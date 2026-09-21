"""Data-leakage audit: prove no gold answer can reach inference code.

A benchmark that leaks its own answers is worthless, and leakage is easy to
introduce by accident (passing the whole question dict into a pipeline, or
"helpfully" seeding retrieval with ``gold_doc_ids``). This script checks the
three ways it could happen and exits non-zero if any of them is possible.

    python tools/audit_leakage.py

Checks
------
1. **Call-site**  - every ``pipe.run(...)`` receives only (question, qid).
2. **Static**     - no module under pipelines/, agents/, reasoning/, retrieval/
   or kg/ references a gold-answer key.
3. **Behavioural**- the decisive test: re-run the benchmark with every gold
   answer replaced by a random string. If accuracy is unchanged, inference
   never saw the gold. (Leaked answers would make accuracy track the
   corruption.)
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Keys that carry ground truth in the question files.
GOLD_KEYS = ("gold_doc_ids", "gold_answers", "expected_answer")

# Inference code: must never read ground truth. The benchmark/ package is
# excluded on purpose - grading is its job.
INFERENCE_DIRS = ("pipelines", "agents", "reasoning", "retrieval", "kg", "utils")

# `gold` is also a legitimate *corpus* field (the gold medallist of an event),
# so a bare "gold" is not evidence of leakage. These patterns are.
LEAK_PATTERNS = [
    re.compile(r"\bgold_doc_ids\b"),
    re.compile(r"\bgold_answers\b"),
    re.compile(r"\bexpected_answer\b"),
    re.compile(r"""\bgold\s*\[\s*["']answers["']\s*\]"""),
    re.compile(r"""\bgold\b\s*\.\s*get\s*\(\s*["']answers["']"""),
]


def _py_files(dirs) -> List[str]:
    out = []
    for d in dirs:
        for root, _, files in os.walk(os.path.join(ROOT, d)):
            if "__pycache__" in root:
                continue
            out.extend(os.path.join(root, f) for f in files if f.endswith(".py"))
    return out


def check_static() -> Tuple[bool, List[str]]:
    """No inference module may reference a gold-answer key."""
    problems = []
    for path in _py_files(INFERENCE_DIRS):
        try:
            src = open(path, encoding="utf-8").read()
        except OSError:
            continue
        for lineno, line in enumerate(src.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or '"""' in stripped:
                continue
            for pat in LEAK_PATTERNS:
                if pat.search(line):
                    rel = os.path.relpath(path, ROOT)
                    problems.append(f"{rel}:{lineno}: {stripped[:90]}")
    return (not problems), problems


def check_call_sites() -> Tuple[bool, List[str]]:
    """Every ``<pipeline>.run(...)`` must pass only question and qid.

    Only *pipeline* receivers are inspected. ``runner.run(questions, ...)`` is
    the benchmark harness and legitimately handles gold, so matching on the
    method name alone would produce a false positive.
    """
    problems = []
    targets = [os.path.join(ROOT, "benchmark", "runner.py"),
               os.path.join(ROOT, "tools", "submit_hidden.py"),
               os.path.join(ROOT, "tools", "baseline_sweep.py")]

    # Receivers that denote a pipeline object in these files.
    PIPELINE_RECEIVERS = {"pipe", "pipeline", "p", "inner", "agent"}

    for path in targets:
        if not os.path.exists(path):
            continue
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "run"):
                continue
            recv = ast.unparse(node.func.value)
            base = recv.split("[")[0].split(".")[0].strip()
            if base not in PIPELINE_RECEIVERS:
                continue  # not a pipeline call (e.g. the benchmark runner)

            rel = os.path.relpath(path, ROOT)
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                text = ast.unparse(arg)
                if text in ("question", "qid", "text"):
                    continue
                if re.search(r"gold|answer|question_dict", text):
                    problems.append(
                        f"{rel}:{node.lineno}: {recv}.run() receives {text!r}")
            if len(node.args) > 2:
                problems.append(
                    f"{rel}:{node.lineno}: {recv}.run() takes >2 positional args")
    return (not problems), problems


def check_behavioural(limit: int = 25) -> Tuple[bool, List[str]]:
    """Corrupt every gold answer; accuracy must be unaffected.

    This is the test that cannot be fooled by clever indirection: if inference
    reads gold at all, replacing gold with noise changes what the pipelines
    produce. We compare *answers*, not scores, so the check is independent of
    the grader.
    """
    from config import config
    from kg.backend import open_graph
    from pipelines import build_pipelines
    from retrieval import load_index
    from utils.llm import LLMHelper

    kg = open_graph(config.benchmark.corpus_path, config,
                    cache_path=config.benchmark.kg_cache_path, force_local=True)
    index = load_index(config.benchmark.corpus_path, kg,
                       vector_backend=config.benchmark.vector_backend)
    pipes = build_pipelines(index, LLMHelper(), config, with_router=False)

    path = config.benchmark.public_questions_path
    raw = [json.loads(l) for l in open(path, encoding="utf-8")
           if l.strip()][:limit]

    def answers_for(questions: List[Dict[str, Any]]) -> Dict[str, str]:
        out = {}
        for q in questions:
            qid = str(q.get("qid") or q.get("id") or "")
            text = str(q.get("question") or q.get("q") or "")
            for pipe in pipes:
                try:
                    out[f"{qid}:{pipe.name}"] = pipe.run(text, qid).answer
                except Exception as exc:
                    out[f"{qid}:{pipe.name}"] = f"<error:{exc.__class__.__name__}>"
        return out

    clean = answers_for(raw)

    poisoned_src = []
    for q in raw:
        p = dict(q)
        for key in ("answer", "answers", "gold_answers"):
            if key in p:
                p[key] = "ZZZ_CORRUPTED_GOLD_ZZZ"
        p["gold_doc_ids"] = ["Q000000000"]
        poisoned_src.append(p)
    poisoned = answers_for(poisoned_src)

    diffs = [k for k in clean if clean[k] != poisoned.get(k)]
    if diffs:
        return False, [f"answer changed when gold was corrupted: {k} "
                       f"({clean[k]!r} -> {poisoned.get(k)!r})" for k in diffs[:10]]
    return True, [f"{len(clean)} pipeline-answers identical with corrupted gold"]


def main() -> int:
    print("=" * 70)
    print(" DATA LEAKAGE AUDIT")
    print("=" * 70)

    failures = 0
    for label, fn in (("static: inference reads no gold key", check_static),
                      ("call-site: run() gets only (question, qid)", check_call_sites),
                      ("behavioural: corrupting gold changes nothing", check_behavioural)):
        print(f"\n[{label}]")
        try:
            ok, notes = fn()
        except Exception as exc:
            print(f"  ERROR: {exc.__class__.__name__}: {exc}")
            failures += 1
            continue
        for note in notes[:12]:
            print(f"  {'-' if ok else '!'} {note}")
        if ok:
            print("  PASS")
        else:
            print(f"  FAIL ({len(notes)} problem(s))")
            failures += 1

    print("\n" + "=" * 70)
    if failures:
        print(f" {failures} CHECK(S) FAILED - do not submit until resolved")
        return 1
    print(" NO LEAKAGE: gold answers are used only for grading, never inference")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
