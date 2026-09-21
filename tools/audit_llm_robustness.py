"""Adversarial LLM audit: prove the LLM path cannot make results worse.

    python tools/audit_llm_robustness.py [--limit 40]

Every accuracy number in this repo was measured with ``--no-llm``. That is an
honest measurement of the deterministic core, but it leaves the LLM-assisted
path completely unexercised - and that is exactly the path a 4B model will take
on the submission machine. A small model is not a neutral addition: it is a
component that confidently returns wrong short answers.

So instead of hoping the model behaves, we assume it does not. This harness
substitutes stub "LLMs" that report ``available=True`` and then behave as badly
as a model plausibly can, and asserts that accuracy does not drop:

    ADVERSARY          behaviour
    -----------------  --------------------------------------------------
    always_disagree    returns a confidently wrong short answer every time
    prose              answers counting questions in prose ("There are ...")
    empty              returns empty / malformed JSON
    truncated          returns a 400-char ramble (a 4B model that overruns)

If accuracy under any adversary is below the deterministic baseline, a guard is
missing and the run on the other machine would silently score lower than the
--no-llm run we validated. That is the failure this file exists to catch.

This is a *bound*, not a prediction: a real model will land somewhere between
the adversary and the oracle. The point is that the floor is the deterministic
score, so enabling the LLM can only help.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ── stub models ─────────────────────────────────────────────────────────────
class _StubLLM:
    """Minimal stand-in exposing the surface utils.llm callers actually use."""

    provider = "stub"

    def __init__(self, behaviour: str) -> None:
        self.behaviour = behaviour
        self.available = True
        self.chat_model = f"stub:{behaviour}"
        self.fast_model = self.chat_model
        self.eval_model = self.chat_model
        self.num_calls = 0
        self.failed_calls = 0
        self.error = ""

    # -- the three entry points pipelines/agents call -----------------------
    def complete(self, prompt: str, system_prompt: str = "", caller: str = "llm",
                 counter=None, **kwargs) -> str:
        self.num_calls += 1
        return self._text()

    def complete_json(self, prompt: str, system_prompt: str = "", caller: str = "llm",
                      counter=None, **kwargs) -> Dict[str, Any]:
        self.num_calls += 1
        if self.behaviour == "empty":
            return {}
        return {"answer": self._text(), "agree": False,
                "reason": "adversarial stub: deliberately overrules the candidate"}

    def chat(self, messages, tools=None, counter=None, model=None,
             caller: str = "chat", **kwargs):
        """ReAct entry point; real signature returns ``(text, tool_calls,
        finish_reason)``. Emitting text but never a tool call is the realistic
        small-model failure: the loop stalls and must fall back."""
        self.num_calls += 1
        return self._text(), [], "stop"

    def stats(self) -> Dict[str, Any]:
        return {"available": True, "provider": self.provider,
                "num_calls": self.num_calls, "failed_calls": 0}

    # -- the bad behaviour ---------------------------------------------------
    def _text(self) -> str:
        if self.behaviour == "always_disagree":
            return "Vladimir Smirnov"          # plausible, short, wrong
        if self.behaviour == "prose":
            return "There are several such events across the Games."
        if self.behaviour == "empty":
            return ""
        if self.behaviour == "truncated":
            return ("Based on the provided context it appears that the answer "
                    "may involve a number of athletes across several editions "
                    "of the Games, however the passages are incomplete and " * 3)
        raise ValueError(self.behaviour)


ADVERSARIES = ["always_disagree", "prose", "empty", "truncated"]


def _guard_unit_tests() -> List[str]:
    """Check the guards reject bad verdicts *without* rejecting good ones.

    An audit that only proves "nothing gets worse" is satisfied by a guard that
    ignores the model entirely. So we also assert the opposite direction: a
    correct, context-supported correction must still be accepted, or the LLM
    has been neutered rather than constrained.
    """
    from utils.llm import refine_answer

    ctx = [{"doc_id": "d1", "title": "Gymnastics",
            "text": "Larisa Latynina won 18 Olympic medals in total."}]

    class _Say:
        available = True

        def __init__(self, ans):
            self.ans = ans

        def complete_json(self, *a, **k):
            return {"answer": self.ans, "agree": False, "reason": "t"}

    cases = [
        # (label, model answer, candidate, qtype, exhaustive, expect_final)
        ("accepts a grounded correction",
         "Larisa Latynina", "Vera Caslavska", "lookup", False, "Larisa Latynina"),
        ("accepts a grounded numeric correction",
         "18", "9", "lookup", False, "18"),
        ("rejects an ungrounded name",
         "Nadia Comaneci", "Larisa Latynina", "lookup", False, "Larisa Latynina"),
        ("rejects prose over a number",
         "There are several.", "18", "lookup", False, "18"),
        ("skips adjudication for aggregation",
         "42", "18", "aggregation", False, "18"),
        ("skips adjudication when exhaustive",
         "42", "18", "lookup", True, "18"),
        ("does not match 18 inside 1980",
         "1980", "18", "lookup", False, "18"),
    ]
    failures = []
    print("\n guard unit tests (both directions):")
    for label, model_ans, cand, qtype, exh, expect in cases:
        got, _changed, payload = refine_answer(
            _Say(model_ans), "q?", cand, ctx, None,
            qtype=qtype, exhaustive=exh)
        ok = got.strip() == expect
        print(f"   {'ok  ' if ok else 'FAIL'} {label:<40} -> {got!r} "
              f"[{payload.get('reason', '')[:28]}]")
        if not ok:
            failures.append(f"guard/{label}: expected {expect!r}, got {got!r}")
    return failures


# ── harness ─────────────────────────────────────────────────────────────────
def _load(limit: int) -> List[Dict[str, Any]]:
    from config import config
    path = config.benchmark.public_questions_path
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return rows[:limit]


def _score(pipes, rows) -> Dict[str, float]:
    """Accuracy per pipeline. Uses the project's own evaluator with the judge
    off, so the LLM under test can never influence its own grade."""
    from benchmark.evaluator import Evaluator
    from benchmark.runner import _gold_of      # same gold normalisation as a real run

    ev = Evaluator(llm=None)
    out: Dict[str, float] = {}
    for pipe in pipes:
        hits = 0
        for row in rows:
            qid = row.get("id") or row.get("qid") or ""
            question = row.get("question", "")
            gold = _gold_of(row) or {"answers": [], "doc_ids": []}
            try:
                res = pipe.run(question, qid)
            except Exception as exc:                  # a crash is a failure, not a skip
                print(f"    ! {pipe.name} raised on {qid}: "
                      f"{exc.__class__.__name__}: {exc}")
                continue
            verdict = ev.evaluate(question, gold["answers"], res.answer,
                                  res.citations, gold["doc_ids"])
            hits += bool(verdict.is_correct)
        out[pipe.name] = hits / len(rows) if rows else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=40,
                    help="questions from the public set (default 40)")
    args = ap.parse_args()

    from config import config
    from kg.backend import open_graph
    from pipelines import build_pipelines
    from retrieval import load_index

    rows = _load(args.limit)
    print(f"\n{'=' * 70}\n ADVERSARIAL LLM AUDIT  ({len(rows)} questions)\n{'=' * 70}")
    print(" Substituting deliberately-wrong models for the real one and")
    print(" checking that accuracy never falls below the deterministic run.\n")

    kg = open_graph(config.benchmark.corpus_path, config,
                    cache_path=config.benchmark.kg_cache_path, force_local=True)
    index = load_index(config.benchmark.corpus_path, kg,
                       vector_backend=config.benchmark.vector_backend)

    # Baseline: no LLM at all. This is the floor every adversary must match.
    base = _score(build_pipelines(index, None, config, with_router=False), rows)
    print(" baseline (--no-llm):")
    for name, acc in base.items():
        print(f"   {name:<20} {acc:6.0%}")

    failures: List[str] = []
    failures += _guard_unit_tests()
    for behaviour in ADVERSARIES:
        stub = _StubLLM(behaviour)
        scores = _score(build_pipelines(index, stub, config, with_router=False), rows)
        print(f"\n adversary '{behaviour}'  ({stub.num_calls} calls made):")
        for name, acc in scores.items():
            delta = acc - base.get(name, 0.0)
            flag = "  <-- REGRESSION" if delta < -1e-9 else ""
            print(f"   {name:<20} {acc:6.0%}  ({delta:+.0%}){flag}")
            if delta < -1e-9:
                failures.append(f"{behaviour}/{name}: {base[name]:.0%} -> {acc:.0%}")
        if stub.num_calls == 0:
            print("   note: no calls made - the LLM path was not reached here")

    print(f"\n{'=' * 70}")
    if failures:
        print(" FAIL - a bad model degrades these pipelines:")
        for line in failures:
            print(f"   - {line}")
        print("\n A guard is missing. Do not run with a small model until this")
        print(" passes, or the submission will score below the --no-llm run.")
        print("=" * 70)
        return 1
    print(" PASS - no adversary can score below the deterministic baseline.")
    print(" The LLM is upside-only: guards keep a wrong verdict from replacing")
    print(" a correct structured answer.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
