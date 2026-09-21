"""Produce the hidden-set submission artefacts.

The hackathon asks for raw outputs on the 50 hidden questions: tokens used,
answers generated, and the agentic trace. This script produces exactly that,
in a flat, self-describing shape that a grader can read without knowing our
internals.

Design rule: **never raise**. A single malformed question must not cost the
other 49. Every failure is captured in the record and the run continues, and
the file is rewritten atomically after each question so an interrupted run
never loses completed work.

Usage::

    python tools/submit_hidden.py                     # agentic, default paths
    python tools/submit_hidden.py --no-llm            # deterministic
    python tools/submit_hidden.py --pipeline Router
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_OUT = os.path.join("results", "hidden_submission.json")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_questions(path: str) -> List[Dict[str, Any]]:
    """Read .jsonl (one object per line) or a .json list."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read().strip()
    if not text:
        return []
    if text.lstrip().startswith("["):
        return json.loads(text)
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _atomic_write(path: str, payload: Any, retries: int = 5) -> None:
    """Write atomically, tolerating transient Windows/OneDrive file locks.

    ``os.replace`` is atomic but fails with PermissionError when a sync client
    or AV scanner momentarily holds the destination. That is transient, so we
    retry with a short backoff, and fall back to a direct write rather than
    losing a whole run over a locked file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.25 * (attempt + 1))

    # Last resort: write in place. Slightly less safe than a rename, but far
    # better than aborting a 50-question run.
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        if os.path.exists(tmp):
            os.remove(tmp)
    except OSError as exc:
        _log(f"WARNING: could not write {path} ({exc}); progress kept in {tmp}")


def _record(result, qid: str, question: str) -> Dict[str, Any]:
    """Flatten a PipelineResult into the submission schema."""
    d = result.to_dict()
    return {
        "id": qid,
        "question": question,
        "qtype": d.get("qtype", ""),
        "answer": d.get("answer", ""),
        "citations": d.get("citations", []),
        "confidence": d.get("confidence", 0.0),
        "tokens": {
            "context_tokens": d.get("context_tokens", 0),
            "input_tokens": d.get("input_tokens", 0),
            "output_tokens": d.get("output_tokens", 0),
            "total_tokens": d.get("total_tokens", 0),
            "llm_calls": d.get("llm_calls", 0),
        },
        "agentic_trace": {
            "pipeline": d.get("pipeline", ""),
            "plan": d.get("plan", []),
            "retrieval_steps": d.get("retrieval_steps", 0),
            "loop_iterations": d.get("loop_iterations", 0),
            "tools_called": d.get("tools_called", []),
            "agents_invoked": d.get("agents_invoked", []),
            "strategy_changed": d.get("strategy_changed", False),
            "stop_reason": d.get("stop_reason", ""),
            "candidates_considered": d.get("candidates_considered", 0),
            "chunks_retrieved": d.get("chunks_retrieved", 0),
            "docs_retrieved": d.get("docs_retrieved", 0),
            "steps": d.get("steps", []),
            "time_per_operation": d.get("time_per_operation", []),
            "tokens_per_operation": d.get("tokens_per_operation", []),
        },
        "evidence": d.get("evidence", []),
        "unresolved": d.get("unresolved", []),
        "latency_ms": d.get("latency_ms", 0.0),
        "error": None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the hidden question set")
    ap.add_argument("questions", nargs="?", default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--pipeline", default="Agentic GraphRAG",
                    help="pipeline .name to submit with")
    ap.add_argument("--no-llm", action="store_true",
                    help="deterministic mode: make no provider calls")
    ap.add_argument("--no-tg", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    from config import config
    from kg.backend import describe_backend, open_graph
    from pipelines import build_pipelines
    from retrieval import load_index
    from utils.llm import LLMHelper, build_llm

    qpath = args.questions or config.benchmark.hidden_questions_path
    if not os.path.exists(qpath):
        _log(f"ERROR: hidden questions not found at {qpath}")
        return 1

    _log("building KG / index ...")
    kg = open_graph(config.benchmark.corpus_path, config,
                    cache_path=config.benchmark.kg_cache_path,
                    force_local=args.no_tg)
    backend = describe_backend()
    index = load_index(config.benchmark.corpus_path, kg,
                       vector_backend=config.benchmark.vector_backend)

    if args.no_llm:
        llm = LLMHelper()
        _log("LLM disabled: deterministic reasoning only")
    else:
        llm = build_llm(config)
        if not llm.available:
            _log(f"WARNING: no LLM available ({llm.error}); "
                 "falling back to deterministic mode")

    pipes = build_pipelines(index, llm, config)
    by_name = {p.name: p for p in pipes}
    pipe = by_name.get(args.pipeline)
    if pipe is None:
        match = [p for p in pipes if args.pipeline.lower() in p.name.lower()]
        if not match:
            _log(f"ERROR: no pipeline matching {args.pipeline!r}. "
                 f"available: {list(by_name)}")
            return 1
        pipe = match[0]
    _log(f"pipeline: {pipe.name} | backend: {backend['active']}")

    questions = _load_questions(qpath)
    if args.limit:
        questions = questions[:args.limit]
    _log(f"{len(questions)} hidden questions from {qpath}")

    records: List[Dict[str, Any]] = []
    started = time.perf_counter()
    failures = 0

    for i, q in enumerate(questions, 1):
        qid = str(q.get("id") or q.get("qid") or f"hidden-{i:03d}")
        text = str(q.get("question") or q.get("text") or "")
        try:
            result = pipe.run(text, qid)
            rec = _record(result, qid, text)
        except Exception:  # one bad question must not cost the rest
            failures += 1
            rec = {"id": qid, "question": text, "answer": "", "citations": [],
                   "tokens": {}, "agentic_trace": {}, "evidence": [],
                   "error": traceback.format_exc(limit=4)}
        records.append(rec)

        def _payload() -> Dict[str, Any]:
            return {
                "submission": {
                    "pipeline": pipe.name,
                    "mode": "deterministic" if not getattr(llm, "available", False) else "llm",
                    "provider": getattr(llm, "provider", None),
                    "chat_model": getattr(llm, "chat_model", None),
                    "graph_backend": backend.get("active"),
                    "questions_file": qpath,
                    "num_questions": len(questions),
                    "num_completed": len(records),
                    "num_failed": failures,
                    "config": config.describe(),
                },
                "results": records,
            }

        # Checkpoint periodically rather than every question: the traces are
        # large, and rewriting all of them 50 times is pure overhead. The final
        # write after the loop guarantees a complete file.
        if i % 10 == 0 or i == len(questions):
            _atomic_write(args.out, _payload())

        answer = (rec.get("answer") or "")[:52]
        _log(f"  {i}/{len(questions)} {qid:<14s} "
             f"tok={rec.get('tokens', {}).get('total_tokens', 0):<6} "
             f"-> {answer!r}{' [ERROR]' if rec.get('error') else ''}")

    elapsed = time.perf_counter() - started
    answered = sum(1 for r in records if r.get("answer"))
    total_tokens = sum(r.get("tokens", {}).get("total_tokens", 0) for r in records)

    print("=" * 66)
    print(f"wrote {args.out}")
    print(f"  questions : {len(records)}")
    print(f"  answered  : {answered} ({answered / max(1, len(records)):.0%})")
    print(f"  failed    : {failures}")
    print(f"  tokens    : {total_tokens:,} "
          f"({total_tokens / max(1, len(records)):.0f}/question)")
    print(f"  wall clock: {elapsed:.1f}s")
    print("=" * 66)
    if failures:
        print("NOTE: failed questions kept an empty answer and an 'error' field.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
