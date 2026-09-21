"""Why did each pipeline get it wrong? Failure triage for a live run.

    python tools/triage_failures.py results/llm_results.json --pipeline "Agentic GraphRAG"

Accuracy tells you *how many* you lost; it never tells you *what to fix*. This
groups the wrong answers by their likely cause so effort goes where the losses
actually are, instead of to whichever knob is easiest to turn.

Causes are inferred from the record, not from the gold answer, so this is a
diagnostic and not a second grader.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split()).strip(" \t\"'.")


def _is_num(text: str) -> bool:
    return bool(re.fullmatch(r"\s*-?\d+(?:[.,]\d+)?\s*", text or ""))


def classify_failure(rec: Dict[str, Any], golds: List[str]) -> str:
    """Best-effort cause of a wrong answer, from the record alone."""
    pred = _norm(rec.get("answer", ""))
    gold_n = [_norm(g) for g in golds]
    meta = rec.get("metadata") or {}
    adj = meta.get("adjudication") or {}

    if not pred:
        return "empty_answer"

    # The LLM overwrote a candidate that was already right.
    cand = _norm(adj.get("candidate", ""))
    if cand and cand in gold_n and pred not in gold_n:
        return "llm_overwrote_correct_candidate"

    # Off-by-one / wrong count: both numeric, both present, different.
    if _is_num(pred) and any(_is_num(g) for g in gold_n):
        if pred not in gold_n:
            try:
                diff = min(abs(float(pred) - float(g))
                           for g in gold_n if _is_num(g))
                return ("count_off_by_one" if diff == 1
                        else f"count_wrong_by_{int(diff)}" if diff < 10
                        else "count_far_off")
            except ValueError:
                return "count_wrong"

    # Right idea, wrong surface form (substring either direction).
    for g in gold_n:
        if g and (g in pred or pred in g):
            return "phrasing_or_partial_match"

    if rec.get("stop_reason") in ("insufficient_retrieved_evidence",
                                  "no_evidence"):
        return "evidence_not_retrieved"

    if adj.get("reason", "").startswith("verdict_"):
        return f"guard_{adj['reason']}"

    return "wrong_entity"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--pipeline", default="Agentic GraphRAG")
    ap.add_argument("--show", type=int, default=12,
                    help="how many failing questions to print in full")
    args = ap.parse_args()

    with open(args.path, encoding="utf-8") as fh:
        data = json.load(fh)
    rows = data if isinstance(data, list) else data.get("results", data)

    by_cause: Counter = Counter()
    by_type_cause: Dict[str, Counter] = defaultdict(Counter)
    examples: List[Dict[str, Any]] = []
    total = wrong = 0

    # Records are nested per question: {qid, gold, pipelines: {name: record}}.
    for row in rows:
        rec = (row.get("pipelines") or {}).get(args.pipeline)
        if not rec:
            continue
        total += 1
        ev = rec.get("evaluation") or {}
        if ev.get("is_correct"):
            continue
        wrong += 1
        golds = (row.get("gold") or {}).get("answers") or []
        if isinstance(golds, str):
            golds = [golds]
        cause = classify_failure(rec, golds)
        by_cause[cause] += 1
        qtype = row.get("qtype") or rec.get("qtype", "?")
        by_type_cause[qtype][cause] += 1
        examples.append({"qid": row.get("qid"), "qtype": qtype,
                         "cause": cause, "q": row.get("question", "")[:90],
                         "pred": rec.get("answer", "")[:60],
                         "gold": (golds[0] if golds else "")[:60],
                         "stop": rec.get("stop_reason", "")})

    print(f"\n{'=' * 76}")
    print(f" FAILURE TRIAGE  {args.pipeline}   {wrong}/{total} wrong "
          f"({(total - wrong) / total:.0%} correct)" if total else " no records")
    print("=" * 76)

    if not wrong:
        print("  no failures to triage")
        return 0

    print("\n by cause (this is your fix-list, ordered):")
    for cause, n in by_cause.most_common():
        print(f"   {n:>3}  {cause}")

    print("\n by qtype:")
    for qtype, counter in sorted(by_type_cause.items(),
                                 key=lambda kv: -sum(kv[1].values())):
        causes = ", ".join(f"{c}={n}" for c, n in counter.most_common(3))
        print(f"   {qtype:<14} {sum(counter.values()):>3}  ({causes})")

    print(f"\n failing questions (first {args.show}):")
    for ex in examples[:args.show]:
        print(f"\n   {ex['qid']}  [{ex['qtype']}]  {ex['cause']}")
        print(f"     Q    {ex['q']}")
        print(f"     pred {ex['pred']!r}")
        print(f"     gold {ex['gold']!r}")
        if ex["stop"]:
            print(f"     stop {ex['stop']}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
