"""Validate the structured solver against the 100 public gold answers.

Usage: python tools/validate_solvers.py
"""
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from kg.builder import load_or_build
from kg.textutil import normalize
from reasoning.query_parser import parse_question
from reasoning.solvers import StructuredSolver

CORPUS = "corpus-20260919T043338Z-1-001/corpus/corpus.jsonl"
QUESTIONS = "questions-20260919T043312Z-1-001/questions/eval_public.jsonl"


def correct(gold_list, pred):
    pred_n = normalize(pred)
    if not pred_n:
        return False
    for g in gold_list:
        gn = normalize(g)
        if not gn:
            continue
        if pred_n == gn or gn in pred_n or pred_n in gn:
            return True
    return False


def main() -> int:
    kg = load_or_build(CORPUS)
    solver = StructuredSolver(kg)

    questions = []
    with open(QUESTIONS, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                questions.append(json.loads(line))

    by_type = defaultdict(Counter)
    failures = []
    for q in questions:
        spec = parse_question(q["question"], kg)
        res = solver.solve(q["question"], spec)
        ok = correct(q["answer"], res.answer)
        by_type[q["qtype"]]["total"] += 1
        by_type[q["qtype"]]["ok"] += int(ok)
        if not ok:
            failures.append((q, spec, res))

    total = sum(c["total"] for c in by_type.values())
    ok = sum(c["ok"] for c in by_type.values())
    print(f"\n{'='*70}\nOVERALL {ok}/{total} = {ok/total:.1%}\n{'='*70}")
    for t, c in sorted(by_type.items()):
        print(f"  {t:12s} {c['ok']:3d}/{c['total']:3d} = {c['ok']/c['total']:.1%}")

    print(f"\n--- FAILURES ({len(failures)}) ---")
    for q, spec, res in failures:
        print(f"\n[{q['qid']}:{q['qtype']}] {q['question']}")
        print(f"   GOLD : {q['answer']}")
        print(f"   PRED : {res.answer!r}  conf={res.confidence} method={res.method}")
        print(f"   SLOTS: qtype={spec.qtype} sport={spec.sport!r} yr={spec.year} "
              f"season={spec.season!r} desc={spec.event_desc!r} venue={spec.venue!r}")
        print(f"          date={spec.date_text!r} thr={spec.threshold} before={spec.before_year} "
              f"title={spec.target_title!r}")
        if res.unresolved:
            print(f"   UNRESOLVED: {res.unresolved}")
        for st in res.steps[-2:]:
            print(f"   STEP: {json.dumps(st, ensure_ascii=False)[:300]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())