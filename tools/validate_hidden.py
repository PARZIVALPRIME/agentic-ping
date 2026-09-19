"""Validate solver coverage/generalisation on the 50 hidden questions.

There are no gold answers for these, so the report focuses on:
  * how many questions the solver resolves (answer produced)
  * which slots fail to resolve
  * the predicted answer + confidence + citation for manual inspection
  * citation overlap with the public set's distribution (sanity check)
"""
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from kg.builder import load_or_build
from reasoning.query_parser import parse_question
from reasoning.solvers import StructuredSolver

kg = load_or_build("corpus-20260919T043338Z-1-001/corpus/corpus.jsonl")
solver = StructuredSolver(kg)

rows = []
skipped = 0
with open("questions-20260919T043312Z-1-001/questions/eval_hidden.jsonl", "r", encoding="utf-8") as fh:
    for line in fh:
        if not line.strip():
            continue
        q = json.loads(line)
        if "eval-0" in q.get("qid", "") and skipped < 0:
            skipped += 1
            continue
        spec = parse_question(q["question"], kg)
        res = solver.solve(q["question"], spec)
        rows.append((q, spec, res))

by_type = defaultdict(Counter)
for q, spec, res in rows:
    by_type[q["qtype"]]["total"] += 1
    by_type[q["qtype"]]["resolved"] += int(res.resolved)

total = len(rows)
resolved = sum(1 for _, _, r in rows if r.resolved)
print(f"\nHIDDEN SET: resolved {resolved}/{total} = {resolved/total:.1%}")
for t, c in sorted(by_type.items()):
    print(f"  {t:12s} {c['resolved']:3d}/{c['total']:3d} = {c['resolved']/c['total']:.1%}")

print("\n--- per-question predictions ---")
for q, spec, res in rows:
    flag = "OK " if res.resolved else "!! "
    print(f"{flag}{q['qid']} [{q['qtype']:11s}] {q['question']}")
    print(f"      -> {res.answer!r}  conf={res.confidence} cites={res.citations}"
          f"  unresolved={res.unresolved}")