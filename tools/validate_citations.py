"""Report citation precision/recall against gold_doc_ids on the public set."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from kg.builder import load_or_build
from reasoning.query_parser import parse_question
from reasoning.solvers import StructuredSolver

kg = load_or_build("corpus-20260919T043338Z-1-001/corpus/corpus.jsonl")
solver = StructuredSolver(kg)

per_type = {}
with open("questions-20260919T043312Z-1-001/questions/eval_public.jsonl", "r", encoding="utf-8") as fh:
    for line in fh:
        if not line.strip():
            continue
        q = json.loads(line)
        spec = parse_question(q["question"], kg)
        res = solver.solve(q["question"], spec)
        gold = set(q.get("gold_doc_ids") or [])
        cited = set(res.citations)
        inter = gold & cited
        rec = len(inter) / len(gold) if gold else 0.0
        prec = len(inter) / len(cited) if cited else 0.0
        st = per_type.setdefault(q["qtype"], {"n": 0, "rec": 0.0, "prec": 0.0, "hit": 0})
        st["n"] += 1
        st["rec"] += rec
        st["prec"] += prec
        st["hit"] += int(bool(inter))

print(f"{'qtype':12s} {'n':>4s} {'citation_recall':>16s} {'citation_precision':>19s} {'any_hit':>8s}")
tot = {"n": 0, "rec": 0.0, "prec": 0.0, "hit": 0}
for t, st in sorted(per_type.items()):
    print(f"{t:12s} {st['n']:4d} {st['rec']/st['n']:16.3f} {st['prec']/st['n']:19.3f} "
          f"{st['hit']}/{st['n']}")
    for k in tot:
        tot[k] += st[k]
print(f"{'ALL':12s} {tot['n']:4d} {tot['rec']/tot['n']:16.3f} {tot['prec']/tot['n']:19.3f} "
      f"{tot['hit']}/{tot['n']}")