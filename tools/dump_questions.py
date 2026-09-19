"""Dump public questions grouped by qtype for design review."""
import json
import sys
from collections import defaultdict

PATH = "questions-20260919T043312Z-1-001/questions/eval_public.jsonl"
groups = defaultdict(list)
with open(PATH, "r", encoding="utf-8") as fh:
    for line in fh:
        q = json.loads(line)
        groups[q["qtype"]].append(q)

out = []
for t, qs in groups.items():
    out.append(f"\n===== {t} ({len(qs)}) =====")
    for q in qs:
        out.append(f"{q['qid']}: {q['question']}")
        out.append(f"    GOLD: {q['answer']}  docs={len(q['gold_doc_ids'])}")
sys.stdout.reconfigure(encoding="utf-8")
print("\n".join(out))