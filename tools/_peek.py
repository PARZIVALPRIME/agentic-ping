import sys, os, json
sys.path.insert(0, os.getcwd())
from reasoning.query_parser import parse_question, classify
p="questions-20260919T043312Z-1-001/questions/eval_hidden.jsonl"
rows=[json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]
parsed=0; olympics=0
from collections import Counter
qt=Counter()
for r in rows:
    q=r["question"]
    if "Olympics" in q: olympics+=1
    sp=parse_question(q, None)
    if sp and (sp.year or sp.sport or sp.season): parsed+=1
    qt[classify(q)]+=1
print(f"hidden n={len(rows)}  mention 'Olympics': {olympics}/{len(rows)}")
print(f"parser extracted structure: {parsed}/{len(rows)}")
print("classify dist:", dict(qt))
