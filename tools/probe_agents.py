"""List the agent/operation pairs the agentic pipeline actually executed."""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

path = sys.argv[1] if len(sys.argv) > 1 else "results/public_results_v1.json"
entries = json.load(open(path, encoding="utf-8"))

counts = collections.Counter()
pairs = set()
for e in entries:
    ag = e["pipelines"]["Agentic GraphRAG"]
    counts.update(ag.get("agents_invoked") or [])
    for st in ag.get("steps") or []:
        pairs.add((st.get("agent"), st.get("operation")))

print("agents_invoked frequency:")
for name, n in counts.most_common():
    print(f"  {name:<20s} {n}")
print("\nstep agent | operation:")
for a, o in sorted(pairs):
    print(f"  {a:<20s} | {o}")
