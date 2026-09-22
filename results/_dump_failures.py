"""Dump the questions each arm got wrong, with gold vs predicted, so we can
   decide whether an LLM-side (non-deterministic) fix is worth attempting."""
import io
import json
import sys
from collections import Counter

OUT = io.StringIO()
def say(*a):
    print(*a, file=OUT)

rows = json.load(open("results/llm_full.json", encoding="utf-8"))
names = list(rows[0]["pipelines"].keys())
say("arms:", names)

for arm in names:
    wrong = []
    for r in rows:
        p = r["pipelines"][arm]
        ev = p.get("evaluation") or {}
        if not ev.get("correct"):
            wrong.append(r)
    say("\n" + "=" * 100)
    say(f"{arm}: {len(wrong)} wrong / {len(rows)}")
    say("=" * 100)
    say("  by type:", dict(Counter(r["qtype"] for r in wrong)))
    for r in wrong:
        p = r["pipelines"][arm]
        ev = p.get("evaluation") or {}
        say(f"\n  [{r['qid']}] {r['qtype']}  match={ev.get('match_type')}")
        say(f"    Q    : {r['question'][:160]}")
        say(f"    gold : {str(r['gold'].get('answers'))[:110]}")
        say(f"    pred : {str(p.get('answer'))[:110]}")
        say(f"    stop : {p.get('stop_reason')}  conf={p.get('confidence')}  steps={p.get('retrieval_steps')}")

with open("results/_failures.txt", "w", encoding="utf-8") as f:
    f.write(OUT.getvalue())
print("wrote results/_failures.txt")

