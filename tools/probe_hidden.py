"""Inspect the hidden-set agentic records (answers, tokens, methods)."""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

entries = json.load(open("results/hidden_results.json", encoding="utf-8"))
print(f"entries: {len(entries)}\n")

for name in ("RAG", "GraphRAG", "Agentic GraphRAG"):
    toks = [e["pipelines"].get(name, {}).get("total_tokens", 0) for e in entries]
    empty = [e["qid"] for e in entries
             if not (e["pipelines"].get(name, {}).get("answer") or "").strip()]
    methods = Counter((e["pipelines"].get(name, {}).get("method") or "?")
                      for e in entries)
    stops = Counter((e["pipelines"].get(name, {}).get("stop_reason") or "?")
                    for e in entries)
    print(f"{name}: avg_tok={sum(toks)/len(toks):.1f} min={min(toks)} max={max(toks)}")
    print(f"  empty answers ({len(empty)}): {empty[:12]}")
    print(f"  methods: {dict(methods)}")
    print(f"  stops:   {dict(stops)}\n")

print("sample agentic records:")
for e in entries[:4] + entries[-2:]:
    ag = e["pipelines"]["Agentic GraphRAG"]
    print(f"  {e['qid']} [{e['qtype']}] {e['question'][:78]}")
    print(f"     ans={ag.get('answer')!r} method={ag.get('method')} tok={ag.get('total_tokens')} "
          f"steps={ag.get('retrieval_steps')} stop={ag.get('stop_reason')} conf={ag.get('confidence')}")
