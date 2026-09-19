"""Diagnose why the agentic pipeline reports zero LLM tokens."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

entries = json.load(open("results/public_results.json", encoding="utf-8"))
by_pipe = {}
for e in entries:
    for name, rec in (e.get("pipelines") or {}).items():
        by_pipe.setdefault(name, []).append(rec)

for name, recs in by_pipe.items():
    tot = sum(r.get("total_tokens", 0) for r in recs)
    calls = sum(r.get("llm_calls", 0) for r in recs)
    with_ops = sum(1 for r in recs if r.get("tokens_per_operation"))
    print(f"{name:<18s} n={len(recs):<4d} total_tokens={tot:<8d} "
          f"llm_calls={calls:<5d} recs_with_per_op={with_ops}")

ag = by_pipe.get("Agentic GraphRAG", [])
print("\nagentic sample record:")
r = ag[-1]
for k in ("total_tokens", "input_tokens", "output_tokens", "llm_calls",
          "context_tokens", "method", "stop_reason", "confidence"):
    print(f"  {k} = {r.get(k)!r}")
print("  tokens_per_operation:", json.dumps(r.get("tokens_per_operation"), indent=2)[:600])
print("  metadata keys:", list((r.get("metadata") or {}).keys()))
print("  adjudication:", json.dumps((r.get("metadata") or {}).get("adjudication"))[:300])
print("  classification:", json.dumps((r.get("metadata") or {}).get("classification"))[:300])