"""Print a readable report from results/metrics_summary.json.

Usage:
    python tools/summary_report.py [results/metrics_summary.json]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

path = sys.argv[1] if len(sys.argv) > 1 else "results/metrics_summary.json"
s = json.load(open(path, encoding="utf-8"))

print(f"questions: {s['num_questions']}   wall clock: {s.get('wall_clock_s')}s   "
      f"evaluator: {s.get('evaluator')}\n")

for name, p in s["pipelines"].items():
    print(f"{name}")
    print(f"  accuracy      {p['accuracy']:.1%}  ({p['correct']}/{p['num_evaluated']})")
    print(f"  avg tokens    {p['avg_total_tokens']:.0f}  "
          f"(ctx {p['avg_context_tokens']:.0f}, "
          f"in {p['avg_input_tokens']:.0f}, out {p['avg_output_tokens']:.0f})")
    print(f"  avg latency   {p['avg_latency_ms']:.0f} ms   LLM calls "
          f"{p['avg_llm_calls']:.1f}   steps {p['avg_retrieval_steps']:.1f}")
    print(f"  chunks        {p['avg_chunks_retrieved']:.1f} avg retrieved, "
          f"{p['avg_candidates_considered']:.1f} candidates considered")
    print(f"  citations     precision {p['avg_citation_precision']}  "
          f"recall {p['avg_citation_recall']}")
    print(f"  match types   {p['match_types']}")
    print(f"  stop reasons  {p.get('stop_reasons')}")
    if p.get("agents_used"):
        print(f"  agents used   {p['agents_used']}")
    print("  by question type")
    for qt, t in p["by_type"].items():
        acc = t.get("accuracy")
        flag = "✓" if acc == 1 else ("·" if acc == 0 else "~")
        print(f"    {flag} {qt:<12s} {t.get('correct', 0):>3d}/{t['n']:<3d}"
              + (f"  {acc:.0%}" if acc is not None else "  –"))
    print()

rows = s.get("when_agents_matter", [])
wins = sum(1 for r in rows if r["delta"] > 0)
losses = sum(1 for r in rows if r["delta"] < 0)
print(f"AGENTIC vs GRAPHRAG: +{wins} fixed, -{losses} broken, "
      f"{sum(1 for r in rows if r['delta'] == 0)} same")
for r in rows:
    if r["delta"] != 0:
        sign = "+" if r["delta"] > 0 else "-"
        print(f"  {sign} {r['qid']:<9s} {r['qtype']:<12s} "
              f"agentic {r['agentic_tokens']:>6d} tok, graphrag {r['graphrag_tokens']:>6d} tok")