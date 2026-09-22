"""Headline report for the LLM-path runs (public 100 + hidden 50)."""
import json

for label, path in (("PUBLIC 100", "results/llm_full_summary.json"),
                    ("HIDDEN 50", "results/hidden_llm_summary.json")):
    s = json.load(open(path, encoding="utf-8"))
    print("=" * 100)
    print(f"{label}  ({s['num_questions']} questions)")
    print("=" * 100)
    for name, p in s["pipelines"].items():
        acc = p.get("accuracy")
        acc_s = f"{acc * 100:5.1f}%" if acc is not None else "  n/a "
        print(f"\n  {name}")
        print(f"    accuracy      : {acc_s}   correct={p['correct']}/{p['num_evaluated']}"
              f"  (answered {p.get('num_answered', p['num_questions'])})")
        print(f"    tokens/q      : {p['avg_total_tokens']:.0f}   llm_calls/q={p['avg_llm_calls']:.2f}"
              f"   retrieval_steps={p['avg_retrieval_steps']:.1f}   chunks={p['avg_chunks_retrieved']:.1f}")
        print(f"    latency       : {p['avg_latency_ms'] / 1000:.2f}s/query")
        print(f"    by_type       : " + "  ".join(
            f"{k}={v['correct']}/{v['n']}" for k, v in p["by_type"].items()))
        if p.get("stop_reasons"):
            print(f"    stop_reasons  : {p['stop_reasons']}")
        if p.get("strategy_adapted") is not None:
            print(f"    adaptation    : strategy_adapted={p['strategy_adapted']}  agents={p['agents_used']}")
    print()

# Router / submission arm
sub = json.load(open("results/hidden_submission.json", encoding="utf-8"))["submission"]
print("=" * 100)
print("SUBMISSION ARM (hidden)")
print("=" * 100)
for k in ("pipeline", "mode", "provider", "chat_model", "graph_backend",
          "num_questions", "num_completed", "num_answered", "total_tokens", "source_run"):
    print(f"  {k:16s}: {sub.get(k)}")
