import json, sys
sys.stdout.reconfigure(encoding="utf-8")
old = {r["qid"]: r for r in json.load(open("results/llm_results.json", encoding="utf-8"))}
new = {r["qid"]: r for r in json.load(open("results/batch/b_pub_01.json", encoding="utf-8"))}
o, n = old["pub-007"], new["pub-007"]
print("Q:", n.get("question"))
print("gold:", n.get("gold") or n.get("answers"))
for tag, rec in (("OLD", o), ("NEW", n)):
    print(tag, "pipeline keys:", list(rec.get("pipelines", {}).keys()))
    for pname, p in rec.get("pipelines", {}).items():
        print(tag, pname, "answer:", repr(p.get("answer")), "| method:", p.get("method"))
    md = rec.get("pipelines", {}).get("agentic_graphrag", {}).get("metadata", {})
    if md:
        print(tag, "meta:", {k: md.get(k) for k in ("run_mode", "agent_mode", "stop_reason", "agents_used") if k in md})
    print(tag, "eval:", rec.get("pipelines", {}).get("agentic_graphrag", {}).get("evaluation"))
