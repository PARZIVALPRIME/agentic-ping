import json, sys
sys.stdout.reconfigure(encoding="utf-8")
main = json.load(open("results/hidden_results.json", encoding="utf-8"))
v4 = json.load(open("results/_hidden_mh_v4.json", encoding="utf-8"))
idx = {r["qid"]: i for i, r in enumerate(main)}
changed = []
for r in v4:
    if r["qid"] in idx:
        old = main[idx[r["qid"]]]["pipelines"]["Agentic GraphRAG"]["answer"]
        main[idx[r["qid"]]] = r
        new = r["pipelines"]["Agentic GraphRAG"]["answer"]
        changed.append(f"{r['qid']}: {old!r} -> {new!r}")
with open("results/hidden_results.json", "w", encoding="utf-8") as f:
    json.dump(main, f, ensure_ascii=False, indent=1)
print("merged", len(changed), "records")
for c in changed:
    print(" ", c)
