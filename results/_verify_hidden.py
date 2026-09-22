import json

kg = json.load(open("results/knowledge_graph.json", encoding="utf-8"))
rows = json.load(open("results/hidden_llm.json", encoding="utf-8"))

# 1. eval-002 / eval-010: nations fields straight from the KG event
for qid in ("eval-002", "eval-010"):
    r = [x for x in rows if x["qid"] == qid][0]
    title_frag = "Men's foil" if qid == "eval-002" else "Flyweight"
    year = "1988" if qid == "eval-002" else "1996"
    hits = [e for e in kg["events"].values()
            if title_frag in str(e.get("title", "")) and year in str(e.get("title", ""))]
    for ev in hits:
        fields = {k: v for k, v in ev.items()
                  if "nation" in k.lower() or "compet" in k.lower()}
        print(qid, "KG:", ev.get("title"), "->", fields)

# 2. eval-004: which events match venue+date
print()
r = [x for x in rows if x["qid"] == "eval-004"][0]
print("eval-004 question:", r["question"])
for e in kg["events"].values():
    if "Sydney International Shooting Centre" in str(e.get("venue", "")) \
       and "22 September 2000" in str(e.get("date_raw") or e.get("date") or ""):
        print("  candidate:", e.get("title"), "| gold:", e.get("gold"))

# 3. eval-003: cycling 2008 >N competitors — recount from KG
print()
r = [x for x in rows if x["qid"] == "eval-003"][0]
print("eval-003 question:", r["question"])
import re
m = re.search(r"more than (\d+)", r["question"])
thr = int(m.group(1)) if m else 0
cnt = 0
for e in kg["events"].values():
    if "Cycling at the 2008 Summer Olympics" in str(e.get("title", "")):
        c = str(e.get("competitors", ""))
        if c.isdigit() and int(c) > thr:
            cnt += 1
print("  recount from KG:", cnt, "| Graph:", r["pipelines"]["GraphRAG"]["answer"],
      "| Agent:", r["pipelines"]["Agentic GraphRAG"]["answer"])
