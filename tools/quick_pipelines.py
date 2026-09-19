"""Quick accuracy check for RAG vs GraphRAG on a question subset."""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from kg.builder import load_or_build
from kg.textutil import normalize
from pipelines.graphrag_pipeline import GraphRagPipeline
from pipelines.rag_pipeline import RagPipeline
from retrieval import load_index
from utils.llm import build_llm

CORPUS = "corpus-20260919T043338Z-1-001/corpus/corpus.jsonl"
QUESTIONS = "questions-20260919T043312Z-1-001/questions/eval_public.jsonl"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def correct(gold_list, pred):
    p = normalize(pred)
    if not p:
        return False
    return any(p == normalize(g) or normalize(g) in p or p in normalize(g)
               for g in gold_list if normalize(g))


def main():
    kg = load_or_build(CORPUS)
    index = load_index(CORPUS, kg, vector_backend="sparse-tfidf")
    llm = build_llm(type("C", (), {"llm": type("L", (), {"provider": "none"})})())
    print("llm available:", llm.available, "| index:", index.stats())

    qs = []
    with open(QUESTIONS, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                qs.append(json.loads(line))
    qs = qs[:LIMIT]

    pipelines = [
        RagPipeline(index, llm=llm, top_k=5),
        GraphRagPipeline(index, llm=llm, top_k=10, num_hops=2),
    ]
    for pipe in pipelines:
        stats = Counter()
        lat = []
        toks = []
        for q in qs:
            res = pipe.run(q["question"], q["qid"])
            ok = correct(q["answer"], res.answer)
            stats[q["qtype"] + ":total"] += 1
            stats[q["qtype"] + ":ok"] += int(ok)
            lat.append(res.latency_ms)
            toks.append(res.total_tokens)
        total = sum(v for k, v in stats.items() if k.endswith(":total"))
        ok = sum(v for k, v in stats.items() if k.endswith(":ok"))
        print(f"\n=== {pipe.name}: {ok}/{total} = {ok/total:.1%} "
              f"| avg {sum(lat)/len(lat):.0f}ms | avg {sum(toks)/len(toks):.0f} tok")
        for t in ("lookup", "multi_hop", "temporal", "aggregation", "superlative"):
            tt = stats.get(t + ":total", 0)
            to = stats.get(t + ":ok", 0)
            if tt:
                print(f"   {t:12s} {to:3d}/{tt:3d} = {to/tt:.0%}")


if __name__ == "__main__":
    main()