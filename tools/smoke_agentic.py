"""Smoke test: run the three pipelines on one question per type."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from config import config
from kg.builder import load_or_build
from kg.textutil import normalize
from pipelines import build_pipelines
from retrieval import load_index
from utils.llm import build_llm

CORPUS = config.benchmark.corpus_path
QUESTIONS = config.benchmark.public_questions_path
TYPES = ("lookup", "multi_hop", "temporal", "aggregation", "superlative")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 1   # per type
PIPELINE = sys.argv[2] if len(sys.argv) > 2 else "all"


def correct(gold_list, pred):
    p = normalize(pred)
    if not p:
        return False
    return any(p == normalize(g) or normalize(g) in p or p in normalize(g)
               for g in gold_list if normalize(g))


def main():
    kg = load_or_build(CORPUS)
    index = load_index(CORPUS, kg, vector_backend=config.benchmark.vector_backend)
    llm = build_llm(config)
    print(f"llm available: {llm.available} ({llm.chat_model}) | index: {index.stats()}\n")

    pipelines = build_pipelines(index, llm, config)
    if PIPELINE != "all":
        pipelines = [p for p in pipelines if p.name.lower().startswith(PIPELINE)]

    by_type = {}
    with open(QUESTIONS, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            q = json.loads(line)
            by_type.setdefault(q["qtype"], []).append(q)

    for pipe in pipelines:
        print("=" * 100)
        print(f"PIPELINE: {pipe.name}")
        print("=" * 100)
        for qtype in TYPES:
            for q in by_type.get(qtype, [])[:LIMIT]:
                t0 = time.time()
                res = pipe.run(q["question"], q["qid"])
                ok = correct(q["answer"], res.answer)
                print(f"\n[{qtype}] {q['question']}")
                print(f"  gold      : {q['answer']}")
                print(f"  answer    : {res.answer!r}   -> {'CORRECT' if ok else 'WRONG'}")
                print(f"  method    : {res.method} | conf {res.confidence} | stop {res.stop_reason}")
                print(f"  steps     : {res.retrieval_steps} retrieval / {len(res.steps)} total "
                      f"| agents {res.agents_invoked}")
                print(f"  retrieval : chunks={res.chunks_retrieved} docs={res.docs_retrieved} "
                      f"candidates={res.candidates_considered}")
                print(f"  cost      : tok={res.total_tokens} (in {res.input_tokens}/out "
                      f"{res.output_tokens}) llm_calls={res.llm_calls} "
                      f"latency={res.latency_ms:.0f}ms")
                if res.strategy_changed:
                    print(f"  ADAPTED   : {res.metadata['strategy_changes']}")
                print(f"  citations : {res.citations[:4]}")


if __name__ == "__main__":
    main()