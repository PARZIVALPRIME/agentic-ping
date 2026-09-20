"""Targeted check of the agentic verifier and the venue+date tool (live LLM).

Runs the agentic pipeline on the questions that failed in the previous run plus
controls that passed, and prints what the tool loop submitted next to what the
deterministic verifier kept or corrected. This validates the ``verify_answer``
pass and the ``resolve_event`` tool; it spends provider calls, so it takes an
explicit list of qids.

Usage:
    python tools/probe_agentic_verify.py                 # known failures + controls
    python tools/probe_agentic_verify.py pub-045 pub-087
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from benchmark.evaluator import Evaluator
from config import config
from kg.backend import open_graph
from pipelines import build_pipelines
from retrieval import load_index
from utils.llm import build_llm

# The six questions the previous live run got wrong, plus four it got right.
DEFAULT_QIDS = ("pub-037", "pub-038", "pub-045", "pub-060", "pub-086", "pub-087",
                "pub-001", "pub-005", "pub-002", "pub-009")


def main() -> int:
    wanted = tuple(sys.argv[1:]) or DEFAULT_QIDS
    corpus = config.benchmark.corpus_path

    print("building KG / index ...")
    kg = open_graph(corpus, config, cache_path=config.benchmark.kg_cache_path,
                    force_local=True)
    index = load_index(corpus, kg, vector_backend=config.benchmark.vector_backend)
    llm = build_llm(config)
    print(f"llm: {llm.available} ({llm.provider}/{llm.chat_model})")

    questions = {}
    with open(config.benchmark.public_questions_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                q = json.loads(line)
                questions[q["qid"]] = q

    agentic = [p for p in build_pipelines(index, llm, config)
               if p.name == "Agentic GraphRAG"][0]
    evaluator = Evaluator(None, use_llm_judge=False)

    passed = 0
    for qid in wanted:
        q = questions.get(qid)
        if q is None:
            print(f"{qid}: not in the public set")
            continue
        result = agentic.run(q["question"], qid)
        verdict = evaluator.evaluate(q["question"], q["answer"], result.answer,
                                     result.citations, q.get("doc_ids") or [])
        ok = bool(getattr(verdict, "is_correct", False))
        passed += int(ok)
        checks = [c for c in (result.metadata.get("strategy_changes") or [])
                  if c.startswith("verification[")]
        print(f"\n[{qid}:{q['qtype']}] {'OK ' if ok else 'MISS'} {q['question']}")
        print(f"   GOLD    : {q['answer']}")
        print(f"   ANSWER  : {result.answer!r}  (method={result.method}, "
              f"stop={result.stop_reason}, conf={result.confidence})")
        print(f"   expected: {verdict.match_type} | citations={result.citations[:3]}")
        for check in checks:
            print(f"   VERIFIED: {check}")
    print(f"\n{passed}/{len(wanted)} correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
