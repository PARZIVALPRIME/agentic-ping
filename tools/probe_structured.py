"""Probe the structure-aware retrieval layer (no LLM, no provider calls).

Reports, per question type, what the structured route answers versus gold, and
how often the candidate set was complete. This is the deterministic ceiling of
the structure-aware retrieval step that RAG/GraphRAG now run before answering.

Usage:
    python tools/probe_structured.py [limit]
"""
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from kg.builder import load_or_build
from kg.textutil import normalize
from reasoning.query_parser import parse_question
from retrieval import load_index
from retrieval.structured import StructuredRetriever, choose_candidate

CORPUS = "corpus-20260919T043338Z-1-001/corpus/corpus.jsonl"
QUESTIONS = "questions-20260919T043312Z-1-001/questions/eval_public.jsonl"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def correct(gold_list, pred):
    p = normalize(pred)
    if not p:
        return False
    return any(p == normalize(g) or normalize(g) in p or p in normalize(g)
               for g in gold_list if normalize(g))


class _NoExtraction:
    """Stand-in for a pipeline's passage extractor (empty candidate)."""
    answer = ""
    citations = []
    evidence = []
    confidence = 0.0
    method = "extractive"


def main() -> int:
    kg = load_or_build(CORPUS)
    index = load_index(CORPUS, kg, vector_backend="sparse-tfidf")
    retriever = StructuredRetriever(kg, index)

    questions = []
    with open(QUESTIONS, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                questions.append(json.loads(line))
    questions = questions[:LIMIT]

    by_type = defaultdict(Counter)
    failures = []
    for q in questions:
        spec = parse_question(q["question"], kg)
        got = retriever.retrieve(q["question"], spec)
        choice = choose_candidate(spec, _NoExtraction(), got)
        ok = correct(q["answer"], choice.answer)
        stats = by_type[q["qtype"]]
        stats["total"] += 1
        stats["ok"] += int(ok)
        stats["hits"] += len(got.hits)
        stats["complete"] += int(got.complete)
        stats["verified"] += int(got.verified)
        if not ok:
            failures.append((q, got, choice))

    total = sum(c["total"] for c in by_type.values())
    ok = sum(c["ok"] for c in by_type.values())
    print(f"\n{'=' * 70}\nSTRUCTURED ROUTE {ok}/{total} = {ok / total:.1%}\n{'=' * 70}")
    for t, c in sorted(by_type.items()):
        print(f"  {t:12s} {c['ok']:3d}/{c['total']:3d} = {c['ok'] / c['total']:5.1%}"
              f"  complete={c['complete']:3d} verified={c['verified']:3d}"
              f" avg_hits={c['hits'] / c['total']:.1f}")

    print(f"\n--- FAILURES ({len(failures)}) ---")
    for q, got, choice in failures:
        print(f"\n[{q['qid']}:{q['qtype']}] {q['question']}")
        print(f"   GOLD : {q['answer']}")
        print(f"   PRED : {choice.answer!r} src={choice.source} verified={choice.verified}")
        print(f"   METHOD: {got.method} candidates={got.candidates} "
              f"complete={got.complete} hits={len(got.hits)}")
        if got.unresolved:
            print(f"   UNRESOLVED: {got.unresolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
