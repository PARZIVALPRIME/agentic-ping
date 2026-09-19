"""Prepare the Olympics corpus for the benchmark.

This is the data-ingestion step of the pipeline. It builds the two artefacts
every pipeline reads from, so that the benchmark itself does no parsing work:

1. **Knowledge graph** -> ``results/knowledge_graph.json``
   Events, athletes, venues, games and WON_BY edges extracted from each
   article title + leading ``[Infobox ...]`` block (``kg/builder.py``).
2. **Retrieval index** -> in-memory (rebuilt per run, ~4 s)
   130-word overlapping chunks with BM25 postings and a sparse TF-IDF vector
   space (``retrieval/index.py``, ``retrieval/vector.py``).

Both are derived purely from the corpus, which keeps the three-pipeline
comparison reproducible: the same graph and the same chunks are available to
RAG (vector only), GraphRAG (vector + graph) and the agentic pipeline (any
tool, repeated).

Usage
-----
    python setup/ingest_corpus.py                      # build + report
    python setup/ingest_corpus.py --rebuild            # force KG rebuild
    python setup/ingest_corpus.py --corpus path.jsonl  # explicit corpus
    python setup/ingest_corpus.py --vector-backend sparse-tfidf
    python setup/ingest_corpus.py --sport Gymnastics   # inspect one sport

A TigerGraph (Savanna / Community Edition) backend is *optional*: the graph
model in ``kg/model.py`` and the vector backends in ``retrieval/vector.py`` are
drop-in extension points, and ``.env`` carries ``TG_*`` settings for that path.
Nothing in this script needs a running database.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

KG_CACHE = "results/knowledge_graph.json"


def _fmt(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) else str(value)


def report_kg(kg, corpus_path: str) -> None:
    stats = kg.stats()
    print("\n" + "=" * 66)
    print("KNOWLEDGE GRAPH")
    print("=" * 66)
    for key in ("num_events", "num_athletes", "num_edges", "num_games",
                "num_venues", "num_sports", "num_distractors"):
        if key in stats:
            print(f"  {key:<20s} {_fmt(stats[key])}")
    print("  " + "-" * 30)
    for key in ("with_competitors", "with_nations", "with_venue", "with_date"):
        if key in stats:
            coverage = stats[key] / max(stats.get("num_events", 1), 1)
            print(f"  {key:<20s} {_fmt(stats[key]):>9s}  ({coverage:.0%} of events)")
    print(f"  {'cache':<20s} {KG_CACHE}")
    print(f"  {'corpus':<20s} {corpus_path}")


def report_index(index) -> None:
    stats = index.stats()
    print("\n" + "=" * 66)
    print("RETRIEVAL INDEX")
    print("=" * 66)
    for key, value in stats.items():
        print(f"  {key:<20s} {_fmt(value)}")
    words = [len(getattr(c, "text", "").split()) for c in index.chunks[:20000]]
    if words:
        print(f"  {'avg_chunk_words':<20s} {sum(words) / len(words):.1f}")
        print(f"  {'max_chunk_words':<20s} {max(words)}")


def inspect_sport(kg, sport: str) -> None:
    events = kg.events_for_sport(sport)
    print("\n" + "=" * 66)
    print(f"SPORT: {sport}  ({len(events)} event articles)")
    print("=" * 66)
    for ev in events[:25]:
        print(f"  {ev.doc_id:<58s} {ev.games_key:<22s} venue={ev.venue or '-'}")
    if len(events) > 25:
        print(f"  ... and {len(events) - 25} more")


def main(argv=None) -> int:
    from config import config
    from kg.builder import load_or_build
    from retrieval import load_index

    ap = argparse.ArgumentParser(description="Build the KG + retrieval index")
    ap.add_argument("--corpus", default=config.benchmark.corpus_path)
    ap.add_argument("--kg-cache", default=KG_CACHE)
    ap.add_argument("--rebuild", action="store_true", help="ignore the KG cache")
    ap.add_argument("--vector-backend", default=config.benchmark.vector_backend,
                    choices=["auto", "sparse-tfidf", "hashing-tfidf",
                             "sentence-transformer"])
    ap.add_argument("--sport", default="", help="inspect events for one sport")
    ap.add_argument("--json", action="store_true", help="machine-readable stats")
    args = ap.parse_args(argv)

    if not os.path.exists(args.corpus):
        print(f"corpus not found: {args.corpus}", file=sys.stderr)
        return 2

    t0 = time.time()
    print(f"corpus      : {args.corpus}")
    print(f"building KG : {'(forced rebuild)' if args.rebuild else '(cache if present)'}")
    kg = load_or_build(args.corpus, args.kg_cache, rebuild=args.rebuild)
    t_kg = time.time() - t0

    t1 = time.time()
    print(f"building index (backend={args.vector_backend}) ...")
    index = load_index(args.corpus, kg, vector_backend=args.vector_backend)
    t_idx = time.time() - t1

    if args.json:
        print(json.dumps({"kg": kg.stats(), "index": index.stats(),
                          "kg_seconds": round(t_kg, 2),
                          "index_seconds": round(t_idx, 2)}, indent=2))
        return 0

    report_kg(kg, args.corpus)
    report_index(index)
    if args.sport:
        inspect_sport(kg, args.sport)

    print("\n" + "=" * 66)
    print(f"ready in {t_kg + t_idx:.1f}s  (kg {t_kg:.1f}s, index {t_idx:.1f}s)")
    print("next:  python run_benchmark.py --limit 5   # then drop --limit")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())