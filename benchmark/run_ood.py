"""One-command CLI for running the Out-of-Distribution (OOD) 7-dimension reasoning benchmark.

Usage:
  python -m benchmark.run_ood
  python benchmark/run_ood.py [--pipeline PIPELINE] [--no-llm] [--out-path PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmark.generalization import GENERALIZATION_SUITE, run_generalization_evaluation
from config import config
from kg.backend import open_graph
from pipelines import build_pipelines
from retrieval import load_index
from utils.llm import LLMHelper


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 7-dimension OOD Generalization Suite")
    parser.add_argument("--pipeline", default="Router", choices=["Router", "Agentic GraphRAG", "GraphRAG", "RAG"],
                        help="Pipeline to evaluate (default: Router)")
    parser.add_argument("--no-llm", action="store_true", help="Run without calling external LLM APIs")
    parser.add_argument("--out-path", default="", help="Custom output JSON path")
    args = parser.parse_args()

    print(f"\n{'='*72}\nTigerGraph Agentic GraphRAG: 7-Dimension OOD Benchmark CLI\n{'='*72}")
    print(f"Target Pipeline: {args.pipeline}")
    print(f"LLM Available:   {not args.no_llm}")
    print(f"OOD Suite Size:  {len(GENERALIZATION_SUITE)} questions spanning 7 reasoning dimensions\n")

    kg = open_graph(config.benchmark.corpus_path, config,
                    cache_path=config.benchmark.kg_cache_path,
                    force_local=True)
    index = load_index(config.benchmark.corpus_path, kg,
                       vector_backend=os.getenv("VECTOR_BACKEND", config.benchmark.vector_backend))
    llm = None if args.no_llm else LLMHelper()
    pipes = build_pipelines(index, llm, config, with_router=True)
    target_pipe = next((p for p in pipes if p.name == args.pipeline), None)
    if not target_pipe:
        print(f"Error: Pipeline '{args.pipeline}' not found in built pipelines.")
        return 1

    summary = run_generalization_evaluation(target_pipe, verbose=True)

    out_path = args.out_path or os.path.join(_ROOT, "results", "generalization_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"Saved canonical OOD results to: {out_path}")

    # Return 0 if all questions pass, 1 otherwise
    return 0 if summary.get("total_passed") == len(GENERALIZATION_SUITE) else 1


if __name__ == "__main__":
    sys.exit(main())
