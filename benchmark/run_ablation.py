"""One-command CLI for running the 8-Way Architectural Ablation Matrix.

Usage:
  python -m benchmark.run_ablation
  python benchmark/run_ablation.py [--per-type N] [--allow-llm] [--out-path PATH]
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

from benchmark.evaluator import Evaluator
from config import config
from kg.backend import open_graph
from pipelines.ablations import build_8way_ablations
from retrieval import load_index
from tools.run_ablation_study import run_ablation_study, select_stratified_questions
from utils.llm import LLMHelper


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 8-Way Architectural Ablation Matrix")
    parser.add_argument("--questions", default=config.benchmark.public_questions_path,
                        help="Path to questions file (default: eval_public.jsonl)")
    parser.add_argument("--per-type", type=int, default=2,
                        help="Number of questions per category for stratified evaluation (default: 2, pass 0 for all)")
    parser.add_argument("--allow-llm", action="store_true",
                        help="Explicitly permit provider calls (off by default)")
    parser.add_argument("--out-path", default="", help="Custom output path for results JSON")
    args = parser.parse_args()

    print(f"\n{'='*74}\nTigerGraph Agentic GraphRAG: 8-Way Architectural Ablation CLI\n{'='*74}")
    print(f"Questions Source: {args.questions}")
    print(f"Per-Type Sample:  {args.per_type if args.per_type > 0 else 'ALL'}")
    print(f"LLM Mode:         {'Active (explicitly enabled)' if args.allow_llm else 'Deterministic (default)'}\n")

    kg = open_graph(config.benchmark.corpus_path, config,
                    cache_path=config.benchmark.kg_cache_path,
                    force_local=True)
    index = load_index(config.benchmark.corpus_path, kg,
                       vector_backend=os.getenv("VECTOR_BACKEND", config.benchmark.vector_backend))
    llm = LLMHelper() if args.allow_llm else None

    pipelines = build_8way_ablations(index, llm=llm)
    evaluator = Evaluator(llm=llm, use_llm_judge=args.allow_llm)

    if args.per_type > 0:
        questions = select_stratified_questions(args.questions, per_type=args.per_type)
    else:
        with open(args.questions, "r", encoding="utf-8") as fh:
            questions = [json.loads(line) for line in fh if line.strip()]

    study_results = run_ablation_study(
        questions, pipelines, evaluator, verbose=True,
        dataset_path=args.questions,
        run_mode="live" if args.allow_llm else "deterministic",
    )

    out_path = args.out_path or os.path.join(_ROOT, "results", "ablation_8way_matrix.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(study_results, fh, indent=2, ensure_ascii=False)
    print(f"Saved canonical 8-way ablation matrix to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
