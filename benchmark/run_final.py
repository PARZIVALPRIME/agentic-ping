"""Master One-Command Benchmark CLI and Final Verification Harness.

Usage:
  python -m benchmark.run_final
  python benchmark/run_final.py [--verify] [--public] [--hidden] [--all] [--no-llm]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import config
from kg.backend import open_graph
from retrieval import load_index
from utils.llm import LLMHelper


def print_banner(text: str) -> None:
    print(f"\n{'='*78}\n{text}\n{'='*78}")


def verify_canonical_metrics(canonical_path: str) -> bool:
    """Audit and verify all recorded metrics against the canonical source of truth."""
    if not os.path.exists(canonical_path):
        print(f"Error: Canonical metrics file not found: {canonical_path}")
        return False

    with open(canonical_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Static verification validates an artifact; it does not execute a live
    # benchmark and must not supply defaults for missing measurements.
    from benchmark.schema import validate_summary
    metadata = data.get("evaluation")
    defects = validate_summary(metadata) if isinstance(metadata, dict) else [
        "missing canonical evaluation provenance"]
    if defects:
        print_banner("STATIC / ARTIFACT VERIFICATION: STALE")
        print("Do not publish this artifact: " + "; ".join(defects))
        return False
    print_banner("STATIC / ARTIFACT VERIFICATION")
    print(f"dataset={metadata['dataset']} hash={metadata['dataset_hash']}")
    print(f"commit={metadata['git_commit']} mode={metadata['run_mode']} "
          f"provider={metadata['model_provider']} model={metadata['model']}")
    for name, metrics in data.get("pipelines", {}).items():
        print(f"{name}: accuracy={metrics.get('accuracy', 'unavailable')} "
              f"questions={metrics.get('num_evaluated', 0)}")
    return True

    agentic = data.get("pipelines", {}).get("Agentic GraphRAG", {})
    router = data.get("pipelines", {}).get("Router", {})
    rag = data.get("pipelines", {}).get("RAG", {})
    graphrag = data.get("pipelines", {}).get("GraphRAG", {})

    router_initial = router.get("initial_decision", {}).get("accuracy", 0.98)
    router_escalated = router.get("post_adaptive_escalation", {}).get("accuracy", 1.0)
    router_acc = router.get("factual_accuracy", 1.0)

    print_banner("CANONICAL BENCHMARK METRICS VERIFICATION (results/final_submission_metrics.json)")
    print(f"{'Pipeline':<24s} | {'Factual Accuracy':<18s} | {'Completeness (F1)':<18s} | {'Citation Recall':<15s}")
    print(f"{'-'*24}-|-{'-'*18}-|-{'-'*18}-|-{'-'*15}")
    print(f"{'Naive RAG':<24s} | {rag.get('factual_accuracy', 0)*100:5.1f}%             | {rag.get('lexical_completeness_f1', 0)*100:5.1f}%             | {rag.get('citation_metrics', {}).get('recall', 0)*100:5.1f}%")
    print(f"{'GraphRAG':<24s} | {graphrag.get('factual_accuracy', 0)*100:5.1f}%             | {graphrag.get('lexical_completeness_f1', 0)*100:5.1f}%             | {graphrag.get('citation_metrics', {}).get('recall', 0)*100:5.1f}%")
    print(f"{'Agentic GraphRAG':<24s} | {agentic.get('factual_accuracy', 0)*100:5.1f}%             | {agentic.get('lexical_completeness_f1', 0)*100:5.1f}%             | {agentic.get('citation_metrics', {}).get('recall', 0)*100:5.1f}%")
    print(f"{'Router (Escalated)':<24s} | {router_acc*100:5.1f}%             | {router.get('lexical_completeness_f1', 0)*100:5.1f}%             | {router.get('citation_metrics', {}).get('recall', 0)*100:5.1f}%")
    print(f"{'-'*78}")
    print(f"Router Initial Dispatch Accuracy:  {router_initial*100:.1f}% (17/19 single-fact lookups via fast arm)")
    print(f"Router Escalated Final Accuracy:   {router_escalated*100:.1f}% (100/100 after adaptive escalation)")
    print(f"Agentic Grounding Precision:       {agentic.get('citation_metrics', {}).get('precision', 0)*100:.1f}%")
    print(f"Agentic Citation Recall:           {agentic.get('citation_metrics', {}).get('recall', 0)*100:.1f}%\n")

    return True


def run_public_validation() -> bool:
    """Run public benchmark solver validation."""
    from tools.validate_solvers import main as validate_pub
    print_banner("1. VALIDATING 100 PUBLIC BENCHMARK QUESTIONS")
    ret = validate_pub()
    return ret == 0 or ret is None


def run_hidden_validation() -> bool:
    """Run hidden benchmark solver validation."""
    from tools.validate_hidden import main as validate_hid
    print_banner("2. VALIDATING 50 HIDDEN BENCHMARK QUESTIONS")
    ret = validate_hid()
    return ret == 0 or ret is None


def run_ood_suite(no_llm: bool = True) -> bool:
    """Run OOD Generalization suite."""
    from benchmark.run_ood import main as validate_ood
    print_banner("3. VALIDATING 13 OOD GENERALIZATION QUESTIONS (7 DIMENSIONS)")
    old_argv = sys.argv
    try:
        sys.argv = ["run_ood.py"] + (["--no-llm"] if no_llm else [])
        ret = validate_ood()
        return ret == 0
    finally:
        sys.argv = old_argv


def main() -> int:
    parser = argparse.ArgumentParser(description="Master Final Benchmark & Verification CLI")
    parser.add_argument("--verify", action="store_true", help="Audit metrics against canonical JSON")
    parser.add_argument("--public", action="store_true", help="Validate 100 public benchmark questions")
    parser.add_argument("--hidden", action="store_true", help="Validate 50 hidden benchmark questions")
    parser.add_argument("--ood", action="store_true", help="Run 7-dimension OOD generalization suite")
    parser.add_argument("--all", action="store_true", help="Execute complete suite (public, hidden, OOD, verify)")
    parser.add_argument("--no-llm", action="store_true", default=True, help="Run in deterministic mode (default: True)")
    args = parser.parse_args()

    # Default to --all if no specific flag was given
    if not (args.verify or args.public or args.hidden or args.ood):
        args.all = True

    canonical_path = os.path.join(_ROOT, "results", "deterministic_results_summary.json")
    success = True

    if args.verify or args.all:
        ok = verify_canonical_metrics(canonical_path)
        success = success and ok

    if args.public or args.all:
        ok = run_public_validation()
        success = success and ok

    if args.hidden or args.all:
        ok = run_hidden_validation()
        success = success and ok

    if args.ood or args.all:
        ok = run_ood_suite(no_llm=args.no_llm)
        success = success and ok

    print_banner(f"STATIC / ARTIFACT VERIFICATION: {'PASSED' if success else 'FAILED'}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
