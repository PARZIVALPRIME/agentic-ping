"""Verify the numbers quoted in docs/ablation_study.md against the real
summary file. Documentation that drifts from the data is worse than none.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "results", "ablation_summary.json")

# What docs/ablation_study.md claims: name -> (overall, aggregation, superlative)
CLAIMED = {
    "RAG": (0.39, 0.00, 0.00),
    "GraphRAG": (0.61, 0.33, 0.50),
    "Agentic GraphRAG": (1.00, 1.00, 1.00),
    "Router": (1.00, 1.00, 1.00),
    "Agentic-NoPlanner": (0.00, 0.00, 0.00),
    "Agentic-NoEnumeration": (0.86, 0.38, 0.90),
    "Agentic-NoVerifier": (1.00, 1.00, 1.00),
    "Agentic-NoGapDetector": (1.00, 1.00, 1.00),
}
TOL = 0.015


def _acc(node) -> float:
    if not isinstance(node, dict):
        return float("nan")
    for key in ("accuracy", "exact_match", "acc"):
        if key in node:
            try:
                return float(node[key])
            except (TypeError, ValueError):
                pass
    correct, total = node.get("correct"), node.get("total")
    if isinstance(correct, (int, float)) and total:
        return correct / total
    return float("nan")


def main() -> int:
    if not os.path.exists(SUMMARY):
        print(f"missing {SUMMARY}\nrun: python run_benchmark.py --no-llm "
              f"--no-tg --ablations --out results/ablation_study.json "
              f"--summary results/ablation_summary.json")
        return 1

    summary = json.load(open(SUMMARY, encoding="utf-8"))
    pipelines = summary.get("pipelines", {})

    print(f"{'pipeline':<26} {'overall':>18} {'aggregation':>18} {'superlative':>18}")
    print("-" * 84)

    failures = 0
    for name, (c_all, c_agg, c_sup) in CLAIMED.items():
        node = pipelines.get(name)
        if node is None:
            print(f"{name:<26} {'MISSING from summary':>18}")
            failures += 1
            continue
        by_type = node.get("by_type", {}) or {}
        actual = (_acc(node), _acc(by_type.get("aggregation", {})),
                  _acc(by_type.get("superlative", {})))
        claimed = (c_all, c_agg, c_sup)

        cells, bad = [], False
        for a, c in zip(actual, claimed):
            match = (a == a) and abs(a - c) <= TOL  # a == a filters NaN
            bad = bad or not match
            cells.append(f"{a:.2f} vs {c:.2f}{'' if match else ' X'}")
        failures += 1 if bad else 0
        print(f"{name:<26} " + " ".join(f"{c:>18}" for c in cells))

    print("-" * 84)
    if failures:
        print(f"{failures} MISMATCH(ES): update docs/ablation_study.md to match "
              f"the data (or re-run the study).")
        return 1
    print("docs/ablation_study.md matches results/ablation_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
