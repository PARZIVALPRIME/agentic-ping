"""Verify the numbers quoted in docs/ against the real result files.
Documentation that drifts from the data is worse than none.

Checks:
  * docs/ablation_study.md   vs results/ablation_summary.json
  * docs/baseline_ceiling.md vs results/baseline_sweep.json
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "results", "ablation_summary.json")
SWEEP = os.path.join(ROOT, "results", "baseline_sweep.json")

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

# What docs/baseline_ceiling.md claims: (pipeline, k) -> (accuracy, ctx_tokens)
CLAIMED_SWEEP = {
    ("RAG", 5): (0.39, 251),
    ("RAG", 160): (0.91, 9680),
    ("GraphRAG", 20): (0.64, 1768),
    ("GraphRAG", 40): (0.61, 3511),
    ("GraphRAG", 160): (0.74, 13785),
}

TOL = 0.015
TOK_TOL = 0.02  # 2% relative on token counts


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


def _check_sweep() -> int:
    """docs/baseline_ceiling.md vs results/baseline_sweep.json."""
    if not os.path.exists(SWEEP):
        print(f"\nskip: no {SWEEP} yet "
              f"(run: python tools/baseline_sweep.py --ks 5,10,20,40,80,160)")
        return 0

    data = json.load(open(SWEEP, encoding="utf-8"))
    results = data.get("results", {})

    print(f"\n{'baseline sweep':<26} {'accuracy':>20} {'ctx tokens':>22}")
    print("-" * 84)

    failures = 0
    for (name, k), (c_acc, c_tok) in sorted(CLAIMED_SWEEP.items()):
        row = (results.get(name) or {}).get(str(k)) or (results.get(name) or {}).get(k)
        if not row:
            print(f"{name + f' k={k}':<26} {'MISSING from sweep':>20}")
            failures += 1
            continue
        a_acc = float(row.get("accuracy", float("nan")))
        a_tok = float(row.get("avg_context_tokens", float("nan")))
        acc_ok = abs(a_acc - c_acc) <= TOL
        tok_ok = abs(a_tok - c_tok) <= max(1.0, TOK_TOL * c_tok)
        failures += 0 if (acc_ok and tok_ok) else 1
        print(f"{name + f' k={k}':<26} "
              f"{f'{a_acc:.2f} vs {c_acc:.2f}' + ('' if acc_ok else ' X'):>20} "
              f"{f'{a_tok:,.0f} vs {c_tok:,}' + ('' if tok_ok else ' X'):>22}")
    return failures


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

    failures += _check_sweep()

    print("-" * 84)
    if failures:
        print(f"{failures} MISMATCH(ES): update the docs to match the data "
              f"(or re-run the study).")
        return 1
    print("docs match the result files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
