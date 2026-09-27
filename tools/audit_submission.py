"""Deterministic pre-submission artifact audit; never invokes a model provider."""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from benchmark.schema import validate_summary


def load(relative: str):
    path = os.path.join(ROOT, relative)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    failures = 0
    deterministic = load("results/deterministic_results_summary.json")
    if not isinstance(deterministic, dict):
        print("FAIL deterministic public summary is missing")
        failures += 1
    else:
        defects = validate_summary(deterministic.get("evaluation", {}))
        if defects:
            print("FAIL deterministic public summary: " + "; ".join(defects))
            failures += 1
        else:
            meta = deterministic["evaluation"]
            print(f"OK deterministic public artifact: {meta['dataset_size']} questions; "
                  f"mode={meta['run_mode']}; backend={meta['backend']['active']}")
            if meta["run_mode"] != "deterministic":
                print("FAIL deterministic artifact has an incorrect run mode")
                failures += 1

    hidden = load("results/hidden_submission.json")
    if not isinstance(hidden, dict):
        print("FAIL hidden submission artifact is missing")
        failures += 1
    else:
        submission = hidden.get("submission", {})
        count = submission.get("num_completed")
        print(f"OK hidden submission artifact: completed={count}; local gold unavailable")

    ablation = load("results/ablation_8way_matrix.json")
    if not isinstance(ablation, dict):
        print("FAIL ablation artifact is missing")
        failures += 1
    else:
        print(f"OK ablation artifact: {ablation.get('num_questions')} questions; "
              "historical artifact lacks canonical provenance")

    stale = load("results/metrics_summary.json")
    if isinstance(stale, dict) and "evaluation" not in stale:
        print("WARN metrics_summary.json is historical/stale and must not be published as current")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
