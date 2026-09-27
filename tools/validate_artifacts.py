"""Publication gate for benchmark and ablation artifacts; makes no model calls."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.schema import validate_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate canonical evaluation provenance.")
    parser.add_argument("artifacts", nargs="*", default=[
        "results/metrics_summary.json", "results/ablation_8way_matrix.json"])
    args = parser.parse_args()
    failures = 0
    for path in args.artifacts:
        if not os.path.exists(path):
            print(f"MISSING {path}")
            failures += 1
            continue
        with open(path, encoding="utf-8") as handle:
            artifact = json.load(handle)
        meta = artifact.get("evaluation")
        if not isinstance(meta, dict):
            print(f"STALE {path}: lacks canonical evaluation provenance; do not publish as current")
            failures += 1
            continue
        defects = validate_summary(meta)
        if defects:
            print(f"INVALID {path}: {', '.join(defects)}")
            failures += 1
        else:
            print(f"OK {path}: {meta['run_mode']} dataset_size={meta['dataset_size']}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
