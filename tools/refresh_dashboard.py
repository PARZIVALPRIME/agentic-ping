"""Regenerate a published dashboard page from a results file, and check it.

Publishing a page is three steps that have to agree with one another: the
summary has to describe the same run as the results file, the page has to be
built with that summary, and the page has to survive the validator. Done by hand
they drift. Concretely: ``results/public_results.json`` held 100 questions
beside a ``results/metrics_summary.json`` describing 10, because a later smoke
run reused the default paths - and the summary drives the headline table while
the entries drive the per-question table, the traces and the provider panel, so
that page would have shown two different runs at once.

This tool makes the mismatch impossible to publish by accident: it compares the
summary's question count against the results file and stops, unless
``--rebuild-summary`` says out loud that the summary should be rebuilt from the
results file (in which case provider telemetry is reported as unknown, because
only a live run knows it).

Usage::

    python tools/refresh_dashboard.py                        # public -> index.html
    python tools/refresh_dashboard.py --all                  # + hidden page
    python tools/refresh_dashboard.py --target hidden
    python tools/refresh_dashboard.py --name partial_public.html \
        --summary results/_partial_public_summary.json --rebuild-summary
    python tools/refresh_dashboard.py --results results/deterministic_results.json \
        --summary results/deterministic_results_summary.json \
        --name baseline.html --allow-deterministic

Exit codes: 0 published and valid, 1 the page or its data failed a check,
2 nothing to publish.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from benchmark.dashboard_generator import generate   # noqa: E402
from benchmark.runner import load_entries, summarize_entries  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover - non-reconfigurable stream
    pass

#: The page each sweep is published to, and the pair of files it is built from.
TARGETS = {
    "public": {"results": "results/public_results.json",
               "summary": "results/metrics_summary.json",
               "name": "index.html"},
    "hidden": {"results": "results/hidden_llm.json",
               "summary": "results/hidden_llm_summary.json",
               "name": "dashboard_hidden_llm.html"},
}


def _summary_questions(path: str) -> int:
    """``num_questions`` of a summary file, or 0 when it is missing/unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return int((json.load(fh) or {}).get("num_questions") or 0)
    except (OSError, ValueError, TypeError):
        return 0



def _prepare_summary(summary_path: str, entries: List[Dict[str, Any]],
                     source: str, rebuild: bool) -> None:
    """Make the summary describe the same run as the results file, or stop."""
    have, want = _summary_questions(summary_path), len(entries)
    if have == want and not rebuild:
        print(f"summary : {summary_path} ({have} questions, matches {source})")
        return
    if not rebuild:
        raise SystemExit(
            f"\n{summary_path} describes {have or 'no'} question(s) but "
            f"{source} holds {want}.\n"
            f"  Refusing: the page's headline table comes from the summary while "
            f"its per-question table, traces and provider panel come from the "
            f"results file, so it would show two different runs at once.\n"
            f"  fix: re-run the benchmark with --summary {summary_path}, or pass "
            f"--rebuild-summary to build the summary from {source} (provider "
            f"telemetry is then reported as unknown, because only a live run "
            f"knows it).")
    summary = summarize_entries(entries, summary_path, source=source)
    print(f"summary : rebuilt from {source} ({summary['num_questions']} "
          f"questions, run_mode={summary.get('run_mode')}) -> {summary_path}")


def refresh(results: str, summary: str, name: str, out_dir: str = "dashboard",
            rebuild: bool = False, require_llm: bool = True) -> int:
    """Build one page from a results file and validate it.

    ``require_llm`` is on by default: publishing a run in which the provider
    never answered as the front-page accuracy table is the single most damaging
    mix-up this repository guards against, and the generator refuses it.
    """
    if not os.path.exists(results):
        print(f"missing results file: {results}")
        return 2
    entries = load_entries(results)
    if not entries:
        print(f"{results}: no entries")
        return 2

    _prepare_summary(summary, entries, results, rebuild)
    page = generate(results, summary, out_dir, require_llm=require_llm, name=name)

    validator = os.path.join(ROOT, "tools", "validate_dashboard.py")
    rc = subprocess.call([sys.executable, validator, os.path.basename(page)],
                         cwd=ROOT)
    if rc != 0:
        print(f"validation FAILED for {page} (exit {rc})")
        return 1
    print(f"published: {page} (validated)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate a dashboard page from "
                                             "a results file, then validate it.")
    ap.add_argument("--results", default=TARGETS["public"]["results"])
    ap.add_argument("--summary", default=TARGETS["public"]["summary"])
    ap.add_argument("--name", default=TARGETS["public"]["name"],
                    help="output file inside --out; the page links "
                         "styles.css/dashboard.js relatively")
    ap.add_argument("--out", default="dashboard")
    ap.add_argument("--rebuild-summary", action="store_true",
                    help="rebuild the summary from the results file when the "
                         "two describe different runs (telemetry becomes unknown)")
    ap.add_argument("--all", action="store_true",
                    help="refresh the public page and then the hidden page")
    ap.add_argument("--target", choices=sorted(TARGETS),
                    help="shorthand for one of the published pages")
    ap.add_argument("--allow-deterministic", action="store_true",
                    help="publish even when no pipeline recorded provider "
                         "output (for the --no-llm baseline page)")
    args = ap.parse_args(argv)

    if args.target:
        args.results = TARGETS[args.target]["results"]
        args.summary = TARGETS[args.target]["summary"]
        args.name = TARGETS[args.target]["name"]

    if args.all:
        # Public first: it is the page the accuracy table is quoted from.
        plan = [dict(TARGETS["public"]), dict(TARGETS["hidden"])]
    else:
        plan = [{"results": args.results, "summary": args.summary,
                 "name": args.name}]

    for target in plan:
        print(f"\n=== {target['name']} ===")
        rc = refresh(target["results"], target["summary"], target["name"],
                     out_dir=args.out, rebuild=args.rebuild_summary,
                     require_llm=not args.allow_deterministic)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
