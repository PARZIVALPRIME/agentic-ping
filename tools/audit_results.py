"""Audit a results file: did the LLM actually contribute, and is it current?

Two failure modes this makes visible. Both produce a perfectly valid results
file that is easy to misread, and both have been observed in this project:

1. **Silent deterministic fallback.** A run whose provider calls all fail
   (quota exhausted, rate limiting, unusable completions) still finishes. Every
   answer then comes from the deterministic solvers, the token counters read
   zero, and the headline table reads "100% accuracy at 0 tokens" - which is a
   deterministic result wearing an LLM run's clothes. The tell is latency: the
   record still spends seconds waiting on a call that recorded nothing
   (``llm_calls == 0`` with ``latency_ms`` in the thousands).
2. **Stale results.** A file written by an earlier revision of the code can
   still be attached to today's dashboard. The audit checks each record against
   the fields the current pipelines write, and (optionally) compares answers
   against a ``--no-llm`` baseline.

Usage::

    python tools/audit_results.py                                  # public set
    python tools/audit_results.py results/regression_final.json
    python tools/audit_results.py --baseline results/deterministic_results.json

Exit code is non-zero when a pipeline took part in a run that used the provider
yet recorded no calls of its own, so the tool can gate numbers before they are
published.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover - non-tty
    pass

# Fields the current pipelines write into ``metadata``. A file that predates
# them was produced by an older revision, so its numbers no longer describe the
# code that ships alongside it.
CURRENT_METADATA = {
    "Agentic GraphRAG": ("agent_mode", "tool_calls", "plan_skips", "synth_source"),
}
# A record slower than this with no recorded call spent real time on something
# the counters did not attribute to a provider call.
UNACCOUNTED_MS = 1000


def load_entries(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else []


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    return (ordered[mid] if len(ordered) % 2
            else (ordered[mid - 1] + ordered[mid]) / 2.0)


def _status(slot: Dict[str, Any], file_used_provider: bool) -> str:
    if slot["with_calls"] and slot["records"]:
        return "llm" if slot["with_calls"] == slot["records"] else "partial-llm"
    return "provider-failed" if file_used_provider else "deterministic"


def audit(entries: List[Dict[str, Any]]) -> "OrderedDict[str, Dict[str, Any]]":
    """Aggregate per-pipeline evidence about how each answer was produced."""
    stats: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for entry in entries:
        for name, rec in (entry.get("pipelines") or {}).items():
            slot = stats.setdefault(name, {
                "records": 0, "evaluated": 0, "correct": 0,
                "with_calls": 0, "tokens": 0, "output_tokens": 0,
                "latencies": [], "unaccounted_records": 0, "unaccounted_ms": 0.0,
                "methods": Counter(), "stop_reasons": Counter(),
                "classifications": Counter(), "adjudications": Counter(),
                "tool_ops": 0, "schema_missing": [],
            })
            slot["records"] += 1
            ev = rec.get("evaluation")
            if ev:
                slot["evaluated"] += 1
                slot["correct"] += 1 if ev.get("is_correct") else 0
            calls = int(rec.get("llm_calls") or 0)
            slot["with_calls"] += 1 if calls > 0 else 0
            slot["tokens"] += int(rec.get("total_tokens") or 0)
            slot["output_tokens"] += int(rec.get("output_tokens") or 0)
            latency = float(rec.get("latency_ms") or 0.0)
            slot["latencies"].append(latency)
            if calls == 0 and latency >= UNACCOUNTED_MS:
                slot["unaccounted_records"] += 1
                slot["unaccounted_ms"] += latency
            slot["methods"][str(rec.get("method"))] += 1
            slot["stop_reasons"][str(rec.get("stop_reason"))] += 1
            slot["tool_ops"] += sum(
                1 for s in (rec.get("steps") or [])
                if str(s.get("operation", "")).startswith("tool:"))
            meta = rec.get("metadata") or {}
            slot["classifications"][str(
                meta.get("classification", {}).get("method"))] += 1
            adj = meta.get("adjudication")
            slot["adjudications"][str(adj.get("reason")) if isinstance(adj, dict)
                                  else "absent"] += 1
            if slot["records"] == 1:
                slot["schema_missing"] = [k for k in CURRENT_METADATA.get(name, ())
                                          if k not in meta]
    return stats


def compare_to_baseline(entries: List[Dict[str, Any]],
                        baseline: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """How much of this file agrees with a deterministic (``--no-llm``) run."""
    base: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for entry in baseline:
        for name, rec in (entry.get("pipelines") or {}).items():
            base.setdefault(name, {})[entry.get("qid")] = rec

    out: Dict[str, Dict[str, Any]] = {}
    for name, recs in base.items():
        same = total = 0
        for entry in entries:
            rec = (entry.get("pipelines") or {}).get(name)
            other = recs.get(entry.get("qid"))
            if not rec or not other:
                continue
            total += 1
            if str(rec.get("answer", "")).strip() == str(other.get("answer", "")).strip():
                same += 1
        if total:
            out[name] = {"compared": total, "identical_answers": same,
                         "share": round(same / total, 4)}
    return out
    return (ordered[mid] if len(ordered) % 2
            else (ordered[mid - 1] + ordered[mid]) / 2.0)
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Audit a benchmark results file.")
    ap.add_argument("results", nargs="?", default="results/public_results.json")
    ap.add_argument("--baseline", default="results/deterministic_results.json",
                    help="deterministic results file to compare answers against")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the baseline comparison (equivalent to no "
                         "deterministic run being available)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.results):
        print(f"no such results file: {args.results}")
        return 2
    entries = load_entries(args.results)
    if not entries:
        print(f"{args.results}: no entries")
        return 2

    stats = audit(entries)
    file_used_provider = any(s["with_calls"] for s in stats.values())

    print(f"audit: {args.results}  ({len(entries)} questions)")
    print(f"provider used somewhere in this file: {file_used_provider}")
    print()
    print(f"{'pipeline':<18s} {'n':>4s} {'acc':>7s} {'w/calls':>8s} "
          f"{'tokens':>8s} {'out':>6s} {'p50 ms':>8s} {'unacct':>7s} status")
    for name, slot in stats.items():
        acc = (slot["correct"] / slot["evaluated"]) if slot["evaluated"] else None
        print(f"{name:<18s} {slot['records']:>4d} "
              f"{('%.0f%%' % (acc * 100)) if acc is not None else 'n/a':>7s} "
              f"{slot['with_calls']:>8d} {slot['tokens']:>8d} "
              f"{slot['output_tokens']:>6d} {_median(slot['latencies']):>8.0f} "
              f"{slot['unaccounted_records']:>7d} {_status(slot, file_used_provider)}")

    print()
    for name, slot in stats.items():
        print(f"[{name}]")
        print(f"  method         : {dict(slot['methods'])}")
        print(f"  stop reasons   : {dict(slot['stop_reasons'])}")
        print(f"  classification : {dict(slot['classifications'])}")
        print(f"  adjudication   : {dict(slot['adjudications'])}")
        print(f"  tool: ops      : {slot['tool_ops']}")
        if slot["schema_missing"]:
            print(f"  schema         : MISSING {slot['schema_missing']} "
                  f"(file predates the current pipeline)")
        if slot["unaccounted_records"]:
            print(f"  unaccounted    : {slot['unaccounted_records']} record(s) took "
                  f">= {UNACCOUNTED_MS} ms with llm_calls=0 "
                  f"({slot['unaccounted_ms'] / 1000.0:.0f}s total)")

    if args.baseline and not args.no_baseline and os.path.exists(args.baseline):
        delta = compare_to_baseline(entries, load_entries(args.baseline))
        if delta:
            print()
            print(f"vs {args.baseline}:")
            for name, d in delta.items():
                print(f"  {name:<18s} identical answers {d['identical_answers']}"
                      f"/{d['compared']} ({d['share']:.0%})")

    problems: List[str] = []
    provider_failed = False
    stale = False
    for name, slot in stats.items():
        if _status(slot, file_used_provider) == "provider-failed":
            provider_failed = True
            problems.append(
                f"{name}: no provider call recorded in {slot['records']} records, "
                f"but other pipelines in the same file did call the provider - "
                f"its answers came from the deterministic path")
        if slot["schema_missing"]:
            stale = True
            problems.append(f"{name}: results written by an older revision "
                            f"(missing {slot['schema_missing']})")

    print()
    if problems:
        print("FINDINGS")
        for problem in problems:
            print(f"  - {problem}")
        if provider_failed:
            print("  The answers are the deterministic ones: re-run with quota "
                  "available before publishing these numbers as an LLM result.")
        if stale:
            print("  Re-run to refresh this file against the current pipelines; "
                  "the schema it was written with no longer matches the code "
                  "shipped beside it.")
        return 1
    if not file_used_provider:
        print("NOTE: no provider calls anywhere in this file - a deterministic "
              "run (expected for --no-llm). Accuracy is still valid; the cost "
              "columns are not comparable to an LLM run.")
    else:
        print("OK: every pipeline that took part in this file recorded provider "
              "activity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())