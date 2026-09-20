"""Audit a hidden-evaluation answer sheet (no gold answers available).

Reports, per pipeline: how many questions produced an answer, how many answers
are empty, how many look like refusals/prose instead of a corpus span, and how
answers changed against a previous snapshot. Used to compare runs when the gold
answers are withheld (accuracy can only be scored by the organisers).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List

try:  # answers contain non-ASCII venue/name spans; cp1252 consoles choke on them
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MALFORMED = (
    "does not contain", "no information", "cannot answer", "cannot determine",
    "unable to", "insufficient", "i don't know", "i see that",
    "based on the context", "the provided context", "not specified",
    "no relevant",
)

PROSE_CHARS = 90


def load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else []


def audit(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"records": 0, "answered": 0, "empty": 0, "malformed": 0,
                 "prose": 0, "tokens": 0})
    for record in entries:
        for name, result in (record.get("pipelines") or {}).items():
            slot = stats[name]
            slot["records"] += 1
            answer = (result.get("answer") or "").strip()
            slot["tokens"] += int(result.get("total_tokens", 0) or 0)
            if not answer:
                slot["empty"] += 1
                continue
            slot["answered"] += 1
            low = answer.lower()
            if any(marker in low for marker in MALFORMED):
                slot["malformed"] += 1
            if len(answer) > PROSE_CHARS:
                slot["prose"] += 1
    return dict(stats)


def agreement(new_entries: List[Dict[str, Any]],
              old_entries: List[Dict[str, Any]]) -> Dict[str, Counter]:
    """Per-pipeline counts of same/different answers between two runs."""
    old = {r.get("qid"): r.get("pipelines") or {} for r in old_entries}
    changes: Dict[str, Counter] = defaultdict(Counter)
    for record in new_entries:
        previous = old.get(record.get("qid")) or {}
        for name, result in (record.get("pipelines") or {}).items():
            before = previous.get(name) or {}
            a = (result.get("answer") or "").strip().lower()
            b = (before.get("answer") or "").strip().lower()
            if not previous:
                changes[name]["no_baseline"] += 1
            elif a == b:
                changes[name]["same"] += 1
            else:
                changes[name]["changed"] += 1
    return {k: v for k, v in changes.items()}


def main(argv: List[str]) -> int:
    current = argv[0] if argv else "results/hidden_results.json"
    baseline = argv[1] if len(argv) > 1 else ""
    pipeline = argv[2] if len(argv) > 2 else ""
    entries = load(current)
    entries = load(current)
    print(f"# hidden answer-sheet audit :: {current} ({len(entries)} questions)")
    for name, slot in sorted(audit(entries).items()):
        print(f"  {name:<18s} records={slot['records']} answered={slot['answered']} "
              f"empty={slot['empty']} refusal={slot['malformed']} "
              f"long_prose={slot['prose']} tokens={slot['tokens']}")
    if baseline:
        old = load(baseline)
        print(f"\n# agreement vs {baseline} ({len(old)} questions)")
        for name, counts in sorted(agreement(entries, old).items()):
            print(f"  {name:<18s} same={counts['same']} changed={counts['changed']} "
                  f"no_baseline={counts['no_baseline']}")
        if pipeline:
            print(f"\n# changed answers :: {pipeline}")
            diff_answers(entries, old, pipeline)
    return 0


def diff_answers(new_entries: List[Dict[str, Any]],
                 old_entries: List[Dict[str, Any]], pipeline: str,
                 qtypes: tuple = ()) -> None:
    """Print question/old/new triples for one pipeline's changed answers."""
    old = {r.get("qid"): r.get("pipelines") or {} for r in old_entries}
    for record in new_entries:
        if qtypes and record.get("qtype") not in qtypes:
            continue
        before = (old.get(record.get("qid")) or {}).get(pipeline) or {}
        after = (record.get("pipelines") or {}).get(pipeline) or {}
        a = (after.get("answer") or "").strip()
        b = (before.get("answer") or "").strip()
        if a == b:
            continue
        print(f"[{record.get('qid')}:{record.get('qtype')}]")
        print(f"   old: {b[:110]!r}  ({before.get('synth_source') or before.get('method')})")
        print(f"   new: {a[:110]!r}  ({after.get('synth_source') or after.get('method')})")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
