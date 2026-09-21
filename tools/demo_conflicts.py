"""Round 2 conflict-resolution demo / self-test.

Run:  python tools/demo_conflicts.py

Exercises every precedence rule with realistic corpus-shaped conflicts and
asserts the expected adjudication, so a regression fails loudly.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reasoning.conflicts import detect_conflicts, resolve, summarise

CASES = [
    ("agreement", "gold_medallist", [
        {"value": "Chen Ding", "doc_id": "Q1050909", "year": 2012},
        {"value": "Chen Ding", "doc_id": "Q26233122", "year": 2012},
    ], "Chen Ding", "no_conflict"),

    ("doping reallocation", "gold_medallist", [
        {"value": "Athlete A", "doc_id": "Q111", "year": 2012,
         "text": "Athlete A won the gold medal."},
        {"value": "Athlete B", "doc_id": "Q222", "year": 2016,
         "text": "Athlete A was disqualified for doping; the medal was "
                 "reallocated to Athlete B."},
    ], "Athlete B", "authority_correction"),

    ("superseded record", "olympic_record", [
        {"value": "9.85", "doc_id": "Q300", "year": 1996},
        {"value": "9.69", "doc_id": "Q301", "year": 2008},
        {"value": "9.63", "doc_id": "Q302", "year": 2012},
    ], "9.63", "recency"),

    ("dissolved nation", "nation", [
        {"value": "Soviet Union", "doc_id": "Q400"},
        {"value": "Russia", "doc_id": "Q401"},
    ], "Russia", "entity_succession"),

    ("undated disagreement", "venue", [
        {"value": "London Velopark", "doc_id": "Q500"},
        {"value": "London Velopark", "doc_id": "Q501"},
        {"value": "Lee Valley VeloPark", "doc_id": "Q502"},
    ], "London Velopark", "majority"),
]


def main() -> int:
    failures = 0
    print("=" * 70)
    print(" Round 2: conflicting / evolving fact resolution")
    print("=" * 70)

    for label, field_name, candidates, want_value, want_rule in CASES:
        r = resolve(field_name, candidates)
        ok = (str(r.resolved) == want_value and r.rule == want_rule)
        failures += 0 if ok else 1
        print(f"\n[{'PASS' if ok else 'FAIL'}] {label}  ({field_name})")
        print(f"   candidates : {[c['value'] for c in candidates]}")
        print(f"   resolved   : {r.resolved!r}   rule={r.rule} "
              f"confidence={r.confidence}")
        print(f"   because    : {r.explanation}")
        if r.superseded:
            print(f"   superseded : {[s['value'] for s in r.superseded]}")
        if not ok:
            print(f"   EXPECTED   : {want_value!r} via {want_rule}")

    facts = {field: cands for _, field, cands, _, _ in CASES}
    report = summarise([resolve(f, c) for f, c in facts.items()])
    print("\n" + "-" * 70)
    print(f" fields examined : {report['fields_examined']}")
    print(f" conflicts found : {report['conflicts_found']}")
    print(f" rules applied   : {report['rules_applied']}")
    print(f" min confidence  : {report['min_confidence']}")
    print("-" * 70)
    print(" ALL PASS" if not failures else f" {failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
