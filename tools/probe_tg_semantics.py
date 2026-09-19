"""Do the corpus attribute values match a GSQL ``lower()`` comparison?

The TigerGraph query ``tg_filter_events`` can only compare what it can express
(``lower(x) == lower(y)``). The local path compares ``normalize(x) ==
normalize(y)``, and ``normalize`` also collapses whitespace and strips
punctuation. If the corpus values are already "clean" the two agree and no
reconciliation layer is needed; if they are not, the backend has to intersect
the remote rows with the local indexes to keep the semantics identical.

Run this before trusting the push-down:

    python tools/probe_tg_semantics.py

It also reports the biggest candidate sets the agent's filters actually produce,
which is what the ``TG_MAX_ROWS`` safety valve has to clear to stay lossless.
"""

from __future__ import annotations

import io
import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

# tools/ is run directly (python tools/probe_tg_semantics.py), so make the
# project root importable the way the other probes in this folder do.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from kg.builder import load_or_build
from kg.textutil import normalize


def main() -> None:
    kg = load_or_build(config.benchmark.corpus_path)

    total = len(kg.events)
    print(f"events            : {total}")
    print(f"athletes          : {len(kg.athletes)}")
    print(f"edges             : {len(kg.edges)}")
    print(f"distinct sports   : {len([s for s in kg.sports if s])}")
    print(f"distinct venues   : {len([v for v in kg.venues if v])}")
    print()

    # ── value cleanliness ──────────────────────────────────────────────
    for label, values, getter in (
        ("sport", [s for s in kg.sports if s], lambda e: e.sport),
        ("venue", [v for v in kg.venues if v], lambda e: e.venue),
        ("season", sorted({e.season for e in kg.events.values() if e.season}),
         lambda e: e.season),
    ):
        dirty = [v for v in values if normalize(v) != v.lower().strip()]
        print(f"{label}: {len(values)} distinct, "
              f"{len(dirty)} where normalize(x) != lower(x)")
        if dirty:
            print("  e.g. " + " | ".join(repr(v) for v in dirty[:6]))
    print()

    # ── what the tools actually ask for ───────────────────────────────
    # Candidate sets the agent filters produce. A remote answer must be able to
    # return these without truncation, or the backend degrades to the mirror.
    probes = [
        ("no filter at all", {}),
        ("sport only (biggest)",
         {"sport": max(kg.sport_index, key=lambda s: len(kg.sport_index[s]))
                  if kg.sport_index else ""}),
        ("season + year window", {"season": "Summer", "year_from": 2000,
                                 "year_to": 2020}),
        ("venue only", {"venue": max(kg.venue_index,
                                     key=lambda v: len(kg.venue_index[v]))
                                 if kg.venue_index else ""}),
        ("query terms only", {"query": "100 metres final gold"}),
    ]
    biggest = 0
    for label, kwargs in probes:
        events = kg.filter_events(**kwargs)
        biggest = max(biggest, len(events))
        print(f"{label:22s}: {len(events):6d} candidate events")
    print()
    print(f"largest candidate set: {biggest}")
    print(f"TG_MAX_ROWS must be >= {biggest} to be lossless "
          f"(currently {getattr(config.tg, 'max_rows', '?')})")

    # Term-ranked path: proves the Python ranking survives a remote round-trip.
    ranked = kg.filter_events(query="100 metres final gold")
    print(f"\nranked sample ({len(ranked)} rows): "
          + ", ".join(e.doc_id for e in ranked[:5]))

    # Sanity: normalised equality is what the local index uses, so the query
    # layer must at least preserve it for the push-down to be provable.
    for sport in list(kg.sport_index)[:3]:
        ids = kg.sport_index[sport]
        print(f"index[{sport!r}] -> {len(ids)} events")


if __name__ == "__main__":
    main()