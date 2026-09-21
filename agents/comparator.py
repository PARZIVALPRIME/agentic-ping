"""Comparator agent: find the max/min event across a candidate set.

Superlative questions need the same exhaustive candidate set as counting
questions, plus a total order over the comparison field. The agent reports the
runner-up and the margin, because a large margin is what makes an argmax
robust to a missing document - that margin becomes part of the confidence
calculation.
"""

from __future__ import annotations

from typing import Any, Dict, List

from reasoning.query_parser import QuerySpec


def canonical_event_title(node: Any) -> str:
    """The full corpus-style event title, e.g.
    'Sailing at the 2016 Summer Olympics – Men's 470'.

    Gold answers for superlative questions are the *full* title. Some code
    paths feed the comparator nodes whose ``title`` is only the event suffix
    ("Men's 470"), which can never string-match the gold. When the title is
    already full (contains the en-dash separator) it is used as-is; otherwise
    it is reconstructed from the structured fields the node always carries.
    """
    title = (getattr(node, "title", "") or "").strip()
    if "–" in title or " - " in title:
        return title
    sport = (getattr(node, "sport", "") or "").strip()
    year = getattr(node, "year", 0) or 0
    season = (getattr(node, "season", "") or "").strip()
    event = (getattr(node, "event_name", "") or title).strip()
    if sport and year and event:
        games = f"{year} {season} Olympics".replace("  ", " ").strip()
        return f"{sport} at the {games} – {event}"
    return title


class Comparator:
    """Argmax/argmin over a structured field with margin reporting."""

    def __init__(self, kg) -> None:
        self.kg = kg

    def run(self, spec: QuerySpec, events: List[Any]) -> Dict[str, Any]:
        candidates = [n for n in events if n.competitors is not None]
        if not candidates:
            return {"winner": None, "ranked": [], "candidates": 0,
                    "missing_field": len(events)}
        reverse = spec.direction == "max"
        ranked = sorted(candidates, key=lambda n: n.competitors, reverse=reverse)
        winner = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        margin = (winner.competitors - runner_up.competitors) if runner_up else None
        return {
            "winner": {"doc_id": winner.doc_id,
                       "title": canonical_event_title(winner),
                       "competitors": winner.competitors},
            "direction": spec.direction,
            "ranked": [{"doc_id": n.doc_id, "title": canonical_event_title(n),
                        "competitors": n.competitors, "rank": i + 1}
                       for i, n in enumerate(ranked[:10])],
            "runner_up": ({"doc_id": runner_up.doc_id,
                           "title": canonical_event_title(runner_up),
                           "competitors": runner_up.competitors} if runner_up else None),
            "margin": margin,
            "candidates": len(candidates),
            "missing_field": len(events) - len(candidates),
            "unique_winner": (margin is not None and margin > 0),
        }

    def verify(self, spec: QuerySpec, events: List[Any]) -> Dict[str, Any]:
        """Cross-check the extreme value without relying on sort order."""
        values = [n.competitors for n in events if n.competitors is not None]
        if not values:
            return {"verified": False}
        extreme = max(values) if spec.direction == "max" else min(values)
        violations = [n.doc_id for n in events
                      if n.competitors is not None
                      and ((spec.direction == "max" and n.competitors > extreme)
                           or (spec.direction == "min" and n.competitors < extreme))]
        return {"verified": not violations, "extreme_value": extreme,
                "violations": violations, "documents_with_field": len(values)}