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
            "winner": {"doc_id": winner.doc_id, "title": winner.title,
                       "competitors": winner.competitors},
            "direction": spec.direction,
            "ranked": [{"doc_id": n.doc_id, "title": n.title,
                        "competitors": n.competitors, "rank": i + 1}
                       for i, n in enumerate(ranked[:10])],
            "runner_up": ({"doc_id": runner_up.doc_id, "title": runner_up.title,
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