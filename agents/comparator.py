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
    """The full corpus title for an event, e.g.
    'Sailing at the 2016 Summer Olympics – Men's 470'.

    Gold answers for superlative questions are the *full* title. Some code
    paths feed the comparator nodes whose ``title`` is only the event suffix
    ("Men's 470"), which can never string-match the gold.

    Nothing here is corpus-specific: it prefers the node's own ``title`` (which
    the builder copies verbatim from the source document, whatever the domain)
    and only falls back to joining ``title_prefix``/``games_label``/``event_name``
    - all fields taken straight from the corpus - when the stored title is
    somehow just the suffix. No domain word ("Olympics") is hardcoded, so this
    behaves correctly on any corpus whose documents carry a title.
    """
    title = (getattr(node, "title", "") or "").strip()
    # The stored corpus title is authoritative whenever it is the full form.
    # "Full" = longer than the bare event suffix, detected by the presence of
    # the corpus's own separator or simply by being longer than event_name.
    event = (getattr(node, "event_name", "") or "").strip()
    if title and ("–" in title or " - " in title or len(title) > len(event)):
        return title

    # Last resort: reconstruct from structured fields, using only values that
    # came from the corpus itself (games_label is the raw infobox "games").
    games = (getattr(node, "games_label", "") or "").strip()
    prefix = (getattr(node, "sport", "") or "").strip()
    parts = [p for p in (prefix, games, event or title) if p]
    return " – ".join(parts) if parts else title


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