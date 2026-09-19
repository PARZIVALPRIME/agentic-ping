"""Aggregation agent: count events satisfying a threshold across many documents.

This agent is the reason the agentic pipeline exists. Counting questions cannot
be answered by top-k retrieval at all: the answer depends on the *complete*
event set for a sport at a Games. The agent therefore (1) enumerates the set
from the graph, (2) filters on the structured field, and (3) reports how many
documents lacked the field so the gap detector can chase them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from reasoning.query_parser import QuerySpec
from reasoning.solvers import _threshold_pass, resolve_sport


class Aggregator:
    """Counts/filters over an exhaustive candidate set."""

    def __init__(self, kg) -> None:
        self.kg = kg

    def run(self, spec: QuerySpec, events: List[Any]) -> Dict[str, Any]:
        kept, rejected, missing = [], [], []
        for node in events:
            if node.competitors is None:
                missing.append(node)
                continue
            if _threshold_pass(node.competitors, spec.comparator, spec.threshold):
                kept.append(node)
            else:
                rejected.append(node)

        complete = len(missing) == 0 and bool(events)
        return {
            "count": len(kept),
            "comparator": spec.comparator,
            "threshold": spec.threshold,
            "candidates": len(events),
            "kept": [{"doc_id": n.doc_id, "title": n.title,
                      "competitors": n.competitors} for n in kept],
            "rejected": [{"doc_id": n.doc_id, "title": n.title,
                          "competitors": n.competitors} for n in rejected],
            "missing_field": [{"doc_id": n.doc_id, "title": n.title} for n in missing],
            "field_completeness": (round(1.0 - len(missing) / len(events), 3)
                                   if events else 0.0),
            "exhaustive": complete,
        }

    def verify_by_recount(self, spec: QuerySpec, events: List[Any]) -> Dict[str, Any]:
        """Independent recount used by the evidence evaluator.

        Recomputes the filter using a different code path (list comprehension
        over a set of the passing doc_ids) and compares. A mismatch would signal
        a bug in either the candidate set or the filter.
        """
        values = {n.doc_id: n.competitors for n in events if n.competitors is not None}
        op = {
            "gt": lambda v: v > (spec.threshold or 0),
            "gte": lambda v: v >= (spec.threshold or 0),
            "lt": lambda v: v < (spec.threshold or 0),
            "lte": lambda v: v <= (spec.threshold or 0),
        }.get(spec.comparator, lambda v: v > (spec.threshold or 0))
        recounted = len({d for d, v in values.items() if op(v)})
        return {"recount": recounted, "documents_with_field": len(values)}