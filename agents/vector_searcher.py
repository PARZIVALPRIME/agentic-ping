"""Vector searcher agent: text retrieval over the passage/chunk index.

Used as (a) the primary evidence source for lookup and venue/date questions,
and (b) the fallback whenever entity linking or graph traversal fails to locate
a candidate set, so the agentic pipeline never ends a run with nothing.
"""

from __future__ import annotations

from typing import Any, Dict, List


class VectorSearcher:
    """Wraps the corpus index so retrieval is a first-class agent action."""

    def __init__(self, index) -> None:
        self.index = index

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return self.index.vector_search(query, top_k)

    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return self.index.hybrid_search(query, top_k)

    def graph_search(self, query: str, top_k: int = 10, num_hops: int = 2) -> List[Dict[str, Any]]:
        return self.index.graph_search(query, top_k, num_hops)

    def refine_query(self, spec, base: str, widening: int = 0) -> str:
        """Build a retrieval query, optionally widened by dropping constraints.

        widening=0 -> the question as asked
        widening=1 -> sport + games + event descriptor (no threshold prose)
        widening=2 -> sport + event descriptor only
        widening=3 -> event descriptor only
        """
        if widening <= 0:
            return base
        parts: List[str] = []
        if widening <= 2 and spec.sport:
            parts.append(spec.sport)
        if widening <= 1 and spec.year:
            parts.append(str(spec.year))
        if widening <= 1 and spec.season:
            parts.append(spec.season)
        if spec.event_desc:
            parts.append(spec.event_desc)
        if spec.venue:
            parts.append(spec.venue)
        if spec.date_text:
            parts.append(spec.date_text)
        if spec.target_title:
            parts.append(spec.target_title)
        if spec.sport and spec.sport not in parts:
            parts.insert(0, spec.sport)
        return " ".join(parts) if parts else base