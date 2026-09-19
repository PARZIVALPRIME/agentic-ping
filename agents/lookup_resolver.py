"""Lookup + multi-hop resolver agent.

Handles the two "single answer" families:

  * lookup    - resolve a named article and read one field from it
  * multi_hop - resolve (venue, date) -> event page -> gold medal winner

Both are *linking* problems rather than aggregation problems, so this agent
returns the matched vertex together with the competing candidates and their
scores, letting the evidence evaluator decide whether the match is safe.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from kg.textutil import best_match, normalize, similarity
from reasoning.query_parser import QuerySpec
from reasoning.solvers import _date_score, _venue_sport_affinity, resolve_venue


class LookupResolver:
    """Resolves article titles and (venue, date) pairs to event vertices."""

    def __init__(self, kg) -> None:
        self.kg = kg

    # ── lookup ─────────────────────────────────────────────────────────
    def resolve_article(self, spec: QuerySpec) -> Dict[str, Any]:
        target = spec.target_title
        if not target:
            return {"matched": None, "score": 0.0, "field": None}
        doc_id = self.kg.title_index.get(normalize(target))
        score = 1.0
        if doc_id is None:
            match, score = best_match(target, list(self.kg.title_index.keys()))
            if match is not None and score >= 0.85:
                doc_id = self.kg.title_index[match]
        if doc_id is None:
            return {"matched": None, "score": round(score, 3), "field": None,
                    "searched": len(self.kg.title_index)}
        node = self.kg.events[doc_id]
        return {
            "matched": node,
            "score": round(score, 3),
            "field": "nations",
            "value": None if node.nations is None else str(node.nations),
            "competitors": node.competitors,
        }

    # ── multi-hop ──────────────────────────────────────────────────────
    def resolve_venue_date(self, spec: QuerySpec, top_n: int = 5) -> Dict[str, Any]:
        if not spec.venue:
            return {"matched": None, "score": 0.0, "candidates": []}
        keys, venue_score = resolve_venue(spec.venue, self.kg)
        if not keys:
            return {"matched": None, "score": 0.0, "candidates": [],
                    "venue_resolved": []}

        ids: List[str] = []
        for key in keys:
            ids.extend(self.kg.venue_index.get(key, []))
        candidates = [self.kg.events[d] for d in dict.fromkeys(ids)]

        if spec.year:
            narrowed = [e for e in candidates if e.year == spec.year]
            if spec.season:
                narrowed = [e for e in narrowed if e.season == spec.season]
            if narrowed:
                candidates = narrowed

        scored = sorted(
            ((_date_score(spec, e), _venue_sport_affinity(spec.venue, e), e)
             for e in candidates),
            key=lambda x: (x[0], x[1]), reverse=True)
        if not scored:
            return {"matched": None, "score": 0.0, "candidates": [],
                    "venue_resolved": keys, "venue_score": round(venue_score, 3)}

        date_score, affinity, best = scored[0]
        return {
            "matched": best,
            "score": round(date_score, 3),
            "venue_affinity": round(affinity, 3),
            "venue_resolved": keys,
            "venue_score": round(venue_score, 3),
            "value": best.gold,
            "candidates": [
                {"doc_id": e.doc_id, "title": e.title,
                 "date": e.date_raw or e.dates_raw, "venue": e.venue,
                 "gold": e.gold, "date_score": round(s, 3),
                 "venue_affinity": round(a, 3)}
                for s, a, e in scored[:top_n]
            ],
            "num_candidates": len(candidates),
        }

    def disambiguate(self, spec: QuerySpec, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Explain which signal separated the top candidate from the runner-up."""
        if len(candidates) < 2:
            return {"ambiguous": False}
        top, second = candidates[0], candidates[1]
        gap = top["date_score"] - second["date_score"]
        return {
            "ambiguous": gap < 0.1 and top["venue_affinity"] == second["venue_affinity"],
            "top": top["title"],
            "runner_up": second["title"],
            "date_score_gap": round(gap, 3),
            "separating_signal": ("venue/sport affinity" if gap < 0.1
                                  else "date match strength"),
        }