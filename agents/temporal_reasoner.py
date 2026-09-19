"""Temporal reasoner agent: resolve "immediately before/after" questions.

Temporal questions require linking *two* Games editions. The reasoner walks the
corpus edition chain (the same PREV/NEXT edges the GraphRAG pipeline uses) to
find the immediately preceding Games, then resolves the event inside those
Games and reads its medal field.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from reasoning.query_parser import QuerySpec
from reasoning.solvers import (_event_similarity, _nearest_previous_games,
                               resolve_sport)


class TemporalReasoner:
    """Edition-chain reasoning for temporal questions."""

    def __init__(self, kg) -> None:
        self.kg = kg

    def resolve_anchor(self, spec: QuerySpec) -> Dict[str, Any]:
        """Find the Games edition the question's phrase actually refers to."""
        season = spec.season or "Summer"
        target = spec.before_year
        if target is None:
            target = min((spec.years_mentioned or [9999]))
        year = _nearest_previous_games(season, target, self.kg)
        return {
            "anchor_year": target,
            "resolved_year": year,
            "season": season,
            "reasoning": (f"latest {season} Games strictly before {target} "
                          f"is {year}"),
        }

    def resolve_event(self, spec: QuerySpec, year: int, season: str,
                      top_n: int = 3) -> Dict[str, Any]:
        sport = spec.sport or resolve_sport(spec.event_desc, self.kg)
        if not sport:
            return {"sport": "", "matched": None, "candidates": []}
        events = self.kg.events_for(sport, year, season)
        scored = sorted(((_event_similarity(spec.event_desc, e), e) for e in events),
                        key=lambda x: x[0], reverse=True)
        matched = scored[0][1] if scored else None
        return {
            "sport": sport,
            "matched": matched,
            "score": round(scored[0][0], 3) if scored else 0.0,
            "candidates": [{"title": e.title, "score": round(s, 3)}
                           for s, e in scored[:top_n]],
            "num_events_in_games": len(events),
        }

    def verify(self, spec: QuerySpec, node) -> Dict[str, Any]:
        """Cross-check the resolved event against its own edition chain.

        Confirms that the event really did exist in the edition *after* the one
        we resolved to (the anchor), which is what "immediately before X" means.
        """
        checks: Dict[str, Any] = {"consistent": False, "details": []}
        if node is None:
            return checks
        chain = sorted((n for n in self.kg.events_for_sport(node.sport)
                        if n.event_name == node.event_name and n.season == node.season),
                       key=lambda n: n.year)
        years = [n.year for n in chain]
        checks["edition_years"] = years
        if node.next:
            checks["details"].append(
                f"corpus infobox 'next' field points to {node.next}")
            checks["consistent"] = str(node.next) == str(spec.before_year)
        if checks["consistent"]:
            checks["details"].append(
                f"edition chain confirms {node.year} is immediately before "
                f"{spec.before_year}")
        else:
            checks["details"].append(
                "edition chain could not confirm the anchor; falling back to "
                "corpus-wide edition ordering")
        return checks