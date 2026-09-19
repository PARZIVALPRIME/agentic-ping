"""Entity linker agent: resolve question phrases onto graph vertices.

This is the step that makes graph traversal possible. It maps the question's
sport phrase, venue phrase and event descriptor onto concrete vertices
(``Sport``, ``Venue``, ``Games``, ``EventPage``) and reports the match scores so
the orchestrator can decide whether to trust the link or fall back to text
retrieval.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from reasoning.solvers import resolve_sport, resolve_venue
from reasoning.query_parser import QuerySpec


class EntityLinker:
    """Links (sport, venue, games, edition-chain) mentions to graph vertices."""

    def __init__(self, kg) -> None:
        self.kg = kg

    def link(self, spec: QuerySpec) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "sport_vertex": None, "sport_score": 0.0,
            "venue_vertices": [], "venue_score": 0.0,
            "games_vertex": None, "edition_chain": [],
        }

        sport = spec.sport or resolve_sport(spec.sport or spec.event_desc, self.kg)
        if sport:
            out["sport_vertex"] = sport
            out["sport_score"] = 1.0 if spec.sport else 0.85

        if spec.venue:
            keys, score = resolve_venue(spec.venue, self.kg)
            out["venue_vertices"] = [self.kg.venue_id(k) for k in keys]
            out["venue_keys"] = keys
            out["venue_score"] = round(score, 3)

        if spec.year and spec.season:
            out["games_vertex"] = self.kg.games_id(spec.year, spec.season)
        if spec.before_year is not None:
            from reasoning.solvers import _nearest_previous_games

            season = spec.season or "Summer"
            prev = _nearest_previous_games(season, spec.before_year, self.kg)
            out["edition_chain"] = [f"{spec.before_year} {season}", f"{prev} {season}"]
            if prev is not None:
                out["games_vertex"] = self.kg.games_id(prev, season)
                out["resolved_prev_year"] = prev

        return out

    def linked_event_pages(self, spec: QuerySpec, limit: int = 40) -> List[Any]:
        """EventPage vertices implied by the question's slots.

        This is the graph traversal the agentic pipeline uses for counting and
        comparison questions: (sport ∪ venue ∪ edition) -> candidate set.
        """
        found: Dict[str, Any] = {}

        if spec.sport:
            sport = resolve_sport(spec.sport, self.kg)
            if sport and spec.year and spec.season:
                for node in self.kg.events_for(sport, spec.year, spec.season):
                    found[node.doc_id] = node
            elif sport:
                for node in self.kg.events_for_sport(sport)[:limit]:
                    found[node.doc_id] = node

        if spec.before_year is not None:
            from reasoning.solvers import _nearest_previous_games

            season = spec.season or "Summer"
            prev = _nearest_previous_games(season, spec.before_year, self.kg)
            sport = spec.sport or resolve_sport(spec.event_desc, self.kg)
            if prev is not None and sport:
                for node in self.kg.events_for(sport, prev, season):
                    found[node.doc_id] = node

        if spec.venue:
            keys, _score = resolve_venue(spec.venue, self.kg)
            for key in keys:
                for node in self.kg.events_at_venue(key):
                    found[node.doc_id] = node

        return list(found.values())[:limit]