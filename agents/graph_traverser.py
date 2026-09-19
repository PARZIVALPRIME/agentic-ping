"""Graph traverser agent: walk the knowledge graph from linked vertices.

Given vertices produced by the entity linker, this agent performs the
*structural* retrieval that no vector index can do: enumerate every event page
belonging to a sport at a Games, walk the PREV/NEXT edition chain, and gather
same-venue siblings. Its output is the candidate set that counting and
comparison questions are decided over.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from reasoning.query_parser import QuerySpec
from reasoning.solvers import _nearest_previous_games, resolve_sport


class GraphTraverser:
    """Structural retrieval over (Sport, Games, Venue, PREV/NEXT) edges."""

    def __init__(self, kg) -> None:
        self.kg = kg

    def events_for_sport_games(self, sport: str, year: int, season: str) -> List[Any]:
        return self.kg.events_for(sport, year, season)

    def previous_editions(self, sport: str, event_name: str, year: int,
                          season: str) -> List[Any]:
        """Editions of the same event that are strictly older than ``year``."""
        out = []
        for node in self.kg.events_for_sport(sport):
            if node.season != season or node.year >= year:
                continue
            if node.event_name == event_name:
                out.append(node)
        return sorted(out, key=lambda n: n.year)

    def edition_chain(self, sport: str, event_name: str, season: str) -> List[Any]:
        chain = [n for n in self.kg.events_for_sport(sport)
                 if n.season == season and n.event_name == event_name]
        return sorted(chain, key=lambda n: n.year)

    def neighbours_of(self, doc_id: str) -> List[Dict[str, Any]]:
        out = []
        for relation, direction, other_id, attrs in self.kg.neighbours(doc_id):
            out.append({"relation": relation, "direction": direction,
                        "other_id": other_id, "attrs": attrs})
        return out

    def expand_candidates(self, spec: QuerySpec, num_hops: int = 1,
                          limit: int = 60) -> Dict[str, Any]:
        """The main entry point: produce the candidate event set for a question."""
        result: Dict[str, Any] = {"strategy": "", "events": [], "relations": []}

        # 1. sport + games (exhaustive enumeration - this is the key advantage
        #    over top-k retrieval for counting questions)
        if spec.sport and spec.year and spec.season:
            sport = resolve_sport(spec.sport, self.kg)
            if sport:
                events = self.kg.events_for(sport, spec.year, spec.season)
                if events:
                    result["strategy"] = "IN_SPORT + PART_OF"
                    result["events"] = events[:limit]
                    result["relations"] = ["IN_SPORT", "PART_OF"]
                    return result

        # 2. temporal: resolve the previous Games, then the event inside it
        if spec.before_year is not None:
            season = spec.season or "Summer"
            prev = _nearest_previous_games(season, spec.before_year, self.kg)
            sport = spec.sport or resolve_sport(spec.event_desc, self.kg)
            if prev is not None and sport:
                events = self.kg.events_for(sport, prev, season)
                result["strategy"] = "PREV_GAMES + IN_SPORT"
                result["events"] = events[:limit]
                result["relations"] = ["PREV_GAMES", "IN_SPORT"]
                result["resolved_year"] = prev
                return result

        # 3. venue-anchored traversal (link venue -> events -> candidate set)
        if spec.venue:
            from reasoning.solvers import resolve_venue

            keys, score = resolve_venue(spec.venue, self.kg)
            events = []
            for key in keys:
                events.extend(self.kg.events_at_venue(key))
            if events:
                result["strategy"] = "HELD_AT"
                result["events"] = events[:limit]
                result["relations"] = ["HELD_AT"]
                result["venue_score"] = round(score, 3)
                return result

        # 4. fallback: whole sport (used by widening strategies)
        if spec.sport:
            events = self.kg.events_for_sport(spec.sport)
            result["strategy"] = "IN_SPORT"
            result["events"] = events[:limit]
            result["relations"] = ["IN_SPORT"]
        return result