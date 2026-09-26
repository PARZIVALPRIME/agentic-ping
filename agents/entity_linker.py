"""Entity linker agent: resolve question phrases onto graph vertices.

This is the step that makes graph traversal possible. It maps the question's
sport phrase, venue phrase and event descriptor onto concrete vertices
(``Sport``, ``Venue``, ``Games``, ``EventPage``) and reports the match scores so
the orchestrator can decide whether to trust the link or fall back to text
retrieval.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from reasoning.solvers import (resolve_previous_edition, resolve_sport,
                               resolve_venue, season_candidates)
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

        if spec.year:
            seasons = [spec.season] if spec.season else season_candidates(spec, self.kg)
            vertices = {s: self.kg.games_id(spec.year, s) for s in seasons
                        if self.kg.games_id(spec.year, s)}
            if len(vertices) == 1:
                season, vertex = next(iter(vertices.items()))
                out["games_vertex"] = vertex
                out["resolved_season"] = season
                self._pin_season(spec, season, out, why="single games vertex for year")
            elif vertices:
                # Both seasons exist for this year; the season stays open until
                # the sport/event evidence picks one (never defaulted).
                out["games_vertices"] = vertices
                out["season_ambiguous"] = True
        if spec.before_year is not None:
            edition = resolve_previous_edition(spec, self.kg)
            out["edition_resolution"] = edition
            resolved_season = edition.get("season")
            resolved_year = edition.get("year")
            if resolved_year is not None and resolved_season:
                out["edition_chain"] = [f"{spec.before_year} {resolved_season}",
                                        f"{resolved_year} {resolved_season}"]
                out["games_vertex"] = self.kg.games_id(resolved_year, resolved_season)
                out["resolved_prev_year"] = resolved_year
                self._pin_season(spec, resolved_season, out,
                                 why=f"edition resolved by {edition.get('method')}")
            else:
                out["season_unresolved"] = True
                out["edition_chain"] = []

        return out

    @staticmethod
    def _pin_season(spec: QuerySpec, season: str, out: Dict[str, Any],
                    why: str = "") -> None:
        """Record a season established from evidence rather than from wording.

        The decision is written onto the spec so every later agent works from the
        same Games edition, and onto the linker's output so the trace shows that
        the season was derived (and how) rather than assumed.
        """
        if not season:
            return
        if not spec.season:
            spec.season = season
            spec.season_unresolved = False
            out["season_resolved_by"] = why or "evidence"


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
            sport = spec.sport or resolve_sport(spec.event_desc, self.kg)
            if sport:
                edition = resolve_previous_edition(spec, self.kg)
                if edition.get("year") is not None and edition.get("season"):
                    pairs = [(edition["season"], edition["year"])]
                else:
                    # Undetermined season: gather every candidate edition rather
                    # than committing to one. The event-descriptor match then
                    # decides, and the evidence audit records the ambiguity.
                    pairs = [(c["season"], c["year"])
                             for c in edition.get("candidates", [])
                             if c.get("year") is not None]
                for season, year in pairs:
                    for node in self.kg.events_for(sport, year, season):
                        found[node.doc_id] = node


        if spec.venue:
            keys, _score = resolve_venue(spec.venue, self.kg)
            for key in keys:
                for node in self.kg.events_at_venue(key):
                    found[node.doc_id] = node

        return list(found.values())[:limit]