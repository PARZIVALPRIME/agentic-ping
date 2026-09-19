"""Knowledge-graph data model for the Olympics corpus.

The graph is *built from the corpus itself* (infoboxes + titles) so that no
external knowledge is required and the "corpus is the only source of truth"
rule holds. It contains:

  Vertices
    - EventPage : one per corpus article that is an Olympic event page
    - Games     : (year, season) e.g. (2012, "Summer")
    - Sport     : e.g. "Athletics"
    - Venue     : e.g. "Eton Dorney"
    - Athlete   : medal winners named in the infobox

  Edges
    - EventPage -PART_OF->  Games
    - EventPage -IN_SPORT-> Sport
    - EventPage -HELD_AT->  Venue
    - EventPage -WON_BY->   Athlete   (attrs: medal = gold|silver|bronze)
    - EventPage -PREV->     EventPage (previous edition of the same event)
    - EventPage -NEXT->     EventPage (next edition of the same event)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class EventNode:
    doc_id: str
    title: str
    url: str = ""
    sport: str = ""
    year: int = 0
    season: str = ""          # "Summer" | "Winter" | ""
    games_label: str = ""     # raw infobox `games`, e.g. "2012 Summer"
    event_name: str = ""      # e.g. "Men's K-2 1000 metres"
    competitors: Optional[int] = None
    nations: Optional[int] = None
    venue: str = ""
    venues: str = ""
    date_raw: str = ""
    dates_raw: str = ""
    gold: str = ""
    silver: str = ""
    bronze: str = ""
    win_value: str = ""
    win_label: str = ""
    prev: str = ""
    next: str = ""
    approx_tokens: int = 0
    prev_doc_id: Optional[str] = None
    next_doc_id: Optional[str] = None
    months_found: List[str] = field(default_factory=list)
    days_found: List[str] = field(default_factory=list)

    @property
    def games_key(self) -> str:
        return f"{self.year} {self.season}".strip()

    @property
    def heading(self) -> str:
        """Short human label, e.g. 'Athletics - Men's marathon (2008 Summer)'."""
        ev = f" - {self.event_name}" if self.event_name else ""
        return f"{self.sport}{ev} ({self.games_key} Olympics)"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AthleteNode:
    athlete_id: str
    name: str
    medals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphEdge:
    edge_type: str
    src: str
    dst: str
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KnowledgeGraph:
    """In-memory property graph with lookup indexes."""

    def __init__(self) -> None:
        self.events: Dict[str, EventNode] = {}
        self.athletes: Dict[str, AthleteNode] = {}
        self.games: Set[str] = set()
        self.sports: Set[str] = set()
        self.venues: Set[str] = set()
        self.edges: List[GraphEdge] = []
        self.sport_games_index: Dict[str, List[str]] = {}
        self.venue_index: Dict[str, List[str]] = {}
        self.sport_index: Dict[str, List[str]] = {}
        self.event_name_index: Dict[str, List[str]] = {}
        self.title_index: Dict[str, str] = {}
        self.sport_vocabulary: List[str] = []
        self.non_event_doc_ids: List[str] = []
        self.coverage: Dict[str, int] = {}

    # ── construction ────────────────────────────────────────────────────
    def add_event(self, node: EventNode) -> None:
        self.events[node.doc_id] = node

    def finalise(self) -> None:
        """Build derived indexes once all vertices/edges are added."""
        from .textutil import normalize

        for doc_id, ev in self.events.items():
            self.games.add(ev.games_key)
            self.sports.add(ev.sport)
            if ev.venue:
                self.venues.add(ev.venue)
            self.sport_index.setdefault(normalize(ev.sport), []).append(doc_id)
            self.venue_index.setdefault(normalize(ev.venue), []).append(doc_id)
            if ev.event_name:
                self.event_name_index.setdefault(normalize(ev.event_name), []).append(doc_id)
            self.title_index[normalize(ev.title)] = doc_id
            key = f"{normalize(ev.sport)}|{ev.year}|{ev.season.lower()}"
            self.sport_games_index.setdefault(key, []).append(doc_id)

        self.sport_vocabulary = sorted((s for s in self.sports if s), key=len, reverse=True)
        self.coverage = {
            "num_events": len(self.events),
            "num_athletes": len(self.athletes),
            "num_games": len(self.games),
            "num_sports": len(self.sports),
            "num_venues": len(self.venues),
            "num_edges": len(self.edges),
            "num_non_event_docs": len(self.non_event_doc_ids),
            "with_competitors": sum(1 for e in self.events.values() if e.competitors is not None),
            "with_nations": sum(1 for e in self.events.values() if e.nations is not None),
            "with_venue": sum(1 for e in self.events.values() if e.venue),
            "with_date": sum(1 for e in self.events.values() if (e.date_raw or e.dates_raw)),
        }

    # ── ids ─────────────────────────────────────────────────────────────
    @staticmethod
    def games_id(year: int, season: str) -> str:
        return f"GAMES::{year}::{season}"

# ── persistence ─────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return {
            "events": {k: v.to_dict() for k, v in self.events.items()},
            "athletes": {k: v.to_dict() for k, v in self.athletes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "games": sorted(self.games),
            "sports": sorted(s for s in self.sports if s),
            "venues": sorted(v for v in self.venues if v),
            "coverage": self.coverage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        kg = cls()
        for doc_id, raw in data.get("events", {}).items():
            kg.events[doc_id] = EventNode(**raw)
        for aid, raw in data.get("athletes", {}).items():
            kg.athletes[aid] = AthleteNode(**raw)
        for raw in data.get("edges", []):
            kg.edges.append(GraphEdge(**raw))
        kg.games = set(data.get("games", []))
        kg.sports = set(data.get("sports", []))
        kg.venues = set(data.get("venues", []))
        kg.coverage = data.get("coverage", {})
        kg.non_event_doc_ids = list(data.get("non_event_doc_ids", []))
        kg._rebuild_indexes()
        return kg

    def _rebuild_indexes(self) -> None:
        from .textutil import normalize

        self.sport_index, self.venue_index = {}, {}
        self.event_name_index, self.title_index = {}, {}
        self.sport_games_index = {}
        for doc_id, ev in self.events.items():
            self.sport_index.setdefault(normalize(ev.sport), []).append(doc_id)
            self.venue_index.setdefault(normalize(ev.venue), []).append(doc_id)
            if ev.event_name:
                self.event_name_index.setdefault(normalize(ev.event_name), []).append(doc_id)
            self.title_index[normalize(ev.title)] = doc_id
            key = f"{normalize(ev.sport)}|{ev.year}|{ev.season.lower()}"
            self.sport_games_index.setdefault(key, []).append(doc_id)
        self.sport_vocabulary = sorted((s for s in self.sports if s), key=len, reverse=True)

    # ── graph queries ───────────────────────────────────────────────────
    def events_for(self, sport: str, year: int, season: str) -> List[EventNode]:
        from .textutil import normalize

        key = f"{normalize(sport)}|{year}|{season.lower()}"
        return [self.events[d] for d in self.sport_games_index.get(key, [])]

    def events_for_sport(self, sport: str) -> List[EventNode]:
        from .textutil import normalize

        return [self.events[d] for d in self.sport_index.get(normalize(sport), [])]

    def events_at_venue(self, venue: str) -> List[EventNode]:
        from .textutil import normalize

        return [self.events[d] for d in self.venue_index.get(normalize(venue), [])]

    def events_at_games(self, year: int, season: str) -> List[EventNode]:
        key = f"|{year}|{season.lower()}"
        out = []
        for idx_key, ids in self.sport_games_index.items():
            if idx_key.endswith(key):
                out.extend(self.events[d] for d in ids)
        return out

    def filter_events(self, query: str = "", sport: str = "", year_from=None,
                      year_to=None, season: str = "", venue: str = "",
                      cap: int = 0) -> List["EventNode"]:
        """Event set for a structured filter (the one query the tools rely on).

        This lives on the graph model rather than in the tools so that a remote
        backend (``tg/backend.py``) can override it and push the same filter down
        to the database, while callers keep one call site.

        Ordering is part of the contract: corpus/graph order unless ``query`` is
        given, in which case the candidates are ranked by term overlap and then
        title. ``cap`` is the row budget and is applied *before* ranking, so a
        capped local answer covers the same rows as a capped remote answer - that
        is what lets the TigerGraph backend verify its first reply against this
        method object-for-object.
        """
        events = list(self.events.values())
        if sport:
            by_sport = self.events_for_sport(sport)
            if by_sport:
                events = by_sport
        if venue:
            by_venue = self.events_at_venue(venue)
            if by_venue:
                events = by_venue
        if year_from is not None:
            events = [e for e in events if e.year and e.year >= int(year_from)]
        if year_to is not None:
            events = [e for e in events if e.year and e.year <= int(year_to)]
        if season:
            events = [e for e in events if e.season.lower() == season.lower()]
        if cap and int(cap) > 0:
            events = events[:int(cap)]
        if query:
            events = rank_events_by_terms(events, query)
        return events

    def neighbours(self, doc_id: str) -> List[tuple]:
        """Return (edge_type, direction, other_id) triples for a vertex."""
        out = []
        for e in self.edges:
            if e.src == doc_id:
                out.append((e.edge_type, "out", e.dst, e.attrs))
            elif e.dst == doc_id:
                out.append((e.edge_type, "in", e.src, e.attrs))
        return out

    def stats(self) -> Dict[str, Any]:
        return dict(self.coverage)
    @staticmethod
    def sport_id(sport: str) -> str:
        return f"SPORT::{sport}"

    @staticmethod
    def venue_id(venue: str) -> str:
        return f"VENUE::{venue}"

    @staticmethod
    def athlete_id(name: str) -> str:
        from .textutil import normalize

        return f"ATHLETE::{normalize(name)}"


def rank_events_by_terms(events: List[EventNode], query: str) -> List[EventNode]:
    """Rank candidate events by term overlap with ``query``.

    The only piece of filtering that stays in Python when a remote backend
    answers the query (see ``tg/backend.py``): the database applies the
    attribute filters, this orders the survivors so the agent sees the most
    relevant rows first. Events with no matching term are dropped, matching the
    local-only behaviour, and the original order is kept when nothing matches so
    a broad query still returns candidates.
    """
    from .textutil import normalize

    terms = [t for t in normalize(query).split() if len(t) > 2]
    if not terms:
        return events
    scored = []
    for ev in events:
        hay = normalize(f"{ev.title} {ev.event_name} {ev.sport} {ev.venue}")
        hits = sum(1 for t in terms if t in hay)
        if hits:
            scored.append((hits, ev))
    if not scored:
        return events
    scored.sort(key=lambda pair: (-pair[0], pair[1].title))
    return [ev for _hits, ev in scored]
