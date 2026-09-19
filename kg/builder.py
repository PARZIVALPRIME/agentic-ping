"""Build a knowledge graph from the Olympics corpus.

Everything is derived from the corpus itself: the article title gives
(sport, games year, season, event) and the leading `[Infobox ...]` block gives
competitors, nations, venue, dates and medal winners. Articles that are not
Olympic event pages (the corpus also contains films, biographies, companies,
...) are recorded as distractors so the retrieval layers can account for them.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .model import AthleteNode, EventNode, GraphEdge, KnowledgeGraph
from .textutil import days_in, months_in, normalize

TITLE_RE = re.compile(
    r"^(?P<sport>.+?)\s+at\s+the\s+(?P<year>\d{4})\s+(?P<season>Summer|Winter)\s+Olympics"
    r"(?:\s*[\u2013\u2014-]\s*(?P<event>.+))?$"
)
FIELD_RE = re.compile(r"^\s{0,4}([a-z_]+)\s*:\s?(.*)$")
INT_RE = re.compile(r"^\s*(\d+)\s*$")
MEDAL_FIELDS = ("gold", "silver", "bronze")


def parse_infobox(text: str) -> Dict[str, str]:
    """Extract the leading ``[Infobox ...]`` key/value block, if present."""
    fields: Dict[str, str] = {}
    if not text or not text.lstrip().startswith("[Infobox"):
        return fields
    started = False
    for line in text.split("\n"):
        stripped = line.strip()
        if line.lstrip().startswith("[Infobox"):
            started = True
            continue
        if not started:
            continue
        if not stripped:
            break
        if line.startswith("["):
            break
        m = FIELD_RE.match(line)
        if m:
            key, value = m.group(1), m.group(2).strip()
            if value:
                fields[key] = value
    return fields


def parse_title(title: str) -> Optional[Tuple[str, int, str, str]]:
    """Return (sport, year, season, event_name) or None if not an event page."""
    m = TITLE_RE.match(title.strip())
    if not m:
        return None
    return (
        m.group("sport").strip(),
        int(m.group("year")),
        m.group("season"),
        (m.group("event") or "").strip(),
    )


def to_int(value: str) -> Optional[int]:
    if value is None:
        return None
    m = INT_RE.match(str(value))
    if m:
        return int(m.group(1))
    m2 = re.match(r"^\s*(\d+)", str(value))
    return int(m2.group(1)) if m2 else None


def iter_corpus(corpus_path: str) -> Iterator[Dict[str, Any]]:
    with open(corpus_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _event_key(sport: str, event_name: str) -> str:
    return f"{normalize(sport)}::{normalize(event_name)}"


def build_kg(corpus_path: str, progress: bool = True) -> KnowledgeGraph:
    """Parse the corpus and return a fully-linked :class:`KnowledgeGraph`."""
    kg = KnowledgeGraph()
    event_keys: Dict[str, List[str]] = {}
    pending_links: List[Tuple[str, str, str]] = []

    count = 0
    for idx, doc in enumerate(iter_corpus(corpus_path)):
        count = idx + 1
        doc_id = doc.get("doc_id") or f"doc_{idx}"
        title = doc.get("title") or ""
        text = doc.get("text") or ""
        parsed = parse_title(title)
        if not parsed:
            kg.non_event_doc_ids.append(doc_id)
            continue

        sport, year, season, event_name = parsed
        ib = parse_infobox(text)
        node = EventNode(
            doc_id=doc_id,
            title=title,
            url=doc.get("url", ""),
            sport=sport,
            year=year,
            season=season,
            games_label=ib.get("games", ""),
            event_name=event_name or ib.get("event", ""),
            competitors=to_int(ib.get("competitors", "")),
            nations=to_int(ib.get("nations", "")),
            venue=ib.get("venue", ""),
            venues=ib.get("venues", ""),
            date_raw=ib.get("date", ""),
            dates_raw=ib.get("dates", ""),
            gold=ib.get("gold", ""),
            silver=ib.get("silver", ""),
            bronze=ib.get("bronze", ""),
            win_value=ib.get("win_value", ""),
            win_label=ib.get("win_label", ""),
            prev=ib.get("prev", ""),
            next=ib.get("next", ""),
            approx_tokens=int(doc.get("approx_tokens", 0) or 0),
        )
        date_blob = f"{node.date_raw} {node.dates_raw} {node.games_label}"
        node.months_found = sorted(months_in(date_blob))
        node.days_found = sorted(days_in(date_blob), key=lambda d: int(d))
        kg.add_event(node)

        event_keys.setdefault(_event_key(node.sport, node.event_name), []).append(doc_id)
        if node.prev:
            pending_links.append((doc_id, "prev", node.prev))
        if node.next:
            pending_links.append((doc_id, "next", node.next))

    _link_editions(kg, event_keys, pending_links)
    _build_medal_vertices(kg)
    kg.finalise()

    if progress:
        print(f"[kg] parsed {count} docs -> {len(kg.events)} event pages, "
              f"{len(kg.non_event_doc_ids)} non-event docs")
    return kg


def _link_editions(
    kg: KnowledgeGraph,
    event_keys: Dict[str, List[str]],
    pending_links: List[Tuple[str, str, str]],
) -> None:
    """Resolve ``prev``/``next`` infobox years into PREV/NEXT edges."""
    year_lookup: Dict[str, Dict[int, str]] = {}
    for key, ids in event_keys.items():
        by_year = {}
        for doc_id in ids:
            by_year[kg.events[doc_id].year] = doc_id
        year_lookup[key] = by_year

    for doc_id, field, year_raw in pending_links:
        year = to_int(year_raw)
        if year is None:
            continue
        node = kg.events[doc_id]
        key = _event_key(node.sport, node.event_name)
        target = year_lookup.get(key, {}).get(year)
        if target and target != doc_id:
            if field == "prev":
                node.prev_doc_id = target
                kg.edges.append(GraphEdge("PREV", doc_id, target))
            else:
                node.next_doc_id = target
                kg.edges.append(GraphEdge("NEXT", doc_id, target))


def _build_medal_vertices(kg: KnowledgeGraph) -> None:
    """Create athlete vertices + WON_BY edges, and structural graph edges."""
    for doc_id, node in kg.events.items():
        kg.edges.append(GraphEdge(
            "PART_OF", doc_id, kg.games_id(node.year, node.season),
            {"year": node.year, "season": node.season, "label": node.games_label},
        ))
        if node.sport:
            kg.edges.append(GraphEdge("IN_SPORT", doc_id, kg.sport_id(node.sport)))
        if node.venue:
            kg.edges.append(GraphEdge("HELD_AT", doc_id, kg.venue_id(node.venue)))
        for medal in MEDAL_FIELDS:
            winner = getattr(node, medal)
            if not winner:
                continue
            aid = kg.athlete_id(winner)
            athlete = kg.athletes.get(aid)
            if athlete is None:
                athlete = AthleteNode(athlete_id=aid, name=winner)
                kg.athletes[aid] = athlete
            athlete.medals.append(doc_id)
            kg.edges.append(GraphEdge("WON_BY", doc_id, aid, {"medal": medal}))


def load_or_build(corpus_path: str, cache_path: str = "results/knowledge_graph.json",
                  rebuild: bool = False) -> KnowledgeGraph:
    """Load the graph from cache when available, otherwise build and cache it."""
    import os

    if not rebuild and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            return KnowledgeGraph.from_dict(json.load(fh))
    kg = build_kg(corpus_path)
    save_kg(kg, cache_path)
    return kg


def save_kg(kg: KnowledgeGraph, cache_path: str) -> None:
    import os

    payload = kg.to_dict()
    payload["non_event_doc_ids"] = kg.non_event_doc_ids
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
