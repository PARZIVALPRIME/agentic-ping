"""Graph tools exposed to the LLM as native function-call schemas.

The knowledge graph and corpus index already contain everything needed to answer
the benchmark questions; what changes here is *who reasons over it*. Instead of
Python deciding "this is an aggregation question, so call the counting solver",
the tools below give the model primitives to explore with, and the model decides
which to call, in what order, and how to combine the results.

Tool design rules
-----------------
* **Data access, not answers.**  ``get_event_values`` returns the raw list of
  values; the model does the counting/comparing. There is deliberately no
  ``answer_aggregation`` tool - that would put the reasoning back in Python.
* **Compact results.**  Rows are trimmed and capped so a multi-step loop stays
  inside a metered token budget.
* **Terminal tool.**  ``submit_answer`` is the only way to finish, which makes
  "the model decided it was done" an explicit, observable event.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

MAX_ROWS = 40
MAX_TEXT_CHARS = 900

# Infobox fields the model may aggregate over. Kept to scalar fields so the
# returned list is small enough to reason over directly.
AGG_FIELDS = ("competitors", "nations", "year", "gold", "silver", "bronze",
              "venue", "event_name", "win_value", "date_raw", "title")

EDGE_TYPES = ("PREV", "NEXT", "WON_BY", "HELD_AT", "IN_SPORT", "PART_OF")


def _schema(name: str, description: str, props: Dict[str, Any],
            required: Optional[List[str]] = None) -> Dict[str, Any]:
    """OpenAI tool-calling schema. Kept terse: these definitions are re-sent on
    every step, so their length is a per-step cost on metered providers."""
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props,
                       "required": required or []}}}


_STR = {"type": "string"}
_INT = {"type": "integer"}


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    _schema("search_events",
            "Find event pages in the knowledge graph. Combine filters for "
            "precision. Returns rows with the doc_id used by other tools.",
            {"query": dict(_STR, description="match on title/event name"),
             "sport": dict(_STR, description="e.g. 'Athletics'"),
             "year_from": _INT, "year_to": _INT,
             "season": dict(_STR, description="'Summer' or 'Winter'"),
             "venue": _STR,
             "limit": dict(_INT, description=f"rows (default 25, cap {MAX_ROWS})")},
            []),

    _schema("search_passages",
            "Full-text search over corpus passages. Use for prose evidence or "
            "when the event is not in the graph.",
            {"query": dict(_STR, description="natural-language query"),
             "top_k": dict(_INT, description="passages (default 5, max 10)")},
            ["query"]),

    _schema("get_event_values",
            "Raw list of values for one field across every event matching the "
            "filters. You count, sum, sort or find extremes over these values "
            "yourself. Includes the matching doc_ids.",
            {"field": dict(_STR, enum=list(AGG_FIELDS),
                           description="field to extract"),
             "sport": _STR, "year_from": _INT, "year_to": _INT,
             "season": _STR, "venue": _STR,
             "query": dict(_STR, description="match on title/event name")},
            ["field"]),

    _schema("get_event_details",
            "Complete record for one event page (infobox fields, venue, dates, "
            "medal winners) plus a text excerpt.",
            {"doc_id": dict(_STR, description="doc_id from search_events")},
            ["doc_id"]),

    _schema("traverse_graph",
            "Follow one graph edge: PREV/NEXT walk an event's editions "
            "(before/after questions), WON_BY lists medal winners, HELD_AT the "
            "venue, IN_SPORT/PART_OF the sport and Games.",
            {"doc_id": _STR,
             "edge_type": dict(_STR, enum=list(EDGE_TYPES)),
             "direction": dict(_STR, enum=["out", "in", "both"],
                               description="default 'both'")},
            ["doc_id", "edge_type"]),

    _schema("submit_answer",
            "Submit the final answer and stop. Call once, when confident. "
            "'answer' is a short verbatim span: a number, a name, an event "
            "title. Cite the doc_ids that justify it.",
            {"answer": dict(_STR, description="short final answer"),
             "reasoning": dict(_STR, description="1-2 sentences on how you got it"),
             "doc_ids": {"type": "array", "items": _STR,
                         "description": "supporting doc_ids"}},
            ["answer"]),
]

TOOL_NAMES = [t["function"]["name"] for t in TOOL_SCHEMAS]


class GraphTools:
    """Executes the tool schemas above against the knowledge graph + index.

    Every executor returns a JSON-serialisable dict. Results carry a ``doc_ids``
    list whenever they touch documents, so the model can cite them directly and
    the harness can score citation precision/recall.
    """

    def __init__(self, kg, index) -> None:
        self.kg = kg
        self.index = index
        self.calls: List[Dict[str, Any]] = []

    # ── helpers ────────────────────────────────────────────────────────
    def _rows(self, events, limit: int = 25) -> List[Dict[str, Any]]:
        out = []
        for ev in events[:limit]:
            out.append({
                "doc_id": ev.doc_id, "title": ev.title,
                "sport": ev.sport, "year": ev.year, "season": ev.season,
                "event_name": ev.event_name, "venue": ev.venue,
            })
        return out

    def _filter(self, query: str = "", sport: str = "", year_from: Optional[int] = None,
                year_to: Optional[int] = None, season: str = "",
                venue: str = "") -> List[Any]:
        """Candidate event set from the graph indexes, then attribute filters."""
        events = list(self.kg.events.values())
        if sport:
            by_sport = self.kg.events_for_sport(sport)
            if by_sport:
                events = by_sport
        if venue:
            by_venue = self.kg.events_at_venue(venue)
            if by_venue:
                events = by_venue
        if year_from is not None:
            events = [e for e in events if e.year and e.year >= int(year_from)]
        if year_to is not None:
            events = [e for e in events if e.year and e.year <= int(year_to)]
        if season:
            events = [e for e in events if e.season.lower() == season.lower()]
        if query:
            from kg.textutil import normalize

            q = normalize(query)
            terms = [t for t in q.split() if len(t) > 2]
            scored = []
            for ev in events:
                hay = normalize(f"{ev.title} {ev.event_name} {ev.sport} {ev.venue}")
                hits = sum(1 for t in terms if t in hay)
                if hits:
                    scored.append((hits, ev))
            if scored:
                scored.sort(key=lambda p: (-p[0], p[1].title))
                events = [ev for _h, ev in scored]
        return events

    # ── executors ──────────────────────────────────────────────────────
    def search_events(self, query: str = "", sport: str = "", year_from=None,
                      year_to=None, season: str = "", venue: str = "",
                      limit: int = 25) -> Dict[str, Any]:
        limit = max(1, min(int(limit or 25), MAX_ROWS))
        events = self._filter(query, sport, year_from, year_to, season, venue)
        rows = self._rows(events, limit)
        return {"count": len(events), "returned": len(rows), "events": rows,
                "doc_ids": [r["doc_id"] for r in rows]}

    def search_passages(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        top_k = max(1, min(int(top_k or 5), 10))
        hits = self.index.hybrid_search(query, top_k=top_k)
        out = []
        for h in hits:
            out.append({"doc_id": h.get("doc_id"), "title": h.get("title", ""),
                        "score": round(float(h.get("score", 0.0)), 3),
                        "text": (h.get("text") or "")[:MAX_TEXT_CHARS]})
        return {"returned": len(out), "passages": out,
                "doc_ids": list(dict.fromkeys(h["doc_id"] for h in out if h.get("doc_id")))}

    def get_event_values(self, field: str, sport: str = "", year_from=None,
                         year_to=None, season: str = "", venue: str = "",
                         query: str = "") -> Dict[str, Any]:
        """Raw values for one field - the model performs the aggregation itself."""
        if field not in AGG_FIELDS:
            return {"error": f"unknown field '{field}'. valid: {list(AGG_FIELDS)}"}
        events = self._filter(query, sport, year_from, year_to, season, venue)
        values, pairs, skipped = [], [], 0
        for ev in events:
            raw = getattr(ev, field, None)
            if raw in (None, "", 0):
                skipped += 1
                continue
            values.append(raw)
            pairs.append({"doc_id": ev.doc_id, "value": raw})
        return {"field": field, "num_matching_events": len(events),
                "num_with_value": len(values), "num_missing": skipped,
                "values": values[:MAX_ROWS * 2],
                "pairs": pairs[:MAX_ROWS * 2],
                "note": ("values are in graph order, not sorted - sort/count them "
                         "yourself"),
                "doc_ids": [p["doc_id"] for p in pairs[:MAX_ROWS * 2]]}

    def get_event_details(self, doc_id: str) -> Dict[str, Any]:
        ev = self.kg.events.get(doc_id)
        if ev is None:
            resolved = self.index.resolve_doc_id(doc_id) if self.index else None
            ev = self.kg.events.get(resolved) if resolved else None
        if ev is None:
            doc = self.index.get_doc(doc_id) if self.index else None
            if doc is not None:
                return {"doc_id": doc.doc_id, "title": doc.title,
                        "in_graph": False,
                        "infobox": dict(doc.infobox or {}),
                        "text": (doc.text or "")[:MAX_TEXT_CHARS],
                        "doc_ids": [doc.doc_id]}
            return {"error": f"no document or graph vertex '{doc_id}'"}
        payload = {
            "doc_id": ev.doc_id, "title": ev.title, "url": ev.url,
            "in_graph": True, "sport": ev.sport, "year": ev.year,
            "season": ev.season, "games": ev.games_key,
            "event_name": ev.event_name, "venue": ev.venue or ev.venues,
            "date": ev.date_raw or ev.dates_raw,
            "competitors": ev.competitors, "nations": ev.nations,
            "gold": ev.gold, "silver": ev.silver, "bronze": ev.bronze,
            "win": f"{ev.win_value} {ev.win_label}".strip(),
            "prev": ev.prev_doc_id, "next": ev.next_doc_id,
            "doc_ids": [ev.doc_id],
        }
        doc = self.index.get_doc(ev.doc_id) if self.index else None
        if doc is not None:
            payload["text"] = (doc.text or "")[:MAX_TEXT_CHARS]
        return payload

    def traverse_graph(self, doc_id: str, edge_type: str,
                       direction: str = "both") -> Dict[str, Any]:
        edge_type = (edge_type or "").upper()
        if edge_type not in EDGE_TYPES:
            return {"error": f"unknown edge_type '{edge_type}'. valid: {list(EDGE_TYPES)}"}
        if doc_id not in self.kg.events and not doc_id.startswith(("SPORT::", "VENUE::",
                                                                 "ATHLETE::")):
            resolved = self.index.resolve_doc_id(doc_id) if self.index else None
            doc_id = resolved or doc_id
        hops = []
        for etype, direction_seen, other, attrs in self.kg.neighbours(doc_id):
            if etype != edge_type:
                continue
            if direction != "both" and direction_seen != direction:
                continue
            hops.append({"direction": direction_seen, "target": other,
                         "attrs": attrs or {},
                         "title": getattr(self.kg.events.get(other), "title", "")})
        doc_ids = [h["target"] for h in hops if h["target"] in self.kg.events]
        return {"from": doc_id, "edge_type": edge_type, "num_found": len(hops),
                "neighbours": hops[:MAX_ROWS], "doc_ids": doc_ids[:MAX_ROWS]}

    def get_games_events(self, year: int, season: str = "", sport: str = "") -> Dict[str, Any]:
        events = self.kg.events_at_games(int(year), season or "")
        if sport:
            from kg.textutil import normalize

            s = normalize(sport)
            events = [e for e in events if normalize(e.sport) == s]
        rows = self._rows(events, MAX_ROWS)
        return {"year": year, "season": season, "sport": sport,
                "count": len(events), "returned": len(rows),
                "events": rows, "doc_ids": [r["doc_id"] for r in rows]}

    # ── terminal tool ──────────────────────────────────────────────────
    def submit_answer(self, answer: str = "", reasoning: str = "",
                      doc_ids: Optional[List[str]] = None,
                      confidence: float = 0.9) -> Dict[str, Any]:
        """Terminate the loop with a final answer.

        This is the agent's explicit completion signal: the *model* decides when
        the investigation is over, instead of the harness inferring it from a
        confidence score.
        """
        cited = [d for d in (doc_ids or []) if d]
        return {"accepted": True, "answer": str(answer).strip()[:300],
                "reasoning": str(reasoning or "").strip()[:600],
                "confidence": float(confidence or 0.9),
                "doc_ids": cited}

    # ── dispatch ───────────────────────────────────────────────────────
    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run one tool call, timing it and recording it in ``self.calls``."""
        import time

        t0 = time.perf_counter()
        fn = getattr(self, name, None)
        error = ""
        if fn is None or name not in TOOL_NAMES:
            result: Dict[str, Any] = {"error": f"unknown tool '{name}'"}
            error = result["error"]
        else:
            try:
                result = fn(**(args or {}))
            except TypeError as exc:
                result = {"error": f"bad arguments for {name}: {exc}"}
                error = result["error"]
            except Exception as exc:
                result = {"error": f"{exc.__class__.__name__}: {exc}"}
                error = result["error"]
        record = {"tool": name, "args": args or {},
                  "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                  "error": error,
                  "doc_ids": result.get("doc_ids", []) if isinstance(result, dict) else [],
                  "summary": _summarise(name, result)}
        self.calls.append(record)
        return result


def _summarise(name: str, result: Any) -> str:
    """One-line description of a tool result for the trace/dashboard."""
    if not isinstance(result, dict):
        return str(result)[:120]
    if result.get("error"):
        return f"error: {result['error']}"[:160]
    if name == "search_events":
        return f"{result.get('count', 0)} event(s), {result.get('returned', 0)} shown"
    if name == "search_passages":
        return f"{result.get('returned', 0)} passage(s)"
    if name == "get_event_values":
        return (f"field={result.get('field')} values={result.get('num_with_value', 0)}"
                f"/{result.get('num_matching_events', 0)}")
    if name == "get_event_details":
        return f"{result.get('title', '')[:60]} ({result.get('games', '')})"
    if name == "traverse_graph":
        return f"{result.get('edge_type')} -> {result.get('num_found', 0)} neighbour(s)"
    if name == "get_games_events":
        return f"{result.get('count', 0)} event(s) at {result.get('year')}"
    if name == "submit_answer":
        return f"answer={str(result.get('answer', ''))[:60]}"
    return json.dumps(result)[:120]
