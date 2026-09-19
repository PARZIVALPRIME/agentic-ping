"""Push the corpus-built knowledge graph into TigerGraph, idempotently.

The graph itself is still produced by ``kg/builder.py`` - this module only ships
it to the database, so there is one definition of the model (Python) and one of
the storage layout (``tg/schema.gsql``).

Design notes that matter when something goes wrong at 2am:

* **Idempotent.** Everything is an upsert keyed by the ids the model already uses
  (``Q303623``, ``ATHLETE::name``, ``SPORT::Athletics``, ``GAMES::2012::Summer``),
  so re-running after a partial load repairs the graph instead of duplicating it.
* **Batched, and a bad batch does not stop the load.** Objects are flushed in
  batches of ``batch_size``; a rejected batch is recorded and the load continues,
  because one malformed document must not cost the other 2,950.
* **Edge-safe.** An edge is only sent when both endpoints were part of this load,
  so ``--limit`` test loads cannot fail on dangling targets.
* **Text is optional and truncated.** Article text is what makes
  ``get_event_details`` useful from the database, but it dominates payload size;
  ``with_text=False`` ships only the structured attributes.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from kg.builder import iter_corpus
from kg.model import KnowledgeGraph

from .client import TigerGraphClient, TigerGraphError

VERTEX_TYPES = ("Event", "Athlete", "Games", "Sport", "Venue")
EDGE_TYPES = ("PREV", "NEXT", "WON_BY", "HELD_AT", "IN_SPORT", "PART_OF")
# Types checked by the verifier and by the backend's startup probe.
VERIFY_VERTEX_TYPES = VERTEX_TYPES

# Attributes carried on the Event vertex. This tuple is the contract with
# tg/schema.gsql: the schema test asserts every name here is declared there, so a
# new model field cannot silently fail to reach the database.
EVENT_ATTRS = ("title", "url", "sport", "year", "season", "games_label",
               "event_name", "competitors", "nations", "venue", "venues",
               "date_raw", "dates_raw", "gold", "silver", "bronze",
               "win_value", "win_label", "prev", "next", "approx_tokens",
               "prev_doc_id", "next_doc_id", "months_found", "days_found")
# ``seq`` is deliberately *not* an EventNode field: it is the document position
# in the corpus graph, injected by the loader while it enumerates. It lives on the
# server only, so tg_filter_events can order rows exactly like the local scan and
# the backend's remote-vs-mirror check is an exact comparison.
SEQ_ATTR = "seq"
INT_ATTRS = ("year", "competitors", "nations", "approx_tokens", SEQ_ATTR)
LIST_ATTRS = ("months_found", "days_found")
TEXT_LIMIT = 6000


def id_type(v_id: str) -> str:
    """Vertex type implied by a model id (the ids are type-prefixed by design)."""
    if v_id.startswith("ATHLETE::"):
        return "Athlete"
    if v_id.startswith("SPORT::"):
        return "Sport"
    if v_id.startswith("VENUE::"):
        return "Venue"
    if v_id.startswith("GAMES::"):
        return "Games"
    return "Event"


def _coerce(name: str, value: Any) -> Any:
    """Force a model value into the type declared in the schema."""
    if name in LIST_ATTRS:
        return [str(v) for v in (value or [])]
    if name in INT_ATTRS:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0
    if value is None:
        return ""
    return str(value)


def event_payload(node: Any, text: str = "", seq: int = 0) -> Dict[str, Any]:
    """Attribute dict for one Event vertex (typed; text only when provided).

    ``seq`` carries the corpus position so server-side result order matches the
    local scan; it is supplied by the loader, not read from the node.
    """
    payload = {name: _coerce(name, getattr(node, name, None))
               for name in EVENT_ATTRS}
    payload[SEQ_ATTR] = int(seq)
    if text:
        payload["text"] = text
    return payload


def games_payload(label: str) -> Dict[str, Any]:
    """Attributes for a Games vertex from its ``"2012 Summer"`` label."""
    parts = str(label).split()
    year = int(parts[0]) if parts and parts[0].isdigit() else 0
    season = parts[1] if len(parts) > 1 else ""
    return {"label": str(label), "year": year, "season": season}


def event_texts(corpus_path: str, wanted: Optional[Iterable[str]] = None,
                limit_chars: int = TEXT_LIMIT) -> Dict[str, str]:
    """Read article text for the documents being pushed.

    Streamed from the corpus rather than from the KG cache, because the cache
    deliberately stores only ``approx_tokens``. ``wanted`` bounds memory to the
    slice being loaded.
    """
    keep = set(wanted) if wanted is not None else None
    texts: Dict[str, str] = {}
    for doc in iter_corpus(corpus_path):
        doc_id = doc.get("doc_id")
        if not doc_id or (keep is not None and doc_id not in keep):
            continue
        texts[doc_id] = (doc.get("text") or "")[:limit_chars]
    return texts


def install_schema(client: TigerGraphClient, schema_path: str = "",
                   drop: bool = False) -> str:
    """Install ``tg/schema.gsql`` (types + queries) via the GSQL server.

    Savanna has no GSQL-server port; there the same file is pasted into the
    console's GSQL editor and this call is simply not used. ``drop`` prepends
    ``DROP GRAPH`` so a schema change can be applied cleanly on Community
    Edition.
    """
    import os

    path = schema_path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "schema.gsql")
    with open(path, "r", encoding="utf-8") as fh:
        script = fh.read()
    if drop:
        script = f"DROP GRAPH {client.graphname}\n" + script
    return client.gsql(script, tag="benchmark")


def _send(client: TigerGraphClient, payload: Dict[str, Any], report: Dict[str, Any],
          what: str) -> bool:
    """One upsert call, with the failure recorded instead of raised."""
    try:
        client.upsert(payload)
        report["batches"] += 1
        return True
    except TigerGraphError as exc:
        if len(report["errors"]) < 5:
            report["errors"].append({"what": what, "error": str(exc)[:300]})
        return False


def push_graph(kg: KnowledgeGraph, client: TigerGraphClient, corpus_path: str = "",
               batch_size: int = 500, max_events: int = 0, with_text: bool = True,
               progress: bool = True) -> Dict[str, Any]:
    """Load the whole graph. Returns a report; never raises for data problems.

    ``max_events`` loads a prefix only, for smoke tests against a real server.
    """
    report: Dict[str, Any] = {"graph": client.graphname, "errors": [], "batches": 0,
                              "vertices": {}, "edges": {}, "text_included": with_text}
    started = time.time()

    event_ids = list(kg.events.keys())
    if max_events:
        event_ids = event_ids[:int(max_events)]
    report["capped_at"] = len(event_ids)
    texts = (event_texts(corpus_path, event_ids)
             if corpus_path and with_text else {})

    # ── vertices ───────────────────────────────────────────────────────
    pushed: set = set()
    sent_events = 0
    buffer: Dict[str, Any] = {}
    for index, doc_id in enumerate(event_ids, 1):
        node = kg.events[doc_id]
        buffer[doc_id] = event_payload(node, texts.get(doc_id, ""),
                                      seq=index)
        pushed.add(doc_id)
        if len(buffer) >= batch_size:
            if _send(client, {"vertices": {"Event": buffer}}, report, "Event batch"):
                sent_events += len(buffer)
            buffer = {}
            if progress:
                print(f"[tg-loader] Event {min(index, len(event_ids))}/{len(event_ids)} "
                      f"({time.time() - started:.1f}s)", flush=True)
    if buffer:
        if _send(client, {"vertices": {"Event": buffer}}, report, "Event batch"):
            sent_events += len(buffer)
    report["vertices"]["Event"] = {"sent": sent_events,
                                   "target": len(event_ids),
                                   "local_total": len(kg.events)}

    others = {
        "Athlete": {aid: {"name": a.name, "medals": list(a.medals)}
                    for aid, a in kg.athletes.items()},
        "Sport": {kg.sport_id(s): {"name": s} for s in sorted(kg.sports) if s},
        "Venue": {kg.venue_id(v): {"name": v} for v in sorted(kg.venues) if v},
        "Games": {kg.games_id(games_payload(g)["year"], games_payload(g)["season"]):
                  games_payload(g) for g in sorted(kg.games)},
    }
    for vtype, vertices in others.items():
        sent = 0
        keys = list(vertices)
        for start in range(0, len(keys), batch_size):
            chunk = {k: vertices[k] for k in keys[start:start + batch_size]}
            if _send(client, {"vertices": {vtype: chunk}}, report, f"{vtype} batch"):
                sent += len(chunk)
        pushed.update(vertices.keys())
        report["vertices"][vtype] = {"sent": sent, "target": len(keys)}

    # ── edges ──────────────────────────────────────────────────────────
    buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
    skipped = 0
    for edge in kg.edges:
        etype = str(edge.edge_type).upper()
        if etype not in EDGE_TYPES:
            skipped += 1
            continue
        if edge.src not in pushed or edge.dst not in pushed:
            skipped += 1          # capped load / dangling target: never send it
            continue
        src_type = id_type(edge.src)
        bucket = buckets.setdefault((src_type, etype), {})
        bucket.setdefault(edge.src, {})[edge.dst] = dict(edge.attrs or {})
    for (src_type, etype), edges in buckets.items():
        sent_edges = 0
        keys = list(edges)
        for start in range(0, len(keys), batch_size):
            chunk = {k: edges[k] for k in keys[start:start + batch_size]}
            if _send(client, {"edges": {src_type: {etype: chunk}}}, report,
                     f"{etype} batch"):
                sent_edges += sum(len(targets) for targets in chunk.values())
        report["edges"][etype] = {"sent": sent_edges, "sources": len(keys)}
    report["edges_skipped"] = skipped
    report["local_edges_total"] = len(kg.edges)

    report["seconds"] = round(time.time() - started, 2)
    report["requests"] = getattr(client, "requests", 0)
    report["retries_used"] = getattr(client, "retries_used", 0)
    report["ok"] = not report["errors"]
    return report


def remote_counts(client: TigerGraphClient) -> Dict[str, Any]:
    """Per-type counts from the server: installed query first, stats second.

    Two sources because Savanna's RESTPP exposes ``/graph/{g}/stats`` while some
    Community builds only answer it after a first query; ``tg_stats`` is installed
    by ``schema.gsql`` and always works.
    """
    try:
        data = client.run_query("tg_stats")
        if isinstance(data, dict) and data:
            return {k.lstrip("@"): v for k, v in data.items()}
    except TigerGraphError:
        pass
    try:
        stats = client.graph_stats()
    except TigerGraphError as exc:
        return {"error": str(exc)}
    results = stats.get("results") if isinstance(stats, dict) else None
    if isinstance(results, dict):
        node = results.get("Vertex") or {}
        return {k: (v.get("COUNT") if isinstance(v, dict) else v)
                for k, v in node.items()}
    return {}


def verify_counts(kg: KnowledgeGraph, client: TigerGraphClient) -> Dict[str, Any]:
    """Compare local vs remote object counts, per vertex type.

    A mismatch is reported, never raised: the point of the verifier is to tell a
    human whether an ingest is complete, and a partial graph is the normal state
    while a load is still running.
    """
    local = {
        "Event": len(kg.events),
        "Athlete": len(kg.athletes),
        "Games": len(kg.games),
        "Sport": len({s for s in kg.sports if s}),
        "Venue": len({v for v in kg.venues if v}),
    }
    remote = remote_counts(client)
    rows: Dict[str, Any] = {}
    for vtype, expected in local.items():
        got = remote.get(vtype)
        rows[vtype] = {
            "local": expected,
            "remote": got,
            "match": (int(got) == expected) if isinstance(got, (int, float)) else None,
        }
    missing = client.missing_types(VERIFY_VERTEX_TYPES)
    ok = not missing and all(row["match"] is not False for row in rows.values())
    return {"ok": bool(ok), "types": rows, "missing_vertex_types": missing,
            "remote_raw": remote, "requests": getattr(client, "requests", 0),
            "retries_used": getattr(client, "retries_used", 0)}


def dump_report(report: Dict[str, Any]) -> str:
    """JSON string for a report, safe for anything the server returned."""
    return json.dumps(report, indent=2, default=str)