"""A KnowledgeGraph whose hot paths are answered by TigerGraph.

``GraphTools`` (the ReAct tool layer) touches the graph through exactly two entry
points:

    filter_events(...)   filtered event search - powers aggregation/superlative
    neighbours(doc_id)   one-hop traversal     - powers multi-hop/temporal

This class overrides those two so the work happens in the database
(``tg_filter_events`` / ``tg_neighbours``), and keeps the corpus-built graph as a
*mirror* for everything else: passage retrieval, the vocabulary accessors, titles
for hop targets, and the offline path.

Two properties make the swap safe, and they are the reason this file exists:

1. Remote rows are rehydrated into the same ``EventNode`` objects the local path
   produces, including derived fields (months_found/days_found). Downstream code
   cannot tell which backend answered, so a TigerGraph run stays comparable to a
   local one.
2. The first remote answer per entry point is checked against the mirror. A server
   graph that disagrees with the corpus (partial ingest, query built against an
   older schema) disables the remote path for the rest of the run, loudly, instead
   of quietly returning worse answers. Errors degrade the same way, per call.

Row budget
----------
``max_rows`` (``TG_MAX_ROWS``) is a *safety valve*, not a tuning knob: the graph
tools report the size of the candidate set (``count``, ``num_matching_events``) and
hand back the raw values behind it, so a budget below the largest candidate set
this corpus produces would quietly change the answers the agent can reach. The
shipped default (100000) is deliberately lossless;
``tools/probe_tg_semantics.py`` prints the number to clear (2210 here) and
``tools/test_tg_backend.py`` exercises a deliberately small budget to prove that
truncation stays *reported* rather than silent.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from kg.model import EventNode, KnowledgeGraph, rank_events_by_terms

from .client import TigerGraphClient, TigerGraphError, TigerGraphUnavailable

# EventNode fields that may arrive from the server; anything else is ignored, so a
# schema drift can never raise a TypeError in the middle of a benchmark run.
_EVENT_FIELDS = set(EventNode.__dataclass_fields__)
_INT_FIELDS = ("year", "competitors", "nations", "approx_tokens")


def accumulators(client: TigerGraphClient, name: str,
                 params: Optional[Dict[str, Any]] = None,
                 method: str = "GET") -> Dict[str, Any]:
    """Run an installed query and return every PRINTed accumulator.

    ``client.run_query`` unwraps a single accumulator, which is not enough here:
    ``tg_filter_events`` deliberately prints both ``@@ids`` (ordered ids, cheap to
    compare) and ``@@rows`` (the vertices themselves). An empty dict means the
    query answered nothing usable, and the caller degrades to the mirror.
    """
    results = client.run_query(name, params or {}, method=method, raw=True)
    if isinstance(results, list) and results and isinstance(results[0], dict):
        return {str(k).lstrip("@"): v for k, v in results[0].items()}
    return {}


def vertex_rows(value: Any) -> Optional[List[Dict[str, Any]]]:
    """Extract ``[{v_id, attributes}]`` from any RESTPP vertex-set shape.

    ``PRINT @@rows`` arrives as ``[{"Event": {...}}]`` on some versions and as a
    flat list on others, so both are accepted. ``None`` means "not parseable",
    which the caller treats as a query failure - the safe direction when answers
    are being scored.
    """
    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(item, dict) for item in value):
            out: List[Dict[str, Any]] = []
            for item in value:
                if "attributes" in item or "v_id" in item:
                    out.append({"v_id": item.get("v_id") or item.get("id") or "",
                                "attributes": item.get("attributes") or {}})
                    continue
                for v_id, attrs in item.items():
                    if isinstance(attrs, dict):
                        out.append({"v_id": v_id,
                                    "attributes": attrs.get("attributes", attrs)})
            return out
        return None
    if isinstance(value, dict):
        if not value:
            return []
        if "v_id" in value:
            return [{"v_id": value.get("v_id") or "",
                     "attributes": value.get("attributes") or {}}]
        for item in value.values():
            if isinstance(item, list):
                return vertex_rows(item)
            if isinstance(item, dict):
                return [{"v_id": v_id, "attributes": attrs.get("attributes", attrs)}
                        for v_id, attrs in value.items()]
        return None
    return None


def edge_rows(value: Any) -> Optional[List[Tuple[str, str, str, Dict[str, Any]]]]:
    """Extract ``(src, dst, edge_type, attrs)`` from a printed edge accumulator."""
    if isinstance(value, dict):
        for item in value.values():
            found = edge_rows(item)
            if found is not None:
                return found
        return None
    if not isinstance(value, list):
        return None
    out: List[Tuple[str, str, str, Dict[str, Any]]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        # Versions differ: flat from_id/to_id, nested {"e": {...}}, or camelCase.
        row = item
        if "from_id" not in row and "fromId" not in row and isinstance(row.get("e"), dict):
            row = row["e"]
        src = row.get("from_id") or row.get("fromId")
        dst = row.get("to_id") or row.get("toId")
        if not src or not dst:
            return None
        etype = (row.get("edge_type") or row.get("edgeType") or row.get("e_type")
                 or row.get("type") or "")
        out.append((str(src), str(dst), str(etype).upper(),
                    dict(row.get("attributes") or {})))
    return out


class _Disagreement(TigerGraphError):
    """The server answered, and the answer contradicts the mirror.

    Separated from the other ``TigerGraphError``s on purpose: this one means the
    graph in the database is not the graph this corpus produced (foreign rows, or
    a payload whose accumulators contradict each other), which is a defect worth
    reporting differently from an outage.
    """


def _consistent(remote_ids: List[Any], local_ids: List[Any], cap: int) -> bool:
    """Is the remote answer consistent with the same query run locally?

    Not a row-order comparison: ordering can legitimately differ between a graph
    query and a Python scan, and ``rank_events_by_terms`` is applied to the
    remote rows afterwards anyway. What matters is that the server is answering
    from the *same* graph:

    * every remote id must be one the mirror has - a foreign row means the
      database holds vertices this corpus never produced, or the query is
      answering from a different graph, and
    * when the mirror's whole answer fits in the row budget, the two sets must be
      identical. This is the assertion that catches a *partial* ingest: if the
      corpus says 8 hops (or 213 matching events) and the server answers 5, the
      missing rows would become silently missing candidates.
    * when the mirror's answer is larger than the budget, the query's ``LIMIT``
      (and therefore ``@@ids``) is only ever a page of it, so all that can be
      proven per call is membership - the page must be full of rows the mirror
      owns. Completeness is a property of the ingest, not of one page, and is
      checked separately by ``tg.loader.verify_counts`` at load time.

    ``local_ids`` must therefore be the *uncapped* mirror answer and ``cap`` is
    only the query's row budget. Empty-vs-nonempty is always a failure: a schema
    that exists but holds no data must never silently become the answer path.
    """
    budget = max(1, int(cap))
    if not remote_ids:
        return not local_ids
    if not set(remote_ids) <= set(local_ids):
        return False
    if len(local_ids) <= budget:
        return set(remote_ids) == set(local_ids)
    return True


class TigerGraphBackend(KnowledgeGraph):
    """Mirror of the corpus graph, with TigerGraph answering the agent's queries."""

    def __init__(self, mirror: KnowledgeGraph, client: TigerGraphClient,
                 graphname: str = "", max_rows: int = 200,
                 verbose: bool = False) -> None:
        super().__init__()
        # Shared, not copied: the mirror is read-only during a run and a 15k-object
        # deep copy per pipeline would show up in the latency numbers we report.
        self.events = mirror.events
        self.athletes = mirror.athletes
        self.edges = mirror.edges
        self.games = mirror.games
        self.sports = mirror.sports
        self.venues = mirror.venues
        self.coverage = dict(mirror.coverage)
        self.non_event_doc_ids = list(mirror.non_event_doc_ids)
        self._rebuild_indexes()

        self.client = client
        self.graphname = graphname or getattr(client, "graphname", "")
        self.max_rows = max(1, int(max_rows))
        self.verbose = verbose
        # Entry point -> why it stopped using the server. Per entry point on
        # purpose: a graph missing one installed query (say tg_neighbours) must
        # not take the other path down with it, and the reason is kept for the
        # run summary instead of only a log line.
        self.disabled: Dict[str, str] = {}
        # None = not answered remotely yet, so the next answer is still checked.
        # True = agreed with the mirror. False = disagreed (that path is off).
        self.checked: Dict[str, Optional[bool]] = {"filter_events": None,
                                                   "neighbours": None}
        self.counters: Dict[str, int] = {"filter_events": 0, "neighbours": 0,
                                         "remote_rows": 0, "fallbacks": 0}
        self.notes: List[str] = []

    # ── reporting ─────────────────────────────────────────────────────
    def log(self, message: str) -> None:
        self.notes.append(message)
        if self.verbose:
            print(f"[tg-backend] {message}", flush=True)

    def remote_stats(self) -> Dict[str, Any]:
        """What the remote path actually did - copied into the run summary."""
        return {"remote_enabled": not self.disabled, "graphname": self.graphname,
                "disabled": dict(self.disabled),
                "verified": dict(self.checked), "counters": dict(self.counters),
                "client_requests": getattr(self.client, "requests", 0),
                "notes": list(self.notes[-6:])}

    def _remote_ok(self, where: str) -> bool:
        return self.disabled.get(where) is None

    def _degrade(self, where: str, exc: Exception, disagreed: bool = False) -> None:
        """Stop answering ``where`` from the server and record the reason.

        ``disagreed`` separates the two ways this happens, because the run
        summary is read by a human deciding whether the ingest is sound:

        * the server could not be asked (down, query not installed, unreadable
          payload) - ``checked`` stays ``None``: nothing was compared;
        * the server answered and the answer did not match the mirror -
          ``checked`` becomes ``False``, which is an ingest defect, not an
          outage.
        """
        self.counters["fallbacks"] += 1
        reason = f"{type(exc).__name__}: {str(exc)[:200]}"
        self.disabled[where] = reason
        if disagreed and self.checked.get(where) is None:
            self.checked[where] = False
        self.log(f"{where}: remote query unusable ({reason}); using the mirror")

    # ── hot path 1: filtered event search ──────────────────────────────
    def _hydrate(self, rows: List[Dict[str, Any]]) -> List[EventNode]:
        """Rebuild the exact objects the local path would have produced.

        Server attributes are filtered through the dataclass fields, so a column
        that exists only in the database (``text``, ``seq``) is ignored rather
        than raising. ``months_found``/``days_found`` come back as JSON lists and
        are what the temporal helpers read.
        """
        out: List[EventNode] = []
        for row in rows:
            attrs = {k: v for k, v in (row.get("attributes") or {}).items()
                     if k in _EVENT_FIELDS}
            attrs.setdefault("doc_id", row.get("v_id") or "")
            for name in _INT_FIELDS:
                if name in attrs and attrs[name] is not None:
                    try:
                        attrs[name] = int(attrs[name])
                    except (TypeError, ValueError):
                        attrs[name] = None
            for name in ("months_found", "days_found"):
                value = attrs.get(name)
                attrs[name] = [str(v) for v in value] if isinstance(value, list) else []
            if not attrs.get("doc_id"):
                continue
            try:
                out.append(EventNode(**attrs))
            except TypeError as exc:      # unknown/missing field: keep the row out
                self.log(f"could not hydrate {attrs.get('doc_id')}: {exc}")
        return out

    def _local_filter(self, query: str, sport: str, year_from: Optional[int],
                      year_to: Optional[int], season: str, venue: str,
                      cap: int) -> List[EventNode]:
        """The same query answered from the mirror, with the same row budget."""
        return KnowledgeGraph.filter_events(
            self, query=query, sport=sport, year_from=year_from, year_to=year_to,
            season=season, venue=venue, cap=cap)

    def _filter_page(self, sport: str, venue: str, season: str,
                     year_from: Optional[int], year_to: Optional[int],
                     budget: int) -> Tuple[List[EventNode], List[str]]:
        """One page of ``tg_filter_events``, reassembled in corpus order.

        Returns ``(rows, ids)``: the hydrated vertices the server answered with,
        and the ids it claimed, in its own order. ``@@ids`` is what carries that
        order - ``@@rows`` is a ``SetAccum``, so it arrives as an unordered vertex
        set and re-reading the mirror to sort it would defeat the point of having
        the database answer. Both accumulators are bounded by the query's
        ``LIMIT``, so the ids describe exactly the page that was returned.

        Raises ``_Disagreement`` when the two accumulators contradict each other
        (the query claims a row it did not print), and a plain
        ``TigerGraphError`` when the payload is simply unreadable. The caller
        degrades to the mirror either way.
        """
        acc = accumulators(self.client, "tg_filter_events", {
            "sport": sport or "", "venue": venue or "", "season": season or "",
            "year_from": int(year_from or 0), "year_to": int(year_to or 0),
            "max_rows": budget})
        id_order = acc.get("ids")
        rows = vertex_rows(acc.get("rows"))
        if rows is None:
            raise TigerGraphError(
                f"unreadable tg_filter_events payload: {str(acc)[:200]}")
        by_id: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            by_id.setdefault(str(row.get("v_id") or ""), row)
        page_ids = ([str(v) for v in id_order] if isinstance(id_order, list)
                    else list(by_id))
        missing = [v for v in page_ids if v not in by_id]
        if missing:
            raise _Disagreement(
                f"{len(missing)} of {len(page_ids)} row ids have no vertex in "
                f"@@rows (e.g. {missing[:3]})")
        hydrated = self._hydrate([by_id[v] for v in page_ids])
        if len(hydrated) != len(page_ids):
            raise _Disagreement(
                f"only {len(hydrated)} of {len(page_ids)} rows could be "
                f"rebuilt as event vertices")
        return hydrated, page_ids

    def filter_events(self, query: str = "", sport: str = "", year_from=None,
                      year_to=None, season: str = "", venue: str = "",
                      cap: int = 0) -> List[EventNode]:
        """Filtered event search: TigerGraph applies the filter, Python ranks.

        ``cap`` is the row budget (the backend defaults to ``max_rows``). The
        remote page and the mirror answer are capped before ranking so the agent
        sees the same shape either way; the first-use check compares the page the
        server claimed against the *uncapped* mirror answer, so a
        truncated-to-budget page is only required to be a subset of what the
        mirror holds (see ``_consistent``).
        """
        budget = int(cap) or self.max_rows
        if not self._remote_ok("filter_events"):
            return self._local_filter(query, sport, year_from, year_to, season,
                                      venue, budget)
        try:
            remote, page_ids = self._filter_page(sport, venue, season, year_from,
                                                 year_to, budget)
        except _Disagreement as exc:
            self._degrade("filter_events", exc, disagreed=True)
            return self._local_filter(query, sport, year_from, year_to, season,
                                      venue, budget)
        except (TigerGraphError, TigerGraphUnavailable) as exc:
            self._degrade("filter_events", exc)
            return self._local_filter(query, sport, year_from, year_to, season,
                                      venue, budget)

        if self.checked["filter_events"] is None:
            full = self._local_filter(query, sport, year_from, year_to, season,
                                      venue, 0)          # uncapped, for checking
            foreign = sorted(set(page_ids) - {e.doc_id for e in full})[:3]
            if not _consistent(page_ids, [e.doc_id for e in full], budget):
                self._degrade("filter_events", TigerGraphError(
                    f"server answered {len(page_ids)} rows for a filter the "
                    f"corpus answers with {len(full)} (sport={sport!r} "
                    f"venue={venue!r} years={year_from}-{year_to}); "
                    f"{len(foreign)} of them are not in the corpus "
                    f"(e.g. {foreign}) - is the ingest complete?"),
                    disagreed=True)
                return self._local_filter(query, sport, year_from, year_to,
                                          season, venue, budget)
            self.checked["filter_events"] = True
            self.log(f"filter_events verified against the mirror "
                     f"({len(page_ids)} rows)")
            if len(full) > len(page_ids) >= budget:
                # Verified as far as one page can be, but the caller asked for more
                # rows than the budget allows. Say so, loudly: a run whose numbers
                # come from a truncated candidate set is a different measurement.
                self.log(f"filter_events: row budget {budget} is below the "
                         f"candidate set ({len(full)} rows for sport={sport!r} "
                         f"venue={venue!r} years={year_from}-{year_to}) - raise "
                         f"TG_MAX_ROWS for a lossless run")

        # Counted only now: a page that failed the check above was not an answer
        # this backend served, and the run summary must not imply that it was.
        self.counters["filter_events"] += 1
        self.counters["remote_rows"] += len(remote)
        remote = remote[:budget]
        return rank_events_by_terms(remote, query) if query else remote

    # ── hot path 2: one-hop traversal ──────────────────────────────────
    def _local_neighbours(self, doc_id: str, cap: int = 0) -> List[tuple]:
        """The same hop answered from the mirror, in the same tuple shape."""
        triples = KnowledgeGraph.neighbours(self, doc_id)
        return triples[:int(cap)] if cap and int(cap) > 0 else triples

    def _hydrate_edges(self, rows: Any, doc_id: str) -> List[tuple]:
        """Turn printed edges into the ``(edge_type, direction, other, attrs)``
        tuples the local path returns.

        The direction is recovered by asking which end is the requested vertex,
        because an undirected GSQL pattern (``-(e)-``) does not record which side
        the traversal started from - and direction is what the temporal helpers
        read to walk PREV/NEXT the right way. Rows that do not mention this
        vertex are dropped rather than trusted.
        """
        out: List[tuple] = []
        parsed = edge_rows(rows)
        if parsed is None:
            return out
        for src, dst, etype, attrs in parsed:
            if src == doc_id:
                out.append((etype, "out", dst, attrs))
            elif dst == doc_id:
                out.append((etype, "in", src, attrs))
        return out

    def neighbours(self, doc_id: str, cap: int = 0) -> List[tuple]:
        """One-hop traversal served by TigerGraph, verified on first use.

        Returns the identical tuple shape as ``KnowledgeGraph.neighbours`` so
        ``GraphTools.traverse_graph`` cannot tell which backend answered.
        """
        budget = int(cap) or self.max_rows
        if not self._remote_ok("neighbours"):
            return self._local_neighbours(doc_id, budget)
        try:
            # ``vertex_id``: the parameter name the installed query declares. It
            # accepts any vertex type, so SPORT::/VENUE::/ATHLETE::/GAMES:: ids
            # walk too - that is what the multi-hop tools rely on.
            acc = accumulators(self.client, "tg_neighbours",
                               {"vertex_id": doc_id, "max_rows": budget})
            if acc.get("edges") is None:
                raise TigerGraphError(
                    f"unexpected tg_neighbours payload: {str(acc)[:200]}")
            hops = self._hydrate_edges(acc.get("edges"), doc_id)
        except (TigerGraphError, TigerGraphUnavailable) as exc:
            self._degrade("neighbours", exc)
            return self._local_neighbours(doc_id, budget)

        if self.checked["neighbours"] is None:
            full = self._local_neighbours(doc_id, 0)      # uncapped, for checking
            if not _consistent([(h[0], h[1], h[2]) for h in hops],
                               [(h[0], h[1], h[2]) for h in full], budget):
                self._degrade("neighbours", TigerGraphError(
                    f"server returned {len(hops)} hops from {doc_id} but the "
                    f"corpus graph has {len(full)} - is the ingest complete?"),
                    disagreed=True)
                return self._local_neighbours(doc_id, budget)
            self.checked["neighbours"] = True
            self.log(f"neighbours verified against the mirror "
                     f"({len(hops)} hops from {doc_id})")
            if len(full) > len(hops) >= budget:
                self.log(f"neighbours: row budget {budget} is below {doc_id}'s "
                         f"{len(full)} hops - raise TG_MAX_ROWS for a lossless run")
        # Counted only now: hops that failed the check above were not served.
        self.counters["neighbours"] += 1
        self.counters["remote_rows"] += len(hops)
        return hops[:budget]