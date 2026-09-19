"""An in-process RESTPP double, so the TigerGraph path is testable offline.

Why this exists (and why it is not just "a mock"): the risk in this backend is not
that the HTTP call fails - that degrades safely - but that the *server's* answer is
subtly unlike the local one. So this fake implements the semantics of the queries in
``tg/queries.gsql`` as literally as it can:

    tg_filter_events   same WHERE clauses, corpus order (``seq``), row budget
    tg_neighbours      undirected hops, one accumulator row per edge
    tg_event           one Event by primary id
    tg_stats           per-type counts

It also speaks the same RESTPP envelopes (``{"results": [{"@@acc": ...}]}``,
``{"error": false, ...}``), upsert payloads and token handshake, so ``TigerGraphClient``
runs unmodified against it. That makes ``tools/test_tg_backend.py`` a real check of
the backend, the client, the loader and the query contract together - no TigerGraph
install, no network, runnable in the same pass as the rest of the benchmark.

Fault injection is deliberate: ``fail_queries`` / ``fail_upserts`` / ``phantom_ids``
let the test prove that degradation and verification actually fire.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlsplit

# Vertex types and their primary-id attribute, mirroring tg/schema.gsql. The id
# attribute matters because the installed queries compare against these names.
ID_ATTR = {"Event": "doc_id", "Athlete": "athlete_id", "Games": "gid",
           "Sport": "sid", "Venue": "vid"}
EDGE_TYPES = ("PREV", "NEXT", "PART_OF", "IN_SPORT", "HELD_AT", "WON_BY")
QUERIES = ("tg_filter_events", "tg_neighbours", "tg_event", "tg_stats")


def _query_params(query: Dict[str, List[str]]) -> Dict[str, str]:
    """Flatten a parsed query string to ``{name: first_value}``."""
    return {k: v[0] for k, v in query.items() if v}


class FakeHTTPError(Exception):
    """Raised inside the fake to produce a non-200 RESTPP response."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class FakeTigerGraph:
    """In-memory graph plus the RESTPP routes the client and loader use."""

    def __init__(self, graphname: str = "OlympicsKG") -> None:
        self.graphname = graphname
        self.vertices: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.edges: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
        self.installed: List[str] = list(QUERIES)
        # fault injection (used by the test to prove safe degradation)
        self.fail_queries: Dict[str, int] = {}
        self.fail_upserts = False
        self.phantom_ids: Dict[str, List[str]] = {}
        self.token_requests = 0
        self.query_calls: List[str] = []
        self.upserts = 0
        self.lock = threading.Lock()

    # ─ data ───────────────────────────────────────────────────────────
    def add_vertex(self, vtype: str, v_id: str, attributes: Dict[str, Any]) -> None:
        self.vertices.setdefault(vtype, {})[v_id] = dict(attributes)

    def add_edge(self, src_type: str, etype: str, src: str, dst: str,
                 attrs: Optional[Dict[str, Any]] = None) -> None:
        bucket = self.edges.setdefault(src_type, {}).setdefault(etype, {})
        bucket.setdefault(src, {})[dst] = dict(attrs or {})

    def counts(self) -> Dict[str, int]:
        out = {vtype: len(rows) for vtype, rows in self.vertices.items()}
        for blocks in self.edges.values():
            for etype, sources in blocks.items():
                out[etype] = sum(len(t) for t in sources.values())
        return out

    def _vertex_json(self, vtype: str, v_id: str) -> Dict[str, Any]:
        return {"v_id": v_id, "v_type": vtype,
                "attributes": self.vertices.get(vtype, {}).get(v_id, {})}

    # ── query emulation (mirrors tg/queries.gsql) ──────────────────────
    def tg_filter_events(self, p: Dict[str, str]) -> Dict[str, Any]:
        sport = (p.get("sport") or "").lower()
        venue = (p.get("venue") or "").lower()
        season = (p.get("season") or "").lower()
        year_from = int(p.get("year_from") or 0)
        year_to = int(p.get("year_to") or 0)
        max_rows = int(p.get("max_rows") or 100000)
        rows = []
        for v_id, attrs in self.vertices.get("Event", {}).items():
            if sport and str(attrs.get("sport", "")).lower() != sport:
                continue
            if venue and str(attrs.get("venue", "")).lower() != venue:
                continue
            if season and str(attrs.get("season", "")).lower() != season:
                continue
            year = int(attrs.get("year") or 0)
            if year_from and year < year_from:
                continue
            if year_to and year > year_to:
                continue
            rows.append((int(attrs.get("seq") or 0), v_id, attrs))
        rows.sort(key=lambda r: r[0])            # ORDER BY v.seq ASC
        rows = rows[:max_rows]                   # LIMIT bounds the ACCUM clause, so
        #                                          both accumulators are one page
        # Fault injection: vertices the corpus never produced, appended to *both*
        # accumulators exactly like a real row would be. The point is that the
        # backend has to catch it by comparing against its mirror, not by noticing
        # a malformed payload - so this must look like any other page.
        for offset, phantom in enumerate(self.phantom_ids.get("tg_filter_events", [])):
            rows = rows + [(max_rows + offset, phantom, {})]
        return {"@@ids": [r[1] for r in rows],
                "@@rows": [self._vertex_json("Event", v_id)
                           for _seq, v_id, _attrs in rows]}

    def tg_neighbours(self, p: Dict[str, str]) -> Dict[str, Any]:
        vertex_id = p.get("vertex_id") or ""
        wanted = (p.get("edge_type") or "").upper()
        max_rows = int(p.get("max_rows") or 200)
        out: List[Dict[str, Any]] = []
        # An undirected hop: every edge with either end at this vertex. The
        # direction is *not* recorded here on purpose - the backend must derive it
        # from from_id/to_id, exactly as it has to against the real server.
        for src_type, blocks in self.edges.items():
            for etype, sources in blocks.items():
                if wanted and etype != wanted:
                    continue
                for src, targets in sources.items():
                    for dst, attrs in targets.items():
                        if vertex_id not in (src, dst):
                            continue
                        out.append({"from_id": src, "from_type": src_type,
                                    "to_id": dst, "to_type": "", "e_type": etype,
                                    "attributes": attrs, "directed": True})
                        if len(out) >= max_rows:
                            return {"@@edges": out}
        return {"@@edges": out}

    def tg_event(self, p: Dict[str, str]) -> Dict[str, Any]:
        v_id = p.get("doc_id") or ""
        attrs = self.vertices.get("Event", {}).get(v_id)
        return {"@@rows": [self._vertex_json("Event", v_id)] if attrs else []}

    def tg_stats(self, _p: Dict[str, str]) -> Dict[str, Any]:
        c = self.counts()
        return {"@@events": c.get("Event", 0), "@@athletes": c.get("Athlete", 0),
                "@@games": c.get("Games", 0), "@@sports": c.get("Sport", 0),
                "@@venues": c.get("Venue", 0)}

    def run_query(self, name: str, params: Dict[str, str]) -> Dict[str, Any]:
        with self.lock:
            self.query_calls.append(name)
            if name in self.fail_queries:
                raise FakeHTTPError(self.fail_queries[name],
                                    f"injected failure for {name}")
            if name not in self.installed:
                raise FakeHTTPError(404, f"query {name} is not installed")
            fn = getattr(self, name, None)
            if fn is None:
                raise FakeHTTPError(404, f"no emulation for {name}")
            return fn(params)

    # ── RESTPP shapes ──────────────────────────────────────────────────
    def upsert(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.fail_upserts:
            raise FakeHTTPError(500, "injected upsert failure")
        accepted_v = accepted_e = 0
        for vtype, rows in (payload.get("vertices") or {}).items():
            for v_id, attrs in rows.items():
                self.add_vertex(vtype, v_id, attrs or {})
                accepted_v += 1
        for src_type, blocks in (payload.get("edges") or {}).items():
            for etype, sources in blocks.items():
                for src, targets in sources.items():
                    for dst, attrs in targets.items():
                        self.add_edge(src_type, etype, src, dst, attrs or {})
                        accepted_e += 1
        return {"error": False, "results": {"accepted_vertices": accepted_v,
                                            "accepted_edges": accepted_e}}

    def stats(self) -> Dict[str, Any]:
        vertex = {v: {"COUNT": len(rows)}
                  for v, rows in self.vertices.items()}
        edge = {}
        for blocks in self.edges.values():
            for etype, sources in blocks.items():
                edge[etype] = {"COUNT": sum(len(t) for t in sources.values())}
        return {"error": False, "results": {"Vertex": vertex, "Edge": edge}}

    def request(self, method: str, path: str,
                query: Dict[str, List[str]], body: Optional[bytes]) -> Any:
        """Dispatch one request the way RESTPP would; returns the JSON payload."""
        parts = [unquote(p) for p in path.strip("/").split("/") if p]
        if parts and parts[0] == "restpp":
            parts = parts[1:]
        if not parts:
            raise FakeHTTPError(404, f"unhandled path {path}")
        head = parts[0]

        if head == "echo":
            return {"error": False, "message": "Hello GSQL"}
        if head == "version":
            return {"error": False, "message": "fake", "version": {"product": "tigergraph",
                                                                  "version": "0.0-fake"}}
        if head == "requesttoken":
            self.token_requests += 1
            return {"token": f"fake-token-{self.token_requests}",
                    "expiration": 4102444800}          # 2100-01-01, far future
        if head == "graph" and len(parts) >= 2:
            graph = parts[1]
            if graph != self.graphname:
                raise FakeHTTPError(404, f"graph {graph} does not exist")
            if len(parts) == 3 and parts[2] == "stats":
                return self.stats()
            if method == "POST" and len(parts) == 2:
                if not body:
                    return {"error": False, "results": {}}
                return self.upsert(json.loads(body.decode("utf-8")))
            if len(parts) >= 4 and parts[2] == "vertices":
                vtype = parts[3]
                if vtype not in self.vertices:
                    raise FakeHTTPError(404, f"vertex type {vtype} is not defined")
                if len(parts) == 4:
                    rows = list(self.vertices[vtype].items())
                    limit = int((query.get("limit") or ["0"])[0] or 0)
                    if limit:
                        rows = rows[:limit]
                    return {"error": False,
                            "results": [self._vertex_json(vtype, i) for i, _a in rows]}
                v_id = parts[4]
                if v_id not in self.vertices[vtype]:
                    return {"error": False, "results": []}
                return {"error": False, "results": [self._vertex_json(vtype, v_id)]}
            if len(parts) >= 5 and parts[2] == "edges":
                src_type, src_id, etype = parts[3], parts[4], parts[5]
                rows = (self.edges.get(src_type, {}).get(etype, {})
                        .get(src_id, {}))
                return {"error": False,
                        "results": [{"from_id": src_id, "to_id": dst,
                                     "e_type": etype, "attributes": attrs}
                                    for dst, attrs in rows.items()]}
            raise FakeHTTPError(404, f"unhandled graph path {path}")
        if head == "query" and len(parts) >= 2:
            graph = parts[1]
            if graph != self.graphname:
                raise FakeHTTPError(404, f"graph {graph} does not exist")
            if len(parts) == 2:
                return {"error": False, "results": [{n: {} for n in self.installed}]}
            name = parts[2]
            params = _query_params(query)
            if method == "POST" and body:
                params.update({k: str(v) for k, v in json.loads(body).items()})
            payload = self.run_query(name, params)
            return {"error": False, "results": [payload]}
        if head == "gsqlserver":
            # Schema/query installation: accepted and (for the fake) a no-op.
            return {"error": False, "message": "fake GSQL server accepted the script"}
        raise FakeHTTPError(404, f"unhandled path {path}")


class _Handler(BaseHTTPRequestHandler):
    """Thin HTTP skin over ``FakeTigerGraph.request``."""

    server_version = "FakeTigerGraph/1.0"

    def log_message(self, *args: Any) -> None:      # keep test output clean
        pass

    def _dispatch(self, method: str) -> None:
        fake: FakeTigerGraph = getattr(self.server, "fake")
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        split = urlsplit(self.path)
        try:
            payload = fake.request(method, split.path, parse_qs(split.query), body)
            self._send(200, payload)
        except FakeHTTPError as exc:
            self._send(exc.status, {"error": True, "message": str(exc)})
        except Exception as exc:                   # noqa: BLE001 - test harness
            self._send(500, {"error": True,
                             "message": f"{type(exc).__name__}: {exc}"})

    def do_GET(self) -> None:                      # noqa: N802 - http.server API
        self._dispatch("GET")

    def do_POST(self) -> None:                     # noqa: N802 - http.server API
        self._dispatch("POST")

    def _send(self, status: int, payload: Any) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class FakeServerHandle:
    """A running fake: ``url`` for a client, ``graph`` for assertions."""

    def __init__(self, graph: FakeTigerGraph, httpd: ThreadingHTTPServer,
                 thread: threading.Thread) -> None:
        self.graph = graph
        self.httpd = httpd
        self.thread = thread
        self.url = f"http://{httpd.server_address[0]}:{httpd.server_address[1]}"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def install(self, *names: str) -> None:
        """Restrict the installed queries (to exercise the missing-query path)."""
        self.graph.installed = [n for n in names]

    def __enter__(self) -> "FakeServerHandle":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def start_fake_server(graph: Optional[FakeTigerGraph] = None,
                      host: str = "127.0.0.1", port: int = 0) -> FakeServerHandle:
    """Start the fake on an ephemeral port and return its handle."""
    graph = graph or FakeTigerGraph()
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.fake = graph                            # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return FakeServerHandle(graph, httpd, thread)