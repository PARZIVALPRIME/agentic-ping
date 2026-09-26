"""Minimal, dependency-free TigerGraph RESTPP / GSQL-server client.

Why hand-rolled instead of ``pyTigerGraph``: the benchmark's core must run with
the packages in ``requirements.txt`` CORE, and the client only needs a handful of
endpoints. Everything is stdlib ``urllib``, so setting ``TG_ENABLED=true`` never
requires installing anything.

Endpoints used (all documented RESTPP behaviour):

    POST /restpp/requesttoken                    -> bearer token
    GET  /restpp/echo                            -> liveness probe
    GET  /restpp/version                         -> server version
    GET  /restpp/graph/{g}/stats                 -> vertex/edge counts
    POST /restpp/graph/{g}                       -> batch vertex/edge upsert
    GET  /restpp/graph/{g}/vertices/{type}       -> vertex scan / schema probe
    GET  /restpp/graph/{g}/vertices/{type}/{id}  -> single vertex
    GET  /restpp/graph/{g}/edges/{t}/{id}/{e}    -> edges of one vertex
    GET  /restpp/query/{g}                       -> installed queries
    GET|POST /restpp/query/{g}/{name}            -> run an installed query
    POST /gsqlserver/gsql/file                   -> install schema / queries

Failures are typed so callers can tell "server is down" (degrade to the local
graph) from "this query returned something unusable" (fall back for that call):

    TigerGraphUnavailable  - transport/auth problems (a retryable outage)
    TigerGraphError        - the server answered with an error for a request
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, List, Optional


class TigerGraphError(RuntimeError):
    """The server answered, but with an error."""

    def __init__(self, message: str, status: int = 0, payload: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload


class TigerGraphUnavailable(TigerGraphError):
    """The server could not be reached / authorised (degrade to local graph)."""


def _join(host: str, port: str) -> str:
    """Attach the RESTPP port unless the host already carries one.

    Local installs use ``http://localhost`` + 9000; remote installs give a portless
    URL that already serves RESTPP on 443. One rule
    covers both: only append a port when the host URL has none.
    """
    host = (host or "").strip().rstrip("/")
    if not host:
        return "http://localhost"
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    parts = urllib.parse.urlsplit(host)
    if parts.port or not port:
        return host
    return f"{parts.scheme}://{parts.netloc}:{port}{parts.path.rstrip('/')}"


class TigerGraphClient:
    """RESTPP / GSQL client with token auth, batching and bounded retries."""

    def __init__(self, host: str = "http://localhost", graphname: str = "OlympicsKG",
                 username: str = "tigergraph", password: str = "tigergraph",
                 restpp_port: str = "9000", gs_port: str = "14240",
                 token: str = "", timeout: float = 30.0, retries: int = 2,
                 verbose: bool = False) -> None:
        self.host = host
        self.graphname = graphname
        self.username = username
        self.password = password
        self.timeout = float(timeout or 30.0)
        self.retries = max(0, int(retries))
        self.verbose = verbose
        self._restpp = _join(host, restpp_port)
        self._gs = _join(host, gs_port)
        self._token = token or ""
        self._token_expiry = 0.0
        self.requests = 0
        self.retries_used = 0

    # ── logging ────────────────────────────────────────────────────────
    def log(self, message: str) -> None:
        if self.verbose:
            print(f"[tg] {message}", flush=True)

    # ── low level HTTP ─────────────────────────────────────────────────
    def _auth_headers(self, use_token: bool = True) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if use_token and self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _basic_headers(self) -> Dict[str, str]:
        import base64

        raw = f"{self.username}:{self.password}".encode("utf-8")
        return {"Authorization": "Basic " + base64.b64encode(raw).decode("ascii"),
                "Accept": "application/json"}

    def _open(self, url: str, method: str, body: Optional[bytes],
              headers: Dict[str, str]) -> Any:
        request = urllib.request.Request(url, data=body, method=method,
                                         headers=dict(headers))
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            if exc.code in (401, 403):
                raise TigerGraphUnavailable(
                    f"authentication failed ({exc.code}) for {url}: {detail}",
                    exc.code, detail) from exc
            # 404 on a query/vertex type is a normal "not installed" answer.
            raise TigerGraphError(f"HTTP {exc.code} for {url}: {detail}",
                                  exc.code, detail) from exc
        except urllib.error.URLError as exc:
            raise TigerGraphUnavailable(f"cannot reach {url}: {exc.reason}") from exc
        except OSError as exc:  # socket timeout, connection reset, ...
            raise TigerGraphUnavailable(f"cannot reach {url}: {exc}") from exc
        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw  # /restpp/echo returns bare text

    def request(self, method: str, url: str, params: Optional[Dict[str, Any]] = None,
                payload: Optional[Any] = None, form: Optional[Dict[str, Any]] = None,
                use_token: bool = True) -> Any:
        """One HTTP call with a small retry budget for transient failures."""
        if params:
            clean = {k: v for k, v in params.items() if v is not None and v != ""}
            if clean:
                url = f"{url}?{urllib.parse.urlencode(clean)}"
        body: Optional[bytes] = None
        headers = self._auth_headers(use_token)
        if form is not None:
            body = urllib.parse.urlencode(form).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        elif payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        attempt = 0
        while True:
            self.requests += 1
            try:
                return self._open(url, method, body, headers)
            except TigerGraphUnavailable:
                if attempt >= self.retries:
                    raise
            except TigerGraphError as exc:
                if attempt >= self.retries or (exc.status and exc.status < 500):
                    raise
            attempt += 1
            self.retries_used += 1
            time.sleep(min(4.0, 0.5 * (2 ** (attempt - 1))))

    # ── auth / liveness ────────────────────────────────────────────────
    def ensure_token(self, force: bool = False) -> str:
        """Fetch (and cache) a RESTPP bearer token."""
        if self._token and not force and (not self._token_expiry
                                          or time.time() < self._token_expiry - 30):
            return self._token
        try:
            data = self.request("POST", f"{self._restpp}/requesttoken",
                                form={"secret": self.password}, use_token=False)
        except TigerGraphError:
            # Community Edition also accepts basic auth on this endpoint.
            data = self.request("GET", f"{self._restpp}/requesttoken",
                                use_token=False)
        if isinstance(data, dict) and data.get("token"):
            self._token = str(data["token"])
            self._token_expiry = float(data.get("expiration") or 0)
            self.log(f"token acquired (expires in "
                     f"{max(0, int(self._token_expiry - time.time()))}s)")
            return self._token
        raise TigerGraphUnavailable(
            f"no token in /requesttoken response: {str(data)[:200]}")

    def ping(self) -> str:
        """Liveness probe: returns the server's echo string."""
        try:
            self.ensure_token()
        except TigerGraphError as exc:
            raise TigerGraphUnavailable(f"token request failed: {exc}") from exc
        data = self.request("GET", f"{self._restpp}/echo")
        if isinstance(data, dict):
            if data.get("error"):
                raise TigerGraphError(f"echo failed: {data}")
            return str(data.get("message") or "ok")
        return str(data).strip() or "ok"

    def version(self) -> Dict[str, Any]:
        data = self.request("GET", f"{self._restpp}/version")
        return data if isinstance(data, dict) else {"version": data}

    # ── schema / graph ─────────────────────────────────────────────────
    def graph_stats(self) -> Dict[str, Any]:
        data = self.request("GET", f"{self._restpp}/graph/{self.graphname}/stats")
        if isinstance(data, dict) and data.get("error"):
            raise TigerGraphError(f"stats failed: {data}")
        return data if isinstance(data, dict) else {}

    def upsert(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Batch upsert: ``{"vertices": {...}}`` and/or ``{"edges": {...}}``."""
        data = self.request("POST", f"{self._restpp}/graph/{self.graphname}",
                            payload=payload)
        if isinstance(data, dict) and data.get("error"):
            raise TigerGraphError(f"upsert rejected: {data}")
        return data if isinstance(data, dict) else {}

    def upsert_vertices(self, vertex_type: str,
                        vertices: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        return self.upsert({"vertices": {vertex_type: vertices}})

    def upsert_edges(self, src_type: str, edge_type: str,
                     edges: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Edges are keyed by ``"<src_id>"`` -> ``{"<dst_id>": {attrs}}``."""
        return self.upsert({"edges": {src_type: {edge_type: edges}}})

    def get_vertices(self, vertex_type: str, limit: int = 0,
                     skip: int = 0) -> List[Dict[str, Any]]:
        """Scan a vertex type. Returns ``[{v_id, attributes}, ...]``."""
        data = self.request(
            "GET", f"{self._restpp}/graph/{self.graphname}/vertices/{vertex_type}",
            params={"limit": limit or None, "skip": skip or None})
        return _rows(data)

    def get_vertex(self, vertex_type: str, v_id: str) -> Dict[str, Any]:
        data = self.request(
            "GET", f"{self._restpp}/graph/{self.graphname}/vertices/{vertex_type}/"
                   f"{urllib.parse.quote(str(v_id), safe='')}")
        rows = _rows(data)
        return rows[0] if rows else {}

    def get_edges(self, src_type: str, src_id: str, edge_type: str,
                  limit: int = 0) -> List[Dict[str, Any]]:
        data = self.request(
            "GET", f"{self._restpp}/graph/{self.graphname}/edges/{src_type}/"
                   f"{urllib.parse.quote(str(src_id), safe='')}/{edge_type}",
            params={"limit": limit or None})
        return _rows(data)

    def vertex_type_exists(self, vertex_type: str) -> bool:
        try:
            self.get_vertices(vertex_type, limit=1)
            return True
        except TigerGraphError:
            return False

    def missing_types(self, vertex_types: Iterable[str]) -> List[str]:
        return [v for v in vertex_types if not self.vertex_type_exists(v)]

    # ── queries ────────────────────────────────────────────────────────
    def installed_queries(self) -> List[str]:
        """Names of the queries installed on the graph (empty when unsure)."""
        try:
            data = self.request("GET", f"{self._restpp}/query/{self.graphname}")
        except TigerGraphError:
            return []
        results = data.get("results") if isinstance(data, dict) else data
        if isinstance(results, list) and results and isinstance(results[0], dict):
            return [n for n in results[0] if not n.startswith("_")]
        return []

    def run_query(self, name: str, params: Optional[Dict[str, Any]] = None,
                  method: str = "GET", raw: bool = False) -> Any:
        """Run an installed query and unwrap the RESTPP envelope.

        RESTPP wraps results as ``{"results": [ ... ]}``; a query that ``PRINT``s
        an accumulator comes back as ``[{"@@name": value}]``. Callers want the
        value, so both envelopes are stripped here. A query the server does not
        have raises ``TigerGraphError`` (no retry on 404) so the backend can fall
        back for that single call instead of abandoning the run.
        """
        url = f"{self._restpp}/query/{self.graphname}/{name}"
        if method == "POST":
            data = self.request("POST", url, payload=params or {})
        else:
            data = self.request("GET", url, params=params or {})
        if isinstance(data, dict) and data.get("error"):
            raise TigerGraphError(f"query {name} failed: {str(data)[:300]}")
        results = data.get("results") if isinstance(data, dict) else data
        if raw:
            return results
        if isinstance(results, list):
            if not results:
                return []
            first = results[0]
            if isinstance(first, dict):
                for key, value in first.items():
                    if key.startswith("@@"):
                        return value
                return results
            return results
        return results if results is not None else []

    def run_accumulators(self, name: str, params: Optional[Dict[str, Any]] = None,
                         method: str = "GET") -> Dict[str, Any]:
        """Run a query and return *every* printed accumulator, ``@@`` stripped.

        ``run_query`` gives callers one value; a query that prints several
        accumulators (``tg_filter_events`` prints the ordered ids and the vertex
        rows together) needs them all, and the keys must stay distinguishable.
        Non-accumulator results come back under ``"results"``.
        """
        data = self.request("GET" if method == "GET" else "POST",
                            f"{self._restpp}/query/{self.graphname}/{name}",
                            params=params or {}, payload=params or {})
        if isinstance(data, dict) and data.get("error"):
            raise TigerGraphError(f"query {name} failed: {str(data)[:300]}")
        results = data.get("results") if isinstance(data, dict) else data
        if not isinstance(results, list) or not results:
            return {}
        first = results[0]
        out: Dict[str, Any] = {}
        if isinstance(first, dict):
            for key, value in first.items():
                out[key[2:] if key.startswith("@@") else key] = value
        else:
            out["results"] = results
        return out

    # ── GSQL server (schema install, best effort) ──────────────────────
    def gsql(self, script: str, tag: Optional[str] = None) -> Any:
        """Submit raw GSQL to the GSQL server (schema/query installation).

        Savanna installs the schema through its console; on Community Edition
        this endpoint does the same job as ``gsql schema.gsql``. Failures are
        reported rather than hidden, because the loader wants to print the exact
        manual command when this route is not available.
        """
        url = f"{self._gs}/gsqlserver/gsql/file"
        if tag:
            url = f"{url}?tag={urllib.parse.quote(tag)}"
        request = urllib.request.Request(
            url, data=script.encode("utf-8"), method="POST",
            headers={**self._basic_headers(), "Content-Type": "text/plain"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            raise TigerGraphError(
                f"GSQL server rejected the script (HTTP {exc.code}): "
                f"{exc.read().decode('utf-8', 'replace')[:400]}", exc.code) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise TigerGraphUnavailable(f"GSQL server unreachable: {exc}") from exc
        return raw


def _rows(data: Any) -> List[Dict[str, Any]]:
    """Normalise RESTPP's several vertex/edge result shapes into row dicts."""
    if isinstance(data, dict):
        if data.get("error"):
            return []
        data = data.get("results", data)
    out: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        for v_id, attrs in data.items():
            if isinstance(attrs, dict) and "attributes" in attrs:
                out.append({"v_id": v_id, "attributes": attrs.get("attributes") or {}})
            elif isinstance(attrs, dict):
                out.append({"v_id": v_id, "attributes": attrs})
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            if "v_id" in item:
                out.append({"v_id": item["v_id"],
                            "attributes": item.get("attributes") or {}})
                continue
            for v_id, attrs in item.items():
                if isinstance(attrs, dict):
                    out.append({"v_id": v_id,
                                "attributes": attrs.get("attributes", attrs)})
    return out