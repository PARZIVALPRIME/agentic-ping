"""Backend selection: the one place that decides local vs TigerGraph.

``open_graph`` is the only entry point the benchmark and the pipelines use, so
turning TigerGraph on is a configuration change and nothing else:

    config: TG_ENABLED=true + TG_HOST/TG_GRAPHNAME/TG_USERNAME/TG_PASSWORD
    code:   kg = open_graph(corpus_path, config)    # instead of load_or_build

Behaviour that matters for a judged demo:

* ``TG_ENABLED`` unset -> the local graph, exactly as before (the ``tg`` package
  is never imported, so the default path stays dependency- and network-free).
* ``TG_ENABLED=true`` and the server answers -> ``TigerGraphBackend``; the local
  graph is still built and kept as a mirror and fallback.
* ``TG_ENABLED=true`` but the server is down or the schema is not installed ->
  the local graph, with a printed reason. A network problem never fails a run.

``describe_backend`` reports what is in force, for the run log and the dashboard.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .builder import load_or_build
from .model import KnowledgeGraph


@dataclass
class BackendInfo:
    """What actually answered the graph calls in this process."""

    requested: str = "local"          # per configuration
    active: str = "local"             # local | tigergraph
    graphname: str = ""
    reason: str = ""                  # why local, when remote was requested
    remote_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_LAST_INFO = BackendInfo()
# The object that actually answered the last open_graph call, kept so
# describe_backend() can report *live* counters. Snapshotting them at startup
# would report zeros for every run, which is exactly the metric that
# distinguishes a real TigerGraph run from one that fell back mid-flight.
_LAST_GRAPH: Optional[KnowledgeGraph] = None


def last_backend_info() -> BackendInfo:
    """Info for the most recent ``open_graph`` call (for summaries and logs)."""
    return _LAST_INFO


def _flag(value: Any) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _tg_config(cfg: Any) -> Any:
    """Pull a ``TigerGraphConfig`` out of a config object or a plain mapping."""
    tg = getattr(cfg, "tg", None)
    if tg is None and isinstance(cfg, dict):
        tg = cfg.get("tg")
    return tg


def probe_tigergraph(cfg: Any) -> tuple:
    """Build a client from ``config.tg`` and verify it can serve the graph.

    Returns ``(client_or_None, reason)``. The reason is written for a human
    reading a run log, because every failure here ends in a local fallback and a
    silent fallback would make the dashboard claim a backend it never used.
    """
    from tg.client import TigerGraphClient, TigerGraphError

    required = ("Event", "Athlete", "Games", "Sport", "Venue")
    tg = _tg_config(cfg)
    if tg is None:
        return None, "no [tg] configuration section"
    client = TigerGraphClient(
        host=tg.host, graphname=tg.graphname, username=tg.username,
        password=tg.password, restpp_port=tg.restpp_port, gs_port=tg.gs_port,
        token=tg.token, timeout=float(getattr(tg, "timeout", 30.0)),
        retries=int(getattr(tg, "retries", 2)),
        verbose=bool(getattr(tg, "verbose", False)))
    try:
        client.ping()
    except TigerGraphError as exc:
        return None, f"unreachable at {tg.host} ({exc})"
    try:
        missing = client.missing_types(required)
    except TigerGraphError as exc:
        return None, f"schema not readable ({exc})"
    if missing:
        return None, (f"vertex types missing: {', '.join(missing)} "
                      f"-> run: python tools/tg_ingest.py --install --push")
    queries = client.installed_queries()
    if not queries:
        return None, ("no installed queries; the backend needs tg_filter_events, "
                      "tg_event, tg_neighbours and tg_stats "
                      "-> run: python tools/tg_ingest.py --install")
    return client, ""


def open_graph(corpus_path: str, cfg: Any = None,
               cache_path: str = "results/knowledge_graph.json",
               force_local: bool = False,
               verbose: bool = False) -> KnowledgeGraph:
    """Load the knowledge graph from the configured backend.

    Accepts an object with a ``.tg`` attribute (the project's ``config``) or a
    plain mapping, so callers can pass a hand-built config in tests.
    """
    global _LAST_INFO, _LAST_GRAPH

    if cfg is None:
        try:
            from config import config as cfg  # type: ignore
        except Exception:
            cfg = None
    tg_cfg = _tg_config(cfg)

    enabled = _flag(os.getenv("TG_ENABLED")) or bool(getattr(tg_cfg, "enabled", False))
    graph_name = str(getattr(tg_cfg, "graphname", "") or "")
    client, reason = (None, "")
    if enabled and not force_local:
        client, reason = probe_tigergraph(cfg)

    if client is not None:
        # The mirror is always built: it backs passage retrieval, the vocabulary
        # accessors and the fallback path, and it is what the remote answers are
        # verified against on first use.
        mirror = load_or_build(corpus_path, cache_path=cache_path)
        from tg.backend import TigerGraphBackend

        backend = TigerGraphBackend(
            mirror, client, graphname=graph_name,
            max_rows=int(getattr(tg_cfg, "max_rows", 200)),
            verbose=verbose or bool(getattr(tg_cfg, "verbose", False)))
        _LAST_GRAPH = backend
        _LAST_INFO = BackendInfo(requested="tigergraph", active="tigergraph",
                                 graphname=graph_name, reason="")
        print(f"[kg] backend=tigergraph graph={graph_name} "
              f"host={getattr(tg_cfg, 'host', '')} "
              f"(local graph retained as mirror/fallback)")
        return backend

    if enabled:
        print(f"[kg] TigerGraph requested but unusable: {reason or 'forced local'}")
        print("[kg] falling back to the corpus-built local graph")
    kg = load_or_build(corpus_path, cache_path=cache_path)
    _LAST_GRAPH = kg
    _LAST_INFO = BackendInfo(
        requested="tigergraph" if enabled else "local", active="local",
        graphname=graph_name,
        reason=(reason or "forced local (--no-tg)") if enabled
               else ("TG_ENABLED not set" if tg_cfg is None else "disabled in config"))
    return kg


def describe_backend() -> Dict[str, Any]:
    """Summary of the active backend for the run log and the dashboard.

    ``remote_stats`` is read from the live backend rather than a startup
    snapshot, so a run that lost the remote path half way through reports that in
    its summary instead of claiming a healthy TigerGraph run.
    """
    info = _LAST_INFO
    stats = getattr(_LAST_GRAPH, "remote_stats", None)
    if callable(stats):
        try:
            info.remote_stats = stats()
        except Exception as exc:                   # never fail over reporting
            info.remote_stats = {"error": f"{type(exc).__name__}: {exc}"}
    return info.to_dict()