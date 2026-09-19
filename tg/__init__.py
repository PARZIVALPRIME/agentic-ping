"""TigerGraph backend for the Olympics knowledge graph.

The benchmark normally builds its knowledge graph from the corpus
(``kg/builder.py``). This package makes TigerGraph a drop-in alternative store:

    tg/client.py    RESTPP + GSQL-server client (stdlib HTTP, no dependency)
    tg/schema.gsql  graph schema, loading jobs and the installed queries
    tg/loader.py    corpus-built graph -> TigerGraph (idempotent upserts)
    tg/backend.py   a KnowledgeGraph subclass served by TigerGraph queries
    tg/fake_server.py  in-process RESTPP double used by tools/test_tg_backend.py

``kg/backend.py::open_graph`` is the only entry point the rest of the code uses,
so enabling TigerGraph is a configuration change (``TG_ENABLED=true``) and an
outage degrades to the local graph instead of failing a run.
"""

from .backend import TigerGraphBackend
from .client import TigerGraphClient, TigerGraphError, TigerGraphUnavailable
from .loader import push_graph, verify_counts

__all__ = [
    "TigerGraphBackend",
    "TigerGraphClient",
    "TigerGraphError",
    "TigerGraphUnavailable",
    "push_graph",
    "verify_counts",
]