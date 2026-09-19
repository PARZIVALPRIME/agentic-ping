"""Prove that a *running* graph store answers the agent's tools.

``tools/test_tg_backend.py`` proves the backend offline, in-process, against the
fake RESTPP double. This probe proves the same wiring end to end through the path
the benchmark itself uses - ``kg.backend.open_graph`` - against whatever host
``TG_HOST`` points at, fake or real:

    python tools/tg_fake_server.py --port 19123        # or a real TigerGraph
    $env:TG_ENABLED="true"; $env:TG_HOST="http://127.0.0.1:19123"
    python tools/probe_tg_live.py

What it checks, in order, and why each one matters for a demo:

  1. the configured host is selected, not silently replaced by the mirror
  2. every graph tool answers *identically* to the corpus-built mirror, so a
     TigerGraph run stays comparable to a local one question for question
  3. the answers really came from the server (counters moved, first answer
     verified against the mirror) - the difference between a database-backed run
     and a locally-answered one that merely points at a database
  4. a one-page row budget degrades loudly instead of looking like a small answer

Exit code is non-zero on the first failure, so it can gate a demo. Note that the
push-down is exercised by the agent's tool layer (``GraphTools``); the purely
deterministic solvers read the mirror's indexes, which is why a ``--no-llm``
benchmark run with no tool calls shows zero remote answers even when the backend
is perfectly healthy.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config                                     # noqa: E402
from agents.tools import GraphTools                           # noqa: E402
from kg.backend import open_graph                             # noqa: E402
from kg.builder import load_or_build                          # noqa: E402
from tg.backend import TigerGraphBackend                      # noqa: E402

FAILED = 0


def check(label: str, ok: bool, detail: str = "") -> bool:
    global FAILED
    if not ok:
        FAILED += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {label}"
          + (f"  <- {detail}" if detail and not ok else ""))
    return ok


def tg_cfg(host: str, max_rows: int) -> dict:
    """A config object shaped like ``config.tg``, to point a probe anywhere."""
    return {"tg": SimpleNamespace(
        host=host, graphname=config.tg.graphname, username="tigergraph",
        password="tigergraph", restpp_port="", gs_port="", token="",
        timeout=15.0, retries=0, verbose=False, max_rows=max_rows, enabled=True)}


def tool_calls(mirror) -> list:
    """Both hot paths, plus an aggregation and enough hops to be convincing."""
    doc_ids = [e.doc_id for e in mirror.events.values()][:400]
    with_prev = next(e.doc_id for e in mirror.events.values() if e.prev_doc_id)
    sport = max(mirror.sport_index, key=lambda s: len(mirror.sport_index[s]))
    calls = [
        ("search_events", {"sport": sport, "limit": 10}),
        ("search_events", {"query": "100 metres gold", "sport": "Athletics",
                           "limit": 5}),
        ("search_events", {"season": "Summer", "year_from": 2000, "year_to": 2020,
                           "limit": 10}),
        ("get_event_values", {"field": "competitors", "sport": sport}),
        ("traverse_graph", {"doc_id": with_prev, "edge_type": "PREV"}),
        ("traverse_graph", {"doc_id": doc_ids[0], "edge_type": "PART_OF"}),
        ("get_games_events", {"year": 2012, "season": "Summer"}),
    ]
    calls += [("traverse_graph", {"doc_id": d, "edge_type": "WON_BY"})
              for d in doc_ids[:12]]
    return calls


def main() -> int:
    corpus = config.benchmark.corpus_path
    cache = config.benchmark.kg_cache_path
    print(f"probe: TG_ENABLED={os.getenv('TG_ENABLED')!r} "
          f"TG_HOST={os.getenv('TG_HOST')!r}")
    mirror = load_or_build(corpus, cache_path=cache)
    print(f"mirror: {len(mirror.events)} events, {len(mirror.edges)} edges\n")

    print("1. selection through kg.backend.open_graph")
    graph = open_graph(corpus, config, cache_path=cache)
    if not check("the configured host was selected",
                 isinstance(graph, TigerGraphBackend),
                 f"active={type(graph).__name__} - is TG_ENABLED/TG_HOST set, and "
                 f"has the graph been ingested (tools/tg_fake_server.py)?"):
        return 1

    print("\n2. tool parity: remote answers vs the corpus-built mirror")
    local_tools, remote_tools = GraphTools(mirror, None), GraphTools(graph, None)
    calls = tool_calls(mirror)
    same, first_bad = 0, ""
    for name, args in calls:
        a = local_tools.execute(name, dict(args))
        b = remote_tools.execute(name, dict(args))
        if a == b:
            same += 1
        elif not first_bad:
            first_bad = f"{name} {args} -> local={str(a)[:110]} remote={str(b)[:110]}"
    check(f"all {len(calls)} tool answers identical to the mirror",
          same == len(calls), first_bad)

    print("\n3. the answers came from the server")
    stats = graph.remote_stats()
    counters = stats["counters"]
    check("remote filter_events served", counters["filter_events"] > 0, str(counters))
    check("remote hops served", counters["neighbours"] > 0, str(counters))
    check("remote rows arrived", counters["remote_rows"] > 0, str(counters))
    check("neighbours answer verified against the mirror",
          stats["verified"]["neighbours"] is True, str(stats["verified"]))
    check("no path silently degraded", not stats["disabled"], str(stats["disabled"]))
    print(f"       {counters} | {stats['client_requests']} RESTPP requests")
    for note in stats["notes"]:
        print(f"       note: {note}")

    print("\n4. a one-page row budget is reported, not silent")
    host = os.getenv("TG_HOST") or config.tg.host
    small = open_graph(corpus, tg_cfg(host, max_rows=5), cache_path=cache)
    if isinstance(small, TigerGraphBackend):
        small_tools = GraphTools(small, None)
        got = small_tools.execute("search_events", {"sport": "Athletics"})
        small_stats = small.remote_stats()
        note = next((n for n in small_stats["notes"] if "row budget" in n), "")
        print(f"       budget=5 -> {got.get('count')} candidates | "
              f"fell back: {list(small_stats['disabled'])}")
        print(f"       note: {note or '(none)'}")
        check("the truncation is reported in the run notes", bool(note),
              str(small_stats["notes"]))
        check("the remote page is still the mirror's own rows",
              got.get("count", 0) > 0 and not small_stats["disabled"],
              str(small_stats["counters"]))
    else:
        check("a pointed-at host was used for the budget probe", False,
              f"active={type(small).__name__}")

    print(f"\n{'=' * 60}\nFAILURES: {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())