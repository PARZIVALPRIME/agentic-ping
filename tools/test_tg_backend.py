"""Offline proof that the TigerGraph backend is wired correctly.

Runs the whole remote path against the in-process RESTPP double in
``tg/fake_server.py`` - loader, client, backend, tools, backend selection - and
compares every answer against the local corpus graph. No server, no network, no
credentials, so this runs in the same pass as the rest of the benchmark:

    python tools/test_tg_backend.py

What is asserted (and why each one matters):

 1. loader round-trip       every event/edge reaches the server, and the verifier
                            agrees on per-type counts.
 2. filter parity           remote and local ``filter_events`` return identical
                            doc_ids for sport/venue/season/year/query filters -
                            the aggregation and superlative paths.
 3. hop parity              remote and local ``neighbours`` agree, with direction
                            derived from the undirected edge rows - the multi-hop
                            and temporal paths.
 4. tool parity             ``GraphTools`` produces identical output on both
                            backends, so the agent cannot tell which one answered
                            (measured against a lossless row budget - the tools
                            report the candidate-set size, so a budget that
                            truncates would change ``count``/``num_matching_events``).
 5. verification fires      a server that returns a phantom row is detected and
                            the mirror answers instead (checked=False, reason kept).
 6. degradation fires       a missing query and a dead server both fall back to
                            the mirror with identical answers, and per entry point
                            (a missing tg_neighbours leaves tg_filter_events on).
 7. backend selection       ``kg.backend.open_graph`` picks TigerGraph when enabled
                            and local when not (and honours force_local).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config                                   # noqa: E402
from kg.backend import describe_backend, open_graph         # noqa: E402
from kg.builder import load_or_build                        # noqa: E402
from agents.tools import GraphTools                         # noqa: E402
from tg import loader as tg_loader                          # noqa: E402
from tg.backend import TigerGraphBackend                    # noqa: E402
from tg.client import TigerGraphClient                      # noqa: E402
from tg.fake_server import FakeTigerGraph, start_fake_server  # noqa: E402

PASSED = 0
FAILED = 0


def check(label: str, ok: bool, detail: str = "") -> bool:
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}" + (f"  <- {detail}" if detail else ""))
    return ok


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def ids_of(events) -> list:
    return [getattr(e, "doc_id", None) for e in events]


def hops_of(backend, doc_id: str) -> list:
    return [(h[0], h[1], h[2]) for h in backend.neighbours(doc_id)]


def make_client(url: str) -> TigerGraphClient:
    return TigerGraphClient(host=url, graphname="OlympicsKG",
                            username="tigergraph", password="tigergraph",
                            retries=0, timeout=10.0)


def make_backend(mirror, url: str, max_rows: int = 200) -> TigerGraphBackend:
    return TigerGraphBackend(mirror, make_client(url), graphname="OlympicsKG",
                             max_rows=max_rows)


# ── 1. loader round-trip ───────────────────────────────────────────────
def test_loader(mirror, handle) -> None:
    section("1. loader round-trip (RESTPP upsert)")
    report = tg_loader.push_graph(mirror, make_client(handle.url),
                                  batch_size=500, with_text=False,
                                  progress=False)
    check("push reports no errors", report["ok"], str(report.get("errors")))
    check("all events sent",
          report["vertices"]["Event"]["sent"] == len(mirror.events),
          f"{report['vertices']['Event']['sent']} != {len(mirror.events)}")
    check("every edge type sent",
          all(v["sent"] > 0 for v in report["edges"].values()),
          str(report["edges"]))
    check("no edges skipped",
          report.get("edges_skipped", 0) == 0, str(report.get("edges_skipped")))
    verify = tg_loader.verify_counts(mirror, make_client(handle.url))
    check("counts match per type", verify["ok"],
          f"{verify['types']} missing={verify['missing_vertex_types']}")
    check("installed queries visible",
          "tg_filter_events" in make_client(handle.url).installed_queries())


# ── 2/3. parity on both hot paths ──────────────────────────────────────
def test_filter_parity(mirror, remote, handle) -> None:
    section("2. filter parity: tg_filter_events vs local scan")
    cases = [
        ("no filter (budget 40)", dict(), 40),
        ("sport only", dict(sport="Canoeing"), 40),
        ("sport + year range", dict(sport="Athletics", year_from=1990,
                                    year_to=2010), 40),
        ("season", dict(season="Summer"), 25),
        ("venue", dict(venue="Eton Dorney"), 25),
        ("query ranking", dict(query="100 metres final gold", sport="Athletics"),
         25),
    ]
    for label, kwargs, cap in cases:
        local = mirror.filter_events(cap=cap, **kwargs)
        got = remote.filter_events(cap=cap, **kwargs)
        check(f"identical rows: {label}", ids_of(local) == ids_of(got),
              f"local={ids_of(local)[:3]}... remote={ids_of(got)[:3]}...")
    check("first remote answer was verified against the mirror",
          remote.checked["filter_events"] is True, str(remote.checked))
    check("remote rows came from the server",
          remote.counters["filter_events"] > 0, str(remote.counters))
    check("no fallback needed", not remote.disabled, str(remote.disabled))
    check("server attributes were used (not a local re-read)",
          remote.counters["remote_rows"] > 0, str(remote.counters))

    # A row budget below the candidate set is a safety valve, not a silent
    # truncation: the tools report the size of the candidate set, so the run has
    # to say which measurement its numbers came from.
    small = make_backend(mirror, handle.url, max_rows=5)
    got_small = small.filter_events(sport="Athletics", cap=5)
    note = next((n for n in small.notes if "row budget" in n), "")
    check("a budget below the candidate set is reported in the run notes",
          bool(note), str(small.notes))
    check("the truncated page is still the mirror's own rows",
          ids_of(got_small) == ids_of(mirror.filter_events(sport="Athletics",
                                                          cap=5)),
          f"remote={ids_of(got_small)[:3]}")


def test_hop_parity(mirror, remote) -> None:
    section("3. hop parity: tg_neighbours vs local adjacency")
    sample = [e.doc_id for e in mirror.events.values()
              if e.prev_doc_id or e.next_doc_id][:3]
    check("found events with PREV/NEXT to walk", bool(sample), str(sample))
    for doc_id in sample:
        local = hops_of(mirror, doc_id)
        got = hops_of(remote, doc_id)
        check(f"identical hops: {doc_id}", sorted(local) == sorted(got),
              f"local={sorted(local)} remote={sorted(got)}")
    check("neighbours answer was verified against the mirror",
          remote.checked["neighbours"] is True, str(remote.checked))
    check("neighbour hops were served remotely",
          remote.counters["neighbours"] > 0, str(remote.counters))
    check("edge attributes survived the round trip",
          all(isinstance(h[3], dict) for h in remote.neighbours(sample[0])),
          str(remote.neighbours(sample[0])[:2]))


# ── 4. the agent layer cannot tell the difference ──────────────────────
def test_tool_parity(mirror, remote) -> None:
    section("4. tool parity: GraphTools on both backends")
    local_tools = GraphTools(mirror, None)
    remote_tools = GraphTools(remote, None)
    calls = [
        ("search_events sport", "search_events", {"sport": "Canoeing", "limit": 10}),
        ("search_events query", "search_events",
         {"query": "women's 100 metres gold", "limit": 5}),
        ("get_event_values", "get_event_values",
         {"field": "competitors", "sport": "Athletics", "year_from": 2000}),
        ("traverse_graph PART_OF", "traverse_graph",
         {"doc_id": "2012 Summer Olympics", "edge_type": "PART_OF"}),
    ]
    doc_id = None
    for event in mirror.events.values():
        if event.prev_doc_id:
            doc_id = event.doc_id
            break
    calls.append(("traverse_graph PREV", "traverse_graph",
                  {"doc_id": doc_id, "edge_type": "PREV"}))
    for label, name, args in calls:
        a = local_tools.execute(name, dict(args))
        b = remote_tools.execute(name, dict(args))
        check(f"identical tool result: {label}", a == b,
              f"local={str(a)[:160]} remote={str(b)[:160]}")


# ── 5. verification catches a lying server ─────────────────────────────
def test_verification(mirror, handle) -> None:
    section("5. verification: a phantom row must be caught")
    handle.graph.phantom_ids["tg_filter_events"] = ["Q_DOES_NOT_EXIST"]
    remote = make_backend(mirror, handle.url, max_rows=40)
    local = mirror.filter_events(cap=40)
    got = remote.filter_events(cap=40)
    check("phantom row detected (checked=False)",
          remote.checked["filter_events"] is False, str(remote.checked))
    check("reason recorded for the run summary",
          "filter_events" in remote.disabled, str(remote.disabled))
    check("mirror answer served instead", ids_of(got) == ids_of(local),
          f"remote={ids_of(got)[:3]}")
    check("server no longer consulted for that path",
          remote.counters["filter_events"] == 0, str(remote.counters))
    handle.graph.phantom_ids.clear()


# ── 6. degradation, per entry point ────────────────────────────────────
def test_degradation(mirror, handle) -> None:
    section("6. degradation: missing query, then a dead server")
    handle.install("tg_filter_events", "tg_event", "tg_stats")   # no neighbours
    remote = make_backend(mirror, handle.url, max_rows=40)
    doc_id = next(e.doc_id for e in mirror.events.values() if e.prev_doc_id)
    got_filter = remote.filter_events(sport="Canoeing", cap=10)
    check("filter path still remote despite the missing tg_neighbours query",
          remote.checked["filter_events"] is True, str(remote.checked))
    check("hops fall back to the mirror unchanged",
          sorted(hops_of(remote, doc_id)) == sorted(hops_of(mirror, doc_id)),
          f"remote={sorted(hops_of(remote, doc_id))}")
    # Asserted after the hop call on purpose: the neighbours path is only probed
    # when something asks for hops, so this is the first moment it can be known.
    check("missing query only disabled its own path",
          list(remote.disabled) == ["neighbours"], str(remote.disabled))
    check("filter rows still correct after the other path degraded",
          ids_of(got_filter) == ids_of(mirror.filter_events(sport="Canoeing",
                                                            cap=10)))
    handle.install("tg_filter_events", "tg_neighbours", "tg_event", "tg_stats")

    # A server that simply is not there any more. It gets its own fake so the
    # shared one is still alive for the sections that follow - stopping it in
    # place would make every later section look like a backend failure.
    gone = start_fake_server(FakeTigerGraph("OlympicsKG"))
    gone_url = gone.url
    gone.stop()
    dead = make_backend(mirror, gone_url, max_rows=40)
    check("events still answerable with the server down",
          ids_of(dead.filter_events(cap=20)) == ids_of(mirror.filter_events(cap=20)))
    check("hops still answerable with the server down",
          sorted(hops_of(dead, doc_id)) == sorted(hops_of(mirror, doc_id)))
    check("both paths report why they degraded",
          set(dead.disabled) == {"filter_events", "neighbours"},
          str(dead.disabled))
    check("fallbacks counted", dead.counters["fallbacks"] >= 2,
          str(dead.counters))


# ── 7. backend selection ───────────────────────────────────────────────
class _Cfg:
    """Minimal stand-in for the project config with a pointable TG section."""

    def __init__(self, url: str, enabled: bool) -> None:
        self.tg = type("TG", (), {
            "host": url, "graphname": "OlympicsKG", "username": "tigergraph",
            "password": "tigergraph", "restpp_port": "", "gs_port": "",
            "token": "", "timeout": 10.0, "retries": 0, "verbose": False,
            "max_rows": 40, "enabled": enabled})()


def test_selection(mirror, handle) -> None:
    section("7. backend selection (kg.backend.open_graph)")
    corpus = config.benchmark.corpus_path
    cache = config.benchmark.kg_cache_path
    kg_tg = open_graph(corpus, _Cfg(handle.url, True), cache_path=cache)
    check("TigerGraph selected when enabled",
          describe_backend()["active"] == "tigergraph", str(describe_backend()))
    check("the selected graph is the remote backend",
          isinstance(kg_tg, TigerGraphBackend), type(kg_tg).__name__)

    kg_local = open_graph(corpus, _Cfg(handle.url, False), cache_path=cache)
    check("local graph when TG_ENABLED is off",
          describe_backend()["active"] == "local"
          and not isinstance(kg_local, TigerGraphBackend), str(describe_backend()))

    kg_forced = open_graph(corpus, _Cfg(handle.url, True), cache_path=cache,
                           force_local=True)
    check("force_local wins over the flag",
          describe_backend()["active"] == "local"
          and not isinstance(kg_forced, TigerGraphBackend), str(describe_backend()))

    # a dead endpoint must not raise, and must say why it fell back
    dead = open_graph(corpus, _Cfg("http://127.0.0.1:1", True), cache_path=cache)
    info = describe_backend()
    check("unreachable server degrades instead of raising",
          info["active"] == "local" and not isinstance(dead, TigerGraphBackend),
          str(info))
    check("the reason is recorded", bool(info["reason"]), str(info))


# ── main ───────────────────────────────────────────────────────────────
def main() -> int:
    print("TigerGraph backend verification (offline, fake RESTPP)")
    print(f"corpus: {config.benchmark.corpus_path}")
    mirror = load_or_build(config.benchmark.corpus_path,
                           cache_path=config.benchmark.kg_cache_path)
    print(f"mirror: {len(mirror.events)} events, {len(mirror.edges)} edges, "
          f"{len(mirror.athletes)} athletes")

    handle = start_fake_server(FakeTigerGraph("OlympicsKG"))
    print(f"fake RESTPP at {handle.url}")
    try:
        test_loader(mirror, handle)
        remote = make_backend(mirror, handle.url, max_rows=200)
        test_filter_parity(mirror, remote, handle)
        test_hop_parity(mirror, remote)
        # Tool parity runs against a lossless row budget, the one TG_MAX_ROWS
        # defaults to: the tools report the size of the candidate set and the raw
        # values behind it, so a budget that truncates would legitimately change
        # ``count``/``num_matching_events`` for a reason that has nothing to do
        # with the backend being wired correctly.
        test_tool_parity(mirror, make_backend(mirror, handle.url,
                                             max_rows=len(mirror.events)))
        test_verification(mirror, handle)
        test_degradation(mirror, handle)
        test_selection(mirror, handle)
    finally:
        if handle.thread.is_alive():
            handle.stop()

    print(f"\n{'=' * 60}")
    print(f"PASSED {PASSED}   FAILED {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())