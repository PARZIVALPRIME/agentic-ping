"""Install the TigerGraph schema/queries and push the corpus graph.

The one-stop entry point the backend's own error messages point at
(``kg/backend.py`` says "run: python tools/tg_ingest.py --install --push"):

    python tools/tg_ingest.py --status              # what the server has now
    python tools/tg_ingest.py --install             # schema + queries + INSTALL
    python tools/tg_ingest.py --push                # upsert the corpus graph (+ verify)
    python tools/tg_ingest.py --install --push      # the whole ingest, in order
    python tools/tg_ingest.py --print               # print the GSQL for a manual paste

Connection settings come from the environment exactly as the benchmark reads
them (``TG_HOST``, ``TG_GRAPHNAME``, ``TG_USERNAME``, ``TG_PASSWORD``,
``TG_RESTPP_PORT``, ``TG_GS_PORT``, ``TG_TOKEN``), so "ingested successfully
into *this* server" means the same server ``TG_ENABLED=true`` will later use.

``--install`` talks to the GSQL server (port 14240). TigerGraph Savanna clouds
do not expose that port; there, run ``--print`` and paste the output into the
GSQL editor instead - the resulting schema/queries are identical.

``--push`` builds (or loads from cache) the same local graph the benchmark
uses, then streams idempotent upserts and finishes with a per-type count
verification (``tg/loader.verify_counts``). A mismatch is reported loudly and
the tool exits non-zero, because a partial ingest is exactly the failure mode
the backend's first-answer check is designed to catch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config as app_config  # noqa: E402  (project root on sys.path first)
from kg.builder import load_or_build  # noqa: E402
from tg import loader as tg_loader  # noqa: E402
from tg.client import TigerGraphClient, TigerGraphError  # noqa: E402

TG_DIR = os.path.dirname(os.path.abspath(tg_loader.__file__))
VERIFY_VERTEX_TYPES = ("Event", "Athlete", "Games", "Sport", "Venue")


def make_client(args: argparse.Namespace) -> TigerGraphClient:
    tg = app_config.tg
    return TigerGraphClient(
        host=args.host or tg.host, graphname=tg.graphname,
        username=tg.username, password=tg.password,
        restpp_port=tg.restpp_port, gs_port=tg.gs_port, token=tg.token,
        timeout=float(tg.timeout), retries=int(tg.retries),
        verbose=bool(tg.verbose))


def read_gsql(name: str) -> str:
    with open(os.path.join(TG_DIR, name), "r", encoding="utf-8") as fh:
        return fh.read()


def do_install(client: TigerGraphClient, drop: bool) -> bool:
    """Post schema.gsql then queries.gsql, then INSTALL QUERY ALL. Best effort."""
    print(f"[tg-ingest] installing schema into graph {client.graphname!r} "
          f"(GSQL server {client._gs})")
    if drop:
        try:
            print("[tg-ingest] drop:", client.gsql(f"DROP GRAPH {client.graphname}",
                                                   tag="benchmark") or "ok")
        except TigerGraphError as exc:
            print(f"[tg-ingest] drop failed (continuing): {exc}")
    ok = True
    for name in ("schema.gsql", "queries.gsql"):
        try:
            answer = client.gsql(read_gsql(name), tag="benchmark")
            print(f"[tg-ingest] {name}: {str(answer)[:200] or 'ok'}")
        except TigerGraphError as exc:
            ok = False
            print(f"[tg-ingest] {name} FAILED: {exc}")
    if not ok:
        return False
    try:
        print("[tg-ingest] INSTALL QUERY ALL:",
              str(client.gsql("USE GRAPH " + client.graphname + "\n"
                              "INSTALL QUERY ALL", tag="benchmark"))[:200] or "ok")
    except TigerGraphError as exc:
        print(f"[tg-ingest] INSTALL QUERY ALL failed (queries may already be "
              f"installed): {exc}")
    try:
        print("[tg-ingest] installed queries:",
              ", ".join(client.installed_queries()) or "(none reported)")
    except TigerGraphError:
        pass
    return True


def do_push(client: TigerGraphClient, args: argparse.Namespace) -> bool:
    kg = load_or_build(args.corpus, cache_path=args.kg_cache)
    print(f"[tg-ingest] graph built: {len(kg.events)} events, "
          f"{len(kg.athletes)} athletes, {len(kg.edges)} edges")
    report = tg_loader.push_graph(
        kg, client, corpus_path=args.corpus, batch_size=args.batch_size,
        max_events=args.max_events, with_text=not args.no_text)
    if args.json:
        print(json.dumps({"push": report}, indent=2))
    else:
        print(f"[tg-ingest] pushed in {report['batches']} batches "
              f"({report['seconds']}s): " +
              ", ".join(f"{v}={n.get('sent', 0)}/{n.get('target', '?')}"
                        for v, n in report["vertices"].items()) +
              "; edges: " + ", ".join(f"{e}={n.get('sent', 0)}"
                                      for e, n in report["edges"].items()))
    for err in report["errors"]:
        print(f"[tg-ingest] ERROR {err.get('what')}: {err.get('error')}")
    if not report["ok"]:
        return False
    if args.no_verify:
        return True

    verify = tg_loader.verify_counts(kg, client)
    if args.json:
        print(json.dumps({"verify": verify}, indent=2))
    else:
        for vtype, row in verify["types"].items():
            mark = {True: "OK", False: "MISMATCH", None: "?"}[row["match"]]
            print(f"[tg-ingest] verify {vtype}: local={row['local']} "
                  f"remote={row['remote']} {mark}")
        if verify["missing_vertex_types"]:
            print("[tg-ingest] verify: missing vertex types "
                  f"{verify['missing_vertex_types']} - schema not installed?")
    return bool(verify["ok"])


def do_status(client: TigerGraphClient) -> int:
    print(f"[tg-ingest] server {client.host} graph {client.graphname!r}")
    try:
        print("[tg-ingest] echo:", client.ping())
    except TigerGraphError as exc:
        print(f"[tg-ingest] unreachable: {exc}")
        return 1
    missing = client.missing_types(VERIFY_VERTEX_TYPES)
    print("[tg-ingest] vertex types:",
          "all present" if not missing else f"missing {missing}")
    try:
        print("[tg-ingest] installed queries:",
              ", ".join(client.installed_queries()) or "(none reported)")
    except TigerGraphError as exc:
        print(f"[tg-ingest] installed queries: unreadable ({exc})")
    print("[tg-ingest] remote counts:", tg_loader.remote_counts(client))
    return 1 if missing else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--install", action="store_true",
                    help="post tg/schema.gsql + tg/queries.gsql to the GSQL server")
    ap.add_argument("--push", action="store_true",
                    help="upsert the corpus-built graph into the server")
    ap.add_argument("--status", action="store_true",
                    help="report what the server has and exit")
    ap.add_argument("--print", dest="print_gsql", action="store_true",
                    help="print the GSQL files for a manual paste (Savanna)")
    ap.add_argument("--drop", action="store_true",
                    help="prepend DROP GRAPH before installing (schema changes)")
    ap.add_argument("--corpus", default=app_config.benchmark.corpus_path)
    ap.add_argument("--kg-cache", default=app_config.benchmark.kg_cache_path)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--max-events", type=int, default=0,
                    help="push a prefix of the events only (smoke tests)")
    ap.add_argument("--no-text", action="store_true",
                    help="skip article text on Event vertices")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the post-push count verification")
    ap.add_argument("--host", default="", help="override TG_HOST")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable push/verify reports")
    args = ap.parse_args(argv)

    if not (args.install or args.push or args.status or args.print_gsql):
        ap.print_help()
        return 2

    if args.print_gsql:
        for name in ("schema.gsql", "queries.gsql"):
            print(f"/* {'-' * 10} tg/{name} {'-' * 10} */")
            print(read_gsql(name))
        return 0

    client = make_client(args)
    if args.status:
        return do_status(client)
    ok = True
    if args.install:
        ok = do_install(client, drop=args.drop)
    if args.push:
        ok = do_push(client, args) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
