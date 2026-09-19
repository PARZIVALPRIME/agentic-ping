"""Serve the in-process RESTPP double as a real HTTP endpoint.

``tools/test_tg_backend.py`` drives the backend in-process; this script does the
same thing for the *whole benchmark*, so a TigerGraph-backed run is reproducible
with no TigerGraph install, no network and no credentials:

    python tools/tg_fake_server.py --port 19123            # terminal 1
    $env:TG_ENABLED="true"; $env:TG_HOST="http://127.0.0.1:19123"
    python run_benchmark.py --limit 3 --no-llm             # terminal 2

It ingests the corpus-built graph through the loader's own path before serving,
so the backend receives real rows and its first-use verification against the
local mirror passes. That is what makes the run's backend metadata meaningful:
without an ingest the probe would still connect, and the first query would be
rejected as an incomplete graph (which is the fail-safe, not a backend test).

Caveat, stated plainly: this emulates the *semantics* of ``tg/queries.gsql``, not
TigerGraph itself. Use it to prove the wiring end to end; use a real server to
prove the GSQL.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config                                        # noqa: E402
from kg.builder import load_or_build                             # noqa: E402
from tg import loader as tg_loader                               # noqa: E402
from tg.client import TigerGraphClient                           # noqa: E402
from tg.fake_server import FakeTigerGraph, start_fake_server      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the fake RESTPP server")
    ap.add_argument("--port", type=int, default=19123,
                    help="port to bind on 127.0.0.1 (0 = pick a free one)")
    ap.add_argument("--graphname", default=config.tg.graphname)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--with-text", action="store_true",
                    help="ship article text too (bigger payloads, longer load)")
    ap.add_argument("--no-push", action="store_true",
                    help="serve an empty graph (to watch the verification refuse it)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    handle = start_fake_server(FakeTigerGraph(args.graphname), port=args.port)
    print(f"fake RESTPP at {handle.url}  graph={args.graphname}", flush=True)
    if not args.no_push:
        print("building the corpus graph and pushing it ...", flush=True)
        started = time.time()
        kg = load_or_build(config.benchmark.corpus_path,
                           cache_path=config.benchmark.kg_cache_path)
        client = TigerGraphClient(host=handle.url, graphname=args.graphname,
                                  username="tigergraph", password="tigergraph",
                                  retries=0, timeout=60.0)
        report = tg_loader.push_graph(kg, client, batch_size=args.batch_size,
                                      with_text=args.with_text, progress=False)
        verify = tg_loader.verify_counts(kg, client)
        print(f"ingested {len(kg.events)} events, {len(kg.edges)} edges in "
              f"{time.time() - started:.1f}s - counts match: {verify['ok']}",
              flush=True)
        if not report["ok"] or not verify["ok"]:
            print("ingest problems:", report.get("errors"), verify.get("types"),
                  flush=True)

    print("serving; press Ctrl+C to stop", flush=True)
    try:
        while handle.thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        handle.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
