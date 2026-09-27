"""Production API Server for TigerGraph Agentic GraphRAG.

Exposes production endpoints:
  - POST /query      : Execute an Olympic question against any of the 4 pipelines
  - GET  /health     : Liveness probe
  - GET  /readiness  : Readiness probe confirming graph and index availability
  - GET  /metrics    : Prometheus and JSON operational metrics telemetry
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

# Ensure project root is on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import config
from kg.backend import describe_backend, open_graph
from pipelines import build_pipelines
from retrieval import load_index
from utils.llm import LLMHelper, build_llm
from utils.observability import TraceContext, logger


class APIService:
    """Manages loaded pipelines, graph backend, and telemetry."""

    _instance: Optional["APIService"] = None

    def __init__(self, corpus_path: Optional[str] = None,
                 force_local: bool = False, no_llm: bool = False) -> None:
        self.corpus_path = corpus_path or config.benchmark.corpus_path
        self.force_local = force_local
        self.no_llm = no_llm
        self.t0 = time.time()
        self.total_queries = 0
        self.query_latencies: list[float] = []
        self.pipeline_counts: Dict[str, int] = {
            "Router": 0, "Agentic GraphRAG": 0, "GraphRAG": 0, "RAG": 0
        }

        print("[api] initializing KnowledgeGraph and Index ...", flush=True)
        self.kg = open_graph(self.corpus_path, config,
                            cache_path=config.benchmark.kg_cache_path,
                            force_local=self.force_local)
        self.index = load_index(self.corpus_path, self.kg,
                               vector_backend=config.benchmark.vector_backend)
        if self.no_llm:
            self.llm = LLMHelper()
        else:
            self.llm = build_llm(config)

        built = build_pipelines(self.index, self.llm, config, with_router=True)
        self.pipelines: Dict[str, Any] = {p.name: p for p in built}
        print("[api] ready to serve requests.", flush=True)

    @classmethod
    def get_service(cls, force_local: bool = False, no_llm: bool = False) -> "APIService":
        if cls._instance is None:
            cls._instance = APIService(force_local=force_local, no_llm=no_llm)
        return cls._instance

    def query(self, question: str, pipeline_name: str = "Router", qid: str = "") -> Dict[str, Any]:
        pipe = self.pipelines.get(pipeline_name) or self.pipelines["Router"]
        trace = TraceContext.create(qid=qid)
        t_start = time.perf_counter()

        logger.info(f"Received query for {pipe.name}", trace_id=trace.trace_id, question=question[:100])
        with trace.start_span("pipeline_execution", agent=pipe.name):
            result = pipe.run(question, qid=qid)

        dur_ms = (time.perf_counter() - t_start) * 1000.0
        self.total_queries += 1
        self.query_latencies.append(dur_ms)
        self.pipeline_counts[pipe.name] = self.pipeline_counts.get(pipe.name, 0) + 1

        res_dict = result.to_dict()
        res_dict["trace_id"] = trace.trace_id
        res_dict["trace_context"] = trace.to_dict()
        return res_dict

    def metrics(self) -> Dict[str, Any]:
        uptime_s = time.time() - self.t0
        avg_lat = sum(self.query_latencies) / len(self.query_latencies) if self.query_latencies else 0.0
        return {
            "uptime_seconds": round(uptime_s, 1),
            "total_queries": self.total_queries,
            "average_latency_ms": round(avg_lat, 2),
            "pipeline_counts": dict(self.pipeline_counts),
            "backend": describe_backend(),
            "num_docs": self.index.num_docs if self.index else 0,
            "num_chunks": self.index.num_chunks if self.index else 0,
        }


class APIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for REST API."""

    service: Optional[APIService] = None

    def _send_json(self, status: int, data: Dict[str, Any]) -> None:
        blob = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        svc = APIHandler.service or APIService.get_service()

        if path in ("", "/health"):
            self._send_json(200, {
                "status": "healthy",
                "service": "tigergraph-agentic-graphrag-api",
                "version": "2.0.0",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
        elif path == "/readiness":
            backend = describe_backend()
            ready = bool(svc.kg and svc.index)
            self._send_json(200 if ready else 503, {
                "status": "ready" if ready else "not_ready",
                "graph_backend": backend.get("active", "local"),
                "graph_name": backend.get("graphname", ""),
                "num_docs": svc.index.num_docs if svc.index else 0,
                "num_chunks": svc.index.num_chunks if svc.index else 0,
            })
        elif path == "/metrics":
            self._send_json(200, svc.metrics())
        else:
            self._send_json(404, {"error": "not_found", "path": self.path})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        if path != "/query":
            self._send_json(404, {"error": "endpoint_not_found", "path": self.path})
            return

        content_len = int(self.headers.get("Content-Length", 0))
        if not content_len:
            self._send_json(400, {"error": "empty_request_body"})
            return

        try:
            body = json.loads(self.rfile.read(content_len).decode("utf-8"))
        except Exception as exc:
            self._send_json(400, {"error": f"malformed_json: {exc}"})
            return

        question = body.get("question")
        if not question or not str(question).strip():
            self._send_json(400, {"error": "missing_required_field: question"})
            return

        pipeline_name = body.get("pipeline", "Router")
        qid = body.get("qid", "")

        svc = APIHandler.service or APIService.get_service()
        try:
            result = svc.query(question=str(question).strip(),
                               pipeline_name=str(pipeline_name).strip(),
                               qid=str(qid).strip())
            self._send_json(200, result)
        except Exception as exc:
            logger.error(f"Query processing failed: {exc}", exc_info=True)
            self._send_json(500, {"error": "internal_query_error", "message": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy default http.server logging in favour of structured logs
        pass


def run_self_test(host: str = "127.0.0.1", port: int = 8089) -> bool:
    """Run an automated self-test of the production API layer."""
    import urllib.request

    print("=== Testing Production API Layer ===")
    svc = APIService.get_service(force_local=True, no_llm=True)
    APIHandler.service = svc

    server = HTTPServer((host, port), APIHandler)
    import threading
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)

    base = f"http://{host}:{port}"
    passed = 0
    try:
        # 1. Test /health
        with urllib.request.urlopen(f"{base}/health") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 200 and data.get("status") == "healthy"
            print("  PASS GET /health -> healthy")
            passed += 1

        # 2. Test /readiness
        with urllib.request.urlopen(f"{base}/readiness") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 200 and data.get("status") == "ready"
            print("  PASS GET /readiness -> ready")
            passed += 1

        # 3. Test /metrics
        with urllib.request.urlopen(f"{base}/metrics") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 200 and "total_queries" in data
            print("  PASS GET /metrics -> metrics rollup available")
            passed += 1

        # 4. Test POST /query
        req_data = json.dumps({
            "question": "How many events in 1996 had more than 100 competitors?",
            "pipeline": "Router",
            "qid": "test-api-001"
        }).encode("utf-8")
        req = urllib.request.Request(f"{base}/query", data=req_data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 200
            assert "answer" in data and "trace_id" in data
            print(f"  PASS POST /query -> answer='{data.get('answer')}' (pipeline={data.get('pipeline')})")
            passed += 1

    finally:
        server.shutdown()
        server.server_close()

    print(f"Production API Test Results: {passed}/4 Passed.")
    return passed == 4


def main() -> None:
    parser = argparse.ArgumentParser(description="Production API Server for Agentic GraphRAG")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Binding port (default: 8000)")
    parser.add_argument("--no-tg", action="store_true", help="Force local graph backend")
    parser.add_argument("--no-llm", action="store_true", help="Deterministic mode without LLM calls")
    parser.add_argument("--test", action="store_true", help="Run automated self-test and exit")
    args = parser.parse_args()

    if args.test:
        ok = run_self_test()
        sys.exit(0 if ok else 1)

    svc = APIService.get_service(force_local=args.no_tg, no_llm=args.no_llm)
    APIHandler.service = svc

    server = HTTPServer((args.host, args.port), APIHandler)
    print(f"[api] Serving HTTP on http://{args.host}:{args.port} ...", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[api] Server stopped by user.", flush=True)
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
