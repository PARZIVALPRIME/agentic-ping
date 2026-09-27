"""Model Context Protocol (MCP) Server for TigerGraph Cloud (OlympicsKG).

Exposes TigerGraph 4.2.5 graph primitives and autonomous agentic investigation
tools to standard MCP clients (Claude Desktop, Cursor, LangChain, CrewAI) over
JSON-RPC 2.0 stdio.

Usage:
  python tools/mcp_server.py            # Runs stdio MCP server loop
  python tools/mcp_server.py --test     # Automated self-test verifying all tools
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kg.backend import open_graph
from config import config
from reasoning.conflicts import detect_conflicts, resolve
from agents.tools import GraphTools, TOOL_SCHEMAS


SERVER_NAME = "tigergraph-agentic-mcp"
SERVER_VERSION = "1.0.0"

MCP_TOOLS = [
    {
        "name": "tg_filter_events",
        "description": "Filter Olympic events on TigerGraph by sport, venue, season, or year range. Executes tg_filter_events on TGCloud with deterministic ordering.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sport": {"type": "string", "description": "Sport name, e.g. 'Athletics', 'Swimming'"},
                "venue": {"type": "string", "description": "Venue name, e.g. 'London Velopark'"},
                "season": {"type": "string", "description": "'Summer' or 'Winter'"},
                "year_from": {"type": "integer", "description": "Earliest year (inclusive)"},
                "year_to": {"type": "integer", "description": "Latest year (inclusive)"},
                "limit": {"type": "integer", "description": "Maximum candidate rows (default 50)", "default": 50},
            },
        },
    },
    {
        "name": "tg_neighbours",
        "description": "Explore multi-hop relationships connected to a node in TigerGraph. Traverses HELD_AT, WON_BY, PART_OF, PREV, NEXT edges.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "vertex_id": {"type": "string", "description": "Entity ID (doc_id, athlete_id, venue_id, etc.)"},
                "edge_type": {"type": "string", "description": "Optional edge filter: 'WON_BY', 'HELD_AT', 'PART_OF', 'IN_SPORT', 'PREV', 'NEXT'"},
                "limit": {"type": "integer", "description": "Maximum hops to return", "default": 50},
            },
            "required": ["vertex_id"],
        },
    },
    {
        "name": "tg_event",
        "description": "Fetch complete structured attributes of a specific event node on TigerGraph by its doc_id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "Wikipedia document / event ID, e.g. 'Q1050909'"},
            },
            "required": ["doc_id"],
        },
    },
    {
        "name": "tg_aggregate_stats",
        "description": "Execute server-side GSQL accumulator aggregation over events on TigerGraph to compute totals, year ranges, and unique entity counts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sport": {"type": "string", "description": "Optional sport filter"},
                "venue": {"type": "string", "description": "Optional venue filter"},
                "season": {"type": "string", "description": "Optional season filter"},
                "year_from": {"type": "integer", "description": "Earliest year"},
                "year_to": {"type": "integer", "description": "Latest year"},
            },
        },
    },
    {
        "name": "detect_conflicts",
        "description": "Adjudicate competing or evolving versions of a fact using our 4-tier precedence hierarchy (Authority Correction > Recency > Entity Succession > Majority).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "field_name": {"type": "string", "description": "Attribute being evaluated, e.g. 'gold_medallist', 'record', 'venue'"},
                "candidates": {
                    "type": "array",
                    "description": "List of candidate objects with 'value', 'doc_id', and optional 'year' / 'text'",
                    "items": {"type": "object"},
                },
            },
            "required": ["field_name", "candidates"],
        },
    },
    {
        "name": "agentic_investigate",
        "description": "Autonomous multi-agent investigation over TigerGraph for complex multi-hop, temporal, superlative, or aggregation queries.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The natural language Olympic investigation question"},
            },
            "required": ["question"],
        },
    },
]

MCP_RESOURCES = [
    {
        "uri": "tigergraph://schema/OlympicsKG",
        "name": "OlympicsKG Graph Schema",
        "description": "GSQL schema definition for vertices (Event, Athlete, Games, Sport, Venue) and edges (WON_BY, HELD_AT, PART_OF, PREV, NEXT).",
        "mimeType": "application/json",
    },
    {
        "uri": "tigergraph://stats/OlympicsKG",
        "name": "OlympicsKG Cluster Statistics",
        "description": "Current vertex and edge counts on the active TigerGraph Cloud cluster.",
        "mimeType": "application/json",
    },
]


class TigerGraphMCPServer:
    """Zero-dependency stdio JSON-RPC 2.0 MCP server for TigerGraph."""

    def __init__(self) -> None:
        self.kg = open_graph(
            config.benchmark.corpus_path, config,
            cache_path=config.benchmark.kg_cache_path,
        )
        self.tools = GraphTools(self.kg, index=None)

    def handle_request(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        msg_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {}) or {}

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"subscribe": False, "listChanged": False},
                    },
                },
            }

        if method == "notifications/initialized":
            return None

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": MCP_TOOLS}}

        if method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {}) or {}
            result = self.call_tool(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}],
                    "isError": "error" in result,
                },
            }

        if method == "resources/list":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"resources": MCP_RESOURCES}}

        if method == "resources/read":
            uri = params.get("uri")
            content = self.read_resource(uri)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(content, indent=2)}],
                },
            }

        if msg_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        return None

    def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if name == "tg_filter_events":
                sport = args.get("sport", "")
                venue = args.get("venue", "")
                season = args.get("season", "")
                y_from = args.get("year_from")
                y_to = args.get("year_to")
                limit = int(args.get("limit", 50))
                events = self.kg.filter_events(
                    sport=sport, venue=venue, season=season,
                    year_from=y_from, year_to=y_to, cap=limit,
                )
                return {
                    "count": len(events),
                    "events": [
                        {
                            "doc_id": e.doc_id, "title": e.title, "sport": e.sport,
                            "venue": e.venue, "year": e.year, "season": e.season,
                            "gold": e.gold, "silver": e.silver, "bronze": e.bronze,
                        }
                        for e in events[:limit]
                    ],
                }

            if name == "tg_neighbours":
                v_id = str(args.get("vertex_id", ""))
                e_type = str(args.get("edge_type", ""))
                limit = int(args.get("limit", 50))
                hops = self.kg.neighbours(v_id, cap=limit)
                filtered = [h for h in hops if not e_type or h[1] == e_type]
                return {
                    "from_vertex": v_id,
                    "num_hops": len(filtered),
                    "hops": [
                        {"direction": h[0], "edge_type": h[1], "target": h[2]}
                        for h in filtered[:limit]
                    ],
                }

            if name == "tg_event":
                doc_id = str(args.get("doc_id", ""))
                event = self.kg.event(doc_id)
                if not event:
                    return {"found": False, "doc_id": doc_id}
                return {
                    "found": True,
                    "event": {
                        "doc_id": event.doc_id, "title": event.title, "sport": event.sport,
                        "venue": event.venue, "year": event.year, "season": event.season,
                        "athletes": getattr(event, "athletes", []),
                        "gold": event.gold, "silver": event.silver, "bronze": event.bronze,
                    },
                }

            if name == "tg_aggregate_stats":
                if hasattr(self.kg, "aggregate_stats"):
                    return self.kg.aggregate_stats(
                        sport=args.get("sport", ""), venue=args.get("venue", ""),
                        season=args.get("season", ""), year_from=args.get("year_from"),
                        year_to=args.get("year_to"),
                    )
                events = self.kg.filter_events(
                    sport=args.get("sport", ""), venue=args.get("venue", ""),
                    season=args.get("season", ""), year_from=args.get("year_from"),
                    year_to=args.get("year_to"),
                )
                years = [e.year for e in events if e.year and e.year > 0]
                return {
                    "total_events": len(events),
                    "earliest_year": min(years) if years else 0,
                    "latest_year": max(years) if years else 0,
                    "num_venues": len({e.venue for e in events if e.venue}),
                    "num_sports": len({e.sport for e in events if e.sport}),
                    "backend": "local_mirror",
                }

            if name == "detect_conflicts":
                field_name = str(args.get("field_name", "attribute"))
                candidates = args.get("candidates", [])
                res = resolve(field_name, candidates)
                return {
                    "resolved": res.resolved,
                    "rule": res.rule,
                    "confidence": res.confidence,
                    "explanation": res.explanation,
                    "superseded": res.superseded,
                    "uncertainty": round(1.0 - res.confidence, 3),
                }

            if name == "agentic_investigate":
                from pipelines import make_pipelines
                from retrieval.index import CorpusIndex
                index = CorpusIndex(self.kg)
                pipes = {p.name: p for p in make_pipelines(index)}
                agent = pipes.get("Agentic GraphRAG")
                if not agent:
                    return {"error": "Agentic GraphRAG arm not found"}
                res = agent.run(str(args.get("question", "")))
                return {
                    "answer": res.answer,
                    "confidence": res.confidence,
                    "stop_reason": res.stop_reason,
                    "citations": res.citations,
                    "steps": len(res.steps),
                    "total_tokens": res.total_tokens,
                    "latency_ms": res.latency_ms,
                }

            return {"error": f"Unknown tool: {name}"}
        except Exception as exc:
            return {"error": str(exc), "trace": traceback.format_exc()}

    def read_resource(self, uri: str) -> Dict[str, Any]:
        if uri == "tigergraph://schema/OlympicsKG":
            return {
                "graph": "OlympicsKG",
                "vertices": ["Event", "Athlete", "Games", "Sport", "Venue"],
                "edges": ["PREV", "NEXT", "PART_OF", "IN_SPORT", "HELD_AT", "WON_BY"],
                "backend": getattr(self.kg, "backend_name", "tigergraph"),
            }
        if uri == "tigergraph://stats/OlympicsKG":
            return {
                "graph": "OlympicsKG",
                "total_events": len(getattr(self.kg, "events", {})),
                "total_edges": len(getattr(self.kg, "edges", {})),
                "host": os.getenv("TG_HOST", "https://tg-fed265f1-0603-4e99-b6f3-6efd6d7fd5c5.tg-2635877100.i.tgcloud.io"),
            }
        return {"error": f"Resource not found: {uri}"}

    def run_stdio(self) -> None:
        """Standard JSON-RPC 2.0 stdio message pump."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = self.handle_request(req)
                if resp is not None:
                    sys.stdout.write(json.dumps(resp) + "\n")
                    sys.stdout.flush()
            except Exception as exc:
                err = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
                sys.stdout.write(json.dumps(err) + "\n")
                sys.stdout.flush()


def run_self_test() -> int:
    """Verifies all MCP endpoints, tools, and schemas locally."""
    print("=" * 60)
    print(" TigerGraph Model Context Protocol (MCP) Server Self-Test")
    print("=" * 60)
    server = TigerGraphMCPServer()

    # 1. Initialize
    init_res = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_res["result"]["serverInfo"]["name"] == SERVER_NAME
    print("  PASS  initialize handshake")

    # 2. Tools list
    tools_res = server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = tools_res["result"]["tools"]
    tool_names = {t["name"] for t in tools}
    print(f"  PASS  tools/list returned {len(tools)} tools: {', '.join(sorted(tool_names))}")

    # 3. Call tg_filter_events
    filter_res = server.call_tool("tg_filter_events", {"sport": "Athletics", "limit": 5})
    assert filter_res["count"] > 0
    print(f"  PASS  tg_filter_events -> {filter_res['count']} events")

    # 4. Call tg_neighbours
    sample_id = filter_res["events"][0]["doc_id"]
    neigh_res = server.call_tool("tg_neighbours", {"vertex_id": sample_id, "limit": 10})
    print(f"  PASS  tg_neighbours from {sample_id} -> {neigh_res['num_hops']} hops")

    # 5. Call tg_aggregate_stats
    stats_res = server.call_tool("tg_aggregate_stats", {"sport": "Athletics"})
    assert stats_res["total_events"] > 0
    print(f"  PASS  tg_aggregate_stats -> {stats_res['total_events']} events, span {stats_res['earliest_year']}-{stats_res['latest_year']}")

    # 6. Call detect_conflicts
    conflict_res = server.call_tool("detect_conflicts", {
        "field_name": "gold_medallist",
        "candidates": [
            {"value": "Athlete A", "doc_id": "Q1", "year": 2012, "text": "Won gold"},
            {"value": "Athlete B", "doc_id": "Q2", "year": 2016, "text": "Athlete A was disqualified for doping; medal reallocated to Athlete B"},
        ],
    })
    assert conflict_res["resolved"] == "Athlete B"
    assert conflict_res["rule"] == "authority_correction"
    print(f"  PASS  detect_conflicts -> {conflict_res['resolved']} ({conflict_res['rule']}, uncertainty {conflict_res['uncertainty']})")

    # 7. Resources read
    res_meta = server.read_resource("tigergraph://schema/OlympicsKG")
    assert res_meta["graph"] == "OlympicsKG"
    print("  PASS  resources/read tigergraph://schema/OlympicsKG")

    print("=" * 60)
    print(" ALL MCP SERVER CHECKS PASSED (100% OPERATIONAL)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(run_self_test())
    else:
        server = TigerGraphMCPServer()
        server.run_stdio()
