"""Deterministic tests for the agent's tool layer (no LLM, no quota).

Every tool the model can call is exercised against the real knowledge graph and
index, so a schema/handler mismatch (a tool the model can call but that does not
exist, or one that crashes on valid arguments) fails here instead of silently
degrading a benchmark run.

    python tools/test_tools.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from config import config  # noqa: E402
from kg.builder import load_or_build  # noqa: E402
from retrieval import load_index  # noqa: E402
from agents.tools import (AGG_FIELDS, EDGE_TYPES, TOOL_NAMES,  # noqa: E402
                          TOOL_SCHEMAS, GraphTools)

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" - {detail}" if detail else ""))


kg = load_or_build(config.benchmark.corpus_path)
index = load_index(config.benchmark.corpus_path, kg,
                   vector_backend=config.benchmark.vector_backend)
tools = GraphTools(kg, index)

# ── 1. schema/handler agreement ────────────────────────────────────────
names = [t["function"]["name"] for t in TOOL_SCHEMAS]
check("every schema names a callable GraphTools method",
      all(callable(getattr(tools, n, None)) for n in names),
      f"schemas={names}")
check("TOOL_NAMES matches the schemas",
      TOOL_NAMES == names, f"TOOL_NAMES={TOOL_NAMES}")
check("submit_answer is the terminal tool", "submit_answer" in names)
check("no tool computes an aggregation for the model",
      not any(k in " ".join(names) for k in ("count", "max", "min", "sum")),
      "the model must reason over get_event_values itself")

# ── 2. real calls with valid arguments ─────────────────────────────────
found = tools.search_events(sport="Biathlon", season="Winter", year_from=2018,
                            year_to=2018, limit=5)
check("search_events returns rows and doc_ids",
      found.get("count", 0) > 0 and bool(found.get("doc_ids")),
      f"count={found.get('count')} docs={len(found.get('doc_ids', []))}")
doc_id = found["doc_ids"][0]

vals = tools.get_event_values("competitors", sport="Biathlon", season="Winter",
                              year_from=2018, year_to=2018)
check("get_event_values returns raw values, not a computed answer",
      vals.get("num_with_value", 0) > 0 and "answer" not in vals,
      f"values={vals.get('num_with_value')}/{vals.get('num_matching_events')}")

detail = tools.get_event_details(doc_id)
check("get_event_details resolves a graph vertex",
      detail.get("in_graph") is True and detail.get("doc_id") == doc_id,
      f"{detail.get('title', '')[:50]}")

passages = tools.search_passages("men's 20 kilometres walk gold medal", top_k=3)
check("search_passages returns citable passages",
      passages.get("returned", 0) > 0 and bool(passages.get("doc_ids")),
      f"returned={passages.get('returned')}")

hops_out = tools.traverse_graph(doc_id, "IN_SPORT", "out")
check("traverse_graph follows a directed edge",
      hops_out.get("num_found", 0) > 0,
      f"IN_SPORT -> {hops_out.get('num_found')} hop(s)")

prev = tools.traverse_graph(doc_id, "PREV", "both")
check("traverse_graph PREV supports the temporal pattern",
      prev.get("num_found", 0) > 0, f"PREV -> {prev.get('num_found')} hop(s)")

submit = tools.submit_answer(answer="5", reasoning="counted", doc_ids=[doc_id])
check("submit_answer accepts the final answer",
      submit.get("accepted") is True and submit.get("answer") == "5")

# ── 3. error paths must be recoverable, not fatal ──────────────────────
check("unknown field is rejected with a hint",
      "error" in tools.get_event_values("not_a_field"))
check("unknown edge type is rejected with a hint",
      "error" in tools.traverse_graph(doc_id, "NOT_AN_EDGE"))
check("missing required argument is rejected with a hint",
      "error" in tools.get_event_details(""))            # empty doc_id
bad = tools.execute("no_such_tool", {})
check("unknown tool is rejected by the dispatcher", "error" in bad)
check("errors never raise out of execute",
      all(isinstance(tools.execute(n, {"doc_id": "nope", "edge_type": "PREV",
                                       "field": "competitors",
                                       "query": "zzz"}).get("error", ""), str)
          for n in ("get_event_details", "traverse_graph")))

# ── 4. every call is recorded for the trace ────────────────────────────
recorded = [c["tool"] for c in tools.calls]
check("execute() records every call for the trace",
      len(recorded) >= 3 and all(c["latency_ms"] >= 0 for c in tools.calls),
      f"{len(recorded)} execute()-routed calls recorded")
check("records carry a human-readable summary",
      all(c["summary"] for c in tools.calls),
      tools.calls[0]["summary"][:60])

# ── 5. all declared fields/edge types are actually usable ──────────────
bad_fields = [f for f in AGG_FIELDS
              if "error" in tools.get_event_values(f, sport="Athletics",
                                                   year_from=2008, year_to=2008)]
check("every AGG_FIELD enum value is accepted", not bad_fields,
      f"rejected={bad_fields}")
bad_edges = [e for e in EDGE_TYPES if "error" in tools.traverse_graph(doc_id, e)]
check("every EDGE_TYPES enum value is accepted", not bad_edges,
      f"rejected={bad_edges}")

print(f"\nTOOLS TESTED: {len(names)}  PASSED: {len(PASS)}  FAILURES: {len(FAIL)}")
if FAIL:
    print("failed:", json.dumps(FAIL, indent=2))
    sys.exit(1)