"""Pipeline 3 - Agentic GraphRAG.

The differentiator. Where RAG does one retrieval and GraphRAG does one retrieval
plus one expansion, this pipeline *investigates*: it classifies the question,
plans a typed sequence of agent operations, runs them against the knowledge
graph and the passage index, audits the evidence, splices in recovery steps
when gaps remain, and finally synthesises an answer that an LLM adjudicates
against the retrieved passages.

It therefore reaches documents that no top-k retriever can reach - the complete
event set for counting questions, the immediately-previous edition for temporal
questions - and it reports *why* it stopped (confidence, budget, staleness).
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from utils.metrics import count_tokens

from .base import PipelineResult

RETRIEVAL_OPS = {
    "link_entities", "traverse_graph", "vector_search", "resolve_article",
    "resolve_venue_date", "resolve_anchor", "resolve_event",
    "edition_chain_fallback",
}
EVIDENCE_OPS = {
    "count_threshold", "arg_extreme", "resolve_article", "resolve_venue_date",
    "resolve_event", "edition_chain_fallback", "render_answer",
}
LLM_PREFIXES = ("agent.", "planner.", "classifier.", "llm.")


class AgenticPipeline:
    """Adaptive, tool-using GraphRAG (the agentic pipeline)."""

    name = "Agentic GraphRAG"

    def __init__(self, index, llm=None, cfg: Dict[str, Any] = None,
                 expect_agentic: bool = None) -> None:
        from agents.orchestrator import OrchestratorAgent

        self.index = index
        self.kg = index.kg if index is not None else None
        self.llm = llm
        self.engine = OrchestratorAgent(self.kg, index, llm=llm, cfg=cfg)

        # Fail fast if the run intends to be agentic but cannot be. By default
        # we infer that intent from "an LLM was supplied and is available": a
        # deterministic (--no-llm) run passes llm=None and is never blocked,
        # while an LLM run that silently can't reach ReAct is stopped at
        # startup instead of quietly scoring the deterministic pipeline.
        if expect_agentic is None:
            expect_agentic = bool(llm and getattr(llm, "available", False))
        if expect_agentic:
            self.engine.preflight_or_raise()

    def run(self, question: str, qid: str = "") -> PipelineResult:
        started = time.perf_counter()
        state = self.engine.run(question, qid)

        result = PipelineResult(pipeline=self.name, question=question,
                                qtype=state.qtype)
        result.answer = state.answer
        result.citations = list(state.citations)
        result.confidence = state.confidence
        result.stop_reason = state.stop_reason
        result.method = state.synth_source or "structured_agentic"

        trace_ops = [s.operation for s in state.trace]
        tool_names = [op.split("tool:", 1)[1] for op in trace_ops
                      if op.startswith("tool:")]
        result.retrieval_steps = sum(1 for op in trace_ops
                                     if op in RETRIEVAL_OPS
                                     or (op.startswith("tool:")
                                         and op != "tool:submit_answer"))
        reasoning_ops = {
            "get_event_values", "detect_conflicts", "submit_answer",
            "count_threshold", "verify_by_recount", "arg_extreme",
            "verify_extreme", "evaluate_evidence", "render_answer",
            "final_answer", "retry",
        }
        result.reasoning_steps = sum(1 for op in trace_ops
                                     if op.replace("tool:", "") in reasoning_ops)
        result.tools_called = list(dict.fromkeys(
            tool_names if tool_names else [op for op in trace_ops]))
        result.agents_invoked = list(dict.fromkeys(s.agent for s in state.trace))
        result.chunks_retrieved = len(state.chunks)
        result.docs_retrieved = len(state.documents)
        result.plan = [f"{s.agent}.{s.operation}" for s in state.plan]
        result.loop_iterations = state.iterations
        result.strategy_changed = bool(state.strategy_changes)
        result.candidates_considered = len(state.candidates)

        result.steps = []
        for step in state.trace:
            entry = step.to_dict()
            entry["new_documents"] = step.new_documents
            result.steps.append(entry)
        result.time_per_operation = [
            {"operation": s.operation, "agent": s.agent, "detail": s.detail[:160],
             "latency_ms": round(s.latency_ms, 2)}
            for s in state.trace
        ]
        result.tokens_per_operation = state.tokens.per_operation
        result.input_tokens = state.tokens.input_tokens
        result.output_tokens = state.tokens.output_tokens
        result.total_tokens = state.tokens.total_tokens
        result.llm_calls = sum(1 for op in state.tokens.per_operation
                               if str(op.get("operation", "")).startswith(LLM_PREFIXES))

        observed_payloads = [
            json.dumps(s.observation.get("result", s.observation.get("summary", "")),
                       ensure_ascii=False, default=str)
            for s in state.trace if s.observation
        ]
        chunk_texts = [
            c.get("text") or json.dumps(c, ensure_ascii=False, default=str)
            for c in state.chunks.values() if c
        ]
        context_text = "\n".join(observed_payloads + chunk_texts)
        result.context_tokens = count_tokens(context_text)

        retrieval_method_map = {
            "search_passages": "hybrid",
            "vector_search": "vector",
            "search_events": "graph",
            "traverse_graph": "graph",
            "get_event_details": "graph",
            "resolve_article": "graph",
            "get_event_values": "accumulator",
            "aggregate_stats": "accumulator",
            "count_threshold": "accumulator",
        }
        selected_methods = set()
        for op in trace_ops:
            clean = op.replace("tool:", "")
            if clean in retrieval_method_map:
                selected_methods.add(retrieval_method_map[clean])

        accepted_why = ""
        if state.trace and isinstance(state.trace[-1].observation, dict):
            accepted_why = state.trace[-1].observation.get("answer_accepted_because", "")

        tool_steps = [s for s in state.trace if s.operation.startswith("tool:")]
        if tool_steps:
            result.evidence = [
                {"agent": s.agent, "operation": s.operation,
                 "observation": s.observation} for s in tool_steps]
        else:
            result.evidence = [
                {"agent": s.agent, "operation": s.operation,
                 "observation": s.observation}
                for s in state.trace if s.operation in EVIDENCE_OPS]
        result.unresolved = list(state.missing_info)
        result.uncertainty = round(max(0.0, min(1.0, 1.0 - float(state.confidence or 0.0))), 3)
        result.metadata = {
            "pipeline": self.name,
            "agent_mode": self.engine.cfg.get("agent_mode"),
            "classification": state.classification,
            "slot_report": state.slot_report,
            "parse_report": state.parse_report,
            "spec": state.spec.to_dict(),
            "strategy_changes": state.strategy_changes,
            "resolved_gaps": state.resolved_gaps,
            "gaps_remaining": state.missing_info,
            "synth_source": state.synth_source,
            "rationale": state.rationale,
            "adjudication": state.adjudication,
            "conflicts": state.fact_conflicts,
            "uncertainty": result.uncertainty,
            "tool_calls": list(state.tool_calls),
            "docs_touched": len(state.documents),
            "retrieval_methods_selected": sorted(selected_methods) if selected_methods else ["graph"],
            "persona_handoffs": [s.agent for s in state.trace if s.agent],
            "stopping_decision": {
                "when_step": state.iterations,
                "when_tokens": result.total_tokens,
                "stop_reason": state.stop_reason,
                "why": (
                    f"Answer verified via {accepted_why or 'evidence grounding'} with confidence {state.confidence:.2f}"
                    if state.stop_reason == "submitted_answer" else
                    f"Investigation terminated: {state.stop_reason}"
                ),
                "stopping_condition_met": True,
            },
            "plan_skips": [{"agent": s.agent, "operation": s.operation,
                            "skip_reason": s.skip_reason}
                           for s in state.plan if s.skipped],
        }
        result.latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        result.metadata["stopping_decision"]["when_latency_ms"] = result.latency_ms
        return result