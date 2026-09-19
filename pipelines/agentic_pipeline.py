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

    def __init__(self, index, llm=None, cfg: Dict[str, Any] = None) -> None:
        from agents.orchestrator import OrchestratorAgent

        self.index = index
        self.kg = index.kg if index is not None else None
        self.llm = llm
        self.engine = OrchestratorAgent(self.kg, index, llm=llm, cfg=cfg)

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

        context_text = "\n".join(c.get("text", "") for c in state.chunks.values())
        result.context_tokens = count_tokens(context_text)
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
        result.metadata = {
            "pipeline": self.name,
            "agent_mode": self.engine.cfg.get("agent_mode"),
            "classification": state.classification,
            "slot_report": state.slot_report,
            "spec": state.spec.to_dict(),
            "strategy_changes": state.strategy_changes,
            "resolved_gaps": state.resolved_gaps,
            "gaps_remaining": state.missing_info,
            "synth_source": state.synth_source,
            "rationale": state.rationale,
            "adjudication": state.adjudication,
            "tool_calls": list(state.tool_calls),
            "docs_touched": len(state.documents),
            "plan_skips": [{"agent": s.agent, "operation": s.operation,
                            "skip_reason": s.skip_reason}
                           for s in state.plan if s.skipped],
        }
        result.latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return result