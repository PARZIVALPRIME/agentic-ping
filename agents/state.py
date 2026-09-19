"""Agent state: the shared blackboard the specialised agents read and write.

The state is the *harness* referenced in the plan. It records the plan, the
evidence accumulated so far, the full execution trace, confidence and the
information gaps that remain. Agents never talk to each other directly; they
read and extend this object, which makes every run inspectable after the fact
(the dashboard replays these traces).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from reasoning.query_parser import QuerySpec
from utils.metrics import TokenCounter


@dataclass
class PlannedStep:
    """A step in the orchestrator's investigation plan."""

    agent: str
    operation: str
    reason: str = ""
    optional: bool = False
    done: bool = False
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"agent": self.agent, "operation": self.operation, "reason": self.reason,
                "optional": self.optional, "done": self.done, "skipped": self.skipped,
                "skip_reason": self.skip_reason}


@dataclass
class ExecutedStep:
    """A completed unit of work appended to the trace."""

    index: int
    agent: str
    operation: str
    detail: str = ""
    observation: Dict[str, Any] = field(default_factory=dict)
    confidence_after: float = 0.0
    new_documents: int = 0
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index, "agent": self.agent, "operation": self.operation,
            "detail": self.detail, "observation": self.observation,
            "confidence_after": self.confidence_after,
            "new_documents": self.new_documents,
            "latency_ms": round(self.latency_ms, 2),
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
        }


class AgentState:
    """Blackboard shared by all agents for one question."""

    def __init__(self, question: str, spec: QuerySpec, qid: str = "") -> None:
        self.question = question
        self.qid = qid
        self.spec = spec
        self.qtype = spec.qtype
        self.kind = (spec.qtype if spec.qtype in
                     ("lookup", "multi_hop", "temporal", "aggregation", "superlative")
                     else "lookup")
        self.classification: Dict[str, Any] = {}
        self.slot_report: Dict[str, Any] = {}
        self.synth_source: str = ""
        self.adjudication: Dict[str, Any] = {}
        self.rationale: str = ""

        self.plan: List[PlannedStep] = []
        self.plan_cursor = 0
        self.trace: List[ExecutedStep] = []
        self.tokens = TokenCounter()

        # accumulated evidence
        self.documents: Dict[str, Any] = {}          # doc_id -> EventNode touched
        self.chunks: Dict[str, Dict[str, Any]] = {}  # doc_id -> retrieved passage
        self.slots: Dict[str, Any] = {}              # resolved structured facts
        self.candidates: List[str] = []              # doc_ids under consideration

        self.answer: str = ""
        self.citations: List[str] = []
        self.confidence: float = 0.0
        self.missing_info: List[str] = []
        self.resolved_gaps: List[str] = []
        self.strategy_changes: List[str] = []
        self.proposed_gaps: List[str] = []   # gaps already targeted by a recovery step
        self.stop_reason: str = ""
        self.iterations = 0
        self.tool_calls: List[Dict[str, Any]] = []  # records from the ReAct loop
        self._t0 = time.perf_counter()

    # ── plan management ────────────────────────────────────────────────
    def set_plan(self, steps: List[PlannedStep], reason: str = "") -> None:
        self.plan = steps
        self.plan_cursor = 0
        if reason:
            self.strategy_changes.append(f"plan: {reason}")

    def next_step(self) -> Optional[PlannedStep]:
        while self.plan_cursor < len(self.plan):
            step = self.plan[self.plan_cursor]
            self.plan_cursor += 1
            if not step.skipped:
                return step
        return None

    def append_step(self, step: PlannedStep, reason: str = "") -> None:
        self.plan.append(step)
        self.strategy_changes.append(reason or f"appended {step.agent}.{step.operation}")

    # ── evidence management ────────────────────────────────────────────
    def add_documents(self, nodes: Any) -> int:
        """Register graph vertices; returns how many were new."""
        if nodes is None:
            nodes = []
        if not isinstance(nodes, (list, tuple)):
            nodes = [nodes]
        new = 0
        for node in nodes:
            doc_id = getattr(node, "doc_id", None)
            if doc_id is None and isinstance(node, dict):
                doc_id = node.get("doc_id")
            if doc_id is None and isinstance(node, str):
                doc_id = node          # the tool layer reports bare doc_ids
            if not doc_id:
                continue
            if doc_id not in self.documents:
                new += 1
            self.documents[doc_id] = node
            if doc_id not in self.candidates:
                self.candidates.append(doc_id)
        return new

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        new = 0
        for chunk in chunks:
            doc_id = chunk.get("doc_id")
            if not doc_id:
                continue
            if doc_id not in self.chunks:
                new += 1
            self.chunks[doc_id] = chunk
            if doc_id not in self.candidates:
                self.candidates.append(doc_id)
        return new

    def note_gap(self, gap: str) -> None:
        if gap and gap not in self.missing_info:
            self.missing_info.append(gap)

    def resolve_gap(self, gap: str) -> None:
        if gap in self.missing_info:
            self.missing_info.remove(gap)
            self.resolved_gaps.append(gap)

    # ── trace ──────────────────────────────────────────────────────────
    def record(self, agent: str, operation: str, detail: str = "",
               observation: Optional[Dict[str, Any]] = None,
               new_documents: int = 0, latency_ms: float = 0.0,
               input_tokens: int = 0, output_tokens: int = 0) -> ExecutedStep:
        step = ExecutedStep(
            index=len(self.trace) + 1,
            agent=agent,
            operation=operation,
            detail=detail,
            observation=observation or {},
            confidence_after=round(self.confidence, 3),
            new_documents=new_documents,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.trace.append(step)
        return step

    # ── derived views ──────────────────────────────────────────────────
    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    @property
    def num_steps(self) -> int:
        return len(self.trace)

    @property
    def stale_streak(self) -> int:
        """How many trailing steps discovered no new documents."""
        streak = 0
        for step in reversed(self.trace):
            if step.new_documents > 0:
                break
            streak += 1
        return streak

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "qtype": self.qtype,
            "kind": self.kind,
            "classification": self.classification,
            "slot_report": self.slot_report,
            "spec": self.spec.to_dict(),
            "plan": [s.to_dict() for s in self.plan],
            "trace": [s.to_dict() for s in self.trace],
            "num_steps": self.num_steps,
            "documents_touched": len(self.documents),
            "candidates": len(self.candidates),
            "confidence": round(self.confidence, 3),
            "missing_info": list(self.missing_info),
            "resolved_gaps": list(self.resolved_gaps),
            "strategy_changes": list(self.strategy_changes),
            "stop_reason": self.stop_reason,
            "answer": self.answer,
            "iterations": self.iterations,
            "tool_calls": list(self.tool_calls),
            "citations": list(self.citations),
            "synth_source": self.synth_source,
            "adjudication": self.adjudication,
            "rationale": self.rationale,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }
