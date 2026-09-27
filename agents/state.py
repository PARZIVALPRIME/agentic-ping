"""Agent state: the shared blackboard the specialised agents read and write.

The state is the *harness* referenced in the plan. It records the plan, the
evidence accumulated so far, the full execution trace, confidence and the
information gaps that remain. Agents never talk to each other directly; they
read and extend this object, which makes every run inspectable after the fact
(the dashboard replays these traces).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from kg.textutil import normalize
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
    cumulative_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index, "agent": self.agent, "operation": self.operation,
            "detail": self.detail, "observation": self.observation,
            "confidence_after": self.confidence_after,
            "new_documents": self.new_documents,
            "latency_ms": round(self.latency_ms, 2),
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "cumulative_tokens": self.cumulative_tokens,
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
        #: Audit trail of how the question became a QuerySpec: which slots came
        #: from the semantic (LLM) parse, which from the deterministic template
        #: parser, which were rejected, and where the two disagreed. Written by
        #: ``OrchestratorAgent.run`` from
        #: ``reasoning.query_parser.parse_question_with_llm`` and serialised in
        #: ``to_dict`` so the claim is checkable in the result files.
        self.parse_report: Dict[str, Any] = {}
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
        #: Round 2: the outcome of adjudicating competing versions of a fact
        #: (rule applied, versions superseded, residual confidence) plus the
        #: uncertainty carried on the result. Both are surfaced in the trace and
        #: the pipeline metadata rather than staying inside the resolver.
        self.fact_conflicts: Dict[str, Any] = {}
        self.uncertainty: float = 0.0
        self._t0 = time.perf_counter()

        # Explicit stateful investigation attributes
        self.trace_id: str = f"trace-{qid or 'anon'}-{uuid.uuid4().hex[:8]}"
        self.normalized_query: str = normalize(question) if question else ""
        self.entities: List[str] = self._extract_entities(spec)
        self.constraints: Dict[str, Any] = self._extract_constraints(spec)
        self.inferred_capabilities: List[str] = self._infer_capabilities(self.kind)
        self.current_strategy: str = f"investigate_{self.kind}"
        self.retrieved_evidence: List[Dict[str, Any]] = []
        self.graph_evidence: List[Dict[str, Any]] = []
        self.candidate_answers: List[Dict[str, Any]] = []
        self.evidence_bundle: Dict[str, Any] = {}
        self.retries: int = 0

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

    @staticmethod
    def _extract_entities(spec: QuerySpec) -> List[str]:
        entities: List[str] = []
        if getattr(spec, "sport", ""):
            entities.append(f"Sport:{spec.sport}")
        if getattr(spec, "venue", ""):
            entities.append(f"Venue:{spec.venue}")
        if getattr(spec, "target_title", ""):
            entities.append(f"Event:{spec.target_title}")
        if getattr(spec, "event_desc", ""):
            entities.append(f"Description:{spec.event_desc}")
        return entities

    @staticmethod
    def _extract_constraints(spec: QuerySpec) -> Dict[str, Any]:
        constraints: Dict[str, Any] = {}
        if getattr(spec, "year", 0):
            constraints["year"] = spec.year
        if getattr(spec, "season", ""):
            constraints["season"] = spec.season
        if getattr(spec, "before_year", None) is not None:
            constraints["before_year"] = spec.before_year
        if getattr(spec, "threshold", None) is not None:
            constraints["threshold"] = spec.threshold
            constraints["comparator"] = getattr(spec, "comparator", "gt")
        if getattr(spec, "direction", ""):
            constraints["direction"] = spec.direction
        if getattr(spec, "date_text", ""):
            constraints["date_text"] = spec.date_text
        return constraints

    @staticmethod
    def _infer_capabilities(kind: str) -> List[str]:
        mapping = {
            "lookup": ["single_fact_retrieval", "entity_linking"],
            "multi_hop": ["venue_date_linking", "graph_neighbourhood", "winner_traversal"],
            "temporal": ["chronological_sequence", "temporal_anchor_linking", "edition_traversal"],
            "aggregation": ["set_enumeration", "filter_threshold", "exhaustive_count", "accumulator_stats"],
            "superlative": ["set_enumeration", "extreme_reduction", "total_ordering", "argmax_argmin"],
        }
        return list(mapping.get(kind, ["general_investigation", "vector_search"]))

    def adopt_spec(self, spec: QuerySpec, kind: Optional[str] = None) -> None:
        """Install a parsed spec and keep ``qtype``/``kind`` in sync with it.

        Used once, at the top of a run, after the semantic parse: the parse may
        have reclassified the question (the model reads the requirement, not the
        wording), and every downstream agent routes on ``kind``. ``kind`` is
        passed in by the caller so the mapping has a single definition.
        """
        self.spec = spec
        self.qtype = spec.qtype
        self.kind = kind or (spec.qtype if spec.qtype in
                             ("lookup", "multi_hop", "temporal", "aggregation",
                              "superlative") else "lookup")
        self.entities = self._extract_entities(spec)
        self.constraints = self._extract_constraints(spec)
        self.inferred_capabilities = self._infer_capabilities(self.kind)
        self.current_strategy = f"investigate_{self.kind}"

    @property
    def completed_steps(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.trace]

    @property
    def pending_information_gaps(self) -> List[str]:
        return list(self.missing_info)

    @property
    def conflicts(self) -> Dict[str, Any]:
        return self.fact_conflicts

    @property
    def tool_history(self) -> List[Dict[str, Any]]:
        return list(self.tool_calls)

    @property
    def agent_history(self) -> List[str]:
        return list(dict.fromkeys(s.agent for s in self.trace))

    @property
    def termination_reason(self) -> str:
        return self.stop_reason

    @termination_reason.setter
    def termination_reason(self, val: str) -> None:
        self.stop_reason = val

    def information_inventory(self) -> Dict[str, Any]:
        """Inventory of currently established facts, evidence, and remaining gaps."""
        return {
            "query": self.question,
            "normalized_query": self.normalized_query,
            "entities": list(self.entities),
            "constraints": dict(self.constraints),
            "inferred_capabilities": list(self.inferred_capabilities),
            "current_strategy": self.current_strategy,
            "documents_count": len(self.documents),
            "passages_count": len(self.chunks),
            "candidates_count": len(self.candidates),
            "has_answer": bool(self.answer),
            "current_answer": self.answer,
            "confidence": round(self.confidence, 3),
            "uncertainty": round(self.uncertainty, 3),
            "pending_gaps": list(self.missing_info),
            "resolved_gaps": list(self.resolved_gaps),
            "conflicts_detected": bool(self.fact_conflicts.get("had_conflict", False)),
        }

    def evaluate_uncertainty_reduction(self, step_op: str, observation: Dict[str, Any]) -> Tuple[float, str]:
        """Quantify whether the completed action reduced uncertainty."""
        prev_unc = self.uncertainty
        new_unc = round(max(0.0, min(1.0, 1.0 - float(self.confidence or 0.0))), 3)
        self.uncertainty = new_unc
        delta = round(prev_unc - new_unc, 3)
        if delta > 0.02:
            reason = f"Action '{step_op}' reduced uncertainty by {delta:.2f} (new confidence: {self.confidence:.2f})"
        elif delta < -0.02:
            reason = f"Action '{step_op}' raised uncertainty by {-delta:.2f} due to conflicting evidence"
        else:
            reason = f"Action '{step_op}' maintained uncertainty at {self.uncertainty:.2f}"
        return delta, reason

    def record_candidate(self, candidate: str, source: str, conf: float, evidence: Any = None) -> None:
        """Record an answer candidate with provenance and score."""
        clean = (candidate or "").strip()
        if not clean:
            return
        self.candidate_answers.append({
            "candidate": clean,
            "source": source,
            "confidence": round(conf, 3),
            "evidence": evidence,
            "step_index": len(self.trace),
        })

    def record_graph_evidence(self, op: str, data: Dict[str, Any]) -> None:
        """Record graph vertices, edges, or traversal paths observed."""
        self.graph_evidence.append({
            "operation": op,
            "timestamp_ms": round((time.perf_counter() - self._t0) * 1000.0, 2),
            "data": data,
        })

    def record_retrieved_evidence(self, op: str, data: Dict[str, Any]) -> None:
        """Record retrieved text passages and provenance."""
        self.retrieved_evidence.append({
            "operation": op,
            "timestamp_ms": round((time.perf_counter() - self._t0) * 1000.0, 2),
            "data": data,
        })

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
            "trace_id": self.trace_id,
            "qid": self.qid,
            "question": self.question,
            "normalized_query": self.normalized_query,
            "entities": self.entities,
            "constraints": self.constraints,
            "inferred_capabilities": self.inferred_capabilities,
            "current_strategy": self.current_strategy,
            "qtype": self.qtype,
            "kind": self.kind,
            "classification": self.classification,
            "slot_report": self.slot_report,
            "parse_report": self.parse_report,
            "spec": self.spec.to_dict(),
            "plan": [s.to_dict() for s in self.plan],
            "trace": [s.to_dict() for s in self.trace],
            "completed_steps": self.completed_steps,
            "num_steps": self.num_steps,
            "documents_touched": len(self.documents),
            "candidates": len(self.candidates),
            "candidate_answers": self.candidate_answers,
            "evidence_bundle": self.evidence_bundle,
            "retrieved_evidence": self.retrieved_evidence,
            "graph_evidence": self.graph_evidence,
            "confidence": round(self.confidence, 3),
            "uncertainty": round(self.uncertainty, 3),
            "missing_info": list(self.missing_info),
            "pending_information_gaps": self.pending_information_gaps,
            "resolved_gaps": list(self.resolved_gaps),
            "strategy_changes": list(self.strategy_changes),
            "stop_reason": self.stop_reason,
            "termination_reason": self.termination_reason,
            "answer": self.answer,
            "iterations": self.iterations,
            "tool_calls": list(self.tool_calls),
            "tool_history": self.tool_history,
            "agent_history": self.agent_history,
            "citations": list(self.citations),
            "synth_source": self.synth_source,
            "adjudication": self.adjudication,
            "conflicts": self.conflicts,
            "rationale": self.rationale,
            "retries": self.retries,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }
