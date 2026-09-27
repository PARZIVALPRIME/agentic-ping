"""Pipeline 4 - Router: classify first, then dispatch on capability and cost.

The benchmark's headline question is not "is the agent better" but "when does a
question *need* the agent, and when does a cheaper pipeline suffice". A router
answers that concretely: it classifies the question, then dispatches to the
cheapest pipeline whose capabilities actually cover what the question requires.

ROUTING ON CAPABILITY - NOT ON MEASURED ACCURACY
------------------------------------------------
An earlier version of this pipeline routed with a table of per-question-type
accuracies measured on the public evaluation set, sending each question to the
cheapest arm within 2 points of the best measured arm. That is benchmark-derived
routing. It encodes how well each arm happened to score on the questions it was
graded on, so it cannot transfer to differently-worded or unseen questions, and
it degenerates to "send everything to the most capable arm" the moment any arm
looks perfect. It has been removed.

What replaces it is a statement about *requirements and costs*, both of which are
properties of the pipeline rather than of any evaluation set:

* a question whose answer lives in one named document is answerable by top-k
  retrieval, and RAG is the cheapest arm that does it;
* a question whose evidence must be assembled across documents (an exhaustive
  candidate set for a count or an extreme) or across editions (the Games before
  another) cannot be answered by any top-k retriever - no k closes the set - so
  it goes to the agentic pipeline;
* a question the classifier could not type, or types only with low confidence,
  goes to the most capable pipeline, because routing an unclassified question
  cheaply risks a structurally unanswerable result.

The classifier's confidence is what carries the third case: a low-confidence
type is treated as "unknown" rather than acted upon.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from utils.metrics import TokenCounter
from utils.thresholds import thresholds

from .base import PipelineResult


def _ablate_router() -> bool:
    """ABLATE_ROUTER=1 bypasses capability routing entirely.

    Ablation E: every question escalates to the most capable arm, which is
    exactly the degenerate behaviour the router exists to avoid. Comparing
    this run against the routed run isolates the routing decision's own
    contribution (accuracy *and* cost).
    """
    return os.getenv("ABLATE_ROUTER", "").strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Route:
    """A routing decision expressed as a requirement, not a measurement."""

    target: str          # pipeline that satisfies the requirement
    requirement: str     # what this question type needs to be answerable
    why: str             # why that pipeline meets the requirement (incl. cost)


#: Capability routing table. Contains no accuracy figures by construction.
CAPABILITY_ROUTES: Dict[str, Route] = {
    "lookup": Route(
        "RAG",
        "a single fact held by one named document",
        "top-k vector retrieval reaches that document and is the cheapest arm "
        "(1 LLM call, ~5 chunks); no graph traversal or iteration is required"),
    "multi_hop": Route(
        "GraphRAG",
        "two sources linked: (venue, date) -> event page -> target field",
        "graph neighborhood expansion bridges the (venue, date) -> event traversal "
        "efficiently; escalates adaptively to Agentic GraphRAG if confidence < 0.85"),
    "temporal": Route(
        "Agentic GraphRAG",
        "the edition immediately before/after a stated one",
        "the PREV/NEXT edition chain must be walked and confirmed; top-k "
        "retrieval cannot follow an edge"),
    "aggregation": Route(
        "Agentic GraphRAG",
        "an exhaustive candidate set across many documents",
        "a count is only correct if every candidate is enumerated, and no top-k "
        "window contains the complete set"),
    "superlative": Route(
        "Agentic GraphRAG",
        "an exhaustive candidate set plus a total order over a field",
        "the extreme need not be in the retrieved window at all, so the set must "
        "be enumerated and compared rather than sampled"),
}

#: Used when the question type is unknown or the classifier is unsure.
UNKNOWN_ROUTE = Route(
    "Agentic GraphRAG",
    "not established from the question",
    "the most capable arm: routing an unclassified question to a pipeline that "
    "structurally cannot answer it is worse than paying for the agent")




class RouterPipeline:
    """Classify first, then dispatch to the cheapest capable pipeline."""

    name = "Router"

    def __init__(self, index, pipelines: List[Any], llm=None) -> None:
        from agents.classifier import QuestionClassifier

        self.index = index
        self.llm = llm
        self.kg = index.kg if index is not None else None
        self.classifier = QuestionClassifier(self.kg, llm)
        self._by_name = {p.name: p for p in pipelines if p.name != self.name}

    def _dispatch_target(self, qtype: str,
                         confidence: float = 1.0) -> Tuple[str, Route, str]:
        """Pick a pipeline from the question's capability and the classifier's confidence.

        Returns ``(target, route, note)``. No measured accuracy is consulted: the
        decision rests on what the question type requires and what each pipeline
        can do, with a low-confidence classification escalated to the most
        capable arm rather than acted upon.
        """
        min_confidence = thresholds().router_min_confidence
        if _ablate_router():
            return (UNKNOWN_ROUTE.target, UNKNOWN_ROUTE,
                    "router ablation: capability routing bypassed, everything "
                    "escalates to the most capable arm")
        route = CAPABILITY_ROUTES.get(qtype)
        note = ""
        if route is None:
            route = UNKNOWN_ROUTE
            note = f"'{qtype or 'unknown'}' is not a recognised question type"
        elif confidence < min_confidence:
            route = UNKNOWN_ROUTE
            note = (f"classification confidence {confidence:.2f} is below "
                    f"{min_confidence}: escalated instead of trusted")

        target = route.target
        if target not in self._by_name:
            for candidate in ("Agentic GraphRAG", "GraphRAG", "RAG"):
                if candidate in self._by_name:
                    target = candidate
                    break
            else:  # degenerate --pipelines selection
                target = next(iter(self._by_name))
            note = ((note + "; ") if note else "") + \
                f"'{route.target}' was not built in this run, using '{target}'"
        return target, route, note


    def run(self, question: str, qid: str = "") -> PipelineResult:
        started = time.perf_counter()
        counter = TokenCounter()

        t0 = time.perf_counter()
        try:
            classification = self.classifier.classify(question, counter=counter)
        except Exception as exc:  # classification must never sink a question
            classification = {"qtype": "", "method": "error",
                              "reason": f"{exc.__class__.__name__}: {exc}",
                              "confidence": 0.0}
        classify_ms = (time.perf_counter() - t0) * 1000.0

        qtype = str(classification.get("qtype", "") or "")
        try:
            confidence = float(classification.get("confidence", 1.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        target, route, route_note = self._dispatch_target(qtype, confidence)

        inner = self._by_name[target].run(question, qid)

        # Adaptive Escalation: If the dispatched arm reports insufficient evidence,
        # low confidence (< 0.85), or an empty answer, escalate to the agentic arm.
        # This guarantees 100% investigation coverage without paying agentic token
        # costs on clean, unambiguous lookups.
        escalation_note = ""
        agentic_arm = "Agentic GraphRAG"
        if target != agentic_arm and agentic_arm in self._by_name:
            needs_escalation = (
                (inner.confidence is not None and inner.confidence < 0.85)
                or inner.stop_reason in ("insufficient_retrieved_evidence", "no_candidate_docs_retrieved")
                or not (inner.answer or "").strip()
            )
            if needs_escalation:
                orig_target = target
                escalation_note = (
                    f"adaptive escalation: {orig_target} yielded confidence={inner.confidence} "
                    f"stop_reason='{inner.stop_reason}'; escalated to {agentic_arm}"
                )
                escalated_run = self._by_name[agentic_arm].run(question, qid)
                # Combine telemetry: router honestly accounts for all resources spent in both passes
                escalated_run.context_tokens += inner.context_tokens
                escalated_run.input_tokens += inner.input_tokens
                escalated_run.output_tokens += inner.output_tokens
                escalated_run.total_tokens += inner.total_tokens
                escalated_run.latency_ms += inner.latency_ms
                escalated_run.llm_calls += inner.llm_calls
                escalated_run.retrieval_steps += inner.retrieval_steps
                escalated_run.reasoning_steps += getattr(inner, "reasoning_steps", 0)
                escalated_run.chunks_retrieved += inner.chunks_retrieved
                escalated_run.docs_retrieved = max(escalated_run.docs_retrieved, inner.docs_retrieved)
                escalated_run.steps = inner.steps + escalated_run.steps
                inner = escalated_run
                target = agentic_arm

        # Present the inner run as a Router run, but keep every routing fact
        # visible so the decision is auditable rather than magic.
        result = PipelineResult(**inner.to_dict())
        result.pipeline = self.name
        result.qtype = inner.qtype or qtype

        routing_step = {
            "step": 0,
            "agent": "Router",
            "operation": "classify_and_route",
            "detail": f"qtype={qtype or 'unknown'} -> {target}" + (f" ({escalation_note})" if escalation_note else ""),
            "observation": {
                "qtype": qtype,
                "routed_to": target,
                "requirement": route.requirement,
                "capability_reason": route.why,
                "method": classification.get("method"),
                "confidence": classification.get("confidence"),
                "min_confidence": thresholds().router_min_confidence,
                "reason": classification.get("reason"),
                "routing_note": (route_note + ("; " + escalation_note if escalation_note else "")).strip("; "),
                "escalated": route is UNKNOWN_ROUTE or bool(escalation_note),
            },
        }
        result.steps = [routing_step] + list(inner.steps)
        result.time_per_operation = (
            [{"operation": "classify_and_route", "agent": "Router",
              "detail": f"-> {target}", "latency_ms": round(classify_ms, 2)}]
            + list(inner.time_per_operation))

        # Routing costs whatever the classifier spent, on top of the inner run.
        result.input_tokens = inner.input_tokens + counter.input_tokens
        result.output_tokens = inner.output_tokens + counter.output_tokens
        result.total_tokens = inner.total_tokens + counter.total_tokens
        result.tokens_per_operation = (list(counter.per_operation)
                                       + list(inner.tokens_per_operation))
        result.llm_calls = inner.llm_calls + (1 if counter.per_operation else 0)

        result.agents_invoked = ["Router"] + list(inner.agents_invoked)
        result.plan = [f"Router.route->{target}"] + list(inner.plan)
        result.metadata = {
            **dict(inner.metadata),
            "router": {
                "routed_to": target,
                "qtype": qtype,
                "classification": classification,
                "routing_table": {q: r.target for q, r in CAPABILITY_ROUTES.items()},
                "fallback_used": qtype not in CAPABILITY_ROUTES,
            },
            "inner_pipeline": target,
        }
        result.latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return result
