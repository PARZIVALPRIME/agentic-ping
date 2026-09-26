"""Shared pipeline contract and the full per-question metric schema.

Every pipeline returns a :class:`PipelineResult`. The schema deliberately
captures *process* metrics (steps, tools, agents, stop reason) alongside
*outcome* metrics (answer, citations, tokens, latency) because the hackathon
question is not only "which pipeline is more accurate" but "what did the extra
computation buy us".
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from utils.metrics import TokenCounter


@dataclass
class StepRecord:
    """A single observable unit of work inside a pipeline run."""

    name: str
    operation: str
    detail: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    """Result of running one pipeline on one question."""

    pipeline: str = ""
    question: str = ""
    qtype: str = ""
    answer: str = ""
    citations: List[str] = field(default_factory=list)

    # ── outcome / cost metrics ────────────────────────────────────────
    latency_ms: float = 0.0
    retrieval_steps: int = 0
    tools_called: List[str] = field(default_factory=list)
    agents_invoked: List[str] = field(default_factory=list)
    chunks_retrieved: int = 0
    docs_retrieved: int = 0
    context_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0

    # ── agentic-process metrics ───────────────────────────────────────
    method: str = ""
    strategy_changed: bool = False
    stop_reason: str = ""
    confidence: float = 0.0
    #: Residual doubt, reported explicitly (``1 - confidence``) because Round 2
    #: asks for *uncertainty* to be carried rather than a bare confidence score.
    #: A pipeline that says "0.6 confident" and one that says "40% uncertain"
    #: are the same fact stated in the two directions the rubric names.
    uncertainty: float = 0.0
    plan: List[str] = field(default_factory=list)
    loop_iterations: int = 0
    candidates_considered: int = 0

    # ── provenance ────────────────────────────────────────────────────
    steps: List[Dict[str, Any]] = field(default_factory=list)
    time_per_operation: List[Dict[str, Any]] = field(default_factory=list)
    tokens_per_operation: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    unresolved: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Pipeline(Protocol):
    """A pipeline maps a question to a :class:`PipelineResult`."""

    name: str

    def run(self, question: str, qid: str = "") -> PipelineResult:
        ...


class Timer:
    """Small context manager that records elapsed milliseconds."""

    def __init__(self, sink: List[Dict[str, Any]], operation: str, detail: str = "") -> None:
        self.sink = sink
        self.operation = operation
        self.detail = detail
        self.elapsed_ms = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        self.sink.append({"operation": self.operation, "detail": self.detail,
                          "latency_ms": round(self.elapsed_ms, 2)})


def finalise_result(result: PipelineResult, counter: TokenCounter,
                    timings: List[Dict[str, Any]], started_at: float) -> PipelineResult:
    """Attach token + timing breakdowns and the total wall-clock latency."""
    result.input_tokens = counter.input_tokens
    result.output_tokens = counter.output_tokens
    result.total_tokens = counter.total_tokens
    result.tokens_per_operation = counter.per_operation
    result.time_per_operation = timings
    result.latency_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
    # Uncertainty is the complement of confidence, clipped to [0, 1] so a
    # pipeline that reports 1.0 confidence still reports 0.0 uncertainty.
    result.uncertainty = round(max(0.0, min(1.0, 1.0 - float(result.confidence or 0.0))), 3)
    return result