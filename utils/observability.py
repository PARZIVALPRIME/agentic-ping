"""Enterprise Observability Subsystem.

Provides structured JSON logging, distributed trace context propagation (trace_id, span_id),
and real-time execution telemetry recording for production operations.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from utils.security import SecretMasker


@dataclass
class TraceContext:
    """Distributed tracing context for a single query investigation."""

    trace_id: str
    qid: str = ""
    start_time: float = field(default_factory=time.perf_counter)
    spans: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls, qid: str = "") -> "TraceContext":
        tid = f"trace-{qid or 'anon'}-{uuid.uuid4().hex[:12]}"
        return cls(trace_id=tid, qid=qid)

    def start_span(self, name: str, agent: str = "", operation: str = "") -> "Span":
        return Span(context=self, name=name, agent=agent, operation=operation)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "qid": self.qid,
            "elapsed_ms": round((time.perf_counter() - self.start_time) * 1000.0, 2),
            "num_spans": len(self.spans),
            "spans": list(self.spans),
        }


class Span:
    """An individual operation span within a trace."""

    def __init__(self, context: TraceContext, name: str, agent: str = "", operation: str = ""):
        self.context = context
        self.name = name
        self.agent = agent
        self.operation = operation
        self.span_id = uuid.uuid4().hex[:8]
        self.t0 = 0.0

    def __enter__(self) -> "Span":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.t0) * 1000.0
        record = {
            "span_id": self.span_id,
            "name": self.name,
            "agent": self.agent,
            "operation": self.operation,
            "duration_ms": round(duration_ms, 2),
            "error": str(exc_val) if exc_val else None,
        }
        self.context.spans.append(record)
        return False


class StructuredLogger:
    """Structured JSON logger outputting parseable NDJSON for observability pipelines."""

    def __init__(self, name: str = "agentic-graphrag"):
        self.name = name

    def log(self, level: str, message: str, trace_id: str = "", **kwargs: Any) -> None:
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level": level.upper(),
            "logger": self.name,
            "message": SecretMasker.mask_text(message),
            "trace_id": trace_id or "global",
            **SecretMasker.mask_dict(kwargs),
        }
        line = json.dumps(entry, ensure_ascii=False)
        sys.stderr.write(line + "\n")
        sys.stderr.flush()

    def info(self, message: str, trace_id: str = "", **kwargs: Any) -> None:
        self.log("INFO", message, trace_id=trace_id, **kwargs)

    def warning(self, message: str, trace_id: str = "", **kwargs: Any) -> None:
        self.log("WARNING", message, trace_id=trace_id, **kwargs)

    def error(self, message: str, trace_id: str = "", **kwargs: Any) -> None:
        self.log("ERROR", message, trace_id=trace_id, **kwargs)


logger = StructuredLogger()
