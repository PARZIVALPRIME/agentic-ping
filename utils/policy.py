"""Centralized execution policy and budget enforcement layer.

Defines hard and soft constraints on agentic exploration: step limits,
replan budgets, LLM call limits, token ceilings, latency timeouts, and
exponential backoff recovery policies.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ExecutionPolicy:
    """Centralized execution policy enforcing resource budgets and termination bounds."""

    # Step & loop bounds
    max_steps: int = _i("POLICY_MAX_STEPS", 15)
    max_replans: int = _i("POLICY_MAX_REPLANS", 2)
    max_widen_attempts: int = _i("POLICY_MAX_WIDEN_ATTEMPTS", 2)
    stale_step_limit: int = _i("POLICY_STALE_STEP_LIMIT", 3)

    # Confidence & sufficiency gates
    confidence_threshold: float = _f("POLICY_CONFIDENCE_THRESHOLD", 0.90)
    min_sufficient_confidence: float = _f("POLICY_MIN_SUFFICIENT_CONFIDENCE", 0.85)

    # Call & token budgets
    max_llm_calls_per_query: int = _i("POLICY_MAX_LLM_CALLS", 12)
    max_tokens_per_query: int = _i("POLICY_MAX_TOKENS", 60000)

    # Timeouts & backoff
    timeout_seconds: float = _f("POLICY_TIMEOUT_SECONDS", 45.0)
    backoff_base_s: float = _f("POLICY_BACKOFF_BASE_S", 0.5)
    backoff_max_s: float = _f("POLICY_BACKOFF_MAX_S", 4.0)
    max_retries: int = _i("POLICY_MAX_RETRIES", 3)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_DEFAULT_POLICY = ExecutionPolicy()


def get_execution_policy() -> ExecutionPolicy:
    """Retrieve the active centralized execution policy."""
    return _DEFAULT_POLICY


def check_budget(state: Any, policy: Optional[ExecutionPolicy] = None) -> Tuple[bool, str]:
    """Verify whether the ongoing investigation is within its allocated policy budget."""
    pol = policy or _DEFAULT_POLICY

    # 1. Step budget check
    if getattr(state, "num_steps", 0) >= pol.max_steps:
        return False, f"max_steps_exceeded ({state.num_steps} >= {pol.max_steps})"

    # 2. Token budget check
    tokens = getattr(state, "tokens", None)
    if tokens is not None:
        total_tok = getattr(tokens, "total_tokens", 0)
        if total_tok >= pol.max_tokens_per_query:
            return False, f"token_budget_exceeded ({total_tok} >= {pol.max_tokens_per_query})"

    # 3. Timeout check
    elapsed = getattr(state, "elapsed_ms", 0.0) / 1000.0
    if elapsed >= pol.timeout_seconds:
        return False, f"timeout_exceeded ({elapsed:.1f}s >= {pol.timeout_seconds:.1f}s)"

    return True, "within_budget"


def calculate_backoff(attempt: int, policy: Optional[ExecutionPolicy] = None) -> float:
    """Calculate exponential backoff duration with jitter for autonomous recovery."""
    pol = policy or _DEFAULT_POLICY
    base = pol.backoff_base_s * (2.0 ** max(0, attempt - 1))
    return min(pol.backoff_max_s, base)
