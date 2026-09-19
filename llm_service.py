"""
Agentic GraphRAG — LLM Service Wrapper

Unified interface for multiple LLM providers (OpenAI, Gemini, Azure, Groq).
Tracks token usage per call for benchmarking metrics.

Rate-limit policy
-----------------
Free/on-demand provider tiers (Groq in particular) reject bursts with HTTP 429.
The naive strategy — fire immediately, retry on failure — wastes a whole
exponential-backoff cycle per rejected call and makes a long benchmark look
hung. Instead we *pace* requests client-side:

* ``LLM_MIN_INTERVAL_S`` (env, default 0.0 = off) enforces a minimum spacing
  between request starts, so a 100-question benchmark stays under the per-minute
  quota instead of colliding with it.
* When a 429 does arrive we wait the server's ``Retry-After`` hint when it is
  present, otherwise an exponential backoff with jitter, and we never wait
  longer than ``LLM_MAX_BACKOFF_S`` (default 20s) for a single attempt.

The total time any single call can spend retrying is capped, so one unlucky
question cannot stall the whole run.
"""

import json
import os
import random
import threading
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from tenacity import (retry, retry_if_exception, stop_after_attempt,
                      stop_after_delay, wait_exponential_jitter)

logger = logging.getLogger(__name__)


# ── client-side pacing + backoff ──────────────────────────────────────────
class _Pacer:
    """Serialises request starts so we stay under the provider's rate limit.

    Two budgets are enforced because providers meter both:

    * ``LLM_MIN_INTERVAL_S`` - minimum spacing between request *starts*.
    * ``LLM_TPM``            - approximate tokens-per-minute budget. Each call
      declares its estimated prompt size and the pacer sleeps long enough that
      a sliding one-minute window stays inside the budget.

    The token budget is what actually matters for this corpus: adjudication
    prompts are ~1-2k tokens, so a free Groq tier accepts only a few calls per
    minute. Without it the runner 429-thrashes and looks frozen.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last = 0.0
        self._window: List[Tuple[float, int]] = []   # (timestamp, tokens)
        self.min_interval = float(os.getenv("LLM_MIN_INTERVAL_S", "0") or 0)
        self.tpm = int(os.getenv("LLM_TPM", "0") or 0)
        # Askers only know the prompt size; the provider meters prompt+completion.
        # We reserve a small completion estimate up front and top it up with the
        # real usage afterwards, otherwise long agent runs 429 just below the cap.
        self.output_reserve = int(os.getenv("LLM_OUTPUT_RESERVE", "250") or 0)
        self.rate_limit_events = 0
        self.pace_sleeps = 0
        self.pace_slept_s = 0.0

    def wait(self, prompt_tokens: int = 0) -> None:
        """Sleep until this request fits inside both pacing budgets."""
        if self.min_interval <= 0 and self.tpm <= 0:
            return
        asked = max(prompt_tokens, 0) + self.output_reserve
        with self._lock:
            slept = 0.0
            now = time.monotonic()
            if self.min_interval > 0:
                gap = now - self._last
                if gap < self.min_interval:
                    delay = self.min_interval - gap
                    time.sleep(delay)
                    slept += delay
                    now = time.monotonic()
            if self.tpm > 0:
                # Loop: a single sleep only expires the oldest entry, which is
                # not necessarily enough to make room for this request.
                for _ in range(60):
                    self._window = [(t, n) for (t, n) in self._window
                                    if t > now - 60.0]
                    used = sum(n for _t, n in self._window)
                    if used + asked <= self.tpm or not self._window:
                        break
                    delay = max(0.5, 60.0 - (now - self._window[0][0]))
                    time.sleep(delay)
                    slept += delay
                    now = time.monotonic()
                self._window.append((now, asked))
            self._last = time.monotonic()
            if slept > 0:
                self.pace_sleeps += 1
                self.pace_slept_s += slept

    def record_output(self, output_tokens: int) -> None:
        """Top up the window with the completion tokens a call actually used."""
        if self.tpm <= 0 or output_tokens <= 0:
            return
        with self._lock:
            self._window.append((time.monotonic(), int(output_tokens)))

    def note_rate_limit(self) -> None:
        with self._lock:
            self.rate_limit_events += 1

    def stats(self) -> Dict[str, Any]:
        return {"min_interval_s": self.min_interval,
                "tpm_budget": self.tpm,
                "rate_limit_events": self.rate_limit_events,
                "pace_sleeps": self.pace_sleeps,
                "pace_slept_s": round(self.pace_slept_s, 1)}


class LLMCircuitOpen(RuntimeError):
    """Raised when the provider is known to be rate-limiting us.

    Deliberately *not* retryable: retrying while the quota is exhausted is what
    makes a long benchmark look hung. Callers fall back to deterministic logic.
    """


class _Circuit:
    """Trips after N consecutive rate-limit failures, then fails fast.

    A benchmark run must never stall on a provider that has run out of quota.
    After ``LLM_CIRCUIT_THRESHOLD`` consecutive 429s the circuit opens and every
    later call raises :class:`LLMCircuitOpen` immediately (no retry, no sleep).
    After ``LLM_CIRCUIT_COOLDOWN_S`` one probe call is allowed through; if it
    succeeds the circuit closes, otherwise it opens again.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.threshold = int(os.getenv("LLM_CIRCUIT_THRESHOLD", "3") or 3)
        self.cooldown = float(os.getenv("LLM_CIRCUIT_COOLDOWN_S", "60") or 60)
        self.consecutive = 0
        self.opened_at = 0.0
        self.trips = 0
        self.blocked_calls = 0

    @property
    def is_open(self) -> bool:
        if self.opened_at <= 0:
            return False
        return (time.monotonic() - self.opened_at) < self.cooldown

    def check(self) -> None:
        """Raise if the circuit is open and the cooldown has not elapsed."""
        if self.is_open:
            with self._lock:
                self.blocked_calls += 1
            raise LLMCircuitOpen(
                f"provider rate-limited; circuit open "
                f"({self.cooldown:.0f}s cooldown, {self.trips} trip(s))")

    def record_rate_limit(self) -> None:
        with self._lock:
            self.consecutive += 1
            if self.consecutive >= self.threshold:
                self.opened_at = time.monotonic()
                self.trips += 1

    def record_success(self) -> None:
        with self._lock:
            self.consecutive = 0
            self.opened_at = 0.0

    def stats(self) -> Dict[str, Any]:
        return {"threshold": self.threshold, "cooldown_s": self.cooldown,
                "trips": self.trips, "blocked_calls": self.blocked_calls,
                "open": self.is_open}


CIRCUIT = _Circuit()

PACER = _Pacer()
_MAX_BACKOFF = float(os.getenv("LLM_MAX_BACKOFF_S", "20") or 20)
_MAX_CALL_SECONDS = float(os.getenv("LLM_MAX_CALL_S", "90") or 90)


def _estimate_tokens(text: str) -> int:
    """Cheap prompt-size estimate (~4 chars/token, matching utils.metrics)."""
    if not text:
        return 0
    return max(1, int(round(len(text) / 4.0)))


def _status_of(exc: BaseException) -> Optional[int]:
    for attr in ("status_code", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _retry_after_of(exc: BaseException) -> Optional[float]:
    """Read a server ``Retry-After`` hint (seconds) when the SDK exposes it."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _is_transient(exc: BaseException) -> bool:
    """Only rate limits and transport hiccups are worth retrying."""
    if isinstance(exc, LLMCircuitOpen):
        return False          # fail fast: retrying an exhausted quota is useless
    status = _status_of(exc)
    if status in (408, 409, 429, 500, 502, 503, 504):
        return True
    name = exc.__class__.__name__.lower()
    if any(k in name for k in ("ratelimit", "timeout", "connection",
                               "apiconnection", "internalserver")):
        return True
    return False


def _classify(exc: BaseException) -> str:
    return "rate_limit" if _status_of(exc) == 429 or "ratelimit" in \
        exc.__class__.__name__.lower() else "transient"


def wait_rate_limit(retry_state) -> float:
    """Backoff that honours ``Retry-After`` and stays under ``_MAX_BACKOFF``."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is not None and _classify(exc) == "rate_limit":
        PACER.note_rate_limit()
        CIRCUIT.record_rate_limit()
        hint = _retry_after_of(exc)
        if hint is not None:
            return min(hint + 0.25, _MAX_BACKOFF)
    # exponential with jitter: ~1, 2, 4, 8 ... capped
    exp = min(2 ** max(0, retry_state.attempt_number - 1), _MAX_BACKOFF / 2)
    return min(exp + random.uniform(0, 0.75), _MAX_BACKOFF)


RETRY_POLICY = dict(
    stop=(stop_after_attempt(5) | stop_after_delay(_MAX_CALL_SECONDS)),
    wait=wait_rate_limit,
    retry=retry_if_exception(_is_transient),
    reraise=True,
)



@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0
    caller: str = ""


@dataclass
class ToolCall:
    """One function call requested by the model."""
    id: str
    name: str
    arguments: str = "{}"      # raw JSON string as returned by the provider

    def parsed_args(self) -> Dict[str, Any]:
        try:
            out = json.loads(self.arguments or "{}")
            return out if isinstance(out, dict) else {}
        except json.JSONDecodeError:
            return {}


@dataclass
class ChatTurn:
    """Result of one tool-calling chat turn."""
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str = ""


class TokenTracker:
    """Accumulates token usage across multiple calls."""

    def __init__(self) -> None:
        # NOTE: this must be an instance attribute. Using dataclasses.field()
        # in a plain class leaves a Field object on the class, so every
        # `self.calls.append(...)` raised AttributeError - which silently turned
        # every LLM call into a failure and reported zero token usage.
        self.calls: List[TokenUsage] = []

    def record(self, usage: TokenUsage):
        self.calls.append(usage)

    @property
    def total_input(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    @property
    def total_latency_ms(self) -> float:
        return sum(c.latency_ms for c in self.calls)

    @property
    def num_calls(self) -> int:
        return len(self.calls)

    def to_dict(self) -> dict:
        return {
            "total_input_tokens": self.total_input,
            "total_output_tokens": self.total_output,
            "total_tokens": self.total_tokens,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "num_calls": self.num_calls,
            "calls": [
                {
                    "caller": c.caller,
                    "model": c.model,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "total_tokens": c.total_tokens,
                    "latency_ms": round(c.latency_ms, 2),
                }
                for c in self.calls
            ],
        }

    def reset(self):
        self.calls.clear()


class LLMService:
    """Unified LLM interface with token tracking."""

    def __init__(self, provider: str, api_key: str, model: str,
                 temperature: float = 0.0, **kwargs):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.tracker = TokenTracker()
        self._client = None
        # Generation budget. Reasoning-capable models (e.g. gpt-oss) bill
        # reasoning tokens as completion tokens, so callers cap the output.
        self.max_completion_tokens = int(kwargs.get("max_completion_tokens") or 0) or None
        self.reasoning_effort = kwargs.get("reasoning_effort") or ""

        if provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)

        elif provider == "gemini":
            from google import genai
            self._client = genai.Client(api_key=api_key)

        elif provider == "azure":
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=kwargs.get("azure_endpoint", ""),
                api_version=kwargs.get("api_version", "2024-10-21"),
            )

        elif provider == "groq":
            # Groq exposes an OpenAI-compatible endpoint. Use OpenAI client with custom base_url.
            from openai import OpenAI
            base_url = kwargs.get("groq_base_url") or "https://api.groq.com/openai/v1"
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @retry(**RETRY_POLICY)
    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        caller: str = "unknown",
    ) -> Tuple[str, TokenUsage]:
        """Generate a completion and track token usage.

        Requests are paced client-side (see module docstring) and retried only
        on transient/rate-limit failures, with the server's Retry-After hint.

        Returns:
            Tuple of (response_text, token_usage)
        """
        t0 = time.time()
        # Fail fast while the provider is known to be rate-limiting us, instead
        # of burning a full retry cycle per call (which looks like a hang).
        CIRCUIT.check()
        # Tell the pacer how big this request is: the token budget, not the
        # request count, is what free provider tiers actually enforce.
        PACER.wait(_estimate_tokens(prompt) + _estimate_tokens(system_prompt))

        if self.provider == "gemini":
            out = self._complete_gemini(prompt, system_prompt, caller, t0)
        else:
            out = self._complete_openai_compat(prompt, system_prompt, caller, t0)
        PACER.record_output(out[1].output_tokens)
        CIRCUIT.record_success()
        return out

    # ── tool calling ───────────────────────────────────────────────────
    @retry(**RETRY_POLICY)
    def chat(self, messages: List[Dict[str, Any]],
             tools: Optional[List[Dict[str, Any]]] = None,
             tool_choice: str = "auto",
             caller: str = "chat",
             model: Optional[str] = None) -> ChatTurn:
        """One tool-calling turn against an OpenAI-compatible endpoint.

        ``messages`` is the full conversation in OpenAI format (system/user/
        assistant/tool). ``model`` overrides the configured model for this call
        only (per-call, so a caller can run on a cheaper model without changing
        what the judge or the other agents use). Returns the assistant turn:
        either text, function calls to execute, or both.
        """
        use_model = model or self.model
        t0 = time.time()
        CIRCUIT.check()
        size = sum(_estimate_tokens(str(m.get("content") or "")) for m in messages)
        size += sum(_estimate_tokens(json.dumps(t.get("function", {})))
                    for t in (tools or []))
        PACER.wait(size)

        kwargs: Dict[str, Any] = dict(self._generation_kwargs())
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        response = self._client.chat.completions.create(
            model=use_model, messages=messages,
            temperature=self.temperature, **kwargs)

        choice = response.choices[0]
        msg = choice.message
        calls: List[ToolCall] = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            calls.append(ToolCall(id=getattr(tc, "id", "") or "",
                                  name=tc.function.name,
                                  arguments=tc.function.arguments or "{}"))
        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
            model=use_model,
            latency_ms=(time.time() - t0) * 1000, caller=caller)
        self.tracker.record(usage)
        PACER.record_output(usage.output_tokens)
        CIRCUIT.record_success()
        return ChatTurn(text=(msg.content or ""), tool_calls=calls,
                        usage=usage, finish_reason=choice.finish_reason or "")

    def _generation_kwargs(self) -> Dict[str, Any]:
        """Provider-specific generation limits, omitted when unset."""
        kwargs: Dict[str, Any] = {}
        if self.max_completion_tokens:
            kwargs["max_completion_tokens"] = self.max_completion_tokens
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return kwargs

    def _complete_openai_compat(
        self, prompt: str, system_prompt: str, caller: str, t0: float
    ) -> Tuple[str, TokenUsage]:
        """OpenAI-compatible completion (OpenAI, Azure, Groq)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self._generation_kwargs(),
        )

        text = response.choices[0].message.content or ""
        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
            model=self.model,
            latency_ms=(time.time() - t0) * 1000,
            caller=caller,
        )
        self.tracker.record(usage)
        return text, usage

    def _complete_gemini(
        self, prompt: str, system_prompt: str, caller: str, t0: float
    ) -> Tuple[str, TokenUsage]:
        """Google Gemini completion."""
        from google.genai import types

        contents = prompt
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=system_prompt if system_prompt else None,
        )

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        text = response.text or ""

        # Extract token usage from response metadata
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            model=self.model,
            latency_ms=(time.time() - t0) * 1000,
            caller=caller,
        )
        self.tracker.record(usage)
        return text, usage

    def complete_json(
        self,
        prompt: str,
        system_prompt: str = "",
        caller: str = "unknown",
    ) -> Tuple[Any, TokenUsage]:
        """Generate a completion and parse as JSON."""
        # Add JSON instruction to system prompt
        json_system = (system_prompt or "") + (
            "\n\nYou MUST respond with valid JSON only. No markdown, no code fences, no explanation."
        )
        text, usage = self.complete(prompt, json_system, caller)

        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            return json.loads(text), usage
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from LLM response: {text[:200]}")
            return {"error": "json_parse_failed", "raw": text}, usage


class EmbeddingService:
    """Unified embedding interface with token tracking."""

    def __init__(self, provider: str, api_key: str, model: str):
        self.provider = provider
        self.model = model
        self._client = None

        if provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)

        elif provider == "gemini":
            from google import genai
            self._client = genai.Client(api_key=api_key)

        elif provider == "groq":
            # Groq doesn't have embeddings — fall back to OpenAI-compatible
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)

        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def embed(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        if self.provider == "gemini":
            return self._embed_gemini(texts)
        else:
            return self._embed_openai_compat(texts)

    def _embed_openai_compat(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def _embed_gemini(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            response = self._client.models.embed_content(
                model=self.model,
                contents=text,
            )
            results.append(list(response.embeddings[0].values))
        return results


def create_llm_service(config) -> LLMService:
    """Factory function to create an LLM service from config."""
    kwargs = {}
    if config.provider == "azure":
        kwargs["azure_endpoint"] = config.azure_endpoint

    return LLMService(
        provider=config.provider,
        api_key=config.api_key,
        model=config.completion_model,
        temperature=config.temperature,
        **kwargs,
    )


def create_chat_llm(config) -> LLMService:
    """Create an LLM service specifically for chat/answer generation."""
    kwargs = {}
    if config.provider == "azure":
        kwargs["azure_endpoint"] = config.azure_endpoint

    return LLMService(
        provider=config.provider,
        api_key=config.api_key,
        model=config.chat_model,
        temperature=config.temperature,
        **kwargs,
    )


def create_eval_llm(config) -> LLMService:
    """Create an LLM service specifically for evaluation."""
    kwargs = {}
    if config.provider == "azure":
        kwargs["azure_endpoint"] = config.azure_endpoint

    return LLMService(
        provider=config.provider,
        api_key=config.api_key,
        model=config.eval_model,
        temperature=0.0,
        **kwargs,
    )


def create_embedding_service(config) -> EmbeddingService:
    """Factory function to create an embedding service from config."""
    return EmbeddingService(
        provider=config.provider,
        api_key=config.api_key,
        model=config.embedding_model,
    )
