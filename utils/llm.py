"""Thin LLM façade over :mod:`llm_service`.

Design rule for this project: **the LLM is an accelerator, never a
requirement.** When a provider key is configured the pipelines use it for
answer synthesis, evidence judging and LLM-as-judge evaluation; when no key is
present every pipeline still runs end-to-end using deterministic logic. This
keeps the benchmark reproducible on any machine.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from utils.metrics import TokenCounter, count_tokens

logger = logging.getLogger(__name__)


class LLMHelper:
    """Provider-agnostic completion helper with token accounting."""

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None,
                 chat_model: Optional[str] = None, fast_model: Optional[str] = None,
                 eval_model: Optional[str] = None, temperature: float = 0.0,
                 max_tokens: int = 0, reasoning_effort: str = "",
                 base_url: Optional[str] = None) -> None:
        self.provider = provider
        self.chat_model = chat_model
        self.fast_model = fast_model or chat_model
        self.eval_model = eval_model or chat_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self._service = None
        self.available = False
        self.error = ""
        self.num_calls = 0
        self.failed_calls = 0
        self.last_error = ""

        if not provider or not api_key:
            self.error = "no provider/API key configured"
            return
        try:
            from llm_service import LLMService

            self._service = LLMService(provider=provider, api_key=api_key,
                                       model=self.chat_model or "", temperature=temperature,
                                       max_completion_tokens=max_tokens or None,
                                       reasoning_effort=reasoning_effort or None,
                                       ollama_base_url=base_url or "")
            # A local server that is not running would otherwise report
            # available=True (the client is lazy) and then fail every call of a
            # long sweep, which is how a run silently ends up deterministic.
            reachable, why = self._service.ping()
            if not reachable:
                self.error = f"local model server not reachable: {why}"
                logger.warning("LLM unavailable (%s); running deterministically", self.error)
                return
            self.available = True
        except Exception as exc:  # missing SDK, bad key, ...
            self.error = f"{exc.__class__.__name__}: {exc}"
            logger.warning("LLM unavailable (%s); running deterministically", self.error)

        # ── completion helpers ──────────────────────────────────────────────
    def complete(self, prompt: str, system_prompt: str = "", caller: str = "llm",
                 counter: Optional[TokenCounter] = None,
                 model: Optional[str] = None) -> Tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens).

        On rate-limit exhaustion the call returns empty rather than raising, so
        callers fall back to deterministic logic. Only the first failure of each
        kind is logged, to keep a long run's log readable.
        """
        if not self.available:
            return "", 0, 0
        if model and model != self._service.model:
            self._service.model = model
        try:
            text, usage = self._service.complete(prompt, system_prompt, caller)
            self.num_calls += 1
        except Exception as exc:
            self.failed_calls += 1
            self.last_error = f"{exc.__class__.__name__}: {exc}"[:200]
            if self.failed_calls <= 3 or self.failed_calls % 50 == 0:
                logger.warning("LLM call '%s' failed (%d so far): %s",
                               caller, self.failed_calls, self.last_error)
            return "", 0, 0
        if counter is not None:
            counter.add(caller, usage.input_tokens, usage.output_tokens,
                        detail=(model or self._service.model))
        return text, usage.input_tokens, usage.output_tokens

    def chat(self, messages: List[Dict[str, Any]],
             tools: Optional[List[Dict[str, Any]]] = None,
             counter: Optional[TokenCounter] = None,
             model: Optional[str] = None, caller: str = "chat"):
        """One tool-calling turn; returns ``(text, tool_calls, finish_reason)``.

        ``tool_calls`` is a list of dicts ``{"id", "name", "arguments"}`` with
        ``arguments`` already parsed. Returns an empty triple when the provider
        is unavailable or rate-limiting, which lets the caller fall back to
        deterministic reasoning instead of failing.
        """
        if not self.available:
            return "", [], ""
        try:
            turn = self._service.chat(messages, tools=tools, model=model,
                                      caller=caller)
            self.num_calls += 1
        except Exception as exc:
            self.failed_calls += 1
            self.last_error = f"{exc.__class__.__name__}: {exc}"[:200]
            if self.failed_calls <= 3 or self.failed_calls % 50 == 0:
                logger.warning("LLM chat '%s' failed (%d so far): %s",
                               caller, self.failed_calls, self.last_error)
            return "", [], ""
        if counter is not None:
            counter.add(caller, turn.usage.input_tokens, turn.usage.output_tokens,
                        detail=(model or self._service.model))
        calls = [{"id": tc.id, "name": tc.name,
                  "arguments": tc.parsed_args(), "raw": tc.arguments}
                 for tc in turn.tool_calls]
        return turn.text, calls, turn.finish_reason

    def complete_json(self, prompt: str, system_prompt: str = "", caller: str = "llm",
                      counter: Optional[TokenCounter] = None,
                      model: Optional[str] = None) -> Dict[str, Any]:
        if not self.available:
            return {}
        text, _i, _o = self.complete(prompt, system_prompt, caller, counter, model)
        if not text:
            return {}
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(l for l in cleaned.split("\n")
                                if not l.strip().startswith("```"))
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("could not parse JSON from %s response", caller)
            return {}

    def stats(self) -> Dict[str, Any]:
        """Provider/model identity plus rate-limit telemetry for the summary."""
        out: Dict[str, Any] = {
            "available": self.available,
            "provider": self.provider,
            "chat_model": self.chat_model,
            "eval_model": self.eval_model,
            "num_calls": self.num_calls,
            "failed_calls": self.failed_calls,
            "degraded": bool(self.failed_calls and not self.num_calls),
        }
        if self.error:
            out["error"] = self.error
        if self.last_error:
            out["last_error"] = self.last_error
        try:
            from llm_service import CIRCUIT, PACER

            out.update(PACER.stats())
            out["circuit"] = CIRCUIT.stats()
        except Exception:
            pass
        return out


def build_llm(config) -> LLMHelper:
    """Create an :class:`LLMHelper` from the project configuration."""
    llm = config.llm
    provider = getattr(llm, "provider", None)
    if provider == "none" or provider == "":
        return LLMHelper()
    return LLMHelper(
        provider=provider,
        api_key=llm.api_key,
        chat_model=llm.chat_model,
        fast_model=getattr(llm, "fast_model", None),
        eval_model=llm.eval_model,
        temperature=llm.temperature,
        max_tokens=getattr(llm, "max_tokens", 0),
        reasoning_effort=getattr(llm, "reasoning_effort", ""),
        base_url=(getattr(llm, "ollama_base_url", "") if provider == "ollama" else None),
    )


def render_context(items: List[Dict[str, Any]], max_chars_per_item: int = 1200,
                   max_items: int = 8) -> str:
    """Render retrieved evidence into a grounded context block for prompting."""
    blocks = []
    for i, item in enumerate(items[:max_items], start=1):
        text = (item.get("text") or "").strip()
        if len(text) > max_chars_per_item:
            text = text[:max_chars_per_item] + " ..."
        doc_id = item.get("doc_id") or item.get("chunk_id") or f"item-{i}"
        blocks.append(f"[{i}] doc_id={doc_id} title={item.get('title', '')}\n{text}")
    return "\n\n".join(blocks)


ANSWER_SYSTEM_PROMPT = (
    "You are a meticulous QA assistant for an Olympics corpus. Answer the "
    "question using ONLY the provided context. Follow these rules strictly:\n"
    "1. Answer the question directly and concisely (a number, a name, or a title).\n"
    "2. For counting or comparison questions, treat the corpus as the only truth "
    "and count/compare across every relevant document in the context.\n"
    "3. If the context is insufficient, reply with your best supported answer and "
    "note the uncertainty briefly.\n"
    "4. Never use outside knowledge; never invent documents."
)


def answer_prompt(question: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


# ── LLM adjudication (shared by all three pipelines) ──────────────────────
# The deterministic layer produces a *candidate* answer from structured evidence
# (infobox fields, graph counts, argmax over the candidate set). The LLM then
# adjudicates between that candidate and the retrieved passages. It may correct
# the candidate only when the passages contradict it, which keeps the LLM from
# inventing counts or names while still letting it fix linking mistakes.

ADJUDICATE_SYSTEM_PROMPT = """You verify answers for an Olympics question-answering system.

You receive: the question, a CANDIDATE answer produced by deterministic structured reasoning over a knowledge graph, and the retrieved CONTEXT passages.

Rules:
1. The candidate answer is derived from structured corpus fields (infobox values, exhaustive graph counts, argmax over an event set). Treat it as the default answer.
2. Keep the candidate if the context supports it or does not contradict it.
3. Replace it ONLY if the context clearly contradicts it (e.g. the candidate names the wrong event, the wrong edition, or the wrong field).
4. If the candidate is empty, answer from the context.
5. The answer must be a short verbatim span from the corpus: a number, a person's name, an event title, or a medal-winner name. Never explain, never hedge, never add units beyond what the corpus states.
6. If neither the candidate nor the context supports any answer, return an empty string.

Reply with JSON only:
{"answer": "<final answer>", "agree": true|false, "reason": "<one short sentence>"}"""


def refine_answer(llm, question: str, candidate: str,
                  context_items: List[Dict[str, Any]],
                  counter: Optional[TokenCounter] = None,
                  caller: str = "llm.refine") -> Tuple[str, bool, Dict[str, Any]]:
    """Adjudicate a structured candidate answer against retrieved context.

    Returns ``(answer, changed, payload)``. When the LLM is unavailable the
    candidate is returned unchanged, so every pipeline keeps working offline.
    """
    candidate = (candidate or "").strip()
    if llm is None or not getattr(llm, "available", False):
        return candidate, False, {"reason": "llm_unavailable"}
    if not context_items and not candidate:
        return "", False, {"reason": "no_evidence"}

    # Keep the adjudication prompt small: on metered tiers the prompt size, not
    # the request count, is the binding constraint. 6 x ~450 chars is enough to
    # adjudicate a candidate answer against the corpus.
    context = render_context(context_items, max_chars_per_item=450, max_items=6)
    prompt = (f"Question: {question}\n\n"
              f"CANDIDATE answer: {candidate if candidate else '(none - deterministic '
                 'reasoning found no answer)'}\n\n"
              f"CONTEXT:\n{context}\n\n"
              "Return the JSON verdict now.")
    payload = llm.complete_json(prompt, ADJUDICATE_SYSTEM_PROMPT,
                                caller=caller, counter=counter)
    final = str(payload.get("answer", "") or "").strip()
    agree = bool(payload.get("agree", False))

    # Guard rails: a verdict that fails to produce a usable short answer keeps
    # the deterministic candidate rather than degrading the pipeline.
    if not final or len(final) > 160:
        return candidate, False, {"reason": "verdict_unusable",
                                  "raw_answer": final[:120]}
    changed = _norm(final) != _norm(candidate)
    return final, changed, {"agree": agree, "reason": str(payload.get("reason", ""))[:200],
                            "candidate": candidate}


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split()).strip(" \t\"'.")