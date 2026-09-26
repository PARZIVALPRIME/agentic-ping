"""Thin LLM façade over :mod:`llm_service`.

Design rule for this project: **the LLM is an accelerator, never a
requirement.** When a provider key is configured the pipelines use it for
answer synthesis, evidence judging and LLM-as-judge evaluation; when no key is
present every pipeline still runs end-to-end using deterministic logic. This
keeps the benchmark reproducible on any machine.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

from utils.metrics import TokenCounter, count_tokens

logger = logging.getLogger(__name__)


class LLMHelper:
    """Provider-agnostic completion helper with token accounting."""

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None,
                 chat_model: Optional[str] = None, fast_model: Optional[str] = None,
                 eval_model: Optional[str] = None, temperature: float = 0.0,
                 eval_temperature: float = 0.0,
                 max_tokens: int = 0, reasoning_effort: str = "",
                 base_url: Optional[str] = None) -> None:
        self.provider = provider
        self.chat_model = chat_model
        self.fast_model = fast_model or chat_model
        self.eval_model = eval_model or chat_model
        self.temperature = temperature
        #: Sampling temperature for verification/judging calls. The agent model
        #: may explore (temperature > 0) while scoring and adjudication stay
        #: greedy, so a run is reproducible even though investigation is not.
        self.eval_temperature = eval_temperature
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
                                       base_url=base_url or "",
                                       ollama_base_url=base_url or "",
                                       gemini_base_url=base_url or "")
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

    @contextmanager
    def _at_temperature(self, temperature: Optional[float]) -> Iterator[None]:
        """Run the enclosed call at a different sampling temperature.

        Verification (adjudication, judging) must stay greedy even while the
        agent explores, so one call can pin its own temperature without changing
        the agent's sampling.
        """
        service = self._service
        if temperature is None or service is None:
            yield
            return
        previous = getattr(service, "temperature", None)
        try:
            service.temperature = temperature
            yield
        finally:
            if previous is not None:
                service.temperature = previous

        # ── completion helpers ──────────────────────────────────────────────
    def complete(self, prompt: str, system_prompt: str = "", caller: str = "llm",
                 counter: Optional[TokenCounter] = None,
                 model: Optional[str] = None,
                 temperature: Optional[float] = None) -> Tuple[str, int, int]:
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
            with self._at_temperature(temperature):
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
             model: Optional[str] = None, caller: str = "chat",
             temperature: Optional[float] = None):
        """One tool-calling turn; returns ``(text, tool_calls, finish_reason)``.

        ``tool_calls`` is a list of dicts ``{"id", "name", "arguments"}`` with
        ``arguments`` already parsed. Returns an empty triple when the provider
        is unavailable or rate-limiting, which lets the caller fall back to
        deterministic reasoning instead of failing.
        """
        if not self.available:
            return "", [], ""
        try:
            with self._at_temperature(temperature):
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
                  "arguments": tc.parsed_args(), "raw": tc.arguments,
                  "extra_content": getattr(tc, "extra_content", None)}
                 for tc in turn.tool_calls]
        return turn.text, calls, turn.finish_reason

    def complete_json(self, prompt: str, system_prompt: str = "", caller: str = "llm",
                      counter: Optional[TokenCounter] = None,
                      model: Optional[str] = None,
                      temperature: Optional[float] = None) -> Dict[str, Any]:
        if not self.available:
            return {}
        text, _i, _o = self.complete(prompt, system_prompt, caller, counter, model,
                                     temperature=temperature)
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
        eval_temperature=getattr(llm, "eval_temperature", 0.0),
        max_tokens=getattr(llm, "max_tokens", 0),
        reasoning_effort=getattr(llm, "reasoning_effort", ""),
        base_url=(
            getattr(llm, "ollama_base_url", "") if provider == "ollama"
            else (getattr(llm, "gemini_base_url", "") if provider == "gemini"
                  else (getattr(llm, "groq_base_url", "") if provider == "groq" else None))
        ),
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
    "You are a meticulous QA assistant for {corpus_label}. Answer the "
    "question using ONLY the provided context. Follow these rules strictly:\n"
    "1. Answer the question directly and concisely (a number, a name, or a title).\n"
    "2. For counting or comparison questions, treat the corpus as the only truth "
    "and count/compare across every relevant document in the context.\n"
    "3. If the context is insufficient, reply with your best supported answer and "
    "note the uncertainty briefly.\n"
    "4. Never use outside knowledge; never invent documents."
)


def _corpus_label() -> str:
    """How prompts should name the corpus (configuration, not code)."""
    try:
        from config import config as _config

        return _config.domain.corpus_label
    except Exception:  # configuration is optional for library use
        return "the corpus"


def answer_system_prompt() -> str:
    """The answering system prompt, with the corpus named from configuration."""
    return ANSWER_SYSTEM_PROMPT.format(corpus_label=_corpus_label())


def answer_prompt(question: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


# ── LLM adjudication (shared by all three pipelines) ──────────────────────
# The deterministic layer produces a *candidate* answer from structured evidence
# (infobox fields, graph counts, argmax over the candidate set). The LLM then
# adjudicates between that candidate and the retrieved passages. It may correct
# the candidate only when the passages contradict it, which keeps the LLM from
# inventing counts or names while still letting it fix linking mistakes.

ADJUDICATE_SYSTEM_PROMPT = """You verify answers for a QA system over {corpus_label}.

You receive: the question, a CANDIDATE answer produced by deterministic structured reasoning over a knowledge graph, and the retrieved CONTEXT passages.

Rules:
1. The candidate answer is derived from structured corpus fields (infobox values, exhaustive graph counts, argmax over an event set). Treat it as the default answer.
2. Keep the candidate if the context supports it or does not contradict it.
3. Replace it ONLY if the context clearly contradicts it (e.g. the candidate names the wrong event, the wrong edition, or the wrong field).
4. If the candidate is empty, answer from the context.
5. The answer must be a short verbatim span from the corpus: a number, a person's name, an event title, or a medal-winner name. Never explain, never hedge, never add units beyond what the corpus states.
6. If neither the candidate nor the context supports any answer, return an empty string.

Reply with JSON only:
{{"answer": "<final answer>", "agree": true|false, "reason": "<one short sentence>"}}"""


def adjudicate_system_prompt() -> str:
    """The adjudication prompt, with the corpus named from configuration."""
    return ADJUDICATE_SYSTEM_PROMPT.format(corpus_label=_corpus_label())


def refine_answer(llm, question: str, candidate: str,
                  context_items: List[Dict[str, Any]],
                  counter: Optional[TokenCounter] = None,
                  caller: str = "llm.refine",
                  qtype: str = "",
                  exhaustive: bool = False) -> Tuple[str, bool, Dict[str, Any]]:
    """Adjudicate a structured candidate answer against retrieved context.

    Returns ``(answer, changed, payload)``. When the LLM is unavailable the
    candidate is returned unchanged, so every pipeline keeps working offline.

    ``qtype``/``exhaustive`` gate the one case where adjudication is actively
    harmful: an answer computed *exhaustively* over the full candidate set
    (a count, a max/min over 60+ events) cannot be checked against the handful
    of passages that fit in the prompt. The model sees six documents, cannot
    see the other fifty-four, and will still return a confident different
    number. Asking it to adjudicate there is not verification - it is inviting
    a hallucination to overwrite arithmetic that is correct by construction.
    """
    candidate = (candidate or "").strip()
    if llm is None or not getattr(llm, "available", False):
        return candidate, False, {"reason": "llm_unavailable"}
    if not context_items and not candidate:
        return "", False, {"reason": "no_evidence"}

    # Never adjudicate an exhaustively-computed aggregate against a partial view.
    if candidate and (exhaustive or qtype in ("aggregation", "superlative")):
        return candidate, False, {
            "reason": "adjudication_skipped_exhaustive",
            "detail": (f"'{qtype or 'aggregate'}' answers are computed over the "
                       "complete candidate set; the prompt can only show a "
                       "subset, so the model cannot validly overrule them"),
            "candidate": candidate}

    # Keep the adjudication prompt small: on metered tiers the prompt size, not
    # the request count, is the binding constraint. 6 x ~450 chars is enough to
    # adjudicate a candidate answer against the corpus.
    context = render_context(context_items, max_chars_per_item=450, max_items=6)
    prompt = (f"Question: {question}\n\n"
              f"CANDIDATE answer: {candidate if candidate else '(none - deterministic '
                 'reasoning found no answer)'}\n\n"
              f"CONTEXT:\n{context}\n\n"
              "Return the JSON verdict now.")
    payload = llm.complete_json(prompt, adjudicate_system_prompt(),
                                caller=caller, counter=counter,
                                # Verification stays greedy even when the agent
                                # model explores: scoring must not wobble.
                                temperature=getattr(llm, "eval_temperature", None))
    final = str(payload.get("answer", "") or "").strip()
    agree = bool(payload.get("agree", False))

    # Guard rails: a verdict that fails to produce a usable short answer keeps
    # the deterministic candidate rather than degrading the pipeline.
    if not final or len(final) > 160:
        return candidate, False, {"reason": "verdict_unusable",
                                  "raw_answer": final[:120]}

    # A numeric candidate may only be replaced by another number. Small models
    # answer counting questions with prose ("There are several events..."),
    # which would otherwise destroy a correct integer.
    if candidate and _is_numeric(candidate) and not _is_numeric(final):
        return candidate, False, {"reason": "verdict_rejected_non_numeric",
                                  "raw_answer": final[:120],
                                  "candidate": candidate}

    changed = _norm(final) != _norm(candidate)

    # Grounding check. An adjudicator's job is to pick the right answer *out of
    # the context it was shown*; it is not licensed to invent a new one. So a
    # replacement is only accepted when it actually occurs in that context.
    # This is the single guard that matters most for a small local model: the
    # characteristic 4B failure is a fluent, confident, short answer with no
    # support in the passages, which is indistinguishable from a real
    # correction by any other test. A genuine correction is by definition
    # quoted from the evidence, so it passes untouched.
    if changed and candidate and not _grounded(final, context):
        return candidate, False, {"reason": "verdict_rejected_ungrounded",
                                  "detail": ("the proposed answer does not appear "
                                             "in the retrieved context"),
                                  "raw_answer": final[:120],
                                  "candidate": candidate}

    return final, changed, {"agree": agree, "reason": str(payload.get("reason", ""))[:200],
                            "candidate": candidate}


def _grounded(answer: str, context: str) -> bool:
    """True when the answer is supported by the context that was shown.

    Two ways to qualify, because answers come in two shapes:
      * a span ("Larisa Latynina") -> must appear verbatim after normalisation;
      * a number ("18")            -> must appear as a standalone token, so that
                                      "18" is not matched by "1980".
    Very short answers (<=2 chars) skip the check: they are too collision-prone
    for substring matching to mean anything either way.
    """
    ans, ctx = _norm(answer), _norm(context)
    if not ctx or len(ans) <= 2:
        return True
    if _is_numeric(ans):
        return bool(re.search(rf"(?<!\d){re.escape(ans.strip())}(?!\d)", ctx))
    return ans in ctx


def _is_numeric(text: str) -> bool:
    """True when the string is a bare number (the shape of a count answer)."""
    return bool(re.fullmatch(r"\s*-?\d+(?:[.,]\d+)?\s*", text or ""))


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split()).strip(" \t\"'.")