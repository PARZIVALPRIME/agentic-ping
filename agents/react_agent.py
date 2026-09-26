"""LLM-driven tool-calling agent (ReAct over the knowledge graph).

This is the difference between *using* an LLM and *being driven by* one. The
deterministic pipeline decides a strategy in Python and then calls solvers; here
the model receives the question plus a set of graph tools and decides for itself:

    search_events -> get_event_values -> (reason over the raw values) -> submit_answer
    search_events -> traverse_graph(PREV) -> get_event_details      -> submit_answer

Nothing in this module knows what an "aggregation question" is. Counting,
argmax, temporal chaining and multi-hop linking all emerge from the model
choosing tools and combining their raw outputs. That is the point: generality
comes from the loop, not from question-type detection in Python.

The loop stops when the model calls ``submit_answer`` (explicit completion), when
it replies with a plain-text answer, when it exhausts the step budget, or when the
provider becomes unavailable - in which case the orchestrator falls back to the
deterministic path so the benchmark never loses a question.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from kg.textutil import normalize, tokens
from utils.llm import LLMHelper
from config import config as _app_config
from utils.thresholds import thresholds

from .state import AgentState, ExecutedStep
from .tools import TOOL_SCHEMAS, GraphTools

#: The agent's operating instructions. The corpus is named from configuration so
#: pointing the system at another document collection is a config change, not a
#: code change - nothing in the loop depends on the subject matter.
SYSTEM_PROMPT_TEMPLATE = """You are an agent answering questions about \
{entity_label}s in {corpus_label}. The corpus is the ONLY source of truth; if the \
tools show nothing, say so instead of guessing.

Work iteratively: call tools, read the results, decide the next step. Do the \
reasoning YOURSELF - get_event_values returns raw values, and you count, compare \
or sort them.

Patterns:
- count/filter: search_events -> get_event_values -> count the matching entries. \
`num_matching_events` is the candidate-set size, not the answer.
- largest/most/fewest (superlative): get_event_values -> pick the extreme -> get_event_details \
on its doc_id. If the question asks which event, athlete, sport, or nation had the most/least/highest, \
the answer is the entity's title/name, NOT the count integer.
- before/after: find the anchor event/edition -> traverse_graph with PREV or NEXT \
-> read the year.
- who won what, where: search_events -> get_event_details or traverse_graph(WON_BY).
- prose evidence: search_passages.
- disagreement: if two sources state different versions of the same fact (a \
record, a venue name, a medallist, a nation), call detect_conflicts with the \
versions and their doc_ids instead of picking one silently; the later statement, \
an explicit correction and a successor state outrank the older version.

Answer rules: `answer` is a SHORT verbatim span from the corpus (a number, a \
person's name, an event title). If asked for an entity/event ("which ..."), provide the entity name/title. \
If asked for a count/quantity ("how many ..."), provide the number alone. Cite the doc_ids that justify it. \
Call submit_answer once, when you are confident.
"""


def system_prompt() -> str:
    """The agent system prompt, with the corpus named from configuration."""
    domain = getattr(_app_config, "domain", None)
    return SYSTEM_PROMPT_TEMPLATE.format(
        corpus_label=getattr(domain, "corpus_label", "the corpus"),
        entity_label=getattr(domain, "entity_label", "corpus record"))


#: Backwards-compatible constant: probes and dashboards import this name.
SYSTEM_PROMPT = system_prompt()

MAX_TOOL_RESULT_CHARS = 1600
# Older tool payloads are digested before each turn: the model keeps the two
# most recent observations verbatim and does not re-pay for stale ones.
KEEP_VERBATIM = 2
DIGEST_CHARS = 340


class ReActAgent:
    """Tool-calling agent loop in which the model chooses every step."""

    name = "ReActAgent"
    version = "tool-calling/v1"

    def __init__(self, kg, index, llm: Optional[LLMHelper] = None,
                 cfg: Optional[Dict[str, Any]] = None) -> None:
        self.kg = kg
        self.index = index
        self.llm = llm
        self.cfg = cfg or {}
        self.max_steps = int(self.cfg.get("react_max_steps", 8) or 8)
        self.max_retries = int(self.cfg.get("react_retries", 2) or 2)
        self.model = self.cfg.get("react_model") or None

    @property
    def available(self) -> bool:
        return bool(self.llm is not None and getattr(self.llm, "available", False))

    def run(self, question: str, qid: str = "", spec=None,
            state: Optional[AgentState] = None) -> AgentState:
        """Investigate ``question`` with tool calls until the model submits."""
        state = state or AgentState(question, spec, qid)
        tools = GraphTools(self.kg, self.index)

        if not self.available:
            state.stop_reason = "provider_unavailable"
            return state

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": f"Question: {question}"},
        ]
        state.strategy_changes.append("strategy: LLM tool-calling loop (ReAct)")
        retries_left = self.max_retries
        # Everything the tools have returned, kept verbatim: this is the evidence
        # an answer is validated against (see validate_answer).
        evidence_blobs: List[str] = []
        rejections_left = 2

        for step in range(1, self.max_steps + 1):
            state.iterations = step
            t0 = time.perf_counter()
            before_in = state.tokens.input_tokens
            before_out = state.tokens.output_tokens
            _digest_history(messages)
            evidence_text = "\n".join(evidence_blobs)

            text, calls, finish = self.llm.chat(
                messages, tools=TOOL_SCHEMAS, counter=state.tokens,
                model=self.model, caller="agent.react")

            if not calls and not text:
                # The provider either failed or returned an unparseable tool
                # call (seen on Groq's gpt-oss when reasoning tokens run long).
                # One nudge usually recovers; only then give up and let the
                # orchestrator fall back to the deterministic path.
                if retries_left > 0:
                    retries_left -= 1
                    state.strategy_changes.append(
                        f"retry: unparseable model turn ({retries_left} left)")
                    state.trace.append(ExecutedStep(
                        index=len(state.trace) + 1, agent=self.name,
                        operation="retry",
                        detail="model turn had neither text nor tool calls",
                        observation={"stage": step, "retries_left": retries_left},
                        latency_ms=(time.perf_counter() - t0) * 1000))
                    messages.append({
                        "role": "user",
                        "content": "Your previous response could not be parsed. "
                                   "Reply with exactly one valid tool call.",
                    })
                    continue
                state.stop_reason = "provider_unavailable"
                break

            if not calls:
                # replied in prose instead of calling submit_answer
                state.trace.append(ExecutedStep(
                    index=len(state.trace) + 1, agent=self.name,
                    operation="final_answer",
                    detail=_clip(text, 200),
                    observation={"text": _clip(text, 400),
                                 "finish_reason": finish},
                    confidence_after=round(state.confidence, 3),
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    input_tokens=state.tokens.input_tokens - before_in,
                    output_tokens=state.tokens.output_tokens - before_out))
                if text and not state.answer:
                    # A prose reply means the model did not use submit_answer, so
                    # the answer is only kept if it passes the same schema and
                    # evidence checks the tool path runs - not because it looks
                    # like the right shape for the benchmark.
                    extracted = _extract_answer(text)
                    ok, why = validate_answer(extracted, spec, evidence_text)
                    if ok:
                        state.answer = extracted
                        state.trace[-1].observation["answer_accepted_because"] = why
                    else:
                        state.trace[-1].observation["rejected_answer"] = _clip(extracted, 200)
                        state.trace[-1].observation["rejected_reason"] = (
                            f"{why}; deferring to the deterministic planner")
                state.stop_reason = "text_answer"
                break

            messages.append(_assistant_message(text, calls))

            submitted = False
            for call in calls:
                result = tools.execute(call["name"], call["arguments"])
                record = tools.calls[-1]
                blob = json.dumps(result, default=str, ensure_ascii=False)
                evidence_blobs.append(blob)
                messages.append({
                    "role": "tool", "tool_call_id": call["id"],
                    "content": _clip(blob, MAX_TOOL_RESULT_CHARS)})
                state.trace.append(ExecutedStep(
                    index=len(state.trace) + 1, agent=self.name,
                    operation=f"tool:{call['name']}",
                    detail=", ".join(f"{k}={v}" for k, v in
                                     (call["arguments"] or {}).items())[:200],
                    observation={"tool": call["name"],
                                 "args": call["arguments"] or {},
                                 "summary": record["summary"],
                                 "result": _clip(json.dumps(result, default=str, ensure_ascii=False), 900)},
                    confidence_after=round(state.confidence, 3),
                    new_documents=_absorb(state, call["name"], result),
                    latency_ms=record["latency_ms"],
                    input_tokens=state.tokens.input_tokens - before_in,
                    output_tokens=state.tokens.output_tokens - before_out))
                if call["name"] == "submit_answer":
                    candidate = str(result.get("answer", "")).strip()
                    ok, why = validate_answer(candidate, spec, evidence_text)
                    if ok:
                        state.answer = candidate
                        state.rationale = str(result.get("reasoning", ""))[:600]
                        state.trace[-1].observation["answer_accepted_because"] = why
                        submitted = True
                    elif rejections_left > 0:
                        # Tell the model why it was refused and let it try again.
                        # The loop's own evidence is the judge, so this is a
                        # correction round rather than a failed investigation.
                        rejections_left -= 1
                        state.trace[-1].observation["rejected_answer"] = _clip(candidate, 200)
                        state.trace[-1].observation["rejected_reason"] = why
                        messages.append({
                            "role": "user",
                            "content": ("Your submitted answer was not accepted: "
                                        f"{why}. Re-read the tool results and call "
                                        "submit_answer again with an answer the "
                                        "evidence supports."),
                        })
                    else:
                        state.trace[-1].observation["rejected_answer"] = _clip(candidate, 200)
                        state.trace[-1].observation["rejected_reason"] = why
                        state.strategy_changes.append(
                            f"react: dropped an unvalidated submission ({why})")

            if submitted:
                state.stop_reason = "submitted_answer"
                break
        else:
            state.stop_reason = "max_steps"

        state.tool_calls = list(tools.calls)
        state.synth_source = "react_tool_calling"
        if not state.stop_reason:
            state.stop_reason = "max_steps"
        state.confidence = {"submitted_answer": 0.9, "text_answer": 0.6}.get(
            state.stop_reason, 0.3 if state.answer else 0.0)
        return state


def _digest_history(messages: List[Dict[str, Any]]) -> None:
    """Shrink all but the most recent tool results in place.

    The full payloads stay in the run trace; only the wire conversation is
    compacted, so token cost per step stops growing with the investigation
    length while the model still sees what it just learned.
    """
    positions = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    for idx in positions[:-KEEP_VERBATIM] if KEEP_VERBATIM else positions:
        content = str(messages[idx].get("content") or "")
        if len(content) > DIGEST_CHARS:
            messages[idx]["content"] = content[:DIGEST_CHARS] + " ...(trimmed)"


def _assistant_message(text: str, calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Rebuild the assistant turn in OpenAI wire format so the model sees its
    own previous tool calls (required for multi-turn function calling)."""
    tool_calls = []
    for c in calls:
        tc_dict: Dict[str, Any] = {
            "id": c["id"],
            "type": "function",
            "function": {
                "name": c["name"],
                "arguments": c.get("raw") or json.dumps(c.get("arguments") or {}),
            },
        }
        if c.get("extra_content"):
            tc_dict["extra_content"] = c["extra_content"]
        tool_calls.append(tc_dict)
    msg: Dict[str, Any] = {
        "role": "assistant",
        "tool_calls": tool_calls,
    }
    if text:
        msg["content"] = text
    return msg


def _absorb(state: AgentState, tool: str, result: Any) -> int:
    """Fold a tool result into the shared blackboard. Returns new doc count."""
    if not isinstance(result, dict):
        return 0
    doc_ids = [d for d in (result.get("doc_ids") or []) if d]
    new = state.add_documents(doc_ids)
    if tool == "search_events":
        state.add_chunks(result.get("events") or [])
    elif tool == "search_passages":
        state.add_chunks(result.get("passages") or [])
    elif tool in ("get_event_details", "traverse_graph", "get_games_events"):
        state.add_chunks(result.get("events") or [])
    if tool == "get_event_values":
        state.slots.setdefault("value_sets", []).append({
            "field": result.get("field"),
            "num_with_value": result.get("num_with_value"),
            "values": result.get("values") or [],
        })
    for doc_id in doc_ids:
        if doc_id not in state.citations:
            state.citations.append(doc_id)
    return new


def _clip(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def _extract_answer(text: str) -> str:
    """Best-effort short answer from a prose reply (used only as a fallback)."""
    import re

    match = re.search(r"(?im)^\s*(?:final\s+)?answer\s*[:\-]\s*(.+)$", text)
    if match:
        return match.group(1).strip().strip('"').strip("*")[:200]
    last = [line.strip() for line in text.splitlines() if line.strip()]
    return last[-1][:200] if last else ""


# ── answer validation ──────────────────────────────────────────────────────
#
# An answer is accepted on two grounds and no others:
#
#   * SCHEMA   - the shape the question demands. "How many ..." has a numeric
#                answer because the corpus field behind it is numeric; "which
#                event ..." has an entity as its answer. This is a property of
#                the question's semantics, not of any benchmark's formatting.
#   * EVIDENCE - a span answer must occur in the tool output that produced it.
#                An answer the model cannot point at in its own evidence is a
#                guess, however fluent.
#
# This replaces an earlier heuristic that asserted what benchmark golds look
# like ("Benchmark golds are spans and numbers", ">14 words is a sentence about
# a span") plus a hedge-phrase blocklist. Those were statements about the
# evaluation set, and they fail exactly where they are most confident: a
# fluent, confident, wrong answer has the same shape as a right one.

_EXPECTED_SHAPE = {"aggregation": "number", "superlative": "entity"}
_NUMBER_RE = re.compile(r"^\s*-?\d[\d,]*(?:\.\d+)?\s*$")


def _is_number(text: str) -> bool:
    """True when the string is a bare number (the shape of a count)."""
    return bool(_NUMBER_RE.match(text or ""))


def _grounded_in_evidence(answer: str, evidence: str, overlap: float = 1.0) -> bool:
    """True when the answer occurs in, or is fully covered by, the evidence."""
    if not evidence or not normalize(answer):
        return False
    norm_ans = normalize(answer)
    norm_ev = normalize(evidence)
    if norm_ans in norm_ev:
        return True
    try:
        decoded_ev = normalize(evidence.encode("utf-8").decode("unicode_escape"))
        if norm_ans in decoded_ev:
            return True
    except Exception:
        pass
    wanted = {t for t in tokens(answer) if t}
    if not wanted:
        return False
    have = set(tokens(evidence))
    return len(wanted & have) / len(wanted) >= overlap


def validate_answer(answer: str, spec: Any, evidence: str) -> Tuple[bool, str]:
    """Accept or reject an answer on schema and evidence grounds.

    Returns ``(ok, why)``. The reason is written into the run trace, so a
    rejected answer is visible rather than silently swapped.
    """
    text = (answer or "").strip()
    if not text:
        return False, "empty answer"
    th = thresholds()
    if len(text) > th.answer_max_chars:
        return False, (f"longer than {th.answer_max_chars} characters: no corpus "
                       "field produces a paragraph")
    numeric = _is_number(text)
    shape = _EXPECTED_SHAPE.get(str(getattr(spec, "qtype", "") or ""), "")
    if shape == "number" and not numeric:
        return False, "this question asks for a count, so the answer must be a number"
    if shape == "entity" and numeric:
        return False, "this question asks which event/person, so the answer must name it"
    if numeric:
        # A count is *derived* by the model from the value lists it was shown, so
        # it cannot be re-found verbatim in the evidence; a span must be.
        return True, "schema: numeric answer"
    if len(text) <= th.grounding_min_chars:
        return True, "too short to check against evidence"
    if _grounded_in_evidence(text, evidence, th.grounding_token_overlap):
        return True, "evidence: present in tool output"
    return False, "the answer does not appear in any tool output it was shown"
