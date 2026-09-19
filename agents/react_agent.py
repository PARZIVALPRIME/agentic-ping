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
import time
from typing import Any, Dict, List, Optional

from utils.llm import LLMHelper

from .state import AgentState, ExecutedStep
from .tools import TOOL_SCHEMAS, GraphTools

SYSTEM_PROMPT = """You are an agent answering questions about Olympic events from \
a corpus of Wikipedia articles. The corpus is the ONLY source of truth; if the \
tools show nothing, say so instead of guessing.

Work iteratively: call tools, read the results, decide the next step. Do the \
reasoning YOURSELF - get_event_values returns raw values, and you count, compare \
or sort them.

Patterns:
- count/filter: search_events -> get_event_values -> count the matching entries. \
`num_matching_events` is the candidate-set size, not the answer.
- largest/most/fewest: get_event_values -> pick the extreme -> get_event_details \
on its doc_id.
- before/after: find the anchor event -> traverse_graph with PREV or NEXT -> read \
the year.
- who won what, where: search_events -> get_event_details or traverse_graph(WON_BY).
- prose evidence: search_passages.

Answer rules: `answer` is a SHORT verbatim span from the corpus (a number, a \
person's name, an event title). Cite the doc_ids that justify it. A count is \
answered with the number alone. Call submit_answer once, when you are confident.
"""

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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ]
        state.strategy_changes.append("strategy: LLM tool-calling loop (ReAct)")
        retries_left = self.max_retries

        for step in range(1, self.max_steps + 1):
            state.iterations = step
            t0 = time.perf_counter()
            before_in = state.tokens.input_tokens
            before_out = state.tokens.output_tokens
            _digest_history(messages)

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
                    state.answer = _extract_answer(text)
                state.stop_reason = "text_answer"
                break

            messages.append(_assistant_message(text, calls))

            submitted = False
            for call in calls:
                result = tools.execute(call["name"], call["arguments"])
                record = tools.calls[-1]
                messages.append({
                    "role": "tool", "tool_call_id": call["id"],
                    "content": _clip(json.dumps(result, default=str),
                                     MAX_TOOL_RESULT_CHARS)})
                state.trace.append(ExecutedStep(
                    index=len(state.trace) + 1, agent=self.name,
                    operation=f"tool:{call['name']}",
                    detail=", ".join(f"{k}={v}" for k, v in
                                     (call["arguments"] or {}).items())[:200],
                    observation={"tool": call["name"],
                                 "args": call["arguments"] or {},
                                 "summary": record["summary"],
                                 "result": _clip(json.dumps(result, default=str), 900)},
                    confidence_after=round(state.confidence, 3),
                    new_documents=_absorb(state, call["name"], result),
                    latency_ms=record["latency_ms"],
                    input_tokens=state.tokens.input_tokens - before_in,
                    output_tokens=state.tokens.output_tokens - before_out))
                if call["name"] == "submit_answer":
                    state.answer = str(result.get("answer", "")).strip()
                    state.rationale = str(result.get("reasoning", ""))[:600]
                    submitted = True

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
    return {
        "role": "assistant",
        "content": text or "",
        "tool_calls": [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"],
                          "arguments": json.dumps(c["arguments"] or {})}}
            for c in calls
        ],
    }


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
