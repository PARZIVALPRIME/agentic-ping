"""Synthesizer agent: produce the final grounded answer with citations.

The agentic pipeline's answers are *derived from structured evidence* (graph
fields the agents read directly), so the deterministic renderer is primary:
it cannot hallucinate a count or a name. An LLM is used only as a last-resort
fallback when no structured answer could be derived, answering strictly over
the retrieved passages - mirroring the RAG/GraphRAG generation contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from reasoning.query_parser import QuerySpec
from utils.metrics import TokenCounter


@dataclass
class SynthResult:
    answer: str = ""
    citations: List[str] = field(default_factory=list)
    rationale: str = ""
    source: str = "structured"   # structured | llm_fallback | none
    confidence: float = 0.0


class Synthesizer:
    """Renders structured evidence into a concise, comparable answer."""

    def __init__(self, llm=None) -> None:
        self.llm = llm

    # ── deterministic rendering ────────────────────────────────────────
    # Each question type owns one slot on the blackboard; the renderer reads
    # exactly that slot, so a structured answer can only come from the agent
    # that produced it.
    SLOTS = {
        "lookup": "lookup",
        "multi_hop": "venue_date",
        "temporal": "temporal_event",
        "aggregation": "aggregation",
        "superlative": "comparison",
    }

    def render(self, spec: QuerySpec, kind: str, slots: Dict[str, Any]) -> SynthResult:
        renderer = {
            "lookup": self._render_lookup,
            "multi_hop": self._render_medal,
            "temporal": self._render_medal,
            "aggregation": self._render_count,
            "superlative": self._render_superlative,
        }.get(kind)
        if renderer is None:
            return SynthResult(source="none", rationale=f"unsupported kind '{kind}'")
        slot = slots.get(self.SLOTS[kind]) or {}
        return renderer(spec, slot)

    def _render_lookup(self, spec: QuerySpec, slot: Dict[str, Any]) -> SynthResult:
        node = slot.get("matched")
        value = slot.get("value")
        if node is None or value in (None, ""):
            return SynthResult(source="none",
                               rationale="target article or field not resolved")
        return SynthResult(
            answer=str(value),
            citations=[node.doc_id],
            rationale=f"{node.title} lists {value} participating nations "
                      f"(infobox field 'nations')",
            confidence=0.9 if slot.get("score", 0) >= 0.9 else 0.75,
        )

    def _render_medal(self, spec: QuerySpec, slot: Dict[str, Any]) -> SynthResult:
        node = slot.get("matched")
        gold = getattr(node, "gold", "") if node is not None else ""
        if node is None:
            return SynthResult(source="none", rationale="no event matched")
        if not gold:
            return SynthResult(source="none",
                               rationale="matched event has no gold field in corpus")
        citations = [node.doc_id]
        next_doc = getattr(node, "next_doc_id", None)
        if spec.qtype == "temporal" and next_doc:
            citations.append(next_doc)  # anchor edition confirms the chain
        return SynthResult(
            answer=gold,
            citations=citations,
            rationale=f"gold medallist read from '{node.title}' "
                      f"({node.games_key} Olympics)",
            confidence=0.88,
        )

    def _render_count(self, spec: QuerySpec, slot: Dict[str, Any]) -> SynthResult:
        count = slot.get("count")
        if count is None:
            return SynthResult(source="none", rationale="aggregation produced no count")
        op = {"gt": "more than", "gte": "at least", "lt": "fewer than",
              "lte": "at most"}.get(slot.get("comparator", "gt"), "more than")
        return SynthResult(
            answer=str(count),
            citations=[k["doc_id"] for k in slot.get("kept", [])],
            rationale=f"{count} of {slot.get('candidates', 0)} enumerated events "
                      f"had {op} {slot.get('threshold')} competitors",
            confidence=0.9 if slot.get("exhaustive") else 0.7,
        )

    def _render_superlative(self, spec: QuerySpec, slot: Dict[str, Any]) -> SynthResult:
        winner = slot.get("winner") or {}
        title = winner.get("title")
        if not title:
            return SynthResult(source="none", rationale="no winner determined")
        direction = "highest" if slot.get("direction") == "max" else "lowest"
        return SynthResult(
            answer=title,
            citations=[winner.get("doc_id", "")],
            rationale=f"'{title}' had the {direction} competitor count "
                      f"({winner.get('competitors')}) among "
                      f"{slot.get('candidates', 0)} enumerated events",
            confidence=0.9 if slot.get("unique_winner") else 0.7,
        )

    # ── LLM fallback (only when structure failed) ──────────────────────
    def llm_fallback(self, question: str, chunks: List[Dict[str, Any]],
                     counter: Optional[TokenCounter] = None) -> Tuple[str, List[str]]:
        if not self.llm or not getattr(self.llm, "available", False) or not chunks:
            return "", []
        from utils.llm import answer_prompt, answer_system_prompt, render_context

        context = render_context(chunks)
        text, _i, _o = self.llm.complete(answer_prompt(question, context),
                                         answer_system_prompt(),
                                         caller="agent.synthesize", counter=counter)
        line = (text or "").strip().split("\n")[0].strip()
        for prefix in ("Answer:", "answer:", "ANSWER:"):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        line = line.strip().strip('"').strip()
        ids = list(dict.fromkeys(c.get("doc_id", "") for c in chunks[:8])) if line else []
        return line, [i for i in ids if i]
