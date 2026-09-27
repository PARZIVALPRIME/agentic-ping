"""Explicit evidence structures and sufficiency gate for evidence-first answering.

In evidence-first answering, an answer is never accepted on prose confidence alone:
it must be substantiated by an explicit bundle of grounded evidence items
(graph vertices, edges, passage spans, accumulator values) that pass a typed
sufficiency gate before being finalized.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from reasoning.query_parser import QuerySpec


@dataclass
class EvidenceItem:
    """A discrete, verifiable unit of evidence grounded in the corpus or graph."""

    doc_id: str
    source_type: str       # "graph_vertex" | "graph_edge" | "passage" | "infobox" | "accumulator"
    field: str             # "gold" | "nations" | "competitors" | "venue" | "edition_chain" | "count"
    value: Any             # The raw extracted value or fact
    snippet: str = ""      # Verbatim text snippet or edge description
    grounded: bool = True  # True if verified against corpus/graph documents
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceBundle:
    """An assembled collection of evidence items supporting an answer."""

    items: List[EvidenceItem] = field(default_factory=list)
    sufficiency_verdict: bool = False
    confidence: float = 0.0
    uncertainty: float = 0.0
    gaps: List[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [it.to_dict() for it in self.items],
            "sufficiency_verdict": self.sufficiency_verdict,
            "confidence": round(self.confidence, 3),
            "uncertainty": round(self.uncertainty, 3),
            "gaps": list(self.gaps),
            "rationale": self.rationale,
            "num_items": len(self.items),
            "num_grounded": sum(1 for it in self.items if it.grounded),
        }


class SufficiencyGate:
    """Evaluates whether accumulated evidence satisfies the question's structural contract."""

    @staticmethod
    def audit_evidence(spec: QuerySpec,
                       kind: str,
                       candidate_answer: str,
                       citations: List[str],
                       state_documents: Dict[str, Any],
                       state_chunks: Dict[str, Any],
                       slots: Dict[str, Any],
                       min_confidence: float = 0.85) -> EvidenceBundle:
        """Run multi-faceted sufficiency verification on candidate answer and evidence."""
        items: List[EvidenceItem] = []
        gaps: List[str] = []
        clean_ans = (candidate_answer or "").strip()

        if not clean_ans:
            return EvidenceBundle(
                items=[],
                sufficiency_verdict=False,
                confidence=0.0,
                uncertainty=1.0,
                gaps=["empty_answer"],
                rationale="No candidate answer was produced.",
            )

        # 1. Grounding check on citations
        grounded_cites: List[str] = []
        unobserved_cites: List[str] = []
        for cite in citations:
            if cite in state_documents or cite in state_chunks:
                grounded_cites.append(cite)
            else:
                unobserved_cites.append(cite)

        if unobserved_cites:
            gaps.append(f"unobserved_citations:{len(unobserved_cites)}")

        # 2. Structural sufficiency per question kind
        confidence = 0.90
        rationale_parts = []

        if kind == "lookup":
            lookup_slot = slots.get("lookup") or {}
            matched = lookup_slot.get("matched")
            if matched:
                doc_id = getattr(matched, "doc_id", "")
                val = lookup_slot.get("value")
                items.append(EvidenceItem(
                    doc_id=doc_id,
                    source_type="infobox",
                    field="nations",
                    value=val,
                    snippet=f"{getattr(matched, 'title', '')}: {val} nations",
                    grounded=bool(doc_id in state_documents),
                    confidence=float(lookup_slot.get("score", 0.9)),
                ))
                rationale_parts.append(f"Lookup matched article {getattr(matched, 'title', '')}")
            else:
                gaps.append("unresolved_lookup_article")
                confidence = min(confidence, 0.50)

        elif kind in ("multi_hop", "temporal"):
            event_slot = slots.get("temporal_event") if kind == "temporal" else slots.get("venue_date")
            event_slot = event_slot or {}
            matched = event_slot.get("matched")
            if matched:
                doc_id = getattr(matched, "doc_id", "")
                gold = getattr(matched, "gold", "")
                items.append(EvidenceItem(
                    doc_id=doc_id,
                    source_type="graph_vertex",
                    field="gold",
                    value=gold,
                    snippet=f"Event {getattr(matched, 'title', '')} gold medallist: {gold}",
                    grounded=bool(doc_id in state_documents),
                    confidence=0.90,
                ))
                rationale_parts.append(f"Matched event {getattr(matched, 'title', '')} for {kind}")
            else:
                gaps.append(f"unresolved_{kind}_event")
                confidence = min(confidence, 0.50)

            if kind == "temporal":
                t_verify = slots.get("temporal_verify") or {}
                if not t_verify.get("consistent", True):
                    gaps.append("edition_chain_inconsistent")
                    confidence = min(confidence, 0.65)

        elif kind == "aggregation":
            agg_slot = slots.get("aggregation") or {}
            count = agg_slot.get("count")
            recount_v = slots.get("agg_verification") or {}
            recount = recount_v.get("recount")
            exhaustive = bool(agg_slot.get("exhaustive"))

            if count is not None:
                items.append(EvidenceItem(
                    doc_id="aggregation_accumulator",
                    source_type="accumulator",
                    field="count",
                    value=count,
                    snippet=f"Counted {count} events meeting threshold",
                    grounded=True,
                    confidence=0.92 if exhaustive else 0.75,
                ))
                if recount is not None and recount != count:
                    gaps.append("recount_mismatch")
                    confidence = min(confidence, 0.60)
                else:
                    rationale_parts.append(f"Count {count} verified with recount agreement")
            else:
                gaps.append("missing_aggregation_count")
                confidence = min(confidence, 0.40)

        elif kind == "superlative":
            comp_slot = slots.get("comparison") or {}
            winner = comp_slot.get("winner") or {}
            if winner.get("title"):
                doc_id = winner.get("doc_id", "")
                items.append(EvidenceItem(
                    doc_id=doc_id,
                    source_type="graph_vertex",
                    field="competitors",
                    value=winner.get("competitors"),
                    snippet=f"Winner {winner.get('title')} competitors: {winner.get('competitors')}",
                    grounded=bool(doc_id in state_documents),
                    confidence=0.92 if comp_slot.get("unique_winner") else 0.75,
                ))
                rationale_parts.append(f"Superlative extreme {winner.get('title')} verified")
            else:
                gaps.append("unresolved_superlative_winner")
                confidence = min(confidence, 0.40)

        # 3. Ground citations from chunks if not already added
        for cid in grounded_cites:
            if cid in state_chunks and not any(it.doc_id == cid for it in items):
                chunk = state_chunks[cid]
                items.append(EvidenceItem(
                    doc_id=cid,
                    source_type="passage",
                    field="text",
                    value=chunk.get("text", "")[:120],
                    snippet=chunk.get("text", "")[:160],
                    grounded=True,
                    confidence=0.88,
                ))

        sufficient = (confidence >= min_confidence) and (len(gaps) == 0) and bool(clean_ans)
        uncertainty = round(max(0.0, min(1.0, 1.0 - confidence)), 3)
        rationale = "; ".join(rationale_parts) or f"Evidence gathered across {len(items)} source items"

        return EvidenceBundle(
            items=items,
            sufficiency_verdict=sufficient,
            confidence=round(confidence, 3),
            uncertainty=uncertainty,
            gaps=gaps,
            rationale=rationale,
        )
