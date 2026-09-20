"""Structure-aware retrieval: exhaustive candidate sets from typed corpus fields.

The corpus is not only prose. Every event page carries a typed infobox (sport,
Games, competitors, nations, venue, dates, medals), and the pages are related
to each other (the previous edition of the same event, the other events at the
same venue). Vector retrieval flattens all of that into bag-of-words text, which
is why a top-k retriever structurally cannot answer two question families:

* **complete-set questions** - "how many X had more than N competitors",
  "which X had the most competitors" - the answer depends on *every* event of
  that sport at those Games, not on the k most similar passages;
* **relational questions** - "immediately before 2016", "held at V on D" - the
  evidence is a *different* document from the one the question lexically
  matches, so similarity is the wrong signal (measured: 100% of temporal gold
  documents are retrieved, yet only 36% are answered, because the retrieved
  page is the anchor edition, not the answer edition).

This module exposes the deterministic, exhaustive access that the agentic
pipeline's solvers already use, shaped like a *retrieval* result: canonical
infobox chunks that the shared extractor and the LLM adjudicator read exactly
like vector hits, plus the structured candidate and whether the candidate set
was complete.

Design rules
------------
* **Nothing is invented here.** Every value comes from the knowledge graph built
  from the corpus; this layer only selects, renders and checks completeness.
* **Retrieval-shaped output.** Hits use the same ``{doc_id, chunk_id, title,
  text, score, method}`` shape as :class:`retrieval.index.CorpusIndex`, so a
  pipeline merges them into its context without special cases.
* **Graceful degradation.** When the question's slots cannot be resolved, or the
  corpus has no structured page for them, the retrieval contributes nothing and
  the caller keeps its previous, passage-only behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from reasoning.query_parser import QuerySpec, parse_question
from reasoning.solvers import StructuredSolver, resolve_sport

from .index import infobox_summary

# Question types whose evidence is structural rather than lexical.
STRUCTURED_QTYPES = ("aggregation", "superlative", "temporal", "multi_hop", "lookup")

# Types where a *complete* candidate set is required for the answer to be
# trustworthy at all (counting / argmax over an enumerated population).
COMPLETE_SET_QTYPES = ("aggregation", "superlative")

# Types where the answer is a single resolved document: there the structured
# retrieval is verified when its resolution is confident enough. Lookup is the
# same mechanism over the article title, which the question names verbatim.
MIN_VERIFIED_CONFIDENCE = {"temporal": 0.60, "multi_hop": 0.72, "lookup": 0.85}

# Context budget: an exhaustive set can be large (a sport at a Games can hold
# ~50 events), so hits are capped. The *answer* is still computed over the whole
# set; only the rendered context is bounded, and the truncation is reported.
MAX_STRUCTURED_HITS = 24


@dataclass
class StructuredRetrieval:
    """Result of one structure-aware retrieval step."""

    qtype: str = ""
    hits: List[Dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    confidence: float = 0.0
    citations: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    method: str = "structured"
    candidates: int = 0
    complete: bool = False
    verified: bool = False
    hits_truncated: int = 0
    steps: List[Dict[str, Any]] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return bool(self.answer)

    @property
    def doc_ids(self) -> List[str]:
        return [h["doc_id"] for h in self.hits]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qtype": self.qtype,
            "method": self.method,
            "answer": self.answer,
            "confidence": round(self.confidence, 3),
            "candidates": self.candidates,
            "complete_set": self.complete,
            "verified": self.verified,
            "hits": len(self.hits),
            "hits_truncated": self.hits_truncated,
            "citations": list(self.citations),
            "unresolved": list(self.unresolved),
            "steps": list(self.steps),
        }


class StructuredRetriever:
    """Retrieval over the corpus' typed fields, backed by the graph solvers."""

    name = "StructureRetriever"

    def __init__(self, kg, index=None) -> None:
        self.kg = kg
        self.index = index
        self.solver = StructuredSolver(kg) if kg is not None else None

    # ── public API ─────────────────────────────────────────────────────
    def applies(self, spec: QuerySpec) -> bool:
        """Is this a question whose evidence the typed structure must supply?"""
        return self.solver is not None and spec.qtype in STRUCTURED_QTYPES

    def retrieve(self, question: str, spec: Optional[QuerySpec] = None,
                 max_hits: int = MAX_STRUCTURED_HITS) -> StructuredRetrieval:
        spec = spec or parse_question(question, self.kg)
        out = StructuredRetrieval(qtype=spec.qtype)
        if not self.applies(spec):
            return out

        solve = self.solver.solve(question, spec)
        out.method = f"structured:{solve.method or 'unsupported'}"
        out.answer = solve.answer or ""
        out.confidence = float(solve.confidence or 0.0)
        out.citations = [c for c in (solve.citations or []) if c]
        out.evidence = list(solve.evidence or [])
        out.candidates = int(solve.candidates_considered or 0)
        out.steps = list(solve.steps or [])
        out.unresolved = list(solve.unresolved or [])

        # Independent check that the candidate set really is the whole
        # population: re-enumerate it from the typed fields instead of trusting
        # the solver's own bookkeeping.
        population = self._population(spec)
        out.complete = self._is_complete(population)

        nodes = self._ordered_nodes(solve, population)
        hits = [h for h in (self._hit(n, spec, solve) for n in nodes) if h]
        if len(hits) > max_hits:
            out.hits_truncated = len(hits) - max_hits
            hits = hits[:max_hits]
        out.hits = hits
        out.verified = self._verified(spec, out)
        return out

    # ── internals ──────────────────────────────────────────────────────
    def _population(self, spec: QuerySpec) -> List[Any]:
        """The whole event population the question is quantified over."""
        if spec.qtype not in COMPLETE_SET_QTYPES:
            return []
        sport = resolve_sport(spec.sport or spec.event_desc, self.kg)
        if not sport or not spec.year or not spec.season:
            return []
        return list(self.kg.events_for(sport, spec.year, spec.season))

    @staticmethod
    def _is_complete(population: List[Any]) -> bool:
        """Complete = every event of the population carries the counted field."""
        if not population:
            return False
        return all(getattr(e, "competitors", None) is not None for e in population)

    def _ordered_nodes(self, solve, population: List[Any]) -> List[Any]:
        """Documents to render, most informative first.

        For counting/argmax questions the whole population is rendered (it *is*
        the evidence). Otherwise the solver's own evidence and citations define
        the set: the resolved event, plus the anchor edition for temporal
        questions.
        """
        ids: List[str] = []
        if population:
            ids.extend(e.doc_id for e in population)
        for item in (solve.evidence or []):
            if isinstance(item, dict) and item.get("doc_id"):
                ids.append(str(item["doc_id"]))
        ids.extend(c for c in (solve.citations or []) if c)

        nodes: List[Any] = []
        seen = set()
        for doc_id in ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            node = self.kg.events.get(doc_id)
            if node is not None:
                nodes.append(node)
        return nodes

    def _hit(self, node, spec: QuerySpec, solve) -> Optional[Dict[str, Any]]:
        """Render one event as a retrieval-shaped, canonical infobox chunk."""
        doc = self.index.get_doc(node.doc_id) if self.index is not None else None
        text = infobox_summary(doc) if doc is not None else self._render(node)
        if not text:
            return None
        relation = self._relation(node, spec, solve)
        return {
            "chunk_id": f"{node.doc_id}::structured",
            "doc_id": node.doc_id,
            "title": node.title,
            "text": text,
            "score": 0.99 if relation == "candidate_set" else 0.95,
            "method": f"structured:{relation}",
        }

    @staticmethod
    def _relation(node, spec: QuerySpec, solve) -> str:
        """Why this document is in the structured evidence set."""
        if spec.qtype in COMPLETE_SET_QTYPES:
            return "candidate_set"
        if spec.qtype == "temporal":
            return "answer_edition" if node.doc_id in (solve.citations or []) \
                else "anchor_edition"
        if spec.qtype == "multi_hop":
            return "venue_date_candidate"
        if spec.qtype == "lookup":
            return "title_resolution"
        return "structured"

    @staticmethod
    def _render(node) -> str:
        """Fallback rendering when the corpus index is not available."""
        fields = [
            ("sport", node.sport), ("year", node.year), ("season", node.season),
            ("competitors", node.competitors), ("nations", node.nations),
            ("venue", node.venue or node.venues),
            ("date", node.date_raw or node.dates_raw),
            ("gold", node.gold), ("silver", node.silver), ("bronze", node.bronze),
        ]
        body = " | ".join(f"{k}: {v}" for k, v in fields if v not in (None, ""))
        return f"Infobox of {node.title} -> {body}"

    @staticmethod
    def _verified(spec: QuerySpec, out: StructuredRetrieval) -> bool:
        """Can this structured answer be trusted over a passage-derived one?"""
        if not out.resolved:
            return False
        if spec.qtype in COMPLETE_SET_QTYPES:
            return out.complete
        threshold = MIN_VERIFIED_CONFIDENCE.get(spec.qtype)
        return bool(threshold is not None and out.confidence >= threshold)

# ── candidate reconciliation ───────────────────────────────────────────────

@dataclass
class CandidateChoice:
    """The candidate answer a pipeline will adjudicate, and where it came from."""

    answer: str = ""
    citations: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    method: str = ""
    source: str = "extractive"      # extractive | structured | none
    verified: bool = False
    note: str = ""


def choose_candidate(spec: QuerySpec, extracted,
                     structured: StructuredRetrieval) -> CandidateChoice:
    """Pick between the passage-derived candidate and the structured one.

    Policy (matched to the measured failure profile of the retrieval-only
    pipelines):

    * counting / argmax: the passage candidate is arithmetic over an incomplete
      set, so the exhaustive structured candidate always wins when available;
    * temporal / multi_hop: the structured candidate wins when its resolution is
      confident, because the passage candidate is usually the *anchor* document
      the question names rather than the document that holds the answer;
    * lookup: the structured candidate wins when the article title resolves
      exactly (the question names it verbatim); otherwise the passage candidate
      is kept and the structured one is the fallback.
    """
    passage = CandidateChoice(
        answer=(getattr(extracted, "answer", "") or "").strip(),
        citations=list(getattr(extracted, "citations", []) or []),
        evidence=list(getattr(extracted, "evidence", []) or []),
        confidence=float(getattr(extracted, "confidence", 0.0) or 0.0),
        method=getattr(extracted, "method", "extractive") or "extractive",
        source="extractive",
    )

    def _from_structured(note: str) -> CandidateChoice:
        return CandidateChoice(
            answer=structured.answer,
            citations=list(structured.citations),
            evidence=list(structured.evidence),
            confidence=structured.confidence,
            method=structured.method,
            source="structured",
            verified=structured.verified,
            note=note,
        )

    if not structured.resolved:
        passage.note = ("structured retrieval produced no candidate; "
                        "passage-only answer kept")
        return passage

    if spec.qtype in COMPLETE_SET_QTYPES:
        return _from_structured(
            f"exhaustive candidate set ({structured.candidates} events, "
            f"complete={structured.complete})")

    if spec.qtype in ("temporal", "multi_hop", "lookup"):
        if structured.verified:
            return _from_structured(
                f"structured resolution verified (confidence "
                f"{structured.confidence:.2f})")
        if not passage.answer:
            return _from_structured(
                "structured resolution below the verification threshold, but no "
                "passage candidate was available")
        passage.note = (f"passage-only answer kept: structured resolution "
                        f"confidence {structured.confidence:.2f} below threshold")
        return passage

    if passage.answer:
        passage.note = "passage-only answer kept (structured route not applicable)"
        return passage
    return _from_structured("structured fallback for an unresolved passage answer")


def merge_context(chunks: List[Dict[str, Any]],
                  hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Append structured hits to retrieved chunks, one entry per document.

    A structured hit replaces a passage for the same document: it carries the
    canonical infobox rendering, which is what the extractor and the adjudicator
    know how to read.
    """
    order: List[str] = []
    best: Dict[str, Dict[str, Any]] = {}
    for chunk in list(hits) + list(chunks or []):
        doc_id = chunk.get("doc_id") or chunk.get("chunk_id")
        if not doc_id:
            continue
        if doc_id not in best:
            best[doc_id] = chunk
            order.append(doc_id)
    return [best[d] for d in order]
