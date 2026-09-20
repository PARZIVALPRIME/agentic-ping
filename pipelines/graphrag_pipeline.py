"""Pipeline 2 - GraphRAG (hybrid retrieval + knowledge-graph expansion).

Single-shot like RAG, but the retrieved seed passages are expanded along the
knowledge graph: previous/next editions of the same event, other events at the
same venue, and the other events of that sport at the same Games. This is the
classic "vector + graph" retrieval recipe, and it is the cheapest way to answer
questions whose evidence is a *neighbour* of what the question literally says.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from reasoning.query_parser import classify, parse_question
from retrieval.structured import (StructuredRetrieval, StructuredRetriever,
                                  choose_candidate, merge_context)
from utils.llm import (ANSWER_SYSTEM_PROMPT, answer_prompt, refine_answer,
                       render_context)
from utils.metrics import TokenCounter, count_tokens

from .base import PipelineResult, Timer, finalise_result
from .extractive import ExtractiveAnswerer

EMPTY_STRUCTURED = StructuredRetrieval()


class GraphRagPipeline:
    """Hybrid retrieval, graph-neighbourhood expansion, structure-aware
    retrieval, then answer."""

    name = "GraphRAG"

    def __init__(self, index, llm=None, top_k: int = 10, num_hops: int = 2,
                 answerer: Optional[ExtractiveAnswerer] = None) -> None:
        self.index = index
        self.llm = llm
        self.top_k = top_k
        self.num_hops = num_hops
        self.answerer = answerer or ExtractiveAnswerer()
        self.structured = StructuredRetriever(
            index.kg if index is not None else None, index)


    def run(self, question: str, qid: str = "") -> PipelineResult:
        started = time.perf_counter()
        counter = TokenCounter()
        timings: List[Dict[str, Any]] = []
        result = PipelineResult(pipeline=self.name, question=question,
                                qtype=classify(question))
        counter.add_text("prompt:question", input_text=question)

        with Timer(timings, "hybrid_search", "lexical + vector RRF") as timer:
            seeds = self.index.hybrid_search(question, max(3, self.top_k // 2))
        with Timer(timings, "graph_expand", f"{self.num_hops}-hop neighbourhood") as timer:
            expanded = self.index.graph_search(question, self.top_k, self.num_hops)

        spec = parse_question(question, self.index.kg if self.index else None)

        # Structure-aware retrieval: graph expansion reaches *neighbouring*
        # documents, which is still a similarity-shaped guess about which
        # neighbours matter. The typed fields say exactly which documents form
        # the candidate set (counting/argmax), or which single document is the
        # resolved answer (temporal chaining, venue+date linking).
        structured = EMPTY_STRUCTURED
        if self.structured.applies(spec):
            with Timer(timings, "structured_retrieve",
                       f"{spec.qtype} over typed infobox fields"):
                structured = self.structured.retrieve(question, spec)
        context = merge_context(expanded, structured.hits)

        result.retrieval_steps = 2 + (1 if structured.hits else 0)
        result.tools_called = (["hybrid_search", "structural_retrieve"]
                               + (["structured_retrieve"] if structured.hits else []))
        result.chunks_retrieved = len(context)
        result.docs_retrieved = len({c["doc_id"] for c in context})
        result.metadata = {
            "retriever": "hybrid+graph",
            "num_hops": self.num_hops,
            "seed_doc_ids": [c["doc_id"] for c in seeds],
            "expanded_doc_ids": [c["doc_id"] for c in expanded],
            "graph_relations": sorted({c.get("method", "") for c in expanded}),
            "retrieved_titles": [c["title"] for c in context],
            "structured_retrieval": structured.to_dict(),
            "structured_doc_ids": structured.doc_ids,
        }
        context_text = "\n".join(c.get("text", "") for c in context)
        counter.add_text("context:graph_expanded", input_text=context_text,
                         detail=f"{len(context)} documents "
                                f"({len(structured.hits)} structured)")
        result.context_tokens = count_tokens(context_text)

        result.steps.append({
            "step": 1, "agent": "HybridRetriever", "operation": "hybrid_search",
            "detail": f"RRF fusion, {len(seeds)} seed documents",
            "doc_ids": [c["doc_id"] for c in seeds],
        })
        result.steps.append({
            "step": 2, "agent": "GraphTraverser", "operation": "structural_retrieve",
            "detail": f"{self.num_hops}-hop expansion over PREV/NEXT, SAME_VENUE, "
                      f"SAME_SPORT_GAMES",
            "doc_ids": [c["doc_id"] for c in expanded],
            "new_docs": len({c["doc_id"] for c in expanded}
                            - {c["doc_id"] for c in seeds}),
        })
        if structured.hits:
            result.steps.append({
                "step": 3, "agent": "StructureRetriever",
                "operation": "structured_retrieve",
                "detail": (f"{structured.method}: {structured.candidates} candidate "
                           f"events from typed infobox fields "
                           f"(complete={structured.complete})"),
                "doc_ids": structured.doc_ids,
            })

        # 1. deterministic candidate read out of the retrieved document set
        with Timer(timings, "extractive_answer", "read infobox fields from passages"):
            extracted = self.answerer.answer(question, context, spec)

        # 2. reconcile the passage candidate with the structured one
        choice = choose_candidate(spec, extracted, structured)
        result.answer = choice.answer
        result.citations = choice.citations
        result.confidence = choice.confidence
        result.evidence = choice.evidence
        result.method = choice.method
        result.unresolved = (list(structured.unresolved) if choice.source == "structured"
                             else list(extracted.unresolved))
        result.metadata["candidate_source"] = choice.source
        result.metadata["candidate_verified"] = choice.verified
        result.metadata["candidate_note"] = choice.note
        result.metadata["docs_with_infobox"] = extracted.fields_parsed
        result.stop_reason = self._stop_reason(choice, extracted)

        # 3. LLM adjudication over the merged (graph + structured) context. A
        #    structure-verified candidate is not overwritten; the disagreement is
        #    recorded instead.
        if self.llm is not None and getattr(self.llm, "available", False):
            with Timer(timings, "llm_adjudicate", "verify candidate against graph context"):
                final, changed, payload = refine_answer(
                    self.llm, question, choice.answer, context, counter,
                    caller="graphrag.adjudicate")
            result.llm_calls = 1
            payload["candidate_source"] = choice.source
            payload["overridden"] = bool(changed and choice.verified)
            if final and not payload["overridden"]:
                result.answer = _clean(final)
            elif payload["overridden"]:
                payload["reason"] = ("structure-verified candidate kept; "
                                     f"adjudicator proposed {_clean(final)!r}")
            result.metadata["adjudication"] = payload
            result.metadata["llm_revised"] = bool(changed and not payload["overridden"])
            if changed and not payload["overridden"]:
                result.confidence = min(0.9, result.confidence + 0.05)
                result.stop_reason = "llm_revised_extraction"
            if not result.citations:
                result.citations = list(dict.fromkeys(c["doc_id"] for c in context))

        result.plan = (["hybrid_search", "graph_expand", "structured_retrieve", "answer"]
                       if structured.hits
                       else ["hybrid_search", "graph_expand", "answer"])
        result.loop_iterations = 1
        return finalise_result(result, counter, timings, started)

    @staticmethod
    def _stop_reason(choice, extracted) -> str:
        """Why the run stopped, from the candidate that won."""
        if choice.source == "structured":
            return "structured_verified" if choice.verified else "structured_resolved"
        return "resolved" if extracted.resolved else "insufficient_retrieved_evidence"



def _clean(text: str) -> str:
    if not text:
        return ""
    line = text.strip().split("\n")[0].strip()
    for prefix in ("Answer:", "answer:", "ANSWER:"):
        if line.startswith(prefix):
            line = line[len(prefix):].strip()
    return line.strip().strip('"').strip()