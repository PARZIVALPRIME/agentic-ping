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
from utils.llm import (ANSWER_SYSTEM_PROMPT, answer_prompt, refine_answer,
                       render_context)
from utils.metrics import TokenCounter, count_tokens

from .base import PipelineResult, Timer, finalise_result
from .extractive import ExtractiveAnswerer


class GraphRagPipeline:
    """Hybrid retrieval, then graph-neighbourhood expansion, then answer."""

    name = "GraphRAG"

    def __init__(self, index, llm=None, top_k: int = 10, num_hops: int = 2,
                 answerer: Optional[ExtractiveAnswerer] = None) -> None:
        self.index = index
        self.llm = llm
        self.top_k = top_k
        self.num_hops = num_hops
        self.answerer = answerer or ExtractiveAnswerer()

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

        result.retrieval_steps = 2
        result.tools_called = ["hybrid_search", "structural_retrieve"]
        result.chunks_retrieved = len(expanded)
        result.docs_retrieved = len({c["doc_id"] for c in expanded})
        result.metadata = {
            "retriever": "hybrid+graph",
            "num_hops": self.num_hops,
            "seed_doc_ids": [c["doc_id"] for c in seeds],
            "expanded_doc_ids": [c["doc_id"] for c in expanded],
            "graph_relations": sorted({c.get("method", "") for c in expanded}),
            "retrieved_titles": [c["title"] for c in expanded],
        }
        context_text = "\n".join(c.get("text", "") for c in expanded)
        counter.add_text("context:graph_expanded", input_text=context_text,
                         detail=f"{len(expanded)} documents")
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

        spec = parse_question(question, self.index.kg if self.index else None)

        # 1. deterministic candidate read out of the expanded document set
        with Timer(timings, "extractive_answer", "read infobox fields from passages"):
            extracted = self.answerer.answer(question, expanded, spec)
        result.answer = extracted.answer
        result.citations = extracted.citations
        result.confidence = extracted.confidence
        result.evidence = extracted.evidence
        result.unresolved = extracted.unresolved
        result.method = extracted.method
        result.metadata["docs_with_infobox"] = extracted.fields_parsed
        result.stop_reason = ("resolved" if extracted.resolved
                              else "insufficient_retrieved_evidence")

        # 2. LLM adjudication over the graph-expanded context
        if self.llm is not None and getattr(self.llm, "available", False):
            with Timer(timings, "llm_adjudicate", "verify candidate against graph context"):
                final, changed, payload = refine_answer(
                    self.llm, question, extracted.answer, expanded, counter,
                    caller="graphrag.adjudicate")
            result.llm_calls = 1
            if final:
                result.answer = _clean(final)
            result.metadata["adjudication"] = payload
            result.metadata["llm_revised"] = changed
            if changed:
                result.confidence = min(0.9, result.confidence + 0.05)
                result.stop_reason = "llm_revised_extraction"
            if not result.citations:
                result.citations = list(dict.fromkeys(c["doc_id"] for c in expanded))

        result.plan = ["hybrid_search", "graph_expand", "answer"]
        result.loop_iterations = 1
        return finalise_result(result, counter, timings, started)


def _clean(text: str) -> str:
    if not text:
        return ""
    line = text.strip().split("\n")[0].strip()
    for prefix in ("Answer:", "answer:", "ANSWER:"):
        if line.startswith(prefix):
            line = line[len(prefix):].strip()
    return line.strip().strip('"').strip()