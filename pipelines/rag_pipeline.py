"""Pipeline 1 - classic vector RAG.

The baseline: embed the question, take the top-k most similar chunks, and answer
from those passages in a single shot. No graph traversal, no iteration, no
knowledge of which documents should exist. This is what the other two pipelines
must beat to justify their cost.
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


class RagPipeline:
    """Single-shot vector retrieval + grounded answer generation."""

    name = "RAG"

    def __init__(self, index, llm=None, top_k: int = 5,
                 answerer: Optional[ExtractiveAnswerer] = None) -> None:
        self.index = index
        self.llm = llm
        self.top_k = top_k
        self.answerer = answerer or ExtractiveAnswerer()

    def run(self, question: str, qid: str = "") -> PipelineResult:
        started = time.perf_counter()
        counter = TokenCounter()
        timings: List[Dict[str, Any]] = []
        result = PipelineResult(pipeline=self.name, question=question,
                                qtype=classify(question))

        counter.add_text("prompt:question", input_text=question)

        with Timer(timings, "vector_search", f"top_k={self.top_k}") as timer:
            chunks = self.index.vector_search(question, self.top_k)
        result.retrieval_steps = 1
        result.tools_called = ["similarity_search"]
        result.chunks_retrieved = len(chunks)
        result.docs_retrieved = len({c["doc_id"] for c in chunks})
        result.metadata = {
            "retriever": "vector",
            "top_k": self.top_k,
            "retrieved_doc_ids": [c["doc_id"] for c in chunks],
            "retrieved_titles": [c["title"] for c in chunks],
        }
        counter.add_text("context:chunks", input_text="\n".join(c["text"] for c in chunks),
                         detail=f"{len(chunks)} chunks")
        result.context_tokens = count_tokens("\n".join(c["text"] for c in chunks))

        result.steps.append({
            "step": 1,
            "agent": "Retriever",
            "operation": "vector_search",
            "detail": f"top-{self.top_k} chunks by cosine similarity",
            "doc_ids": [c["doc_id"] for c in chunks],
        })

        spec = parse_question(question, self.index.kg if self.index else None)

        # 1. deterministic candidate: read the answer out of the passages
        with Timer(timings, "extractive_answer", "read infobox fields from passages"):
            extracted = self.answerer.answer(question, chunks, spec)
        result.answer = extracted.answer
        result.citations = extracted.citations
        result.confidence = extracted.confidence
        result.evidence = extracted.evidence
        result.unresolved = extracted.unresolved
        result.method = extracted.method
        result.metadata["extraction_method"] = extracted.method
        result.metadata["docs_with_infobox"] = extracted.fields_parsed
        result.stop_reason = ("resolved" if extracted.resolved
                              else "insufficient_retrieved_evidence")

        # 2. LLM adjudication: may correct the candidate using the passages
        if self.llm is not None and getattr(self.llm, "available", False):
            with Timer(timings, "llm_adjudicate", "verify candidate against passages"):
                final, changed, payload = refine_answer(
                    self.llm, question, extracted.answer, chunks, counter,
                    caller="rag.adjudicate")
            result.llm_calls = 1
            if final:
                result.answer = _clean(final)
            result.metadata["adjudication"] = payload
            result.metadata["llm_revised"] = changed
            if changed:
                result.confidence = min(0.9, result.confidence + 0.05)
                result.stop_reason = "llm_revised_extraction"
            if not result.citations:
                result.citations = list(dict.fromkeys(c["doc_id"] for c in chunks))

        result.plan = ["vector_search", "answer"]
        result.loop_iterations = 1
        return finalise_result(result, counter, timings, started)


def _clean(text: str) -> str:
    """Normalise an LLM answer into a single comparable line."""
    if not text:
        return ""
    line = text.strip().split("\n")[0].strip()
    for prefix in ("Answer:", "answer:", "ANSWER:"):
        if line.startswith(prefix):
            line = line[len(prefix):].strip()
    return line.strip().strip('"').strip()