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
from retrieval.structured import (StructuredRetrieval, StructuredRetriever,
                                  choose_candidate, merge_context)
from utils.llm import (ANSWER_SYSTEM_PROMPT, answer_prompt, refine_answer,
                       render_context)
from utils.metrics import TokenCounter, count_tokens

from .base import PipelineResult, Timer, finalise_result
from .extractive import ExtractiveAnswerer

EMPTY_STRUCTURED = StructuredRetrieval()


class RagPipeline:
    """Single-shot vector retrieval + structure-aware retrieval + grounded answer.

    ``use_structured=False`` disables the structure-aware retrieval step and
    leaves the pipeline as pure vector top-k. That variant is published as the
    *ablation* arm: same extractor, same adjudicator, same prompt, one retrieval
    mechanism removed, so the difference between the two arms is exactly what
    the structured layer contributes.
    """

    name = "RAG"

    def __init__(self, index, llm=None, top_k: int = 5,
                 answerer: Optional[ExtractiveAnswerer] = None,
                 use_structured: bool = True, name: str = "") -> None:
        self.index = index
        self.llm = llm
        self.top_k = top_k
        self.answerer = answerer or ExtractiveAnswerer()
        self.use_structured = use_structured
        if name:
            self.name = name
        self.structured = StructuredRetriever(
            index.kg if index is not None else None, index)


    def run(self, question: str, qid: str = "") -> PipelineResult:
        started = time.perf_counter()
        counter = TokenCounter()
        timings: List[Dict[str, Any]] = []
        result = PipelineResult(pipeline=self.name, question=question,
                                qtype=classify(question))

        counter.add_text("prompt:question", input_text=question)
        spec = parse_question(question, self.index.kg if self.index else None)

        with Timer(timings, "vector_search", f"top_k={self.top_k}") as timer:
            chunks = self.index.vector_search(question, self.top_k)
        vector_doc_ids = [c["doc_id"] for c in chunks]

        # Structure-aware retrieval: the complete candidate set for counting /
        # argmax questions, and the resolved document for relational ones. This
        # is a *retrieval* step (same substrate, typed fields instead of
        # similarity); it contributes nothing when the slots do not resolve.
        structured = EMPTY_STRUCTURED
        if self.use_structured and self.structured.applies(spec):
            with Timer(timings, "structured_retrieve",
                       f"{spec.qtype} over typed infobox fields"):
                structured = self.structured.retrieve(question, spec)
        context = merge_context(chunks, structured.hits)

        result.retrieval_steps = 1 + (1 if structured.hits else 0)
        result.tools_called = ["similarity_search"] + (
            ["structured_retrieve"] if structured.hits else [])
        result.chunks_retrieved = len(context)
        result.docs_retrieved = len({c["doc_id"] for c in context})
        result.metadata = {
            "retriever": "vector",
            "retrieval_mode": ("vector+structured" if self.use_structured
                               else "vector_only (ablation)"),
            "top_k": self.top_k,
            "retrieved_doc_ids": vector_doc_ids,
            "retrieved_titles": [c["title"] for c in chunks],
            "structured_retrieval": structured.to_dict(),
            "structured_doc_ids": structured.doc_ids,
        }
        context_text = "\n".join(c.get("text", "") for c in context)
        counter.add_text("context:chunks", input_text=context_text,
                         detail=f"{len(context)} chunks ({len(structured.hits)} structured)")
        result.context_tokens = count_tokens(context_text)

        result.steps.append({
            "step": 1,
            "agent": "Retriever",
            "operation": "vector_search",
            "detail": f"top-{self.top_k} chunks by cosine similarity",
            "doc_ids": vector_doc_ids,
        })
        if structured.hits:
            result.steps.append({
                "step": 2,
                "agent": "StructureRetriever",
                "operation": "structured_retrieve",
                "detail": (f"{structured.method}: {structured.candidates} candidate "
                           f"events from typed infobox fields "
                           f"(complete={structured.complete})"),
                "doc_ids": structured.doc_ids,
            })

        # 1. deterministic candidate: read the answer out of the passages
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

        # 3. LLM adjudication: may correct a passage-derived candidate using the
        #    context. A structure-verified candidate is not overwritten - the
        #    disagreement is recorded instead, because a verified answer came
        #    from the complete candidate set rather than from k passages.
        if self.llm is not None and getattr(self.llm, "available", False):
            with Timer(timings, "llm_adjudicate", "verify candidate against passages"):
                final, changed, payload = refine_answer(
                    self.llm, question, choice.answer, context, counter,
                    caller="rag.adjudicate")
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

        result.plan = (["vector_search", "structured_retrieve", "answer"]
                       if structured.hits else ["vector_search", "answer"])
        result.loop_iterations = 1
        return finalise_result(result, counter, timings, started)

    @staticmethod
    def _stop_reason(choice, extracted) -> str:
        """Why the run stopped, from the candidate that won."""
        if choice.source == "structured":
            return "structured_verified" if choice.verified else "structured_resolved"
        return "resolved" if extracted.resolved else "insufficient_retrieved_evidence"



def _clean(text: str) -> str:
    """Normalise an LLM answer into a single comparable line."""
    if not text:
        return ""
    line = text.strip().split("\n")[0].strip()
    for prefix in ("Answer:", "answer:", "ANSWER:"):
        if line.startswith(prefix):
            line = line[len(prefix):].strip()
    return line.strip().strip('"').strip()