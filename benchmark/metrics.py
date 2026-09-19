"""Metrics collection: turns per-question pipeline records into aggregates.

The collector is intentionally dumb: it accumulates the per-question entries
the runner produces (one entry per question, one record per pipeline) and
computes the summary tables the dashboard renders:

- accuracy overall and per question type, per pipeline
- cost: tokens (context/input/output/total), latency, LLM calls
- process: retrieval steps, chunks, agents invoked, stop reasons
- grounding: citation precision/recall against gold_doc_ids
- the "when agents matter" delta: per-question Agentic-vs-GraphRAG outcome
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

QTYPES = ("lookup", "multi_hop", "temporal", "aggregation", "superlative")


def _mean(values: List[float]) -> Optional[float]:
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 4) if nums else None


class MetricsCollector:
    """Accumulates per-question records and computes aggregate summaries."""

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def add(self, entry: Dict[str, Any]) -> None:
        self.entries.append(entry)

    # ── aggregation ────────────────────────────────────────────────────
    def summarize(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        pipeline_names: List[str] = []
        for entry in self.entries:
            for name in (entry.get("pipelines") or {}):
                if name not in pipeline_names:
                    pipeline_names.append(name)

        summary: Dict[str, Any] = {
            "num_questions": len(self.entries),
            "pipelines": {name: self._pipeline_summary(name)
                          for name in pipeline_names},
            "when_agents_matter": self._delta_table(),
        }
        if extra:
            summary.update(extra)
        return summary

    def _pipeline_summary(self, name: str) -> Dict[str, Any]:
        recs = [e["pipelines"][name] for e in self.entries
                if name in (e.get("pipelines") or {})]
        if not recs:
            return {}
        evaluated = [r for r in recs if r.get("evaluation")]

        out: Dict[str, Any] = {
            "num_questions": len(recs),
            "num_evaluated": len(evaluated),
            "correct": sum(1 for r in evaluated if r["evaluation"]["is_correct"]),
        }
        out["accuracy"] = round(out["correct"] / len(evaluated), 4) \
            if evaluated else None

        # per question-type accuracy
        by_type: Dict[str, Any] = {}
        for qtype in QTYPES:
            sub = [r for r in recs if r.get("qtype") == qtype]
            sub_eval = [r for r in sub if r.get("evaluation")]
            by_type[qtype] = {
                "n": len(sub),
                "correct": sum(1 for r in sub_eval
                               if r["evaluation"]["is_correct"]),
            }
            if sub_eval:
                by_type[qtype]["accuracy"] = round(
                    by_type[qtype]["correct"] / len(sub_eval), 4)
        out["by_type"] = by_type

        # cost + process
        out["avg_latency_ms"] = _mean([r.get("latency_ms", 0) for r in recs])
        out["avg_context_tokens"] = _mean([r.get("context_tokens", 0) for r in recs])
        out["avg_input_tokens"] = _mean([r.get("input_tokens", 0) for r in recs])
        out["avg_output_tokens"] = _mean([r.get("output_tokens", 0) for r in recs])
        out["avg_total_tokens"] = _mean([r.get("total_tokens", 0) for r in recs])
        out["avg_llm_calls"] = _mean([r.get("llm_calls", 0) for r in recs])
        out["avg_retrieval_steps"] = _mean([r.get("retrieval_steps", 0) for r in recs])
        out["avg_chunks_retrieved"] = _mean([r.get("chunks_retrieved", 0) for r in recs])
        out["avg_candidates_considered"] = _mean(
            [r.get("candidates_considered", 0) for r in recs])
        out["total_tokens"] = sum(r.get("total_tokens", 0) for r in recs)

        # grounding
        prec = [r["evaluation"]["citation_precision"] for r in evaluated
                if r["evaluation"].get("citation_precision") is not None]
        rec = [r["evaluation"]["citation_recall"] for r in evaluated
               if r["evaluation"].get("citation_recall") is not None]
        out["avg_citation_precision"] = _mean(prec)
        out["avg_citation_recall"] = _mean(rec)

        # process markers (mostly agentic)
        out["match_types"] = dict(Counter(
            r["evaluation"]["match_type"] for r in evaluated))
        stops = Counter(r.get("stop_reason", "") for r in recs if r.get("stop_reason"))
        if stops:
            out["stop_reasons"] = dict(stops)
        adapted = sum(1 for r in recs if r.get("strategy_changed"))
        if any(r.get("strategy_changed") is not None for r in recs):
            out["strategy_adapted"] = adapted
        out["agents_used"] = sorted({a for r in recs
                                     for a in (r.get("agents_invoked") or [])})
        return out

    def _delta_table(self) -> List[Dict[str, Any]]:
        """Per-question Agentic vs GraphRAG correctness delta (scatter data)."""
        rows = []
        for e in self.entries:
            pipes = e.get("pipelines") or {}
            agentic = pipes.get("Agentic GraphRAG") or pipes.get("agentic")
            graphrag = pipes.get("GraphRAG") or pipes.get("graphrag")
            if not agentic or not graphrag:
                continue
            a_ev, g_ev = agentic.get("evaluation"), graphrag.get("evaluation")
            if not a_ev or not g_ev:
                continue
            rows.append({
                "qid": e.get("qid"),
                "qtype": e.get("qtype"),
                "agentic_correct": a_ev["is_correct"],
                "graphrag_correct": g_ev["is_correct"],
                "agentic_tokens": agentic.get("total_tokens", 0),
                "graphrag_tokens": graphrag.get("total_tokens", 0),
                "delta": int(a_ev["is_correct"]) - int(g_ev["is_correct"]),
            })
        return rows
