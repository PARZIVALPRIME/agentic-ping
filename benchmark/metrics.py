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


def llm_activity(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per-pipeline provider contribution, derived only from the records.

    Every results file records ``llm_calls`` and tokens per pipeline, so "did the
    LLM actually answer anything in this run?" is answerable after the fact. It
    needs to be: a run whose provider calls all failed (or a ``--no-llm`` run)
    still writes a complete, plausible-looking file in which every answer came
    from the deterministic path. Without this block a reader cannot tell such a
    file apart from an LLM-backed one - "0 LLM calls" looks like a property of
    the architecture instead of a property of the run.

    Lives here (not in ``benchmark.runner``) so the dashboard generator can call
    it without importing the harness, KG builders or evaluator.
    """
    activity: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        for name, rec in (entry.get("pipelines") or {}).items():
            if not isinstance(rec, dict):
                continue
            slot = activity.setdefault(name, {
                "records": 0, "records_with_calls": 0, "records_answering": 0,
                "calls": 0, "total_tokens": 0, "output_tokens": 0,
                "unusable_adjudications": 0,
            })
            slot["records"] += 1
            calls = int(rec.get("llm_calls") or 0)
            out_tokens = int(rec.get("output_tokens") or 0)
            slot["calls"] += calls
            slot["total_tokens"] += int(rec.get("total_tokens") or 0)
            slot["output_tokens"] += out_tokens
            if calls > 0:
                slot["records_with_calls"] += 1
            if out_tokens > 0:
                # The provider returned prose for this record, so the model - not
                # just the client - ran. A record can carry llm_calls > 0 with
                # zero output tokens when the call failed (429, timeout, bad
                # JSON), which is exactly the case that looks like "the LLM was
                # used" while every answer still came from the solvers.
                slot["records_answering"] += 1
            adj = (rec.get("metadata") or {}).get("adjudication")
            if isinstance(adj, dict) and adj.get("reason") == "verdict_unusable":
                slot["unusable_adjudications"] += 1
    for slot in activity.values():
        slot["llm_contributed"] = slot["records_answering"] > 0
    return activity


def llm_activity_warnings(activity: Dict[str, Dict[str, Any]]) -> List[str]:
    """Human-readable warnings about a run's provider contribution.

    Shared by the live runner and ``--summarize-only`` so both describe the same
    situation the same way. An empty list means every pipeline recorded provider
    calls for every question.
    """
    warnings: List[str] = []
    if not activity:
        return warnings
    active = {name: slot for name, slot in activity.items()
              if slot["records_with_calls"] > 0}
    if not active:
        warnings.append("no pipeline recorded a provider call: this results file "
                        "is a deterministic run (see --no-llm)")
    elif len(active) < len(activity):
        idle = ", ".join(sorted(set(activity) - set(active)))
        warnings.append("mixed LLM activity: no provider calls recorded for "
                        f"{idle} - those records came from the deterministic path")
    partial = {name: slot for name, slot in active.items()
               if slot["records_with_calls"] < slot["records"]}
    if partial:
        detail = ", ".join(f"{name} {slot['records_with_calls']}/{slot['records']}"
                           for name, slot in sorted(partial.items()))
        warnings.append("provider calls were recorded on only part of the run "
                        f"({detail}) - the remaining records are deterministic")
    # Calls were attempted but no completion text came back: the client counted
    # the request, the provider served nothing (429/timeout/bad JSON), and the
    # answers are therefore deterministic. This is the "0 tokens, 1 call"
    # signature of a run that looks LLM-backed and is not.
    silent = {name: slot for name, slot in active.items()
              if slot["output_tokens"] == 0}
    if silent:
        detail = ", ".join(f"{name} {slot['calls']} call(s)"
                           for name, slot in sorted(silent.items()))
        warnings.append("provider calls returned no completion text "
                        f"({detail}): those answers came from the deterministic "
                        "path - check the API key, quota and circuit breaker")
    return warnings


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
