"""Executable Runner for Comprehensive 8-Way Ablation Study.

Evaluates 8 distinct architectural variants to rigorously quantify the marginal
contribution of every individual component in the Agentic GraphRAG system:
  1. Agentic-NoGraph: Pure vector similarity retrieval (graph traversal disabled)
  2. Agentic-NoVector: Pure topological graph traversal (dense vector search disabled)
  3. Agentic-NoLoop: Single-pass execution (dynamic loops and replanning disabled)
  4. Agentic-NoConflictResolution: First fact accepted (4-tier precedence matrix disabled)
  5. Agentic-NoAccumulators: Client-side row retrieval (TigerGraph GSQL accumulators disabled)
  6. Agentic-NoReformulation: Single retrieval attempt (query widening disabled)
  7. Agentic-NoEntityResolution: Exact surface matching (alias linking disabled)
  8. Agentic-Full: Complete system with all specialized agents and accumulators active
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import config
from kg.backend import describe_backend, open_graph
from pipelines.ablations import build_8way_ablations
from retrieval import load_index
from utils.llm import LLMHelper, build_llm
from benchmark.evaluator import Evaluator


def select_stratified_questions(questions_path: str, per_type: int = 4) -> List[Dict[str, Any]]:
    """Select a stratified subset of questions evenly distributed across types."""
    with open(questions_path, "r", encoding="utf-8") as fh:
        all_q = [json.loads(line) for line in fh if line.strip()]

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for q in all_q:
        qt = q.get("qtype", "general")
        by_type.setdefault(qt, []).append(q)

    selected: List[Dict[str, Any]] = []
    for qt, qlist in sorted(by_type.items()):
        selected.extend(qlist[:per_type])
    return selected


def run_ablation_study(questions: List[Dict[str, Any]], pipelines: List[Any],
                       evaluator: Evaluator, verbose: bool = True) -> Dict[str, Any]:
    """Execute the full 8-way ablation study."""
    matrix: Dict[str, Dict[str, Any]] = {}
    per_question_results: List[Dict[str, Any]] = []

    print(f"\n{'='*78}\nRunning 8-Way Ablation Study on {len(questions)} Questions across {len(pipelines)} Pipelines\n{'='*78}")

    for pipe in pipelines:
        name = pipe.name
        matrix[name] = {
            "num_evaluated": len(questions),
            "correct": 0,
            "accuracy": 0.0,
            "latencies_ms": [],
            "tokens": [],
            "by_type": {},
        }

    for idx, q in enumerate(questions, 1):
        qid = q.get("qid", f"q-{idx}")
        qtext = q.get("question", "")
        qtype = q.get("qtype", "general")
        golds = q.get("answer") or q.get("answers") or []
        gold_docs = q.get("gold_doc_ids") or []

        q_entry = {"qid": qid, "qtype": qtype, "question": qtext, "results": {}}

        for pipe in pipelines:
            name = pipe.name
            t0 = time.perf_counter()
            try:
                res = pipe.run(qtext, qid=qid)
                ans = res.answer
                cites = res.citations
                toks = getattr(res, "total_tokens", 0)
            except Exception as exc:
                ans = f"ERROR: {exc}"
                cites = []
                toks = 0
            dur_ms = (time.perf_counter() - t0) * 1000.0

            ev = evaluator.evaluate(qtext, golds, ans, cites, gold_docs, qtype=qtype)

            m = matrix[name]
            m["latencies_ms"].append(dur_ms)
            m["tokens"].append(toks)
            if ev.is_correct:
                m["correct"] += 1

            t_stat = m["by_type"].setdefault(qtype, {"total": 0, "correct": 0})
            t_stat["total"] += 1
            if ev.is_correct:
                t_stat["correct"] += 1

            q_entry["results"][name] = {
                "answer": ans,
                "is_correct": ev.is_correct,
                "match_type": ev.match_type,
                "latency_ms": round(dur_ms, 1),
            }

        per_question_results.append(q_entry)
        if verbose:
            stat_str = " ".join(f"{p.name.replace('Agentic-', '')[:7]}={'OK' if q_entry['results'][p.name]['is_correct'] else 'F'}"
                                for p in pipelines)
            print(f"[{idx:02d}/{len(questions):02d}] {qid} {qtype:<12s} {stat_str}")

    # Compute final aggregations
    full_pipe_name = "Agentic-Full"
    baseline_acc = 1.0
    if full_pipe_name in matrix and matrix[full_pipe_name]["num_evaluated"] > 0:
        baseline_acc = matrix[full_pipe_name]["correct"] / matrix[full_pipe_name]["num_evaluated"]

    summary_table: List[Dict[str, Any]] = []
    for name, m in matrix.items():
        total = m["num_evaluated"]
        corr = m["correct"]
        acc = round(corr / total, 4) if total > 0 else 0.0
        m["accuracy"] = acc
        m["avg_latency_ms"] = round(sum(m["latencies_ms"]) / len(m["latencies_ms"]), 1) if m["latencies_ms"] else 0.0
        m["avg_tokens"] = round(sum(m["tokens"]) / len(m["tokens"]), 1) if m["tokens"] else 0.0
        delta_acc = round(acc - baseline_acc, 4)
        m["delta_accuracy"] = delta_acc

        # Per type accuracies
        type_accs = {t: round(s["correct"] / s["total"], 2) for t, s in m["by_type"].items() if s["total"] > 0}
        m["type_accuracies"] = type_accs

        summary_table.append({
            "variant": name,
            "accuracy": f"{acc * 100:.1f}%",
            "delta_acc": f"{delta_acc * 100:+.1f}%",
            "avg_latency_ms": f"{m['avg_latency_ms']:.1f}ms",
            "by_type": type_accs,
        })

    # Print Formatted Markdown Table
    print(f"\n{'='*78}\n8-WAY ABLATION STUDY RESULTS SUMMARY\n{'='*78}")
    print(f"{'Variant':<32s} | {'Accuracy':<10s} | {'Delta':<10s} | {'Avg Latency':<12s}")
    print(f"{'-'*32}-|-{'-'*10}-|-{'-'*10}-|-{'-'*12}")
    for row in summary_table:
        print(f"{row['variant']:<32s} | {row['accuracy']:<10s} | {row['delta_acc']:<10s} | {row['avg_latency_ms']:<12s}")
    print(f"{'='*78}\n")

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_questions": len(questions),
        "baseline_pipeline": full_pipe_name,
        "matrix": matrix,
        "summary_table": summary_table,
        "questions": per_question_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 8-Way Ablation Study")
    parser.add_argument("--questions", default=config.benchmark.public_questions_path)
    parser.add_argument("--per-type", type=int, default=4, help="Questions per question type (default: 4 = 20 total)")
    parser.add_argument("--no-tg", action="store_true", help="Force local graph")
    parser.add_argument("--no-llm", action="store_true", help="Deterministic mode")
    parser.add_argument("--out", default=os.path.join(_ROOT, "results", "ablation_8way_matrix.json"))
    args = parser.parse_args()

    print("[ablation] initializing KnowledgeGraph and Index ...")
    kg = open_graph(config.benchmark.corpus_path, config,
                    cache_path=config.benchmark.kg_cache_path,
                    force_local=args.no_tg)
    index = load_index(config.benchmark.corpus_path, kg,
                       vector_backend=config.benchmark.vector_backend)
    llm = LLMHelper() if args.no_llm else build_llm(config)

    pipelines = build_8way_ablations(index, llm=llm)
    evaluator = Evaluator(llm, use_llm_judge=False)

    questions = select_stratified_questions(args.questions, per_type=args.per_type)
    summary = run_ablation_study(questions, pipelines, evaluator, verbose=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"[ablation] Detailed 8-way ablation matrix written to: {args.out}")


if __name__ == "__main__":
    main()
