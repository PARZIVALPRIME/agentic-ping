"""Generalization & Out-of-Distribution (OOD) Test Suite.

Tests reasoning depth across 7 challenging structural dimensions:
  1. 2-hop cross-entity traversal
  2. 3-hop multi-relation graph paths
  3. Cyclic entity queries (A -> B -> C -> A)
  4. Negation queries ("Which events had NO competitors from...")
  5. Multi-edition temporal ordering across 3+ Olympic cycles
  6. Multi-constraint aggregation (year AND sport AND threshold)
  7. Counterfactual & unanswerable queries (detecting impossibility without hallucination)
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@dataclass
class GeneralizationCase:
    qid: str
    dimension: str
    question: str
    gold_answers: List[str]
    gold_doc_ids: List[str]
    reasoning_hops: int
    is_counterfactual: bool = False
    expected_refusal: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# 14 curated synthetic and stress-test benchmark questions spanning the 7 dimensions
GENERALIZATION_SUITE: List[GeneralizationCase] = [
    # Dimension 1: 2-Hop Traversal
    GeneralizationCase(
        qid="ood-2hop-01",
        dimension="2_hop_traversal",
        question="Which venue hosted the event won by Chen Ding at the 2012 Summer Olympics?",
        gold_answers=["Olympic Stadium", "The Mall", "London Olympic Stadium"],
        gold_doc_ids=["Q1050909"],
        reasoning_hops=2,
        notes="Hop 1: Athlete -> Event (Men's 20km walk), Hop 2: Event -> Venue (The Mall / Olympic Stadium)"
    ),
    GeneralizationCase(
        qid="ood-2hop-02",
        dimension="2_hop_traversal",
        question="What country did the competitor who won gold at London Velopark on 4 August 2012 represent?",
        gold_answers=["Great Britain", "United Kingdom", "GBR"],
        gold_doc_ids=["Q2297633"],
        reasoning_hops=2,
        notes="Hop 1: Venue & Date -> Event -> Athlete (Dani King/Laura Trott), Hop 2: Athlete -> Country"
    ),

    # Dimension 2: 3-Hop Traversal
    GeneralizationCase(
        qid="ood-3hop-01",
        dimension="3_hop_traversal",
        question="In which city was the Olympics held where Naim Süleymanoğlu won gold at Olympic Weightlifting Gymnasium?",
        gold_answers=["Seoul"],
        gold_doc_ids=["Q25239316"],
        reasoning_hops=3,
        notes="Hop 1: Venue -> Event, Hop 2: Event -> Edition (1988 Summer Olympics), Hop 3: Edition -> Host City"
    ),
    GeneralizationCase(
        qid="ood-3hop-02",
        dimension="3_hop_traversal",
        question="What sport was contested at the venue where Carolina Marín won gold in August 2016?",
        gold_answers=["Badminton"],
        gold_doc_ids=["Q25301483"],
        reasoning_hops=3,
        notes="Hop 1: Athlete -> Event, Hop 2: Event -> Venue (Riocentro Pavilion 4), Hop 3: Venue -> Sport (Badminton)"
    ),

    # Dimension 3: Cyclic Queries
    GeneralizationCase(
        qid="ood-cyclic-01",
        dimension="cyclic_queries",
        question="Starting from Chen Ding, find the 2012 event he won, locate the sport of that event, and list all gold medallists from that sport at the 2012 Games.",
        gold_answers=["Chen Ding"],
        gold_doc_ids=["Q1050909"],
        reasoning_hops=4,
        notes="Cyclic loop testing self-referential graph closure"
    ),

    # Dimension 4: Negation Queries
    GeneralizationCase(
        qid="ood-negation-01",
        dimension="negation_queries",
        question="According to the provided corpus, did biathlon at the 2018 Winter Olympics have any event with 0 competitors?",
        gold_answers=["No", "0", "None"],
        gold_doc_ids=["Q47091419"],
        reasoning_hops=2,
        notes="Negation condition testing absence of empty events"
    ),
    GeneralizationCase(
        qid="ood-negation-02",
        dimension="negation_queries",
        question="Which sailing events at the 2000 Summer Olympics had fewer than 10 competitors?",
        gold_answers=["None", "0"],
        gold_doc_ids=["Q7400327"],
        reasoning_hops=2,
        notes="Strict lower bound negative constraint"
    ),

    # Dimension 5: Temporal Ordering Across 3+ Editions
    GeneralizationCase(
        qid="ood-temporal-01",
        dimension="multi_edition_temporal",
        question="List the host cities of the Summer Olympics in 2008, 2012, and 2016 in chronological order.",
        gold_answers=["Beijing, London, Rio de Janeiro", "Beijing, London, Rio"],
        gold_doc_ids=[],
        reasoning_hops=3,
        notes="Temporal sequencing across 3 distinct Olympic editions"
    ),
    GeneralizationCase(
        qid="ood-temporal-02",
        dimension="multi_edition_temporal",
        question="Which Olympic edition occurred immediately between the 2008 Beijing Olympics and the 2016 Rio Olympics?",
        gold_answers=["2012 London Olympics", "2012 Summer Olympics", "London 2012", "2012"],
        gold_doc_ids=[],
        reasoning_hops=2,
        notes="Intermediate predecessor/successor resolution"
    ),

    # Dimension 6: Multi-Constraint Aggregation
    GeneralizationCase(
        qid="ood-multi-agg-01",
        dimension="multi_constraint_aggregation",
        question="According to the corpus, how many cycling events at the 2000 Summer Olympics had between 30 and 40 competitors?",
        gold_answers=["2"],
        gold_doc_ids=["Q1856784", "Q2133123"],
        reasoning_hops=2,
        notes="Dual boundary filtering: lower bound >= 30 and upper bound <= 40"
    ),
    GeneralizationCase(
        qid="ood-multi-agg-02",
        dimension="multi_constraint_aggregation",
        question="How many alpine skiing events at the 2014 Winter Olympics had more than 100 competitors?",
        gold_answers=["2"],
        gold_doc_ids=["Q15054994", "Q15054997"],
        reasoning_hops=2,
        notes="High threshold filtering requiring accumulator aggregation"
    ),

    # Dimension 7: Counterfactual & Unanswerable Queries
    GeneralizationCase(
        qid="ood-counterfactual-01",
        dimension="counterfactual_unanswerable",
        question="If the 2020 Tokyo Summer Olympics were held in Antarctica in 1850, who won the gold medal in women's skateboarding?",
        gold_answers=["Unanswerable", "Not in corpus", "Hypothetical / Did not occur", "None"],
        gold_doc_ids=[],
        reasoning_hops=1,
        is_counterfactual=True,
        expected_refusal=True,
        notes="Testing hallucination refusal on historically impossible prompt"
    ),
    GeneralizationCase(
        qid="ood-counterfactual-02",
        dimension="counterfactual_unanswerable",
        question="Who won the gold medal in the men's 5000 metres swimming event at the 1996 Summer Olympics?",
        gold_answers=["Unanswerable", "Event did not exist", "No such event", "None"],
        gold_doc_ids=[],
        reasoning_hops=1,
        is_counterfactual=True,
        expected_refusal=True,
        notes="Testing refusal for non-existent Olympic event"
    ),
]


def run_generalization_evaluation(pipeline, verbose: bool = True) -> Dict[str, Any]:
    """Execute the generalization test suite against a given pipeline."""
    from benchmark.evaluator import Evaluator

    evaluator = Evaluator(None, use_llm_judge=False)
    results_by_dim: Dict[str, List[Dict[str, Any]]] = {}
    dim_scores: Dict[str, Dict[str, float]] = {}

    t_start = time.perf_counter()
    total_correct = 0

    if verbose:
        print(f"\n{'='*70}\nRunning Generalization & OOD Evaluation ({len(GENERALIZATION_SUITE)} questions)\n{'='*70}")

    for idx, case in enumerate(GENERALIZATION_SUITE, 1):
        t0 = time.perf_counter()
        try:
            res = pipeline.run(case.question, qid=case.qid)
            ans = res.answer
            cites = res.citations
        except Exception as exc:
            ans = f"ERROR: {exc}"
            cites = []

        dur_ms = (time.perf_counter() - t0) * 1000.0

        # Special check for counterfactual refusal
        if case.expected_refusal:
            refusal_terms = ["unanswerable", "not exist", "no such", "not in corpus", "cannot be answered", "none", "hypothetical"]
            ans_lower = ans.lower()
            is_ok = any(t in ans_lower for t in refusal_terms) or ans.strip() in ("", "0")
            match_type = "refusal_passed" if is_ok else "failed_refusal"
            accuracy_score = 1.0 if is_ok else 0.0
        else:
            ev = evaluator.evaluate(case.question, case.gold_answers, ans, cites, case.gold_doc_ids)
            is_ok = ev.is_correct
            match_type = ev.match_type
            accuracy_score = ev.accuracy_score

        if is_ok:
            total_correct += 1

        record = {
            "qid": case.qid,
            "dimension": case.dimension,
            "question": case.question,
            "prediction": ans,
            "gold_answers": case.gold_answers,
            "is_correct": is_ok,
            "match_type": match_type,
            "latency_ms": round(dur_ms, 1),
            "hops": case.reasoning_hops,
        }

        if case.dimension not in results_by_dim:
            results_by_dim[case.dimension] = []
        results_by_dim[case.dimension].append(record)

        if verbose:
            tag = "PASS" if is_ok else "FAIL"
            print(f"  [{idx:02d}/{len(GENERALIZATION_SUITE):02d}] {case.dimension:<28s} {tag} -> {ans[:50]!r} ({dur_ms:.0f}ms)")

    # Aggregate by dimension
    for dim, records in results_by_dim.items():
        corr = sum(1 for r in records if r["is_correct"])
        tot = len(records)
        avg_lat = sum(r["latency_ms"] for r in records) / tot if tot else 0.0
        dim_scores[dim] = {
            "accuracy": round(corr / tot, 4),
            "passed": corr,
            "total": tot,
            "avg_latency_ms": round(avg_lat, 1),
        }

    total_s = time.perf_counter() - t_start
    overall_acc = total_correct / len(GENERALIZATION_SUITE)

    summary = {
        "pipeline": pipeline.name,
        "total_questions": len(GENERALIZATION_SUITE),
        "total_passed": total_correct,
        "overall_accuracy": round(overall_acc, 4),
        "total_time_seconds": round(total_s, 2),
        "dimension_scores": dim_scores,
        "records": results_by_dim,
    }

    if verbose:
        print(f"\nGeneralization Test Summary: {total_correct}/{len(GENERALIZATION_SUITE)} Passed ({overall_acc*100:.1f}%) in {total_s:.1f}s\n")
    return summary


def main() -> None:
    from config import config
    from kg.backend import open_graph
    from pipelines import build_pipelines
    from retrieval import load_index
    from utils.llm import LLMHelper

    kg = open_graph(config.benchmark.corpus_path, config,
                    cache_path=config.benchmark.kg_cache_path,
                    force_local=True)
    index = load_index(config.benchmark.corpus_path, kg,
                       vector_backend=config.benchmark.vector_backend)
    llm = LLMHelper()
    pipes = build_pipelines(index, llm, config, with_router=True)
    router = next(p for p in pipes if p.name == "Router")

    summary = run_generalization_evaluation(router, verbose=True)
    out_path = os.path.join(_ROOT, "results", "generalization_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"Wrote generalization results to {out_path}")


if __name__ == "__main__":
    main()
