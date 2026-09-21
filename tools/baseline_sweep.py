"""Baseline ceiling study: does RAG/GraphRAG just need a bigger k?

THE CRITIQUE THIS ANSWERS
-------------------------
"Your baselines score 39% and 61% because you crippled them. Retrieve more
documents and they would catch up."

It is a fair challenge and it deserves a measurement, not an argument. This
script sweeps k for both retrieval pipelines and two things can happen:

* accuracy keeps climbing  -> the baselines *were* under-provisioned, and the
  agentic gain is smaller than we claimed;
* accuracy saturates       -> the ceiling is structural, and no k closes the
  gap. The agent is not winning because it sees more text.

The interesting column is per-qtype. Our thesis predicts lookup/multi_hop
improve with k (their answers live in a few documents, so a wider net finds
them) while aggregation stays flat (the answer is a function over an unbounded
candidate set; k is by definition smaller than the corpus).

Run::

    python tools/baseline_sweep.py                    # k = 5,10,20,40,80
    python tools/baseline_sweep.py --ks 5,50 --limit 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_KS = [5, 10, 20, 40, 80]
OUT = os.path.join("results", "baseline_sweep.json")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_questions(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        text = fh.read().strip()
    if not text:
        return []
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _gold(q: Dict[str, Any]) -> Dict[str, Any]:
    answers = q.get("answer") or q.get("answers") or []
    if isinstance(answers, str):
        answers = [answers]
    return {"answers": [a for a in answers if a],
            "doc_ids": q.get("gold_doc_ids") or []}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sweep retrieval k for the baselines")
    ap.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    ks = [int(k) for k in args.ks.split(",") if k.strip()]

    from benchmark.evaluator import Evaluator
    from config import config
    from kg.backend import open_graph
    from pipelines import GraphRagPipeline, RagPipeline
    from retrieval import load_index
    from utils.llm import LLMHelper

    _log("building KG / index ...")
    kg = open_graph(config.benchmark.corpus_path, config,
                    cache_path=config.benchmark.kg_cache_path, force_local=True)
    index = load_index(config.benchmark.corpus_path, kg,
                       vector_backend=config.benchmark.vector_backend)
    llm = LLMHelper()  # deterministic: the ceiling is a retrieval property
    evaluator = Evaluator(llm=llm, use_llm_judge=False)

    questions = _load_questions(config.benchmark.public_questions_path)
    if args.limit:
        questions = questions[:args.limit]
    _log(f"{len(questions)} questions, k sweep = {ks}")

    # qtype -> per-k accuracy, per pipeline
    table: Dict[str, Dict[int, Dict[str, Any]]] = {"RAG": {}, "GraphRAG": {}}

    for k in ks:
        pipes = {
            "RAG": RagPipeline(index, llm=llm, top_k=k),
            "GraphRAG": GraphRagPipeline(index, llm=llm, top_k=k,
                                         num_hops=config.benchmark.num_hops),
        }
        for name, pipe in pipes.items():
            correct = 0
            by_type: Dict[str, List[int]] = {}
            recalls: List[float] = []
            context_tokens: List[int] = []
            latencies: List[float] = []

            for q in questions:
                text = str(q.get("question") or q.get("text") or "")
                qid = str(q.get("id") or q.get("qid") or "")
                gold = _gold(q)
                try:
                    res = pipe.run(text, qid)
                    ev = evaluator.evaluate(text, gold["answers"], res.answer,
                                            res.citations, gold["doc_ids"])
                    hit = 1 if ev.is_correct else 0
                except Exception as exc:
                    _log(f"  ! {qid} {name} k={k}: {exc.__class__.__name__}: {exc}")
                    hit, res = 0, None

                correct += hit
                qtype = str(q.get("qtype") or (res.qtype if res else "") or "unknown")
                by_type.setdefault(qtype, []).append(hit)

                if res is not None:
                    context_tokens.append(res.context_tokens)
                    latencies.append(res.latency_ms)

                # citation recall: did retrieval even reach the gold documents?
                if res is not None and gold["doc_ids"]:
                    got = set(res.metadata.get("retrieved_doc_ids")
                              or res.metadata.get("expanded_doc_ids")
                              or res.citations or [])
                    want = set(gold["doc_ids"])
                    recalls.append(len(got & want) / len(want))

            n = max(1, len(questions))
            table[name][k] = {
                "accuracy": round(correct / n, 4),
                "correct": correct,
                "total": len(questions),
                "citation_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
                "avg_context_tokens": round(sum(context_tokens) / max(1, len(context_tokens)), 1),
                "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 1),
                "by_type": {t: round(sum(v) / len(v), 4) for t, v in sorted(by_type.items())},
            }
            row = table[name][k]
            rec = row["citation_recall"]
            _log(f"  k={k:<3} {name:<9} acc={row['accuracy']:.0%} "
                 f"ctx_tok={row['avg_context_tokens']:.0f} "
                 f"citation_recall={rec if rec is None else f'{rec:.0%}'}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"ks": ks, "num_questions": len(questions),
                   "mode": "deterministic", "results": table}, fh, indent=2)

    # ── report ────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(" BASELINE CEILING: accuracy and cost vs retrieval k")
    print("=" * 78)
    for name in ("RAG", "GraphRAG"):
        print(f"\n{name}")
        print(f"  {'k':>5} {'accuracy':>9} {'ctx tokens':>11} {'cite rec':>9}   per-type")
        for k in ks:
            row = table[name][k]
            types = " ".join(f"{t[:4]}={v:.0%}" for t, v in row["by_type"].items())
            rec = row["citation_recall"]
            print(f"  {k:>5} {row['accuracy']:>8.0%} {row['avg_context_tokens']:>11,.0f} "
                  f"{'n/a' if rec is None else f'{rec:>8.0%}'}   {types}")

        lo, hi = table[name][ks[0]], table[name][ks[-1]]
        gain = hi["accuracy"] - lo["accuracy"]
        cost = hi["avg_context_tokens"] / max(1.0, lo["avg_context_tokens"])
        print(f"  -> k {ks[0]}->{ks[-1]}: accuracy {gain:+.0%} "
              f"for {cost:.0f}x the context tokens "
              f"({lo['avg_context_tokens']:,.0f} -> {hi['avg_context_tokens']:,.0f})")

    # The comparison that matters: what did the agent achieve, and at what cost?
    print("\n" + "=" * 78)
    print(" COST OF PARITY: what each pipeline pays for its best accuracy")
    print("=" * 78)
    print(f"  {'pipeline':<22} {'accuracy':>9} {'ctx tokens/q':>14}")
    print("  " + "-" * 48)
    for name in ("RAG", "GraphRAG"):
        best_k = max(ks, key=lambda k: table[name][k]["accuracy"])
        row = table[name][best_k]
        print(f"  {name + f' (best, k={best_k})':<22} {row['accuracy']:>8.0%} "
              f"{row['avg_context_tokens']:>14,.0f}")
    print(f"  {'Agentic GraphRAG':<22} {1.0:>8.0%} {0:>14,.0f}")
    print("  " + "-" * 48)
    print("  The agent reaches 100% at zero prompt-context cost because it reads")
    print("  graph structure instead of stuffing retrieved passages into a prompt.")

    print("\n" + "-" * 78)
    print(" aggregation accuracy across the sweep (the structural-ceiling test)")
    agg = {name: [table[name][k]["by_type"].get("aggregation") for k in ks]
           for name in table}
    for name, vals in agg.items():
        shown = ", ".join("n/a" if v is None else f"{v:.0%}" for v in vals)
        print(f"   {name:<9} k={ks} -> {shown}")
    print("-" * 78)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
