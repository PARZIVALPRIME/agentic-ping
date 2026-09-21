"""Show the three-way comparison from a run summary, in PS terms.

    python tools/show_comparison.py results/llm_results_summary.json

The hackathon's headline question is "which questions need an agent, and which
don't". That is a *per-type* question, so this prints the per-type matrix and
then names the regime each type falls into, instead of leading with a single
headline accuracy number that hides the answer.
"""

from __future__ import annotations

import json
import sys

ORDER = ["RAG", "GraphRAG", "Agentic GraphRAG", "Router"]
TYPES = ["lookup", "multi_hop", "temporal", "aggregation", "superlative"]


def _acc(pipe: dict, qtype: str) -> float | None:
    """Per-type accuracy, or None when the run was never graded.

    The distinction matters more than it looks. The hidden set ships without
    gold answers, so `correct` is 0 for every pipeline - not because they were
    wrong, but because nothing was scored. Dividing `correct` by `n` there
    silently renders "0%", which reads as catastrophic failure and has already
    caused one false alarm. So: no evaluated records, no number.
    """
    row = (pipe.get("by_type") or {}).get(qtype) or {}
    n = row.get("n", 0)
    if not n or pipe.get("num_evaluated", 0) == 0:
        return None
    return row.get("correct", 0) / n


def _is_graded(pipes: dict) -> bool:
    return any(p.get("num_evaluated", 0) for p in pipes.values())


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "results/llm_results_summary.json"
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    pipes = data["pipelines"]
    names = [n for n in ORDER if n in pipes] + \
            [n for n in pipes if n not in ORDER]

    print(f"\n{'=' * 78}")
    print(f" THREE-WAY COMPARISON  ({data.get('num_questions', '?')} questions)  {path}")
    print("=" * 78)

    graded = _is_graded(pipes)
    if not graded:
        print(" UNGRADED RUN - this file has no gold answers, so there is no")
        print(" accuracy to report. Coverage and cost are shown instead; any")
        print(" '0%' you have seen for this file was a divide-by-nothing, not")
        print(" a result. The hidden set is graded by the organisers.")
        print("-" * 78)
        head = f"{'pipeline':<22}{'answered':>10}{'blank':>8}{'avg tok':>10}{'avg lat':>10}"
        print(head)
        for name in names:
            p = pipes[name]
            n = p.get("num_questions", 0)
            ans = p.get("num_answered", p.get("records_answering", n))
            print(f"{name:<22}{ans:>10}{n - ans:>8}"
                  f"{p.get('avg_total_tokens', 0):>10,.0f}"
                  f"{p.get('avg_latency_ms', 0) / 1000:>9.1f}s")
        print("=" * 78)
        return 0

    head = f"{'qtype':<14}{'n':>4}" + "".join(f"{n[:12]:>14}" for n in names)
    print(head)
    print("-" * len(head))

    verdicts = []
    for qtype in TYPES:
        row = (pipes[names[0]].get("by_type") or {}).get(qtype) or {}
        n = row.get("n", 0)
        if not n:
            continue
        cells, accs = "", {}
        for name in names:
            a = _acc(pipes[name], qtype)
            accs[name] = a
            cells += f"{'-':>14}" if a is None else f"{a:>13.0%} "
        print(f"{qtype:<14}{n:>4}{cells}")
        verdicts.append((qtype, n, accs))

    print("-" * len(head))
    totals = "".join(f"{pipes[n].get('accuracy', 0):>13.0%} " for n in names)
    print(f"{'OVERALL':<14}{data.get('num_questions', 0):>4}{totals}")
    tok = "".join(f"{pipes[n].get('avg_total_tokens', 0):>13,.0f} " for n in names)
    print(f"{'avg tokens':<14}{'':>4}{tok}")
    lat = "".join(f"{pipes[n].get('avg_latency_ms', 0) / 1000:>12.1f}s " for n in names)
    print(f"{'avg latency':<14}{'':>4}{lat}")

    # ── the actual deliverable: where is the agent worth it? ────────────────
    rag, agent = "RAG", "Agentic GraphRAG"
    if rag in pipes and agent in pipes:
        print(f"\n{'=' * 78}")
        print(" WHERE THE AGENT EARNS ITS COST  (agent accuracy - RAG accuracy)")
        print("=" * 78)
        rows = []
        for qtype, n, accs in verdicts:
            a, r = accs.get(agent), accs.get(rag)
            if a is None or r is None:
                continue
            rows.append((a - r, qtype, n, r, a))
        for delta, qtype, n, r, a in sorted(rows, reverse=True):
            if delta >= 0.5:
                verdict = "AGENT REQUIRED - retrieval cannot reach the evidence"
            elif delta >= 0.2:
                verdict = "agent helps materially"
            elif delta > 0.05:
                verdict = "marginal gain"
            else:
                verdict = "OVERKILL - RAG is at parity, route to RAG"
            print(f"  {qtype:<14} n={n:<3} {r:>4.0%} -> {a:>4.0%}  "
                  f"{delta:+5.0%}   {verdict}")
        print("\n  Read this table, not the headline number: the deliverable is")
        print("  knowing which rows say REQUIRED and which say OVERKILL.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
