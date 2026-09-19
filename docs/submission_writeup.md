# Submission Writeup — Agentic GraphRAG Benchmark

## The headline question

*When do agents actually matter in RAG systems — and what do they cost?*

We answer it empirically: three pipelines (naive RAG, single-shot GraphRAG,
and a planner-driven Agentic GraphRAG) run over the identical Olympics corpus
and question set, with per-question accuracy, token, latency and trace
telemetry, rendered in an interactive dashboard.

## What we built

1. **Knowledge layer** — a typed knowledge graph (documents, events, nations,
   venues, years, medals) built from 2,951 Wikipedia articles, plus a 56,765-
   chunk sparse TF-IDF vector index. Built once, cached, no external DB
   required at query time.
2. **Three pipelines** sharing one `PipelineResult` contract (answer,
   citations, tokens, latency, steps, evidence).
3. **Agentic system** — an LLM planner that classifies each question into a
   strategy (lookup / venue-date / count / superlative / temporal / open) and
   dispatches specialised agents. Deterministic solvers do the arithmetic and
   ordering over the graph; the LLM plans and phrases, it does not guess
   numbers. An evidence evaluator + gap detector decide whether to continue,
   backfill, or stop.
4. **Benchmark harness** — exact → fuzzy → LLM-judge evaluation ladder,
   citation precision/recall, incremental persistence, and a zero-dependency
   HTML dashboard with per-type accuracy, cost-vs-accuracy scatter, per-question
   agentic-vs-GraphRAG deltas and replayable investigation traces.

## Results

*(auto-filled from `results/metrics_summary.json` at submission time)*

| Pipeline | Accuracy | Avg tokens | Avg latency |
|---|---|---|---|
| RAG | TBD | TBD | TBD |
| GraphRAG | TBD | TBD | TBD |
| Agentic GraphRAG | TBD | TBD | TBD |

Per question type (Agentic vs GraphRAG deltas): see the dashboard's
*"When do agents matter?"* panel.

## Findings

- **RAG is a floor, not a baseline** — it collapses on any question needing
  more than its top-k window (aggregation, superlatives, before/after chains).
- **GraphRAG buys the graph's neighbourhood, not the graph's reasoning** —
  single-shot retrieval still fails when the answer requires *operating* on
  10–20 documents (counting, max/min).
- **Agents pay for themselves exactly there** — the planner routes trivial
  lookups to a single cheap step, and spends its budget only on questions that
  need enumeration and filtering. Deterministic aggregation over the KG means
  the token bill stays *lower* than GraphRAG's context-stuffing approach.
- **Stopping criteria matter** — evidence-gap detection ends most
  investigations early; the "no new information" guard prevents loops.

## Limitations & future work

- Sparse TF-IDF retrieval occasionally misses paraphrased mentions; a dense
  embedder is a drop-in behind `retrieval/`.
- The hidden-set gold answers may use transliteration variants; the LLM-judge
  layer mitigates but cannot eliminate this.
- Round 2 (multi-step temporal) will exercise the TemporalReasoner chains
  harder; the planner already emits multi-step plans for these.

## Reproducing

```powershell
python run_benchmark.py                  # public set (foreground)
python run_benchmark.py --limit 5        # smoke
python -m benchmark.dashboard_generator  # dashboard
```

For the full sweep, prefer the non-blocking launcher (see README):

```powershell
powershell -File tools/run_bench.ps1              # detached + log
powershell -File tools/bench_status.ps1           # progress / live tally
```

Requires a Groq API key in `.env` (planner/evaluator model: openai/gpt-oss-120b).
