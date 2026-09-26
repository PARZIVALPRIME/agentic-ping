# Submission Writeup — Agentic GraphRAG Benchmark

## The headline question

*When do agents actually matter in RAG systems — and what do they cost?*

We answer it empirically: four pipelines (naive RAG, single-shot GraphRAG, a
planner-driven Agentic GraphRAG, and a Router that escalates only when the cheap
arm is unsure) run over the identical Olympics corpus and question set, with
per-question accuracy, token, latency and trace telemetry, rendered in an
interactive dashboard.

## What we built

1. **Knowledge layer** — a typed knowledge graph (documents, events, nations,
   venues, years, medals) built from 2,951 Wikipedia articles, plus a 56,765-
   chunk sparse TF-IDF vector index. Built once, cached, no external DB
   required at query time.
2. **Four pipelines** sharing one `PipelineResult` contract (answer,
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

Measured live on the 100-question public set (`results/metrics_summary.json`),
local provider: Ollama `qwen3.5:4b`, no external API, no rate-limit sleeps.

| Pipeline | Accuracy | Avg total tokens | Avg context tokens | Avg latency |
|---|---|---|---|---|
| RAG | 43.0% (43/100) | 1,084 | 251 | 4.6 s |
| GraphRAG | 61.0% (61/100) | 1,850 | 886 | 4.6 s |
| Agentic GraphRAG | **88.0%** (88/100) | 14,004 | 189 | 78.4 s |
| Router | 86.0% (86/100) | 7,950 | 170 | 53.7 s |

Accuracy by question type (scored against the question set's own `qtype`, so
the buckets are identical across pipelines):

| Pipeline | lookup | multi_hop | temporal | aggregation | superlative |
|---|---|---|---|---|---|
| RAG | 17/19 | 17/28 | 8/22 | 1/21 | 0/10 |
| GraphRAG | 14/19 | 24/28 | 11/22 | 7/21 | 5/10 |
| Agentic GraphRAG | 19/19 | 19/28 | 20/22 | 20/21 | 10/10 |
| Router | 17/19 | 17/28 | 21/22 | 21/21 | 10/10 |

Hidden set (50 questions, no gold answers): the Agentic GraphRAG arm answered
50/50 with 0 errors and 1,042,857 tokens; `tools/validate_hidden.py` resolves
50/50 slots (100%). GraphRAG left 1 blank, Router 5 (all multi-hop/lookup),
RAG 33.
Per question type and per-question traces: see the dashboard's *"When do agents
matter?"* panel.

## Findings

- **RAG is a floor, not a baseline** — it collapses on any question needing
  more than its top-k window: 1/21 aggregation and 0/10 superlatives.
- **GraphRAG buys the graph's neighbourhood, not the graph's reasoning** —
  one hop of context lifts lookup to 14/19 and superlatives to 5/10, but it
  still fails when the answer requires *operating* on 10–20 documents
  (7/21 aggregation).
- **Agents pay for themselves exactly there** — the planner routes trivial
  lookups to a single cheap step (19/19) and spends its budget only on
  questions that need enumeration and filtering: 20/21 aggregation, 10/10
  superlatives, 20/22 temporal, versus GraphRAG's 7/21, 5/10 and 11/22. That
  is +27 points overall.
- **The gain is not free, and the honest number is the token bill.** Agentic
  GraphRAG costs ~7.6× GraphRAG's total tokens (14,004 vs 1,850) and ~17× its
  latency. It reads *fewer* context tokens per retrieval (189 vs 886) because
  it re-queries instead of stuffing — but it re-queries often, and every step
  re-sends the system prompt and tool schemas.
- **The Router is the value pick.** It matches the Agentic arm on aggregation
  (21/21), superlatives (10/10) and temporal (21/22) at 57% of the tokens and
  69% of the latency, escalating only when its confidence drops
  (`THRESH_ROUTER_MIN_CONFIDENCE`).
- **Where the agent still loses: multi-hop.** GraphRAG scores 24/28 on chained
  lookups; the Agentic arm scores 19/28 (Router 17/28). A single graph walk
  already answers "who won the event held at X on Y" in one retrieval, so on
  those 5 questions the agent's extra steps add cost and noise. This is the
  clearest remaining headroom, and it is a planner-shape problem, not a
  retrieval one.
- **Stopping criteria matter** — evidence-gap detection ends most
  investigations early; the "no new information" guard prevents loops.

## Limitations & future work

- Sparse TF-IDF retrieval occasionally misses paraphrased mentions; a dense
  embedder is a drop-in behind `retrieval/`.
- The hidden-set gold answers may use transliteration variants; the LLM-judge
  layer mitigates but cannot eliminate this.
- Round 2 (conflicting, evolving facts) is handled by explicit precedence —
  authority correction → recency → entity succession → majority — surfaced in the
  trace and the result metadata (`conflicts`, `uncertainty`). It is *reporting*:
  a conflict caps the confidence and raises a `conflicting_evidence` gap rather
  than rewriting a grounded answer. Two states in one successor lineage (the
  Soviet Union and Russia) are still counted as two entities — deciding whether
  they are one is an answer-level judgement, not a reporting one, so it is left
  to the reader rather than guessed.
- Whether two *editions* restate one fact is not decided automatically: the
  corpus names 415 event groups with a different venue per edition, and the
  infobox alias field links only generic-name collisions. The agent hands the
  versions it actually observed to `detect_conflicts` instead.
- Round 2-style temporal chains are exercised by the TemporalReasoner; the
  planner already emits multi-step plans for them.

## Reproducing

```powershell
python run_benchmark.py                  # public set (foreground)
python run_benchmark.py --limit 5        # smoke
python tools/refresh_dashboard.py --all  # rebuild + validate both pages
```

For the full two-stage sweep (public 100 → hidden 50), prefer the non-blocking
launcher (see README):

```powershell
powershell -File tools/run_sweep.ps1              # detached + per-stage logs
powershell -File tools/bench_status.ps1           # progress / live tally
python tools/validate_hidden.py                   # hidden slot coverage
```

The published numbers were produced with the **local open-weights profile**:
`LLM_PROVIDER=ollama` against `http://localhost:11434`, `qwen3.5:4b` for the
planner, agent and judge (see `.env`; nothing leaves the machine and no quota
is spent). The cloud profile is the drop-in alternative — set
`LLM_PROVIDER=groq` and `GROQ_API_KEY` in `.env` (planner/evaluator:
`openai/gpt-oss-120b`), and keep `LLM_TPM`/`LLM_CIRCUIT_THRESHOLD` live so the
free tier's 8K TPM is paced instead of 429-thrashed.

`docs/ablation_study.md` and `docs/baseline_ceiling.md` quote the deterministic
studies, not this live sweep; `python tools/verify_doc_numbers.py` re-checks
every figure in them against `results/`.
