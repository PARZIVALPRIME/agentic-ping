# Architecture

## System overview

```mermaid
flowchart LR
    subgraph Ingestion["Offline ingestion (cached under .cache/)"]
        C[corpus.jsonl<br/>2,951 articles] --> KG[kg/builder.py<br/>Knowledge Graph:<br/>Docs · Events · Nations · Venues · Years]
        C --> VI[retrieval/index.py<br/>TF-IDF sparse vector index<br/>56,765 chunks]
    end

    Q[Question] --> R1[Pipeline 1: RAG]
    Q --> R2[Pipeline 2: GraphRAG]
    Q --> R3[Pipeline 3: Agentic GraphRAG]

    VI --> R1
    VI --> R2
    KG --> R2
    VI --> R3
    KG --> R3

    R1 --> LLM[LLM service<br/>openai/gpt-oss-120b via Groq]
    R2 --> LLM
    R3 --> LLM

    R1 --> B[Benchmark harness]
    R2 --> B
    R3 --> B
    B --> D[dashboard/index.html]
```

All three pipelines share one output contract (`pipelines/base.py` →
`PipelineResult`): answer, citations, tokens, latency, retrieval steps,
`tools_called`, `agents_invoked`, `stop_reason`, per-step timings and evidence.

## The three pipelines

### Pipeline 1 — RAG (`pipelines/rag_pipeline.py`)

One vector-search shot over the TF-IDF chunk index (`rag_top_k=5`), chunks
re-ranked by lexical overlap, then a single LLM call. Baseline: no graph, no
planning, no retry. 1 retrieval step, ~1.0 LLM calls per question.

### Pipeline 2 — GraphRAG (`pipelines/graphrag_pipeline.py`)

Single-shot *graph-augmented* retrieval:

1. Vector search finds seed documents (`top_k=10`).
2. Entity linking attaches seed docs to KG vertices (nation, event, venue, year).
3. One-hop expansion pulls related documents through graph edges (same-nation
   athletes at the same games, events at the same venue…).
4. Chunks + graph context go to the LLM in **one** call, which also adjudicates
   the structured candidate.

No replanning: the retrieval shape is fixed regardless of question type. 2
retrieval steps, ~1.0 LLM calls, but ~10 chunks of context per question.

### Pipeline 3 — Agentic GraphRAG (`pipelines/agentic_pipeline.py`)

```mermaid
flowchart TB
    Q[question] --> CL[QuestionClassifier<br/>lookup / multi_hop / temporal /<br/>aggregation / superlative]
    CL --> PL[Planner<br/>emits an ordered step plan<br/>heuristic fallback when no LLM]
    PL --> O[Orchestrator<br/>executes, audits, replans]
    O --> A1[LookupResolver<br/>resolve_article · resolve_venue_date]
    O --> A2[TemporalReasoner<br/>resolve_anchor · resolve_event · verify_chain]
    O --> A3[Aggregator<br/>count_threshold · verify_by_recount]
    O --> A4[Comparator<br/>arg_extreme · verify_extreme]
    O --> A5[GraphTraverser / VectorSearcher / EntityLinker<br/>support retrieval]
    A1 & A2 & A3 & A4 & A5 --> EV[EvidenceEvaluator<br/>confidence score]
    EV -- gaps --> GD[GapDetector → planner recovery]
    GD -- spliced steps --> O
    EV -- sufficient / budget out --> S[Synthesizer<br/>grounded answer + citations]
    S -- contested --> ADJ[LLM adjudication]
```

Key mechanisms:

- **Question classifier + LLM planner** (`agents/classifier.py`,
  `agents/planner.py`) — classify into a strategy, then emit an ordered step
  plan (with a heuristic fallback when no LLM is available).
- **Deterministic solvers** (`reasoning/solvers.py`) — counts, superlatives,
  venue/date lookups and temporal ordering are computed over the KG, not
  hallucinated by the LLM. The LLM plans, adjudicates and phrases; it does not
  invent numbers.
- **Evidence evaluation + gap detection** (`agents/evidence_evaluator.py`,
  `agents/gap_detector.py`) — after each step the orchestrator decides:
  continue, backfill a specific gap, or stop.
- **Stopping criteria** — `confidence ≥ 0.9` (→ `confidence_reached`), plan
  exhausted, `max_steps=15`, or no new information for `stale_step_limit=2`
  steps (→ `no_new_information`). Recovery is bounded by `max_replans=3` and
  `max_widen_attempts=2`. A more specific stop reason is never overwritten by
  the staleness guard.
- **Full trace** — every step records its agent, operation, latency, tokens,
  evidence and `confidence_after`; the dashboard replays it as a waterfall.

### Agent routing (measured on the public set)

| Agent | Questions dispatched | Question types served |
|---|---|---|
| `EntityLinker` | 100 | all (always runs) |
| `EvidenceEvaluator` | 100 | all |
| `Synthesizer` | 100 | all |
| `VectorSearcher` | 69 | fallback / support retrieval |
| `GraphTraverser` | 59 | fallback / support retrieval |
| `LookupResolver` | 47 | lookup (19) + multi_hop (28) |
| `TemporalReasoner` | 22 | temporal (22) |
| `Aggregator` | 21 | aggregation (21) |
| `Comparator` | 10 | superlative (10) |

The last four rows show the router working exactly as designed: each
question type reaches its own specialist, and no specialist fires on a type it
does not serve.

## Benchmark harness

```mermaid
flowchart LR
    QS[eval_public.jsonl\n100 q / eval_hidden.jsonl\n50 q] --> R[runner.py]
    R --> P1[Pipelines ×3]
    P1 --> E[evaluator.py\nexact → fuzzy → LLM-judge\ncitation P/R]
    E --> M[metrics.py\nper-type accuracy · tokens\nlatency · agent deltas]
    M --> RES[(results/*.json)]
    M --> DG[dashboard_generator.py]
    DG --> HTML[(dashboard/index.html)]
```

Evaluation ladder (cheap-first): **exact match** → **fuzzy containment**
(normalised) → **LLM-as-judge** (only for near-misses, strict JSON verdict) →
**citation overlap** reported independently as precision/recall against
`gold_doc_ids`. The results file is rewritten after every question, so
interrupted runs keep completed work.

## Running the benchmark

A full sweep is 300 pipeline runs (100 questions × 3 pipelines) and takes
20–30 minutes wall-clock — dominated by LLM rate limits, not compute. Two
properties make this safe and observable:

| Property | Mechanism |
|---|---|
| **Never blocks the terminal** | `tools/run_bench.ps1` launches `python -u` detached with stdout/stderr redirected to `tools/bench_run.txt` and returns immediately |
| **Progress is always inspectable** | `tools/bench_status.ps1` reads that log and prints questions completed, a live per-pipeline OK tally, the last progress lines, stderr tail and the number of entries already written |
| **Crash-safe** | `runner.py` rewrites `results/*.json` atomically (temp file + `os.replace`) after each question; a per-pipeline exception is captured into the record instead of killing the run |
| **Resumable** | `--resume` / `-Resume` loads existing entries and skips those `qid`s, so an interrupted sweep continues rather than restarting |
| **Live progress lines** | one line per question: `[  7/100] pub-007 temporal RAG=F GraphRAG=F Agentic=OK tok=1211 elapsed=78s` |

```mermaid
flowchart LR
    RB[tools/run_bench.ps1] -->|Start-Process python -u| PY[run_benchmark.py]
    PY --> R[benchmark/runner.py]
    R -->|atomic write each question| RES[(results/*.json)]
    R -->|progress lines| LOG[(tools/bench_run.txt)]
    LOG --> BS[tools/bench_status.ps1]
    RES --> DG[dashboard_generator.py] --> HTML[(dashboard/index.html)]
```

## Deployment posture

The system is **self-contained**: the knowledge graph and sparse TF-IDF index
are built once from `corpus.jsonl` and cached under `.cache/`, so at query time
there is no database, container or vector service to run — only an outbound
HTTPS call to the LLM API (Groq, `openai/gpt-oss-120b`). Swapping in a
TigerGraph / dense-embedding backend is a drop-in behind `retrieval/` and
`kg/`, because pipelines only consume the `Index` and `KnowledgeGraph`
interfaces.
