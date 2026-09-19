# Agentic GraphRAG Benchmark — RAG vs GraphRAG vs Agentic

Three retrieval pipelines over the same Olympics corpus (2,951 Wikipedia
articles, 1987–2023), benchmarked on 100 public questions with gold answers:

| Pipeline | What it does |
|---|---|
| **RAG** | one vector-search shot → top chunks → one LLM answer |
| **GraphRAG** | entity linking → 1-hop knowledge-graph expansion + community context → LLM answer |
| **Agentic GraphRAG** | LLM planner chooses per-question strategy (lookup / venue-date / count-superlative / temporal), specialised agents execute each step, evidence-gap checks decide when to stop |

Every run records **accuracy** (exact → fuzzy → LLM-judge), **token cost**,
**latency**, **retrieval steps**, **citation precision/recall** and — for the
agentic pipeline — a full **step-by-step investigation trace**. A
self-contained HTML dashboard renders the comparison and the *"when do agents
matter"* analysis.

## Quick start

```powershell
.\setup\setup_env.ps1          # venv + deps
Copy-Item .env.example .env    # add your GROQ_API_KEY

python run_benchmark.py                      # full public benchmark (100 q × 3 pipelines)
python run_benchmark.py --limit 5            # quick smoke
python run_benchmark.py --pipelines agentic  # single pipeline
python run_benchmark.py --pipelines agentic --mode react   # LLM tool-calling only
python run_benchmark.py --no-llm             # deterministic baseline (no API calls)
python run_benchmark.py <hidden.jsonl> --out results/hidden_results.json

python -m benchmark.dashboard_generator      # rebuild dashboard/index.html
```

### Two run modes

| Mode | Command | Speed | Use for |
|---|---|---|---|
| LLM-assisted | `python run_benchmark.py` | ~15–30 min (provider rate limits) | the headline numbers: LLM classification, slot extraction, answer adjudication, LLM-as-judge |
| Deterministic | `python run_benchmark.py --no-llm` | **~1 min** | reproducible baseline, verifying the harness, running with no quota |

Both modes run all three pipelines end-to-end — the LLM is an accelerator, never
a requirement. If the provider starts rate-limiting mid-run, a circuit breaker
trips and the remaining questions finish deterministically instead of stalling
(`llm.circuit` in `results/*_summary.json` shows how often this happened).

### Is the LLM actually contributing?

That fallback is the reason a results file needs auditing before it is
published. A run whose provider calls *all* fail still finishes: every answer
comes from the deterministic solvers, the token counters read zero, and the
headline table can read *"100% accuracy at 0 tokens"* — a deterministic result
wearing an LLM run's clothes. The tell is latency: the record still spends
seconds on a call that recorded nothing.

```powershell
python tools/audit_results.py                     # audit results/public_results.json
python tools/audit_results.py results/regression_final.json
```

```
pipeline              n     acc  w/calls   tokens    out   p50 ms  unacct status
RAG                 100     39%      100    27838      0     3438       0 llm
GraphRAG            100     61%      100    91350      0     3650       0 llm
Agentic GraphRAG    100    100%        0        0      0     3533     100 provider-failed
```

The exit code is non-zero when a pipeline took part in a run that used the
provider yet recorded no calls of its own, so the tool gates numbers before they
reach the dashboard. It also flags results written by an **older revision** of
the pipelines (a missing `metadata` field means the file no longer describes the
code shipped beside it) and compares answers against the `--no-llm` baseline.

Every summary carries the same record under `provenance.llm_activity`, and the
dashboard renders it in the **Provider contribution** panel. Rebuild a summary
without re-running the pipelines with:

```powershell
python run_benchmark.py --summarize-only --out results/public_results.json
```

Provider/evaluator/backend telemetry is only known during a live run, so a
rebuilt summary reports those blocks as `null` rather than guessing them.

### If a run looks stuck

It isn't — the provider is metering you. Groq's free tier limits **tokens per
minute**, and each adjudication prompt is ~1–2k tokens, so a fixed request
spacing cannot keep up. Fixes, in order of effort:

```powershell
powershell -File tools\run_bench.ps1 -Follow     # always run in the background
```

* Set `LLM_TPM` in `.env` to your tier's limit (check
  <https://console.groq.com/settings/limits>) so the pacer self-throttles.
* Or run `python run_benchmark.py --no-llm` — deterministic mode can never stall.
* Progress is timestamped and flushed per question, with an ETA estimate:

```
[15:39:58]   1/100 pub-001 aggregation RAG=F Graph=OK Agent=OK tok=1422 took=2.1s elapsed=2s eta=3.4m
```

<details>
<summary>Environment variables that control pacing</summary>

| Variable | Default | Meaning |
|---|---|---|
| `LLM_MIN_INTERVAL_S` | `1.5` | minimum spacing between request starts (0 = off) |
| `LLM_TPM` | `6000` | token budget per minute; pacer sleeps to stay under it (0 = off) |
| `LLM_MAX_BACKOFF_S` | `20` | cap on a single retry wait |
| `LLM_MAX_CALL_S` | `90` | cap on total retry time for one call |
| `LLM_CIRCUIT_THRESHOLD` | `3` | consecutive 429s before failing fast |
| `LLM_CIRCUIT_COOLDOWN_S` | `60` | how long the circuit stays open before one probe call |

</details>

### Long runs: don't block your terminal

The full 100 × 3 sweep takes 20–30 minutes (LLM rate limits), which makes a
foreground run look *stuck*. Launch it in the background instead:

```powershell
powershell -File tools\run_bench.ps1                 # starts detached, returns immediately
powershell -File tools\run_bench.ps1 -Follow         # …or stream progress live
powershell -File tools\bench_status.ps1              # progress / live tally / errors, any time
powershell -File tools\bench_status.ps1 -Log tools\bench_hidden.txt -Out results\hidden_results.json
powershell -File tools\run_bench.ps1 -Resume         # continue an interrupted run
```

`run_bench.ps1` launches `python -u` with output redirected to
`tools/bench_run.txt`; `bench_status.ps1` prints questions completed, a live
per-pipeline OK tally, the last progress lines and any stderr. Results are
rewritten after **every** question (`results/*.json`), so an interrupted run
loses at most one question — `--resume` / `-Resume` skips what is already done.

Outputs land in `results/` (`public_results.json`, `metrics_summary.json`),
the dashboard in `dashboard/index.html` (open directly in a browser — no
server needed).

Rebuild the dashboard from any results file:

```powershell
python -m benchmark.dashboard_generator                                   # results/public_results.json
python -m benchmark.dashboard_generator results/deterministic_results.json `
    --summary results/deterministic_results_summary.json
python tools\validate_dashboard.py                                        # sanity-check the generated page
```

## TigerGraph backend (optional)

The graph the agent queries is built from `corpus.jsonl` and cached, but the
same graph can be served by TigerGraph instead — the graph tools then answer
from the database rather than from a local scan, and nothing else in a run
changes. The corpus graph is always built and kept as a **mirror**: every
remote answer is verified against it on first use, and any failure (server
down, schema missing, query missing, partial ingest) degrades to the mirror
with a printed reason and a recorded counter. A TigerGraph run therefore cannot
silently score worse than a local one.

```powershell
python tools/tg_ingest.py --status              # what the server has now
python tools/tg_ingest.py --install --push      # schema + queries + the corpus graph
$env:TG_ENABLED="true"; $env:TG_HOST="http://localhost"   # + TG_USERNAME/TG_PASSWORD
python run_benchmark.py                         # --no-tg forces the local graph
```

| Variable | Default | Meaning |
|---|---|---|
| `TG_ENABLED` | *(unset)* | serve the graph from TigerGraph |
| `TG_HOST` / `TG_RESTPP_PORT` | `https://...tgcloud.io` / `443` | RESTPP endpoint (Community: `http://localhost`, port `9000`) |
| `TG_GRAPHNAME` / `TG_USERNAME` / `TG_PASSWORD` / `TG_TOKEN` | `OlympicsKG` / `tigergraph` / — / — | graph and credentials |
| `TG_MAX_ROWS` | `100000` | row budget per query — a **safety valve**, not a tuning knob: a budget below the largest candidate set (2210 rows here) is reported loudly, never silently applied |
| `TG_TIMEOUT` / `TG_RETRIES` | `30` / `2` | per-request timeout and retries |

No TigerGraph install? The suite ships an in-process RESTPP double:

```powershell
python tools/test_tg_backend.py                       # 48 offline checks, no server needed
python tools/tg_fake_server.py --port 19123           # offline RESTPP double
$env:TG_ENABLED="true"; $env:TG_HOST="http://127.0.0.1:19123"
python tools/probe_tg_live.py                         # the exact benchmark wiring, proven live
```

## Architecture

```
question ──► RAG ─────────────── vector top-k ──────────────► LLM ──► answer
         ──► GraphRAG ──► vector seeds ──► KG 1-hop ───────► LLM ──► answer
         ──► Agentic ──► Planner(LLM) ─┬─ lookup_resolver ──┤
                                       ├─ venue_date solver │  evidence
                                       ├─ count/superlative │  evaluator ─►
                                       ├─ temporal solver   │  gap detector
                                       └─ vector fallback ──┘     │
                                                          Synthesizer ──► answer
```

See `docs/architecture.md` for the full diagram and `docs/submission_writeup.md`
for the hackathon narrative. The knowledge graph + sparse TF-IDF vector index
are built once from `corpus.jsonl` and cached under `.cache/` (`kg/`,
`retrieval/`).

## Repository layout

| Path | Contents |
|---|---|
| `pipelines/` | the 3 pipelines + shared `PipelineResult` schema |
| `agents/` | planner, orchestrator and the specialised agentic agents |
| `reasoning/` | query parser + deterministic solvers (counts, superlatives, temporal) |
| `kg/`, `retrieval/` | knowledge-graph builder and TF-IDF vector index |
| `tg/` | the TigerGraph backend: schema, queries, loader, client, offline test double |
| `benchmark/` | runner, evaluation ladder, metrics aggregation, dashboard generator |
| `dashboard/` | self-contained HTML/CSS/JS metrics dashboard |
| `results/` | benchmark outputs + cached KG/index |
| `tools/` | development probes, validators and the background run scripts |
| `setup/` | environment setup + ingestion scripts |

## Hackathon resources

| Path | Contents |
|---|---|
| `corpus/corpus.jsonl` | 2,951 documents, ~5,471,565 tokens |
| `questions/eval_public.jsonl` | 100 questions, with answers |
| `questions/eval_hidden.jsonl` | 50 questions, without answers |

All three files are JSONL. The **corpus is the only source of truth** — answers
are defined over these documents, not over the real world. Corpus text is
derived from English Wikipedia, licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

