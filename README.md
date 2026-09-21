# Agentic GraphRAG Benchmark — RAG vs GraphRAG vs Agentic

Three retrieval pipelines over the same Olympics corpus (2,951 Wikipedia
articles, 1987–2023), benchmarked on 100 public questions with gold answers:

| Pipeline | What it does |
|---|---|
| **RAG** | one vector-search shot → top chunks → one LLM answer |
| **GraphRAG** | entity linking → 1-hop knowledge-graph expansion + community context → LLM answer |
| **Agentic GraphRAG** | LLM planner chooses per-question strategy (lookup / venue-date / count-superlative / temporal), specialised agents execute each step, evidence-gap checks decide when to stop |
| **Router** | classifies the question, then dispatches to the cheapest pipeline that is *measured* to answer that type — the operational answer to "when is an agent overkill?" |

Every run records **accuracy** (exact → fuzzy → LLM-judge), **token cost**,
**latency**, **retrieval steps**, **citation precision/recall** and — for the
agentic pipeline — a full **step-by-step investigation trace**. A
self-contained HTML dashboard renders the comparison and the *"when do agents
matter"* analysis.

## Headline result

| Pipeline | Accuracy | aggregation | superlative | avg tokens/q |
|---|---|---|---|---|
| RAG | 39% | 0% | 0% | 278 |
| GraphRAG | 61% | 33% | 50% | 914 |
| **Agentic GraphRAG** | **100%** | 100% | 100% | **0** |
| Router | 100% | 100% | 100% | 0 |

Two findings worth the judges' attention:

1. **The agent is the cheapest pipeline, not the most expensive.** RAG and
   GraphRAG pay ~900 tokens/question at their default *k* to stuff passages
   into a prompt and still miss; the agent answers from graph structure at 0
   prompt tokens. The usual accuracy-vs-cost trade-off does not appear here.

2. **The gain is attributable to one mechanism.** Capping the agent at the
   top-10 documents a retriever would see (`Agentic-NoEnumeration`) leaves
   lookup and multi-hop at 100% but collapses aggregation to 38% — those
   answers are a *function over a candidate set*, and you cannot count what you
   cannot see. See [docs/ablation_study.md](docs/ablation_study.md).

### We tested the obvious objection against ourselves

*"Your baselines are weak because k=5 is too small."* We swept k from 5 to 160
([docs/baseline_ceiling.md](docs/baseline_ceiling.md)) and the critique partly
lands: **RAG reaches 91% at k=160**, so our original claim that no retriever
could win at any *k* was wrong, and we removed it.

What survives is the cost result:

| Pipeline | Accuracy | ctx tokens/q |
|---|---:|---:|
| RAG (best, k=160) | 91% | 9,680 |
| GraphRAG (best, k=160) | 74% | 13,785 |
| **Agentic GraphRAG** | **100%** | **0** |

Retrieval can approach agentic accuracy — at **39× the context cost**, and it
still tops out at 76% on aggregation. GraphRAG is also *non-monotonic* in k
(64% at k=20, 61% at k=40): wider retrieval crowds out the correct document.

## Quick start

```powershell
python preflight.py            # verify this machine can run it
python tools\selftest.py       # end-to-end: imports, benchmark, hidden set, docs

.\setup\setup_env.ps1          # venv + deps
Copy-Item .env.example .env    # add your GROQ_API_KEY (optional)

python run_benchmark.py                      # full public benchmark
python run_benchmark.py --limit 5            # quick smoke
python run_benchmark.py --no-llm             # deterministic baseline (no API calls)
python run_benchmark.py --no-llm --ablations # ablation study (which mechanism wins)
python tools\submit_hidden.py --no-llm       # 50 hidden questions -> submission file

python -m benchmark.dashboard_generator      # rebuild dashboard/index.html
```

New to this machine? Start with **[docs/MIGRATION.md](docs/MIGRATION.md)** — it
is a copy-paste path from `git clone` to a complete set of submission
artefacts, entirely offline.

## Round 2 — evolving and conflicting facts

`reasoning/conflicts.py` adjudicates competing versions of a fact by explicit,
explainable precedence:

| Rule | Beats | Example |
|---|---|---|
| `authority_correction` | everything | a doping disqualification reallocates a medal |
| `recency` | succession, majority | a 2012 Olympic record supersedes the 1996 one |
| `entity_succession` | majority | Soviet Union → Russia, Yugoslavia → Serbia |
| `majority` | — | undated venue-name disagreement |

Every decision carries the rule that produced it, the evidence it rested on,
and a confidence that drops when the fact was contested. Verify with
`python tools\demo_conflicts.py`.


### Two run modes

| Mode | Command | Speed | Use for |
|---|---|---|---|
| LLM-assisted | `python run_benchmark.py --out results/llm_results.json --summary results/llm_results_summary.json` | ~15–30 min (provider rate limits) | the headline numbers: LLM classification, slot extraction, answer adjudication, LLM-as-judge |
| Deterministic | `python run_benchmark.py --no-llm --out results/deterministic_results.json --summary results/deterministic_results_summary.json` | **~1 min** | reproducible baseline, verifying the harness, running with no quota |

Both modes run all three pipelines end-to-end — the LLM is an accelerator, never
a requirement. If the provider starts rate-limiting mid-run, a circuit breaker
trips and the remaining questions finish deterministically instead of stalling
(`llm.circuit` in `results/*_summary.json` shows how often this happened).

Keep the two modes in **separate result files**, because they are separate
claims. `results/llm_results.json` is the LLM-assisted run (the headline);
`results/deterministic_results.json` is the `--no-llm` baseline. Overwriting one
with the other is how a deterministic "0 LLM calls" table ends up published as an
LLM result — see below.

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

#### "Why does my dashboard show 0 LLM calls?"

Because the page was built from the wrong file. A `--no-llm` run writes
`"llm": {"available": false, "error": "no provider/API key configured"}` into its
own summary, and its per-record counters are zero by construction. If that file
is fed to the generator as `index.html`, the dashboard faithfully reports a
deterministic run — it is not a bug in the LLM path. Two checks settle it:

```powershell
python -c "import json;s=json.load(open('results/metrics_summary.json'));print(s['llm'])"
python tools/audit_results.py            # non-zero exit => numbers are not an LLM result
```

A live run looks like this instead (`avg_llm_calls` comes straight from the
summary, and every pipeline shows model output):

```
pipeline              n     acc  w/calls   tokens    out   p50 ms  unacct status
RAG                  10     30%       10     8536    591     1584       0 llm
GraphRAG             10     50%       10    17482    897     1465       0 llm
Agentic GraphRAG     10     90%        7    25132   1344    55031       0 partial-llm
```

The agentic pipeline shows `7/10` because its deterministic solvers legitimately
resolve some questions before the ReAct loop needs a call, and one record stopped
with `provider_unavailable` after the circuit breaker tripped — that is the
fallback working as designed, and the audit reports it as `partial-llm` rather
than hiding it.

If a **live** run still shows zero output tokens, the calls are reaching the
provider but the replies are unusable. One round-trip tells you which of the
three causes it is (no call made / reply not parseable / JSON with no answer):

```powershell
python tools/probe_llm_json.py            # verdict + token accounting
python tools/probe_llm_json.py --raw      # also dump the raw completion
```

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

Outputs land in `results/` — `llm_results.json` + `llm_results_summary.json`
(LLM-assisted) and `deterministic_results.json` +
`deterministic_results_summary.json` (`--no-llm`) — and the dashboards in
`dashboard/` (`index.html` for the LLM-assisted run, `baseline.html` for the
deterministic one; open either directly in a browser, no server needed).

Rebuild the dashboard from any results file:

```powershell
python -m benchmark.dashboard_generator                                   # results/public_results.json
python -m benchmark.dashboard_generator results/llm_results.json `
    --summary results/llm_results_summary.json --require-llm              # dashboard/index.html
python -m benchmark.dashboard_generator results/deterministic_results.json `
    --summary results/deterministic_results_summary.json `
    --name baseline.html                                                 # dashboard/baseline.html
python tools\validate_dashboard.py                                        # sanity-check the generated page
```

The page links `styles.css` / `dashboard.js` relatively and `--name` picks the
file inside `--out`, so one `dashboard/` directory can hold both runs:
`index.html` (LLM-assisted) next to `baseline.html` (deterministic). Pass
`--require-llm` on the headline page and the generator **refuses** to build it
from a file in which no pipeline recorded provider output — a deterministic run
can no longer be published as an LLM result by accident.

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

