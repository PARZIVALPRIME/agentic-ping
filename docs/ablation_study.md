# Ablation Study: attributing the agentic gain

A three-bar chart shows *that* the agent wins. It does not show *why*, and a
judge is entitled to suspect the gap comes from one hard-coded solver rather
than from agency. This study removes one mechanism at a time and measures what
breaks.

Reproduce with:

```powershell
python run_benchmark.py --no-llm --no-tg --ablations `
  --out results/ablation_study.json --summary results/ablation_summary.json
```

## Headline results (public set, n=100, deterministic)

| Pipeline | Overall | lookup | multi_hop | temporal | aggregation | superlative | avg tokens |
|---|---|---|---|---|---|---|---|
| RAG | 39% | 84% | 61% | 27% | 0% | 0% | 278 |
| GraphRAG | 61% | 74% | 86% | 50% | 33% | 50% | 914 |
| **Agentic GraphRAG** | **100%** | 100% | 100% | 100% | 100% | 100% | **0** |
| Router | 100% | 100% | 100% | 100% | 100% | 100% | 0 |
| *Agentic − planner* | 0% | 0% | 0% | 0% | 0% | 0% | 0 |
| *Agentic − enumeration* | 86% | 100% | 100% | 100% | **38%** | **90%** | 0 |
| *Agentic − verifier* | 100% | 100% | 100% | 100% | 100% | 100% | 0 |
| *Agentic − gap detector* | 100% | 100% | 100% | 100% | 100% | 100% | 0 |

## What each ablation proves

### 1. Planning is load-bearing (−100 points)

`Agentic-NoPlanner` replaces the five type-specific plan templates with one
generic plan (link → traverse → search → audit → answer) and scores **0%**.

The agentic gain is not "better retrieval plus a good solver". The solver only
fires because the planner selected it for this question type. Remove the
routing of question → strategy and the system answers nothing at all, despite
having every tool still available.

### 2. Enumeration is the mechanism behind the headline claim (−14 points, concentrated)

`Agentic-NoEnumeration` is the decisive experiment. It keeps planning, tools,
gap detection and verification, and removes **only** the ability to see the
full candidate set — the agent is capped at the same top-10 a retriever sees.

The damage is precisely where the thesis predicts:

| qtype | full agent | capped at top-10 | delta |
|---|---|---|---|
| lookup | 100% | 100% | 0 |
| multi_hop | 100% | 100% | 0 |
| temporal | 100% | 100% | 0 |
| **aggregation** | 100% | **38%** | **−62** |
| **superlative** | 100% | **90%** | **−10** |

Lookup and multi-hop are untouched, because their answers live in one or two
documents that top-k retrieval reaches. Aggregation collapses, because the
answer is a **function over a candidate set**: with only 10 documents visible
the agent cannot count what it cannot see.

This isolates the mechanism to a single variable while holding every other
agentic capability fixed.

> **Scope of this claim.** An earlier draft said "no top-k retriever can win at
> any *k*". The [baseline ceiling study](baseline_ceiling.md) refutes that: RAG
> reaches 91% at k=160. The surviving claim is about *cost*, not capability —
> retrieval buys that accuracy with 39× the context tokens (9,680/question vs
> the agent's 0), and still tops out at 76% on aggregation.

### 3. Verification and gap detection are inert here (0 points) — a negative result

`Agentic-NoVerifier` and `Agentic-NoGapDetector` both score **100%**: removing
them changes nothing on this corpus.

We keep this in the study rather than quietly dropping it. The honest reading:
the deterministic solvers read from graph structure that is either present or
absent, so there is no noisy intermediate result for a second pass to correct,
and the first plan is already sufficient. On this corpus, self-verification
earns zero points.

Both mechanisms stay in the default pipeline because they cost ~0 tokens and
are what would catch a regression on a noisier or conflicting corpus (see
[Round 2](#round-2)). But we do not claim accuracy points for them.

## The token result is the counter-intuitive one

Agentic GraphRAG is not only the most accurate pipeline — it is the
**cheapest**, at 0 prompt tokens/question versus GraphRAG's 914.

The reason: RAG and GraphRAG pay tokens to stuff retrieved passages into a
prompt and still miss, while the agent answers from graph structure. The usual
accuracy-versus-cost trade-off does not appear on this corpus. Retrieval
pipelines pay *per question* for accuracy they never reach.

## Why the router routes everything to the agent

The router's table is **derived from measurement**, not assumed
(`pipelines/router_pipeline.py`, `MEASURED_ACCURACY`). It sends each question
type to the cheapest pipeline within 2 points of the best.

Our first table assumed "simple questions don't need an agent" and sent
`lookup`→RAG, `temporal`→GraphRAG. Measured, that router scored **82%** — it
gave away 18 accuracy points to save tokens it did not need to save. The
corrected, data-derived table routes everything to the agent and scores 100%.

**When is an agent overkill?** On this corpus: never. But the router is the
mechanism that would detect it instantly — flip one table entry and the saving
is immediate. The contribution is the *measurement discipline*, not the table.

## Round 2

Conflict handling (`reasoning/conflicts.py`) resolves competing versions of a
fact by explicit precedence: authority correction → recency → entity
succession → majority. Every decision carries the rule that produced it and
the evidence it rested on. Verify with:

```powershell
python tools/demo_conflicts.py     # the resolver, rule by rule
python tools/test_round2.py        # the wiring: dates, audit, gap, tool
python tools/_smoke_conflicts.py 1 # a real agentic run's metadata + trace
```

### What reaches a run

The resolver is not a standalone study: every stage of a run now carries the
version signal and the verdict.

| Stage | What it carries |
| --- | --- |
| `kg/builder.py` | every fact gets `source_type` and an "as of" `fact_version_date` ("28 July 2012" → `2012-07-28`, the Games year when the day is not stated). A stated year that is neither the edition's nor the next one is discarded: two pages in the corpus date themselves to an unrelated edition, and a version date like that would let a page supersede an edition it never took part in. |
| `agents/orchestrator.py` | after the answer is finalised, the documents that *attest* it plus the model's own reading of the passages are handed to the resolver. The verdict is written to the trace, to the observation and to `state.fact_conflicts`; the confidence of the run is capped by the confidence of the adjudication, and `state.uncertainty` (1 − confidence) travels with the result. |
| `agents/gaps.py`, `agents/gap_detector.py` | an adjudication that no rule settled — the versions disagree and neither dates nor authority separate them — is recorded as the `conflicting_evidence` gap, for which the gap detector has a targeted recovery (retrieve the authoritative/superseding passage). |
| `agents/tools.py`, prompt | a `detect_conflicts` tool lets the *agent* hand the versions it saw to the same resolver, and the ReAct prompt tells it to use the tool rather than pick silently. |
| `pipelines/base.py`, `pipelines/agentic_pipeline.py` | every pipeline result reports `uncertainty`, and the agentic metadata reports `conflicts` + `uncertainty`. |

### What it deliberately does not do

- **It never rewrites the answer.** A conflict is reported, priced into the
  confidence and offered to the gap recovery; the grounded answer is only ever
  revised by the adjudication step that already existed.
- **It does not audit derived answers.** A count or an extreme is computed over
  a candidate set in which every document holds a different value by
  construction; comparing those values labelled *every* aggregation "majority,
  undecided" and cut a correct answer's confidence from 0.97 to 0.60 (measured on
  the public set). Derived answers and answers that cannot be attributed to a
  field of the documents that cite them are therefore left unaudited.
- **It does not compare values across editions automatically.** 415 of the 535
  event groups change venue across editions, and the 6 pairs the infobox alias
  field appeared to link were generic-name collisions ("Olympic Stadium" in two
  different cities) — two editions are usually two facilities or two results, not
  two names for one fact. Cross-edition versions the agent has actually observed
  go through `detect_conflicts`, where the model names them.
- **The KG cache is versioned** (`KG_SCHEMA_VERSION`): an older cache is rebuilt
  instead of loaded, because a missing field and an empty field are
  indistinguishable to every consumer — the first build after this change
  silently produced a graph whose facts had no version date at all.

The remaining gap is a *counting* consequence: two different states in one
successor lineage (the Soviet Union and Russia, say) are counted as two
entities. Deciding whether they are one is an answer-level judgement, not a
reporting one, so it is left out rather than guessed.
