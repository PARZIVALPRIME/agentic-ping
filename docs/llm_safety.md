# LLM safety: why enabling the model cannot lower the score

Every headline number in this repo was measured with `--no-llm`. That is an
honest measurement of the deterministic core, but it leaves a real question
unanswered: *what happens when a small local model is switched on?*

This is not a rhetorical concern. A 4B model is not a neutral addition. It is a
component that returns short, fluent, confident answers that are frequently
wrong — which is precisely the shape of output that a naive "let the LLM
double-check the answer" design will accept.

We measured it instead of assuming.

## The measurement

`tools/audit_llm_robustness.py` substitutes stub models that report
`available=True` and then behave as badly as a model plausibly can, and asserts
accuracy never falls below the `--no-llm` floor:

| adversary | behaviour |
|---|---|
| `always_disagree` | returns a confidently wrong short answer every time |
| `prose` | answers counting questions in prose ("There are several…") |
| `empty` | returns empty / malformed JSON |
| `truncated` | emits a 400-char hedging ramble, as a model that overruns its budget does |

## What it found (before the fixes)

The audit was not a formality. On first run it failed **8 of 12** cells:

| adversary | RAG | GraphRAG | Agentic |
|---|---|---|---|
| `always_disagree` | 20% → **5%** | 65% → **20%** | 100% → **0%** |
| `prose` | 20% → **5%** | 65% → **20%** | 100% → **0%** |
| `empty` | 20% | 65% | 100% → **0%** |
| `truncated` | 20% | 65% | 100% → **0%** |

A bad model did not merely fail to help — it **destroyed** a 100% pipeline.
Had this been run blind on the submission machine, the LLM run would have
scored far below the deterministic run we had already validated, and the cause
would have looked like a mystery.

Three distinct defects were responsible.

## The three fixes

**1. Adjudication was ungrounded** (`utils/llm.py::refine_answer`)

The adjudicator's job is to pick the right answer *out of the context it was
shown* — it is not licensed to invent a new one. A replacement is now accepted
only if it actually occurs in that context (`_grounded`). Numbers are matched
as standalone tokens, so `18` is not "supported" by the `1980` in a nearby
date.

This is the single most valuable guard for a small model, because the
characteristic 4B failure — a fluent, short, unsupported answer — is
indistinguishable from a genuine correction by *any other test*. A real
correction is by definition quoted from the evidence, so it passes untouched.

**2. ReAct prose was accepted as an answer** (`agents/react_agent.py`)

Replying in prose instead of calling `submit_answer` already means the model
lost the protocol. Taking the last line of a hedging paragraph as the answer
scored zero *and* suppressed the deterministic fallback — failing worse than
not trying at all. `_answer_shaped()` now rejects hedging phrasing and anything
longer than a span (gold answers are names, years, countries, counts).

**3. `hybrid` mode rescued on emptiness, not on evidence**
(`agents/orchestrator.py`)

`hybrid` promises "the LLM drives; deterministic solvers rescue empty answers".
But an answer the model *typed* never passed through a tool, so nothing in the
system had checked it against the graph. Only an answer submitted via the
`submit_answer` tool has been through the grounded path, so only that one now
skips the deterministic solvers.

Plus two guards added earlier in the same pass:

- **exhaustive answers are never adjudicated.** A count over 60+ enumerated
  events cannot be validly checked against the six passages that fit in a
  prompt. The model sees six documents, cannot see the other fifty-four, and
  will still return a confident different number. That is not verification —
  it is inviting a hallucination to overwrite arithmetic that is correct by
  construction.
- **a numeric candidate may only be replaced by another number.**

## After

```
 adversary 'always_disagree'   RAG 20% (+0%)   GraphRAG 65% (+0%)   Agentic 100% (+0%)
 adversary 'prose'             RAG 20% (+0%)   GraphRAG 65% (+0%)   Agentic 100% (+0%)
 adversary 'empty'             RAG 20% (+0%)   GraphRAG 65% (+0%)   Agentic 100% (+0%)
 adversary 'truncated'         RAG 20% (+0%)   GraphRAG 65% (+0%)   Agentic 100% (+0%)

 PASS - no adversary can score below the deterministic baseline.
```

## The check that keeps this honest

"Nothing gets worse" is trivially satisfied by a guard that ignores the model
entirely. So the audit also asserts the opposite direction — a correct,
context-supported correction must still be **accepted**:

```
 ok   accepts a grounded correction            -> 'Larisa Latynina'
 ok   accepts a grounded numeric correction    -> '18'
 ok   rejects an ungrounded name               -> [verdict_rejected_ungrounded]
 ok   rejects prose over a number              -> [verdict_rejected_non_numeric]
 ok   skips adjudication for aggregation       -> [adjudication_skipped_exhaustive]
 ok   skips adjudication when exhaustive       -> [adjudication_skipped_exhaustive]
 ok   does not match 18 inside 1980            -> [verdict_rejected_ungrounded]
```

The model is **constrained, not neutered**.

## Scope and honest limits

- This establishes a **floor, not a prediction**. A real model lands somewhere
  between the adversary and an oracle; the guarantee is that the floor is the
  deterministic score, so enabling the LLM is upside-only.
- The adversaries are stylised. A model that returns a *grounded but wrong*
  span — one that genuinely appears in the retrieved context — defeats the
  grounding check by construction. That is the residual risk, and it is the
  reason the deterministic path, not the model, remains the primary answerer.
- Measured on 20 public questions (`--limit 20`) for runtime. The baseline
  column differs from the 100-question headline for that reason.

## Reproduce

```bash
python tools/audit_llm_robustness.py --limit 20
python tools/selftest.py            # includes it as a gating stage
```
