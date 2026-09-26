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
| `wrong_qtype` | answers the *classification* request with a legal but wrong label (`lookup`) every time |

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

Five distinct defects were responsible.

## The fixes

**1. The parser credited a slot-less reply with the prompt's defaults**
(`reasoning/query_parser.py::parse_question_semantic`)

This is the defect the audit caught a *second* time, on the branch that added
the semantic parser, and it is the most instructive one.

The parse request shows the model a JSON skeleton, so a compliant model writes
`qtype: "lookup"` for "I see no category". When reading the reply, the parser
fell back to `SLOT_DEFAULTS` for every field the model omitted -
`payload.get(field, SLOT_DEFAULTS[field])`. So a reply with **no slots at all**,
like the adjudication-shaped `{"answer": "Vladimir Smirnov", "agree": false}`
that a confused small model actually produces, was read as the model *choosing*
`qtype = "lookup"`.

That invention then survived the deterministic fill, because the fill test was
"is the slot empty?" and the slot now held `"lookup"`. The consequence was not
a wrong answer but **no answer**: an aggregation question was planned as a
single-article lookup, the article could not be resolved, and the run returned
the empty string. All three non-empty adversaries scored the Agentic pipeline at
15% instead of 100%.

Two invariants replaced it:

- a field the model did not return is *missing*, and missing is filled by the
  templates - so a slot-less reply is credited with nothing (it no longer even
  appears in `corpus_validated`, which is what made the old behaviour
  misattribute the qtype in the run trace);
- the fill is tracked from what the model actually supplied (and the corpus
  accepted) rather than from "the slot is empty", so the dataclass placeholders
  (`qtype="unknown"`, `comparator="gt"`, `direction="max"`) and corpus-rejected
  values can no longer block it. A fill that would not change the spec is not
  recorded, so the report still says the model produced the slots when it did.

**2. Adjudication was ungrounded** (`utils/llm.py::refine_answer`)

The adjudicator's job is to pick the right answer *out of the context it was
shown* — it is not licensed to invent a new one. A replacement is now accepted
only if it actually occurs in that context (`_grounded`). Numbers are matched
as standalone tokens, so `18` is not "supported" by the `1980` in a nearby
date.

This is the single most valuable guard for a small model, because the
characteristic 4B failure — a fluent, short, unsupported answer — is
indistinguishable from a genuine correction by *any other test*. A real
correction is by definition quoted from the evidence, so it passes untouched.

**3. ReAct prose was accepted as an answer** (`agents/react_agent.py`)

Replying in prose instead of calling `submit_answer` already means the model
lost the protocol. Taking the last line of a hedging paragraph as the answer
scored zero *and* suppressed the deterministic fallback — failing worse than
not trying at all. `_answer_shaped()` now rejects hedging phrasing and anything
longer than a span (gold answers are names, years, countries, counts).

**4. `hybrid` mode rescued on emptiness, not on evidence**
(`agents/orchestrator.py`)

`hybrid` promises "the LLM drives; deterministic solvers rescue empty answers".
But an answer the model *typed* never passed through a tool, so nothing in the
system had checked it against the graph. Only an answer submitted via the
`submit_answer` tool has been through the grounded path, so only that one now
skips the deterministic solvers.

**5. A wrong-but-legal question type could still cost the question**
(`agents/orchestrator.py`)

Fix 1 stops a slot-less reply from inventing a question type. It does not stop a
model that returns a *valid* one and is simply wrong - and that is the most
common way a small model fails, because "lookup" is a perfectly good answer to
"which category is this?". No validator can reject it; the classifier's
structural override catches only the counting/extreme/edition questions.

So the guard is evidence-based, like the others: after the deterministic path
runs, if the run has **no answer at all** and the parse had read the question as
a different type from the templates, the whole deterministic path is retried
once under the templates' reading - with the question type pinned so the model's
own vote cannot walk it back, with the abandoned attempt's answer, gaps and
candidate set cleared so the second reading is judged on its own results, and
with a fresh step budget, because the first reading has already spent it.

Three properties make this safe to ship:

- it fires on a **result**, not on suspicion, so a model reading that works is
  still kept and still measured (which is what the ablation study exists to
  measure);
- it can only turn an empty answer into a non-empty one;
- it is unreachable without an LLM, because with no model the parse returns the
  template spec and `spec.qtype` always equals the templates' `rule_qtype` - so
  the deterministic arms' published numbers cannot move.

Measured on the `wrong_qtype` adversary (public set, 20 questions): **15% -> 100%**
for the Agentic pipeline, with the retry alone accounting for 75% -> 100%.

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
 adversary 'wrong_qtype'       RAG 20% (+0%)   GraphRAG 65% (+0%)   Agentic 100% (+0%)

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
