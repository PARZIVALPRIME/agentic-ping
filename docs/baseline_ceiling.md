# Baseline ceiling study: do the baselines just need a bigger *k*?

## The critique this answers

> "Your baselines score 39% and 61% because you crippled them. Retrieve more
> documents and they would catch up."

That is a fair challenge and it deserves a measurement, not an argument.

```powershell
python tools/baseline_sweep.py --ks 5,10,20,40,80,160
```

## Result: the critique partly lands, and we were wrong

Our first write-up claimed *"no top-k retriever can win at any k"*. **The data
refutes that.** RAG reaches **91%** at k=160. The strong structural claim was
false and has been removed.

### RAG

| k | accuracy | ctx tokens/q | citation recall | aggregation |
|---:|---:|---:|---:|---:|
| 5 | 39% | 251 | 67% | 0% |
| 10 | 52% | 518 | 76% | 0% |
| 20 | 61% | 1,074 | 83% | 0% |
| 40 | 71% | 2,251 | 93% | 19% |
| 80 | 80% | 4,678 | 98% | 48% |
| **160** | **91%** | **9,680** | **100%** | 76% |

### GraphRAG

| k | accuracy | ctx tokens/q | citation recall | aggregation |
|---:|---:|---:|---:|---:|
| 5 | 45% | 440 | 56% | 0% |
| 10 | 61% | 886 | 70% | 33% |
| 20 | 64% | 1,768 | 73% | 38% |
| 40 | 61% | 3,511 | 72% | 24% |
| 80 | 66% | 6,958 | 77% | 33% |
| **160** | **74%** | **13,785** | **86%** | 43% |

## What the sweep actually shows

**1. The default k=5 was under-provisioned.** RAG gains +52 points from k=5→160.
Any comparison at k=5 alone would have been unfair, and we publish the corrected
baselines as the honest comparison point.

**2. The real variable is cost, not capability.** RAG buys its 91% with **39×
more context** — 9,680 tokens per question versus 251. The agent reaches 100% at
**0** prompt-context tokens, because it reads graph structure rather than
stuffing passages into a prompt.

| Pipeline | Accuracy | ctx tokens/q |
|---|---:|---:|
| RAG (best, k=160) | 91% | 9,680 |
| GraphRAG (best, k=160) | 74% | 13,785 |
| **Agentic GraphRAG** | **100%** | **0** |

**3. GraphRAG is non-monotonic.** Accuracy *drops* from 64% (k=20) to 61%
(k=40) and lookup accuracy degrades from 89% → 63% before recovering. Graph
expansion on top of a wide retrieval pulls in neighbours that crowd out the
correct document. More context is not monotonically better — a result that only
appears if you actually sweep.

**4. Aggregation is where retrieval pays the most and gains the least.** RAG
needs k≥40 before scoring anything at all, and is still at 76% with 160
documents in context. GraphRAG never exceeds 43% at any k. These questions are
a function over a candidate set, so a retriever must hold the *entire* set in
context to answer, while the agent enumerates it from the graph.

## Revised claim

The defensible statement is **not** "retrieval cannot do this". It is:

> Retrieval-based pipelines can approach agentic accuracy on this corpus, but
> only by paying 30–40× the context cost, and they remain weakest exactly where
> the answer is a function over a candidate set. The agent reaches higher
> accuracy at zero context cost, because it reads structure instead of text.

This is a weaker claim about *capability* and a stronger one about *efficiency*
— and unlike the original, it survives the sweep.

## Caveat

These are deterministic runs: the extractive answerer reads infobox fields from
whatever is in context. A frontier LLM given 9,680 tokens might do better than
our extractor at k=160 — the RAG curve here is a *lower* bound on what a
large-context LLM pipeline could achieve. What would not change is the cost
ratio, which is a property of the retrieval strategy rather than of the reader.
