# Agentic GraphRAG Hackathon — Top-Tier Implementation Plan

> **Goal:** Build a production-grade system that benchmarks RAG vs GraphRAG vs Agentic GraphRAG on the Olympics corpus, with full metrics, architecture, and demo — optimized to score maximally across all 6 judging criteria.

---

## Understanding the Problem Space

### Dataset Profile
| Asset | Details |
|---|---|
| **Corpus** | 2,951 Wikipedia articles about Olympic events (1987–2023), ~5.5M tokens, JSONL format |
| **Public Questions** | 100 questions with verified gold answers and gold_doc_ids |
| **Hidden Questions** | 50 questions without answers (system will be scored against held-out ground truth) |

### Question Type Distribution (Public)
| Type | Count | Avg Gold Docs | Complexity | Best Pipeline |
|---|---|---|---|---|
| `lookup` | 19 | 1.0 | Low — single doc, fact retrieval | RAG sufficient |
| `multi_hop` | 28 | 1.0 | Medium — venue+date→event→winner | GraphRAG shines |
| `temporal` | 22 | 2.0 | Medium — "Olympics before X" requires linking 2 events | GraphRAG/Agentic |
| `aggregation` | 21 | 15.1 | **High** — count across many docs, filter by threshold | **Agentic required** |
| `superlative` | 10 | 13.8 | **High** — find max/min across many docs | **Agentic required** |

> [!IMPORTANT]
> **Key Insight:** This distribution is designed to prove the hackathon thesis. ~31 questions (aggregation + superlative) genuinely need multi-step agentic reasoning across 10-20 docs. ~28 multi_hop and ~22 temporal need graph structure. ~19 lookup can be answered by simple RAG. This is the story our metrics dashboard should tell.

### TigerGraph GraphRAG Repo (v2.0.2)
The official repo already provides:
- **Agentic engine** with Planned (DAG-based) and Reactive (ReAct loop) modes
- **Tool registry** with: `get_schema`, `structural_retrieve`, `hybrid_search`, `similarity_search`, `contextual_search`, `community_search`
- **Retrievers**: HybridRetriever, SimilarityRetriever, CommunityRetriever, SiblingRetriever, EntityRelationshipRetriever
- **Docker Compose** with graphrag, ecc, chat-history, graphrag-ui, nginx

---

## User Review Required

> [!IMPORTANT]
> **Decision 1: TigerGraph Deployment** — Savanna (cloud, zero-install, credits provided) vs Community Edition (Docker, local). I recommend **Savanna for the actual deployment** + the **GraphRAG Docker stack pointing to Savanna**. This gives us cloud DB + local agentic service. If you want fully local, we can use Community Edition in Docker instead.

> [!IMPORTANT]
> **Decision 2: LLM Provider** — Which LLM API key do you have? The system supports OpenAI, Google GenAI, Azure, Bedrock, Groq, Ollama. I recommend **Google GenAI (Gemini 2.5 Flash)** for cost efficiency or **OpenAI GPT-4.1-mini** for reliability. Which do you prefer?

> [!IMPORTANT]
> **Decision 3: Custom Agentic Layer** — We can either:
> - **(A) Extend the existing TigerGraph agentic engine** (modify `agentic_agent.py`, `agentic_planner.py`, add specialized agents) — faster, uses their infra
> - **(B) Build a standalone Python orchestrator** that calls TigerGraph's GraphRAG API as a backend — more control, cleaner 3-pipeline comparison, more impressive for "Innovation" scoring
> - **(C) Hybrid approach (Recommended):** Use TigerGraph's GraphRAG stack for data ingestion + knowledge graph building + retrieval tools, but build our own orchestrator layer on top for the 3-pipeline comparison + benchmarking harness. This gives us the best of both worlds.

---

## Open Questions

> [!NOTE]
> 1. **Team size** — Are you solo or do you have teammates? This affects parallelization strategy.
> 2. **Timeline** — Submission deadline is Sep 24. That gives us ~5 days. Do you want to focus on Round 1 only or also prep for Round 2 (temporal reasoning)?
> 3. **Hardware** — Docker available on your machine? How much RAM? GPU?
> 4. **Existing Savanna account** — Do you already have TigerGraph Savanna credits, or do we need to set that up first?

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BENCHMARKING HARNESS                      │
│  (Python CLI — runs all 3 pipelines on all questions,       │
│   collects metrics, generates dashboard)                     │
├──────────┬──────────────┬────────────────────────────────────┤
│          │              │                                    │
│ Pipeline │  Pipeline 2  │       Pipeline 3                   │
│ 1: RAG   │  GraphRAG    │   Agentic GraphRAG                │
│          │              │                                    │
│ Vector   │ Hybrid       │ ┌──────────────────────┐           │
│ Search → │ Search +     │ │   Orchestrator Agent │           │
│ LLM      │ Graph        │ │   ┌───────────────┐  │           │
│ Answer   │ Traversal →  │ │   │  Question     │  │           │
│          │ LLM Answer   │ │   │  Classifier   │  │           │
│          │              │ │   └───────┬───────┘  │           │
│          │              │ │           │          │           │
│          │              │ │   ┌───────▼───────┐  │           │
│          │              │ │   │  Planner      │  │           │
│          │              │ │   │  (dynamic)    │  │           │
│          │              │ │   └───────┬───────┘  │           │
│          │              │ │           │          │           │
│          │              │ │  ┌────────┼────────┐ │           │
│          │              │ │  ▼        ▼        ▼ │           │
│          │              │ │ Entity  Graph    Vector          │
│          │              │ │ Linker  Traversal Search         │
│          │              │ │  Agent   Agent    Agent          │
│          │              │ │  ┌────────┼────────┐ │           │
│          │              │ │  ▼        ▼        ▼ │           │
│          │              │ │ Aggreg- Evidence  Gap            │
│          │              │ │ ation   Evaluator Detector       │
│          │              │ │  Agent   Agent    Agent          │
│          │              │ │           │          │           │
│          │              │ │   ┌───────▼───────┐  │           │
│          │              │ │   │  Synthesizer  │  │           │
│          │              │ │   └───────────────┘  │           │
│          │              │ └──────────────────────┘           │
└──────────┴──────────────┴────────────────────────────────────┘
                          │
              ┌───────────▼───────────┐
              │    TigerGraph Stack   │
              │  ┌─────────────────┐  │
              │  │ Knowledge Graph │  │
              │  │ (Entities,      │  │
              │  │  Relationships, │  │
              │  │  Communities)   │  │
              │  ├─────────────────┤  │
              │  │ Vector Store    │  │
              │  │ (Chunk          │  │
              │  │  Embeddings)    │  │
              │  ├─────────────────┤  │
              │  │ Document Store  │  │
              │  │ (2,951 articles)│  │
              │  └─────────────────┘  │
              └───────────────────────┘
```

---

## Proposed Changes

### Phase 1: Environment & Data Ingestion (Day 1)

#### [NEW] `setup/` — Environment Setup Scripts

- **`setup/setup_env.sh`** / **`setup/setup_env.ps1`**: Install Python 3.11+, create venv, install deps
- **`setup/config_template.json`**: TigerGraph + LLM config template
- **`setup/ingest_corpus.py`**: Script to load all 2,951 corpus docs into TigerGraph
  - Connects to TigerGraph (Savanna or local)
  - Initializes the GraphRAG schema (SupportAI schema + vector indices)
  - Ingests all corpus.jsonl documents
  - Triggers knowledge graph build (entity extraction, community detection)
  - This leverages the existing GraphRAG API endpoints

---

### Phase 2: Pipeline 1 — RAG (Day 1-2)

#### [NEW] `pipelines/rag_pipeline.py` — Pure Vector RAG

Simple pipeline:
1. Embed the question using the same embedding model
2. Vector similarity search against TigerGraph's vector store (chunk embeddings)
3. Retrieve top-k chunks
4. Feed question + chunks to LLM for answer generation

This uses TigerGraph's `SimilarityRetriever` directly (vector-only, no graph traversal).

**Key design:** Returns `PipelineResult` with: `answer`, `tokens_used`, `chunks_retrieved`, `citations`, `latency`

---

### Phase 3: Pipeline 2 — GraphRAG (Day 2)

#### [NEW] `pipelines/graphrag_pipeline.py` — Graph-Augmented RAG

Uses TigerGraph's `HybridRetriever`:
1. Embed question → vector search to find seed entities/chunks
2. Graph traversal (N hops) from seed nodes to gather related entities, relationships, communities
3. Community search for high-level summaries
4. Combine graph context + vector context → LLM for answer

This is the TigerGraph GraphRAG "Classic" engine in programmatic form.

**Key design:** Single-shot retrieval — no replanning or multi-step reasoning.

---

### Phase 4: Pipeline 3 — Agentic GraphRAG (Day 2-4) ⭐ Core Innovation

#### [NEW] `pipelines/agentic_pipeline.py` — Our Custom Agentic System

This is where we differentiate. Our orchestrator goes beyond the existing TigerGraph agentic engine:

##### 4a. Question Classifier Agent
```python
class QuestionClassifier:
    """Classifies question complexity to determine investigation strategy.
    
    Categories:
    - SIMPLE_LOOKUP: Direct fact retrieval (1 doc)
    - MULTI_HOP: Entity→relationship→entity chains (1-2 docs)
    - TEMPORAL: "Before/after X" requiring event sequencing (2 docs)
    - AGGREGATION: Count/filter across many docs (10-20 docs)
    - SUPERLATIVE: Find max/min across many docs (10-20 docs)
    """
```

##### 4b. Orchestrator Agent (Dynamic Planner)
```python
class OrchestratorAgent:
    """Plans investigation strategy based on question type + available evidence.
    
    For SIMPLE_LOOKUP:
        → SimilaritySearch → Answer (1 step)
    
    For MULTI_HOP:
        → EntityLinker → GraphTraversal → Answer (2-3 steps)
    
    For TEMPORAL:
        → EntityLinker → GraphTraversal → TemporalReasoner → Answer (3-4 steps)
    
    For AGGREGATION:
        → EntityLinker → BroadRetrieval → Aggregator → GapDetector → (loop) → Answer (5-10 steps)
    
    For SUPERLATIVE:
        → BroadRetrieval → Comparator → GapDetector → (loop) → Answer (5-10 steps)
    """
```

##### 4c. Specialized Agents

| Agent | Purpose | When Used |
|---|---|---|
| **EntityLinker** | Resolve entities from question to graph nodes (venue names, event names, dates → vertices) | multi_hop, temporal, aggregation |
| **GraphTraverser** | Walk the knowledge graph from linked entities, follow edges to related entities | multi_hop, temporal |
| **VectorSearcher** | Similarity search for relevant chunks when entities aren't directly found | All types as fallback |
| **TemporalReasoner** | Resolve "before/after" and "immediately before" temporal references | temporal |
| **Aggregator** | Collect and count/filter results across multiple retrieved docs | aggregation |
| **Comparator** | Compare values across docs to find max/min/superlatives | superlative |
| **EvidenceEvaluator** | Assess whether gathered evidence is sufficient to answer | All types |
| **GapDetector** | Identify what information is still missing and suggest next retrieval | aggregation, superlative |
| **Synthesizer** | Generate final grounded answer with citations | All types |

##### 4d. Agent Harness (State Management)
```python
class AgentState:
    """Tracks the investigation state across all steps.
    
    - question: str
    - plan: List[PlannedStep]
    - evidence: Dict[str, Any]  # accumulated evidence from each step
    - steps_executed: List[ExecutedStep]  # full trace
    - tokens_used: TokenCounter
    - missing_info: List[str]  # gaps identified
    - confidence: float  # 0-1, updated after each step
    - stop_reason: str  # why the agent decided to stop
    """
```

##### 4e. Stopping Criteria
The agent stops when:
1. Confidence ≥ 0.9 (evidence is sufficient)
2. Max steps reached (configurable, default 15)
3. No new information discovered in last 2 steps
4. All identified gaps have been addressed

---

### Phase 5: Benchmarking Harness (Day 3-4)

#### [NEW] `benchmark/` — Full Evaluation Suite

##### `benchmark/runner.py`
```python
class BenchmarkRunner:
    """Runs all 3 pipelines on all questions, collects comprehensive metrics."""
    
    def run_all(questions, pipelines):
        for q in questions:
            for pipeline in [rag, graphrag, agentic]:
                result = pipeline.run(q)
                metrics.record(q, pipeline, result)
```

##### `benchmark/metrics.py`
Per-question, per-pipeline metrics:

| Metric | Description |
|---|---|
| `answer` | Generated answer text |
| `is_correct` | Boolean — exact/fuzzy match against gold answer |
| `accuracy_score` | 0-1 from LLM-as-judge evaluation |
| `completeness_score` | 0-1 completeness of answer |
| `context_tokens` | Tokens in retrieved context |
| `input_tokens` | Total LLM input tokens |
| `output_tokens` | Total LLM output tokens |
| `total_tokens` | Grand total tokens |
| `latency_ms` | Wall-clock time |
| `num_retrieval_steps` | Number of retrieval operations |
| `retrieval_methods` | Which methods were used |
| `agents_invoked` | Which specialized agents ran (Agentic only) |
| `tools_called` | Which tools were called |
| `time_per_operation` | Breakdown of time per step |
| `tokens_per_operation` | Breakdown of tokens per step |
| `num_chunks` | Number of chunks retrieved |
| `citations` | Document IDs cited |
| `strategy_changed` | Whether investigation strategy adapted mid-run |
| `stop_reason` | Why the agent stopped |

##### `benchmark/evaluator.py`
LLM-as-judge evaluation:
1. **Exact match** — direct string comparison (normalized)
2. **Fuzzy match** — handles name variations, ordering differences
3. **LLM evaluation** — for complex answers, use an LLM to score [0-1] on correctness + completeness
4. **Citation check** — do cited doc_ids overlap with gold_doc_ids?

##### `benchmark/dashboard.py`
Generates a rich HTML metrics dashboard:
- Accuracy comparison bar charts (by question type × pipeline)
- Token efficiency scatter plots
- Agentic trace waterfall charts
- Per-question detailed comparison table
- Aggregate statistics with confidence intervals

---

### Phase 6: Metrics Dashboard (Day 4)

#### [NEW] `dashboard/` — Interactive Web Dashboard

A single-page HTML/JS/CSS dashboard (no framework) that visualizes:

1. **Pipeline Comparison Overview**
   - Side-by-side accuracy for RAG vs GraphRAG vs Agentic
   - Token cost comparison
   - Latency comparison

2. **Question Type Breakdown**
   - Grouped bar charts showing where each pipeline wins
   - Proves: lookup → RAG sufficient, multi_hop → GraphRAG shines, aggregation/superlative → Agentic necessary

3. **Agentic Investigation Traces**
   - Interactive timeline/waterfall for each agentic run
   - Shows which agents were invoked, what they found, strategy changes

4. **The "When Agents Matter" Analysis**
   - Scatter plot: question complexity vs accuracy delta (Agentic - GraphRAG)
   - Clear visual proof of where agentic adds value vs where it's overkill

---

### Phase 7: Demo & Documentation (Day 5)

#### [NEW] `docs/architecture.md` — Architecture Diagram

Full Mermaid-based architecture diagram showing:
- The 3 pipelines
- Data flow through TigerGraph
- Agent hierarchy and communication
- Evaluation flow

#### [NEW] `docs/README.md` — Project Documentation

- How to set up and run
- Architecture explanation
- Key results and findings
- Limitations and future work

---

## Project Structure

```
d:\gg\
├── README.md                      # Main project README
├── setup/
│   ├── setup_env.ps1              # Windows environment setup
│   ├── config_template.json       # TigerGraph + LLM config
│   └── ingest_corpus.py           # Corpus ingestion script
├── pipelines/
│   ├── __init__.py
│   ├── base.py                    # PipelineResult schema + base class
│   ├── rag_pipeline.py            # Pipeline 1: Pure RAG
│   ├── graphrag_pipeline.py       # Pipeline 2: GraphRAG
│   └── agentic_pipeline.py        # Pipeline 3: Agentic GraphRAG
├── agents/
│   ├── __init__.py
│   ├── state.py                   # AgentState + evidence tracking
│   ├── orchestrator.py            # Main orchestrator agent
│   ├── classifier.py              # Question complexity classifier
│   ├── entity_linker.py           # Entity linking agent
│   ├── graph_traverser.py         # Graph traversal agent
│   ├── vector_searcher.py         # Vector search agent
│   ├── temporal_reasoner.py       # Temporal reasoning agent
│   ├── aggregator.py              # Aggregation agent
│   ├── comparator.py              # Superlative comparison agent
│   ├── evidence_evaluator.py      # Evidence sufficiency evaluator
│   ├── gap_detector.py            # Information gap detector
│   └── synthesizer.py             # Answer synthesis agent
├── benchmark/
│   ├── __init__.py
│   ├── runner.py                  # Main benchmark runner
│   ├── metrics.py                 # Metrics collection
│   ├── evaluator.py               # LLM-as-judge evaluation
│   └── dashboard_generator.py     # Generates HTML dashboard
├── dashboard/
│   ├── index.html                 # Interactive metrics dashboard
│   ├── styles.css                 # Dashboard styling
│   └── dashboard.js               # Dashboard logic + charts
├── results/
│   ├── public_results.json        # Results on 100 public questions
│   ├── hidden_results.json        # Results on 50 hidden questions
│   └── metrics_summary.json       # Aggregate metrics
├── docs/
│   ├── architecture.md            # Architecture diagram
│   └── submission_writeup.md      # Hackathon writeup
├── graphrag/                      # TigerGraph GraphRAG repo (existing)
├── corpus-20260919T.../           # Corpus data (existing)
├── questions-20260919T.../        # Questions data (existing)
└── requirements.txt               # Python dependencies
```

---

## Verification Plan

### Automated Tests
1. **Unit tests** for each agent's core logic
2. **Integration tests** for each pipeline end-to-end on 5 sample questions
3. **Benchmark run** on all 100 public questions → verify accuracy against gold answers
4. **Token counting** verification — ensure metrics are captured accurately

### Manual Verification
1. Run all 3 pipelines on 100 public questions, verify accuracy ≥ 70% for Agentic
2. Run on 50 hidden questions, capture raw outputs for submission
3. Visual inspection of dashboard charts
4. Demo video recording of the system in action

### Success Metrics (Targets)
| Pipeline | Target Accuracy | Token Budget |
|---|---|---|
| RAG | ~40-50% (baseline) | Low |
| GraphRAG | ~55-65% | Medium |
| Agentic GraphRAG | **~75-85%** | Higher but justified |

The key deliverable is **proving the accuracy gap is worth the token cost**, especially on aggregation and superlative questions.

---

## Implementation Timeline

| Day | Tasks |
|---|---|
| **Day 1 (Sep 19)** | Environment setup, TigerGraph connect, corpus ingestion, knowledge graph build |
| **Day 2 (Sep 20)** | Pipeline 1 (RAG) + Pipeline 2 (GraphRAG) + start Pipeline 3 agents |
| **Day 3 (Sep 21)** | Pipeline 3 (Agentic) — orchestrator, specialized agents, harness |
| **Day 4 (Sep 22)** | Benchmark runner, evaluator, metrics dashboard, full benchmark run |
| **Day 5 (Sep 23)** | Polish dashboard, architecture diagram, demo video, writeup, submission |

---

## Innovation Highlights (for judging)

1. **Question Complexity Classifier** — Automatically routes questions to the cheapest pipeline that can answer them correctly
2. **Dynamic Investigation Planning** — The orchestrator builds a DAG of retrieval steps tailored to each question type
3. **Evidence Sufficiency Scoring** — Confidence-based stopping criteria prevent unnecessary token spend
4. **Gap Detection + Backfilling** — For aggregation queries, the system identifies which docs it's missing and specifically retrieves them
5. **"When Agents Matter" Analysis** — The dashboard proves, with data, exactly where agentic reasoning adds value and where simpler approaches suffice — directly answering the hackathon's headline question
6. **Cost-Benefit Visualization** — Scatter plots showing accuracy gain vs token cost for each question, making the tradeoff tangible
