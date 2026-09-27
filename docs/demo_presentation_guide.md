# TGCloud Hackathon — Demo Video Script & Presentation Guide

This guide provides the presentation narrative, timed video recording script, slide outline, and Q&A defense playbook for presenting **Autonomous Agentic GraphRAG on TigerGraph Cloud** to hackathon judges.

---

## 1. Executive Pitch & Headline Finding

> **The Core Research Question:**
> *"When does a complex question require an autonomous agentic investigation rather than a single GraphRAG or RAG retrieval? And are the additional reasoning and retrieval steps worth the token cost?"*

### The Verdict:
1. **RAG is a floor, not a baseline (42% overall)**: Simple vector retrieval completely collapses on set operations (0% on aggregation, 0% on superlatives). Top-$k$ chunks cannot capture an entire candidate set.
2. **GraphRAG buys the neighborhood, not the reasoning (62% overall)**: Adding 1-hop and 2-hop graph expansion improves entity linking and multi-hop lookups (86%), but still fails to compute cross-event extremes or aggregations (33% aggregation, 50% superlative). Expanding context without active reasoning also introduces noise that degrades single-fact lookup accuracy (74% vs 89% for RAG).
3. **Agentic GraphRAG decisively conquers set operations (100% overall)**: By querying structured TigerGraph vertices dynamically, the agent reaches 100% across all categories (21/21 aggregation, 10/10 superlatives, 28/28 multi-hop, 22/22 temporal, 19/19 lookups) while consuming **only 95 average context tokens** per question (vs 886 for GraphRAG).
4. **The Router is the production value pick (98% overall)**: Escalates only when the query demands multi-hop traversals or set operations, providing an optimal operational compromise between latency and accuracy.

---

## 2. 3–5 Minute Demo Video Script

| Timestamp | Visual Screen | Voiceover Script |
|---|---|---|
| **0:00 - 0:35** | **Title Slide & Architecture Diagram (`docs/architecture.md`)** | "Welcome judges. In this hackathon, we set out to answer the fundamental question facing enterprise RAG systems today: When do agents actually matter, and what do they cost? We built four pipelines—Naive RAG, single-shot GraphRAG, Autonomous Agentic GraphRAG, and a capability-based Router—all connected live to a TigerGraph 4.2.5 Enterprise cluster on TGCloud, driven by Gemini 3.8 Flash." |
| **0:35 - 1:20** | **Terminal: Live TGCloud Verification (`tools/tg_ingest.py --status`)** | "Here you can see our live TigerGraph Cloud cluster `OlympicsKG`. We deployed a complete graph schema with 2,210 event nodes, over 5,000 athlete vertices, and 16,000 typed edges. We push query operations directly to TigerGraph via RESTPP v2 and installed GSQL queries like `tg_filter_events` and `tg_neighbours`. Every answer is verified against dual-backend parity guards." |
| **1:20 - 2:30** | **Dashboard (`dashboard/index.html`): Accuracy & Token Scatter** | "Looking at the public benchmark of 100 questions, the contrast is stark. Naive RAG scores 42%, failing completely on aggregation and superlatives because top-$k$ text chunks cannot perform counting or comparison. GraphRAG improves to 62% through graph neighborhood expansion, but tops out because more text context does not equal mathematical reasoning. Agentic GraphRAG achieves a perfect 100% accuracy, while using *9× fewer context tokens* because it queries structured graph vertices on demand." |
| **2:30 - 3:15** | **Dashboard: Trace Waterfall, Subgraph Explorer & Conflict Matrix** | "Let's inspect an investigation trace. You can see our specialized collaborative agent personas in action: the GraphNavigatorAgent discovers seed events, the TemporalAuditorAgent analyzes chronologies, and the ConflictAdjudicatorAgent resolves disputed facts. Below the waterfall, our zero-dependency SVG Subgraph Traversal Network maps topological graph paths across Event, Sport, Venue, and Medallist nodes. In Round 2, our 4-tier Conflict Matrix demonstrates authority corrections, recency, and entity succession on real Olympic controversies." |
| **3:15 - 3:50** | **Live Studio & Model Context Protocol (MCP) Server** | "For interactive evaluation, we built a Live Investigation Studio directly into the dashboard. Judges can test custom Olympic questions and inspect real-time Router dispatch and token savings. Furthermore, our native Model Context Protocol (MCP) server enables any external agent—Claude Desktop, Cursor, LangChain—to query TigerGraph Cloud over standard JSON-RPC 2.0 stdio." |
| **3:50 - 4:25** | **Hidden Benchmark Dashboard (`dashboard/dashboard_hidden_llm.html`)** | "On the 50-question held-out evaluation set, our agentic system resolved 100% of all slots with zero errors across 1.07 million tokens. Zero data leakage was audited across static AST checks and corrupted ground truth tests. The system answers purely from graph and document evidence." |
| **4:25 - 4:50** | **Conclusion & Takeaways** | "In conclusion: Agents are overkill for single-fact lookups, where cheap RAG already scores 89%. But for set operations, aggregations, multi-hop traversals, and conflicting data, Agentic GraphRAG on TigerGraph is indispensable. Thank you." |

---

## 3. Live Judging Q&A Defense Playbook

### Q1: "How do you know the agent isn't just memorizing or hallucinating answers?"
**Answer:**
> "We enforce three rigorous guards:
> 1. **Zero Data Leakage Audit (`tools/audit_leakage.py`)**: We verified by AST analysis that inference code never accesses gold labels, and by behavioural testing that corrupting gold answers produces identical pipeline predictions.
> 2. **Deterministic Solvers for Arithmetic**: The LLM plans and phrases; it does not hallucinate numbers or dates. Counts, superlatives, and temporal chains are executed over the TigerGraph knowledge graph.
> 3. **Citation & Evidence Grounding**: Every answer is required to supply document citations (`citations`), extracted evidence snippets, and residual uncertainty (`uncertainty = 1 - confidence`)."

### Q2: "Why does GraphRAG use more context tokens than Agentic GraphRAG?"
**Answer:**
> "GraphRAG is single-shot: to answer a question, it expands the entity neighborhood and stuffs up to 10 chunks plus related graph facts into one large prompt (averaging 886 context tokens per question). In contrast, Agentic GraphRAG re-queries the graph with surgical filters (`get_event_values`, `search_events`), retrieving only specific vertex attributes (averaging only 95 context tokens per retrieval). The agent re-queries more often, but reads far less irrelevant text."

### Q3: "How does your system handle Round 2 (evolving and conflicting facts)?"
**Answer:**
> "We implemented `reasoning/conflicts.py` with an explainable 4-tier precedence hierarchy:
> 1. `authority_correction`: Disqualifications, medal strippings, or doping revisions supersede earlier podium records.
> 2. `recency`: More recent dated records supersede older editions.
> 3. `entity_succession`: Historical entities map to recognized successor nations (e.g., Soviet Union → Russia).
> 4. `majority`: Used only when competing versions have no dating or authority markers.
> When a conflict is detected, the agent records the superseded versions, logs the adjudication rule in the trace, and lowers the confidence score to reflect residual uncertainty."

### Q4: "Can external agentic frameworks (Cursor, Claude Desktop, CrewAI) integrate with your TigerGraph system?"
**Answer:**
> "Yes, natively. We built an official Model Context Protocol (MCP) server in `tools/mcp_server.py`. Over standard JSON-RPC 2.0 stdio, it exposes our installed TigerGraph 4.2.5 GSQL queries (`tg_filter_events`, `tg_neighbours`, `tg_aggregate_stats`) and reasoning engines (`detect_conflicts`, `agentic_investigate`). Any MCP client can connect with a single configuration line in their `claude_desktop_config.json`."

### Q5: "How does your Live Investigation Studio work for un-indexed or unseen judge questions?"
**Answer:**
> "The Live Studio in `dashboard/index.html` features dual modes: for benchmark questions, it loads real verified execution traces and TGCloud server logs; for arbitrary judge queries, it performs real-time capability classification (aggregation, superlative, temporal, multi-hop, lookup), computes the optimal token-efficient route, and simulates the multi-agent persona collaboration steps."

### Q6: "How can Accuracy be 100% if Completeness is 97.6%?"
**Answer:**
> "Accuracy and Completeness measure two fundamentally different properties in our evaluation harness (`benchmark/evaluator.py`):
> 1. **Accuracy (100.0%)** measures **factual correctness**: did the pipeline correctly answer the question and identify the true target Olympic entity, number, venue, or athlete? Evaluated via deterministic normalization, fuzzy containment, and strict LLM-as-judge validation, 100/100 answers are factually verified.
> 2. **Completeness (97.6%)** measures **token-level SQuAD $F_1$ lexical overlap** against raw scraped Wikipedia gold strings.
> The delta is caused entirely by surface formatting and punctuation artifacts in scraped Wikipedia strings. For example, in `pub-015`, the raw Wikipedia gold answer concatenated athlete names without spaces (`'Dani KingLaura TrottJoanna Rowsell'`). The agent cleanly outputted `'Dani King, Laura Trott and Joanna Rowsell'`. The factual entity match is 100% correct, but token-level F1 is penalised due to whitespace separation. Similarly, in `pub-004` and `pub-084`, the gold strings included redundant article-title prefixes (`'Athletics at the 2008 Summer Olympics – Men\'s marathon'`), whereas the agent returned the precise event requested (`'Men\'s marathon'`)."

### Q7: "Why does the Router report 98% in some places and 100% in others?"
**Answer:**
> "Our terminology distinguishes two precise stages of routing architecture:
> 1. **Initial Capability Routing achieves 98.0% (98/100)**: Pre-routing dispatches each question by $O(1)$ structural complexity (`lookup` $\to$ RAG, `multi_hop` $\to$ GraphRAG, set operations $\to$ Agentic GraphRAG). On 100 public questions, initial dispatch gets 100% on multi-hop (28/28), temporal (22/22), aggregation (21/21), and superlatives (10/10), but 17/19 on single-shot lookups because RAG missed vector chunks for `pub-042` and `pub-047`.
> 2. **Confidence-Based Adaptive Escalation achieves 100.0%**: When the fast arm returns confidence below 0.85 or empty evidence, the Router automatically escalates to Agentic GraphRAG, recovering the 2 failed lookups.
> We honestly report the static 98.0% benchmark to preserve baseline transparency, while demonstrating that dynamic escalation provides a 100% safety net."

### Q8: "Why should I use a 13-second agent when GraphRAG gives me an answer in under 2 seconds?"
**Answer:**
> "A 1.9-second answer is useless if it is wrong. In our public benchmark:
> - On **Aggregation** ($n=21$): GraphRAG gets **33.3%** accuracy (RAG gets **0.0%**).
> - On **Superlatives** ($n=10$): GraphRAG gets **50.0%** accuracy (RAG gets **0.0%**).
> - On **Temporal chains** ($n=22$): GraphRAG gets **54.5%** accuracy.
> GraphRAG is sub-2-second only because it executes a single-shot heuristic neighborhood expansion. It structurally hits a ceiling because top-$k$ text chunks cannot perform counting or comparison over open candidate sets.
> In contrast, Agentic GraphRAG takes 12.9s because it conducts an **autonomous multi-turn investigation** (averaging 5.6 LLM turns, dynamic tool calling across `tg_filter_events`, `tg_neighbours`, `get_event_values`, and `detect_conflicts`).
> Crucially, **the Router eliminates unnecessary agent latency**: on single-fact `lookup` queries, the Router bypasses the agent loop entirely, cutting latency to 4.2s and reducing token consumption by **83.8%** (from 8,390 to 1,362 tokens). You only pay for the agent when the query's structural complexity demands it."

---

## 4. Social Media Announcement Template (X / LinkedIn)

```
Thrilled to submit our Autonomous Agentic GraphRAG system for the @TigerGraph Hackathon! 🐯🚀

We tackled the core question: When do agents actually matter in RAG?
Comparing RAG vs GraphRAG vs Agentic GraphRAG live on TigerGraph Cloud 4.2.5 with Gemini:

📊 100% on 100-Q public benchmark & 50/50 on hidden benchmark
🧠 Proved agents decisively conquer set operations (100% vs 0% RAG)
⚡ Agentic uses 9× fewer context tokens (95 vs 886) via structured KG tools
⚖️ Round 2 conflict resolution with explainable 4-tier precedence
📈 Zero-dependency interactive dashboards with replayable traces

Repo: https://github.com/PARZIVALPRIME/agentic-ping

#TigerGraph #GraphRAG #AI #KnowledgeGraph #LLMs #AgenticAI #GenAI
```
