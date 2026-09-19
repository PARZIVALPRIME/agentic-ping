"""Generates the interactive HTML metrics dashboard from benchmark results.

Inputs (produced by ``benchmark.runner``):
- ``results/public_results.json``   per-question records (all pipelines)
- ``results/metrics_summary.json``  aggregate tables

Output:
- ``dashboard/index.html`` - a single self-contained page. Chart logic lives
  in ``dashboard/dashboard.js`` and styling in ``dashboard/styles.css``; the
  generator inlines both plus the result data so the page also works from
  ``file://`` with no server and no framework.

Usage:
    python -m benchmark.dashboard_generator [results.json] [summary.json]
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "dashboard")

HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agentic GraphRAG — Benchmark Dashboard</title>
<link rel="stylesheet" href="styles.css">
</head>
<body>
<header>
  <div>
    <h1>Agentic GraphRAG Benchmark</h1>
    <p class="sub">RAG vs GraphRAG vs Agentic GraphRAG — Olympics corpus, 2,951 documents</p>
  </div>
  <div id="meta" class="meta"></div>
</header>
<main>
  <section id="cards" class="cards"></section>

  <section class="panel">
    <h2>Accuracy by question type</h2>
    <p class="hint">Where each pipeline wins — lookup needs RAG, multi-hop needs the
       graph, aggregation &amp; superlative need agents.</p>
    <div id="byType" class="chart"></div>
    <div id="legend" class="legend"></div>
  </section>

  <section class="panel">
    <h2>The cost of being right</h2>
    <p class="hint">Average total tokens per question (x) vs accuracy (y). Bubble size
       = average wall-clock latency.</p>
    <div id="costScatter" class="chart"></div>
  </section>

  <section class="panel">
    <h2>When do agents matter?</h2>
    <p class="hint">Per-question outcome delta: Agentic vs GraphRAG.
       <span class="chip up">+1 agents fixed it</span>
       <span class="chip down">−1 agents broke it</span>
       <span class="chip same">0 same outcome</span></p>
    <div id="delta" class="chart"></div>
  </section>

  <section class="panel">
    <h2>Graph backend</h2>
    <p class="hint">Which store answered the graph calls, and what the TigerGraph
       query push-down actually served. A run that lost the remote path halfway
       through says so here instead of looking identical to a database-backed one.</p>
    <div id="backend"></div>
  </section>

  <section class="panel">
    <h2>Provider contribution</h2>
    <p class="hint">Which pipelines the LLM actually contributed to. A run whose
       provider calls all failed still finishes — every answer then comes from
       the deterministic solvers and the token counters read zero — so the
       contribution is reported here rather than left to be inferred from the
       accuracy table.</p>
    <div id="llmActivity"></div>
  </section>

  <section class="panel">
    <h2>Agentic investigation traces</h2>
    <p class="hint">Pick a question to replay the agent's plan, steps and evidence.</p>
    <div class="trace-controls">
      <select id="traceSelect"></select>
      <span id="traceMeta" class="hint"></span>
    </div>
    <div id="trace" class="trace"></div>
  </section>

  <section class="panel">
    <h2>Per-question results</h2>
    <div class="table-controls">
      <select id="typeFilter">
        <option value="">all types</option>
      </select>
      <select id="outcomeFilter">
        <option value="">all outcomes</option>
        <option value="agent-win">agentic win</option>
        <option value="agent-loss">agentic loss</option>
        <option value="all-wrong">all wrong</option>
        <option value="all-right">all right</option>
      </select>
      <input id="search" type="search" placeholder="filter questions…">
    </div>
    <div class="table-wrap"><table id="resultsTable"></table></div>
  </section>
</main>
<footer>
  <span id="footMeta"></span>
</footer>
<script>
const DATA = __DATA__;
</script>
<script src="dashboard.js"></script>
</body>
</html>
"""


def _load(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def generate(results_path: str = "results/public_results.json",
             summary_path: str = "results/metrics_summary.json",
             out_dir: str = "dashboard") -> str:
    entries = _load(results_path)
    summary = _load(summary_path) if os.path.exists(summary_path) else {}
    payload = {"entries": entries, "summary": summary,
               "generated": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
               "results_file": results_path}

    os.makedirs(out_dir, exist_ok=True)
    html = HTML_HEAD.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    out_path = os.path.join(out_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"dashboard -> {out_path} ({len(entries)} questions, "
          f"{len(html) / 1024:.0f} KiB)")
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate the metrics dashboard")
    ap.add_argument("results", nargs="?", default="results/public_results.json")
    ap.add_argument("--summary", default="results/metrics_summary.json")
    ap.add_argument("--out", default="dashboard")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    generate(args.results, args.summary, args.out)
