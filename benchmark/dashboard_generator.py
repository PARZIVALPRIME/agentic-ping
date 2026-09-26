"""Generates the interactive HTML metrics dashboard from benchmark results.

Inputs (produced by ``benchmark.runner``):
- ``results/public_results.json``   per-question records (all pipelines)
- ``results/metrics_summary.json``  aggregate tables

The corpus (``config.benchmark.corpus_path``) is read for its document count, so
the header/footer state the size of the corpus the run used instead of a number
typed into this file.

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

from benchmark.metrics import llm_activity, llm_activity_warnings

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
    <p class="sub">RAG vs GraphRAG vs Agentic GraphRAG — __CORPUS_LINE__</p>
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


def _corpus_facts() -> Dict[str, Any]:
    """Measure the corpus for the header/footer line the page has to state.

    The header and the footer both used to hard-code "2,951 documents" - a fact
    about one corpus snapshot, not about the run on the page, and one that has
    already outlived its usefulness once (the footer additionally claimed a
    hosted model and provider that no run in this repository used). Counting is
    a single pass over a 23 MB jsonl (~0.2 s), which is far cheaper than
    publishing a stale number. When the corpus cannot be read the page says the
    size is unrecorded instead of guessing.
    """
    try:
        from config import config
        path = config.benchmark.corpus_path
    except Exception:
        path = ""
    if not path:
        return {}
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    if not os.path.exists(path):
        return {"path": path, "num_docs": 0}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            docs = sum(1 for line in fh if line.strip())
    except OSError:
        return {}
    return {"path": path, "num_docs": docs}


def generate(results_path: str = "results/public_results.json",
             summary_path: str = "results/metrics_summary.json",
             out_dir: str = "dashboard",
             require_llm: bool = False,
             name: str = "index.html") -> str:
    entries = _load(results_path)
    summary = _load(summary_path) if os.path.exists(summary_path) else {}

    # Provider contribution is derived from the per-question records so the panel
    # works even when the summary predates the provenance block (an older run, or
    # one rebuilt from a results file). A summary that carries its own
    # ``provenance.llm_activity`` still wins in the dashboard.
    activity = llm_activity(entries)
    warnings = llm_activity_warnings(activity)
    # Measured here rather than read from the results file: the corpus is an
    # input to the run, and the page states its size in the header and footer.
    corpus = _corpus_facts()

    if require_llm and not any(slot.get("llm_contributed") for slot in activity.values()):
        idle = ", ".join(sorted(activity)) or "no pipeline"
        raise SystemExit(
            "--require-llm: no pipeline recorded provider output in "
            f"{results_path} ({idle}) - this is a deterministic run, so its "
            "accuracy table cannot be presented as an LLM result. Re-run "
            "without --no-llm, or drop the flag to publish it as the baseline.")

    payload = {"entries": entries, "summary": summary,
               "llm_activity": activity, "llm_warnings": warnings,
               "corpus": corpus,
               "generated": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
               "results_file": results_path}

    os.makedirs(out_dir, exist_ok=True)
    docs = int(corpus.get("num_docs") or 0)
    corpus_line = (f"Olympics corpus, {docs:,} documents" if docs
                   else "corpus document count not recorded in this page")
    html = (HTML_HEAD.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
                     .replace("__CORPUS_LINE__", corpus_line))
    out_path = os.path.join(out_dir, name)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"dashboard -> {out_path} ({len(entries)} questions, "
          f"{len(html) / 1024:.0f} KiB, corpus "
          f"{f'{docs:,} documents' if docs else 'size unmeasured'})")
    for name, slot in activity.items():
        print(f"  {name:<18s} provider calls in {slot['records_with_calls']}"
              f"/{slot['records']} records, model output in "
              f"{slot['records_answering']} - {slot['total_tokens']} tokens")
    for warning in warnings:
        print(f"  WARNING: {warning}")
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate the metrics dashboard")
    ap.add_argument("results", nargs="?", default="results/public_results.json")
    ap.add_argument("--summary", default="results/metrics_summary.json")
    ap.add_argument("--out", default="dashboard")
    ap.add_argument("--name", default="index.html",
                    help="output file name inside --out; the page links "
                         "styles.css/dashboard.js relatively, so any name in "
                         "that directory works (e.g. 'baseline.html' to publish "
                         "the deterministic baseline beside the live run)")
    ap.add_argument("--require-llm", action="store_true",
                    help="refuse to build the page unless at least one pipeline "
                         "recorded provider output, so a deterministic run cannot "
                         "be published as an LLM result")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    generate(args.results, args.summary, args.out,
             require_llm=args.require_llm, name=args.name)
