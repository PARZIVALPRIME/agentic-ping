"""Quota forensics: how many LLM calls does a full run need, and what did we spend?

Estimates the adjudication prompt size per pipeline and compares it with the
provider's token budget, then reports the confidence distribution that decides
whether adjudication is even necessary for a given question.
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from utils.llm import ADJUDICATE_SYSTEM_PROMPT, render_context

entries = json.load(open("results/public_results.json", encoding="utf-8"))

# ── 1. prompt-size estimate for one adjudication call ──────────────────────
q = entries[0]["question"]
ctx = [{"doc_id": f"d{i}", "title": "Some Olympics event article title",
        "text": "x " * 350} for i in range(10)]        # ~700 chars each
for items, chars in ((10, 700), (6, 450), (4, 320)):
    body = render_context(ctx[:items], max_chars_per_item=chars, max_items=items)
    total = (len(body) + len(ADJUDICATE_SYSTEM_PROMPT) + len(q) + 120) / 4.0
    print(f"adjudication prompt: items={items:<3d} chars/item={chars:<4d} "
          f"-> ~{total:6.0f} tokens")

print()
# ── 2. what a full public run costs at the current prompt size ─────────────
N = len(entries)
for tok in (2200, 800):
    for pipes in (1, 3):
        print(f"{pipes} pipeline(s) x {N} questions x ~{tok} tok "
              f"= {pipes * N * tok:>9,} tokens")
print("(Groq free tier for gpt-oss-120b is ~200k tokens/day)")

print()
# ── 3. confidence distribution: where is adjudication actually needed? ─────
for name in ("RAG", "GraphRAG", "Agentic GraphRAG"):
    confs, methods = [], {}
    for e in entries:
        r = (e.get("pipelines") or {}).get(name)
        if not r:
            continue
        confs.append(r.get("confidence") or 0.0)
        methods[r.get("method", "?")] = methods.get(r.get("method", "?"), 0) + 1
    if not confs:
        continue
    hi = sum(1 for c in confs if c >= 0.9)
    lo = sum(1 for c in confs if c < 0.9)
    print(f"{name:<18s} n={len(confs)} median_conf={statistics.median(confs):.2f} "
          f"conf>=0.9: {hi}  conf<0.9: {lo}")
    print(f"{'':18s} methods={methods}")