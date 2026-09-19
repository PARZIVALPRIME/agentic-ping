"""Diagnose why LLM adjudications come back 'verdict_unusable'.

The adjudication step (``utils.llm.refine_answer``) is the only place the RAG and
GraphRAG pipelines let the model change an answer. When every record reports
``verdict_unusable`` with zero output tokens, the pipeline still looks healthy
while the LLM contributes nothing - so this probe separates the three possible
causes:

1. the provider is not configured / not reachable (no call is made at all),
2. the call returns text that ``complete_json`` cannot parse (markdown fences,
   reasoning prose, truncation at ``max_completion_tokens``), or
3. the call returns valid JSON that simply carries no usable answer.

Usage:
    python tools/probe_llm_json.py            # one adjudication round-trip
    python tools/probe_llm_json.py --raw      # also dump the raw completion
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from config import config                        # noqa: E402
from utils.llm import refine_answer, build_llm   # noqa: E402
from utils.metrics import TokenCounter           # noqa: E402

CONTEXT_ITEMS = [
    {"doc_id": "Q47091419",
     "title": "Biathlon at the 2018 Winter Olympics - Women's sprint",
     "text": "Competitors: 87. Gold: Laura Dahlmeier. Venue: Alpensia Biathlon Centre."},
    {"doc_id": "Q47105341",
     "title": "Biathlon at the 2018 Winter Olympics - Men's sprint",
     "text": "Competitors: 87. Gold: Arnd Peiffer. Venue: Alpensia Biathlon Centre."},
]
QUESTION = ("According to the provided corpus, how many biathlon events at the "
            "2018 Winter Olympics had more than 73 competitors?")
CANDIDATE = "5"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", action="store_true",
                    help="also send a plain JSON request and dump the raw reply")
    args = ap.parse_args()

    llm = build_llm(config)
    print(f"provider      : {llm.provider!r} model={llm.chat_model!r}")
    print(f"available     : {llm.available}")
    if not llm.available:
        print(f"error         : {llm.error}")
        print("\nPROVIDER UNAVAILABLE - no adjudication can happen. Configure "
              "LLM_PROVIDER/CHAT_MODEL plus the API key in .env, or accept a "
              "deterministic run explicitly with --no-llm.")
        return 2

    counter = TokenCounter()
    final, changed, payload = refine_answer(
        llm, QUESTION, CANDIDATE, CONTEXT_ITEMS, counter, caller="probe.adjudicate")

    print(f"candidate     : {CANDIDATE!r}")
    print(f"final         : {final!r} (changed={changed})")
    print(f"verdict reason: {payload.get('reason')!r}")
    if payload.get("raw_answer"):
        print(f"raw answer    : {payload['raw_answer']!r}")
    print(f"provider calls: {llm.num_calls} ok / {llm.failed_calls} failed")
    print(f"tokens        : {json.dumps(counter.to_dict().get('totals')
                                     or counter.to_dict(), ensure_ascii=False)}")
    for op in counter.to_dict().get("per_operation", [])[-3:]:
        print(f"  op {op.get('operation')}: in={op.get('input_tokens')} "
              f"out={op.get('output_tokens')}")

    if args.raw:
        text, in_tok, out_tok = llm.complete(
            "Reply with JSON only: {\"ok\": true}", caller="probe.raw")
        print("\nraw completion test")
        print(f"  text   : {text[:600]!r}")
        print(f"  tokens : in={in_tok} out={out_tok}")

    # A usable verdict is about the *payload*, not about whether the model
    # happened to agree with the candidate: a corrected answer is a success.
    unusable = {"verdict_unusable", "llm_unavailable", "no_evidence"}
    if final and payload.get("reason") not in unusable:
        print("\nOK: adjudication returned a usable answer path.")
        return 0
    print("\nATTENTION: the adjudication did not yield a usable answer - see the "
          "reason above ('json_parse_failed' means the reply could not be "
          "parsed, 'verdict_unusable' means the JSON carried no answer).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())