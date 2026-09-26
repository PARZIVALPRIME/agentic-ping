"""Compare the slot-parse round-trip at different generation settings.

The semantic parser is the entry point of a run, so a model that cannot produce
*parseable* JSON there silently demotes every question to the template parser -
the run still answers, the metadata still says "rules", and the claim "the model
does the understanding" quietly becomes false. That is the failure this probe
isolates, and the reason the settings are measured rather than assumed.

For one question it asks the model for the slots three ways and reports, per
setting, whether the reply parsed, how long it took and what it cost:

  A  reasoning_effort=none, temperature=0.3, max_tokens=700
  B  reasoning_effort=low,  temperature=0.3, max_tokens=1200
  C  reasoning_effort=none, temperature=0.3, max_tokens=1200

Usage:
    python tools/probe_slot_parse.py            # 1 question, 3 settings
    python tools/probe_slot_parse.py --raw      # + the raw completions
    python tools/probe_slot_parse.py --n 3      # 3 questions
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from config import config                          # noqa: E402
from reasoning.query_parser import SEMANTIC_PARSE_PROMPT  # noqa: E402
from utils.llm import LLMHelper                    # noqa: E402
from utils.metrics import TokenCounter             # noqa: E402

SETTINGS = [
    ("A", "none", 0.3, 700),
    ("B", "low", 0.3, 1200),
    ("C", "none", 0.3, 1200),
]

QUESTIONS_PATH = "questions-20260919T043312Z-1-001/questions/eval_public.jsonl"


def load_questions(n: int):
    out = []
    with open(QUESTIONS_PATH, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true", help="print the raw completions")
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()

    questions = load_questions(args.n)
    print(f"provider={config.llm.provider} model={config.llm.fast_model} "
          f"questions={len(questions)}")

    for label, effort, temp, max_tokens in SETTINGS:
        llm = LLMHelper(provider=config.llm.provider, api_key=config.llm.api_key,
                        chat_model=config.llm.chat_model,
                        fast_model=config.llm.fast_model,
                        eval_model=config.llm.eval_model,
                        temperature=temp,
                        eval_temperature=config.llm.eval_temperature,
                        max_tokens=max_tokens, reasoning_effort=effort,
                        base_url=config.llm.ollama_base_url)
        if not llm.available:
            print(f"\n[{label}] model not available: {llm.error}")
            continue
        print(f"\n[{label}] effort={effort} temp={temp} max_tokens={max_tokens}")
        for q in questions:
            counter = TokenCounter()
            started = time.perf_counter()
            payload = llm.complete_json(
                SEMANTIC_PARSE_PROMPT.format(question=q["question"],
                                             corpus_label=config.domain.corpus_label),
                caller="probe.slot_parse", counter=counter, model=llm.fast_model)
            took = time.perf_counter() - started
            ok = bool(payload) and not payload.get("error")
            shape = (f"qtype={payload.get('qtype')} sport={payload.get('sport')!r} "
                     f"year={payload.get('year')} season={payload.get('season')!r} "
                     f"threshold={payload.get('threshold')} conf={payload.get('confidence')}"
                     if ok else f"UNPARSEABLE payload={payload!r}")
            print(f"   {q['qid']}: {'parsed' if ok else 'FAILED'} in {took:5.1f}s | "
                  f"tokens={counter.total_tokens} | {shape}")
        print(f"   summary: calls={llm.num_calls} failed={llm.failed_calls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
