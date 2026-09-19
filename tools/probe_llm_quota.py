"""Probe the configured LLM: one real call, token accounting, pacer state.

Answers three questions that matter for the benchmark:
1. are LLM calls succeeding right now (or is the key rate-limited)?
2. does the provider report token usage (input/output) that we can bill?
3. how long does a call take with the client-side pacer enabled?

Usage: python tools/probe_llm_quota.py [n_calls]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from config import config                      # noqa: E402
from utils.llm import build_llm                # noqa: E402
from utils.metrics import TokenCounter         # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3

llm = build_llm(config)
print(f"provider={llm.provider} chat={llm.chat_model} fast={llm.fast_model} "
      f"eval={llm.eval_model}")
print(f"available={llm.available} error={llm.error}")
print("pacer:", llm.stats())

counter = TokenCounter()
for i in range(n):
    t0 = time.time()
    text, in_tok, out_tok = llm.complete(
        "Reply with exactly one word: OK", "You are terse.", "probe",
        counter=counter)
    print(f"call {i+1}: {time.time()-t0:6.2f}s in={in_tok} out={out_tok} "
          f"text={text[:60]!r}")

print("counter:", counter.to_dict()["input_tokens"], counter.to_dict()["output_tokens"])
print("calls recorded:", len(counter.per_operation))
print("pacer after:", llm.stats())