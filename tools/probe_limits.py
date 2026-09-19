"""Show Groq rate-limit headers for one minimal call.

Distinguishes the three ways a 429 can happen: per-minute tokens (TPM),
per-minute requests (RPM) and per-day budgets (RPD / TPD). The daily caps are
what make a long benchmark run stop mid-way, and the headers are the only
authoritative source for which one was hit.

    python tools/probe_limits.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import httpx  # noqa: E402

from config import config  # noqa: E402

KEY = os.getenv("GROQ_API_KEY", "")
MODEL = config.llm.chat_model
URL = "https://api.groq.com/openai/v1/chat/completions"

print(f"model: {MODEL}  key: ...{KEY[-6:]}")
resp = httpx.post(
    URL,
    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
          "max_completion_tokens": 8},
    timeout=60,
)
print("status:", resp.status_code)
for name in ("retry-after", "x-ratelimit-limit-requests",
             "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests",
             "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens",
             "x-ratelimit-reset-tokens"):
    print(f"  {name:34s} {resp.headers.get(name)}")
if resp.status_code != 200:
    print("body:", resp.text[:400])
else:
    print("usage:", resp.json().get("usage"))