"""Check remaining quota and native tool-calling support per Groq model.

`authored` is what matters for a long run: gpt-oss models share one 200K
tokens/day budget, so a benchmark that spends the budget on the chat model
leaves nothing for the agent. Pass several model ids to compare.

    python tools/probe_models.py
    python tools/probe_models.py openai/gpt-oss-120b openai/gpt-oss-20b
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import httpx  # noqa: E402

from agents.tools import TOOL_SCHEMAS  # noqa: E402

KEY = os.getenv("GROQ_API_KEY", "")
URL = "https://api.groq.com/openai/v1/chat/completions"
MODELS = sys.argv[1:] or ["openai/gpt-oss-120b", "openai/gpt-oss-20b",
                          "qwen/qwen3.8-27b", "groq/compound-mini"]
TOOLS = TOOL_SCHEMAS[:2]        # search_events, search_passages - enough to test

for model in MODELS:
    payload = {
        "model": model,
        "messages": [
            {"role": "user",
             "content": "Call search_events for biathlon events at the 2018 "
                        "Winter Olympics."},
        ],
        "tools": TOOLS, "tool_choice": "auto",
        "max_completion_tokens": 300, "temperature": 0.0,
    }
    try:
        resp = httpx.post(URL, headers={"Authorization": f"Bearer {KEY}"},
                          json=payload, timeout=120)
    except Exception as exc:
        print(f"{model:26s} transport error: {exc}")
        continue
    left_r = resp.headers.get("x-ratelimit-remaining-requests")
    left_t = resp.headers.get("x-ratelimit-remaining-tokens")
    head = f"{model:26s} http={resp.status_code} rpm_left={left_r} tpm_left={left_t}"
    if resp.status_code != 200:
        try:
            err = resp.json()["error"]["message"]
        except Exception:
            err = resp.text[:200]
        print(f"{head}\n    -> {err[:220]}")
        continue
    msg = resp.json()["choices"][0]["message"]
    calls = [(c["function"]["name"], c["function"]["arguments"][:90])
             for c in (msg.get("tool_calls") or [])]
    print(f"{head}\n    tool_calls={calls} text={str(msg.get('content'))[:80]!r}")