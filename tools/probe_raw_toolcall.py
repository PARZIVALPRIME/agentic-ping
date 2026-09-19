"""One raw tool-calling request straight to the SDK, no pacing/retry/circuit.

Prints the provider's verbatim error body, which is the only way to see which
limit (TPM vs RPD vs TPD vs request size) is actually being enforced.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from agents.react_agent import SYSTEM_PROMPT  # noqa: E402
from agents.tools import TOOL_SCHEMAS  # noqa: E402
from config import config  # noqa: E402
from utils.metrics import count_tokens  # noqa: E402
from utils.llm import build_llm  # noqa: E402

llm = build_llm(config)
client = llm._service._client
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Question: how many biathlon events at the 2018 "
                                "Winter Olympics had more than 73 competitors?"},
]
payload = {"model": config.llm.chat_model, "messages": messages,
           "tools": TOOL_SCHEMAS, "tool_choice": "auto", "temperature": 0.0,
           "max_completion_tokens": 900, "reasoning_effort": "low"}
print("prompt+schemas approx tokens:",
      count_tokens(SYSTEM_PROMPT) + count_tokens(json.dumps(TOOL_SCHEMAS)) + 40)

try:
    resp = client.chat.completions.with_raw_response.create(**payload)
    print("status:", resp.status_code)
    for name in ("retry-after", "x-ratelimit-remaining-requests",
                 "x-ratelimit-remaining-tokens", "x-ratelimit-limit-tokens",
                 "x-ratelimit-reset-tokens"):
        print(f"  {name:32s} {resp.headers.get(name)}")
    msg = resp.parse().choices[0].message
    print("tool_calls:", [(c.function.name, c.function.arguments[:120])
                          for c in (msg.tool_calls or [])])
    print("text:", (msg.content or "")[:200])
except Exception as exc:
    print("FAIL", exc.__class__.__name__)
    body = getattr(exc, "response", None)
    print("  body:", body.text[:700] if body is not None else str(exc)[:700])