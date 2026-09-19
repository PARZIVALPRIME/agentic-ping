"""Send N real ReAct-shaped tool-calling requests back to back and print the
exact provider error for any failure.

Answers "which limit did we actually hit?" empirically instead of guessing from
the aggregated circuit state.

    python tools/probe_toolcalling_limits.py 6
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from agents.react_agent import SYSTEM_PROMPT  # noqa: E402
from agents.tools import TOOL_SCHEMAS  # noqa: E402
from config import config  # noqa: E402
from utils.llm import build_llm  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
llm = build_llm(config)
print(f"model={config.llm.chat_model} available={llm.available}")

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Question: how many biathlon events at the 2018 "
                                "Winter Olympics had more than 73 competitors?"},
]
for i in range(1, N + 1):
    try:
        turn = llm._service.chat(messages, tools=TOOL_SCHEMAS,
                                 caller=f"probe.tc{i}")
        usage = turn.usage
        print(f"[{i}] ok  in={usage.input_tokens} out={usage.output_tokens} "
              f"calls={[c.name for c in turn.tool_calls]} text={turn.text[:60]!r}")
    except Exception as exc:
        body = getattr(exc, "response", None)
        print(f"[{i}] FAIL {exc.__class__.__name__}")
        if body is not None:
            try:
                print("     body:", body.text[:500])
            except Exception:
                pass
        else:
            print("     ", str(exc)[:500])
        break
print("pacer:", llm._service.pacer_stats() if hasattr(llm._service, "pacer_stats")
      else "n/a")