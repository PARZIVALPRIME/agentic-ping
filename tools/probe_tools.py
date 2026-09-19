"""Probe: does openai/gpt-oss-120b on Groq support native function calling?

If native tool-calling works we build a function-calling agent loop; if it does
not, we fall back to text-based ReAct prompting. This probe decides which.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

client = OpenAI(api_key=os.environ["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "count_events",
        "description": "Count Olympic events matching a sport and optional year range.",
        "parameters": {
            "type": "object",
            "properties": {
                "sport": {"type": "string", "description": "e.g. 'Athletics'"},
                "year_from": {"type": "integer"},
                "year_to": {"type": "integer"},
            },
            "required": ["sport"],
        },
    },
}]

for model in ("openai/gpt-oss-120b", "openai/gpt-oss-20b"):
    print(f"=== {model} ===")
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user",
                       "content": "How many Athletics events were held between 2000 and 2012?"}],
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        msg = r.choices[0].message
        print("finish_reason:", r.choices[0].finish_reason)
        print("tool_calls:", json.dumps(
            [{"name": tc.function.name, "args": tc.function.arguments}
             for tc in (msg.tool_calls or [])], indent=2))
        print("content:", (msg.content or "")[:160])
        print("usage:", r.usage.prompt_tokens, "->", r.usage.completion_tokens)
    except Exception as exc:
        print("FAILED:", exc.__class__.__name__, str(exc)[:300])
    print()