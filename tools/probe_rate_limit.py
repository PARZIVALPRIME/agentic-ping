"""Probe the exact rate-limit error message from the provider.

Distinguishes per-minute throttling (fixable with pacing) from daily-quota
exhaustion (not fixable - the run must fail fast and report it).

Run: python tools/probe_rate_limit.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GROQ_API_KEY", "")
print("key:", (key[:12] + "..." + key[-4:]) if key else "MISSING")

try:
    from openai import OpenAI
    import openai
except ImportError:
    print("openai SDK missing")
    raise SystemExit(1)


def attempt(model, label):
    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            temperature=0.0,
        )
        print(f"[{label}] OK in {time.time()-t0:.2f}s -> {r.choices[0].message.content!r} "
              f"usage={r.usage.prompt_tokens}/{r.usage.completion_tokens}")
        return True
    except Exception as exc:
        print(f"[{label}] FAIL in {time.time()-t0:.2f}s -> {type(exc).__name__}")
        print(f"        message: {str(exc)[:400]}")
        resp = getattr(exc, "response", None)
        if resp is not None:
            print(f"        status={getattr(resp,'status_code',None)} "
                  f"headers={dict(list(getattr(resp,'headers',{}).items())[:8])}")
        return False


for model in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
    attempt(model, model)

# list models to confirm what the key may use
try:
    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    models = [m.id for m in client.models.list().data]
    print("\navailable models:", models)
except Exception as exc:
    print("\nmodel list failed:", str(exc)[:200])