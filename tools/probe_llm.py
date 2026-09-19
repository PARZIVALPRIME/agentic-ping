"""Probe the configured LLM provider: auth, model availability, JSON mode."""
import sys

sys.stdout.reconfigure(encoding="utf-8")
from openai import OpenAI

KEY = sys.argv[1] if len(sys.argv) > 1 else ""
BASE = "https://api.groq.com/openai/v1"
MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

client = OpenAI(api_key=KEY, base_url=BASE)
print("models available:")
for m in client.models.list().data:
    print("  -", m.id)

for model in MODELS:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "Answer with JSON only."},
                      {"role": "user", "content": 'Return {"answer": "OK"}'}],
            temperature=0.0,
            max_completion_tokens=64,
        )
        print(f"\n{model}: OK -> {(r.choices[0].message.content or '').strip()[:80]!r}")
        print(f"   usage prompt={r.usage.prompt_tokens} completion={r.usage.completion_tokens}")
    except Exception as exc:
        print(f"\n{model}: FAILED -> {exc.__class__.__name__}: {str(exc)[:160]}")