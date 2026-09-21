"""Migration preflight: verify this machine can run the benchmark.

Run this FIRST on any new machine::

    python preflight.py

Exit code 0 means a deterministic run will work. Non-zero means something is
missing and the message says exactly what. Nothing here touches the network,
so it is safe to run before any API key is configured.
"""

from __future__ import annotations

import importlib
import os
import platform
import sys

REQUIRED = ["numpy"]
OPTIONAL = {
    "dotenv": "reads .env automatically (else export env vars by hand)",
    "openai": "needed only for --llm runs (any OpenAI-compatible provider)",
    "tenacity": "retry/backoff around provider calls",
    "sentence_transformers": "dense embeddings; TF-IDF fallback works without it",
}

OK, WARN, BAD = "  ok   ", "  warn ", "  FAIL "


def _check_python() -> bool:
    print(f"{OK}python {sys.version.split()[0]} on {platform.system()} "
          f"({platform.machine()})")
    if sys.version_info < (3, 9):
        print(f"{BAD}Python >= 3.9 required")
        return False
    return True


def _check_deps() -> bool:
    ok = True
    for mod in REQUIRED:
        try:
            importlib.import_module(mod)
            print(f"{OK}{mod}")
        except ImportError:
            print(f"{BAD}{mod} missing -> pip install -r requirements.txt")
            ok = False
    for mod, why in OPTIONAL.items():
        try:
            importlib.import_module(mod)
            print(f"{OK}{mod} (optional)")
        except ImportError:
            print(f"{WARN}{mod} absent - {why}")
    return ok


def _check_data(cfg) -> bool:
    ok = True
    paths = [("corpus", cfg.benchmark.corpus_path),
             ("public questions", cfg.benchmark.public_questions_path),
             ("hidden questions", cfg.benchmark.hidden_questions_path)]

    for label, path in paths:
        if not path:
            continue
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"{OK}{label}: {path} ({size:,} bytes)")
            if size == 0:
                print(f"{BAD}{label} is empty")
                ok = False
        else:
            missing_is_fatal = label != "hidden questions"
            print(f"{BAD if missing_is_fatal else WARN}{label} not found: {path}")
            ok = ok and not missing_is_fatal
    return ok


def main() -> int:
    print("=" * 62)
    print(" preflight: agentic-ping")
    print("=" * 62)

    ok = _check_python()
    ok = _check_deps() and ok

    try:
        from config import config as cfg
    except Exception as exc:  # config must never explode on a fresh machine
        print(f"{BAD}could not load config: {exc.__class__.__name__}: {exc}")
        return 1

    print(f"{OK}config loaded")
    try:
        described = cfg.describe()
        for key, value in described.items():
            print(f"       {key}: {value}")
    except Exception:
        pass

    ok = _check_data(cfg) and ok

    provider = (cfg.llm.provider or "none").lower()
    if provider in ("", "none"):
        print(f"{WARN}LLM_PROVIDER=none -> only --no-llm runs will work")
    elif not cfg.llm.api_key and provider != "ollama":
        print(f"{WARN}provider '{provider}' has no API key in the environment")
    else:
        print(f"{OK}provider '{provider}' configured")

    if cfg.tg.enabled:
        print(f"{WARN}TG_ENABLED is set - runs will try {cfg.tg.host} "
              f"(local graph is used as fallback, so this cannot fail a run)")
    else:
        print(f"{OK}TigerGraph disabled - local graph only")

    print("-" * 62)
    print(" PASS - try: python run_benchmark.py --no-llm --no-tg --limit 5"
          if ok else " FAIL - fix the items marked FAIL above")
    print("-" * 62)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
