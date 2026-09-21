# Migration guide — running this repo on another machine

Everything below is deterministic and offline. No API key, no TigerGraph, no
network. If these commands pass, the machine can produce a full submission.

## 1. Set up

```powershell
git clone <your-repo-url> agentic-ping
cd agentic-ping

python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
```

The dataset folders (`corpus-*/`, `questions-*/`) must sit next to the code. If
you keep them elsewhere, point at them instead of moving them:

```powershell
$env:CORPUS_PATH           = "D:\data\corpus.jsonl"
$env:PUBLIC_QUESTIONS_PATH = "D:\data\eval_public.jsonl"
$env:HIDDEN_QUESTIONS_PATH = "D:\data\eval_hidden.jsonl"
```

## 2. Verify the machine (always do this first)

```powershell
python preflight.py      # deps, config, dataset paths
python tools\selftest.py # imports + preflight + conflicts + benchmark + hidden set
```

`selftest.py` exits non-zero if anything is broken, so it can gate a
submission. Expected tail:

```
  PASS  imports
  PASS  preflight
  PASS  conflicts (Round 2)
  PASS  benchmark smoke
  PASS  hidden submission
  ALL STAGES PASS - this machine can produce a submission
```

## 3. Produce the submission

```powershell
# Public set: RAG vs GraphRAG vs Agentic vs Router  (~60s)
python run_benchmark.py --no-llm --no-tg `
  --out results\public_results.json --summary results\metrics_summary.json

# Ablation study: which mechanism earns the gain  (~3 min)
python run_benchmark.py --no-llm --no-tg --ablations `
  --out results\ablation_study.json --summary results\ablation_summary.json

# Hidden set: raw answers + tokens + agentic traces  (~5s)
python tools\submit_hidden.py --no-llm --no-tg

# Self-contained dashboard
python -m benchmark.dashboard_generator
```

Deliverables produced:

| File | Contents |
|---|---|
| `results/public_results.json` | per-question, per-pipeline results + traces |
| `results/metrics_summary.json` | accuracy / token / latency aggregates |
| `results/ablation_study.json` | ablation variants |
| `results/hidden_submission.json` | 50 hidden answers, tokens, agentic trace |
| `dashboard/index.html` | self-contained dashboard (open directly) |

## 4. Optional: enable an LLM

Deterministic mode is the reproducible baseline and needs no provider. To run
with a model, create `.env` (see `.env.example`):

```ini
LLM_PROVIDER=ollama
CHAT_MODEL=qwen3.5:4b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

Then drop `--no-llm`. Use `--agent-mode plan` with small (≤7B) models: free-form
tool-calling is unreliable below that size, while the planner/executor path
only asks the model to classify and phrase.

A run that requests the LLM and cannot reach it **fails loudly** rather than
silently degrading into a deterministic run — that failure mode is how a
deterministic run gets mistaken for a broken LLM one.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `PermissionError ... .tmp -> .json` | OneDrive/AV lock. Already handled with retry + fallback; if persistent, move the repo outside the synced folder. |
| `no LLM provider available` | Expected without `.env`. Add `--no-llm`. |
| Dashboard shows "0 LLM calls" | You are viewing a deterministic run. Expected. |
| KG rebuild is slow on first run | ~30s, cached to `results/knowledge_graph.json`. Delete it to force a rebuild. |
| Unicode errors in the console | Cosmetic (PowerShell code page). Files are written UTF-8. |
