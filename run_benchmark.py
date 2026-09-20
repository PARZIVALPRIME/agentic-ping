"""CLI for the benchmark harness.

Examples:
    python run_benchmark.py                                   # public set, all pipelines
    python run_benchmark.py --limit 5                         # quick check
    python run_benchmark.py --pipelines agentic               # agentic only
    python run_benchmark.py path/to/hidden.jsonl --out results/hidden_results.json

All arguments are handled by ``benchmark.runner``'s argparse block:
    questions [positional] | --out | --summary | --pipelines | --limit |
    --offset | --resume | --no-llm | --no-llm-judge | --no-tg |
    --agent-mode (alias --mode) | --types | --per-type | --summarize-only |
    --ablation

Common invocations:
    # deterministic baseline (no provider calls at all)
    python run_benchmark.py --no-llm --out results/deterministic_results.json

    # agentic-only, LLM tool-calling loop, quota-bounded sample
    python run_benchmark.py --pipelines agentic --mode react --per-type 2

    # publish the RAG ablation arm alongside the three main pipelines
    python run_benchmark.py --ablation

    # rebuild the summary/dashboard inputs from an existing results file
    python run_benchmark.py --summarize-only --out results/public_results.json
"""
import runpy

if __name__ == "__main__":
    try:
        from benchmark.runner import cli
        cli()
    except ImportError:  # direct-script fallback
        runpy.run_module("benchmark.runner", run_name="__main__")


