"""CLI for the benchmark harness.

Examples:
    python run_benchmark.py                                   # public set, all pipelines
    python run_benchmark.py --limit 5                         # quick check
    python run_benchmark.py --pipelines agentic               # agentic only
    python run_benchmark.py path/to/hidden.jsonl --out results/hidden_results.json

All arguments are handled by ``benchmark.runner``'s argparse block:
    questions [positional] | --out | --summary | --pipelines | --limit |
    --offset | --no-llm-judge
"""
import runpy

if __name__ == "__main__":
    try:
        from benchmark.runner import cli
        cli()
    except ImportError:  # direct-script fallback
        runpy.run_module("benchmark.runner", run_name="__main__")


