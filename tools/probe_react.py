"""Probe the ReAct tool-calling agent on a few questions, verbosely.

Runs the real knowledge graph + index and prints, per question, the exact tool
sequence the model chose, each tool's summary, the submitted answer and the token
cost. This is the acceptance test for the agentic refactor: if `llm calls` is 0
or `tool:` steps are missing here, the loop is not actually running.

    python tools/probe_react.py                 # 3 questions, hybrid mode
    python tools/probe_react.py 5 react         # 5 questions, react mode only
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from config import config  # noqa: E402
from kg.builder import load_or_build  # noqa: E402
from pipelines import build_pipelines  # noqa: E402
from retrieval import load_index  # noqa: E402
from utils.llm import build_llm  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MODE = sys.argv[2] if len(sys.argv) > 2 else ""
if MODE:
    config.agent.mode = MODE

print(f"mode={config.agent.mode} model={config.llm.chat_model} "
      f"react_max_steps={config.agent.react_max_steps}")

kg = load_or_build(config.benchmark.corpus_path)
index = load_index(config.benchmark.corpus_path, kg,
                   vector_backend=config.benchmark.vector_backend)
llm = build_llm(config)
print(f"llm available: {llm.available}")

pipes = build_pipelines(index, llm, config)
agent = next(p for p in pipes if "agentic" in p.name.lower())

with open(config.benchmark.public_questions_path, "r", encoding="utf-8") as fh:
    questions = [json.loads(line) for line in fh if line.strip()][:N]

totals = {"calls": 0, "in": 0, "out": 0}
for item in questions:
    qid, question = item.get("qid", ""), item["question"]
    gold = item.get("answer") or item.get("gold") or ""
    print("\n" + "=" * 78)
    print(f"{qid} [{item.get('qtype', '')}] {question}")
    print(f"gold: {gold!r}")

    result = agent.run(question, qid)
    state = agent.engine.react

    print(f"method={result.method} stop={result.stop_reason} "
          f"conf={result.confidence}")
    print(f"answer: {result.answer!r}")
    for call in result.metadata.get("tool_calls", []):
        flag = "ERR " if call["error"] else "    "
        print(f"  {flag}{call['tool']}({json.dumps(call['args'], default=str)})"
              f" -> {call['summary']}  [{call['latency_ms']}ms]")
    print(f"llm_calls={result.llm_calls} tools_called={result.tools_called}")
    print(f"tokens: in={result.input_tokens} out={result.output_tokens} "
          f"total={result.total_tokens}")
    totals["calls"] += result.llm_calls
    totals["in"] += result.input_tokens
    totals["out"] += result.output_tokens

print("\n" + "=" * 78)
print(f"TOTAL llm_calls={totals['calls']} in={totals['in']} out={totals['out']}")
print("circuit:", json.dumps(llm.stats().get("circuit", {}), default=str))