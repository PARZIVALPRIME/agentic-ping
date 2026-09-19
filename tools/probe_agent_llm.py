"""Probe: is the Agentic pipeline's orchestrator receiving a working LLM?

Run: python tools/probe_agent_llm.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from config import config
from kg.builder import load_or_build
from pipelines import build_pipelines
from retrieval import load_index
from utils.llm import build_llm

kg = load_or_build(config.benchmark.corpus_path)
index = load_index(config.benchmark.corpus_path, kg,
                   vector_backend=config.benchmark.vector_backend)
llm = build_llm(config)
print("runner-level llm:", type(llm).__name__,
      "available=", getattr(llm, "available", None),
      "chat_model=", getattr(llm, "chat_model", None),
      "fast_model=", getattr(llm, "fast_model", None),
      "error=", getattr(llm, "error", ""))

pipes = {p.name: p for p in build_pipelines(index, llm, config)}
ag = pipes["Agentic GraphRAG"]
eng = ag.engine
print("orchestrator llm: ", type(eng.llm).__name__ if eng.llm else None,
      "available=", getattr(eng.llm, "available", None) if eng.llm else None)
print("same object as runner llm:", eng.llm is llm)
print("classifier llm available:", getattr(eng.classifier.llm, "available", None))
print("synth llm available:", getattr(eng.synth.llm, "available", None))

q = "How many participating nations were there at the 2008 Summer Olympics?"
state = eng.run(q, "probe-001")
print("\n--- agentic run ---")
print("qtype:", state.qtype, "| answer:", repr(state.answer))
print("stop_reason:", state.stop_reason, "| synth_source:", state.synth_source)
print("classification:", state.classification)
print("slot_report.used:", state.slot_report.get("used"))
print("tokens:", state.tokens.to_dict()["total_tokens"],
      "| input:", state.tokens.input_tokens, "output:", state.tokens.output_tokens)
print("per_operation:")
for op in state.tokens.per_operation:
    print("   ", op["operation"], op["total_tokens"], op.get("detail"))
print("llm num_calls:", getattr(eng.llm, "num_calls", None))
print("adjudication:", state.adjudication)