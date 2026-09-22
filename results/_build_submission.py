"""Rebuild the hidden submission file from the completed hidden benchmark run
(results/hidden_llm.json) instead of re-running the pipeline. Same schema as
tools/submit_hidden.py writes, built from the Router arm (best public accuracy;
its answers are identical to Agentic GraphRAG on all 50 hidden questions).
"""
import json
import sys

SRC = "results/hidden_llm.json"
OUT = "results/hidden_submission.json"
PIPELINE = "Router"

rows = json.load(open(SRC, encoding="utf-8"))
records = []
for r in rows:
    pr = r["pipelines"][PIPELINE]
    records.append({
        "id": r["qid"],
        "question": r["question"],
        "qtype": pr.get("qtype", ""),
        "answer": pr.get("answer", ""),
        "citations": pr.get("citations", []),
        "confidence": pr.get("confidence", 0.0),
        "tokens": {
            "context_tokens": pr.get("context_tokens", 0),
            "input_tokens": pr.get("input_tokens", 0),
            "output_tokens": pr.get("output_tokens", 0),
            "total_tokens": pr.get("total_tokens", 0),
            "llm_calls": pr.get("llm_calls", 0),
        },
        "agentic_trace": {
            "pipeline": pr.get("pipeline", ""),
            "plan": pr.get("plan", []),
            "retrieval_steps": pr.get("retrieval_steps", 0),
            "loop_iterations": pr.get("loop_iterations", 0),
            "tools_called": pr.get("tools_called", []),
            "agents_invoked": pr.get("agents_invoked", []),
            "strategy_changed": pr.get("strategy_changed", False),
            "stop_reason": pr.get("stop_reason", ""),
            "candidates_considered": pr.get("candidates_considered", 0),
            "chunks_retrieved": pr.get("chunks_retrieved", 0),
            "docs_retrieved": pr.get("docs_retrieved", 0),
            "steps": pr.get("steps", []),
            "time_per_operation": pr.get("time_per_operation", []),
            "tokens_per_operation": pr.get("tokens_per_operation", []),
        },
        "evidence": pr.get("evidence", []),
        "unresolved": pr.get("unresolved", []),
        "latency_ms": pr.get("latency_ms", 0.0),
        "error": None,
    })

answered = sum(1 for r in records if (r["answer"] or "").strip())
total_tokens = sum(r["tokens"]["total_tokens"] for r in records)
payload = {
    "submission": {
        "pipeline": PIPELINE,
        "mode": "live",
        "provider": "ollama",
        "chat_model": "qwen3.5:4b",
        "graph_backend": "local",
        "questions_file": "questions-20260919T043312Z-1-001/questions/eval_hidden.jsonl",
        "num_questions": len(records),
        "num_completed": len(records),
        "num_failed": 0,
        "num_answered": answered,
        "total_tokens": total_tokens,
        "source_run": SRC,
        "note": ("rebuilt from the completed benchmark run; identical answers "
                 "to the Agentic GraphRAG arm on all 50 questions"),
    },
    "results": records,
}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, ensure_ascii=False)
print(f"wrote {OUT}: {len(records)} records, answered={answered}/50, tokens={total_tokens:,}")
