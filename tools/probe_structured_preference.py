"""Regression test for the agentic verifier's structured-preference gate (no LLM).

Replays the answers a live run actually submitted on the questions it failed,
straight into ``OrchestratorAgent._verify_answer``, and reports what the verifier
made of each one. This isolates the arbitration policy from the tool loop: the
submitted answers are frozen fixtures, so the test is deterministic, instant and
runs without a provider.

Usage:
    python tools/probe_structured_preference.py            # historical failures
    python tools/probe_structured_preference.py pub-060    # named questions
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from config import config                                    # noqa: E402
from kg.backend import open_graph                             # noqa: E402
from kg.textutil import normalize                             # noqa: E402
from reasoning.query_parser import parse_question             # noqa: E402
from retrieval import load_index                              # noqa: E402
from agents.orchestrator import OrchestratorAgent             # noqa: E402
from agents.state import AgentState                           # noqa: E402

# qid -> the answer the live run submitted when it was wrong. Taken from
# results/llm_results.json (the 100-question qwen3.5:4b run) plus batch 01.
HISTORICAL_FAILURES = {
    "pub-007": "- bronze: Raphael Holzdeppe",
    "pub-037": "56",
    "pub-038": "The provided context does not contain information about this event.",
    "pub-045": "- 34 (no",
    "pub-060": "Kaori Matsumoto",
    "pub-086": "The context provided lists speed skating events only.",
    "pub-087": "Total: 38 - 20 = 1",
}


def correct(gold_list, pred):
    p = normalize(pred)
    if not p:
        return False
    return any(p == normalize(g) or normalize(g) in p or p in normalize(g)
               for g in gold_list if normalize(g))


def main() -> int:
    wanted = list(sys.argv[1:]) or sorted(HISTORICAL_FAILURES)
    kg = open_graph(config.benchmark.corpus_path, config,
                    cache_path=config.benchmark.kg_cache_path, force_local=True)
    index = load_index(config.benchmark.corpus_path, kg,
                       vector_backend=config.benchmark.vector_backend)
    agent = OrchestratorAgent(kg, index, llm=None, cfg=None)

    questions = {}
    with open(config.benchmark.public_questions_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                q = json.loads(line)
                questions[q["qid"]] = q

    fixed = total = 0
    print("structured_preference =", agent.cfg.get("structured_preference"))
    for qid in wanted:
        q = questions.get(qid)
        if q is None:
            print(f"{qid}: not in the public set")
            continue
        submitted = HISTORICAL_FAILURES.get(qid, "")
        spec = parse_question(q["question"], kg)
        state = AgentState(q["question"], spec, qid)
        state.kind = agent._kind(spec.qtype)
        state.answer = submitted
        state.confidence = 0.6
        agent._verify_answer(q["question"], state)
        total += 1
        ok = correct(q["answer"], state.answer)
        fixed += int(ok)
        changes = [c for c in (state.strategy_changes or [])
                   if c.startswith("verification[")]
        print(f"\n[{qid}:{q['qtype']}] {'OK ' if ok else 'MISS'} {q['question']}")
        print(f"   GOLD      : {q['answer']}")
        print(f"   SUBMITTED : {submitted!r}")
        print(f"   FINAL     : {state.answer!r}  (source={state.synth_source})")
        for change in changes:
            print(f"   VERIFIED  : {change}")
    print(f"\n{fixed}/{total} historical failures now answered correctly by the "
          f"verifier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
