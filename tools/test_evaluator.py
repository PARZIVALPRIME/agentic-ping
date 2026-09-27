"""Offline unit check for the evaluation ladder (no network, no pipelines)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from benchmark.evaluator import Evaluator
from benchmark.metrics import MetricsCollector
from benchmark.type_evaluator import GroundingAuditor, TypeAwareEvaluator

ev = Evaluator(None, use_llm_judge=False)

cases = [
    # (golds, pred, expected_correct, expected_match_type)
    (["26"], "26", True, "exact"),
    (["Naim Suleymanoglu"], "Naim Suleymanoglu", True, "exact"),
    (["Athletics at the 2008 Summer Olympics - Men's marathon"], "Men's marathon", True, "fuzzy"),
    (["5"], "5", True, "exact"),
    (["4"], "5", False, "none"),
    ([], "", False, "none"),
    (["Chen Ding"], "  chen  ding  ", True, "exact"),
]

fails = 0
for golds, pred, want_ok, want_type in cases:
    got = ev.evaluate("q?", golds, pred, ["d1"], ["d1", "d2"])
    ok = got.is_correct == want_ok and got.match_type == want_type
    cp, cr = got.citation_precision, got.citation_recall
    if not ok:
        fails += 1
    print(f"{'PASS' if ok else 'FAIL'} golds={golds!r} pred={pred!r} "
          f"-> {got.match_type} correct={got.is_correct} cite_p={cp} cite_r={cr}")

# collector aggregation sanity
c = MetricsCollector()
good = ev.evaluate("q?", ["26"], "26", ["d1"], ["d1", "d2"])
bad = ev.evaluate("q?", ["26"], "27", [], ["d1"])
entry = {
    "qid": "q1", "qtype": "lookup", "question": "q?",
    "gold": {"answers": ["26"], "doc_ids": ["d1"]},
    "pipelines": {
        "Agentic GraphRAG": {"answer": "26", "total_tokens": 100,
                             "latency_ms": 500, "evaluation": good.to_dict()},
        "GraphRAG": {"answer": "27", "total_tokens": 50, "latency_ms": 100,
                     "evaluation": bad.to_dict()},
    },
}
c.add(entry)
s = c.summarize()
ag = s["pipelines"]["Agentic GraphRAG"]
gr = s["pipelines"]["GraphRAG"]
assert ag["accuracy"] == 1.0 and gr["accuracy"] == 0.0, s
assert ag["avg_total_tokens"] == 100 and ag["avg_citation_recall"] == 0.5
assert s["when_agents_matter"][0]["delta"] == 1
print("collector summary OK:", {k: ag[k] for k in ("accuracy", "avg_total_tokens",
                                                   "avg_citation_recall")})
# Type-aware evaluation test suite
print("\n--- Type-Aware Evaluator Unit Tests ---")
# 1. Numeric with tolerance
num_res1 = TypeAwareEvaluator.evaluate_numeric(["26"], "26.0", tolerance=0.1)
assert num_res1.is_correct and num_res1.score == 1.0, num_res1
num_res2 = TypeAwareEvaluator.evaluate_numeric(["26"], "27.0", tolerance=0.0)
assert not num_res2.is_correct, num_res2
print("  PASS Numeric Evaluator (exact + tolerance)")

# 2. Entity with diacritics and token reversal
ent_res1 = TypeAwareEvaluator.evaluate_entity(["Naim Süleymanoğlu"], "Naim Suleymanoglu")
assert ent_res1.is_correct, ent_res1
ent_res2 = TypeAwareEvaluator.evaluate_entity(["Chen Ding"], "Ding Chen")
assert ent_res2.is_correct, ent_res2
print("  PASS Entity Evaluator (diacritics + token permutation)")

# 3. Event title prefix tolerance
event_res = TypeAwareEvaluator.evaluate_event(
    ["Athletics at the 2008 Summer Olympics – Men's marathon"],
    "Men's marathon"
)
assert event_res.is_correct, event_res
print("  PASS Event Evaluator (prefix tolerance)")

# 4. Set Jaccard / F1
set_res = TypeAwareEvaluator.evaluate_set(
    ["Gold, Silver, Bronze"],
    "Bronze, Gold, Silver"
)
assert set_res.is_correct and set_res.f1 == 1.0, set_res
print("  PASS Set Evaluator (set permutation F1)")

# 5. Grounding Auditor
ground_res = GroundingAuditor.audit_grounding(
    pred="Naim Süleymanoğlu",
    cited_docs=[{"id": "d1", "text": "Naim Süleymanoğlu won weightlifting gold"}],
    gold_doc_ids=["d1", "d2"]
)
assert ground_res["is_factually_grounded"] is True
assert ground_res["citation_precision"] == 1.0 and ground_res["citation_recall"] == 0.5
print("  PASS Grounding Auditor (citation precision/recall + token containment)")

print(f"\nTotal Failures: {fails}")
sys.exit(1 if fails else 0)
