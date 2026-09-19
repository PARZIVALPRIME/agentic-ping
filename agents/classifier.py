"""Question-type classifier agent.

A thin wrapper around the deterministic rule classifier with an optional LLM
refinement pass. The rules are authoritative by default because they are exact
for this corpus's question templates; the LLM is only consulted when the rules
land on the generic ``lookup`` bucket *and* the question looks structurally
unusual, which is where a hidden-set paraphrase would differ.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from reasoning.query_parser import QuerySpec, classify
from utils.metrics import TokenCounter

CLASSES = ("lookup", "multi_hop", "temporal", "aggregation", "superlative")

CLASSIFIER_PROMPT = """You are classifying an Olympics-corpus question.

Categories:
- lookup       : a single fact about one named article
- multi_hop    : link a venue and/or date to an event (and its winner)
- temporal     : "immediately before/after <Games>", needs two editions linked
- aggregation  : count events satisfying a threshold across many documents
- superlative  : find the max/min event across many documents

Question: {question}

Reply with JSON only: {{"qtype": "<category>", "confidence": <0-1>, "reason": "<short>"}}
"""


class QuestionClassifier:
    """Classifies questions; exposes *why* a route was chosen."""

    def __init__(self, kg=None, llm=None) -> None:
        self.kg = kg
        self.llm = llm

    def classify(self, question: str, spec: Optional[QuerySpec] = None,
                 counter: Optional[TokenCounter] = None) -> Dict[str, Any]:
        """Route the question to a strategy.

        LLM-first: the model is asked to classify *every* question (one cheap
        call on the fast model) and its verdict drives the plan. The rule-based
        classifier always runs too, so we can report agreement and fall back
        when the model is unavailable, rate-limited or returns junk.
        """
        rule_type = classify(question, self.kg)
        qtype = rule_type
        reason = f"rule-based match for the '{rule_type}' template"
        method = "rules"
        confidence = 0.95 if rule_type != "lookup" else 0.7
        agreement = None

        if self.llm is not None and getattr(self.llm, "available", False):
            payload = self.llm.complete_json(
                CLASSIFIER_PROMPT.format(question=question),
                caller="classifier.classify", counter=counter,
                model=self.llm.fast_model)
            candidate = str(payload.get("qtype", "")).strip().lower()
            if candidate in CLASSES:
                agreement = (candidate == rule_type)
                qtype = candidate
                method = "llm" if agreement else "llm_override"
                confidence = float(payload.get("confidence", 0.7) or 0.7)
                reason = str(payload.get("reason", "llm classification"))[:200]
        return {"qtype": qtype, "method": method, "confidence": round(confidence, 3),
                "reason": reason, "rule_type": rule_type,
                "llm_agrees_with_rules": agreement}


def classify_question(question: str, kg=None) -> str:
    """Backwards-compatible functional entry point."""
    return classify(question, kg)