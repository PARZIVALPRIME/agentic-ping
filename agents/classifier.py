"""Question-type classifier agent.

Two classifiers vote, in a deliberate order, and their disagreement is a
first-class output rather than a hidden preference.

WHY THE MODEL GOES FIRST
------------------------
The template classifier in :mod:`reasoning.query_parser` recognises the exact
wording of the evaluation set. That makes it precise on that wording and blind
to everything else: reworded, paraphrased or unseen questions fall through the
templates into the generic ``lookup`` bucket. A classifier that only understands
the sentences it was written against is a lookup table for those sentences, so
it cannot be the entry point.

The LLM is therefore asked first, and its verdict is used whenever it is
well-formed and confident. The templates keep two jobs and no others:

1. VALIDATION - a *positive* template match is hard evidence that the question
   structurally needs a particular capability. "How many ... had more than N
   competitors" cannot be answered by any top-k window: the count is only
   correct over the complete candidate set. When the model's verdict conflicts
   with such a structural requirement (a count, an extreme, an edition chain)
   the requirement wins, and the conflict is recorded in ``rule_disagreement``
   so the routing audit can show how often the model was overruled and why. A
   loose cue such as "held at" is *not* treated as structural evidence: a
   reworded question that mentions a venue is often not a multi-hop question,
   and the model is the better judge there.
2. FALLBACK - when the model is unavailable, returns junk, or is unsure, the
   template verdict is used.

``ABLATE_CLASSIFIER=1`` removes both jobs: the templates are never consulted, so
the run measures the system under a phrasing it was never tuned on (row D of the
generalisation suite).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from config import config as _app_config
from reasoning.query_parser import QuerySpec, classify
from utils.metrics import TokenCounter

CLASSES = ("lookup", "multi_hop", "temporal", "aggregation", "superlative")

#: Types whose answer is only correct over the *complete* candidate set (a count,
#: an extreme) or over a linked chain of editions (before/after). A positive rule
#: match for one of these is treated as structural evidence and outranks the
#: model's verdict; everything else is the model's call.
STRUCTURAL_TYPES = ("aggregation", "superlative", "temporal")

# A model override must clear this bar. Small models are poorly calibrated and
# will happily return 0.9 for a guess, so this is a floor, not a safeguard on
# its own - the real protection is only consulting the model on 'lookup'.
MIN_OVERRIDE_CONFIDENCE = 0.6

CLASSIFIER_PROMPT = """You are classifying a question about {corpus_label}.

Categories:
- lookup       : a single fact about one named article
- multi_hop    : link a venue and/or date to an event (and its winner)
- temporal     : "immediately before/after <edition>", needs two editions linked
- aggregation  : count events satisfying a threshold across many documents
- superlative  : find the max/min event across many documents

Judge the *requirement* the question places on the system, not the words it
happens to use: a question is an aggregation question whenever answering it
means counting over many documents, however it is phrased.

Question: {question}

Reply with JSON only: {{"qtype": "<category>", "confidence": <0-1>, "reason": "<short>"}}
"""


def template_classifier_enabled() -> bool:
    """False under ``ABLATE_CLASSIFIER=1`` (row D of the generalisation suite).

    With the template classifier disabled the model's verdict is the only
    evidence available: the run measures the system on wording it was never
    tuned against. It is a measurement switch, never a default.

    Delegates to :func:`reasoning.query_parser.template_parser_enabled` so that
    the one switch removes the templates from *both* of their jobs - this
    classifier and the slot parser. Two independent readings of the same
    environment variable is exactly how a "no templates anywhere" ablation
    quietly stops being one.
    """
    from reasoning.query_parser import template_parser_enabled

    return template_parser_enabled()


def _corpus_label() -> str:
    """Human label for the corpus, from config (never hardcoded per domain)."""
    return getattr(getattr(_app_config, "domain", None), "corpus_label",
                   "the corpus")


class QuestionClassifier:
    """Classifies questions; exposes *why* a route was chosen."""

    def __init__(self, kg=None, llm=None) -> None:
        self.kg = kg
        self.llm = llm
        # Telemetry: how often the model was asked, whether it agreed with the
        # templates, where it was overruled and where its output was junk.
        # Reported in the dashboard as evidence of who actually decides.
        self.stats = {"asked": 0, "agreed": 0, "overrode": 0,
                      "rejected_low_confidence": 0, "rejected_invalid": 0,
                      "structural_override": 0, "templates_disabled": 0}

    def classify(self, question: str, spec: Optional[QuerySpec] = None,
                 counter: Optional[TokenCounter] = None) -> Dict[str, Any]:
        """Classify ``question``: the model decides, the templates validate.

        Order of evidence (see the module docstring):

        1. the LLM verdict, when it is a valid label at or above
           ``MIN_OVERRIDE_CONFIDENCE`` - the primary path, and the only path for
           wording the templates do not recognise;
        2. a positive template match for a *structural* type (a count, an
           extreme, an edition chain), which outranks a disagreeing model
           verdict because that capability is required whatever the phrasing;
        3. the template verdict alone, when the model is unavailable, unsure or
           unusable;
        4. the generic ``lookup`` bucket, which the router escalates when the
           confidence is low rather than acting on it.
        """
        templates_on = template_classifier_enabled()
        if not templates_on:
            self.stats["templates_disabled"] += 1
        rule_type = classify(question, self.kg) if templates_on else ""

        qtype = rule_type or "lookup"
        confidence = 0.9 if rule_type in STRUCTURAL_TYPES else 0.7
        method = "rules" if rule_type else "default"
        reason = (f"template match for '{rule_type}'" if rule_type
                  else "no template match: generic bucket")
        agreement: Optional[bool] = None
        rule_disagreement: Optional[Dict[str, str]] = None

        llm_ready = self.llm is not None and getattr(self.llm, "available", False)
        if llm_ready:
            self.stats["asked"] += 1
            payload = self.llm.complete_json(
                CLASSIFIER_PROMPT.format(question=question,
                                         corpus_label=_corpus_label()),
                caller="classifier.classify", counter=counter,
                model=self.llm.fast_model)
            candidate = str(payload.get("qtype", "")).strip().lower()
            try:
                cand_conf = float(payload.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                cand_conf = 0.0

            if candidate not in CLASSES:
                # Includes the {"error": "json_parse_failed"} shape that
                # complete_json returns for unparseable output - common on
                # small models, and precisely why we never trust it blindly.
                self.stats["rejected_invalid"] += 1
                reason = (f"template kept: model returned an unusable label "
                          f"({candidate or 'empty'!r})")
            elif cand_conf < MIN_OVERRIDE_CONFIDENCE:
                self.stats["rejected_low_confidence"] += 1
                reason = (f"template kept: model proposed '{candidate}' at "
                          f"{cand_conf:.2f} < {MIN_OVERRIDE_CONFIDENCE}")
            elif rule_type in STRUCTURAL_TYPES and candidate != rule_type:
                # A count, an extreme or an edition chain is a requirement on the
                # pipeline, not a matter of wording: it holds however the
                # question is phrased, so the structural match wins - recorded,
                # never hidden.
                self.stats["structural_override"] += 1
                rule_disagreement = {"llm": candidate, "rules": rule_type}
                method = "rules_structural"
                reason = (f"model said '{candidate}' but the question requires "
                          f"'{rule_type}': a top-k pipeline cannot produce an "
                          f"exhaustive (or edition-linked) result")
            else:
                agreement = (candidate == rule_type) if rule_type else None
                qtype = candidate
                confidence = cand_conf
                method = "llm"
                if agreement is False:
                    self.stats["overrode"] += 1
                    method = "llm_override"
                    rule_disagreement = {"llm": candidate, "rules": rule_type}
                elif agreement is True:
                    self.stats["agreed"] += 1
                reason = str(payload.get("reason", "llm classification"))[:200]

        return {"qtype": qtype, "method": method, "confidence": round(confidence, 3),
                "reason": reason, "rule_type": rule_type,
                "llm_agrees_with_rules": agreement,
                "rule_disagreement": rule_disagreement,
                "templates_enabled": templates_on,
                "policy": "llm_first_templates_validate"}


def classify_question(question: str, kg=None) -> str:
    """Backwards-compatible functional entry point."""
    return classify(question, kg)