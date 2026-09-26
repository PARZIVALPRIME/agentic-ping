"""Central thresholds and tunable constants.

Every magic number in the reasoning path used to be a literal at its point of
use, which made it impossible to audit the full set of tuning knobs or to answer
"what was that cutoff?" without grepping. They now live here, are documented
individually, and can be overridden from the environment (``.env``) exactly like
the rest of the configuration in :mod:`config`.

WHAT IS *NOT* IN HERE
---------------------
None of these values encodes a fact about the corpus, a question's wording, or a
benchmark answer. They are similarity floors, string lengths and confidence
floors - the kind of constant any retrieval system needs. Nothing here can be
tuned to make a specific question answerable.

The domain *schema* constants (``gold``/``silver``/``bronze``, the Olympics graph
relationships) deliberately stay in their modules: they are a statement about the
corpus we were given, not a knob.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Thresholds:
    """Tunable constants for linking, parsing, judging and routing."""

    # ── entity linking / string matching ───────────────────────────────
    #: Minimum fuzzy score for a question phrase to be accepted as a corpus
    #: sport name. Below this the linker reports "unresolved" rather than
    #: guessing, which lets the gap detector try a different route.
    sport_match_min: float = _f("THRESH_SPORT_MATCH_MIN", 0.72)
    #: Same, for venue phrases.
    venue_match_min: float = _f("THRESH_VENUE_MATCH_MIN", 0.72)
    #: Venues scoring within this band of the top hit are all kept, because
    #: one building legitimately appears under several corpus keys.
    venue_match_band: float = _f("THRESH_VENUE_MATCH_BAND", 0.06)
    #: A venue at or above this score is treated as an unambiguous hit.
    venue_strong_min: float = _f("THRESH_VENUE_STRONG_MIN", 0.80)
    #: Weight applied to token-containment similarity (handles "X Beijing").
    venue_containment_weight: float = _f("THRESH_VENUE_CONTAINMENT_WEIGHT", 0.93)
    #: Minimum score to accept a fuzzy article-title match for lookup questions.
    title_match_min: float = _f("THRESH_TITLE_MATCH_MIN", 0.85)

    # ── classification ────────────────────────────────────────────────
    #: An LLM classification only overrides the deterministic parse at or above
    #: this confidence. Small models are badly calibrated, so the effective
    #: protection is that the model is only consulted on the uncertain bucket.
    classifier_min_override_confidence: float = _f(
        "THRESH_CLASSIFIER_MIN_OVERRIDE", 0.6)
    #: Confidence recorded for a deterministic template match.
    rule_confidence: float = _f("THRESH_RULE_CONFIDENCE", 0.95)
    #: Confidence recorded when no template matched (the generic bucket), which
    #: is the signal the router uses to escalate to a more capable pipeline.
    rules_generic_confidence: float = _f("THRESH_RULES_GENERIC_CONFIDENCE", 0.7)

    # ── parsing ───────────────────────────────────────────────────────
    #: Semantic (LLM) parse slots below this confidence are treated as absent
    #: so the deterministic parser can supply them.
    parse_min_confidence: float = _f("THRESH_PARSE_MIN_CONFIDENCE", 0.0)
    #: Plausible range for a Games year; outside this a parsed year is rejected.
    year_min: int = _i("THRESH_YEAR_MIN", 1896)
    year_max: int = _i("THRESH_YEAR_MAX", 2035)
    #: Upper bound for a competitor-count threshold.
    threshold_max: int = _i("THRESH_THRESHOLD_MAX", 100000)

    # ── answer validation ─────────────────────────────────────────────
    #: Longest answer a model may submit; beyond this it is a paragraph, not a
    #: value, and no corpus field could have produced it.
    answer_max_chars: int = _i("THRESH_ANSWER_MAX_CHARS", 160)
    #: Below this length, substring grounding is too collision-prone to mean
    #: anything, so short answers are exempt from the containment check.
    grounding_min_chars: int = _i("THRESH_GROUNDING_MIN_CHARS", 2)
    #: Fraction of an answer's tokens that must appear in the tool evidence for a
    #: span answer to count as grounded (1.0 = verbatim containment).
    grounding_token_overlap: float = _f("THRESH_GROUNDING_TOKEN_OVERLAP", 1.0)

    # ── routing ───────────────────────────────────────────────────────
    #: Below this classification confidence the router escalates to the most
    #: capable pipeline instead of trusting the type-specific route.
    router_min_confidence: float = _f("THRESH_ROUTER_MIN_CONFIDENCE", 0.65)

    # ── conflicting versions of a fact ────────────────────────────────
    #: Confidence at or above which a conflict adjudication counts as decisive.
    #: On the resolver's scale an explicit correction scores 0.95, recency 0.9
    #: and entity succession 0.85, while a bare majority scores 0.5 + 0.4*share
    #: (<= 0.767 for a 2-of-3 split). At 0.8 the floor separates "a rule settled
    #: it" from "the sources are genuinely contested", and only the latter is
    #: recorded as an evidence gap.
    conflict_accept_confidence: float = _f("THRESH_CONFLICT_ACCEPT_CONFIDENCE", 0.8)


THRESHOLDS = Thresholds()


def thresholds() -> Thresholds:
    """The active thresholds (single import point for every module)."""
    return THRESHOLDS
