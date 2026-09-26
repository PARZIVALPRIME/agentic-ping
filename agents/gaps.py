"""Shared evidence-gap vocabulary.

WHY THIS MODULE EXISTS
----------------------
Gaps used to travel as human-readable prose: :mod:`agents.evidence_evaluator`
appended a sentence such as ``"some candidate documents lack 'competitors'"``
and :mod:`agents.gap_detector` matched against its own copy of that sentence in
``_UNRECOVERABLE``. Rewording either side silently disabled the recovery - a
recovery step that stops firing produces no error and no visible change in the
trace, only a worse answer.

Gaps are now enum keys with one message table, so a producer and a consumer can
never drift apart: renaming a key is a compile-time error on both sides.

The keys are also what is recorded in a result file (``unresolved`` /
``gaps_remaining``), which makes them stable across runs and renderings, while
:func:`message` supplies the human text used in traces and the dashboard.
"""

from __future__ import annotations

from enum import Enum
from typing import Set


class Gap(str, Enum):
    """A named reason the gathered evidence is incomplete."""

    #: Some candidates in the set carry no value for the field being counted.
    CANDIDATES_MISSING_FIELD = "candidates_missing_field"
    #: The candidate set is so small it is probably a linking failure.
    CANDIDATE_SET_SMALL = "candidate_set_small"
    #: Very few candidates carry the comparison field.
    FEW_CANDIDATES_WITH_FIELD = "few_candidates_with_field"
    #: No event matched the question's descriptor.
    NO_EVENT_MATCHED = "no_event_matched"
    #: No event matched the venue/date pair.
    NO_VENUE_DATE_MATCH = "no_venue_date_match"
    #: The venue phrase did not resolve onto any candidate events.
    VENUE_UNRESOLVED = "venue_unresolved"
    #: The article the question names was not found in the corpus index.
    ARTICLE_UNRESOLVED = "article_unresolved"
    #: Structural traversal produced no candidate set at all.
    NO_CANDIDATE_SET = "no_candidate_set"
    #: The PREV/NEXT chain could not confirm the anchor edition.
    ANCHOR_NOT_CONFIRMED = "anchor_not_confirmed"
    #: The event descriptor matched only weakly.
    WEAK_EVENT_MATCH = "weak_event_match"
    #: The matched event has no medal field in the corpus.
    EVENT_MISSING_GOLD = "event_missing_gold"
    #: The resolved article has no nations field.
    ARTICLE_MISSING_NATIONS = "article_missing_nations"
    #: The extreme value is not separated from the runner-up.
    WINNER_NOT_SEPARATED = "winner_not_separated"
    #: The question does not state a season and it could not be resolved from
    #: evidence. Never silently defaulted - see ``agents.entity_linker``.
    SEASON_UNRESOLVED = "season_unresolved"
    #: Two or more evidence paths disagree and neither is authoritative.
    CONFLICTING_EVIDENCE = "conflicting_evidence"


MESSAGES = {
    Gap.CANDIDATES_MISSING_FIELD:
        "some candidate documents do not expose the aggregated field",
    Gap.CANDIDATE_SET_SMALL:
        "candidate set is suspiciously small",
    Gap.FEW_CANDIDATES_WITH_FIELD:
        "very few candidates carry the comparison field",
    Gap.NO_EVENT_MATCHED:
        "no event matched the question's descriptor",
    Gap.NO_VENUE_DATE_MATCH:
        "no event matched the venue/date pair",
    Gap.VENUE_UNRESOLVED:
        "venue did not resolve to any candidate events",
    Gap.ARTICLE_UNRESOLVED:
        "target article was not resolved",
    Gap.NO_CANDIDATE_SET:
        "no candidate set from structural traversal",
    Gap.ANCHOR_NOT_CONFIRMED:
        "anchor edition not confirmed by the PREV/NEXT chain",
    Gap.WEAK_EVENT_MATCH:
        "event descriptor match is weak",
    Gap.EVENT_MISSING_GOLD:
        "matched event has no gold medal field in the corpus",
    Gap.ARTICLE_MISSING_NATIONS:
        "resolved article has no nations field",
    Gap.WINNER_NOT_SEPARATED:
        "winner is not separated from the runner-up",
    Gap.SEASON_UNRESOLVED:
        "the question does not state a season and none could be established "
        "from the evidence",
    Gap.CONFLICTING_EVIDENCE:
        "structured and textual evidence disagree",
}

#: Gaps no retrieval action can close: the corpus itself lacks the field, or the
#: question is genuinely undetermined. These are recorded as accepted
#: limitations rather than retried. Note that ``SEASON_UNRESOLVED`` is *not* in
#: this set: an unstated season can be recovered by enumerating every candidate
#: edition, which is a real retrieval action.
UNRECOVERABLE: Set[Gap] = {
    Gap.CANDIDATES_MISSING_FIELD,
    Gap.EVENT_MISSING_GOLD,
    Gap.ARTICLE_MISSING_NATIONS,
    Gap.WINNER_NOT_SEPARATED,
}


def message(gap: "Gap | str") -> str:
    """Human-readable text for a gap key (used in traces and dashboards)."""
    try:
        return MESSAGES[Gap(gap)]
    except (ValueError, KeyError):
        return str(gap)


def is_unrecoverable(gap: "Gap | str") -> bool:
    """True when no retrieval action can close this gap."""
    try:
        return Gap(gap) in UNRECOVERABLE
    except ValueError:
        return False


def as_gap(value: "Gap | str | None") -> "Gap | None":
    """Coerce a stored value back to a :class:`Gap` (or None)."""
    if value is None:
        return None
    if isinstance(value, Gap):
        return value
    try:
        return Gap(value)
    except ValueError:
        return None
