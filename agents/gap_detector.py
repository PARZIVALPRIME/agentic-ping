"""Gap detector agent: turn missing evidence into concrete next actions.

The evidence evaluator names *what* is missing; this agent decides *what to do
about it*. Each gap maps to at most one recovery action (a retrieval widening,
a different index, or an explicit "no retrieval can help - accept the
limitation"). The orchestrator appends the returned actions to its plan, which
is how the investigation adapts mid-run (the `strategy_changed` metric).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from reasoning.query_parser import QuerySpec

from .gaps import Gap, is_unrecoverable


@dataclass
class GapAction:
    """A single recovery step proposed for an unresolved gap."""

    gap: str
    agent: str
    operation: str
    reason: str
    params: Dict[str, Any] = field(default_factory=dict)
    recoverable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"gap": self.gap, "agent": self.agent, "operation": self.operation,
                "reason": self.reason, "recoverable": self.recoverable}


# Gaps that no retrieval action can fix (the corpus itself lacks the field) live
# in :mod:`agents.gaps` as ``UNRECOVERABLE`` - a single definition shared with the
# evaluator that produces them, so the two can never disagree.


class GapDetector:
    """Maps evidence gaps to recovery steps, or marks them unresolvable."""

    def detect(self, spec: QuerySpec, kind: str, gaps: List[str],
               state: Any = None) -> List[GapAction]:
        actions: List[GapAction] = []
        seen: set = set()
        for gap in gaps:
            if gap in seen:
                continue
            seen.add(gap)
            action = self._action_for(spec, kind, gap, state)
            if action is not None and action.gap not in _proposed(state):
                actions.append(action)
        return actions

    # ── gap -> action mapping ──────────────────────────────────────────
    def _action_for(self, spec: QuerySpec, kind: str, gap: str,
                    state: Any) -> Optional[GapAction]:
        if is_unrecoverable(gap):
            return GapAction(gap=gap, agent="GapDetector", operation="accept_limitation",
                             reason="no retrieval action can supply this field",
                             recoverable=False)

        if gap in (Gap.CANDIDATE_SET_SMALL, Gap.FEW_CANDIDATES_WITH_FIELD):
            return self._widen_traversal(spec, gap)

        if gap == Gap.SEASON_UNRESOLVED:
            # The Games edition is not settled. Enumerate every candidate
            # edition rather than committing to one, and let the event match and
            # the evidence audit decide - the alternative is answering a Winter
            # question from the Summer Games.
            return GapAction(gap=gap, agent="GraphTraverser",
                             operation="traverse_graph",
                             reason="season not established from the question or "
                                    "the evidence: enumerate every candidate edition",
                             params={"widen_season": True}, recoverable=True)

        if gap in (Gap.NO_EVENT_MATCHED, Gap.NO_VENUE_DATE_MATCH,
                   Gap.VENUE_UNRESOLVED, Gap.ARTICLE_UNRESOLVED,
                   Gap.NO_CANDIDATE_SET):
            return GapAction(gap=gap, agent="VectorSearcher", operation="vector_search",
                             reason="structural linking failed; fall back to text "
                                    "retrieval over the chunk index",
                             params={"widening": self._widening_for(kind)},
                             recoverable=True)

        if gap == Gap.CONFLICTING_EVIDENCE:
            # Two readings of the same fact disagree. The recovery is not *more*
            # of the same evidence - it is a different source class: the prose
            # passages, where a correction or a newer statement would live. If
            # that finishes the budget, the gap stays recorded as unresolved,
            # which is the honest outcome for a fact the corpus never settles.
            return GapAction(gap=gap, agent="VectorSearcher", operation="vector_search",
                             reason="structured and textual readings of the same fact "
                                    "disagree: retrieve passages for the authoritative "
                                    "or superseding statement",
                             params={"widening": 2, "conflict": True}, recoverable=True)

        if gap == Gap.ANCHOR_NOT_CONFIRMED:
            return GapAction(gap=gap, agent="TemporalReasoner",
                             operation="edition_chain_fallback",
                             reason="verify the anchor against the corpus-wide "
                                    "edition ordering instead of the infobox next-field",
                             recoverable=True)

        if gap == Gap.WEAK_EVENT_MATCH:
            return GapAction(gap=gap, agent="VectorSearcher", operation="vector_search",
                             reason="re-rank the top editions by passage evidence",
                             params={"widening": 2}, recoverable=True)

        # Unknown gap: default to a widened vector search, the safest recovery.
        return GapAction(gap=gap, agent="VectorSearcher", operation="vector_search",
                         reason="generic recovery: widened text retrieval",
                         params={"widening": 1}, recoverable=True)

    def _widen_traversal(self, spec: QuerySpec, gap: str) -> GapAction:
        """Loosen the graph traversal constraints one notch."""
        if spec.venue:
            return GapAction(
                gap=gap, agent="GraphTraverser", operation="traverse_graph",
                reason="candidate set too small: re-traverse without the venue "
                       "constraint, using sport + games only",
                params={"mode": "sport_games"}, recoverable=True)
        if spec.year and spec.season:
            return GapAction(
                gap=gap, agent="GraphTraverser", operation="traverse_graph",
                reason="candidate set too small: re-traverse across the whole sport",
                params={"mode": "sport_wide"}, recoverable=True)
        return GapAction(
            gap=gap, agent="VectorSearcher", operation="vector_search",
            reason="no structural constraint available; use text retrieval",
            params={"widening": 2}, recoverable=True)

    @staticmethod
    def _widening_for(kind: str) -> int:
        return {"aggregation": 1, "superlative": 1, "temporal": 2,
                "multi_hop": 1, "lookup": 2}.get(kind, 1)


def _proposed(state: Any) -> List[str]:
    """Gaps already proposed for in this run (prevents recovery loops)."""
    if state is None:
        return []
    return getattr(state, "proposed_gaps", [])
