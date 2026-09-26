"""Evidence evaluator agent: decide whether the gathered evidence is sufficient.

This is the stopping-criterion brain of the agentic loop. Instead of trusting a
single retrieval confidence number, it audits the evidence itself:

  * candidate-set completeness  (did the graph actually yield a closed set?)
  * field coverage              (how many candidates carried the field?)
  * verification signals        (independent recount / argmax verification)
  * decision margin             (how far is the winner from the runner-up?)
  * retrieval provenance        (were documents reached structurally or by fuzzy
                                 text search, and how strong was the link?)

The output is a calibrated confidence plus an explicit list of what is still
missing - which the gap detector turns into the next retrieval action.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .gaps import Gap

from reasoning.query_parser import QuerySpec


class EvidenceEvaluator:
    """Scores sufficiency and names the remaining gaps."""

    def __init__(self, kg=None) -> None:
        self.kg = kg

    def evaluate(self, spec: QuerySpec, kind: str, evidence: Dict[str, Any],
                 link: Optional[Dict[str, Any]] = None,
                 verification: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        gaps: List[str] = []
        signals: Dict[str, Any] = {}
        base = 0.5

        if kind == "aggregation":
            total = max(1, evidence.get("candidates", 0))
            completeness = evidence.get("field_completeness", 0.0)
            signals["candidate_set_size"] = total
            signals["field_completeness"] = round(completeness, 3)
            signals["exhaustive"] = bool(evidence.get("exhaustive"))
            if verification:
                signals["independent_recount"] = verification.get("recount")
                signals["recount_matches"] = (
                    verification.get("recount") == evidence.get("count"))
            base = 0.62 * completeness + 0.18
            if evidence.get("exhaustive"):
                base += 0.18
            else:
                gaps.append(Gap.CANDIDATES_MISSING_FIELD)
            if total <= 1:
                gaps.append(Gap.CANDIDATE_SET_SMALL)
                base -= 0.15
            if evidence.get("count", 0) == 0:
                base -= 0.05

        elif kind == "superlative":
            signals["candidates"] = evidence.get("candidates", 0)
            signals["margin"] = evidence.get("margin")
            signals["unique_winner"] = evidence.get("unique_winner")
            if verification:
                signals["extreme_verified"] = verification.get("verified")
            base = 0.7
            if evidence.get("unique_winner"):
                base += 0.15
            else:
                base -= 0.1
                gaps.append(Gap.WINNER_NOT_SEPARATED)
            if evidence.get("missing_field"):
                gaps.append(Gap.FEW_CANDIDATES_WITH_FIELD)
                base -= 0.08
            if evidence.get("candidates", 0) <= 2:
                gaps.append(Gap.FEW_CANDIDATES_WITH_FIELD)
                base -= 0.1

        elif kind == "temporal":
            signals.update(link or {})
            # An unstated season that no evidence settled is a real gap: the
            # edition being asked about is not established, so any answer would
            # rest on a guess. It is recoverable (enumerate both editions).
            if (link or {}).get("season_unresolved") or (link or {}).get("season_ambiguous"):
                gaps.append(Gap.SEASON_UNRESOLVED)
            chain = (verification or {}).get("consistent")
            signals["anchor_confirmed"] = chain
            base = 0.62
            if chain:
                base += 0.2
            else:
                gaps.append(Gap.ANCHOR_NOT_CONFIRMED)
            matched = (link or {}).get("matched")
            if matched is None:
                gaps.append(Gap.NO_EVENT_MATCHED)
                base = 0.15
            else:
                match_score = float((link or {}).get("score", 0.0))
                signals["event_match_score"] = match_score
                base += 0.15 * match_score
                if match_score < 0.7:
                    gaps.append(Gap.WEAK_EVENT_MATCH)
                if not getattr(matched, "gold", ""):
                    gaps.append(Gap.EVENT_MISSING_GOLD)
                    base -= 0.2

        elif kind == "multi_hop":
            signals.update({k: v for k, v in (link or {}).items()
                            if k in ("score", "venue_score", "venue_affinity",
                                     "num_candidates")})
            base = 0.55 + 0.3 * float((link or {}).get("score", 0.0))
            if (link or {}).get("venue_affinity"):
                base += 0.05
            if (link or {}).get("matched") is None:
                gaps.append(Gap.NO_VENUE_DATE_MATCH)
                base = 0.15
            elif not getattr((link or {}).get("matched"), "gold", ""):
                gaps.append(Gap.EVENT_MISSING_GOLD)
                base -= 0.2
            if not (link or {}).get("num_candidates"):
                gaps.append(Gap.VENUE_UNRESOLVED)

        else:  # lookup
            signals["title_score"] = (link or {}).get("score")
            base = 0.6 + 0.35 * float((link or {}).get("score", 0.0))
            if (link or {}).get("matched") is None:
                gaps.append(Gap.ARTICLE_UNRESOLVED)
                base = 0.15
            elif (link or {}).get("value") in (None, ""):
                gaps.append(Gap.ARTICLE_MISSING_NATIONS)

        confidence = max(0.0, min(0.97, base))
        return {"confidence": round(confidence, 3), "gaps": gaps, "signals": signals,
                "sufficient": confidence >= 0.85 and not gaps}