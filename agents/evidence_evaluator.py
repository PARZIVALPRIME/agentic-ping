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
                gaps.append("some candidate documents do not expose 'competitors'")
            if total <= 1:
                gaps.append("candidate set is suspiciously small")
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
                gaps.append("winner is not separated from the runner-up")
            if evidence.get("missing_field"):
                gaps.append("some candidate documents lack 'competitors'")
                base -= 0.08
            if evidence.get("candidates", 0) <= 2:
                gaps.append("very few candidates carry the comparison field")
                base -= 0.1

        elif kind == "temporal":
            signals.update(link or {})
            chain = (verification or {}).get("consistent")
            signals["anchor_confirmed"] = chain
            base = 0.62
            if chain:
                base += 0.2
            else:
                gaps.append("anchor edition not confirmed by the PREV/NEXT chain")
            matched = (link or {}).get("matched")
            if matched is None:
                gaps.append("no event matched the question's descriptor")
                base = 0.15
            else:
                match_score = float((link or {}).get("score", 0.0))
                signals["event_match_score"] = match_score
                base += 0.15 * match_score
                if match_score < 0.7:
                    gaps.append("event descriptor match is weak")
                if not getattr(matched, "gold", ""):
                    gaps.append("matched edition has no gold medal field")
                    base -= 0.2

        elif kind == "multi_hop":
            signals.update({k: v for k, v in (link or {}).items()
                            if k in ("score", "venue_score", "venue_affinity",
                                     "num_candidates")})
            base = 0.55 + 0.3 * float((link or {}).get("score", 0.0))
            if (link or {}).get("venue_affinity"):
                base += 0.05
            if (link or {}).get("matched") is None:
                gaps.append("no event matched the venue/date pair")
                base = 0.15
            elif not getattr((link or {}).get("matched"), "gold", ""):
                gaps.append("matched event has no gold medal field")
                base -= 0.2
            if not (link or {}).get("num_candidates"):
                gaps.append("venue did not resolve to any candidate events")

        else:  # lookup
            signals["title_score"] = (link or {}).get("score")
            base = 0.6 + 0.35 * float((link or {}).get("score", 0.0))
            if (link or {}).get("matched") is None:
                gaps.append("target article was not resolved")
                base = 0.15
            elif (link or {}).get("value") in (None, ""):
                gaps.append("resolved article has no 'nations' field")

        confidence = max(0.0, min(0.97, base))
        return {"confidence": round(confidence, 3), "gaps": gaps, "signals": signals,
                "sufficient": confidence >= 0.85 and not gaps}