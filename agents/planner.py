"""Planner agent: build the investigation plan, with the LLM as the strategist.

Two responsibilities, both driven by the same idea - *the rules handle the
templates, the LLM handles the surprises*:

1. **Slot refinement.** The regex parser is exact for the corpus's question
   templates, but a hidden-set paraphrase may not match. When the parse leaves
   required slots empty, the LLM is asked to fill them. Its answer is validated
   field-by-field (whitelists, ranges, membership in the graph's sport
   vocabulary) before it can touch the spec, so a bad generation cannot corrupt
   the deterministic pipeline.
2. **Replanning.** When the evidence evaluator reports gaps, the gap detector
   proposes recovery actions; the LLM picks which one to run and why. Without
   an LLM the planner falls back to the first recoverable action, so the loop
   always makes progress.

Every decision is recorded on the state, which is what makes the agentic traces
in the dashboard readable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agents.state import AgentState, PlannedStep
from config import config as _app_config
from reasoning.query_parser import QuerySpec


def _corpus_label() -> str:
    """How prompts name the corpus: configuration, never hardcoded per domain."""
    return getattr(getattr(_app_config, "domain", None), "corpus_label",
                   "the corpus")

SLOT_PROMPT = """Extract structured slots from a question about {corpus_label}.

Fields (omit any you cannot determine):
  qtype        one of lookup | multi_hop | temporal | aggregation | superlative
  sport        the sport name exactly as it appears in the question
  year         the 4-digit year the question mentions
  season       the season/category the question names (e.g. "Summer", "Winter")
  threshold    the competitor count the question compares against (integer)
  comparator   one of gt | gte | lt | lte  (gt="more than", gte="at least",
               lt="fewer than", lte="at most")
  direction    "max" or "min" for highest/lowest questions
  venue        the venue name the question mentions
  date_text    the date phrase the question mentions (e.g. "28 July 2012")
  event_desc   the event description, with the sport name removed
  target_title the article title for lookup questions
  before_year  the year in an "immediately before <year>" question

Question: {question}

Known sports in the corpus (choose the closest match if the question paraphrases):
{sports}

Reply with JSON only."""

PLAN_PROMPT = """An agent is answering a question about {corpus_label} and its evidence has gaps.

Question: {question}
Question type: {qtype}
Evidence so far: {summary}
Unresolved gaps: {gaps}

Available recovery actions:
{actions}

Pick exactly ONE action index that is most likely to close the most important
gap, or -1 if no action can help. Reply with JSON only:
{{"action": <index>, "reason": "<one short sentence>"}}"""


class Planner:
    """Deterministic plan templates + LLM slot refinement and replanning."""

    def __init__(self, kg=None, llm=None) -> None:
        self.kg = kg
        self.llm = llm

    # ─ plan templates ─────────────────────────────────────────────────
    def plan_for(self, spec: QuerySpec) -> List[PlannedStep]:
        qtype = spec.qtype
        if qtype == "aggregation":
            return [
                PlannedStep("EntityLinker", "link_entities",
                            "resolve sport/games onto graph vertices"),
                PlannedStep("GraphTraverser", "traverse_graph",
                            "enumerate EVERY event of the sport at those Games "
                            "(top-k retrieval cannot close this set)"),
                PlannedStep("Aggregator", "count_threshold",
                            "filter the candidate set on the 'competitors' field"),
                PlannedStep("Aggregator", "verify_by_recount",
                            "recount through an independent code path"),
                PlannedStep("EvidenceEvaluator", "evaluate_evidence",
                            "audit completeness, coverage and the recount"),
                PlannedStep("Synthesizer", "render_answer", "render the count"),
            ]
        if qtype == "superlative":
            return [
                PlannedStep("EntityLinker", "link_entities",
                            "resolve sport/games onto graph vertices"),
                PlannedStep("GraphTraverser", "traverse_graph",
                            "enumerate the full candidate set for argmax/argmin"),
                PlannedStep("Comparator", "arg_extreme",
                            "find the extreme value and its margin"),
                PlannedStep("Comparator", "verify_extreme",
                            "cross-check the extreme without relying on sort order"),
                PlannedStep("EvidenceEvaluator", "evaluate_evidence",
                            "score sufficiency and the runner-up margin"),
                PlannedStep("Synthesizer", "render_answer", "render the winner"),
            ]
        if qtype == "temporal":
            return [
                PlannedStep("EntityLinker", "link_entities",
                            "resolve the Games edition and the event"),
                PlannedStep("TemporalReasoner", "resolve_anchor",
                            "walk the edition chain to the Games immediately before"),
                PlannedStep("TemporalReasoner", "resolve_event",
                            "match the event inside the resolved edition"),
                PlannedStep("TemporalReasoner", "verify_chain",
                            "confirm the anchor via the PREV/NEXT fields"),
                PlannedStep("VectorSearcher", "vector_search",
                            "independent passage evidence for the same edition",
                            optional=True),
                PlannedStep("EvidenceEvaluator", "evaluate_evidence",
                            "score the chain confirmation and match strength"),
                PlannedStep("Synthesizer", "render_answer", "render the medallist"),
            ]

        if qtype == "multi_hop":
            return [
                PlannedStep("EntityLinker", "link_entities",
                            "resolve venue + date phrases onto vertices"),
                PlannedStep("LookupResolver", "resolve_venue_date",
                            "(venue, date) -> event page"),
                PlannedStep("GraphTraverser", "traverse_graph",
                            "same-venue siblings as corroborating evidence",
                            optional=True),
                PlannedStep("VectorSearcher", "vector_search",
                            "independent passage evidence for the event",
                            optional=True),
                PlannedStep("EvidenceEvaluator", "evaluate_evidence",
                            "score the date match and venue/sport affinity"),
                PlannedStep("Synthesizer", "render_answer", "render the gold winner"),
            ]
        return [
            PlannedStep("EntityLinker", "link_entities", "resolve the article title"),
            PlannedStep("LookupResolver", "resolve_article",
                        "title -> event page, read the requested field"),
            PlannedStep("VectorSearcher", "vector_search",
                        "passage evidence for the same article", optional=True),
            PlannedStep("EvidenceEvaluator", "evaluate_evidence",
                        "score the title match and field availability"),
            PlannedStep("Synthesizer", "render_answer", "render the field value"),
        ]

    # ─ LLM slot refinement ────────────────────────────────────────────
    def needs_slot_refinement(self, spec: QuerySpec) -> Tuple[bool, List[str]]:
        """Which required slots are missing for this question type?"""
        required = {
            "aggregation": (("sport", spec.sport), ("year", spec.year),
                            ("threshold", spec.threshold)),
            "superlative": (("sport", spec.sport), ("year", spec.year)),
            "temporal": (("before_year", spec.before_year),
                         ("event_desc", spec.event_desc)),
            "multi_hop": (("venue", spec.venue),),
        }.get(spec.qtype, (("target_title", spec.target_title),))
        missing = [name for name, value in required if not value]
        return bool(missing), missing

    def refine_slots(self, question: str, spec: QuerySpec,
                     counter: Optional[TokenCounter] = None) -> Dict[str, Any]:
        """Ask the LLM to fill empty slots. Returns a report (for the trace)."""
        report: Dict[str, Any] = {"used": False, "filled": [], "rejected": []}
        needs, missing = self.needs_slot_refinement(spec)
        if not needs or self.llm is None or not getattr(self.llm, "available", False):
            return report
        sports = ", ".join(getattr(self.kg, "sport_vocabulary", [])[:80]) or "(unknown)"
        payload = self.llm.complete_json(
            SLOT_PROMPT.format(question=question, sports=sports,
                               corpus_label=_corpus_label()),
            caller="planner.slots", model=self.llm.fast_model, counter=counter)
        if not payload:
            return report
        report["used"] = True
        report["raw"] = {k: v for k, v in payload.items() if v}

        # The LLM may also correct the question type when the rules fell through
        if "qtype" in payload and missing:
            coerced_type = self._coerce("qtype", payload["qtype"], spec)
            if coerced_type and coerced_type != spec.qtype:
                report["filled"].append({"qtype": coerced_type})
                return {"used": True, "replan_qtype": coerced_type,
                        "raw": report["raw"], "filled": report["filled"],
                        "rejected": report["rejected"]}

        for field, value in payload.items():
            if field not in missing or value in (None, "", []):
                continue
            coerced = self._coerce(field, value, spec)
            if coerced is None:
                report["rejected"].append({field: value})
                continue
            setattr(spec, field, coerced)
            report["filled"].append({field: coerced})
        return report

    def _coerce(self, field: str, value: Any, spec: QuerySpec) -> Optional[Any]:
        """Validate an LLM-provided slot value; None means reject."""
        if field in ("year", "before_year", "threshold"):
            try:
                number = int(float(str(value).strip()))
            except (TypeError, ValueError):
                return None
            if field == "threshold":
                return number if 0 < number < 100000 else None
            # Plausible years come from configuration: a hallucinated year is
            # rejected instead of being allowed into a slot.
            domain = _app_config.domain
            return number if domain.year_min <= number <= domain.year_max else None
        text = str(value).strip()
        if not text or len(text) > 120:
            return None
        if field == "season":
            return text.title() if text.title() in ("Summer", "Winter") else None
        if field == "comparator":
            return text.lower() if text.lower() in ("gt", "gte", "lt", "lte") else None
        if field == "direction":
            return text.lower() if text.lower() in ("max", "min") else None
        if field == "qtype":
            return text.lower() if text.lower() in (
                "lookup", "multi_hop", "temporal", "aggregation",
                "superlative") else None
        if field == "sport" and getattr(self.kg, "sport_vocabulary", None):
            from reasoning.solvers import resolve_sport

            return resolve_sport(text, self.kg) or None
        return text

    # ─ LLM replanning ─────────────────────────────────────────────────
    def choose_recovery(self, state: AgentState, actions: List[Any]) -> Dict[str, Any]:
        """Pick which proposed recovery action to run next."""
        if not actions:
            return {"chosen": -1, "reason": "no actions proposed", "method": "none"}
        fallback = next((i for i, a in enumerate(actions) if a.recoverable), -1)
        if fallback < 0:
            return {"chosen": -1, "reason": "all gaps are unrecoverable",
                    "method": "rules"}
        if self.llm is None or not getattr(self.llm, "available", False):
            return {"chosen": fallback, "reason": actions[fallback].reason,
                    "method": "rules"}

        listing = "\n".join(
            f"[{i}] agent={a.agent} operation={a.operation} "
            f"targets_gap={a.gap!r} reason={a.reason}"
            for i, a in enumerate(actions))
        summary = (f"{len(state.documents)} documents, {len(state.chunks)} passages, "
                   f"confidence {state.confidence:.2f}, {state.num_steps} steps run")
        payload = self.llm.complete_json(
            PLAN_PROMPT.format(question=state.question, qtype=state.qtype,
                               summary=summary,
                               gaps="; ".join(state.missing_info) or "none",
                               actions=listing,
                               corpus_label=_corpus_label()),
            caller="planner.replan", model=self.llm.fast_model)
        try:
            chosen = int(payload.get("action", fallback))
        except (TypeError, ValueError):
            chosen = fallback
        if not 0 <= chosen < len(actions) or not actions[chosen].recoverable:
            return {"chosen": fallback, "reason": "llm choice invalid; using rules",
                    "method": "rules_fallback"}
        return {"chosen": chosen,
                "reason": str(payload.get("reason", actions[chosen].reason))[:200],
                "method": "llm"}