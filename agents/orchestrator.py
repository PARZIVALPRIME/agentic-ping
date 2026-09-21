"""Orchestrator agent: the planner-executor loop behind the agentic pipeline.

The orchestrator is the only component that knows the *whole* investigation. It:

1. classifies the question (rules, with an LLM refinement pass),
2. lets the planner refine missing slots with the LLM when the parse is
   incomplete (hidden-set paraphrases),
3. builds a typed plan (a DAG of agent operations) for the question type,
4. executes the plan step by step, recording every observation, token and
   millisecond on the shared blackboard,
5. after each evidence audit, asks the gap detector + planner for recovery
   steps and splices them in before synthesis (the strategy adaptation),
6. stops on one of four explicit criteria, and
7. synthesises the final answer, with the LLM adjudicating the structured
   candidate against the retrieved passages.

The stopping criteria (from the plan):
  * confidence >= threshold          -> evidence is sufficient
  * plan exhausted                   -> nothing left to run
  * stale streak >= limit            -> the last N steps found no new documents
  * max steps reached                -> hard budget
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from reasoning.query_parser import QuerySpec, parse_question
from reasoning.solvers import resolve_sport, resolve_venue
from utils.llm import refine_answer
from utils.metrics import TokenCounter

from .aggregator import Aggregator
from .classifier import QuestionClassifier
from .comparator import Comparator
from .entity_linker import EntityLinker
from .evidence_evaluator import EvidenceEvaluator
from .gap_detector import GapDetector
from .graph_traverser import GraphTraverser
from .lookup_resolver import LookupResolver
from .planner import Planner
from .react_agent import ReActAgent
from .state import AgentState, PlannedStep
from .synthesizer import Synthesizer
from .temporal_reasoner import TemporalReasoner
from .vector_searcher import VectorSearcher

DEFAULT_CFG = {
    "max_steps": 15,
    "max_replans": 3,
    "confidence_threshold": 0.9,
    "stale_step_limit": 2,
    "max_widen_attempts": 2,
    "agent_mode": "hybrid",
    "react_max_steps": 8,
    "react_retries": 2,
    "react_model": "",
}

KINDS = ("lookup", "multi_hop", "temporal", "aggregation", "superlative")


class OrchestratorAgent:
    """Plans, executes, adapts and synthesises - one question per run."""

    def __init__(self, kg, index, llm=None, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.kg = kg
        self.index = index
        self.llm = llm
        self.cfg = dict(DEFAULT_CFG)
        if cfg:
            self.cfg.update({k: v for k, v in cfg.items() if v is not None})

        self.classifier = QuestionClassifier(kg, llm)
        self.planner = Planner(kg, llm)
        self.linker = EntityLinker(kg)
        self.traverser = GraphTraverser(kg)
        self.searcher = VectorSearcher(index)
        self.temporal = TemporalReasoner(kg)
        self.aggregator = Aggregator(kg)
        self.comparator = Comparator(kg)
        self.resolver = LookupResolver(kg)
        self.evaluator = EvidenceEvaluator(kg)
        self.gap_detector = GapDetector()
        self.synth = Synthesizer(llm)
        self.react = ReActAgent(kg, index, llm, self.cfg)

    # ── main entry point ───────────────────────────────────────────────
    def run(self, question: str, qid: str = "") -> AgentState:
        spec = parse_question(question, self.kg)
        state = AgentState(question, spec, qid)
        state.kind = self._kind(spec.qtype)

        mode = str(self.cfg.get("agent_mode", "hybrid") or "hybrid").lower()
        if mode in ("react", "hybrid") and self.react.available:
            self.react.run(question, qid, spec, state)
            state.classification = {
                "qtype": state.kind, "method": "ReActAgent", "mode": mode,
                "stop_reason": state.stop_reason, "steps": state.iterations,
            }
            state.slot_report = {"method": "react_tool_calling",
                                 "tool_calls": len(state.tool_calls)}
            if state.answer or mode == "react":
                # 'react' is the pure-ablation mode: whatever the loop produced
                # is the result, by definition.
                if mode == "react":
                    return state
                # 'hybrid' promises "the LLM drives, deterministic solvers
                # rescue". Rescue has to be triggered by *evidence*, not by
                # emptiness. An answer the model typed in prose never passed
                # through a tool, so nothing in the system has checked it
                # against the graph; a small model produces exactly that -
                # fluent, short, and unsupported. Only an answer submitted via
                # the submit_answer tool has been through the grounded path, so
                # only that one is allowed to skip the deterministic solvers.
                if state.stop_reason == "submitted_answer":
                    return state
                state.strategy_changes.append(
                    f"discarded ungrounded ReAct answer ({state.stop_reason}); "
                    "no tool call backed it")
                state.answer = ""
            state.strategy_changes.append(
                "fallback: deterministic planner/executor (ReAct produced no "
                "grounded answer)")

        self._run_deterministic(question, spec, state)
        return state

    def _run_deterministic(self, question: str, spec: QuerySpec,
                           state: AgentState) -> None:
        """The deterministic planner/executor path (rules + curated solvers)."""
        # 1. classification (rules + LLM refinement on the generic bucket)
        verdict = self.classifier.classify(question, spec, counter=state.tokens)
        if verdict["method"] != "rules" and verdict["qtype"] in KINDS:
            spec.qtype = verdict["qtype"]
            state.kind = self._kind(spec.qtype)
        state.classification = verdict

        # 2. LLM slot refinement when the deterministic parse is incomplete
        slot_report = self.planner.refine_slots(question, spec, counter=state.tokens)
        state.slot_report = slot_report
        if slot_report.get("replan_qtype"):
            spec.qtype = slot_report["replan_qtype"]
            state.qtype = spec.qtype
            state.kind = self._kind(spec.qtype)
            state.strategy_changes.append(
                f"reclassified to '{spec.qtype}' by LLM slot extraction")
        if slot_report.get("filled"):
            state.strategy_changes.append(
                "LLM filled slots: " + ", ".join(
                    f"{k}={v}" for item in slot_report["filled"]
                    for k, v in item.items()))

        # 3. plan
        state.spec = spec
        state.set_plan(self.planner.plan_for(spec),
                       reason=f"template for '{spec.qtype}'")

        # 4. execute the plan (with adaptive recovery)
        self._execute(state)

    # ── execution engine ───────────────────────────────────────────────
    def _execute(self, state: AgentState) -> None:
        replans = 0
        widen_attempts = 0

        while True:
            if state.num_steps >= self.cfg["max_steps"]:
                state.stop_reason = "max_steps_reached"
                break

            step = state.next_step()
            if step is None:
                state.stop_reason = state.stop_reason or "plan_exhausted"
                break

            if step.optional and not self._should_run_optional(state, step):
                step.skipped = True
                step.skip_reason = "optional step skipped: primary evidence already resolved"
                continue

            self._run_step(state, step)

            # adaptive recovery after an evidence audit
            if step.operation == "evaluate_evidence":
                if state.confidence >= self.cfg["confidence_threshold"]:
                    # Evidence is sufficient: stop *investigating*, but still
                    # synthesise the answer - so skip ahead to the Synthesizer.
                    state.stop_reason = "confidence_reached"
                    self._skip_to_synthesis(state)
                    continue
                if (replans < self.cfg["max_replans"]
                        and widen_attempts < self.cfg["max_widen_attempts"]):
                    if self._plan_recovery(state):
                        replans += 1
                        widen_attempts += 1
                        state.iterations = replans
                        continue

            if state.stale_streak >= self.cfg["stale_step_limit"] and state.answer:
                # Preserve a more specific reason (e.g. "confidence_reached") if
                # one was already recorded; staleness is only the fallback.
                state.stop_reason = state.stop_reason or "no_new_information"
                break

        state.stop_reason = state.stop_reason or "plan_exhausted"

        # A run must always produce an answer: if the budget or an early stop
        # pre-empted synthesis, run it now.
        if not state.answer:
            pending = next((s for s in state.plan
                            if s.agent == "Synthesizer" and not s.done), None)
            if pending is not None:
                self._run_step(state, pending)

    def _run_step(self, state: AgentState, step: PlannedStep) -> None:
        before_in = state.tokens.input_tokens
        before_out = state.tokens.output_tokens
        t0 = time.perf_counter()
        try:
            detail, observation, new_docs = self._dispatch(state, step)
        except Exception as exc:  # a failing step must not kill the run
            detail = f"step failed: {exc.__class__.__name__}: {exc}"
            observation = {"error": str(exc)[:300]}
            new_docs = 0
        latency = (time.perf_counter() - t0) * 1000.0
        step.done = True
        state.record(step.agent, step.operation, detail, observation, new_docs,
                     latency_ms=latency,
                     input_tokens=state.tokens.input_tokens - before_in,
                     output_tokens=state.tokens.output_tokens - before_out)

    def _skip_to_synthesis(self, state: AgentState) -> None:
        """Mark every not-yet-run step before the Synthesizer as skipped.

        Used when the evidence audit says the case is already closed: the loop
        then executes only the synthesis step, so a sufficient-evidence run
        still produces an answer (rather than stopping with an empty one).
        """
        seen_synth = False
        for step in state.plan[state.plan_cursor:]:
            if step.agent == "Synthesizer":
                break
            if not step.done and not step.skipped:
                step.skipped = True
                step.skip_reason = "skipped: evidence audited as sufficient"

    def _plan_recovery(self, state: AgentState) -> bool:
        """Detect gaps, ask the planner to choose a recovery, splice it in."""
        actions = self.gap_detector.detect(state.spec, state.kind,
                                           list(state.missing_info), state)
        if not actions:
            return False
        choice = self.planner.choose_recovery(state, actions)
        idx = choice.get("chosen", -1)
        if idx < 0 or idx >= len(actions):
            return False
        action = actions[idx]
        if not action.recoverable:
            state.resolve_gap(action.gap)
            state.strategy_changes.append(
                f"accepted limitation for gap {action.gap!r}: {action.reason}")
            return False

        recovery = PlannedStep(action.agent, action.operation,
                               f"RECOVERY for {action.gap!r}: {action.reason}")
        reevaluate = PlannedStep("EvidenceEvaluator", "evaluate_evidence",
                                 "re-audit after recovery step")
        # splice both before the synthesis step so the answer reflects them
        insert_at = next((i for i, s in enumerate(state.plan[state.plan_cursor:],
                                                  start=state.plan_cursor)
                          if s.agent == "Synthesizer"), len(state.plan))
        state.plan[insert_at:insert_at] = [recovery, reevaluate]
        state.proposed_gaps.append(action.gap)
        state.strategy_changes.append(
            f"replan[{choice.get('method')}] on gap {action.gap!r} -> "
            f"{action.agent}.{action.operation} :: {choice.get('reason', '')}")
        recovery.params = action.params  # type: ignore[attr-defined]
        return True

    # ── dispatcher ─────────────────────────────────────────────────────
    def _dispatch(self, state: AgentState, step: PlannedStep) -> Tuple[str, Dict[str, Any], int]:
        params = getattr(step, "params", {}) or {}
        handler = {
            ("EntityLinker", "link_entities"): self._op_link_entities,
            ("GraphTraverser", "traverse_graph"): self._op_traverse_graph,
            ("VectorSearcher", "vector_search"): self._op_vector_search,
            ("Aggregator", "count_threshold"): self._op_count_threshold,
            ("Aggregator", "verify_by_recount"): self._op_verify_recount,
            ("Comparator", "arg_extreme"): self._op_arg_extreme,
            ("Comparator", "verify_extreme"): self._op_verify_extreme,
            ("TemporalReasoner", "resolve_anchor"): self._op_resolve_anchor,
            ("TemporalReasoner", "resolve_event"): self._op_resolve_event,
            ("TemporalReasoner", "verify_chain"): self._op_verify_chain,
            ("TemporalReasoner", "edition_chain_fallback"): self._op_edition_chain,
            ("LookupResolver", "resolve_article"): self._op_resolve_article,
            ("LookupResolver", "resolve_venue_date"): self._op_resolve_venue_date,
            ("EvidenceEvaluator", "evaluate_evidence"): self._op_evaluate,
            ("GapDetector", "accept_limitation"): self._op_accept_limitation,
            ("Synthesizer", "render_answer"): self._op_render_answer,
        }.get((step.agent, step.operation))
        if handler is None:
            raise ValueError(f"unknown operation {step.agent}.{step.operation}")
        return handler(state, params)

    # ── helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _kind(qtype: str) -> str:
        return qtype if qtype in KINDS else "lookup"

    def _should_run_optional(self, state: AgentState, step: PlannedStep) -> bool:
        """Optional steps add corroboration; skip them only when it is wasteful.

        A passage search is kept whenever an LLM is available (the adjudicator
        needs context) or when the structured answer is still missing, which is
        almost always - so this mainly prunes genuinely redundant traversals.
        """
        if step.operation == "vector_search":
            return bool(getattr(self.llm, "available", False)) or not state.answer
        if step.operation == "traverse_graph":
            return bool(state.documents) or bool(state.slots.get("link"))
        return True

    def _current_events(self, state: AgentState) -> List[Any]:
        """The candidate event set the counting/comparison agents operate on.

        Prefers the traverser's explicit candidate set (which is *the* set that
        gets counted) over everything accumulated in the blackboard.
        """
        candidate_set = state.slots.get("candidate_set")
        if candidate_set:
            return list(candidate_set)
        return list(state.documents.values())

    # ── handlers: linking + retrieval ──────────────────────────────────
    def _op_link_entities(self, state: AgentState, params: Dict[str, Any]):
        spec = state.spec
        link = self.linker.link(spec)
        state.slots["link"] = link
        pages = self.linker.linked_event_pages(spec, limit=60)
        new = state.add_documents(pages)

        parts: List[str] = []
        if link.get("sport_vertex"):
            parts.append(f"sport->{link['sport_vertex']}")
        if link.get("venue_vertices"):
            parts.append(f"venue->{len(link['venue_vertices'])} vertex/vertices "
                         f"(score {link.get('venue_score')})")
        if link.get("games_vertex"):
            parts.append(f"games->{link['games_vertex']}")
        if link.get("resolved_prev_year"):
            parts.append(f"immediately-previous edition->{link['resolved_prev_year']}")
        detail = "; ".join(parts) or "no graph vertices linked; will fall back to text retrieval"
        observation = {
            "sport": link.get("sport_vertex"), "sport_score": link.get("sport_score"),
            "venue_score": link.get("venue_score"), "games": link.get("games_vertex"),
            "edition_chain": link.get("edition_chain", []),
            "implied_event_pages": len(pages),
        }
        return detail, observation, new

    def _op_traverse_graph(self, state: AgentState, params: Dict[str, Any]):
        spec = state.spec
        mode = params.get("mode", "")
        events: List[Any] = []
        strategy, relations = "", []

        if mode == "sport_games" and spec.sport and spec.year and spec.season:
            sport = resolve_sport(spec.sport, self.kg)
            events = self.kg.events_for(sport, spec.year, spec.season) if sport else []
            strategy = "IN_SPORT + PART_OF (widened from venue)"
            relations = ["IN_SPORT", "PART_OF"]
        elif mode == "sport_wide" and spec.sport:
            sport = resolve_sport(spec.sport, self.kg)
            events = self.kg.events_for_sport(sport) if sport else []
            strategy = "IN_SPORT (sport-wide)"
            relations = ["IN_SPORT"]
        else:
            expanded = self.traverser.expand_candidates(spec, limit=80)
            events = expanded.get("events", [])
            strategy = expanded.get("strategy", "")
            relations = expanded.get("relations", [])

        new = state.add_documents(events)
        state.slots["candidate_set"] = events
        detail = (f"structural retrieval: strategy={strategy or 'none'} -> "
                  f"{len(events)} event pages ({new} new)")
        observation = {"strategy": strategy, "relations": relations,
                       "events": len(events), "new_documents": new,
                       "sample_titles": [getattr(e, "title", "") for e in events[:5]]}
        return detail, observation, new

    def _op_vector_search(self, state: AgentState, params: Dict[str, Any]):
        widening = int(params.get("widening", 0) or 0)
        query = self.searcher.refine_query(state.spec, state.question, widening)
        chunks = self.searcher.hybrid_search(query, 12)
        new = state.add_chunks(chunks)
        detail = (f"text retrieval over {self.index.num_chunks} chunks: "
                  f"query={query!r} (widening={widening}) -> {len(chunks)} passages "
                  f"({new} new documents)")
        observation = {"query": query, "widening": widening,
                       "chunks": len(chunks),
                       "doc_ids": [c.get("doc_id") for c in chunks[:8]]}
        return detail, observation, new

    # ─ handlers: counting + comparison ────────────────────────────────
    def _op_count_threshold(self, state: AgentState, params: Dict[str, Any]):
        spec = state.spec
        events = self._current_events(state)
        result = self.aggregator.run(spec, events)
        state.slots["aggregation"] = result
        detail = (f"filtered {result['candidates']} enumerated events on "
                  f"competitors {result['comparator']} {result['threshold']} -> "
                  f"{result['count']} pass "
                  f"(field coverage {result['field_completeness']:.0%})")
        observation = {"count": result["count"], "candidates": result["candidates"],
                       "threshold": result["threshold"],
                       "comparator": result["comparator"],
                       "field_completeness": result["field_completeness"],
                       "exhaustive": result["exhaustive"],
                       "missing_field": len(result["missing_field"]),
                       "kept_doc_ids": [k["doc_id"] for k in result["kept"]][:20]}
        return detail, observation, 0

    def _op_verify_recount(self, state: AgentState, params: Dict[str, Any]):
        spec = state.spec
        events = self._current_events(state)
        verification = self.aggregator.verify_by_recount(spec, events)
        state.slots["agg_verification"] = verification
        primary = (state.slots.get("aggregation") or {}).get("count")
        agrees = verification.get("recount") == primary
        detail = (f"independent recount -> {verification.get('recount')} "
                  f"(documents with the field: {verification.get('documents_with_field')}); "
                  f"{'matches' if agrees else 'MISMATCH vs'} the primary count")
        observation = dict(verification)
        observation["recount_matches"] = agrees
        return detail, observation, 0

    def _op_arg_extreme(self, state: AgentState, params: Dict[str, Any]):
        spec = state.spec
        events = self._current_events(state)
        result = self.comparator.run(spec, events)
        state.slots["comparison"] = result
        winner = result.get("winner") or {}
        detail = (f"arg{'max' if result.get('direction') == 'max' else 'min'} over "
                  f"{result.get('candidates', 0)} events -> "
                  f"{winner.get('title')!r} ({winner.get('competitors')}); "
                  f"margin over runner-up: {result.get('margin')}")
        observation = {"winner": winner.get("title"),
                       "winner_doc_id": winner.get("doc_id"),
                       "winner_value": winner.get("competitors"),
                       "runner_up": (result.get("runner_up") or {}).get("title"),
                       "margin": result.get("margin"),
                       "candidates": result.get("candidates"),
                       "missing_field": result.get("missing_field"),
                       "unique_winner": result.get("unique_winner")}
        return detail, observation, 0

    def _op_verify_extreme(self, state: AgentState, params: Dict[str, Any]):
        spec = state.spec
        events = self._current_events(state)
        verification = self.comparator.verify(spec, events)
        state.slots["cmp_verification"] = verification
        detail = (f"cross-check of the extreme value "
                  f"({'no violations' if verification.get('verified') else 'VIOLATIONS FOUND'}) "
                  f"across {verification.get('documents_with_field')} documents")
        return detail, dict(verification), 0

    # ─ handlers: temporal ─────────────────────────────────────────────
    def _op_resolve_anchor(self, state: AgentState, params: Dict[str, Any]):
        anchor = self.temporal.resolve_anchor(state.spec)
        state.slots["temporal_anchor"] = anchor
        detail = anchor.get("reasoning", "no anchor resolved")
        return detail, dict(anchor), 0

    def _op_resolve_event(self, state: AgentState, params: Dict[str, Any]):
        anchor = state.slots.get("temporal_anchor") or {}
        year = anchor.get("resolved_year")
        season = anchor.get("season") or state.spec.season or "Summer"
        resolved = self.temporal.resolve_event(state.spec, year, season)
        state.slots["temporal_event"] = resolved
        matched = resolved.get("matched")
        new = state.add_documents([matched]) if matched is not None else 0
        detail = (f"matched event {getattr(matched, 'title', None)!r} "
                  f"(score {resolved.get('score')}) among "
                  f"{resolved.get('num_events_in_games')} events in {year} {season}")
        observation = {"sport": resolved.get("sport"), "score": resolved.get("score"),
                       "matched": getattr(matched, "title", None),
                       "num_events_in_games": resolved.get("num_events_in_games"),
                       "candidates": resolved.get("candidates", [])[:3]}
        return detail, observation, new

    def _op_verify_chain(self, state: AgentState, params: Dict[str, Any]):
        event = (state.slots.get("temporal_event") or {}).get("matched")
        checks = self.temporal.verify(state.spec, event)
        state.slots["temporal_verify"] = checks
        detail = ("; ".join(checks.get("details", []))
                  or "no chain confirmation available")
        return detail, dict(checks), 0

    def _op_edition_chain(self, state: AgentState, params: Dict[str, Any]):
        spec = state.spec
        event = (state.slots.get("temporal_event") or {}).get("matched")
        season = spec.season or "Summer"
        chain = (self.traverser.edition_chain(
            getattr(event, "sport", "") or spec.sport,
            getattr(event, "event_name", ""), season) if event is not None else [])
        years = [n.year for n in chain]
        consistent = bool(spec.before_year and any(y == spec.before_year for y in years))
        checks = {
            "consistent": consistent,
            "edition_years": years,
            "fallback": "corpus-wide edition ordering",
            "details": [
                f"edition chain of '{getattr(event, 'event_name', '')}' spans {years}",
                (f"anchor edition {spec.before_year} present in the chain"
                 if consistent else
                 f"anchor edition {spec.before_year} not present in the chain"),
            ],
        }
        state.slots["temporal_verify"] = checks
        return "; ".join(checks["details"]), checks, 0

    # ─ handlers: lookup + multi-hop ───────────────────────────────────
    def _op_resolve_article(self, state: AgentState, params: Dict[str, Any]):
        result = self.resolver.resolve_article(state.spec)
        state.slots["lookup"] = result
        matched = result.get("matched")
        new = state.add_documents([matched]) if matched is not None else 0
        detail = (f"resolved title {getattr(matched, 'title', None)!r} "
                  f"(score {result.get('score')}); "
                  f"field 'nations' = {result.get('value')!r}")
        observation = {"matched": getattr(matched, "title", None),
                       "score": result.get("score"),
                       "value": result.get("value"),
                       "searched_titles": result.get("searched")}
        return detail, observation, new

    def _op_resolve_venue_date(self, state: AgentState, params: Dict[str, Any]):
        result = self.resolver.resolve_venue_date(state.spec)
        state.slots["venue_date"] = result
        disambiguation = self.resolver.disambiguate(
            state.spec, result.get("candidates") or [])
        state.slots["venue_date_disambiguation"] = disambiguation
        matched = result.get("matched")
        new = state.add_documents([matched]) if matched is not None else 0
        detail = (f"venue->{result.get('venue_resolved')} (score "
                  f"{result.get('venue_score')}); date match "
                  f"{result.get('score')} over {result.get('num_candidates')} "
                  f"candidates -> {getattr(matched, 'title', None)!r}")
        observation = {"matched": getattr(matched, "title", None),
                       "score": result.get("score"),
                       "venue_affinity": result.get("venue_affinity"),
                       "num_candidates": result.get("num_candidates"),
                       "disambiguation": disambiguation,
                       "top_candidates": [c.get("title") for c in
                                          (result.get("candidates") or [])[:3]]}
        return detail, observation, new
# ─ handlers: evidence audit + synthesis ───────────────────────────
    def _evidence_for(self, state: AgentState) -> Dict[str, Any]:
        """The evidence dict the evaluator scores, per question type."""
        if state.kind == "aggregation":
            return state.slots.get("aggregation") or {}
        if state.kind == "superlative":
            return state.slots.get("comparison") or {}
        return {}

    def _op_evaluate(self, state: AgentState, params: Dict[str, Any]):
        kind = state.kind
        link = {"temporal": state.slots.get("temporal_event"),
                "multi_hop": state.slots.get("venue_date"),
                "lookup": state.slots.get("lookup")}.get(kind)
        verification = {"aggregation": state.slots.get("agg_verification"),
                        "superlative": state.slots.get("cmp_verification"),
                        "temporal": state.slots.get("temporal_verify")}.get(kind)
        verdict = self.evaluator.evaluate(state.spec, kind, self._evidence_for(state),
                                          link, verification)
        state.confidence = verdict["confidence"]
        for gap in verdict["gaps"]:
            state.note_gap(gap)
        for gap in list(state.missing_info):
            if gap not in verdict["gaps"] and gap in state.proposed_gaps:
                state.resolve_gap(gap)
        detail = (f"sufficiency audit -> confidence {verdict['confidence']:.2f}, "
                  f"{'sufficient' if verdict['sufficient'] else 'insufficient'}"
                  + (f"; gaps: {'; '.join(verdict['gaps'])}" if verdict["gaps"] else ""))
        observation = {"confidence": verdict["confidence"],
                       "sufficient": verdict["sufficient"],
                       "signals": verdict["signals"],
                       "gaps": verdict["gaps"]}
        return detail, observation, 0

    def _op_accept_limitation(self, state: AgentState, params: Dict[str, Any]):
        gap = params.get("gap") or (state.missing_info[0] if state.missing_info else "")
        if gap:
            state.resolve_gap(gap)
            state.proposed_gaps.append(gap)
        return (f"accepted limitation for gap {gap!r}: no retrieval can supply it",
                {"accepted_gap": gap}, 0)

    def _context_items(self, state: AgentState, citations: List[str]) -> List[Dict[str, Any]]:
        """Passages (and cited infoboxes) handed to the LLM adjudicator."""
        from retrieval.index import infobox_summary

        items = sorted(state.chunks.values(),
                       key=lambda c: -float(c.get("score", 0.0) or 0.0))[:10]
        seen = {c.get("doc_id") for c in items}
        for doc_id in citations:
            if not doc_id or doc_id in seen or self.index is None:
                continue
            doc = self.index.get_doc(doc_id)
            if doc is None:
                continue
            items.append({"doc_id": doc_id, "title": doc.title,
                          "text": infobox_summary(doc)})
            seen.add(doc_id)
        return items

    def _op_render_answer(self, state: AgentState, params: Dict[str, Any]):
        rendered = self.synth.render(state.spec, state.kind, state.slots)
        answer = rendered.answer
        citations = list(rendered.citations)
        source = rendered.source
        adjudication: Dict[str, Any] = {}

        context_items = self._context_items(state, citations)
        if not answer:
            # structured reasoning failed -> last-resort LLM answer over passages
            llm_answer, llm_cites = self.synth.llm_fallback(
                state.question, context_items, state.tokens)
            if llm_answer:
                answer, citations = llm_answer, (llm_cites or citations)
                source = "llm_fallback"
        elif self.llm is not None and getattr(self.llm, "available", False) and context_items:
            _agg = state.slots.get("aggregation") or {}
            refined, changed, payload = refine_answer(
                self.llm, state.question, answer, context_items, state.tokens,
                caller=f"agent.adjudicate.{state.kind}",
                qtype=state.kind,
                exhaustive=bool(_agg.get("exhaustive")))
            adjudication = dict(payload)
            adjudication["changed"] = changed
            adjudication["candidate"] = answer
            if refined:
                answer = refined
            if changed:
                source = "structured+llm_revised"
                state.strategy_changes.append(
                    f"LLM revised the structured answer: {payload.get('reason', '')}")

        if not citations:
            citations = list(state.candidates[:3])
        state.answer = (answer or "").strip()
        state.citations = [c for c in dict.fromkeys(citations) if c]
        state.synth_source = source
        state.adjudication = adjudication
        state.rationale = rendered.rationale
        if adjudication.get("agree"):
            state.confidence = min(0.97, state.confidence + 0.03)

        detail = (f"[{source}] answer={state.answer!r} citing "
                  f"{len(state.citations)} document(s) :: {rendered.rationale}")
        observation = {"answer": state.answer, "source": source,
                       "rationale": rendered.rationale,
                       "citations": state.citations,
                       "adjudication": adjudication}
        return detail, observation, 0