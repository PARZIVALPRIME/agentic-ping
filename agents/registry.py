"""Specialized agent registry: formal specifications, typed responsibilities,
inputs, outputs, tools, and success/failure criteria.

Every specialized agent in the multi-agent architecture is formally catalogued
here. This contract is used by the orchestrator, the ReAct loop, the API layer,
and evaluation suites to guarantee auditable multi-agent governance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentRole(str, Enum):
    NAVIGATOR = "graph_navigation"
    TEMPORAL = "temporal_auditing"
    CONFLICT = "conflict_adjudication"
    RETRIEVAL = "vector_search"
    SYNTHESIS = "evidence_synthesis"
    AGGREGATION = "set_aggregation"
    COMPARISON = "extreme_comparison"
    LOOKUP = "lookup_resolution"
    EVALUATION = "evidence_evaluation"
    ROUTING = "capability_routing"


@dataclass(frozen=True)
class AgentSpecification:
    """Formal contract defining a specialized agent's role and boundaries."""

    name: str
    role: AgentRole
    description: str
    tools: List[str]
    inputs: List[str]
    outputs: List[str]
    success_criteria: str
    failure_criteria: str
    telemetry_fields: List[str] = field(default_factory=lambda: [
        "latency_ms", "tokens", "status", "documents_touched"
    ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role.value,
            "description": self.description,
            "tools": list(self.tools),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "success_criteria": self.success_criteria,
            "failure_criteria": self.failure_criteria,
            "telemetry_fields": list(self.telemetry_fields),
        }


AGENT_REGISTRY: Dict[str, AgentSpecification] = {
    "GraphNavigatorAgent": AgentSpecification(
        name="GraphNavigatorAgent",
        role=AgentRole.NAVIGATOR,
        description="Traverses TigerGraph vertices and edges, discovers seed events, and performs multi-hop neighbourhood expansions.",
        tools=["search_events", "traverse_graph", "tg_filter_events", "tg_neighbours"],
        inputs=["sport", "venue", "year", "season", "relations", "seed_doc_id"],
        outputs=["event_nodes", "edge_paths", "candidate_doc_ids"],
        success_criteria="Discovers >= 1 valid graph vertices matching structural constraints with graph provenance.",
        failure_criteria="Zero graph vertices discovered or network partition without local fallback.",
    ),
    "TemporalAuditorAgent": AgentSpecification(
        name="TemporalAuditorAgent",
        role=AgentRole.TEMPORAL,
        description="Resolves Olympic chronologies, edition progressions (PREV/NEXT edges), and anchor year constraints.",
        tools=["get_event_values", "get_event_details", "edition_chain", "resolve_anchor"],
        inputs=["anchor_year", "direction (before/after)", "season", "event_title"],
        outputs=["resolved_year", "chronological_sequence", "edition_years", "anchor_event"],
        success_criteria="Identifies exact chronological predecessor/successor edition and verifies edition chain continuity.",
        failure_criteria="Anchor year missing, discontinuous edition chain, or unresolved temporal relation.",
    ),
    "ConflictAdjudicatorAgent": AgentSpecification(
        name="ConflictAdjudicatorAgent",
        role=AgentRole.CONFLICT,
        description="Adjudicates conflicting, evolving, or contested historical facts using a 4-tier precedence matrix.",
        tools=["detect_conflicts"],
        inputs=["conflicting_statements", "source_documents", "temporal_stamps", "authority_markers"],
        outputs=["adjudicated_value", "applied_rule", "residual_uncertainty", "superseded_values"],
        success_criteria="Applies deterministic 4-tier rule (Authority > Recency > Succession > Majority) with quantified uncertainty.",
        failure_criteria="Undetected factual contradiction or ambiguous unadjudicated values.",
    ),
    "VectorSearcherAgent": AgentSpecification(
        name="VectorSearcherAgent",
        role=AgentRole.RETRIEVAL,
        description="Performs hybrid sparse TF-IDF and dense semantic passage retrieval over unstructured text corpus.",
        tools=["search_passages", "vector_search", "refine_query"],
        inputs=["natural_language_query", "widening_level", "top_k"],
        outputs=["ranked_chunks", "similarity_scores", "doc_ids"],
        success_criteria="Returns top-k high-scoring text passages containing relevant entity keywords.",
        failure_criteria="Zero passages retrieved or similarity score below relevance threshold.",
    ),
    "AggregatorAgent": AgentSpecification(
        name="AggregatorAgent",
        role=AgentRole.AGGREGATION,
        description="Executes scalable in-database and set-wide candidate filtering and counting over candidate events.",
        tools=["count_threshold", "verify_by_recount", "tg_aggregate_stats"],
        inputs=["candidate_events", "field_name", "comparator", "threshold_value"],
        outputs=["exact_count", "kept_doc_ids", "field_completeness_ratio", "recount_match"],
        success_criteria="Exhaustive enumeration of all candidates with independent recount verification.",
        failure_criteria="Incomplete candidate enumeration, missing field data > 50%, or recount mismatch.",
    ),
    "ComparatorAgent": AgentSpecification(
        name="ComparatorAgent",
        role=AgentRole.COMPARISON,
        description="Extracts extreme values (argmax/argmin), computes margins of victory, and verifies total ordering.",
        tools=["arg_extreme", "verify_extreme"],
        inputs=["candidate_events", "target_field", "direction (max/min)"],
        outputs=["winner_entity", "extreme_value", "runner_up_entity", "margin", "unique_winner"],
        success_criteria="Identifies unique extreme value with positive margin over runner-up and zero violations.",
        failure_criteria="Empty candidate set, non-unique winner without tie-break, or ranking violation.",
    ),
    "LookupResolverAgent": AgentSpecification(
        name="LookupResolverAgent",
        role=AgentRole.LOOKUP,
        description="Resolves single-fact article lookups, venue-date affiliations, and specific entity attributes.",
        tools=["resolve_article", "resolve_venue_date", "disambiguate"],
        inputs=["target_title", "venue_name", "date_text", "attribute_name"],
        outputs=["matched_document", "attribute_value", "affinity_score"],
        success_criteria="Direct match to single canonical entity with affinity score >= 0.85.",
        failure_criteria="Multiple ambiguous candidates without resolution or entity not in corpus.",
    ),
    "EvidenceSynthesizerAgent": AgentSpecification(
        name="EvidenceSynthesizerAgent",
        role=AgentRole.SYNTHESIS,
        description="Validates answer schema, strictly grounds citations against observed documents, and compiles final response.",
        tools=["submit_answer", "render_answer", "validate_answer"],
        inputs=["accumulated_evidence", "candidate_answers", "query_spec", "state_documents"],
        outputs=["final_answer", "grounded_citations", "confidence", "rationale"],
        success_criteria="Synthesizes concise, factually correct answer citing only verified document IDs with confidence >= 0.90.",
        failure_criteria="Hallucinated citation IDs, empty answer, or ungrounded speculative claims.",
    ),
    "EvidenceEvaluatorAgent": AgentSpecification(
        name="EvidenceEvaluatorAgent",
        role=AgentRole.EVALUATION,
        description="Audits accumulated blackboard evidence against query requirements and detects residual information gaps.",
        tools=["evaluate_evidence", "detect_gaps"],
        inputs=["query_spec", "question_kind", "blackboard_slots", "verification_results"],
        outputs=["confidence_score", "information_gaps", "sufficiency_verdict"],
        success_criteria="Correctly detects whether current evidence is sufficient to answer or requires replanning.",
        failure_criteria="False positive sufficiency verdict on incomplete evidence.",
    ),
    "RouterAgent": AgentSpecification(
        name="RouterAgent",
        role=AgentRole.ROUTING,
        description="Analyzes question structural capabilities and dispatches to the most cost-effective pipeline with adaptive escalation.",
        tools=["classify_question", "analyze_capabilities", "dispatch_target"],
        inputs=["question_text"],
        outputs=["qtype", "classification_confidence", "dispatched_pipeline", "required_capabilities"],
        success_criteria="Dispatches single-fact queries cheaply and escalates complex multi-hop queries to agentic pipeline.",
        failure_criteria="Misrouting complex set queries to single-shot retrieval without escalation.",
    ),
}


def get_agent_spec(name: str) -> Optional[AgentSpecification]:
    """Retrieve the formal specification for an agent persona."""
    return AGENT_REGISTRY.get(name)


def list_agent_specs() -> List[Dict[str, Any]]:
    """List all registered agent specifications as dictionaries."""
    return [spec.to_dict() for spec in AGENT_REGISTRY.values()]
