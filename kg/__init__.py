"""Knowledge-graph package: model, builder, backend selection and persistence."""

from .backend import BackendInfo, describe_backend, open_graph
from .model import (AthleteNode, EventNode, GraphEdge, KnowledgeGraph,
                    rank_events_by_terms)

__all__ = [
    "KnowledgeGraph", "EventNode", "AthleteNode", "GraphEdge",
    "rank_events_by_terms",
    "open_graph", "describe_backend", "BackendInfo",
]