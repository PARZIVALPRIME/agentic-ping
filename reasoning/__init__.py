"""Reasoning layer: query understanding and deterministic structured solvers."""

from .query_parser import QuerySpec, classify, parse_question
from .solvers import SolveResult, StructuredSolver, resolve_sport, resolve_venue

__all__ = [
    "QuerySpec",
    "classify",
    "parse_question",
    "SolveResult",
    "StructuredSolver",
    "resolve_sport",
    "resolve_venue",
]