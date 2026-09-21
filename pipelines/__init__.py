"""Pipelines package: RAG, GraphRAG, Agentic GraphRAG, Router and ablations."""

from .ablations import ABLATIONS, build_ablations
from .agentic_pipeline import AgenticPipeline
from .base import PipelineResult, StepRecord, Timer, finalise_result
from .extractive import ExtractiveAnswerer, ExtractResult
from .graphrag_pipeline import GraphRagPipeline
from .rag_pipeline import RagPipeline
from .router_pipeline import ROUTING_TABLE, RouterPipeline

__all__ = [
    "PipelineResult",
    "StepRecord",
    "Timer",
    "finalise_result",
    "ExtractiveAnswerer",
    "ExtractResult",
    "RagPipeline",
    "GraphRagPipeline",
    "AgenticPipeline",
    "RouterPipeline",
    "ROUTING_TABLE",
    "ABLATIONS",
    "build_ablations",
    "build_pipelines",
]


def build_pipelines(index, llm=None, config=None, with_router: bool = True,
                    with_ablations: bool = False):
    """Instantiate the pipelines with consistent, comparable settings.

    ``with_ablations`` is opt-in because the ablation variants are a *study*,
    not part of the headline three-way comparison: they triple runtime and
    would clutter the main dashboard chart.
    """
    benchmark = getattr(config, "benchmark", None)
    agent_cfg = getattr(config, "agent", None)
    rag_k = getattr(benchmark, "rag_top_k", 5)
    top_k = getattr(benchmark, "top_k", 10)
    hops = getattr(benchmark, "num_hops", 2)
    agent_settings = ({
        "max_steps": agent_cfg.max_steps,
        "max_replans": agent_cfg.max_replans,
        "confidence_threshold": agent_cfg.confidence_threshold,
        "stale_step_limit": agent_cfg.stale_step_limit,
        "max_widen_attempts": agent_cfg.max_widen_attempts,
        "agent_mode": agent_cfg.mode,
        "react_max_steps": agent_cfg.react_max_steps,
        "react_retries": agent_cfg.react_retries,
        "react_model": agent_cfg.react_model,
    } if agent_cfg is not None else None)

    pipes = [
        RagPipeline(index, llm=llm, top_k=rag_k),
        GraphRagPipeline(index, llm=llm, top_k=top_k, num_hops=hops),
        AgenticPipeline(index, llm=llm, cfg=agent_settings),
    ]

    if with_ablations:
        pipes.extend(build_ablations(index, llm=llm, cfg=agent_settings))

    if with_router:
        # The router dispatches to the three core pipelines, so it must be
        # constructed from them (and must not route to itself).
        pipes.append(RouterPipeline(index, pipes[:3], llm=llm))

    return pipes


