"""Pipelines package: RAG, GraphRAG and Agentic GraphRAG."""

from .agentic_pipeline import AgenticPipeline
from .base import PipelineResult, StepRecord, Timer, finalise_result
from .extractive import ExtractiveAnswerer, ExtractResult
from .graphrag_pipeline import GraphRagPipeline
from .rag_pipeline import RagPipeline

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
    "build_pipelines",
    "RAG_ABLATION_NAME",
]


RAG_ABLATION_NAME = "RAG (vector only)"


def build_pipelines(index, llm=None, config=None, include_ablation: bool = False):
    """Instantiate the pipelines with consistent, comparable settings.

    With ``include_ablation`` a fourth arm is added: the same RAG pipeline with
    the structure-aware retrieval step removed (``RAG_ABLATION_NAME``). It shares
    the extractor, the adjudicator and the prompt with the main RAG arm, so
    differencing the two isolates the structured layer's contribution rather
    than attributing it to the graph.
    """
    benchmark = getattr(config, "benchmark", None)
    agent_cfg = getattr(config, "agent", None)
    rag_k = getattr(benchmark, "rag_top_k", 5)
    top_k = getattr(benchmark, "top_k", 10)
    hops = getattr(benchmark, "num_hops", 2)
    if include_ablation is False and benchmark is not None:
        include_ablation = bool(getattr(benchmark, "ablation_rag", False))
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
        "structured_preference": getattr(agent_cfg, "structured_preference", True),
    } if agent_cfg is not None else None)

    pipelines = [
        RagPipeline(index, llm=llm, top_k=rag_k),
        GraphRagPipeline(index, llm=llm, top_k=top_k, num_hops=hops),
        AgenticPipeline(index, llm=llm, cfg=agent_settings),
    ]
    if include_ablation:
        pipelines.append(RagPipeline(index, llm=llm, top_k=rag_k,
                                     use_structured=False,
                                     name=RAG_ABLATION_NAME))
    return pipelines

