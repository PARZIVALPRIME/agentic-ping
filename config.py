"""
Agentic GraphRAG Hackathon - Central Configuration

Everything is environment-driven with safe defaults so the benchmark runs with
zero configuration (deterministic offline mode). See `.env.example`.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv is optional
    pass

CORPUS_DEFAULT = "corpus-20260919T043338Z-1-001/corpus/corpus.jsonl"
PUBLIC_DEFAULT = "questions-20260919T043312Z-1-001/questions/eval_public.jsonl"
HIDDEN_DEFAULT = "questions-20260919T043312Z-1-001/questions/eval_hidden.jsonl"


def _env(*names: str, default: str = "") -> str:
    """First non-empty value among ``names`` (supports legacy aliases)."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


@dataclass
class TigerGraphConfig:
    """TigerGraph connection configuration (Savanna or Community Edition)."""

    host: str = field(default_factory=lambda: _env("TG_HOST", "TGRAPH_HOST",
                                                   default="http://localhost"))
    graphname: str = field(default_factory=lambda: _env("TG_GRAPHNAME", "TGRAPH_GRAPH_NAME",
                                                        default="OlympicsKG"))
    username: str = field(default_factory=lambda: _env("TG_USERNAME", "TGRAPH_USERNAME",
                                                       default="tigergraph"))
    password: str = field(default_factory=lambda: _env("TG_PASSWORD", "TGRAPH_PASSWORD",
                                                       default="tigergraph"))
    restpp_port: str = field(default_factory=lambda: _env("TG_RESTPP_PORT", default="9000"))
    gs_port: str = field(default_factory=lambda: _env("TG_GS_PORT", default="14240"))
    token: str = field(default_factory=lambda: _env("TG_TOKEN", "TGRAPH_TOKEN"))
    enabled: bool = field(default_factory=lambda: _env("TG_ENABLED", default="").lower()
                          in ("1", "true", "yes"))


@dataclass
class LLMConfig:
    """LLM provider configuration."""

    provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", default="none"))
    completion_model: str = field(default_factory=lambda: _env("COMPLETION_MODEL", default=""))
    chat_model: str = field(default_factory=lambda: _env("CHAT_MODEL", default=""))
    fast_model: str = field(default_factory=lambda: _env("FAST_MODEL", default=""))
    eval_model: str = field(default_factory=lambda: _env("EVAL_MODEL", default=""))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", default=""))
    temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE",
                                                                 default="0.0")))
    max_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_TOKENS", default="0")))
    reasoning_effort: str = field(default_factory=lambda: _env("LLM_REASONING_EFFORT"))

    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    google_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    azure_api_key: Optional[str] = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY"))
    azure_endpoint: Optional[str] = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT"))
    azure_deployment: Optional[str] = field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT"))
    groq_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    groq_base_url: str = field(default_factory=lambda: _env("GROQ_BASE_URL",
                                                            default="https://api.groq.com/openai/v1"))

    def __post_init__(self) -> None:
        defaults = {
            "groq": "openai/gpt-oss-120b",
            "openai": "gpt-4o-mini",
            "gemini": "gemini-2.5-flash",
            "azure": "gpt-4o-mini",
        }
        fallback = defaults.get(self.provider, "")
        self.chat_model = self.chat_model or fallback
        self.completion_model = self.completion_model or self.chat_model
        self.fast_model = self.fast_model or self.chat_model
        self.eval_model = self.eval_model or self.chat_model
        self.embedding_model = self.embedding_model or "all-MiniLM-L6-v2"
        # reasoning-capable models emit reasoning tokens; cap them by default
        if self.provider == "groq" and self.max_tokens == 0:
            self.max_tokens = 900
            self.reasoning_effort = self.reasoning_effort or "low"

    @property
    def api_key(self) -> Optional[str]:
        return {
            "openai": self.openai_api_key,
            "gemini": self.google_api_key,
            "azure": self.azure_api_key,
            "groq": self.groq_api_key,
        }.get(self.provider)


@dataclass
class AgentConfig:
    """Agentic pipeline configuration (planner + stopping criteria)."""

    max_steps: int = field(default_factory=lambda: int(_env("AGENT_MAX_STEPS", default="15")))
    max_replans: int = field(default_factory=lambda: int(_env("AGENT_MAX_REPLANS", default="3")))
    confidence_threshold: float = field(
        default_factory=lambda: float(_env("AGENT_CONFIDENCE_THRESHOLD", default="0.9")))
    stale_step_limit: int = field(
        default_factory=lambda: int(_env("AGENT_STALE_STEP_LIMIT", default="2")))
    max_widen_attempts: int = field(
        default_factory=lambda: int(_env("AGENT_MAX_WIDEN_ATTEMPTS", default="2")))
    # "react"    - the LLM drives every step through tool calls
    # "hybrid"   - the LLM drives; deterministic solvers only rescue empty answers
    # "plan"     - deterministic planner/executor only (no tool-calling loop)
    mode: str = field(default_factory=lambda: _env("AGENT_MODE", default="hybrid"))
    react_max_steps: int = field(
        default_factory=lambda: int(_env("AGENT_REACT_MAX_STEPS", default="6")))
    react_retries: int = field(
        default_factory=lambda: int(_env("AGENT_REACT_RETRIES", default="2")))
    react_model: str = field(default_factory=lambda: _env("AGENT_REACT_MODEL", default=""))


@dataclass
class BenchmarkConfig:
    """Benchmark runner configuration."""

    public_questions_path: str = field(
        default_factory=lambda: _env("PUBLIC_QUESTIONS_PATH", default=PUBLIC_DEFAULT))
    hidden_questions_path: str = field(
        default_factory=lambda: _env("HIDDEN_QUESTIONS_PATH", default=HIDDEN_DEFAULT))
    corpus_path: str = field(default_factory=lambda: _env("CORPUS_PATH", default=CORPUS_DEFAULT))
    results_dir: str = field(default_factory=lambda: _env("RESULTS_DIR", default="results"))
    top_k: int = field(default_factory=lambda: int(_env("RETRIEVAL_TOP_K", default="10")))
    rag_top_k: int = field(default_factory=lambda: int(_env("RAG_TOP_K", default="5")))
    num_hops: int = field(default_factory=lambda: int(_env("RETRIEVAL_NUM_HOPS", default="2")))
    vector_backend: str = field(default_factory=lambda: _env("VECTOR_BACKEND", default="auto"))
    kg_cache_path: str = field(
        default_factory=lambda: _env("KG_CACHE_PATH", default="results/knowledge_graph.json"))
    rebuild_kg: bool = field(default_factory=lambda: _env("REBUILD_KG", default="").lower()
                             in ("1", "true", "yes"))


@dataclass
class Config:
    """Master configuration container."""

    tg: TigerGraphConfig = field(default_factory=TigerGraphConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def describe(self) -> dict:
        """Summary of the active configuration (safe to log / embed in results)."""
        return {
            "llm_provider": self.llm.provider,
            "llm_available": bool(self.llm.provider and self.llm.provider != "none"
                                  and self.llm.api_key),
            "chat_model": self.llm.chat_model or None,
            "eval_model": self.llm.eval_model or None,
            "max_tokens": self.llm.max_tokens,
            "reasoning_effort": self.llm.reasoning_effort or None,
            "tigergraph_enabled": self.tg.enabled,
            "tigergraph_host": self.tg.host if self.tg.enabled else None,
            "vector_backend": self.benchmark.vector_backend,
            "rag_top_k": self.benchmark.rag_top_k,
            "num_hops": self.benchmark.num_hops,
            "agent": {
                "mode": self.agent.mode,
                "max_steps": self.agent.max_steps,
                "confidence_threshold": self.agent.confidence_threshold,
                "stale_step_limit": self.agent.stale_step_limit,
            },
        }


# Singleton
config = Config()
