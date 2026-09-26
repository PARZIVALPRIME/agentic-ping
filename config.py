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


def _int_env(name: str, default: int) -> int:
    """Integer env var; unparseable values fall back instead of raising."""
    try:
        return int(_env(name, default=str(default)))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    """Float env var; unparseable values fall back instead of raising."""
    try:
        return float(_env(name, default=str(default)))
    except (TypeError, ValueError):
        return default


#: Default model per provider. Env-overridable so no vendor or model version is
#: baked into the code: swapping the model is a configuration change.
MODEL_DEFAULTS = {
    "groq": _env("DEFAULT_GROQ_MODEL", default="openai/gpt-oss-120b"),
    "ollama": _env("DEFAULT_OLLAMA_MODEL", default="qwen3.5:2b-q4_K_M"),
    "openai": _env("DEFAULT_OPENAI_MODEL", default="gpt-4o-mini"),
    "gemini": _env("DEFAULT_GEMINI_MODEL", default="gemini-2.5-flash"),
    "azure": _env("DEFAULT_AZURE_MODEL", default="gpt-4o-mini"),
}


@dataclass
class DomainConfig:
    """What the corpus is about - named in prompts, never hardcoded.

    Nothing in the pipeline architecture depends on the subject matter: the
    question ("when is the graph worth its cost?") is the same for patents,
    medicine or sport. Prompts therefore describe the corpus from configuration,
    and the numeric guards (which years are plausible) come from here too, so a
    new corpus is onboarded by editing environment variables instead of code.
    """

    #: How prompts refer to the corpus, e.g. "a corpus of sports-event articles".
    corpus_label: str = field(default_factory=lambda: _env(
        "CORPUS_LABEL", default="a corpus of sports-event articles"))
    #: Singular noun for the things the graph is about ("Olympic event").
    entity_label: str = field(default_factory=lambda: _env(
        "CORPUS_ENTITY_LABEL", default="Olympic event"))
    #: The name of the graph in the external store (TigerGraph etc.).
    graph_name: str = field(default_factory=lambda: _env(
        "GRAPH_NAME", "TG_GRAPHNAME", default="OlympicsKG"))
    #: Plausible range for a year mentioned in a question, used to reject a
    #: hallucinated year instead of letting it into a slot.
    year_min: int = field(default_factory=lambda: _int_env("CORPUS_YEAR_MIN", 1896))
    year_max: int = field(default_factory=lambda: _int_env("CORPUS_YEAR_MAX", 2035))


@dataclass
class TigerGraphConfig:
    """TigerGraph connection configuration (Savanna or Community Edition)."""

    host: str = field(default_factory=lambda: _env("TG_HOST", "TGRAPH_HOST",
                                                   default="http://localhost"))
    graphname: str = field(default_factory=lambda: _env("TG_GRAPHNAME", "TGRAPH_GRAPH_NAME",
                                                        "GRAPH_NAME", default="OlympicsKG"))
    username: str = field(default_factory=lambda: _env("TG_USERNAME", "TGRAPH_USERNAME",
                                                       default="tigergraph"))
    password: str = field(default_factory=lambda: _env("TG_PASSWORD", "TGRAPH_PASSWORD",
                                                       default="tigergraph"))
    restpp_port: str = field(default_factory=lambda: _env("TG_RESTPP_PORT", default="9000"))
    gs_port: str = field(default_factory=lambda: _env("TG_GS_PORT", default="14240"))
    token: str = field(default_factory=lambda: _env("TG_TOKEN", "TGRAPH_TOKEN"))
    enabled: bool = field(default_factory=lambda: _env("TG_ENABLED", default="").lower()
                          in ("1", "true", "yes"))
    max_rows: int = field(default_factory=lambda: _int_env("TG_MAX_ROWS", 100000))
    timeout: float = field(default_factory=lambda: _float_env("TG_TIMEOUT", 30.0))
    retries: int = field(default_factory=lambda: _int_env("TG_RETRIES", 2))
    verbose: bool = field(default_factory=lambda: _env("TG_VERBOSE", default="").lower()
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
    #: Sampling temperature for the *agentic* models. Deliberately > 0 by
    #: default: a temperature of 0 makes every investigation take the same path,
    #: so the system cannot adapt its strategy - and adaptation is exactly what
    #: it is graded on. The judge/eval model stays greedy (see eval_temperature)
    #: so that scoring is reproducible.
    temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE",
                                                                 default="0.3")))
    #: Temperature for evaluation/adjudication calls: 0 keeps *scoring* stable
    #: even while the agent explores.
    eval_temperature: float = field(default_factory=lambda: float(
        _env("LLM_EVAL_TEMPERATURE", default="0.0")))
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
    # Local model server (Ollama). No key and no quota are involved: the api_key
    # below is a placeholder the OpenAI client insists on, and every request goes
    # to localhost, so a run on this provider spends no provider tokens at all.
    ollama_base_url: str = field(default_factory=lambda: _env(
        "OLLAMA_BASE_URL", default="http://localhost:11434/v1"))
    ollama_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OLLAMA_API_KEY"))

    def __post_init__(self) -> None:
        defaults = MODEL_DEFAULTS
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
        # A local model has no quota to conserve, but qwen3.5 spends part of its
        # budget on thinking, so an uncapped completion would ramble for minutes.
        if self.provider == "ollama" and self.max_tokens == 0:
            self.max_tokens = 700

    @property
    def api_key(self) -> Optional[str]:
        return {
            "openai": self.openai_api_key,
            "gemini": self.google_api_key,
            "azure": self.azure_api_key,
            "groq": self.groq_api_key,
            # Ollama ignores the key, but the OpenAI client rejects an empty one.
            "ollama": self.ollama_api_key or "ollama",
        }.get(self.provider)

    @property
    def is_local(self) -> bool:
        """True when the model server is on this machine (no quota, no cost)."""
        return self.provider == "ollama"


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
    #: How a question becomes a QuerySpec: "semantic" (the LLM extracts the
    #: slots, the template parser validates and fills - the default) or "rules"
    #: (templates only, for deterministic and ablation runs). Read through
    #: ``reasoning.query_parser.parse_mode()`` rather than here, because that
    #: module also owns the ABLATE_CLASSIFIER switch.
    parse_mode: str = field(default_factory=lambda: _env("PARSE_MODE", default="semantic"))


@dataclass
class BenchmarkConfig:
    """Benchmark runner configuration."""

    public_questions_path: str = field(
        default_factory=lambda: _env("PUBLIC_QUESTIONS_PATH", default=PUBLIC_DEFAULT))
    hidden_questions_path: str = field(
        default_factory=lambda: _env("HIDDEN_QUESTIONS_PATH", default=HIDDEN_DEFAULT))
    corpus_path: str = field(default_factory=lambda: _env("CORPUS_PATH", default=CORPUS_DEFAULT))
    # Where the corpus-built knowledge graph is cached. Explicit here (rather than
    # relying on kg.builder's default) because a TigerGraph run builds the same
    # mirror and must reuse the same cache file.
    kg_cache_path: str = field(
        default_factory=lambda: _env("KG_CACHE_PATH",
                                     default="results/knowledge_graph.json"))
    results_dir: str = field(default_factory=lambda: _env("RESULTS_DIR", default="results"))
    top_k: int = field(default_factory=lambda: int(_env("RETRIEVAL_TOP_K", default="10")))
    rag_top_k: int = field(default_factory=lambda: int(_env("RAG_TOP_K", default="5")))
    num_hops: int = field(default_factory=lambda: int(_env("RETRIEVAL_NUM_HOPS", default="2")))
    vector_backend: str = field(default_factory=lambda: _env("VECTOR_BACKEND", default="auto"))
    rebuild_kg: bool = field(default_factory=lambda: _env("REBUILD_KG", default="").lower()
                             in ("1", "true", "yes"))


@dataclass
class Config:
    """Master configuration container."""

    tg: TigerGraphConfig = field(default_factory=TigerGraphConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)

    def describe(self) -> dict:
        """Summary of the active configuration (safe to log / embed in results)."""
        return {
            "llm_provider": self.llm.provider,
            "llm_available": bool(self.llm.provider and self.llm.provider != "none"
                                  and self.llm.api_key),
            "chat_model": self.llm.chat_model or None,
            "eval_model": self.llm.eval_model or None,
            "max_tokens": self.llm.max_tokens,
            "temperature": self.llm.temperature,
            "eval_temperature": self.llm.eval_temperature,
            "reasoning_effort": self.llm.reasoning_effort or None,
            "corpus_label": self.domain.corpus_label,
            "graph_name": self.domain.graph_name,
            "plausible_years": [self.domain.year_min, self.domain.year_max],
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
