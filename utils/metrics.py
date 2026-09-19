"""Token accounting.

Uses ``tiktoken`` when it is installed (exact BPE counts) and otherwise falls
back to a deterministic approximation. Every pipeline reports its token cost
through this module so the cost comparison between pipelines is consistent.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

try:  # optional dependency
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - offline environments
    tiktoken = None
    _ENCODER = None

_WORD_RE = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Return the token count of ``text``."""
    if not text:
        return 0
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text))
        except Exception:
            pass
    # Approximation: ~4 characters per token, with word/punctuation structure.
    words = _WORD_RE.findall(text)
    chars = len(text)
    return max(len(words), int(round(chars / 4.0)))


class TokenCounter:
    """Accumulates tokens attributed to named operations."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.per_operation: List[dict] = []
        self.exact = _ENCODER is not None

    def add(self, operation: str, input_tokens: int = 0, output_tokens: int = 0,
            detail: str = "") -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.per_operation.append({
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "detail": detail,
        })

    def add_text(self, operation: str, input_text: str = "", output_text: str = "",
                 detail: str = "") -> None:
        self.add(operation, count_tokens(input_text), count_tokens(output_text), detail)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "exact_tokenizer": self.exact,
            "per_operation": self.per_operation,
        }


def count_many(texts: Iterable[str]) -> int:
    return sum(count_tokens(t) for t in texts)