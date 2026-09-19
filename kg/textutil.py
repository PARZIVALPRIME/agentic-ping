"""Text normalization and fuzzy-matching utilities (dependency-free).

All matching in the system funnels through these helpers so that retrieval,
entity linking and evaluation behave consistently.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable, List, Sequence, Set

_WS_RE = re.compile(r"\s+")
_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")
_EN_DASHES = {"\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2010": "-", "\u2011": "-"}


def repair_mojibake(text: str) -> str:
    """Undo accidental latin-1 <-> utf-8 double encoding when detectable.

    The hackathon corpus is clean UTF-8, but user-provided files may not be.
    The function is conservative: it only returns the repaired string when the
    round-trip succeeds and the repaired value is plausibly "more correct".
    """
    if not text:
        return text
    try:
        repaired = text.encode("latin-1", errors="strict").decode("utf-8", errors="strict")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    # Only accept the repair if it removes the classic mojibake markers.
    markers = ("Ã", "Â", "â€", "Î", "Ð")
    if any(m in text for m in markers) and not any(m in repaired for m in markers):
        return repaired
    return text


def fold_unicode(text: str) -> str:
    """Normalize dashes and strip combining accents (é -> e, ı stays)."""
    for src, dst in _EN_DASHES.items():
        text = text.replace(src, dst)
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize(text: str) -> str:
    """Lowercase, fold accents/dashes, strip punctuation, collapse whitespace."""
    if not text:
        return ""
    text = fold_unicode(text).lower()
    text = _ALNUM_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def tokens(text: str) -> List[str]:
    n = normalize(text)
    return n.split() if n else []


def token_set(text: str) -> Set[str]:
    return set(tokens(text))


def jaccard(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def sequence_ratio(a: str, b: str) -> float:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def containment(a: str, b: str) -> float:
    """Fraction of the shorter token set covered by the longer one."""
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / min(len(ta), len(tb))


def similarity(a: str, b: str) -> float:
    """Blended fuzzy similarity in [0, 1]."""
    return max(
        0.45 * jaccard(a, b) + 0.55 * sequence_ratio(a, b),
        containment(a, b) * 0.9,
    )


def best_match(query: str, candidates: Iterable[str]) -> tuple[str, float] | tuple[None, float]:
    """Return the best fuzzy match for ``query`` among ``candidates``."""
    best: str | None = None
    best_score = 0.0
    for cand in candidates:
        score = similarity(query, cand)
        if score > best_score:
            best, best_score = cand, score
    return best, best_score


_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    # common abbreviations
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def months_in(text: str) -> Set[str]:
    """Return the canonical month names mentioned in ``text``."""
    return {m for m in tokens(text) if m in _MONTHS}


def days_in(text: str) -> Set[str]:
    """Return day-of-month numbers (1-31) mentioned in ``text``."""
    found = set()
    for tok in tokens(text):
        if tok.isdigit():
            val = int(tok)
            if 1 <= val <= 31:
                found.add(str(val))
    return found


def ordinal_words_present(text: str) -> Set[str]:
    return {t for t in tokens(text) if t in {"first", "second", "third", "fourth"}}
