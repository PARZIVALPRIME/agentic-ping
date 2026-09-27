"""Security, Input Sanitization, and Secret Redaction Layer.

Guards against:
  1. Prompt injection & jailbreaks in user questions
  2. GSQL / SQL parameter injection attacks
  3. Path traversal in file/corpus/cache paths
  4. Secret leakage in logs, traces, and metrics
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

# Common prompt injection signatures to neutralize
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+(instructions|prompts|directions)", re.IGNORECASE),
    re.compile(r"(system\s+prompt|system\s+instructions)\s*[:=]", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(in\s+developer\s+mode|unrestricted|jailbroken)", re.IGNORECASE),
    re.compile(r"(print|reveal|expose|output|show)\s+(your\s+)?(api[_\s-]?key|secret|token|password)", re.IGNORECASE),
    re.compile(r"<\s*script\s*>", re.IGNORECASE),
    re.compile(r"\[\s*system\s*\]", re.IGNORECASE),
]

# Secret patterns for redaction
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}", re.IGNORECASE),               # OpenAI key
    re.compile(r"gsk_[A-Za-z0-9_-]{20,}", re.IGNORECASE),              # Groq key
    re.compile(r"AIza[0-9A-Za-z-_]{35}", re.IGNORECASE),               # Google API key
    re.compile(r"bearer\s+[A-Za-z0-9._~+/-]{20,}", re.IGNORECASE),     # Bearer token
    re.compile(r"password[\"']?\s*[:=]\s*[\"']?([^\"'\s,]+)", re.IGNORECASE), # Passwords
    re.compile(r"secret[\"']?\s*[:=]\s*[\"']?([^\"'\s,]+)", re.IGNORECASE),   # Secrets
]


class InputSanitizer:
    """Sanitizes user input questions and parameters before pipeline processing."""

    @staticmethod
    def sanitize_question(text: str) -> Tuple[str, bool]:
        """Strip dangerous characters and detect prompt injection patterns.

        Returns: (sanitized_text, was_flagged)
        """
        if not text:
            return "", False

        cleaned = str(text)

        # 1. Remove zero-width spaces and bidirectional override characters
        cleaned = re.sub(r"[​-‍﻿‪-‮]", "", cleaned)

        # 2. Check for prompt injection signatures
        flagged = False
        for pat in _INJECTION_PATTERNS:
            if pat.search(cleaned):
                flagged = True
                # Neutralize injection text by replacing with safe placeholder
                cleaned = pat.sub("[sanitized_injection_attempt]", cleaned)

        # 3. Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned, flagged

    @staticmethod
    def sanitize_gsql_param(val: Any) -> Any:
        """Sanitize string values destined for GSQL parameter maps."""
        if not isinstance(val, str):
            return val
        # Strip potential GSQL statement separators, comment markers, and null bytes
        cleaned = val.replace(";", "").replace("\x00", "").replace("--", "")
        # Prevent escaping quotes maliciously
        cleaned = cleaned.replace("'", "''")
        return cleaned.strip()

    @staticmethod
    def validate_safe_path(path: str, base_dir: Optional[str] = None) -> str:
        """Prevent path traversal vulnerabilities (e.g. ../../etc/passwd)."""
        clean_path = os.path.normpath(str(path))
        # Disallow null bytes
        if "\x00" in clean_path:
            raise ValueError(f"Path contains forbidden null bytes: {path}")

        # If base directory is specified, verify resolved path stays within boundary
        if base_dir:
            abs_base = os.path.abspath(base_dir)
            abs_target = os.path.abspath(clean_path)
            if not abs_target.startswith(abs_base):
                raise PermissionError(f"Path traversal detected: {path} escapes {base_dir}")

        return clean_path


class SecretMasker:
    """Redacts secrets, passwords, and API keys from text and structured payloads."""

    @staticmethod
    def mask_text(text: str) -> str:
        """Replace detected API keys and secrets with [REDACTED_SECRET]."""
        if not text:
            return ""
        result = str(text)
        for pat in _SECRET_PATTERNS:
            result = pat.sub("[REDACTED_SECRET]", result)
        return result

    @classmethod
    def mask_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively redact secrets from dictionary values and sensitive keys."""
        masked: Dict[str, Any] = {}
        sensitive_keys = {"password", "secret", "token", "api_key", "key", "authorization"}

        for k, v in data.items():
            k_lower = str(k).lower()
            if any(s in k_lower for s in sensitive_keys) and isinstance(v, str):
                masked[k] = "[REDACTED_SECRET]"
            elif isinstance(v, dict):
                masked[k] = cls.mask_dict(v)
            elif isinstance(v, list):
                masked[k] = [cls.mask_dict(item) if isinstance(item, dict)
                             else cls.mask_text(str(item)) if isinstance(item, str)
                             else item for item in v]
            elif isinstance(v, str):
                masked[k] = cls.mask_text(v)
            else:
                masked[k] = v
        return masked
