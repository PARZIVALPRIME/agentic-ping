"""Canonical, versioned metadata for benchmark and ablation artifacts.

The runner records only facts it can observe.  Unknown values remain ``None``;
they are never replaced with zero because zero is a measured value in this
project (for example a deterministic run has zero provider tokens).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from typing import Any, Dict, Iterable, Optional


EVALUATION_SCHEMA_VERSION = "1.0"
EVALUATOR_VERSION = "deterministic-ladder-v1"


def file_sha256(path: str) -> Optional[str]:
    """Return the dataset digest, or ``None`` when no dataset was supplied."""
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> Optional[str]:
    """Best-effort commit ID; source archives deliberately report unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            text=True, timeout=2,
        ).strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def run_metadata(*, dataset_path: str, dataset_size: int, config: Any,
                 pipeline_names: Iterable[str], run_mode: str,
                 backend: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build provenance shared by benchmark, ablation, and generated reports."""
    llm = getattr(config, "llm", None)
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "dataset": dataset_path or None,
        "dataset_hash": file_sha256(dataset_path),
        "dataset_size": dataset_size,
        "git_commit": git_commit(),
        "evaluator_version": EVALUATOR_VERSION,
        "model": getattr(llm, "chat_model", None),
        "model_provider": getattr(llm, "provider", None),
        "temperature": getattr(llm, "temperature", None),
        "run_mode": run_mode,
        "pipeline_names": list(pipeline_names),
        "backend": backend,
    }


def validate_summary(summary: Dict[str, Any]) -> list[str]:
    """Return schema defects for publication gates without rewriting history."""
    required = ("schema_version", "dataset", "dataset_hash", "dataset_size",
                "git_commit", "evaluator_version", "model", "model_provider",
                "temperature", "run_mode", "pipeline_names")
    missing = [key for key in required if key not in summary]
    if summary.get("run_mode") not in {"deterministic", "live", "provider-failed"}:
        missing.append("run_mode must be deterministic, live, or provider-failed")
    return missing
