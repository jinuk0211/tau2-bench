"""Shared validation for the remote TauBench failure-steering protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

MATRIX_SCHEMA = "agent-failure-steering-matrix-v1"
MATRIX_COMPILER_VERSION = "failure-steering-compiler-v2"


def validate_official_airline_splits(
    matrix: Mapping[str, Any],
    official_splits: Mapping[str, list[str]],
) -> None:
    """Require the frozen evaluation set to equal Tau2's official test split."""
    splits = matrix.get("splits") or {}
    train = [str(value) for value in splits.get("train_task_ids") or []]
    validation = [str(value) for value in splits.get("validation_task_ids") or []]
    evaluation = [str(value) for value in splits.get("evaluation_task_ids") or []]
    if len(train) != len(set(train)) or len(validation) != len(set(validation)):
        raise ValueError("failure steering train/validation splits contain duplicates")
    if len(evaluation) != len(set(evaluation)):
        raise ValueError("failure steering evaluation split contains duplicates")
    if set(train).intersection(validation):
        raise ValueError("failure steering train and validation splits overlap")
    if set(train).union(validation) != {
        str(value) for value in official_splits.get("train") or []
    }:
        raise ValueError(
            "failure steering train+validation tasks do not match Tau2 official train"
        )
    if set(evaluation) != {str(value) for value in official_splits.get("test") or []}:
        raise ValueError(
            "failure steering evaluation tasks do not match Tau2 official test"
        )
    if set(evaluation).intersection(train) or set(evaluation).intersection(validation):
        raise ValueError(
            "failure steering evaluation tasks leak into artifact selection"
        )


def load_failure_steering_matrix(path: Path) -> dict[str, Any]:
    """Load a fingerprinted remote matrix and bind it to Tau2's official split."""
    matrix = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if matrix.get("schema_version") != MATRIX_SCHEMA:
        raise ValueError("unsupported failure steering matrix")
    if matrix.get("compiler_version") != MATRIX_COMPILER_VERSION:
        raise ValueError("unsupported failure steering compiler version")
    expected_fingerprint = matrix.get("matrix_fingerprint")
    unsigned = dict(matrix)
    unsigned.pop("matrix_fingerprint", None)
    actual_fingerprint = hashlib.sha256(
        json.dumps(
            unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if expected_fingerprint != actual_fingerprint:
        raise ValueError("failure steering matrix fingerprint mismatch")
    if matrix.get("benchmark") != "taubench-airline":
        raise ValueError("the matrix must target taubench-airline")
    execution = matrix.get("execution") or {}
    if execution.get("mode") != "remote":
        raise ValueError("the failure steering protocol requires remote execution")
    official_path = (
        Path(__file__).resolve().parents[3]
        / "data"
        / "tau2"
        / "domains"
        / "airline"
        / "split_tasks.json"
    )
    official_splits = json.loads(official_path.read_text(encoding="utf-8"))
    validate_official_airline_splits(matrix, official_splits)
    return matrix
