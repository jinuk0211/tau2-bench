"""Shared local Hugging Face backend with optional Jacobian-lens telemetry.

The backend deliberately renders and tokenizes a conversation exactly once.
Those exact ``input_ids`` are passed to ``model.generate`` and written to the
telemetry record.  Observe mode measures the completed generation with a
teacher-forced forward pass, so installing observers cannot change decoding.

The J-vector for token ``w`` at layer ``l`` is defined as::

    v(l, w) = normalize(J_l.T @ W_U[w])

where ``J_l`` is the fitted Jacobian transport and ``W_U[w]`` is the model's
unembedding row.  Positive movement along this vector increases the linear
Jacobian-lens score for ``w``.  ``finite_difference_token_effect`` checks the
effect against the model's actual unembedding around a supplied residual.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

from tau2.agent.jservo import (
    JServoConfig,
    jservo_generation_hooks,
    load_jservo_artifact,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool


class InstrumentationMode(str, Enum):
    """Generation mode for the shared backend."""

    OFF = "off"
    OBSERVE = "observe"
    INTERVENE = "intervene"


@dataclass(frozen=True)
class InterventionConfig:
    """A residual-stream intervention applied to the current final token.

    ``steer`` adds ``strength * vector``. ``ablate`` removes ``strength`` times
    the projection onto the vector. ``patch`` linearly moves the residual
    toward ``vector`` (which is interpreted as the source residual).
    """

    kind: Literal["steer", "ablate", "patch"]
    layer: int
    strength: float = 1.0
    method: str = "legacy"
    concept_token_id: Optional[int] = None
    vector: Optional[tuple[float, ...]] = None
    vector_path: Optional[Path] = None
    artifact_layer: Optional[int] = None
    vector_scaling: Literal["raw", "unit"] = "raw"
    cast_prefill_mode: Literal["all_tokens", "decision_only"] = "all_tokens"
    cast_gate_override: Optional[bool] = None
    cast_comparator_override: Optional[Literal["greater", "less"]] = None
    cast_invert_comparator: bool = False
    mera_alpha_override: Optional[float] = None
    mera_prefill_mode: Literal["all_tokens", "decision_only"] = "all_tokens"
    sadi_top_k: Optional[int] = None
    sadi_units: Optional[tuple[tuple[int, int], ...]] = None
    iti_top_k: Optional[int] = None
    iti_entries: Optional[
        tuple[tuple[int, int, tuple[float, ...], float], ...]
    ] = None
    austeer_top_k: Optional[int] = None
    austeer_units: Optional[tuple[tuple[int, int, float], ...]] = None
    austeer_prefill_mode: Literal["all_tokens", "decision_only"] = "all_tokens"
    turn_indices: tuple[int, ...] = ()
    boundaries: tuple[str, ...] = ()
    apply_prefill_decision: bool = True
    apply_decode: bool = True

    def __post_init__(self) -> None:
        sources = sum(
            value is not None
            for value in (
                self.concept_token_id,
                self.vector,
                self.vector_path,
                self.sadi_units,
                self.iti_entries,
                self.austeer_units,
            )
        )
        if sources != 1:
            raise ValueError(
                "intervention requires exactly one of concept_token_id, vector, "
                "or vector_path"
            )
        if self.layer < 0:
            raise ValueError("intervention layer must be non-negative")
        if self.artifact_layer is not None and self.artifact_layer < 0:
            raise ValueError("intervention artifact_layer must be non-negative")
        if not math.isfinite(self.strength):
            raise ValueError("intervention strength must be finite")
        if any(turn < 0 for turn in self.turn_indices):
            raise ValueError("intervention turn_indices must be non-negative")
        if self.vector_scaling not in {"raw", "unit"}:
            raise ValueError("vector_scaling must be 'raw' or 'unit'")
        if self.method in {"caa", "cast", "sadi"} and self.vector_path is None:
            if self.method != "sadi" or self.sadi_units is None:
                raise ValueError(
                    f"{self.method.upper()} intervention requires a versioned vector_path artifact"
                )
        if self.method == "iti" and self.vector_path is None and self.iti_entries is None:
            raise ValueError("ITI requires a versioned artifact or inline control heads")
        if (
            self.method == "austeer"
            and self.vector_path is None
            and self.austeer_units is None
        ):
            raise ValueError("AUSteer requires a versioned artifact or inline control AUs")
        if self.method == "mera" and self.vector_path is None and not (
            self.vector is not None and self.mera_alpha_override is not None
        ):
            raise ValueError(
                "MERA requires an artifact or an inline control probe with alpha override"
            )
        if self.vector_path is not None and self.method not in {
            "caa",
            "cast",
            "mera",
            "sadi",
            "iti",
            "austeer",
        }:
            raise ValueError("vector_path artifacts require a supported artifact method")
        if self.cast_prefill_mode not in {"all_tokens", "decision_only"}:
            raise ValueError("cast_prefill_mode must be all_tokens or decision_only")
        if self.cast_comparator_override not in {None, "greater", "less"}:
            raise ValueError("cast_comparator_override must be greater or less")
        if self.method == "cast" and self.kind != "steer":
            raise ValueError("CAST currently supports kind='steer' only")
        if self.method == "mera" and self.kind != "steer":
            raise ValueError("MERA currently supports kind='steer' only")
        if self.method == "sadi" and self.kind != "steer":
            raise ValueError("SADI currently supports kind='steer' only")
        if self.method == "iti" and self.kind != "steer":
            raise ValueError("ITI currently supports kind='steer' only")
        if self.method == "austeer" and self.kind != "steer":
            raise ValueError("AUSteer currently supports kind='steer' only")
        if self.method == "sadi" and self.strength < 0.0:
            raise ValueError("SADI strength must be non-negative")
        if self.sadi_top_k is not None and self.sadi_top_k <= 0:
            raise ValueError("sadi_top_k must be positive")
        if self.sadi_units is not None and (
            self.method != "sadi"
            or not self.sadi_units
            or len(set(self.sadi_units)) != len(self.sadi_units)
            or any(layer < 0 or dimension < 0 for layer, dimension in self.sadi_units)
        ):
            raise ValueError("sadi_units must contain unique non-negative (layer, dimension) pairs")
        if self.iti_top_k is not None and self.iti_top_k <= 0:
            raise ValueError("iti_top_k must be positive")
        if self.iti_entries is not None:
            pairs = [(layer, head) for layer, head, _direction, _scale in self.iti_entries]
            if (
                self.method != "iti"
                or not self.iti_entries
                or len(set(pairs)) != len(pairs)
                or any(
                    layer < 0
                    or head < 0
                    or not direction
                    or not all(math.isfinite(value) for value in direction)
                    or not math.isfinite(scale)
                    or scale <= 0.0
                    for layer, head, direction, scale in self.iti_entries
                )
            ):
                raise ValueError("iti_entries contain invalid head directions or scales")
        if self.austeer_top_k is not None and self.austeer_top_k <= 0:
            raise ValueError("austeer_top_k must be positive")
        if self.austeer_units is not None:
            pairs = [(layer, dimension) for layer, dimension, _beta in self.austeer_units]
            if (
                self.method != "austeer"
                or not self.austeer_units
                or len(set(pairs)) != len(pairs)
                or any(
                    layer < 0
                    or dimension < 0
                    or not math.isfinite(beta)
                    or abs(beta) > 1.0
                    for layer, dimension, beta in self.austeer_units
                )
            ):
                raise ValueError("austeer_units contain invalid scalar AUs or beta values")
        if self.austeer_prefill_mode not in {"all_tokens", "decision_only"}:
            raise ValueError("austeer_prefill_mode must be all_tokens or decision_only")
        if self.mera_alpha_override is not None and not 0.0 < self.mera_alpha_override <= 1.0:
            raise ValueError("mera_alpha_override must be in (0, 1]")
        if self.mera_prefill_mode not in {"all_tokens", "decision_only"}:
            raise ValueError("mera_prefill_mode must be all_tokens or decision_only")
        if self.vector is not None and not self.vector:
            raise ValueError("inline intervention vector cannot be empty")

    @classmethod
    def from_dict(
        cls, value: Optional[dict[str, Any]]
    ) -> Optional["InterventionConfig"]:
        """Build a validated intervention from JSON-compatible arguments."""
        if value is None:
            return None
        copied = dict(value)
        if copied.get("vector") is not None:
            copied["vector"] = tuple(float(item) for item in copied["vector"])
        if copied.get("vector_path") is not None:
            copied["vector_path"] = Path(copied["vector_path"])
        if copied.get("turn_indices") is not None:
            if isinstance(copied["turn_indices"], (str, bytes)):
                raise ValueError("turn_indices must be a sequence of integers")
            copied["turn_indices"] = tuple(int(item) for item in copied["turn_indices"])
        if copied.get("boundaries") is not None:
            if isinstance(copied["boundaries"], (str, bytes)):
                raise ValueError("boundaries must be a sequence of strings")
            copied["boundaries"] = tuple(str(item) for item in copied["boundaries"])
        if copied.get("sadi_units") is not None:
            if isinstance(copied["sadi_units"], (str, bytes)):
                raise ValueError("sadi_units must be a sequence of pairs")
            copied["sadi_units"] = tuple(
                (int(item[0]), int(item[1])) for item in copied["sadi_units"]
            )
        if copied.get("iti_entries") is not None:
            if isinstance(copied["iti_entries"], (str, bytes)):
                raise ValueError("iti_entries must be a sequence")
            entries = []
            for item in copied["iti_entries"]:
                if isinstance(item, dict):
                    entries.append(
                        (
                            int(item["layer"]),
                            int(item["head"]),
                            tuple(float(value) for value in item["direction"]),
                            float(item["scale"]),
                        )
                    )
                else:
                    entries.append(
                        (
                            int(item[0]),
                            int(item[1]),
                            tuple(float(value) for value in item[2]),
                            float(item[3]),
                        )
                    )
            copied["iti_entries"] = tuple(entries)
        if copied.get("austeer_units") is not None:
            if isinstance(copied["austeer_units"], (str, bytes)):
                raise ValueError("austeer_units must be a sequence")
            copied["austeer_units"] = tuple(
                (
                    int(item["layer"]),
                    int(item["dimension"]),
                    float(item["beta"]),
                )
                if isinstance(item, dict)
                else (int(item[0]), int(item[1]), float(item[2]))
                for item in copied["austeer_units"]
            )
        return cls(**copied)


@dataclass(frozen=True)
class HFBackendConfig:
    """Configuration that must remain identical across experimental modes."""

    model_name_or_path: str
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    mode: InstrumentationMode = InstrumentationMode.OFF
    seed: int = 42
    max_input_tokens: int = 16384
    selected_layers: tuple[int, ...] = ()
    concept_tokens: dict[str, str | int] = field(default_factory=dict)
    telemetry_path: Optional[Path] = None
    lens_path: Optional[Path] = None
    intervention: Optional[InterventionConfig] = None
    controller: Optional[JServoConfig] = None
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    device: str = "auto"
    dtype: str = "auto"
    sdpa_backend: Literal["auto", "efficient", "flash", "math"] = "auto"
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        controls = int(self.intervention is not None) + int(self.controller is not None)
        if self.mode == InstrumentationMode.INTERVENE and controls != 1:
            raise ValueError(
                "intervene mode requires exactly one intervention or J-Servo controller"
            )
        if self.mode != InstrumentationMode.INTERVENE and controls:
            raise ValueError("interventions and controllers are valid only in intervene mode")
        if self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        if self.sdpa_backend not in {"auto", "efficient", "flash", "math"}:
            raise ValueError(f"unknown SDPA backend: {self.sdpa_backend}")


def _generation_kwargs_for_tokenizer(
    config: HFBackendConfig, tokenizer: Any
) -> dict[str, Any]:
    """Resolve generation defaults from the tokenizer's actual chat tokens."""
    generation_kwargs = dict(config.generation_kwargs)
    generation_kwargs.setdefault("max_new_tokens", 256)
    generation_kwargs.setdefault("do_sample", False)
    tokenizer_eos_token_id = tokenizer.eos_token_id
    if tokenizer_eos_token_id is not None:
        # Some multimodal model configs (including Qwen3.5) expose a text
        # config EOS that differs from the chat template's <|im_end|>.
        # Transformers otherwise generates until max_new_tokens instead of
        # stopping at the tokenizer's actual assistant-turn boundary.
        generation_kwargs.setdefault("eos_token_id", tokenizer_eos_token_id)
    tokenizer_pad_token_id = tokenizer.pad_token_id
    if tokenizer_pad_token_id is None:
        tokenizer_pad_token_id = tokenizer_eos_token_id
    if tokenizer_pad_token_id is not None:
        generation_kwargs.setdefault("pad_token_id", tokenizer_pad_token_id)
    return generation_kwargs


@dataclass(frozen=True)
class ParsedToolCall:
    """Parsed tool call plus character spans in the generated text."""

    name: str
    arguments: dict[str, Any]
    name_span: Optional[tuple[int, int]] = None
    arguments_span: Optional[tuple[int, int]] = None


@dataclass
class BackendGeneration:
    """One backend result and the exact token sequence that produced it."""

    message: AssistantMessage
    prompt_input_ids: list[int]
    generated_ids: list[int]
    rendered_text: str
    telemetry_record: dict[str, Any]


@dataclass
class _ModelBundle:
    model: Any
    tokenizer: Any
    lens_model: Any
    lens: Any = None


_BUNDLE_CACHE: dict[tuple[Any, ...], _ModelBundle] = {}
_BUNDLE_CACHE_LOCK = threading.Lock()
_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_LOCK = threading.Lock()
_STEERING_ARTIFACT_SCHEMA = "agent-steering-vector-v1"
_CAST_ARTIFACT_SCHEMA = "agent-cast-v1"
_MERA_ARTIFACT_SCHEMA = "agent-mera-v1"
_SADI_ARTIFACT_SCHEMA = "agent-sadi-v1"
_ITI_ARTIFACT_SCHEMA = "agent-iti-v1"
_AUSTEER_ARTIFACT_SCHEMA = "agent-austeer-v1"
_STEERING_TENSOR_FIELDS = {
    "direction",
    "unit_direction",
    "positive_mean",
    "negative_mean",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().contiguous().float().cpu().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def _artifact_metadata(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in artifact.items()
        if key not in _STEERING_TENSOR_FIELDS
        and key not in {"metadata_fingerprint", "vector_fingerprint", "direction_norm"}
    }


def load_caa_direction_artifact(
    path: Path,
    *,
    model_id: str,
    layer: int,
    d_model: int,
    scaling: Literal["raw", "unit"] = "raw",
) -> tuple[Any, dict[str, Any]]:
    """Load and identity-check the shared tensor-only CAA artifact."""
    import torch

    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise ValueError("CAA artifact must be a dictionary")
    if artifact.get("schema_version") != _STEERING_ARTIFACT_SCHEMA:
        raise ValueError("unsupported steering artifact schema")
    if artifact.get("method") != "caa":
        raise ValueError("steering artifact is not CAA")
    if artifact.get("orientation") != "positive_minus_negative":
        raise ValueError("CAA artifact has an unknown direction orientation")
    if artifact.get("model_id") != model_id:
        raise ValueError(
            f"CAA artifact model {artifact.get('model_id')!r} does not match "
            f"{model_id!r}"
        )
    if int(artifact.get("layer", -1)) != int(layer):
        raise ValueError(
            f"CAA artifact layer {artifact.get('layer')!r} does not match {layer}"
        )
    metadata_fingerprint = hashlib.sha256(
        json.dumps(
            _artifact_metadata(artifact),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if metadata_fingerprint != artifact.get("metadata_fingerprint"):
        raise ValueError("CAA artifact metadata fingerprint mismatch")
    direction = artifact.get("direction")
    unit_direction = artifact.get("unit_direction")
    if (
        direction is None
        or unit_direction is None
        or direction.ndim != 1
        or unit_direction.shape != direction.shape
    ):
        raise ValueError("CAA artifact vectors are missing or malformed")
    if int(artifact.get("d_model", -1)) != int(d_model):
        raise ValueError("CAA artifact d_model does not match the loaded model")
    if int(direction.numel()) != int(d_model):
        raise ValueError("CAA direction width does not match the loaded model")
    if not bool(torch.isfinite(direction).all()) or not bool(
        torch.isfinite(unit_direction).all()
    ):
        raise ValueError("CAA artifact vectors contain non-finite values")
    norm = torch.linalg.vector_norm(direction.float())
    if not bool(torch.isfinite(norm)) or float(norm) == 0.0:
        raise ValueError("CAA artifact direction has zero or non-finite norm")
    if abs(float(artifact.get("direction_norm", -1.0)) - float(norm)) > max(
        1e-6, float(norm) * 1e-5
    ):
        raise ValueError("CAA artifact direction_norm is inconsistent")
    if not torch.allclose(
        unit_direction.float(),
        direction.float() / norm,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("CAA artifact unit_direction is inconsistent")
    if _tensor_sha256(direction) != artifact.get("vector_fingerprint"):
        raise ValueError("CAA direction fingerprint mismatch")
    vector = direction if scaling == "raw" else unit_direction
    metadata = {
        "schema_version": artifact["schema_version"],
        "method": artifact["method"],
        "model_id": artifact["model_id"],
        "model_revision": artifact.get("model_revision"),
        "layer": int(artifact["layer"]),
        "orientation": artifact["orientation"],
        "positive_label": artifact.get("positive_label"),
        "negative_label": artifact.get("negative_label"),
        "extraction_site": artifact.get("extraction_site"),
        "benchmark": artifact.get("benchmark"),
        "pair_count": int(artifact.get("pair_count", 0)),
        "vector_scaling": scaling,
        "vector_fingerprint": artifact["vector_fingerprint"],
        "metadata_fingerprint": artifact["metadata_fingerprint"],
    }
    return vector.detach().float().cpu(), metadata


def load_cast_artifact(
    path: Path,
    *,
    model_id: str,
    behavior_layer: int,
    d_model: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and identity-check the shared CAST behavior/gate artifact."""
    import torch

    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise ValueError("CAST artifact must be a dictionary")
    if artifact.get("schema_version") != _CAST_ARTIFACT_SCHEMA:
        raise ValueError("unsupported CAST artifact schema")
    if artifact.get("method") != "cast":
        raise ValueError("artifact is not CAST")
    if artifact.get("model_id") != model_id:
        raise ValueError("CAST artifact model does not match the loaded model")
    if int(artifact.get("behavior_layer", -1)) != int(behavior_layer):
        raise ValueError("CAST artifact behavior layer does not match")
    behavior = artifact.get("behavior_direction")
    condition = artifact.get("condition_direction")
    if (
        behavior is None
        or condition is None
        or behavior.ndim != 1
        or condition.shape != behavior.shape
        or int(behavior.numel()) != int(d_model)
        or int(artifact.get("d_model", -1)) != int(d_model)
    ):
        raise ValueError("CAST artifact vectors are missing or malformed")
    if not bool(torch.isfinite(behavior).all()) or not bool(
        torch.isfinite(condition).all()
    ):
        raise ValueError("CAST artifact vectors contain non-finite values")
    if abs(float(behavior.float().norm()) - 1.0) > 1e-5:
        raise ValueError("CAST behavior direction is not unit norm")
    if abs(float(condition.float().norm()) - 1.0) > 1e-5:
        raise ValueError("CAST condition direction is not unit norm")
    metadata = {
        key: value
        for key, value in artifact.items()
        if key
        not in {
            "metadata_fingerprint",
            "behavior_vector_fingerprint",
            "condition_vector_fingerprint",
            "behavior_direction",
            "condition_direction",
        }
    }
    fingerprint = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if fingerprint != artifact.get("metadata_fingerprint"):
        raise ValueError("CAST artifact metadata fingerprint mismatch")
    if _tensor_sha256(behavior) != artifact.get("behavior_vector_fingerprint"):
        raise ValueError("CAST behavior vector fingerprint mismatch")
    if _tensor_sha256(condition) != artifact.get("condition_vector_fingerprint"):
        raise ValueError("CAST condition vector fingerprint mismatch")
    comparator = artifact.get("condition_comparator")
    comparison_mode = artifact.get("condition_comparison_mode")
    if comparator not in {"greater", "less"} or comparison_mode not in {"mean", "last"}:
        raise ValueError("CAST artifact gate configuration is invalid")
    threshold = float(artifact.get("condition_threshold"))
    if not math.isfinite(threshold) or int(artifact.get("condition_layer", -1)) < 0:
        raise ValueError("CAST artifact gate threshold or layer is invalid")
    positive_ids = artifact.get("gate_positive_ids", [])
    negative_ids = artifact.get("gate_negative_ids", [])
    positive_scores = artifact.get("gate_positive_scores", [])
    negative_scores = artifact.get("gate_negative_scores", [])
    if (
        not positive_ids
        or not negative_ids
        or len(positive_ids) != len(positive_scores)
        or len(negative_ids) != len(negative_scores)
        or len(set(positive_ids + negative_ids)) != len(positive_ids) + len(negative_ids)
    ):
        raise ValueError("CAST artifact gate calibration records are malformed")
    scores = [float(value) for value in positive_scores + negative_scores]
    labels = [True] * len(positive_scores) + [False] * len(negative_scores)
    predictions = [
        score > threshold if comparator == "greater" else score < threshold
        for score in scores
    ]
    tp = sum(prediction and label for prediction, label in zip(predictions, labels, strict=True))
    fp = sum(
        prediction and not label
        for prediction, label in zip(predictions, labels, strict=True)
    )
    tn = sum(
        not prediction and not label
        for prediction, label in zip(predictions, labels, strict=True)
    )
    fn = sum(
        not prediction and label
        for prediction, label in zip(predictions, labels, strict=True)
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    expected_metrics = {
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "accuracy": (tp + tn) / len(labels),
    }
    if any(
        abs(float(artifact.get("gate_metrics", {}).get(key, -1.0)) - value) > 1e-9
        for key, value in expected_metrics.items()
    ):
        raise ValueError("CAST artifact gate metrics do not match calibration scores")
    source = {
        "schema_version": artifact["schema_version"],
        "method": artifact["method"],
        "model_id": artifact["model_id"],
        "model_revision": artifact.get("model_revision"),
        "behavior_layer": int(artifact["behavior_layer"]),
        "condition_layer": int(artifact["condition_layer"]),
        "condition_threshold": float(artifact["condition_threshold"]),
        "condition_comparator": comparator,
        "condition_comparison_mode": comparison_mode,
        "benchmark": artifact.get("benchmark"),
        "behavior_pair_count": int(artifact.get("behavior_pair_count", 0)),
        "condition_pair_count": int(artifact.get("condition_pair_count", 0)),
        "behavior_vector_fingerprint": artifact["behavior_vector_fingerprint"],
        "condition_vector_fingerprint": artifact["condition_vector_fingerprint"],
        "metadata_fingerprint": artifact["metadata_fingerprint"],
    }
    return artifact, source


def cast_condition_similarity(
    hidden_states: Any,
    direction: Any,
    *,
    comparison_mode: Literal["mean", "last"] = "mean",
) -> Any:
    """Compute CAST's cosine between h and tanh(P_condition h)."""
    import torch

    hidden = hidden_states.float()
    if hidden.ndim == 2:
        hidden = hidden.mean(dim=0) if comparison_mode == "mean" else hidden[-1]
    if hidden.ndim != 1:
        raise ValueError("CAST condition state must have shape [tokens, d] or [d]")
    vector = direction.to(hidden.device, dtype=hidden.dtype)
    denominator = torch.dot(vector, vector)
    if not bool(torch.isfinite(denominator)) or float(denominator) == 0.0:
        raise ValueError("CAST condition direction is zero or non-finite")
    projected = torch.tanh(vector * (torch.dot(vector, hidden) / denominator))
    hidden_norm = hidden.norm()
    projected_norm = projected.norm()
    if (
        not bool(torch.isfinite(hidden_norm))
        or not bool(torch.isfinite(projected_norm))
        or float(hidden_norm) == 0.0
        or float(projected_norm) == 0.0
    ):
        raise ValueError("CAST condition similarity has a zero or non-finite norm")
    return torch.dot(hidden, projected) / (hidden_norm * projected_norm)


def load_mera_artifact(
    path: Path,
    *,
    model_id: str,
    layer: int,
    d_model: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and identity-check the shared calibrated MERA probe artifact."""
    import torch

    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise ValueError("MERA artifact must be a dictionary")
    if artifact.get("schema_version") != _MERA_ARTIFACT_SCHEMA:
        raise ValueError("unsupported MERA artifact schema")
    if artifact.get("method") != "mera":
        raise ValueError("artifact is not MERA")
    if artifact.get("model_id") != model_id:
        raise ValueError("MERA artifact model does not match the loaded model")
    if int(artifact.get("layer", -1)) != int(layer):
        raise ValueError("MERA artifact layer does not match")
    vector = artifact.get("probe_vector")
    if (
        vector is None
        or vector.ndim != 1
        or int(vector.numel()) != int(d_model)
        or int(artifact.get("d_model", -1)) != int(d_model)
    ):
        raise ValueError("MERA probe vector is missing or malformed")
    if not bool(torch.isfinite(vector).all()) or float(vector.float().norm()) == 0.0:
        raise ValueError("MERA probe vector is zero or non-finite")
    metadata = {
        key: value
        for key, value in artifact.items()
        if key not in {"metadata_fingerprint", "probe_vector_fingerprint", "probe_vector"}
    }
    fingerprint = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if fingerprint != artifact.get("metadata_fingerprint"):
        raise ValueError("MERA artifact metadata fingerprint mismatch")
    if _tensor_sha256(vector) != artifact.get("probe_vector_fingerprint"):
        raise ValueError("MERA probe vector fingerprint mismatch")
    alpha = float(artifact.get("selected_alpha", -1.0))
    if not 0.0 < alpha <= 1.0 or alpha not in {
        float(value) for value in artifact.get("alpha_grid", [])
    }:
        raise ValueError("MERA selected alpha is invalid")
    source = {
        "schema_version": artifact["schema_version"],
        "method": artifact["method"],
        "model_id": artifact["model_id"],
        "model_revision": artifact.get("model_revision"),
        "layer": int(artifact["layer"]),
        "selected_alpha": alpha,
        "selection_metrics": artifact.get("selection_metrics"),
        "selection_objective": artifact.get("selection_objective"),
        "benchmark": artifact.get("benchmark"),
        "train_pair_count": int(artifact.get("train_pair_count", 0)),
        "probe_vector_fingerprint": artifact["probe_vector_fingerprint"],
        "metadata_fingerprint": artifact["metadata_fingerprint"],
    }
    return artifact, source


def load_sadi_artifact(
    path: Path,
    *,
    model_id: str,
    d_model: int,
    top_k: Optional[int] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and identity-check the shared SADI sparse-unit artifact."""
    import torch

    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise ValueError("SADI artifact must be a dictionary")
    if artifact.get("schema_version") != _SADI_ARTIFACT_SCHEMA:
        raise ValueError("unsupported SADI artifact schema")
    if artifact.get("method") != "sadi_hidden":
        raise ValueError("artifact is not SADI hidden-unit steering")
    if artifact.get("model_id") != model_id:
        raise ValueError("SADI artifact model does not match the loaded model")
    if int(artifact.get("d_model", -1)) != int(d_model):
        raise ValueError("SADI artifact width does not match the loaded model")
    units = artifact.get("selected_units")
    scores = artifact.get("unit_scores")
    artifact_top_k = int(artifact.get("top_k", -1))
    if (
        units is None
        or scores is None
        or units.ndim != 2
        or tuple(units.shape) != (artifact_top_k, 2)
        or scores.ndim != 1
        or int(scores.shape[0]) != artifact_top_k
        or units.dtype != torch.int64
        or not bool(torch.isfinite(scores).all())
    ):
        raise ValueError("SADI selected units are missing or malformed")
    requested_top_k = artifact_top_k if top_k is None else int(top_k)
    if requested_top_k <= 0 or requested_top_k > artifact_top_k:
        raise ValueError("SADI requested top_k is outside the artifact")
    layers = {int(value) for value in artifact.get("layers", [])}
    pairs = [(int(layer), int(dimension)) for layer, dimension in units.tolist()]
    if (
        not layers
        or len(set(pairs)) != len(pairs)
        or any(layer not in layers or not 0 <= dimension < d_model for layer, dimension in pairs)
    ):
        raise ValueError("SADI selected unit indices are invalid")
    validation_scores = artifact.get("validation_unit_scores")
    if validation_scores is not None and (
        validation_scores.shape != scores.shape
        or not bool(torch.isfinite(validation_scores).all())
    ):
        raise ValueError("SADI validation unit scores are malformed")
    metadata = {
        key: value
        for key, value in artifact.items()
        if key
        not in {
            "metadata_fingerprint",
            "selected_units_fingerprint",
            "unit_scores_fingerprint",
            "validation_unit_scores_fingerprint",
            "selected_units",
            "unit_scores",
            "validation_unit_scores",
        }
    }
    fingerprint = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if fingerprint != artifact.get("metadata_fingerprint"):
        raise ValueError("SADI artifact metadata fingerprint mismatch")
    if _tensor_sha256(units) != artifact.get("selected_units_fingerprint"):
        raise ValueError("SADI selected unit fingerprint mismatch")
    if _tensor_sha256(scores) != artifact.get("unit_scores_fingerprint"):
        raise ValueError("SADI unit score fingerprint mismatch")
    if validation_scores is not None and _tensor_sha256(validation_scores) != artifact.get(
        "validation_unit_scores_fingerprint"
    ):
        raise ValueError("SADI validation unit score fingerprint mismatch")
    source = {
        "schema_version": artifact["schema_version"],
        "method": artifact["method"],
        "model_id": artifact["model_id"],
        "model_revision": artifact.get("model_revision"),
        "layers": sorted(layers),
        "artifact_top_k": artifact_top_k,
        "requested_top_k": requested_top_k,
        "benchmark": artifact.get("benchmark"),
        "pair_count": int(artifact.get("pair_count", 0)),
        "validation_pair_count": int(artifact.get("validation_pair_count", 0)),
        "selected_units_fingerprint": artifact["selected_units_fingerprint"],
        "metadata_fingerprint": artifact["metadata_fingerprint"],
    }
    return artifact, source


def load_iti_artifact(
    path: Path,
    *,
    model_id: str,
    d_model: int,
    top_k: Optional[int] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and identity-check the shared ITI head artifact."""
    import torch

    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise ValueError("ITI artifact must be a dictionary")
    if artifact.get("schema_version") != _ITI_ARTIFACT_SCHEMA:
        raise ValueError("unsupported ITI artifact schema")
    if artifact.get("method") != "iti":
        raise ValueError("artifact is not ITI")
    if artifact.get("model_id") != model_id:
        raise ValueError("ITI artifact model does not match the loaded model")
    num_heads = int(artifact.get("num_attention_heads", -1))
    head_dim = int(artifact.get("head_dim", -1))
    artifact_top_k = int(artifact.get("top_k", -1))
    layers = [int(value) for value in artifact.get("layers", [])]
    if num_heads * head_dim != d_model or int(artifact.get("d_model", -1)) != d_model:
        raise ValueError("ITI artifact attention shape does not match the model")
    expected_shapes = {
        "selected_heads": (artifact_top_k, 2),
        "head_directions": (artifact_top_k, head_dim),
        "projection_stds": (artifact_top_k,),
        "validation_accuracies": (len(layers), num_heads),
        "probe_weights": (len(layers), num_heads, head_dim),
        "probe_intercepts": (len(layers), num_heads),
    }
    for name, shape in expected_shapes.items():
        tensor = artifact.get(name)
        if tensor is None or tuple(tensor.shape) != shape:
            raise ValueError(f"ITI {name} is missing or malformed")
        if name == "selected_heads":
            if tensor.dtype != torch.int64:
                raise ValueError("ITI selected heads must be int64")
        elif not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"ITI {name} contains non-finite values")
        if _tensor_sha256(tensor) != artifact.get(f"{name}_fingerprint"):
            raise ValueError(f"ITI {name} fingerprint mismatch")
    requested_top_k = artifact_top_k if top_k is None else int(top_k)
    if requested_top_k <= 0 or requested_top_k > artifact_top_k:
        raise ValueError("ITI requested top_k is outside the artifact")
    pairs = [
        (int(layer), int(head)) for layer, head in artifact["selected_heads"].tolist()
    ]
    if len(set(pairs)) != len(pairs) or any(
        layer not in layers or not 0 <= head < num_heads for layer, head in pairs
    ):
        raise ValueError("ITI selected head indices are invalid")
    if not bool(
        torch.allclose(
            artifact["head_directions"].norm(dim=1),
            torch.ones(artifact_top_k),
            atol=1e-5,
        )
    ) or not bool((artifact["projection_stds"] > 0).all()):
        raise ValueError("ITI directions or projection scales are invalid")
    tensor_names = set(expected_shapes)
    metadata = {
        key: value
        for key, value in artifact.items()
        if key != "metadata_fingerprint"
        and key not in tensor_names
        and not key.endswith("_fingerprint")
    }
    fingerprint = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if fingerprint != artifact.get("metadata_fingerprint"):
        raise ValueError("ITI artifact metadata fingerprint mismatch")
    source = {
        "schema_version": artifact["schema_version"],
        "method": artifact["method"],
        "model_id": artifact["model_id"],
        "model_revision": artifact.get("model_revision"),
        "layers": layers,
        "num_attention_heads": num_heads,
        "head_dim": head_dim,
        "artifact_top_k": artifact_top_k,
        "requested_top_k": requested_top_k,
        "benchmark": artifact.get("benchmark"),
        "train_pair_count": int(artifact.get("train_pair_count", 0)),
        "validation_pair_count": int(artifact.get("validation_pair_count", 0)),
        "selected_heads_fingerprint": artifact["selected_heads_fingerprint"],
        "metadata_fingerprint": artifact["metadata_fingerprint"],
    }
    return artifact, source


def load_austeer_artifact(
    path: Path,
    *,
    model_id: str,
    d_model: int,
    top_k: Optional[int] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and identity-check the shared AUSteer scalar-AU artifact."""
    import torch

    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise ValueError("AUSteer artifact must be a dictionary")
    if artifact.get("schema_version") != _AUSTEER_ARTIFACT_SCHEMA:
        raise ValueError("unsupported AUSteer artifact schema")
    if artifact.get("method") != "austeer":
        raise ValueError("artifact is not AUSteer")
    if artifact.get("model_id") != model_id:
        raise ValueError("AUSteer artifact model does not match the loaded model")
    artifact_top_k = int(artifact.get("top_k", -1))
    layers = [int(value) for value in artifact.get("layers", [])]
    if int(artifact.get("d_model", -1)) != int(d_model):
        raise ValueError("AUSteer artifact width does not match the model")
    expected_shapes = {
        "selected_units": (artifact_top_k, 2),
        "selected_betas": (artifact_top_k,),
        "validation_betas": (artifact_top_k,),
    }
    for name, shape in expected_shapes.items():
        tensor = artifact.get(name)
        if tensor is None or tuple(tensor.shape) != shape:
            raise ValueError(f"AUSteer {name} is missing or malformed")
        if name == "selected_units":
            if tensor.dtype != torch.int64:
                raise ValueError("AUSteer selected units must be int64")
        elif not bool(torch.isfinite(tensor).all()) or bool((tensor.abs() > 1.0).any()):
            raise ValueError(f"AUSteer {name} has invalid beta values")
        if _tensor_sha256(tensor) != artifact.get(f"{name}_fingerprint"):
            raise ValueError(f"AUSteer {name} fingerprint mismatch")
    requested_top_k = artifact_top_k if top_k is None else int(top_k)
    if requested_top_k <= 0 or requested_top_k > artifact_top_k:
        raise ValueError("AUSteer requested top_k is outside the artifact")
    pairs = [
        (int(layer), int(dimension))
        for layer, dimension in artifact["selected_units"].tolist()
    ]
    if len(set(pairs)) != len(pairs) or any(
        layer not in layers or not 0 <= dimension < d_model
        for layer, dimension in pairs
    ):
        raise ValueError("AUSteer selected unit indices are invalid")
    tensor_names = set(expected_shapes)
    metadata = {
        key: value
        for key, value in artifact.items()
        if key != "metadata_fingerprint"
        and key not in tensor_names
        and not key.endswith("_fingerprint")
    }
    fingerprint = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if fingerprint != artifact.get("metadata_fingerprint"):
        raise ValueError("AUSteer artifact metadata fingerprint mismatch")
    source = {
        "schema_version": artifact["schema_version"],
        "method": artifact["method"],
        "model_id": artifact["model_id"],
        "model_revision": artifact.get("model_revision"),
        "layers": layers,
        "artifact_top_k": artifact_top_k,
        "requested_top_k": requested_top_k,
        "benchmark": artifact.get("benchmark"),
        "train_pair_count": int(artifact.get("train_pair_count", 0)),
        "validation_pair_count": int(artifact.get("validation_pair_count", 0)),
        "selected_units_fingerprint": artifact["selected_units_fingerprint"],
        "metadata_fingerprint": artifact["metadata_fingerprint"],
    }
    return artifact, source


def mera_closed_form_delta(
    hidden_states: Any,
    probe_vector: Any,
    *,
    alpha: float,
) -> tuple[Any, Any, Any]:
    """Return MERA's exact theta, steering mask, and sigmoid error score."""
    import torch

    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("MERA alpha must be in (0, 1]")
    hidden = hidden_states.float()
    vector = probe_vector.to(hidden.device, dtype=hidden.dtype)
    logits = hidden @ vector
    scores = torch.sigmoid(logits)
    if float(alpha) == 1.0:
        return torch.zeros_like(hidden_states), torch.zeros_like(scores, dtype=torch.bool), scores
    threshold = torch.logit(
        torch.tensor(float(alpha), dtype=hidden.dtype, device=hidden.device)
    )
    condition = scores > float(alpha)
    theta = ((threshold - logits) / (vector.square().sum() + 1e-8)).unsqueeze(-1) * vector
    delta = torch.where(condition.unsqueeze(-1), theta, torch.zeros_like(theta))
    return delta.to(dtype=hidden_states.dtype), condition, scores


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Stable hash for an exact token-id sequence."""
    payload = ",".join(str(int(token_id)) for token_id in token_ids).encode()
    return hashlib.sha256(payload).hexdigest()


def _json_value_span(
    payload: str, key: str, absolute_start: int
) -> Optional[tuple[int, int]]:
    key_match = re.search(rf'"{re.escape(key)}"\s*:\s*', payload)
    if key_match is None:
        return None
    value_start = key_match.end()
    decoder = json.JSONDecoder()
    try:
        _, value_end = decoder.raw_decode(payload[value_start:])
    except json.JSONDecodeError:
        return None
    return absolute_start + value_start, absolute_start + value_start + value_end


def _parsed_call_from_payload(
    payload: str, absolute_start: int
) -> Optional[ParsedToolCall]:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("name"), str):
        return None
    arguments = value.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return ParsedToolCall(
        name=value["name"],
        arguments=arguments,
        name_span=_json_value_span(payload, "name", absolute_start),
        arguments_span=_json_value_span(payload, "arguments", absolute_start),
    )


def _parsed_call_from_xml_payload(
    payload: str, absolute_start: int
) -> Optional[ParsedToolCall]:
    """Parse the native Qwen3.5/3.6 function-and-parameter tool syntax."""
    function = re.search(
        r"<function=([^>\r\n]+)>\s*(.*?)\s*</function>",
        payload,
        re.DOTALL,
    )
    if function is None:
        return None
    name = function.group(1).strip()
    if not name:
        return None
    arguments: dict[str, Any] = {}
    value_spans: list[tuple[int, int]] = []
    parameter_pattern = re.compile(
        r"<parameter=([^>\r\n]+)>\s*(.*?)\s*</parameter>",
        re.DOTALL,
    )
    for parameter in parameter_pattern.finditer(function.group(2)):
        key = parameter.group(1).strip()
        raw_value = parameter.group(2).strip()
        if not key:
            continue
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        arguments[key] = value
        value_start = (
            absolute_start
            + function.start(2)
            + parameter.start(2)
            + len(parameter.group(2))
            - len(parameter.group(2).lstrip())
        )
        value_spans.append((value_start, value_start + len(raw_value)))
    name_offset = len(function.group(1)) - len(function.group(1).lstrip())
    name_start = absolute_start + function.start(1) + name_offset
    arguments_span = None
    if value_spans:
        arguments_span = (value_spans[0][0], value_spans[-1][1])
    return ParsedToolCall(
        name=name,
        arguments=arguments,
        name_span=(name_start, name_start + len(name)),
        arguments_span=arguments_span,
    )


def parse_qwen_tool_calls(text: str) -> list[ParsedToolCall]:
    """Parse Qwen-style ``<tool_call>{...}</tool_call>`` output.

    A plain JSON object (or ``{"tool_calls": [...]}``) is accepted as a
    compatibility fallback for small checkpoints and test doubles.
    """
    parsed: list[ParsedToolCall] = []
    pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    for match in pattern.finditer(text):
        payload = match.group(1)
        call = _parsed_call_from_payload(payload, match.start(1))
        if call is None:
            call = _parsed_call_from_xml_payload(payload, match.start(1))
        if call is not None:
            parsed.append(call)
    if parsed:
        return parsed

    stripped = text.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return []
    values: list[Any]
    if isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
        values = value["tool_calls"]
    else:
        values = [value]
    for item in values:
        if not isinstance(item, dict):
            continue
        payload = _canonical_json(item)
        call = _parsed_call_from_payload(payload, 0)
        if call is not None:
            parsed.append(call)
    return parsed


def _stable_tool_call_id(index: int, call: ParsedToolCall) -> str:
    payload = f"{index}\0{call.name}\0{_canonical_json(call.arguments)}".encode()
    return f"call_{hashlib.sha256(payload).hexdigest()[:16]}"


def assistant_message_from_generation(
    text: str,
    *,
    generation_time_seconds: float,
    prompt_tokens: int,
    completion_tokens: int,
) -> tuple[AssistantMessage, list[ParsedToolCall]]:
    """Convert decoded local-model output into a τ² assistant message."""
    parsed = parse_qwen_tool_calls(text)
    tool_calls = [
        ToolCall(
            id=_stable_tool_call_id(index, call),
            name=call.name,
            arguments=call.arguments,
        )
        for index, call in enumerate(parsed)
    ]
    content = None if tool_calls else text.strip()
    return (
        AssistantMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls or None,
            cost=0.0,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            generation_time_seconds=generation_time_seconds,
        ),
        parsed,
    )


def messages_for_hf(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Convert τ² messages to the format consumed by HF chat templates."""
    converted: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, MultiToolMessage):
            converted.extend(messages_for_hf(message.tool_messages))
        elif isinstance(message, SystemMessage):
            converted.append({"role": "system", "content": message.content})
        elif isinstance(message, UserMessage):
            converted.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            item: dict[str, Any] = {
                "role": "assistant",
                "content": message.content,
            }
            if message.is_tool_call():
                item["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                    }
                    for call in message.tool_calls
                ]
            converted.append(item)
        elif isinstance(message, ToolMessage):
            converted.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        else:
            raise TypeError(f"Unsupported message type: {type(message).__name__}")
    return converted


def normalized_j_vector(jacobian: Any, unembedding_row: Any) -> Any:
    """Return the normalized ``J_l.T @ W_U[w]`` concept direction."""
    import torch

    direction = jacobian.float().T @ unembedding_row.float()
    norm = torch.linalg.vector_norm(direction)
    if not torch.isfinite(norm) or float(norm) == 0.0:
        raise ValueError("cannot normalize a zero or non-finite J-vector")
    return direction / norm


def finite_difference_token_effect(
    lens_model: Any,
    lens: Any,
    *,
    layer: int,
    token_id: int,
    residual: Any,
    epsilon: float = 1e-3,
) -> dict[str, float | bool]:
    """Check that positive J-vector steering raises the intended token locally."""
    import torch

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    weight = lens_model._lm_head.weight[token_id].detach().float().cpu()
    direction = normalized_j_vector(lens.jacobians[layer].cpu(), weight)
    center = residual.detach().float().cpu()

    def token_logit(point: Any) -> Any:
        transported = lens.transport(point, layer)
        return lens_model.unembed(transported)[..., token_id].float().cpu()

    with torch.no_grad():
        minus = token_logit(center - epsilon * direction)
        plus = token_logit(center + epsilon * direction)
    derivative = float(((plus - minus) / (2 * epsilon)).mean())
    return {
        "epsilon": float(epsilon),
        "central_difference": derivative,
        "positive": derivative > 0,
    }


def expanded_gqa_sdpa_forward(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    **kwargs: Any,
) -> tuple[Any, None]:
    """Run SDPA after explicitly expanding grouped KV heads.

    PyTorch's Windows efficient-SDPA kernel does not accept Qwen's native
    ``query_heads != kv_heads`` layout. Transformers normally selects
    ``enable_gqa=True`` when no attention mask is present, which falls back to
    the quadratic math kernel on that platform. Expanding the KV view first
    keeps the fused, memory-efficient kernel eligible.
    """
    from types import SimpleNamespace

    from transformers.integrations.sdpa_attention import repeat_kv
    from transformers.integrations.sdpa_attention import (
        sdpa_attention_forward as transformers_sdpa_forward,
    )

    groups = getattr(module, "num_key_value_groups", 1)
    if groups > 1 and key.shape[1] != query.shape[1]:
        key = repeat_kv(key, groups)
        value = repeat_kv(value, groups)
    proxy = SimpleNamespace(
        num_key_value_groups=1,
        is_causal=getattr(module, "is_causal", True),
    )
    return transformers_sdpa_forward(
        proxy,
        query,
        key,
        value,
        attention_mask,
        **kwargs,
    )


class JSONLTelemetryWriter:
    """Thread-safe append-only writer for hot telemetry artifacts."""

    def __init__(self, path: Path):
        self.path = path
        resolved = str(path.resolve())
        with _PATH_LOCKS_LOCK:
            self._lock = _PATH_LOCKS.setdefault(resolved, threading.Lock())

    def write(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")


def _bundle_cache_key(config: HFBackendConfig) -> tuple[Any, ...]:
    return (
        config.model_name_or_path,
        config.revision,
        config.tokenizer_revision,
        str(config.lens_path.resolve()) if config.lens_path else None,
        config.device,
        config.dtype,
        config.sdpa_backend,
        config.trust_remote_code,
    )


def _resolve_dtype(torch: Any, dtype: str, device: str) -> Any:
    if dtype != "auto":
        value = getattr(torch, dtype, None)
        if value is None:
            raise ValueError(f"unknown torch dtype: {dtype}")
        return value
    return torch.float16 if device.startswith("cuda") else torch.float32


def _load_bundle(config: HFBackendConfig) -> _ModelBundle:
    import jlens
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _resolve_dtype(torch, config.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        revision=config.tokenizer_revision or config.revision,
        trust_remote_code=config.trust_remote_code,
    )
    model_kwargs: dict[str, Any] = {}
    if config.sdpa_backend == "efficient":
        attention_name = "jlens_expanded_gqa_sdpa"
        ALL_ATTENTION_FUNCTIONS.register(attention_name, expanded_gqa_sdpa_forward)
        model_kwargs["attn_implementation"] = attention_name
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        revision=config.revision,
        dtype=dtype,
        trust_remote_code=config.trust_remote_code,
        **model_kwargs,
    )
    model.to(device)
    model.eval()
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    lens = (
        jlens.JacobianLens.load(str(config.lens_path))
        if config.lens_path is not None
        else None
    )
    if lens is not None and lens.d_model != lens_model.d_model:
        raise ValueError(
            f"lens d_model={lens.d_model} does not match model d_model="
            f"{lens_model.d_model}"
        )
    return _ModelBundle(
        model=model, tokenizer=tokenizer, lens_model=lens_model, lens=lens
    )


def _get_or_load_bundle(config: HFBackendConfig) -> _ModelBundle:
    key = _bundle_cache_key(config)
    with _BUNDLE_CACHE_LOCK:
        bundle = _BUNDLE_CACHE.get(key)
        if bundle is None:
            bundle = _load_bundle(config)
            _BUNDLE_CACHE[key] = bundle
        return bundle


def _replace_block_output(output: Any, tensor: Any) -> Any:
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    return tensor


def _find_span_token_indices(
    tokenizer: Any, token_ids: Sequence[int], span: Optional[tuple[int, int]]
) -> list[int]:
    if span is None:
        return []
    start, end = span
    indices: list[int] = []
    previous_length = 0
    for index in range(len(token_ids)):
        prefix = tokenizer.decode(token_ids[: index + 1], skip_special_tokens=False)
        current_length = len(prefix)
        if previous_length < end and current_length > start:
            indices.append(index)
        previous_length = current_length
    return indices


def semantic_prediction_positions(
    tokenizer: Any,
    prompt_length: int,
    generated_ids: Sequence[int],
    parsed_calls: Sequence[ParsedToolCall],
) -> dict[str, list[int]]:
    """Map tool-call character spans to exact next-token prediction positions.

    A position is the token whose residual predicts the following token.  The
    mapping is recorded even in ``off`` mode so the offline all-position
    analyzer can add semantic overlays without rerendering or retokenizing the
    conversation.
    """
    groups: dict[str, list[int]] = {}
    if prompt_length:
        groups["initial_decision"] = [prompt_length - 1]
    for index, call in enumerate(parsed_calls):
        name_tokens = _find_span_token_indices(tokenizer, generated_ids, call.name_span)
        argument_tokens = _find_span_token_indices(
            tokenizer, generated_ids, call.arguments_span
        )
        if name_tokens:
            groups[f"tool_{index}_name"] = [
                prompt_length + token_index - 1 for token_index in name_tokens
            ]
        if argument_tokens:
            groups[f"tool_{index}_arguments"] = [
                prompt_length + token_index - 1 for token_index in argument_tokens
            ]
    return groups


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


class InstrumentedHFBackend:
    """Shared model backend used by baseline, observer, and intervention agents."""

    def __init__(
        self,
        config: HFBackendConfig,
        *,
        bundle: Optional[_ModelBundle] = None,
    ):
        self.config = config
        self.bundle = bundle or _get_or_load_bundle(config)
        self.writer = (
            JSONLTelemetryWriter(config.telemetry_path)
            if config.telemetry_path is not None
            else None
        )
        self._concept_ids = self._resolve_concept_ids()
        self._concept_directions: dict[tuple[int, int], Any] = {}
        self._artifact_directions: dict[
            tuple[str, str], tuple[Any, dict[str, Any]]
        ] = {}
        self._cast_artifacts: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        self._mera_artifacts: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        self._sadi_artifacts: dict[
            tuple[str, Optional[int]], tuple[dict[str, Any], dict[str, Any]]
        ] = {}
        self._iti_artifacts: dict[
            tuple[str, Optional[int]], tuple[dict[str, Any], dict[str, Any]]
        ] = {}
        self._austeer_artifacts: dict[
            tuple[str, Optional[int]], tuple[dict[str, Any], dict[str, Any]]
        ] = {}
        self._jservo_artifacts: dict[str, dict[str, Any]] = {}

    def _jservo_artifact(self, controller: JServoConfig) -> dict[str, Any]:
        key = str(controller.artifact_path)
        cached = self._jservo_artifacts.get(key)
        if cached is None:
            cached = load_jservo_artifact(
                controller.artifact_path,
                expected_model_id=self.config.model_name_or_path,
                expected_model_revision=self.config.revision,
            )
            n_layers = self.bundle.lens_model.n_layers
            artifact_layers = {
                int(layer)
                for mode in cached["modes"].values()
                for layer in [*mode["observation_layers"], *mode["control_layers"]]
            }
            invalid = sorted(layer for layer in artifact_layers if not 0 <= layer < n_layers)
            if invalid:
                raise ValueError(f"J-Servo artifact layers are outside the model: {invalid}")
            self._jservo_artifacts[key] = cached
        return cached

    @contextmanager
    def _generation_control(self, *, turn_index: int, boundaries: Sequence[str]):
        """Install exactly one legacy intervention or adaptive J-Servo controller."""
        controller = self.config.controller
        if controller is not None:
            artifact = self._jservo_artifact(controller)
            with jservo_generation_hooks(
                self.bundle.lens_model.layers,
                artifact=artifact,
                config=controller,
                boundaries=boundaries,
            ) as controller_trace:
                yield (
                    {
                        "requested": False,
                        "active": False,
                        "reason": "adaptive_controller_selected",
                    },
                    controller_trace,
                )
            return
        intervention = self.config.intervention
        if intervention is None:
            active, reason = False, "not_configured"
        else:
            active, reason = self._intervention_is_active(
                intervention,
                turn_index=turn_index,
                boundaries=boundaries,
            )
        with self._intervention_hook(
            active=active,
            selection_reason=reason,
        ) as intervention_trace:
            yield (
                intervention_trace,
                {
                    "requested": False,
                    "active": False,
                    "reason": "not_configured",
                },
            )

    @classmethod
    def from_pretrained(cls, config: HFBackendConfig) -> "InstrumentedHFBackend":
        """Load or reuse the model/tokenizer/lens bundle."""
        return cls(config)

    def _resolve_concept_ids(self) -> dict[str, int]:
        resolved: dict[str, int] = {}
        for alias, value in self.config.concept_tokens.items():
            if isinstance(value, int):
                resolved[alias] = value
                continue
            token_ids = self.bundle.tokenizer.encode(value, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f"concept {alias!r} must resolve to exactly one token; "
                    f"{value!r} produced {token_ids}"
                )
            resolved[alias] = int(token_ids[0])
        return resolved

    def _selected_layers(self) -> tuple[int, ...]:
        n_layers = self.bundle.lens_model.n_layers
        if self.config.selected_layers:
            layers = self.config.selected_layers
        else:
            layers = tuple(sorted({max(0, n_layers // 4), n_layers // 2, n_layers - 1}))
        invalid = [layer for layer in layers if not 0 <= layer < n_layers]
        if invalid:
            raise ValueError(f"selected layers out of range: {invalid}")
        return layers

    def _render(
        self, messages: Sequence[Message], tools: Sequence[Tool]
    ) -> tuple[Any, Any, str, dict[str, Any]]:
        import torch

        hf_messages = messages_for_hf(messages)
        tools_schema = [tool.openai_schema for tool in tools]
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
            **self.config.chat_template_kwargs,
        }
        if tools_schema:
            kwargs["tools"] = tools_schema
        encoded = self.bundle.tokenizer.apply_chat_template(hf_messages, **kwargs)
        if hasattr(encoded, "input_ids"):
            input_ids = encoded.input_ids
            attention_mask = getattr(encoded, "attention_mask", None)
        elif isinstance(encoded, dict):
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
        else:
            input_ids = encoded
            attention_mask = None
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        original_tokens = int(input_ids.shape[1])
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        truncated = original_tokens > self.config.max_input_tokens
        if truncated:
            if self.bundle.tokenizer.truncation_side == "left":
                input_ids = input_ids[:, -self.config.max_input_tokens :]
                attention_mask = attention_mask[:, -self.config.max_input_tokens :]
            else:
                input_ids = input_ids[:, : self.config.max_input_tokens]
                attention_mask = attention_mask[:, : self.config.max_input_tokens]
        device = self.bundle.lens_model.input_device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        rendered = self.bundle.tokenizer.decode(
            input_ids[0].detach().cpu().tolist(), skip_special_tokens=False
        )
        return (
            input_ids,
            attention_mask,
            rendered,
            {
                "pre_truncation_tokens": original_tokens,
                "used_tokens": int(input_ids.shape[1]),
                "truncated": truncated,
                "truncation_side": self.bundle.tokenizer.truncation_side,
            },
        )

    def _intervention_direction(
        self, intervention: InterventionConfig
    ) -> tuple[Any, dict[str, Any]]:
        import torch

        if intervention.vector is not None:
            direction = torch.tensor(intervention.vector, dtype=torch.float32)
            source = {
                "source": "inline_vector",
                "vector_fingerprint": _tensor_sha256(direction),
            }
        elif intervention.vector_path is not None:
            if intervention.method == "cast":
                artifact, source = self._cast_artifact(intervention)
                return artifact["behavior_direction"].detach().float().cpu(), {
                    "source": "artifact",
                    "path": str(intervention.vector_path),
                    **source,
                }
            if intervention.method == "mera":
                artifact, source = self._mera_artifact(intervention)
                return artifact["probe_vector"].detach().float().cpu(), {
                    "source": "artifact",
                    "path": str(intervention.vector_path),
                    **source,
                }
            cache_key = (
                str(intervention.vector_path.resolve()),
                intervention.vector_scaling,
            )
            cached = self._artifact_directions.get(cache_key)
            if cached is None:
                cached = load_caa_direction_artifact(
                    intervention.vector_path,
                    model_id=self.config.model_name_or_path,
                    layer=(
                        intervention.artifact_layer
                        if intervention.artifact_layer is not None
                        else intervention.layer
                    ),
                    d_model=self.bundle.lens_model.d_model,
                    scaling=intervention.vector_scaling,
                )
                self._artifact_directions[cache_key] = cached
            direction, artifact_metadata = cached
            source = {
                "source": "artifact",
                "path": str(intervention.vector_path),
                **artifact_metadata,
            }
        elif intervention.concept_token_id is not None:
            if self.bundle.lens is None:
                raise ValueError("concept intervention requires a fitted lens")
            weight = (
                self.bundle.model.get_output_embeddings()
                .weight[intervention.concept_token_id]
                .detach()
                .float()
                .cpu()
            )
            direction = normalized_j_vector(
                self.bundle.lens.jacobians[intervention.layer].cpu(), weight
            )
            source = {
                "source": "jlens_concept",
                "concept_token_id": int(intervention.concept_token_id),
                "vector_fingerprint": _tensor_sha256(direction),
            }
        else:
            raise AssertionError("validated intervention has no direction source")
        if direction.numel() != self.bundle.lens_model.d_model:
            raise ValueError(
                f"intervention vector has {direction.numel()} elements; expected "
                f"{self.bundle.lens_model.d_model}"
            )
        return direction, source

    def _cast_artifact(
        self, intervention: InterventionConfig
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if intervention.vector_path is None:
            raise ValueError("CAST intervention requires vector_path")
        key = str(intervention.vector_path.resolve())
        cached = self._cast_artifacts.get(key)
        if cached is None:
            cached = load_cast_artifact(
                intervention.vector_path,
                model_id=self.config.model_name_or_path,
                behavior_layer=(
                    intervention.artifact_layer
                    if intervention.artifact_layer is not None
                    else intervention.layer
                ),
                d_model=self.bundle.lens_model.d_model,
            )
            self._cast_artifacts[key] = cached
        return cached

    def _mera_artifact(
        self, intervention: InterventionConfig
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if intervention.vector_path is None:
            raise ValueError("MERA intervention requires vector_path")
        key = str(intervention.vector_path.resolve())
        cached = self._mera_artifacts.get(key)
        if cached is None:
            cached = load_mera_artifact(
                intervention.vector_path,
                model_id=self.config.model_name_or_path,
                layer=(
                    intervention.artifact_layer
                    if intervention.artifact_layer is not None
                    else intervention.layer
                ),
                d_model=self.bundle.lens_model.d_model,
            )
            self._mera_artifacts[key] = cached
        return cached

    def _sadi_artifact(
        self, intervention: InterventionConfig
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if intervention.vector_path is None:
            raise ValueError("SADI intervention requires vector_path")
        key = (str(intervention.vector_path.resolve()), intervention.sadi_top_k)
        cached = self._sadi_artifacts.get(key)
        if cached is None:
            cached = load_sadi_artifact(
                intervention.vector_path,
                model_id=self.config.model_name_or_path,
                d_model=self.bundle.lens_model.d_model,
                top_k=intervention.sadi_top_k,
            )
            self._sadi_artifacts[key] = cached
        return cached

    def _iti_artifact(
        self, intervention: InterventionConfig
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if intervention.vector_path is None:
            raise ValueError("ITI intervention requires vector_path")
        key = (str(intervention.vector_path.resolve()), intervention.iti_top_k)
        cached = self._iti_artifacts.get(key)
        if cached is None:
            cached = load_iti_artifact(
                intervention.vector_path,
                model_id=self.config.model_name_or_path,
                d_model=self.bundle.lens_model.d_model,
                top_k=intervention.iti_top_k,
            )
            self._iti_artifacts[key] = cached
        return cached

    def _austeer_artifact(
        self, intervention: InterventionConfig
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if intervention.vector_path is None:
            raise ValueError("AUSteer intervention requires vector_path")
        key = (str(intervention.vector_path.resolve()), intervention.austeer_top_k)
        cached = self._austeer_artifacts.get(key)
        if cached is None:
            cached = load_austeer_artifact(
                intervention.vector_path,
                model_id=self.config.model_name_or_path,
                d_model=self.bundle.lens_model.d_model,
                top_k=intervention.austeer_top_k,
            )
            self._austeer_artifacts[key] = cached
        return cached

    @staticmethod
    def _intervention_is_active(
        intervention: InterventionConfig,
        *,
        turn_index: int,
        boundaries: Sequence[str],
    ) -> tuple[bool, str]:
        if intervention.turn_indices and turn_index not in intervention.turn_indices:
            return False, "turn_not_selected"
        if intervention.boundaries and not set(intervention.boundaries).intersection(
            boundaries
        ):
            return False, "boundary_not_selected"
        return True, "selected"

    @contextmanager
    def _intervention_hook(self, *, active: bool, selection_reason: str):
        intervention = self.config.intervention
        if intervention is None:
            yield {"requested": False, "active": False, "reason": "not_configured"}
            return
        trace: dict[str, Any] = {
            "requested": True,
            "active": bool(active),
            "reason": selection_reason,
            "method": intervention.method,
            "kind": intervention.kind,
            "layer": int(intervention.layer),
            "artifact_layer": intervention.artifact_layer,
            "strength": float(intervention.strength),
            "turn_indices": list(intervention.turn_indices),
            "trigger_boundaries": list(intervention.boundaries),
            "apply_prefill_decision": intervention.apply_prefill_decision,
            "apply_decode": intervention.apply_decode,
            "prefill_calls": 0,
            "decode_calls": 0,
            "applied_prefill_positions": 0,
            "applied_decode_positions": 0,
        }
        if not active:
            yield trace
            return
        n_layers = self.bundle.lens_model.n_layers
        if not 0 <= intervention.layer < n_layers:
            raise ValueError(
                f"intervention layer {intervention.layer} is outside [0, {n_layers})"
            )
        if intervention.method == "austeer":
            import torch

            if intervention.vector_path is not None:
                artifact, source = self._austeer_artifact(intervention)
                top_k = (
                    int(intervention.austeer_top_k)
                    if intervention.austeer_top_k is not None
                    else int(artifact["top_k"])
                )
                entries = [
                    (
                        int(artifact["selected_units"][index, 0]),
                        int(artifact["selected_units"][index, 1]),
                        float(artifact["selected_betas"][index]),
                    )
                    for index in range(top_k)
                ]
                direction_source = {
                    "source": "artifact",
                    "path": str(intervention.vector_path),
                    **source,
                }
            else:
                assert intervention.austeer_units is not None
                entries = list(intervention.austeer_units)
                top_k = (
                    int(intervention.austeer_top_k)
                    if intervention.austeer_top_k is not None
                    else len(entries)
                )
                if top_k != len(entries):
                    raise ValueError("inline AUSteer top_k must equal the supplied AU count")
                entry_tensor = torch.tensor(entries, dtype=torch.float32)
                direction_source = {
                    "source": "inline_control_aus",
                    "selected_units_fingerprint": _tensor_sha256(entry_tensor),
                    "requested_top_k": top_k,
                }
            grouped: dict[int, list[tuple[int, float]]] = {}
            for layer, dimension, beta in entries:
                grouped.setdefault(int(layer), []).append((int(dimension), float(beta)))
            if any(not 0 <= layer < n_layers for layer in grouped):
                raise ValueError("AUSteer selected layer is outside the model")
            if any(
                not 0 <= dimension < self.bundle.lens_model.d_model
                or not math.isfinite(beta)
                or abs(beta) > 1.0
                for values in grouped.values()
                for dimension, beta in values
            ):
                raise ValueError("AUSteer selected dimensions or beta values are invalid")
            calls = {layer: 0 for layer in grouped}
            sentinel_layer = min(grouped)
            trace.update(
                {
                    "layers": sorted(grouped),
                    "top_k": top_k,
                    "prefill_mode": intervention.austeer_prefill_mode,
                    "direction": direction_source,
                    "units_by_layer": {
                        str(layer): [dimension for dimension, _beta in values]
                        for layer, values in grouped.items()
                    },
                    "applied_prefill_scalars": 0,
                    "applied_decode_scalars": 0,
                }
            )
            handles = []

            def make_austeer_hook(layer: int, values: list[tuple[int, float]]):
                def austeer_hook(_module: Any, inputs: Any) -> Any:
                    hidden = inputs[0]
                    if hidden.ndim != 3:
                        raise ValueError("AUSteer attention output must be rank 3")
                    is_prefill = calls[layer] == 0
                    calls[layer] += 1
                    if layer == sentinel_layer:
                        trace["prefill_calls" if is_prefill else "decode_calls"] += 1
                    if (is_prefill and not intervention.apply_prefill_decision) or (
                        not is_prefill and not intervention.apply_decode
                    ):
                        return None
                    modified = hidden.clone()
                    dimensions = torch.tensor(
                        [dimension for dimension, _beta in values],
                        dtype=torch.long,
                        device=hidden.device,
                    )
                    betas = torch.tensor(
                        [beta for _dimension, beta in values],
                        dtype=hidden.dtype,
                        device=hidden.device,
                    )
                    if is_prefill and intervention.austeer_prefill_mode == "decision_only":
                        selected = modified[:, -1:, :].index_select(-1, dimensions)
                        modified[:, -1:, :].index_copy_(
                            -1,
                            dimensions,
                            selected * (1.0 + float(intervention.strength) * betas),
                        )
                        positions = int(modified.shape[0])
                    else:
                        selected = modified.index_select(-1, dimensions)
                        modified.index_copy_(
                            -1,
                            dimensions,
                            selected * (1.0 + float(intervention.strength) * betas),
                        )
                        positions = int(modified.shape[0] * modified.shape[1])
                    key = (
                        "applied_prefill_scalars"
                        if is_prefill
                        else "applied_decode_scalars"
                    )
                    trace[key] += positions * len(values)
                    return (modified, *inputs[1:])

                return austeer_hook

            try:
                for layer, values in grouped.items():
                    attention = getattr(
                        self.bundle.lens_model.layers[layer], "self_attn", None
                    )
                    projection = getattr(attention, "o_proj", None)
                    if projection is None:
                        raise ValueError(f"model layer {layer} has no attention o_proj")
                    handles.append(
                        projection.register_forward_pre_hook(
                            make_austeer_hook(layer, values)
                        )
                    )
                yield trace
            finally:
                for handle in reversed(handles):
                    handle.remove()
            return
        if intervention.method == "iti":
            import torch

            if intervention.vector_path is not None:
                artifact, source = self._iti_artifact(intervention)
                top_k = (
                    int(intervention.iti_top_k)
                    if intervention.iti_top_k is not None
                    else int(artifact["top_k"])
                )
                num_heads = int(artifact["num_attention_heads"])
                head_dim = int(artifact["head_dim"])
                entries = [
                    (
                        int(artifact["selected_heads"][index, 0]),
                        int(artifact["selected_heads"][index, 1]),
                        artifact["head_directions"][index],
                        float(artifact["projection_stds"][index]),
                    )
                    for index in range(top_k)
                ]
                direction_source = {
                    "source": "artifact",
                    "path": str(intervention.vector_path),
                    **source,
                }
            else:
                assert intervention.iti_entries is not None
                entries = [
                    (layer, head, torch.tensor(direction), scale)
                    for layer, head, direction, scale in intervention.iti_entries
                ]
                top_k = (
                    int(intervention.iti_top_k)
                    if intervention.iti_top_k is not None
                    else len(entries)
                )
                if top_k != len(entries):
                    raise ValueError("inline ITI top_k must equal the supplied head count")
                head_dims = {int(direction.numel()) for _layer, _head, direction, _scale in entries}
                if len(head_dims) != 1:
                    raise ValueError("inline ITI directions must share one head dimension")
                head_dim = next(iter(head_dims))
                if self.bundle.lens_model.d_model % head_dim:
                    raise ValueError("inline ITI head dimension does not divide model width")
                num_heads = self.bundle.lens_model.d_model // head_dim
                entry_tensor = torch.tensor(
                    [
                        [float(layer), float(head), float(scale), *direction.float().tolist()]
                        for layer, head, direction, scale in entries
                    ],
                    dtype=torch.float32,
                )
                direction_source = {
                    "source": "inline_control_heads",
                    "selected_heads_fingerprint": _tensor_sha256(entry_tensor),
                    "requested_top_k": top_k,
                }
            grouped: dict[int, list[tuple[int, Any, float]]] = {}
            for layer, head, direction, scale in entries:
                grouped.setdefault(int(layer), []).append(
                    (int(head), direction.detach().float().cpu(), float(scale))
                )
            if any(not 0 <= layer < n_layers for layer in grouped):
                raise ValueError("ITI selected layer is outside the model")
            if any(
                not 0 <= head < num_heads
                or int(direction.numel()) != head_dim
                or not bool(torch.isfinite(direction).all())
                or abs(float(direction.float().norm()) - 1.0) > 1e-5
                or not math.isfinite(scale)
                or scale <= 0.0
                for values in grouped.values()
                for head, direction, scale in values
            ):
                raise ValueError("ITI selected head directions or scales are invalid")
            calls = {layer: 0 for layer in grouped}
            sentinel_layer = min(grouped)
            trace.update(
                {
                    "layers": sorted(grouped),
                    "top_k": top_k,
                    "num_attention_heads": num_heads,
                    "head_dim": head_dim,
                    "direction": direction_source,
                    "heads_by_layer": {
                        str(layer): [head for head, _direction, _scale in values]
                        for layer, values in grouped.items()
                    },
                    "applied_prefill_heads": 0,
                    "applied_decode_heads": 0,
                }
            )
            handles = []

            def make_iti_hook(layer: int, values: list[tuple[int, Any, float]]):
                def iti_hook(_module: Any, inputs: Any) -> Any:
                    hidden = inputs[0]
                    if hidden.ndim != 3 or int(hidden.shape[-1]) != num_heads * head_dim:
                        raise ValueError("ITI o_proj input has an incompatible head shape")
                    is_prefill = calls[layer] == 0
                    calls[layer] += 1
                    if layer == sentinel_layer:
                        trace["prefill_calls" if is_prefill else "decode_calls"] += 1
                    if (is_prefill and not intervention.apply_prefill_decision) or (
                        not is_prefill and not intervention.apply_decode
                    ):
                        return None
                    modified = hidden.clone()
                    for head, direction, scale in values:
                        start = head * head_dim
                        end = start + head_dim
                        vector = direction.to(device=hidden.device, dtype=hidden.dtype)
                        modified[:, -1, start:end] += (
                            float(intervention.strength) * scale * vector
                        )
                    key = "applied_prefill_heads" if is_prefill else "applied_decode_heads"
                    trace[key] += int(modified.shape[0]) * len(values)
                    return (modified, *inputs[1:])

                return iti_hook

            try:
                for layer, values in grouped.items():
                    attention = getattr(
                        self.bundle.lens_model.layers[layer], "self_attn", None
                    )
                    projection = getattr(attention, "o_proj", None)
                    if projection is None:
                        raise ValueError(f"model layer {layer} has no attention o_proj")
                    handles.append(
                        projection.register_forward_pre_hook(make_iti_hook(layer, values))
                    )
                yield trace
            finally:
                for handle in reversed(handles):
                    handle.remove()
            return
        if intervention.method == "sadi":
            import torch

            if intervention.vector_path is not None:
                artifact, source = self._sadi_artifact(intervention)
                top_k = (
                    int(intervention.sadi_top_k)
                    if intervention.sadi_top_k is not None
                    else int(artifact["top_k"])
                )
                selected_units = artifact["selected_units"][:top_k]
                direction_source = {
                    "source": "artifact",
                    "path": str(intervention.vector_path),
                    **source,
                }
            else:
                selected_units = torch.tensor(
                    intervention.sadi_units,
                    dtype=torch.int64,
                )
                top_k = (
                    int(intervention.sadi_top_k)
                    if intervention.sadi_top_k is not None
                    else int(selected_units.shape[0])
                )
                if top_k != int(selected_units.shape[0]):
                    raise ValueError("inline SADI top_k must equal the supplied unit count")
                direction_source = {
                    "source": "inline_control_units",
                    "selected_units_fingerprint": _tensor_sha256(selected_units),
                    "requested_top_k": top_k,
                }
            units_by_layer: dict[int, list[int]] = {}
            for layer, dimension in selected_units.tolist():
                units_by_layer.setdefault(int(layer), []).append(int(dimension))
            invalid_layers = [
                layer for layer in units_by_layer if not 0 <= layer < n_layers
            ]
            if invalid_layers:
                raise ValueError(f"SADI artifact layers are outside the model: {invalid_layers}")
            if any(
                dimension >= self.bundle.lens_model.d_model
                for dimensions in units_by_layer.values()
                for dimension in dimensions
            ):
                raise ValueError("SADI selected dimension exceeds the model width")
            calls = {layer: 0 for layer in units_by_layer}
            trace.update(
                {
                    "layers": sorted(units_by_layer),
                    "top_k": top_k,
                    "direction": direction_source,
                    "units_by_layer": {
                        str(layer): dimensions
                        for layer, dimensions in units_by_layer.items()
                    },
                    "applied_prefill_scalars": 0,
                    "applied_decode_scalars": 0,
                    "mean_absolute_before": [],
                    "mean_absolute_after": [],
                }
            )
            handles = []
            sentinel_layer = min(units_by_layer)

            def make_sadi_hook(layer: int, dimensions: list[int]):
                def sadi_hook(_module: Any, _inputs: Any, output: Any) -> Any:
                    tensor = output if hasattr(output, "shape") else output[0]
                    if tensor.ndim != 3:
                        raise ValueError(
                            "SADI MLP output must have shape [batch, tokens, d_model]"
                        )
                    is_prefill = calls[layer] == 0
                    calls[layer] += 1
                    if layer == sentinel_layer:
                        trace["prefill_calls" if is_prefill else "decode_calls"] += 1
                    if (is_prefill and not intervention.apply_prefill_decision) or (
                        not is_prefill and not intervention.apply_decode
                    ):
                        return output
                    modified = tensor.clone()
                    index = torch.as_tensor(
                        dimensions,
                        device=modified.device,
                        dtype=torch.long,
                    )
                    before = modified[:, -1, :].index_select(-1, index)
                    after = before * float(intervention.strength)
                    modified[:, -1, :].index_copy_(-1, index, after)
                    key = (
                        "applied_prefill_scalars"
                        if is_prefill
                        else "applied_decode_scalars"
                    )
                    trace[key] += int(before.numel())
                    trace["mean_absolute_before"].append(
                        float(before.detach().float().abs().mean().cpu())
                    )
                    trace["mean_absolute_after"].append(
                        float(after.detach().float().abs().mean().cpu())
                    )
                    return _replace_block_output(output, modified)

                return sadi_hook

            try:
                for layer, dimensions in units_by_layer.items():
                    module = getattr(self.bundle.lens_model.layers[layer], "mlp", None)
                    if module is None:
                        raise ValueError(f"model layer {layer} has no MLP module for SADI")
                    handles.append(
                        module.register_forward_hook(make_sadi_hook(layer, dimensions))
                    )
                yield trace
            finally:
                for handle in reversed(handles):
                    handle.remove()
            return
        if intervention.method == "mera":
            if intervention.vector_path is not None:
                artifact, source = self._mera_artifact(intervention)
                probe = artifact["probe_vector"]
                artifact_layer = int(artifact["layer"])
                alpha = (
                    float(intervention.mera_alpha_override)
                    if intervention.mera_alpha_override is not None
                    else float(artifact["selected_alpha"])
                )
                direction_source = {
                    "source": "artifact",
                    "path": str(intervention.vector_path),
                    **source,
                }
            else:
                import torch

                probe = torch.tensor(intervention.vector, dtype=torch.float32)
                artifact_layer = intervention.artifact_layer
                alpha = float(intervention.mera_alpha_override)
                direction_source = {
                    "source": "inline_control_probe",
                    "probe_vector_fingerprint": _tensor_sha256(probe),
                }
            block = self.bundle.lens_model.layers[intervention.layer]
            module = getattr(block, "post_attention_layernorm", None)
            if module is None:
                raise ValueError(
                    f"model layer {intervention.layer} has no post_attention_layernorm"
                )
            call_index = 0
            trace.update(
                {
                    "artifact_layer": artifact_layer,
                    "direction": direction_source,
                    "alpha": alpha,
                    "mera_prefill_mode": intervention.mera_prefill_mode,
                    "eligible_prefill_positions": 0,
                    "eligible_decode_positions": 0,
                    "prefill_error_probability_mean": None,
                    "prefill_error_probability_max": None,
                    "decode_error_probability_mean": [],
                    "decode_error_probability_max": [],
                }
            )

            def mera_hook(_module: Any, _inputs: Any, output: Any) -> Any:
                nonlocal call_index
                tensor = output if hasattr(output, "shape") else output[0]
                is_prefill = call_index == 0
                call_index += 1
                call_key = "prefill_calls" if is_prefill else "decode_calls"
                trace[call_key] += 1
                if not is_prefill and not intervention.apply_decode:
                    return output
                modified = tensor.clone()
                if is_prefill and intervention.mera_prefill_mode == "decision_only":
                    selected = modified[:, -1:, :]
                else:
                    selected = modified
                delta, condition, scores = mera_closed_form_delta(
                    selected,
                    probe,
                    alpha=alpha,
                )
                delta = delta * float(intervention.strength)
                if is_prefill and intervention.mera_prefill_mode == "decision_only":
                    modified[:, -1:, :] = selected + delta
                else:
                    modified = selected + delta
                eligible_key = (
                    "eligible_prefill_positions"
                    if is_prefill
                    else "eligible_decode_positions"
                )
                applied_key = (
                    "applied_prefill_positions"
                    if is_prefill
                    else "applied_decode_positions"
                )
                trace[eligible_key] += int(condition.numel())
                trace[applied_key] += int(condition.sum().detach().cpu())
                mean_score = float(scores.mean().detach().float().cpu())
                max_score = float(scores.max().detach().float().cpu())
                if is_prefill:
                    trace["prefill_error_probability_mean"] = mean_score
                    trace["prefill_error_probability_max"] = max_score
                else:
                    trace["decode_error_probability_mean"].append(mean_score)
                    trace["decode_error_probability_max"].append(max_score)
                return _replace_block_output(output, modified)

            handle = module.register_forward_hook(mera_hook)
            try:
                yield trace
            finally:
                handle.remove()
            return
        if intervention.method == "cast":
            artifact, source = self._cast_artifact(intervention)
            condition_layer = int(artifact["condition_layer"])
            if not 0 <= condition_layer < n_layers:
                raise ValueError(
                    f"CAST condition layer {condition_layer} is outside [0, {n_layers})"
                )
            if condition_layer > intervention.layer:
                raise ValueError("CAST condition layer must not follow behavior layer")
            comparator = (
                intervention.cast_comparator_override
                or str(artifact["condition_comparator"])
            )
            if intervention.cast_invert_comparator:
                comparator = "less" if comparator == "greater" else "greater"
            threshold = float(artifact["condition_threshold"])
            comparison_mode = str(artifact["condition_comparison_mode"])
            condition_direction = artifact["condition_direction"]
            behavior_direction = artifact["behavior_direction"]
            condition_calls = 0
            behavior_calls = 0
            gate_triggered: Optional[bool] = None
            trace.update(
                {
                    "artifact_layer": int(artifact["behavior_layer"]),
                    "direction": {
                        "source": "artifact",
                        "path": str(intervention.vector_path),
                        **source,
                    },
                    "condition_layer": condition_layer,
                    "behavior_layer": int(intervention.layer),
                    "condition_threshold": threshold,
                    "condition_comparator": comparator,
                    "condition_comparison_mode": comparison_mode,
                    "cast_prefill_mode": intervention.cast_prefill_mode,
                    "cast_gate_override": intervention.cast_gate_override,
                    "cast_invert_comparator": intervention.cast_invert_comparator,
                    "condition_score": None,
                    "natural_gate_triggered": None,
                    "gate_triggered": None,
                    "condition_calls": 0,
                }
            )

            def condition_hook(_module: Any, inputs: Any) -> None:
                nonlocal condition_calls, gate_triggered
                condition_calls += 1
                trace["condition_calls"] = condition_calls
                if condition_calls != 1:
                    return None
                hidden = inputs[0]
                if hidden.ndim != 3 or hidden.shape[0] != 1:
                    raise ValueError("CAST generation requires one rank-3 prompt batch")
                score = cast_condition_similarity(
                    hidden[0],
                    condition_direction,
                    comparison_mode=comparison_mode,
                )
                numeric_score = float(score.detach().float().cpu())
                natural_gate = (
                    numeric_score > threshold
                    if comparator == "greater"
                    else numeric_score < threshold
                )
                gate_triggered = (
                    natural_gate
                    if intervention.cast_gate_override is None
                    else bool(intervention.cast_gate_override)
                )
                trace["condition_score"] = numeric_score
                trace["natural_gate_triggered"] = natural_gate
                trace["gate_triggered"] = gate_triggered
                return None

            def behavior_hook(_module: Any, inputs: Any) -> Any:
                nonlocal behavior_calls
                is_prefill = behavior_calls == 0
                behavior_calls += 1
                call_key = "prefill_calls" if is_prefill else "decode_calls"
                trace[call_key] += 1
                if gate_triggered is None:
                    raise RuntimeError("CAST behavior layer ran before its condition gate")
                if not gate_triggered or (not is_prefill and not intervention.apply_decode):
                    return None
                hidden = inputs[0]
                modified = hidden.clone()
                vector = behavior_direction.to(modified.device, dtype=modified.dtype)
                if is_prefill and intervention.cast_prefill_mode == "decision_only":
                    modified[:, -1, :] += intervention.strength * vector
                    applied = int(modified.shape[0])
                else:
                    modified += intervention.strength * vector
                    applied = int(modified.shape[0] * modified.shape[1])
                dose_key = (
                    "applied_prefill_positions"
                    if is_prefill
                    else "applied_decode_positions"
                )
                trace[dose_key] += applied
                return (modified, *inputs[1:])

            condition_handle = self.bundle.lens_model.layers[
                condition_layer
            ].register_forward_pre_hook(condition_hook)
            behavior_handle = self.bundle.lens_model.layers[
                intervention.layer
            ].register_forward_pre_hook(behavior_hook)
            try:
                yield trace
            finally:
                behavior_handle.remove()
                condition_handle.remove()
            return
        block = self.bundle.lens_model.layers[intervention.layer]
        direction, source = self._intervention_direction(intervention)
        trace["direction"] = source
        call_index = 0

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            nonlocal call_index
            tensor = output if hasattr(output, "shape") else output[0]
            is_prefill = call_index == 0
            call_index += 1
            if is_prefill:
                trace["prefill_calls"] += 1
                should_apply = intervention.apply_prefill_decision
            else:
                trace["decode_calls"] += 1
                should_apply = intervention.apply_decode
            if not should_apply:
                return output
            modified = tensor.clone()
            current = modified[:, -1, :]
            vector = direction.to(current.device, dtype=current.dtype)
            if intervention.kind == "steer":
                current = current + intervention.strength * vector
            elif intervention.kind == "ablate":
                vector_norm = vector.norm()
                if not vector_norm.isfinite() or float(vector_norm) == 0.0:
                    raise ValueError("cannot ablate a zero or non-finite direction")
                vector = vector / vector_norm
                trace["ablation_direction_normalized"] = True
                projection = (current * vector).sum(dim=-1, keepdim=True) * vector
                current = current - intervention.strength * projection
            elif intervention.kind == "patch":
                current = current + intervention.strength * (vector - current)
            else:
                raise ValueError(f"unknown intervention kind: {intervention.kind}")
            modified[:, -1, :] = current
            dose_key = (
                "applied_prefill_positions"
                if is_prefill
                else "applied_decode_positions"
            )
            trace[dose_key] += int(modified.shape[0])
            return _replace_block_output(output, modified)

        handle = block.register_forward_hook(hook)
        try:
            yield trace
        finally:
            handle.remove()

    @contextmanager
    def _sdpa_kernel(self):
        if self.config.sdpa_backend == "auto":
            yield
            return
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backends = {
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "flash": SDPBackend.FLASH_ATTENTION,
            "math": SDPBackend.MATH,
        }
        with sdpa_kernel(backends[self.config.sdpa_backend]):
            yield

    def _concept_scores(self, residual: Any, layer: int) -> dict[str, float]:
        if self.bundle.lens is None or layer not in self.bundle.lens.jacobians:
            return {}
        scores: dict[str, float] = {}
        for alias, token_id in self._concept_ids.items():
            cache_key = (layer, token_id)
            direction = self._concept_directions.get(cache_key)
            if direction is None:
                weight_row = (
                    self.bundle.model.get_output_embeddings()
                    .weight[token_id]
                    .detach()
                    .float()
                    .cpu()
                )
                direction = normalized_j_vector(
                    self.bundle.lens.jacobians[layer].cpu(), weight_row
                )
                self._concept_directions[cache_key] = direction
            scores[alias] = float(residual.float().cpu() @ direction)
        return scores

    def _teacher_forced_measurement(
        self,
        *,
        prompt_ids: Any,
        generated_ids: list[int],
        position_groups: dict[str, list[int]],
    ) -> dict[str, Any]:
        import torch
        from jlens.hooks import ActivationRecorder

        if not generated_ids:
            return {
                "positions": position_groups,
                "residuals": [],
                "motorization": {},
            }
        generated = torch.tensor(
            [generated_ids], device=prompt_ids.device, dtype=prompt_ids.dtype
        )
        full_ids = torch.cat([prompt_ids, generated], dim=1)
        final_layer = self.bundle.lens_model.n_layers - 1
        layers = tuple(sorted(set(self._selected_layers()) | {final_layer}))

        with (
            torch.no_grad(),
            self._sdpa_kernel(),
            ActivationRecorder(self.bundle.lens_model.layers, at=layers) as recorder,
        ):
            self.bundle.lens_model.forward(full_ids)
        full_ids_cpu = full_ids.detach().cpu()[0]

        residual_records: list[dict[str, Any]] = []
        for label, positions in position_groups.items():
            sampled = sorted({positions[0], positions[-1]})
            for position in sampled:
                if position < 0 or position >= full_ids.shape[1]:
                    continue
                for layer in layers:
                    residual = (
                        recorder.activations[layer][0, position].detach().float().cpu()
                    )
                    residual_records.append(
                        {
                            "label": label,
                            "position": position,
                            "layer": layer,
                            "l2_norm": float(torch.linalg.vector_norm(residual)),
                            "concept_scores": self._concept_scores(residual, layer),
                            "vector": residual.tolist(),
                        }
                    )

        motorization: dict[str, Any] = {}
        for label, positions in position_groups.items():
            if label == "initial_decision":
                continue
            final_logprobs: list[float] = []
            lens_logits: dict[int, list[float]] = {
                layer: []
                for layer in layers
                if self.bundle.lens is not None and layer in self.bundle.lens.jacobians
            }
            for position in positions:
                target_position = position + 1
                if not 0 <= position < full_ids.shape[1]:
                    continue
                if not 0 <= target_position < len(full_ids_cpu):
                    continue
                token_id = int(full_ids_cpu[target_position])
                final_residual = (
                    recorder.activations[final_layer][0, position].detach().float()
                )
                final_logits = self.bundle.lens_model.unembed(final_residual)
                final_logprobs.append(
                    float(
                        torch.log_softmax(final_logits.float(), dim=-1)[token_id]
                        .detach()
                        .cpu()
                    )
                )
                for layer in lens_logits:
                    residual = (
                        recorder.activations[layer][0, position].detach().float().cpu()
                    )
                    transported = self.bundle.lens.transport(residual, layer)
                    value = self.bundle.lens_model.unembed(transported)[token_id]
                    lens_logits[layer].append(float(value.detach().float().cpu()))
            motorization[label] = {
                "prediction_positions": positions,
                "mean_final_token_logprob": _mean(final_logprobs),
                "mean_jlens_target_logit": {
                    str(layer): _mean(values) for layer, values in lens_logits.items()
                },
            }
        return {
            "positions": position_groups,
            "residuals": residual_records,
            "motorization": motorization,
        }

    def generate(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[Tool],
        task_id: str,
        turn_index: int,
        boundaries: Sequence[str] = (),
        stop_tool_name: str = "done",
    ) -> BackendGeneration:
        """Generate one assistant turn and optionally collect telemetry."""
        import torch

        (
            input_ids,
            attention_mask,
            rendered_context,
            rendering_metadata,
        ) = self._render(messages, tools)
        prompt_ids = input_ids.detach().cpu()[0].tolist()
        generation_kwargs = _generation_kwargs_for_tokenizer(
            self.config, self.bundle.tokenizer
        )
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        start = time.perf_counter()
        with (
            self._sdpa_kernel(),
            self._generation_control(
                turn_index=turn_index,
                boundaries=boundaries,
            ) as control_traces,
        ):
            generated = self.bundle.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        duration = time.perf_counter() - start
        intervention_trace, controller_trace = control_traces
        sequences = (
            generated.sequences if hasattr(generated, "sequences") else generated
        )
        completion = sequences[0, input_ids.shape[1] :].detach().cpu().tolist()
        raw_text = self.bundle.tokenizer.decode(completion, skip_special_tokens=False)
        clean_text = self.bundle.tokenizer.decode(completion, skip_special_tokens=True)
        message, parsed_calls = assistant_message_from_generation(
            raw_text,
            generation_time_seconds=duration,
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(completion),
        )
        if not parsed_calls and clean_text != raw_text:
            message, parsed_calls = assistant_message_from_generation(
                clean_text,
                generation_time_seconds=duration,
                prompt_tokens=len(prompt_ids),
                completion_tokens=len(completion),
            )
        observed_boundaries = list(boundaries)
        if parsed_calls:
            observed_boundaries.append("pre_tool_call")
        if any(call.name == stop_tool_name for call in parsed_calls):
            observed_boundaries.append("candidate_stop")

        position_groups = semantic_prediction_positions(
            self.bundle.tokenizer,
            len(prompt_ids),
            completion,
            parsed_calls,
        )

        measurement: dict[str, Any] = {}
        if self.config.mode in {
            InstrumentationMode.OBSERVE,
            InstrumentationMode.INTERVENE,
        }:
            measurement = self._teacher_forced_measurement(
                prompt_ids=input_ids,
                generated_ids=completion,
                position_groups=position_groups,
            )
        full_ids_hash = token_ids_sha256([*prompt_ids, *completion])
        record = {
            "schema_version": "tau2-jlens-v2",
            "timestamp": _utc_now(),
            "task_id": task_id,
            "turn_index": turn_index,
            "mode": self.config.mode.value,
            "boundaries": sorted(set(observed_boundaries)),
            "model": {
                "name_or_path": self.config.model_name_or_path,
                "requested_revision": self.config.revision,
                "resolved_revision": getattr(
                    self.bundle.model.config, "_commit_hash", None
                ),
                "requested_tokenizer_revision": self.config.tokenizer_revision
                or self.config.revision,
                "resolved_tokenizer_revision": self.bundle.tokenizer.init_kwargs.get(
                    "_commit_hash"
                ),
            },
            "decoding": self.config.generation_kwargs,
            "sdpa_backend": self.config.sdpa_backend,
            "rendering": rendering_metadata,
            "rendered_context": rendered_context,
            "input_ids": prompt_ids,
            "input_ids_sha256": token_ids_sha256(prompt_ids),
            "generated_ids": completion,
            "generated_ids_sha256": token_ids_sha256(completion),
            "full_ids_sha256": full_ids_hash,
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(completion),
            "generated_text": raw_text,
            "tool_calls": [
                {
                    "name": call.name,
                    "arguments": call.arguments,
                    "name_span": call.name_span,
                    "arguments_span": call.arguments_span,
                }
                for call in parsed_calls
            ],
            "intervention": intervention_trace,
            "controller": controller_trace,
            "semantic_positions": position_groups,
            "measurement": measurement,
        }
        record_id = hashlib.sha256(
            f"{task_id}\0{turn_index}\0{record['input_ids_sha256']}\0"
            f"{record['generated_ids_sha256']}".encode()
        ).hexdigest()[:20]
        record["record_id"] = record_id
        message.raw_data = {
            "provider": "local_hf",
            "jlens_mode": self.config.mode.value,
            "telemetry_record_id": record_id,
            "input_ids_sha256": record["input_ids_sha256"],
            "generated_ids_sha256": record["generated_ids_sha256"],
        }
        if self.writer is not None:
            self.writer.write(record)
        return BackendGeneration(
            message=message,
            prompt_input_ids=prompt_ids,
            generated_ids=completion,
            rendered_text=raw_text,
            telemetry_record=record,
        )
