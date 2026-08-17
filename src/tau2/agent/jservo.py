"""Portable J-Servo controller and pre-execution validation for Tau2.

The artifact is produced by ``jlens-causal-steering`` but this runtime module
has no import-time dependency on that research repository.  This keeps remote
workers reproducible and prevents a silent fallback to a different controller.
"""

from __future__ import annotations

import json
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Literal, Optional, Sequence

from pydantic import ValidationError

from tau2.data_model.message import AssistantMessage, Message
from tau2.environment.tool import Tool

JSERVO_ARTIFACT_SCHEMA = "jlens-jservo-v1"
JSERVO_CONTROLLER_VERSION = "failure-mode-adaptive-v1"
JSERVO_TRACE_SCHEMA = "jlens-jservo-trace-v1"

ControlType = Literal[
    "targeted",
    "fixed_strength",
    "fixed_layer",
    "wrong_mode",
    "random",
    "reverse",
    "validator_only",
]

_VALID_CONTROLS = {
    "targeted",
    "fixed_strength",
    "fixed_layer",
    "wrong_mode",
    "random",
    "reverse",
    "validator_only",
}
_IDENTIFIER_FIELD = re.compile(r"(?:^id$|_id$|identifier$)", re.IGNORECASE)


@dataclass(frozen=True)
class JServoConfig:
    """Runtime-only settings; learned quantities live in the pinned artifact."""

    artifact_path: PurePath
    control_type: ControlType = "targeted"
    mode_override: Optional[str] = None
    layer_override: Optional[int] = None
    fixed_strength: Optional[float] = None
    apply_prefill_decision: bool = True
    apply_decode: bool = True
    validate_schema: bool = True
    require_identifier_provenance: bool = True
    block_repeated_calls: bool = True
    failure_action: Literal["abstain"] = "abstain"

    def __post_init__(self) -> None:
        if self.control_type not in _VALID_CONTROLS:
            raise ValueError(f"unknown J-Servo control_type {self.control_type!r}")
        if self.layer_override is not None and self.layer_override < 0:
            raise ValueError("J-Servo layer_override must be non-negative")
        if self.control_type == "fixed_layer" and self.layer_override is None:
            raise ValueError("fixed_layer J-Servo control requires layer_override")
        if self.control_type == "fixed_strength" and (
            self.fixed_strength is None or not math.isfinite(self.fixed_strength)
        ):
            raise ValueError("fixed_strength J-Servo control requires a finite strength")
        if self.failure_action != "abstain":
            raise ValueError("J-Servo currently permits only failure_action='abstain'")

    @classmethod
    def from_dict(cls, value: Optional[dict[str, Any]]) -> Optional["JServoConfig"]:
        if value is None:
            return None
        copied = dict(value)
        path = copied.get("artifact_path")
        if not isinstance(path, (str, PurePath)) or not str(path):
            raise ValueError("J-Servo artifact_path is required")
        copied["artifact_path"] = Path(str(path))
        if copied.get("layer_override") is not None:
            copied["layer_override"] = int(copied["layer_override"])
        if copied.get("fixed_strength") is not None:
            copied["fixed_strength"] = float(copied["fixed_strength"])
        return cls(**copied)


def validate_jservo_artifact(
    artifact: dict[str, Any],
    *,
    expected_model_id: Optional[str] = None,
    expected_model_revision: Optional[str] = None,
) -> dict[str, Any]:
    """Reject stale, malformed, or cross-model controller artifacts."""
    import torch

    if artifact.get("schema_version") != JSERVO_ARTIFACT_SCHEMA:
        raise ValueError("unsupported J-Servo artifact schema")
    if artifact.get("controller_version") != JSERVO_CONTROLLER_VERSION:
        raise ValueError("unsupported J-Servo controller version")
    if expected_model_id is not None and artifact.get("model_id") != expected_model_id:
        raise ValueError("J-Servo artifact model_id mismatch")
    if (
        expected_model_revision is not None
        and artifact.get("model_revision") != expected_model_revision
    ):
        raise ValueError("J-Servo artifact model_revision mismatch")
    modes = artifact.get("modes")
    if not isinstance(modes, dict) or not modes:
        raise ValueError("J-Servo artifact modes are missing")
    for name, mode in modes.items():
        if not isinstance(mode, dict) or mode.get("mode") != name:
            raise ValueError("J-Servo artifact mode key mismatch")
        if not mode.get("observation_layers") or not mode.get("control_layers"):
            raise ValueError(f"J-Servo mode {name} is missing layer roles")
        layers = mode.get("layers")
        if not isinstance(layers, dict):
            raise ValueError(f"J-Servo mode {name} has no layer payloads")
        for layer in [*mode["observation_layers"], *mode["control_layers"]]:
            payload = layers.get(str(layer))
            if not isinstance(payload, dict):
                raise ValueError(f"J-Servo mode {name} is missing layer {layer}")
            for field in ("margin_direction", "projected_direction", "random_direction"):
                tensor = payload.get(field)
                if (
                    tensor is None
                    or getattr(tensor, "ndim", None) != 1
                    or not bool(torch.isfinite(tensor).all())
                ):
                    raise ValueError(
                        f"J-Servo mode {name} layer {layer} has invalid {field}"
                    )
            if float(payload.get("dose_cap", 0.0)) <= 0:
                raise ValueError(f"J-Servo mode {name} layer {layer} has invalid dose cap")
    return artifact


def load_jservo_artifact(
    path: PurePath,
    *,
    expected_model_id: Optional[str] = None,
    expected_model_revision: Optional[str] = None,
) -> dict[str, Any]:
    import torch

    resolved = Path(path)
    try:
        artifact = torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError:
        artifact = torch.load(resolved, map_location="cpu")
    if not isinstance(artifact, dict):
        raise ValueError("J-Servo artifact must be an object")
    return validate_jservo_artifact(
        artifact,
        expected_model_id=expected_model_id,
        expected_model_revision=expected_model_revision,
    )


def _unit(vector: Any) -> Any:
    value = vector.detach().float()
    norm = value.norm()
    if not bool(__import__("torch").isfinite(norm)) or float(norm) <= 1e-12:
        raise ValueError("J-Servo direction has zero or non-finite norm")
    return value / norm


def minimum_state_edit(
    current: Any,
    *,
    margin_direction: Any,
    projected_direction: Any,
    target_margin: float,
    dose_cap: float,
    cumulative_dose: float,
    cumulative_cap: float,
) -> dict[str, Any]:
    """Compute the minimum protected-space edit for one layer and position."""
    import torch

    point = current.detach().float()
    read = margin_direction.to(device=point.device, dtype=point.dtype)
    write = projected_direction.to(device=point.device, dtype=point.dtype)
    pre_margin = float((point @ read).detach().cpu())
    deficit = max(0.0, float(target_margin) - pre_margin)
    denominator = float((read @ write).detach().cpu())
    result = {
        "pre_margin": pre_margin,
        "target_margin": float(target_margin),
        "deficit": deficit,
        "denominator": denominator,
        "dose_norm": 0.0,
        "predicted_post_margin": pre_margin,
        "feasible": True,
        "reason": "target_already_reached" if deficit == 0 else "selected",
        "delta": torch.zeros_like(point),
    }
    if deficit == 0:
        return result
    if not math.isfinite(denominator) or denominator <= 1e-12:
        result.update(feasible=False, reason="non_positive_control_gain")
        return result
    delta = deficit / denominator * write
    dose = float(delta.norm().detach().cpu())
    if not math.isfinite(dose) or dose > float(dose_cap) + 1e-9:
        result.update(feasible=False, reason="layer_dose_cap_exceeded", dose_norm=dose)
        return result
    if float(cumulative_dose) + dose > float(cumulative_cap) + 1e-9:
        result.update(feasible=False, reason="cumulative_dose_cap_exceeded", dose_norm=dose)
        return result
    result.update(
        delta=delta,
        dose_norm=dose,
        predicted_post_margin=pre_margin + float((delta @ read).detach().cpu()),
    )
    return result


def _replace_output(original: Any, tensor: Any) -> Any:
    if hasattr(original, "shape"):
        return tensor
    if isinstance(original, tuple):
        return (tensor, *original[1:])
    if isinstance(original, list):
        return [tensor, *original[1:]]
    raise TypeError(f"unsupported transformer block output {type(original).__name__}")


@contextmanager
def jservo_generation_hooks(
    blocks: Any,
    *,
    artifact: dict[str, Any],
    config: JServoConfig,
    boundaries: Sequence[str],
):
    """Observe at early layers and spend only the remaining error downstream."""
    validate_jservo_artifact(artifact)
    boundary_set = set(map(str, boundaries))
    candidates = {
        name: mode
        for name, mode in artifact["modes"].items()
        if boundary_set.intersection(mode.get("boundaries") or ())
        and (
            config.mode_override is None
            or name == config.mode_override
            or config.control_type == "wrong_mode"
        )
    }
    if config.control_type == "wrong_mode" and config.mode_override is not None:
        candidates = (
            {config.mode_override: artifact["modes"][config.mode_override]}
            if config.mode_override in artifact["modes"]
            else {}
        )
    layers = sorted(
        {
            int(layer)
            for mode in candidates.values()
            for layer in [*mode["observation_layers"], *mode["control_layers"]]
        }
    )
    trace: dict[str, Any] = {
        "schema_version": JSERVO_TRACE_SCHEMA,
        "requested": True,
        "active": False,
        "reason": "no_boundary_matched" if not candidates else "monitoring",
        "artifact_fingerprint": artifact["artifact_fingerprint"],
        "control_type": config.control_type,
        "boundaries": sorted(boundary_set),
        "candidate_modes": sorted(candidates),
        "selected_modes": [],
        "sites": [],
        "applied_positions": 0,
        "cumulative_dose": 0.0,
        "abstain_requested": False,
        "abstain_reasons": [],
        "controller_abstained": not bool(candidates),
    }
    if not layers:
        yield trace
        return

    calls = {layer: 0 for layer in layers}
    selected_by_site: dict[int, str] = {}
    active_layers_by_site: dict[int, int] = {}
    cumulative_by_mode = {name: 0.0 for name in candidates}
    handles = []

    def make_hook(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            tensor = output if hasattr(output, "shape") else output[0]
            site = calls[layer]
            calls[layer] += 1
            is_prefill = site == 0
            should_apply = (
                config.apply_prefill_decision if is_prefill else config.apply_decode
            )
            current = tensor[:, -1, :]
            observing = [
                (name, mode)
                for name, mode in candidates.items()
                if layer in set(map(int, mode["observation_layers"]))
            ]
            if observing:
                scored = []
                for name, mode in observing:
                    payload = mode["layers"][str(layer)]
                    direction = payload["margin_direction"].to(
                        device=current.device, dtype=current.dtype
                    )
                    margin = float((current[0] @ direction).detach().float().cpu())
                    if config.control_type == "reverse":
                        deficit = (
                            margin - float(payload.get("reverse_gate_threshold", margin))
                        ) / max(float(payload["margin_scale"]), 1e-6)
                        gate_threshold = float(
                            payload.get("reverse_gate_threshold", payload["gate_threshold"])
                        )
                    else:
                        deficit = (float(payload["gate_threshold"]) - margin) / max(
                            float(payload["margin_scale"]), 1e-6
                        )
                        gate_threshold = float(payload["gate_threshold"])
                    triggered = bool(deficit > 0 and mode["steering_eligible"])
                    trace["sites"].append(
                        {
                            "site": site,
                            "phase": "prefill" if is_prefill else "decode",
                            "layer": layer,
                            "mode": name,
                            "role": "observe",
                            "margin": margin,
                            "gate_threshold": gate_threshold,
                            "standardized_deficit": deficit,
                            "triggered": triggered,
                        }
                    )
                    if triggered:
                        scored.append((deficit, name))
                if scored:
                    selected_by_site[site] = max(scored)[1]

            selected = selected_by_site.get(site)
            if selected is None or selected not in candidates:
                return output
            mode = candidates[selected]
            if layer not in set(map(int, mode["control_layers"])):
                return output
            if config.layer_override is not None and layer != config.layer_override:
                return output
            if not should_apply or config.control_type == "validator_only":
                return output
            if active_layers_by_site.get(site, 0) >= int(
                mode["max_active_layers_per_position"]
            ):
                return output
            payload = mode["layers"][str(layer)]
            reverse = (
                config.control_type == "reverse" and "reverse_target_margin" in payload
            )
            result = minimum_state_edit(
                current[0],
                margin_direction=(
                    -payload["margin_direction"] if reverse else payload["margin_direction"]
                ),
                projected_direction=(
                    -payload["projected_direction"]
                    if reverse
                    else payload["projected_direction"]
                ),
                target_margin=float(
                    payload["reverse_target_margin"]
                    if reverse
                    else payload["target_margin"]
                ),
                dose_cap=float(payload["dose_cap"]),
                cumulative_dose=cumulative_by_mode[selected],
                cumulative_cap=float(mode["cumulative_dose_cap"]),
            )
            if config.control_type == "fixed_strength":
                direction = _unit(payload["projected_direction"]).to(
                    device=current.device, dtype=current.dtype
                )
                result["delta"] = (
                    float(config.fixed_strength)
                    * float(payload["residual_scale"])
                    * direction
                )
                result["dose_norm"] = float(result["delta"].norm().detach().cpu())
                result["feasible"] = result["dose_norm"] <= float(payload["dose_cap"])
                result["reason"] = (
                    "selected" if result["feasible"] else "layer_dose_cap_exceeded"
                )
            elif config.control_type == "reverse" and not reverse:
                result["delta"] = -result["delta"]
            if config.control_type in {"fixed_strength", "random"} or (
                config.control_type == "reverse" and not reverse
            ):
                read_direction = (
                    -payload["margin_direction"]
                    if reverse
                    else payload["margin_direction"]
                ).to(device=current.device, dtype=current.dtype)
                result["predicted_post_margin"] = float(
                    result["pre_margin"]
                    + (
                        result["delta"].to(
                            device=current.device, dtype=current.dtype
                        )
                        @ read_direction
                    )
                    .detach()
                    .cpu()
                )
            elif config.control_type == "random":
                random = _unit(payload["random_direction"]).to(
                    device=current.device, dtype=current.dtype
                )
                result["delta"] = float(result["dose_norm"]) * random
            site_record = {key: value for key, value in result.items() if key != "delta"}
            site_record.update(
                {
                    "site": site,
                    "phase": "prefill" if is_prefill else "decode",
                    "layer": layer,
                    "mode": selected,
                    "role": "control",
                    "control_type": config.control_type,
                }
            )
            trace["sites"].append(site_record)
            if not result["feasible"]:
                trace["abstain_requested"] = True
                trace["abstain_reasons"].append(str(result["reason"]))
                return output
            if float(result["dose_norm"]) <= 0:
                return output
            modified = tensor.clone()
            modified[:, -1, :] = current + result["delta"].to(
                device=current.device, dtype=current.dtype
            )
            active_layers_by_site[site] = active_layers_by_site.get(site, 0) + 1
            cumulative_by_mode[selected] += float(result["dose_norm"])
            trace["active"] = True
            trace["reason"] = "selected"
            trace["applied_positions"] += int(modified.shape[0])
            trace["cumulative_dose"] += float(result["dose_norm"])
            if selected not in trace["selected_modes"]:
                trace["selected_modes"].append(selected)
            return _replace_output(output, modified)

        return hook

    try:
        for layer in layers:
            handles.append(blocks[layer].register_forward_hook(make_hook(layer)))
        yield trace
    finally:
        for handle in reversed(handles):
            handle.remove()
        trace["abstain_reasons"] = sorted(set(trace["abstain_reasons"]))
        if not trace["active"]:
            trace["controller_abstained"] = True
            if trace["abstain_requested"]:
                trace["reason"] = "infeasible_edit"
            elif config.control_type == "validator_only" and selected_by_site:
                trace["reason"] = "validator_only"
            elif selected_by_site:
                trace["reason"] = "target_already_reached"
            elif candidates:
                trace["reason"] = "signal_not_confirmed"


def _message_ledger(messages: Sequence[Message]) -> str:
    values = []
    for message in messages:
        if message.content:
            values.append(str(message.content))
        for call in message.tool_calls or []:
            values.append(json.dumps(call.arguments, ensure_ascii=False, sort_keys=True))
    return "\n".join(values).lower()


def _identifier_values(value: Any, *, key: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            found.extend(_identifier_values(child, key=str(child_key)))
    elif isinstance(value, list):
        for child in value:
            found.extend(_identifier_values(child, key=key))
    elif _IDENTIFIER_FIELD.search(key) and isinstance(value, (str, int)):
        found.append((key, str(value)))
    return found


def candidate_fingerprint(message: AssistantMessage) -> Optional[str]:
    if not message.tool_calls:
        return None
    return json.dumps(
        [
            {"name": call.name, "arguments": call.arguments}
            for call in message.tool_calls
        ],
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def validate_candidate_message(
    message: AssistantMessage,
    *,
    tools: Sequence[Tool],
    messages: Sequence[Message],
    boundaries: Sequence[str],
    action_history: Sequence[str],
    config: JServoConfig,
) -> dict[str, Any]:
    """Validate a buffered candidate without executing or consulting gold labels."""
    reasons: list[str] = []
    checks: dict[str, Any] = {
        "schema": "not_applicable",
        "identifier_provenance": "not_applicable",
        "repeated_call": "not_applicable",
        "gold_labels_used": False,
    }
    calls = message.tool_calls or []
    tools_by_name = {tool.name: tool for tool in tools}
    ledger = _message_ledger(messages)
    if calls and config.validate_schema:
        checks["schema"] = "passed"
        for call in calls:
            tool = tools_by_name.get(call.name)
            if tool is None:
                checks["schema"] = "failed"
                reasons.append(f"unknown_tool:{call.name}")
                continue
            try:
                tool.params.model_validate(call.arguments)
            except ValidationError:
                checks["schema"] = "failed"
                reasons.append(f"invalid_arguments:{call.name}")
    if calls and config.require_identifier_provenance:
        checks["identifier_provenance"] = "passed"
        for call in calls:
            for field, value in _identifier_values(call.arguments):
                if value.lower() not in ledger:
                    checks["identifier_provenance"] = "failed"
                    reasons.append(f"unsupported_identifier:{field}")
    if calls and config.block_repeated_calls:
        checks["repeated_call"] = "passed"
        fingerprint = candidate_fingerprint(message)
        strong_boundary = bool(
            set(boundaries)
            & {
                "after_tool_error",
                "after_repeated_tool_error",
                "after_successful_tool_result",
                "after_repeated_tool_call",
                "after_short_tool_cycle",
            }
        )
        if strong_boundary and fingerprint is not None and action_history:
            if fingerprint == action_history[-1]:
                checks["repeated_call"] = "failed"
                reasons.append("repeated_call_without_state_change")
    return {
        "schema_version": "jlens-candidate-validation-v1",
        "valid": not reasons,
        "reasons": sorted(set(reasons)),
        "checks": checks,
        "tool_call_count": len(calls),
    }
