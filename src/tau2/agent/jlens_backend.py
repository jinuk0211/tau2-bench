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
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

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
    concept_token_id: Optional[int] = None
    vector: Optional[tuple[float, ...]] = None

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
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    device: str = "auto"
    dtype: str = "auto"
    sdpa_backend: Literal["auto", "efficient", "flash", "math"] = "auto"
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if self.mode == InstrumentationMode.INTERVENE and self.intervention is None:
            raise ValueError("intervene mode requires an intervention")
        if self.mode != InstrumentationMode.INTERVENE and self.intervention is not None:
            raise ValueError("an intervention is only valid in intervene mode")
        if self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        if self.sdpa_backend not in {"auto", "efficient", "flash", "math"}:
            raise ValueError(f"unknown SDPA backend: {self.sdpa_backend}")


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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

    def _intervention_direction(self, intervention: InterventionConfig) -> Any:
        import torch

        if intervention.vector is not None:
            direction = torch.tensor(intervention.vector, dtype=torch.float32)
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
        else:
            raise ValueError("intervention requires vector or concept_token_id")
        if direction.numel() != self.bundle.lens_model.d_model:
            raise ValueError(
                f"intervention vector has {direction.numel()} elements; expected "
                f"{self.bundle.lens_model.d_model}"
            )
        return direction

    @contextmanager
    def _intervention_hook(self):
        intervention = self.config.intervention
        if intervention is None:
            yield
            return
        block = self.bundle.lens_model.layers[intervention.layer]
        direction = self._intervention_direction(intervention)

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            tensor = output if hasattr(output, "shape") else output[0]
            modified = tensor.clone()
            current = modified[:, -1, :]
            vector = direction.to(current.device, dtype=current.dtype)
            if intervention.kind == "steer":
                current = current + intervention.strength * vector
            elif intervention.kind == "ablate":
                projection = (current * vector).sum(dim=-1, keepdim=True) * vector
                current = current - intervention.strength * projection
            elif intervention.kind == "patch":
                current = current + intervention.strength * (vector - current)
            else:
                raise ValueError(f"unknown intervention kind: {intervention.kind}")
            modified[:, -1, :] = current
            return _replace_block_output(output, modified)

        handle = block.register_forward_hook(hook)
        try:
            yield
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
        parsed_calls: Sequence[ParsedToolCall],
    ) -> dict[str, Any]:
        import torch
        from jlens.hooks import ActivationRecorder

        if not generated_ids:
            return {"positions": {}, "residuals": [], "motorization": {}}
        generated = torch.tensor(
            [generated_ids], device=prompt_ids.device, dtype=prompt_ids.dtype
        )
        full_ids = torch.cat([prompt_ids, generated], dim=1)
        prompt_length = int(prompt_ids.shape[1])
        final_layer = self.bundle.lens_model.n_layers - 1
        layers = tuple(sorted(set(self._selected_layers()) | {final_layer}))
        position_groups: dict[str, list[int]] = {
            "initial_decision": [prompt_length - 1]
        }
        for index, call in enumerate(parsed_calls):
            name_tokens = _find_span_token_indices(
                self.bundle.tokenizer, generated_ids, call.name_span
            )
            argument_tokens = _find_span_token_indices(
                self.bundle.tokenizer, generated_ids, call.arguments_span
            )
            if name_tokens:
                position_groups[f"tool_{index}_name"] = [
                    prompt_length + token_index - 1 for token_index in name_tokens
                ]
            if argument_tokens:
                position_groups[f"tool_{index}_arguments"] = [
                    prompt_length + token_index - 1 for token_index in argument_tokens
                ]

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
        generation_kwargs = dict(self.config.generation_kwargs)
        generation_kwargs.setdefault("max_new_tokens", 256)
        generation_kwargs.setdefault("do_sample", False)
        if self.bundle.tokenizer.pad_token_id is None:
            generation_kwargs.setdefault(
                "pad_token_id", self.bundle.tokenizer.eos_token_id
            )
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        start = time.perf_counter()
        with self._sdpa_kernel(), self._intervention_hook():
            generated = self.bundle.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        duration = time.perf_counter() - start
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

        measurement: dict[str, Any] = {}
        if self.config.mode in {
            InstrumentationMode.OBSERVE,
            InstrumentationMode.INTERVENE,
        }:
            measurement = self._teacher_forced_measurement(
                prompt_ids=input_ids,
                generated_ids=completion,
                parsed_calls=parsed_calls,
            )
        record = {
            "schema_version": "tau2-jlens-v1",
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
            "generated_text": raw_text,
            "tool_calls": [
                {"name": call.name, "arguments": call.arguments}
                for call in parsed_calls
            ],
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
