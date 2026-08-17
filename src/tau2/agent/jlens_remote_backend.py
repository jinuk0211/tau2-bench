"""Authenticated remote execution for the instrumented J-Lens backend.

Tau2 keeps the environment and tool execution local.  Only model generation,
hidden-state measurement, and intervention hooks run on the remote worker.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path, PurePath
from typing import Any, Callable, Optional, Sequence

from tau2.agent.jlens_backend import (
    BackendGeneration,
    HFBackendConfig,
    InstrumentationMode,
    InstrumentedHFBackend,
    InterventionConfig,
    JSONLTelemetryWriter,
)
from tau2.agent.jservo import JServoConfig
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

REMOTE_PROTOCOL_SCHEMA = "tau2-jlens-remote-v1"


@dataclass(frozen=True)
class RemoteExecutionConfig:
    """Connection details; the bearer token is read only from the environment."""

    endpoint: str
    token_env: str = "JLENS_REMOTE_TOKEN"
    timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("remote endpoint must be an HTTP(S) URL")
        if self.timeout_seconds <= 0:
            raise ValueError("remote timeout must be positive")
        if not self.token_env:
            raise ValueError("remote token environment variable is required")


def _json_compatible(value: Any) -> Any:
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, InstrumentationMode):
        return value.value
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    return value


def hf_config_to_wire(config: HFBackendConfig) -> dict[str, Any]:
    """Serialize a backend config without asking the worker to write local paths."""
    payload = _json_compatible(asdict(config))
    payload["telemetry_path"] = None
    return payload


def hf_config_from_wire(value: dict[str, Any]) -> HFBackendConfig:
    """Validate the model/intervention identity received by a remote worker."""
    copied = dict(value)
    copied["mode"] = InstrumentationMode(copied.get("mode", "off"))
    copied["selected_layers"] = tuple(
        int(item) for item in copied.get("selected_layers", ())
    )
    copied["concept_tokens"] = dict(copied.get("concept_tokens", {}))
    copied["chat_template_kwargs"] = dict(copied.get("chat_template_kwargs", {}))
    copied["generation_kwargs"] = dict(copied.get("generation_kwargs", {}))
    copied["telemetry_path"] = None
    if copied.get("lens_path") is not None:
        copied["lens_path"] = Path(str(copied["lens_path"]))
    copied["intervention"] = InterventionConfig.from_dict(copied.get("intervention"))
    copied["controller"] = JServoConfig.from_dict(copied.get("controller"))
    return HFBackendConfig(**copied)


def _message_to_wire(message: Message) -> dict[str, Any]:
    return message.model_dump(mode="json", exclude_none=True)


def _message_from_wire(value: dict[str, Any]) -> Message:
    role = value.get("role")
    if role == "system":
        return SystemMessage.model_validate(value)
    if role == "user":
        return UserMessage.model_validate(value)
    if role == "assistant":
        return AssistantMessage.model_validate(value)
    if role == "tool":
        return ToolMessage.model_validate(value)
    raise ValueError(f"unsupported remote message role: {role!r}")


class _SchemaTool:
    def __init__(self, schema: dict[str, Any]):
        self._schema = schema

    @property
    def openai_schema(self) -> dict[str, Any]:
        return self._schema


class RemoteInstrumentedBackend:
    """Drop-in backend client that never loads a model in the Tau2 process."""

    def __init__(
        self,
        config: HFBackendConfig,
        remote: RemoteExecutionConfig,
        *,
        post: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.config = config
        self.remote = remote
        if post is None:
            import requests

            post = requests.post
        self._post = post
        self.writer = (
            JSONLTelemetryWriter(config.telemetry_path)
            if config.telemetry_path is not None
            else None
        )

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
        token = os.environ.get(self.remote.token_env)
        if not token:
            raise RuntimeError(
                f"remote J-Lens token is missing from {self.remote.token_env}"
            )
        payload = {
            "schema_version": REMOTE_PROTOCOL_SCHEMA,
            "backend_config": hf_config_to_wire(self.config),
            "messages": [_message_to_wire(message) for message in messages],
            "tools": [tool.openai_schema for tool in tools],
            "task_id": task_id,
            "turn_index": int(turn_index),
            "boundaries": list(boundaries),
            "stop_tool_name": stop_tool_name,
        }
        response = self._post(
            f"{self.remote.endpoint.rstrip('/')}/v1/generate",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=self.remote.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("schema_version") != REMOTE_PROTOCOL_SCHEMA:
            raise RuntimeError("remote J-Lens worker returned an unknown schema")
        record = dict(body.get("telemetry_record") or {})
        if self.writer is not None:
            self.writer.write(record)
        message = AssistantMessage.model_validate(body["message"])
        raw_data = dict(message.raw_data or {})
        raw_data.update(
            {
                "provider": "remote_jlens",
                "remote_endpoint_sha256": hashlib.sha256(
                    self.remote.endpoint.encode()
                ).hexdigest(),
            }
        )
        message.raw_data = raw_data
        return BackendGeneration(
            message=message,
            prompt_input_ids=[int(item) for item in body["prompt_input_ids"]],
            generated_ids=[int(item) for item in body["generated_ids"]],
            rendered_text=str(body["rendered_text"]),
            telemetry_record=record,
        )


def execute_remote_payload(
    payload: dict[str, Any],
    *,
    backend_loader: Callable[[HFBackendConfig], InstrumentedHFBackend],
) -> dict[str, Any]:
    """Worker-side protocol implementation, separated for deterministic tests."""
    if payload.get("schema_version") != REMOTE_PROTOCOL_SCHEMA:
        raise ValueError("unsupported remote J-Lens request schema")
    config = hf_config_from_wire(dict(payload["backend_config"]))
    backend = backend_loader(config)
    generation = backend.generate(
        messages=[_message_from_wire(item) for item in payload["messages"]],
        tools=[_SchemaTool(item) for item in payload.get("tools", [])],
        task_id=str(payload["task_id"]),
        turn_index=int(payload["turn_index"]),
        boundaries=[str(item) for item in payload.get("boundaries", [])],
        stop_tool_name=str(payload.get("stop_tool_name", "done")),
    )
    return {
        "schema_version": REMOTE_PROTOCOL_SCHEMA,
        "message": generation.message.model_dump(mode="json", exclude_none=True),
        "prompt_input_ids": generation.prompt_input_ids,
        "generated_ids": generation.generated_ids,
        "rendered_text": generation.rendered_text,
        "telemetry_record": generation.telemetry_record,
    }


def backend_cache_key(config: HFBackendConfig) -> str:
    """Key one remote model bundle and intervention setup without secrets."""
    payload = json.dumps(
        hf_config_to_wire(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()
