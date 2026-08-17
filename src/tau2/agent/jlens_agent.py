"""τ² half-duplex agents backed by local or remote instrumented HF execution."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from tau2.agent.jlens_backend import (
    HFBackendConfig,
    InstrumentationMode,
    InstrumentedHFBackend,
    InterventionConfig,
)
from tau2.agent.jlens_remote_backend import (
    RemoteExecutionConfig,
    RemoteInstrumentedBackend,
)
from tau2.agent.jservo import JServoConfig, validate_candidate_message
from tau2.agent.llm_agent import LLMAgent, LLMAgentState, LLMSoloAgent
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.tasks import Task
from tau2.environment.tool import Tool

_BACKEND_KEYS = {
    "jlens_mode",
    "jlens_path",
    "jlens_telemetry_path",
    "jlens_selected_layers",
    "jlens_concept_tokens",
    "jlens_intervention",
    "jlens_controller",
    "hf_revision",
    "hf_tokenizer_revision",
    "hf_max_input_tokens",
    "hf_chat_template_kwargs",
    "hf_device",
    "hf_dtype",
    "hf_sdpa_backend",
    "hf_trust_remote_code",
    "jlens_remote_endpoint",
    "jlens_remote_timeout_seconds",
    "jlens_remote_token_env",
    "jlens_require_remote",
}


def _tool_result_boundaries(
    message: ToolMessage | MultiToolMessage,
    *,
    action_history: list[str],
) -> list[str]:
    """Describe the observed tool outcome without relying on task-specific turns."""
    tool_messages = (
        message.tool_messages if isinstance(message, MultiToolMessage) else [message]
    )
    boundaries = ["after_tool_result"]
    if any(item.error for item in tool_messages):
        boundaries.append("after_tool_error")
    else:
        boundaries.append("after_successful_tool_result")
    if len(action_history) >= 2 and action_history[-1] == action_history[-2]:
        boundaries.append("after_repeated_tool_call")
        if any(item.error for item in tool_messages):
            boundaries.append("after_repeated_tool_error")
    elif any(
        len(action_history) >= 2 * period
        and action_history[-2 * period : -period] == action_history[-period:]
        for period in (2, 3)
    ):
        boundaries.append("after_short_tool_cycle")
    return boundaries


def _tool_action_fingerprint(message: AssistantMessage) -> Optional[str]:
    """Fingerprint one assistant action while ignoring generated call IDs."""
    if not message.tool_calls:
        return None
    calls = [
        {
            "name": call.name,
            "arguments": call.arguments,
        }
        for call in message.tool_calls
    ]
    return json.dumps(calls, ensure_ascii=False, sort_keys=True, default=str)


def _path_for_run(
    value: Optional[str], task_id: str, simulation_id: Optional[str]
) -> Optional[Path]:
    if value is None:
        return None

    def safe_segment(item: str) -> str:
        return re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", item).strip(" .") or "item"

    return Path(
        value.format(
            task_id=safe_segment(task_id),
            simulation_id=safe_segment(simulation_id or "unspecified"),
        )
    )


def _buffer_and_validate_candidate(
    generation: Any,
    *,
    backend: InstrumentedHFBackend | RemoteInstrumentedBackend,
    tools: list[Tool],
    messages: list[Message],
    boundaries: list[str],
    action_history: list[str],
) -> AssistantMessage:
    """Block invalid J-Servo candidates before Tau2 can execute a tool call."""
    controller = getattr(getattr(backend, "config", None), "controller", None)
    if controller is None:
        return generation.message
    validation = validate_candidate_message(
        generation.message,
        tools=tools,
        messages=messages,
        boundaries=boundaries,
        action_history=action_history,
        config=controller,
    )
    controller_trace = generation.telemetry_record.get("controller") or {}
    controller_abstain = bool(controller_trace.get("abstain_requested"))
    if controller_abstain:
        validation["valid"] = False
        validation["reasons"] = sorted(
            {
                *validation["reasons"],
                *(
                    f"controller:{reason}"
                    for reason in controller_trace.get("abstain_reasons", [])
                ),
            }
        )
    validation["controller_abstain_requested"] = controller_abstain
    validation["parent_record_id"] = generation.telemetry_record.get("record_id")
    generation.telemetry_record["candidate_validation"] = validation
    raw_data = dict(generation.message.raw_data or {})
    raw_data["jlens_candidate_validation"] = validation
    generation.message.raw_data = raw_data
    writer = getattr(backend, "writer", None)
    if writer is not None:
        writer.write(validation)
    if validation["valid"]:
        return generation.message
    return AssistantMessage.text(
        "J-Servo abstained before tool execution.",
        raw_data={
            **raw_data,
            "jlens_abstained": True,
            "jlens_original_candidate": generation.message.model_dump(
                mode="json", exclude_none=True
            ),
        },
        generation_time_seconds=generation.message.generation_time_seconds,
    )


def backend_config_from_agent_args(
    *,
    llm: str,
    llm_args: Optional[dict[str, Any]],
    task_id: str,
    simulation_id: Optional[str] = None,
) -> HFBackendConfig:
    """Split backend/instrumentation arguments from HF generation arguments."""
    args = dict(llm_args or {})
    backend_args = {key: args.pop(key) for key in list(args) if key in _BACKEND_KEYS}
    mode = InstrumentationMode(backend_args.get("jlens_mode", "off"))
    intervention = InterventionConfig.from_dict(backend_args.get("jlens_intervention"))
    controller = JServoConfig.from_dict(backend_args.get("jlens_controller"))
    return HFBackendConfig(
        model_name_or_path=llm,
        revision=backend_args.get("hf_revision"),
        tokenizer_revision=backend_args.get("hf_tokenizer_revision"),
        mode=mode,
        seed=int(args.pop("seed", 42)),
        max_input_tokens=int(backend_args.get("hf_max_input_tokens", 16384)),
        selected_layers=tuple(backend_args.get("jlens_selected_layers", ())),
        concept_tokens=dict(backend_args.get("jlens_concept_tokens", {})),
        telemetry_path=_path_for_run(
            backend_args.get("jlens_telemetry_path"), task_id, simulation_id
        ),
        lens_path=(
            Path(backend_args["jlens_path"]) if backend_args.get("jlens_path") else None
        ),
        intervention=intervention,
        controller=controller,
        chat_template_kwargs=dict(backend_args.get("hf_chat_template_kwargs", {})),
        generation_kwargs=args,
        device=backend_args.get("hf_device", "auto"),
        dtype=backend_args.get("hf_dtype", "auto"),
        sdpa_backend=backend_args.get("hf_sdpa_backend", "auto"),
        trust_remote_code=bool(backend_args.get("hf_trust_remote_code", False)),
    )


def build_jlens_backend(
    *,
    llm: str,
    llm_args: Optional[dict[str, Any]],
    task_id: str,
    simulation_id: Optional[str] = None,
) -> InstrumentedHFBackend | RemoteInstrumentedBackend:
    """Select remote execution explicitly and never silently fall back from it."""
    args = dict(llm_args or {})
    config = backend_config_from_agent_args(
        llm=llm,
        llm_args=args,
        task_id=task_id,
        simulation_id=simulation_id,
    )
    endpoint = args.get("jlens_remote_endpoint")
    if endpoint:
        raw_intervention = args.get("jlens_intervention")
        if config.intervention is not None and isinstance(raw_intervention, dict):
            raw_vector_path = raw_intervention.get("vector_path")
            if raw_vector_path is not None:
                config = replace(
                    config,
                    intervention=replace(
                        config.intervention,
                        vector_path=PurePosixPath(str(raw_vector_path)),
                    ),
                )
        raw_controller = args.get("jlens_controller")
        if config.controller is not None and isinstance(raw_controller, dict):
            raw_artifact_path = raw_controller.get("artifact_path")
            if raw_artifact_path is not None:
                config = replace(
                    config,
                    controller=replace(
                        config.controller,
                        artifact_path=PurePosixPath(str(raw_artifact_path)),
                    ),
                )
        if args.get("jlens_path") is not None:
            config = replace(
                config,
                lens_path=PurePosixPath(str(args["jlens_path"])),
            )
        return RemoteInstrumentedBackend(
            config,
            RemoteExecutionConfig(
                endpoint=str(endpoint),
                token_env=str(args.get("jlens_remote_token_env", "JLENS_REMOTE_TOKEN")),
                timeout_seconds=float(args.get("jlens_remote_timeout_seconds", 600.0)),
            ),
        )
    if bool(args.get("jlens_require_remote", False)):
        raise ValueError(
            "jlens_require_remote=true but jlens_remote_endpoint is missing"
        )
    return InstrumentedHFBackend.from_pretrained(config)


class JLensAgent(LLMAgent):
    """Regular tau2 text agent using an exact-token J-Lens backend."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        task: Task,
        llm: str,
        llm_args: Optional[dict] = None,
        *,
        simulation_id: Optional[str] = None,
        backend: Optional[InstrumentedHFBackend | RemoteInstrumentedBackend] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        self.task = task
        self.backend = backend or build_jlens_backend(
            llm=llm,
            llm_args=llm_args,
            task_id=str(task.id),
            simulation_id=simulation_id,
        )
        self._turn_index = 0
        self._tool_action_history: list[str] = []

    def generate_next_message(
        self,
        message: UserMessage | ToolMessage | MultiToolMessage,
        state: LLMAgentState,
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """Respond normally while recording the exact rendered token stream."""
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("User message cannot be audio in JLensAgent")
        boundaries: list[str] = []
        if self._turn_index == 0:
            boundaries.append("initial_decision")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
            boundaries.extend(
                _tool_result_boundaries(
                    message,
                    action_history=self._tool_action_history,
                )
            )
        else:
            state.messages.append(message)
            if isinstance(message, ToolMessage):
                boundaries.extend(
                    _tool_result_boundaries(
                        message,
                        action_history=self._tool_action_history,
                    )
                )
            elif isinstance(message, UserMessage):
                boundaries.append("after_user_message")
        generation = self.backend.generate(
            messages=state.system_messages + state.messages,
            tools=self.tools,
            task_id=str(self.task.id),
            turn_index=self._turn_index,
            boundaries=boundaries,
        )
        assistant_message = _buffer_and_validate_candidate(
            generation,
            backend=self.backend,
            tools=self.tools,
            messages=state.system_messages + state.messages,
            boundaries=boundaries,
            action_history=self._tool_action_history,
        )
        state.messages.append(assistant_message)
        fingerprint = _tool_action_fingerprint(assistant_message)
        if fingerprint is not None:
            self._tool_action_history.append(fingerprint)
        self._turn_index += 1
        return assistant_message, state


class JLensSoloAgent(LLMSoloAgent):
    """No-user τ² agent using one shared HF/J-Lens backend."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        task: Task,
        llm: str,
        llm_args: Optional[dict] = None,
        *,
        simulation_id: Optional[str] = None,
        backend: Optional[InstrumentedHFBackend | RemoteInstrumentedBackend] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            task=task,
            llm=llm,
            llm_args=llm_args,
        )
        self.backend = backend or build_jlens_backend(
            llm=llm,
            llm_args=llm_args,
            task_id=str(task.id),
            simulation_id=simulation_id,
        )
        self._turn_index = 0
        self._tool_action_history: list[str] = []

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentState:
        """Build prompt state from policy, ticket, and allowed trajectory only."""
        if message_history is None:
            message_history = []
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self,
        message: Optional[UserMessage | ToolMessage | MultiToolMessage],
        state: LLMAgentState,
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """Generate one tool-only action and record semantic boundaries."""
        if isinstance(message, UserMessage):
            raise ValueError("JLensSoloAgent does not support user messages")
        boundaries: list[str] = []
        if self._turn_index == 0:
            boundaries.append("initial_decision")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
            boundaries.extend(
                _tool_result_boundaries(
                    message,
                    action_history=self._tool_action_history,
                )
            )
        elif isinstance(message, ToolMessage):
            state.messages.append(message)
            boundaries.extend(
                _tool_result_boundaries(
                    message,
                    action_history=self._tool_action_history,
                )
            )
        elif message is None:
            if state.messages:
                raise AssertionError("None input is only valid for the first turn")
        else:
            raise TypeError(f"Unsupported agent input: {type(message).__name__}")

        generation = self.backend.generate(
            messages=state.system_messages + state.messages,
            tools=self.tools,
            task_id=str(self.task.id),
            turn_index=self._turn_index,
            boundaries=boundaries,
            stop_tool_name=self.STOP_FUNCTION_NAME,
        )
        assistant_message = _buffer_and_validate_candidate(
            generation,
            backend=self.backend,
            tools=self.tools,
            messages=state.system_messages + state.messages,
            boundaries=boundaries,
            action_history=self._tool_action_history,
        )
        if assistant_message.is_tool_call():
            assistant_message = self._check_if_stop_toolcall(assistant_message)
        state.messages.append(assistant_message)
        fingerprint = _tool_action_fingerprint(assistant_message)
        if fingerprint is not None:
            self._tool_action_history.append(fingerprint)
        self._turn_index += 1
        return assistant_message, state


def create_jlens_direct_solo_agent(tools, domain_policy, **kwargs):
    """Factory for the direct No-User J-Lens condition."""
    return JLensSoloAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        task=kwargs.get("task"),
        simulation_id=kwargs.get("simulation_id"),
    )


def create_jlens_agent(tools, domain_policy, **kwargs):
    """Factory for regular user-agent tau2 conversations with J-Lens traces."""
    return JLensAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        task=kwargs.get("task"),
        simulation_id=kwargs.get("simulation_id"),
    )
