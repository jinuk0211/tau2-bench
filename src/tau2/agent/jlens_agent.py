"""τ² half-duplex agents backed by the local instrumented HF backend."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from tau2.agent.jlens_backend import (
    HFBackendConfig,
    InstrumentationMode,
    InstrumentedHFBackend,
    InterventionConfig,
)
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
    "hf_revision",
    "hf_tokenizer_revision",
    "hf_max_input_tokens",
    "hf_chat_template_kwargs",
    "hf_device",
    "hf_dtype",
    "hf_sdpa_backend",
    "hf_trust_remote_code",
}


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
        chat_template_kwargs=dict(backend_args.get("hf_chat_template_kwargs", {})),
        generation_kwargs=args,
        device=backend_args.get("hf_device", "auto"),
        dtype=backend_args.get("hf_dtype", "auto"),
        sdpa_backend=backend_args.get("hf_sdpa_backend", "auto"),
        trust_remote_code=bool(backend_args.get("hf_trust_remote_code", False)),
    )


class JLensAgent(LLMAgent):
    """Regular tau2 text agent using the exact-token local J-Lens backend."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        task: Task,
        llm: str,
        llm_args: Optional[dict] = None,
        *,
        simulation_id: Optional[str] = None,
        backend: Optional[InstrumentedHFBackend] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        self.task = task
        self.backend = backend or InstrumentedHFBackend.from_pretrained(
            backend_config_from_agent_args(
                llm=llm,
                llm_args=llm_args,
                task_id=str(task.id),
                simulation_id=simulation_id,
            )
        )
        self._turn_index = 0

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
            boundaries.append("after_tool_result")
        else:
            state.messages.append(message)
            if isinstance(message, ToolMessage):
                boundaries.append("after_tool_result")
            elif isinstance(message, UserMessage):
                boundaries.append("after_user_message")
        generation = self.backend.generate(
            messages=state.system_messages + state.messages,
            tools=self.tools,
            task_id=str(self.task.id),
            turn_index=self._turn_index,
            boundaries=boundaries,
        )
        state.messages.append(generation.message)
        self._turn_index += 1
        return generation.message, state


class JLensSoloAgent(LLMSoloAgent):
    """No-user τ² agent using one shared local HF/J-Lens backend."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        task: Task,
        llm: str,
        llm_args: Optional[dict] = None,
        *,
        simulation_id: Optional[str] = None,
        backend: Optional[InstrumentedHFBackend] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            task=task,
            llm=llm,
            llm_args=llm_args,
        )
        self.backend = backend or InstrumentedHFBackend.from_pretrained(
            backend_config_from_agent_args(
                llm=llm,
                llm_args=llm_args,
                task_id=str(task.id),
                simulation_id=simulation_id,
            )
        )
        self._turn_index = 0

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
            boundaries.append("after_tool_result")
        elif isinstance(message, ToolMessage):
            state.messages.append(message)
            boundaries.append("after_tool_result")
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
        assistant_message = generation.message
        if assistant_message.is_tool_call():
            assistant_message = self._check_if_stop_toolcall(assistant_message)
        state.messages.append(assistant_message)
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
