from copy import deepcopy

from tau2.agent.jlens_agent import JLensSoloAgent
from tau2.agent.jlens_backend import BackendGeneration
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
from tau2.runner.build import build_user
from tau2.user.user_simulator import DummyUser


class FakeBackend:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        message = AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="stable",
                    name="create_task",
                    arguments={"task_name": "test task"},
                )
            ],
        )
        return BackendGeneration(
            message=message,
            prompt_input_ids=[1, 2, 3],
            generated_ids=[4],
            rendered_text="tool",
            telemetry_record={},
        )


class FakeTextBackend:
    def generate(self, **_kwargs):
        return BackendGeneration(
            message=AssistantMessage(role="assistant", content="invalid action"),
            prompt_input_ids=[1, 2, 3],
            generated_ids=[4],
            rendered_text="invalid action",
            telemetry_record={},
        )


def test_jlens_solo_prompt_does_not_expose_evaluation_criteria(
    get_environment, base_task
):
    task = deepcopy(base_task)
    secret = "HIDDEN_ASSERTION_SENTINEL_9d7d"
    task.evaluation_criteria.actions[0].arguments["hidden"] = secret
    backend = FakeBackend()
    agent = JLensSoloAgent(
        llm="fake",
        llm_args={},
        tools=get_environment().get_tools(),
        domain_policy=get_environment().get_policy(),
        task=task,
        backend=backend,
    )
    state = agent.get_init_state()
    rendered_source = "\n".join(
        message.content or "" for message in state.system_messages
    )
    assert secret not in rendered_source
    assert str(task.ticket) in rendered_source


def test_jlens_solo_marks_initial_and_post_tool_boundaries(get_environment, base_task):
    backend = FakeBackend()
    agent = JLensSoloAgent(
        llm="fake",
        llm_args={},
        tools=get_environment().get_tools(),
        domain_policy=get_environment().get_policy(),
        task=base_task,
        backend=backend,
    )
    state = agent.get_init_state()
    _, state = agent.generate_next_message(None, state)
    assert backend.calls[0]["boundaries"] == ["initial_decision"]

    result = ToolMessage(
        id="stable",
        role="tool",
        content="created",
        requestor="assistant",
    )
    agent.generate_next_message(result, state)
    assert backend.calls[1]["boundaries"] == ["after_tool_result"]


def test_build_user_constructs_dummy_without_llm_arguments(get_environment, base_task):
    user = build_user(
        "dummy_user",
        get_environment(),
        base_task,
        solo_mode=True,
    )
    assert isinstance(user, DummyUser)


def test_jlens_solo_returns_non_tool_output_for_orchestrator_classification(
    get_environment, base_task
):
    agent = JLensSoloAgent(
        llm="fake",
        llm_args={},
        tools=get_environment().get_tools(),
        domain_policy=get_environment().get_policy(),
        task=base_task,
        backend=FakeTextBackend(),
    )
    message, _ = agent.generate_next_message(None, agent.get_init_state())
    assert message.content == "invalid action"
    assert not message.is_tool_call()
