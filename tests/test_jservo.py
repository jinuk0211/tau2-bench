from pathlib import Path

import pytest

from tau2.agent.jservo import (
    JServoConfig,
    jservo_generation_hooks,
    minimum_state_edit,
    validate_candidate_message,
)
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.environment.tool import Tool


def _controller() -> JServoConfig:
    return JServoConfig(artifact_path=Path("unused.pt"))


def test_zero_error_has_zero_dose_and_target_error_gets_minimum_dose():
    torch = pytest.importorskip("torch")
    reached = minimum_state_edit(
        torch.tensor([2.0, 0.0]),
        margin_direction=torch.tensor([1.0, 0.0]),
        projected_direction=torch.tensor([1.0, 0.0]),
        target_margin=1.0,
        dose_cap=4.0,
        cumulative_dose=0.0,
        cumulative_cap=4.0,
    )
    assert reached["dose_norm"] == 0.0
    correction = minimum_state_edit(
        torch.tensor([0.0, 0.0]),
        margin_direction=torch.tensor([1.0, 0.0]),
        projected_direction=torch.tensor([1.0, 0.0]),
        target_margin=1.0,
        dose_cap=4.0,
        cumulative_dose=0.0,
        cumulative_cap=4.0,
    )
    assert correction["dose_norm"] == pytest.approx(1.0)
    assert correction["predicted_post_margin"] == pytest.approx(1.0)


def test_cap_excess_requests_abstention_in_generation_trace():
    torch = pytest.importorskip("torch")
    blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    payload = {
        "margin_direction": torch.tensor([1.0, 0.0]),
        "projected_direction": torch.tensor([1.0, 0.0]),
        "random_direction": torch.tensor([0.0, 1.0]),
        "gate_threshold": 0.5,
        "target_margin": 2.0,
        "margin_scale": 1.0,
        "dose_cap": 0.1,
        "residual_scale": 1.0,
    }
    artifact = {
        "schema_version": "jlens-jservo-v1",
        "controller_version": "failure-mode-adaptive-v1",
        "artifact_fingerprint": "test",
        "modes": {
            "retry_without_state_change": {
                "mode": "retry_without_state_change",
                "boundaries": ["after_tool_error"],
                "observation_layers": [0],
                "control_layers": [1],
                "max_active_layers_per_position": 1,
                "cumulative_dose_cap": 0.1,
                "steering_eligible": True,
                "layers": {"0": payload, "1": payload},
            }
        },
    }
    hidden = torch.zeros(1, 1, 2)
    with jservo_generation_hooks(
        blocks,
        artifact=artifact,
        config=_controller(),
        boundaries=["after_tool_error"],
    ) as trace:
        hidden = blocks[0](hidden)
        blocks[1](hidden)
    assert trace["abstain_requested"]
    assert trace["abstain_reasons"] == ["layer_dose_cap_exceeded"]
    assert trace["cumulative_dose"] == 0.0


def test_candidate_buffer_requires_schema_provenance_and_blocks_repeat():
    def lookup(record_id: str) -> str:
        """Look up one record."""
        return record_id

    tool = Tool(lookup)
    candidate = AssistantMessage(
        role="assistant",
        tool_calls=[
            ToolCall(name="lookup", arguments={"record_id": "AIR-17"})
        ],
    )
    valid = validate_candidate_message(
        candidate,
        tools=[tool],
        messages=[UserMessage(role="user", content="Please inspect AIR-17")],
        boundaries=["after_user_message"],
        action_history=[],
        config=_controller(),
    )
    assert valid["valid"]
    unsupported = validate_candidate_message(
        candidate,
        tools=[tool],
        messages=[UserMessage(role="user", content="Please inspect the record")],
        boundaries=["after_user_message"],
        action_history=[],
        config=_controller(),
    )
    assert not unsupported["valid"]
    repeated = validate_candidate_message(
        candidate,
        tools=[tool],
        messages=[UserMessage(role="user", content="Please inspect AIR-17")],
        boundaries=["after_successful_tool_result"],
        action_history=[
            '[{"arguments": {"record_id": "AIR-17"}, "name": "lookup"}]'
        ],
        config=_controller(),
    )
    assert not repeated["valid"]
    assert "repeated_call_without_state_change" in repeated["reasons"]
