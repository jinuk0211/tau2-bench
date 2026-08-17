from tau2.agent.jlens_backend import (
    BackendGeneration,
    HFBackendConfig,
    InstrumentationMode,
    InterventionConfig,
)
from tau2.agent.jlens_remote_backend import (
    REMOTE_PROTOCOL_SCHEMA,
    RemoteExecutionConfig,
    RemoteInstrumentedBackend,
    execute_remote_payload,
    hf_config_from_wire,
    hf_config_to_wire,
)
from tau2.agent.jservo import JServoConfig
from tau2.data_model.message import AssistantMessage, SystemMessage, ToolCall


class SchemaTool:
    openai_schema = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a record",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _config(tmp_path):
    return HFBackendConfig(
        model_name_or_path="org/model",
        revision="a" * 40,
        mode=InstrumentationMode.INTERVENE,
        telemetry_path=tmp_path / "trace.jsonl",
        selected_layers=(3, 7),
        generation_kwargs={"max_new_tokens": 4, "do_sample": False},
        intervention=InterventionConfig(
            kind="steer",
            method="legacy",
            layer=7,
            strength=0.5,
            vector=(1.0, 0.0),
            boundaries=("after_tool_error",),
        ),
    )


def test_remote_config_round_trip_omits_client_telemetry_path(tmp_path):
    config = _config(tmp_path)
    wire = hf_config_to_wire(config)
    assert wire["telemetry_path"] is None
    restored = hf_config_from_wire(wire)
    assert restored.model_name_or_path == "org/model"
    assert restored.selected_layers == (3, 7)
    assert restored.intervention.boundaries == ("after_tool_error",)
    assert restored.telemetry_path is None


def test_remote_config_round_trip_preserves_jservo_controller(tmp_path):
    config = HFBackendConfig(
        model_name_or_path="org/model",
        revision="b" * 40,
        mode=InstrumentationMode.INTERVENE,
        telemetry_path=tmp_path / "trace.jsonl",
        controller=JServoConfig(
            artifact_path=tmp_path / "jservo.pt",
            control_type="fixed_strength",
            fixed_strength=0.1,
        ),
    )
    restored = hf_config_from_wire(hf_config_to_wire(config))
    assert restored.controller.control_type == "fixed_strength"
    assert restored.controller.fixed_strength == 0.1
    assert str(restored.controller.artifact_path).endswith("jservo.pt")


def test_remote_client_sends_boundaries_and_records_telemetry(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_JLENS_TOKEN", "secret")
    observed = {}

    class Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "schema_version": REMOTE_PROTOCOL_SCHEMA,
                "message": AssistantMessage(
                    role="assistant",
                    tool_calls=[ToolCall(id="call", name="lookup", arguments={})],
                ).model_dump(mode="json", exclude_none=True),
                "prompt_input_ids": [1, 2],
                "generated_ids": [3],
                "rendered_text": "<tool_call>lookup</tool_call>",
                "telemetry_record": {"record_id": "remote-record"},
            }

    def post(url, **kwargs):
        observed.update({"url": url, **kwargs})
        return Response()

    backend = RemoteInstrumentedBackend(
        _config(tmp_path),
        RemoteExecutionConfig(
            endpoint="https://worker.example",
            token_env="TEST_JLENS_TOKEN",
        ),
        post=post,
    )
    result = backend.generate(
        messages=[SystemMessage(role="system", content="policy")],
        tools=[SchemaTool()],
        task_id="airline-3",
        turn_index=2,
        boundaries=["after_tool_error", "after_repeated_tool_call"],
    )

    assert observed["url"] == "https://worker.example/v1/generate"
    assert observed["headers"] == {"Authorization": "Bearer secret"}
    assert observed["json"]["boundaries"] == [
        "after_tool_error",
        "after_repeated_tool_call",
    ]
    assert result.message.raw_data["provider"] == "remote_jlens"
    assert (tmp_path / "trace.jsonl").is_file()


def test_worker_protocol_reconstructs_messages_tools_and_config(tmp_path):
    config = _config(tmp_path)
    observed = {}

    class Backend:
        def generate(self, **kwargs):
            observed.update(kwargs)
            return BackendGeneration(
                message=AssistantMessage(role="assistant", content="done"),
                prompt_input_ids=[1],
                generated_ids=[2],
                rendered_text="done",
                telemetry_record={"record_id": "r1"},
            )

    payload = {
        "schema_version": REMOTE_PROTOCOL_SCHEMA,
        "backend_config": hf_config_to_wire(config),
        "messages": [
            SystemMessage(role="system", content="policy").model_dump(
                mode="json", exclude_none=True
            )
        ],
        "tools": [SchemaTool.openai_schema],
        "task_id": "airline-3",
        "turn_index": 4,
        "boundaries": ["after_short_tool_cycle"],
        "stop_tool_name": "done",
    }
    body = execute_remote_payload(payload, backend_loader=lambda loaded: Backend())

    assert body["schema_version"] == REMOTE_PROTOCOL_SCHEMA
    assert observed["boundaries"] == ["after_short_tool_cycle"]
    assert observed["tools"][0].openai_schema == SchemaTool.openai_schema
    assert observed["messages"][0].content == "policy"
