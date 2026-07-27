import json
from types import SimpleNamespace

import pytest

from tau2.agent.jlens_backend import (
    HFBackendConfig,
    InstrumentationMode,
    InstrumentedHFBackend,
    JSONLTelemetryWriter,
    _ModelBundle,
    assistant_message_from_generation,
    expanded_gqa_sdpa_forward,
    finite_difference_token_effect,
    messages_for_hf,
    normalized_j_vector,
    parse_qwen_tool_calls,
    token_ids_sha256,
)
from tau2.data_model.message import AssistantMessage, SystemMessage, ToolCall
from tau2.environment.tool import as_tool


def test_parse_qwen_tool_call_and_motorization_spans():
    text = (
        '<tool_call>\n{"name":"set_roaming","arguments":'
        '{"phone_number":"+12025550123","enabled":true}}\n</tool_call>'
    )
    calls = parse_qwen_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "set_roaming"
    assert calls[0].arguments["enabled"] is True
    assert text[slice(*calls[0].name_span)] == '"set_roaming"'
    assert json.loads(text[slice(*calls[0].arguments_span)])["enabled"] is True


def test_local_generation_conversion_uses_stable_tool_ids():
    text = '<tool_call>{"name":"done","arguments":{}}</tool_call>'
    first, _ = assistant_message_from_generation(
        text,
        generation_time_seconds=0.1,
        prompt_tokens=10,
        completion_tokens=5,
    )
    second, _ = assistant_message_from_generation(
        text,
        generation_time_seconds=0.2,
        prompt_tokens=10,
        completion_tokens=5,
    )
    assert first.tool_calls[0].id == second.tool_calls[0].id
    assert first.content is None


def test_hf_message_conversion_preserves_tool_structure():
    messages = [
        SystemMessage(role="system", content="policy"),
        AssistantMessage(
            role="assistant",
            tool_calls=[ToolCall(id="call-1", name="lookup", arguments={"line": 3})],
        ),
    ]
    converted = messages_for_hf(messages)
    function = converted[1]["tool_calls"][0]["function"]
    assert function == {"name": "lookup", "arguments": {"line": 3}}


def test_exact_input_id_hash_is_order_sensitive():
    assert token_ids_sha256([1, 2, 3]) == token_ids_sha256([1, 2, 3])
    assert token_ids_sha256([1, 2, 3]) != token_ids_sha256([3, 2, 1])


def test_jsonl_writer_appends_complete_records(tmp_path):
    path = tmp_path / "jlens.jsonl"
    writer = JSONLTelemetryWriter(path)
    writer.write({"record_id": "a", "input_ids": [1, 2]})
    writer.write({"record_id": "b", "input_ids": [3]})
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["record_id"] for record in records] == ["a", "b"]


def test_j_vector_is_normalized_and_has_positive_linear_effect():
    torch = pytest.importorskip("torch")
    jacobian = torch.tensor([[2.0, 0.0], [0.0, 0.5]])
    unembedding = torch.tensor([1.0, -1.0])
    direction = normalized_j_vector(jacobian, unembedding)
    assert torch.linalg.vector_norm(direction).item() == pytest.approx(1.0)
    residual = torch.tensor([0.2, -0.4])
    epsilon = 1e-3

    def score(point):
        return unembedding @ (jacobian @ point)

    assert score(residual + epsilon * direction) > score(residual - epsilon * direction)


def test_finite_difference_helper_reports_positive_target_effect():
    torch = pytest.importorskip("torch")
    jacobian = torch.tensor([[1.5, 0.0], [0.0, 0.25]])
    unembedding = torch.tensor([[0.5, -1.0], [-0.25, 0.75]])

    class FakeLens:
        jacobians = {0: jacobian}

        @staticmethod
        def transport(residual, layer):
            return residual @ jacobian.T

    class FakeLensModel:
        _lm_head = SimpleNamespace(weight=unembedding)

        @staticmethod
        def unembed(residual):
            return residual @ unembedding.T

    result = finite_difference_token_effect(
        FakeLensModel(),
        FakeLens(),
        layer=0,
        token_id=1,
        residual=torch.tensor([0.2, -0.4]),
    )
    assert result["positive"] is True
    assert result["central_difference"] > 0


def test_observe_mode_preserves_exact_deterministic_generation(tmp_path):
    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0
        truncation_side = "left"
        init_kwargs = {"_commit_hash": "tokenizer-commit"}

        @staticmethod
        def apply_chat_template(_messages, **_kwargs):
            input_ids = torch.tensor([[1, 2, 3]])
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }

        @staticmethod
        def decode(token_ids, skip_special_tokens=False):
            return "".join(chr(token_id) for token_id in token_ids)

        @staticmethod
        def encode(text, add_special_tokens=False):
            return [ord(character) for character in text]

    class FakeLensModel:
        n_layers = 2
        d_model = 4
        layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        input_device = torch.device("cpu")

        def __init__(self, weight):
            self._lm_head = SimpleNamespace(weight=weight)

        def forward(self, input_ids):
            hidden = torch.nn.functional.one_hot(input_ids % 4, num_classes=4).float()
            for layer in self.layers:
                hidden = layer(hidden)
            return hidden

        def unembed(self, residual):
            return residual @ self._lm_head.weight.T

    class FakeModel:
        config = SimpleNamespace(_commit_hash="model-commit")

        def __init__(self, completion, weight):
            self.completion = torch.tensor([completion])
            self._embedding = SimpleNamespace(weight=weight)

        def generate(self, input_ids, attention_mask, **_kwargs):
            return torch.cat([input_ids, self.completion], dim=1)

        def get_output_embeddings(self):
            return self._embedding

    text = '<tool_call>{"name":"done","arguments":{}}</tool_call>'
    completion = [ord(character) for character in text]
    weight = torch.arange(256 * 4, dtype=torch.float32).reshape(256, 4) / 100
    bundle = _ModelBundle(
        model=FakeModel(completion, weight),
        tokenizer=FakeTokenizer(),
        lens_model=FakeLensModel(weight),
    )

    def done():
        """Finish the task."""

    tool = as_tool(done)
    messages = [SystemMessage(role="system", content="policy and ticket")]
    off = InstrumentedHFBackend(
        HFBackendConfig(
            model_name_or_path="fake",
            mode=InstrumentationMode.OFF,
            telemetry_path=tmp_path / "off.jsonl",
        ),
        bundle=bundle,
    ).generate(messages=messages, tools=[tool], task_id="t", turn_index=0)
    observe = InstrumentedHFBackend(
        HFBackendConfig(
            model_name_or_path="fake",
            mode=InstrumentationMode.OBSERVE,
            telemetry_path=tmp_path / "observe.jsonl",
        ),
        bundle=bundle,
    ).generate(messages=messages, tools=[tool], task_id="t", turn_index=0)

    assert off.prompt_input_ids == observe.prompt_input_ids
    assert off.generated_ids == observe.generated_ids
    assert off.message.tool_calls == observe.message.tool_calls
    assert observe.telemetry_record["measurement"]["residuals"]


def test_expanded_gqa_attention_matches_query_head_count():
    torch = pytest.importorskip("torch")
    module = SimpleNamespace(num_key_value_groups=2, is_causal=True)
    query = torch.randn(1, 4, 5, 8)
    key = torch.randn(1, 2, 5, 8)
    value = torch.randn(1, 2, 5, 8)
    output, weights = expanded_gqa_sdpa_forward(
        module,
        query,
        key,
        value,
        attention_mask=None,
    )
    assert output.shape == (1, 5, 4, 8)
    assert weights is None
