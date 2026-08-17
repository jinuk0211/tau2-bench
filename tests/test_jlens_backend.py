import hashlib
import json
from types import SimpleNamespace

import pytest

from tau2.agent.jlens_backend import (
    HFBackendConfig,
    InstrumentationMode,
    InstrumentedHFBackend,
    InterventionConfig,
    JSONLTelemetryWriter,
    _ModelBundle,
    assistant_message_from_generation,
    cast_condition_similarity,
    expanded_gqa_sdpa_forward,
    finite_difference_token_effect,
    load_austeer_artifact,
    load_caa_direction_artifact,
    load_cast_artifact,
    load_iti_artifact,
    load_loreft_artifact,
    load_mera_artifact,
    load_sadi_artifact,
    mera_closed_form_delta,
    messages_for_hf,
    normalized_j_vector,
    parse_qwen_tool_calls,
    semantic_prediction_positions,
    token_ids_sha256,
)
from tau2.data_model.message import AssistantMessage, SystemMessage, ToolCall
from tau2.environment.tool import as_tool


def _write_caa_artifact(torch, path, *, model_id="fake", layer=0):
    direction = torch.tensor([3.0, 4.0])
    unit = direction / direction.norm()
    metadata = {
        "schema_version": "agent-steering-vector-v1",
        "method": "caa",
        "orientation": "positive_minus_negative",
        "model_id": model_id,
        "model_revision": "revision",
        "layer": layer,
        "positive_label": "correct",
        "negative_label": "failure",
        "extraction_site": "assistant_decision",
        "benchmark": "taubench",
        "pair_ids": ["task-18"],
        "pair_count": 1,
        "d_model": 2,
        "calibration_split": {"tasks": [18]},
    }
    artifact = {
        **metadata,
        "metadata_fingerprint": hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "vector_fingerprint": hashlib.sha256(
            direction.contiguous().float().numpy().tobytes()
        ).hexdigest(),
        "direction": direction,
        "unit_direction": unit,
        "direction_norm": float(direction.norm()),
        "positive_mean": direction,
        "negative_mean": torch.zeros_like(direction),
    }
    torch.save(artifact, path)
    return artifact


def _write_cast_artifact(torch, path, *, model_id="fake", behavior_layer=1):
    behavior = torch.tensor([0.0, 1.0])
    condition = torch.tensor([1.0, 0.0])
    metadata = {
        "schema_version": "agent-cast-v1",
        "method": "cast",
        "pca_method": "pca_pairwise",
        "orientation": "positive_over_negative_pair_majority",
        "model_id": model_id,
        "model_revision": "revision",
        "behavior_layer": behavior_layer,
        "condition_layer": 0,
        "condition_threshold": 0.8,
        "condition_comparator": "greater",
        "condition_comparison_mode": "mean",
        "gate_metrics": {
            "f1": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "specificity": 1.0,
            "balanced_accuracy": 1.0,
            "accuracy": 1.0,
        },
        "gate_positive_ids": ["vp0", "vp1"],
        "gate_negative_ids": ["vn0", "vn1"],
        "gate_positive_scores": [0.9, 0.95],
        "gate_negative_scores": [0.1, 0.2],
        "threshold_search": "exact_observed_midpoints",
        "behavior_pair_ids": ["b0", "b1"],
        "condition_pair_ids": ["c0", "c1"],
        "behavior_pair_count": 2,
        "condition_pair_count": 2,
        "d_model": 2,
        "benchmark": "taubench-airline-task18",
        "calibration_split": {"task": "18"},
        "sites": {"behavior_application": "block_input"},
        "source": {"repository": "IBM/activation-steering", "revision": "abc"},
        "behavior_explained_variance_ratio": 1.0,
        "condition_explained_variance_ratio": 1.0,
    }
    artifact = {
        **metadata,
        "metadata_fingerprint": hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "behavior_vector_fingerprint": hashlib.sha256(
            behavior.contiguous().float().numpy().tobytes()
        ).hexdigest(),
        "condition_vector_fingerprint": hashlib.sha256(
            condition.contiguous().float().numpy().tobytes()
        ).hexdigest(),
        "behavior_direction": behavior,
        "condition_direction": condition,
    }
    torch.save(artifact, path)
    return artifact


def _write_mera_artifact(torch, path, *, model_id="fake", layer=0):
    probe = torch.tensor([1.0, 0.0])
    metadata = {
        "schema_version": "agent-mera-v1",
        "method": "mera",
        "model_id": model_id,
        "model_revision": "revision",
        "layer": layer,
        "d_model": 2,
        "probe_fit": "linear_regression_no_intercept_logit_error",
        "target_epsilon": 1e-8,
        "training_rmse_logit": 0.1,
        "train_pair_ids": ["t0", "t1"],
        "train_pair_count": 2,
        "validation_correct_ids": ["c0", "c1"],
        "validation_failure_ids": ["f0", "f1"],
        "validation_correct_scores": [0.1, 0.2],
        "validation_failure_scores": [0.8, 0.9],
        "alpha_grid": [0.5, 0.7],
        "selected_alpha": 0.7,
        "selection_metrics": {
            "f1": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "specificity": 1.0,
            "balanced_accuracy": 1.0,
            "accuracy": 1.0,
        },
        "selection_objective": "heldout_failure_detection_f1",
        "benchmark": "taubench-airline-task18",
        "calibration_split": {"task": "18"},
        "site": "post_attention_layernorm_output_last_assistant_content",
        "source": {"repository": "MERA-steering", "revision": "abc"},
    }
    artifact = {
        **metadata,
        "metadata_fingerprint": hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "probe_vector_fingerprint": hashlib.sha256(
            probe.contiguous().float().numpy().tobytes()
        ).hexdigest(),
        "probe_vector": probe,
    }
    torch.save(artifact, path)
    return artifact


def _write_sadi_artifact(torch, path, *, model_id="fake"):
    units = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    scores = torch.tensor([4.0, 3.0])
    metadata = {
        "schema_version": "agent-sadi-v1",
        "method": "sadi_hidden",
        "model_id": model_id,
        "model_revision": "revision",
        "layers": [0, 1],
        "d_model": 2,
        "pair_ids": ["p0", "p1"],
        "pair_count": 2,
        "top_k": 2,
        "selection": "global_top_positive_mean_correct_minus_failure",
        "positive_selected_count": 2,
        "validation_pair_ids": [],
        "validation_pair_count": 0,
        "validation_positive_selected_count": None,
        "benchmark": "taubench-airline-task18",
        "calibration_split": {"task": "18"},
        "site": "mlp_output_last_assistant_content",
        "source": {"repository": "SADI", "revision": "abc"},
    }
    artifact = {
        **metadata,
        "metadata_fingerprint": hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "selected_units_fingerprint": hashlib.sha256(
            units.contiguous().float().numpy().tobytes()
        ).hexdigest(),
        "unit_scores_fingerprint": hashlib.sha256(
            scores.contiguous().float().numpy().tobytes()
        ).hexdigest(),
        "selected_units": units,
        "unit_scores": scores,
    }
    torch.save(artifact, path)
    return artifact


def _write_iti_artifact(torch, path, *, model_id="fake"):
    selected = torch.tensor([[0, 1]], dtype=torch.int64)
    directions = torch.tensor([[1.0, 0.0]])
    scales = torch.tensor([0.5])
    accuracies = torch.tensor([[0.5, 1.0]])
    weights = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    intercepts = torch.zeros(1, 2)
    metadata = {
        "schema_version": "agent-iti-v1",
        "method": "iti",
        "model_id": model_id,
        "model_revision": "revision",
        "layers": [0],
        "num_attention_heads": 2,
        "head_dim": 2,
        "d_model": 4,
        "top_k": 1,
        "train_pair_ids": ["t0", "t1"],
        "train_pair_count": 2,
        "validation_pair_ids": ["v0", "v1"],
        "validation_pair_count": 2,
        "probe": "binary_l2_logistic_regression_with_intercept",
        "regularization_c": 1.0,
        "selection": "global_top_heldout_head_accuracy",
        "direction": "center_of_mass_correct_minus_failure_train_plus_validation",
        "scale": "sample_std_projection_train_plus_validation",
        "benchmark": "taubench-airline-task18",
        "calibration_split": {"task": "18"},
        "site": "self_attn_o_proj_input_last_assistant_content",
        "source": {"repository": "honest_llama", "revision": "abc"},
    }
    tensors = {
        "selected_heads": selected,
        "head_directions": directions,
        "projection_stds": scales,
        "validation_accuracies": accuracies,
        "probe_weights": weights,
        "probe_intercepts": intercepts,
    }
    artifact = {
        **metadata,
        "metadata_fingerprint": hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        **{
            f"{name}_fingerprint": hashlib.sha256(
                tensor.contiguous().float().numpy().tobytes()
            ).hexdigest()
            for name, tensor in tensors.items()
        },
        **tensors,
    }
    torch.save(artifact, path)
    return artifact


def _write_austeer_artifact(torch, path, *, model_id="fake"):
    units = torch.tensor([[0, 1], [0, 3]], dtype=torch.int64)
    betas = torch.tensor([0.75, -0.5])
    validation = torch.tensor([1.0, -0.5])
    metadata = {
        "schema_version": "agent-austeer-v1",
        "method": "austeer",
        "model_id": model_id,
        "model_revision": "revision",
        "layers": [0],
        "d_model": 4,
        "top_k": 2,
        "window_size": 1,
        "train_pair_ids": ["t0", "t1"],
        "train_pair_count": 2,
        "validation_pair_ids": ["v0", "v1"],
        "validation_pair_count": 2,
        "selection": "global_top_absolute_signed_pair_consistency",
        "application": "activation_times_one_plus_alpha_beta",
        "validation_sign_agreement_count": 2,
        "benchmark": "taubench-airline-task18",
        "calibration_split": {"task": "18"},
        "site": "self_attn_o_proj_input_last_assistant_content",
        "source": {"repository": "AUSteer", "revision": "abc"},
    }
    tensors = {
        "selected_units": units,
        "selected_betas": betas,
        "validation_betas": validation,
    }
    artifact = {
        **metadata,
        "metadata_fingerprint": hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        **{
            f"{name}_fingerprint": hashlib.sha256(
                tensor.contiguous().float().numpy().tobytes()
            ).hexdigest()
            for name, tensor in tensors.items()
        },
        **tensors,
    }
    torch.save(artifact, path)
    return artifact


def _write_loreft_artifact(torch, path, *, model_id="fake"):
    rotations = torch.tensor([[[1.0], [0.0], [0.0], [0.0]]])
    weights = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
    biases = torch.tensor([[0.5]])
    metadata = {
        "schema_version": "agent-loreft-v1",
        "method": "loreft",
        "model_id": model_id,
        "model_revision": "revision",
        "layers": [0],
        "d_model": 4,
        "rank": 1,
        "train_example_ids": ["t0"],
        "train_example_count": 1,
        "validation_example_ids": ["v0"],
        "validation_example_count": 1,
        "formula": "h_plus_learned_source_minus_projection_times_rotation_transpose",
        "benchmark": "taubench-airline-task18",
        "training": {"optimizer": "adamw"},
        "validation_loss": 1.0,
        "site": "block_output",
        "position": "last_prompt_token",
        "source": {"repository": "stanfordnlp/pyreft", "revision": "abc"},
    }
    tensors = {
        "rotations": rotations,
        "learned_weights": weights,
        "learned_biases": biases,
    }
    artifact = {
        **metadata,
        "metadata_fingerprint": hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        **{
            f"{name}_fingerprint": hashlib.sha256(
                tensor.contiguous().float().numpy().tobytes()
            ).hexdigest()
            for name, tensor in tensors.items()
        },
        **tensors,
    }
    torch.save(artifact, path)
    return artifact


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


def test_parse_qwen36_native_xml_tool_call_and_spans():
    text = """<tool_call>
<function=set_roaming>
<parameter=phone_number>
+12025550123
</parameter>
<parameter=enabled>
true
</parameter>
</function>
</tool_call>"""

    calls = parse_qwen_tool_calls(text)

    assert len(calls) == 1
    assert calls[0].name == "set_roaming"
    assert calls[0].arguments == {
        "phone_number": "+12025550123",
        "enabled": True,
    }
    assert text[slice(*calls[0].name_span)] == "set_roaming"
    assert "+12025550123" in text[slice(*calls[0].arguments_span)]


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


def test_semantic_prediction_positions_cover_every_tool_span_token():
    text = '<tool_call>{"name":"done","arguments":{"ok":true}}</tool_call>'
    calls = parse_qwen_tool_calls(text)

    class CharacterTokenizer:
        @staticmethod
        def decode(token_ids, skip_special_tokens=False):
            return "".join(chr(token_id) for token_id in token_ids)

    positions = semantic_prediction_positions(
        CharacterTokenizer(),
        prompt_length=3,
        generated_ids=[ord(character) for character in text],
        parsed_calls=calls,
    )

    assert positions["initial_decision"] == [2]
    assert len(positions["tool_0_name"]) == len('"done"')
    assert len(positions["tool_0_arguments"]) == len('{"ok":true}')


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


def test_caa_artifact_loads_raw_and_unit_vectors(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "caa.pt"
    artifact = _write_caa_artifact(torch, path)

    raw, raw_metadata = load_caa_direction_artifact(
        path,
        model_id="fake",
        layer=0,
        d_model=2,
        scaling="raw",
    )
    unit, unit_metadata = load_caa_direction_artifact(
        path,
        model_id="fake",
        layer=0,
        d_model=2,
        scaling="unit",
    )

    torch.testing.assert_close(raw, artifact["direction"])
    torch.testing.assert_close(unit, artifact["unit_direction"])
    assert raw_metadata["vector_fingerprint"] == artifact["vector_fingerprint"]
    assert unit_metadata["vector_scaling"] == "unit"
    with pytest.raises(ValueError, match="does not match"):
        load_caa_direction_artifact(
            path,
            model_id="other",
            layer=0,
            d_model=2,
        )


def test_caa_artifact_can_be_applied_at_preregistered_wrong_layer(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "caa.pt"
    _write_caa_artifact(torch, path, model_id="fake", layer=0)

    class FakeLensModel:
        n_layers = 2
        d_model = 2
        layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

    bundle = _ModelBundle(
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(encode=lambda *_args, **_kwargs: []),
        lens_model=FakeLensModel(),
    )
    backend = InstrumentedHFBackend(
        HFBackendConfig(
            model_name_or_path="fake",
            mode=InstrumentationMode.INTERVENE,
            intervention=InterventionConfig(
                kind="steer",
                method="caa",
                layer=1,
                artifact_layer=0,
                vector_path=path,
                vector_scaling="unit",
            ),
        ),
        bundle=bundle,
    )

    with backend._intervention_hook(active=True, selection_reason="selected") as trace:
        unchanged = bundle.lens_model.layers[0](torch.zeros(1, 1, 2))
        changed = bundle.lens_model.layers[1](torch.zeros(1, 1, 2))

    torch.testing.assert_close(unchanged, torch.zeros(1, 1, 2))
    torch.testing.assert_close(changed[0, -1], torch.tensor([0.6, 0.8]))
    assert trace["layer"] == 1
    assert trace["artifact_layer"] == 0
    assert trace["direction"]["layer"] == 0


def test_cast_artifact_gate_applies_official_pre_layer_dose(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "cast.pt"
    artifact = _write_cast_artifact(torch, path)
    loaded, metadata = load_cast_artifact(
        path,
        model_id="fake",
        behavior_layer=1,
        d_model=2,
    )
    torch.testing.assert_close(loaded["behavior_direction"], artifact["behavior_direction"])
    assert metadata["condition_layer"] == 0
    score = cast_condition_similarity(
        torch.tensor([[2.0, 1.0], [4.0, 1.0]]),
        artifact["condition_direction"],
    )
    assert float(score) == pytest.approx(3.0 / (10.0**0.5))

    class FakeLensModel:
        n_layers = 2
        d_model = 2
        layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

    bundle = _ModelBundle(
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(encode=lambda *_args, **_kwargs: []),
        lens_model=FakeLensModel(),
    )
    backend = InstrumentedHFBackend(
        HFBackendConfig(
            model_name_or_path="fake",
            mode=InstrumentationMode.INTERVENE,
            intervention=InterventionConfig(
                kind="steer",
                method="cast",
                layer=1,
                artifact_layer=1,
                strength=2.0,
                vector_path=path,
            ),
        ),
        bundle=bundle,
    )
    with backend._intervention_hook(active=True, selection_reason="selected") as trace:
        prompt = torch.tensor([[[2.0, 1.0], [4.0, 1.0], [3.0, 1.0]]])
        prompt = bundle.lens_model.layers[0](prompt)
        prompt = bundle.lens_model.layers[1](prompt)
        decode = torch.tensor([[[1.0, 0.0]]])
        decode = bundle.lens_model.layers[0](decode)
        decode = bundle.lens_model.layers[1](decode)

    torch.testing.assert_close(
        prompt,
        torch.tensor([[[2.0, 3.0], [4.0, 3.0], [3.0, 3.0]]]),
    )
    torch.testing.assert_close(decode, torch.tensor([[[1.0, 2.0]]]))
    assert trace["gate_triggered"] is True
    assert trace["applied_prefill_positions"] == 3
    assert trace["applied_decode_positions"] == 1


def test_mera_artifact_applies_adaptive_post_attention_norm_dose(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "mera.pt"
    artifact = _write_mera_artifact(torch, path)
    loaded, metadata = load_mera_artifact(
        path,
        model_id="fake",
        layer=0,
        d_model=2,
    )
    torch.testing.assert_close(loaded["probe_vector"], artifact["probe_vector"])
    assert metadata["selected_alpha"] == 0.7
    delta, condition, _scores = mera_closed_form_delta(
        torch.tensor([[2.0, 1.0]]), artifact["probe_vector"], alpha=0.5
    )
    assert bool(condition[0])
    torch.testing.assert_close(delta, torch.tensor([[-2.0, 0.0]]))

    class FakeBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.post_attention_layernorm = torch.nn.Identity()

        def forward(self, hidden):
            return self.post_attention_layernorm(hidden)

    class FakeLensModel:
        n_layers = 1
        d_model = 2
        layers = torch.nn.ModuleList([FakeBlock()])

    bundle = _ModelBundle(
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(encode=lambda *_args, **_kwargs: []),
        lens_model=FakeLensModel(),
    )
    backend = InstrumentedHFBackend(
        HFBackendConfig(
            model_name_or_path="fake",
            mode=InstrumentationMode.INTERVENE,
            intervention=InterventionConfig(
                kind="steer",
                method="mera",
                layer=0,
                artifact_layer=0,
                vector_path=path,
                mera_alpha_override=0.5,
            ),
        ),
        bundle=bundle,
    )
    with backend._intervention_hook(active=True, selection_reason="selected") as trace:
        prompt = bundle.lens_model.layers[0](
            torch.tensor([[[-2.0, 1.0], [2.0, 1.0]]])
        )
        decode = bundle.lens_model.layers[0](torch.tensor([[[3.0, 1.0]]]))

    torch.testing.assert_close(prompt, torch.tensor([[[-2.0, 1.0], [0.0, 1.0]]]))
    torch.testing.assert_close(decode, torch.tensor([[[0.0, 1.0]]]))
    assert trace["applied_prefill_positions"] == 1
    assert trace["applied_decode_positions"] == 1


def test_sadi_artifact_scales_selected_mlp_units_at_prefill_only(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "sadi.pt"
    artifact = _write_sadi_artifact(torch, path)
    loaded, metadata = load_sadi_artifact(
        path,
        model_id="fake",
        d_model=2,
        top_k=2,
    )
    torch.testing.assert_close(loaded["selected_units"], artifact["selected_units"])
    assert metadata["requested_top_k"] == 2

    class FakeBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Identity()

        def forward(self, hidden):
            return self.mlp(hidden)

    class FakeLensModel:
        n_layers = 2
        d_model = 2
        layers = torch.nn.ModuleList([FakeBlock(), FakeBlock()])

    bundle = _ModelBundle(
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(encode=lambda *_args, **_kwargs: []),
        lens_model=FakeLensModel(),
    )
    backend = InstrumentedHFBackend(
        HFBackendConfig(
            model_name_or_path="fake",
            mode=InstrumentationMode.INTERVENE,
            intervention=InterventionConfig(
                kind="steer",
                method="sadi",
                layer=0,
                vector_path=path,
                sadi_top_k=2,
                strength=5.0,
                apply_decode=False,
            ),
        ),
        bundle=bundle,
    )
    prompt = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    with backend._intervention_hook(active=True, selection_reason="selected") as trace:
        layer0 = bundle.lens_model.layers[0](prompt)
        layer1 = bundle.lens_model.layers[1](prompt)
        decode = bundle.lens_model.layers[0](torch.ones(1, 1, 2))

    torch.testing.assert_close(layer0[0, -1], torch.tensor([3.0, 20.0]))
    torch.testing.assert_close(layer1[0, -1], torch.tensor([15.0, 4.0]))
    torch.testing.assert_close(decode, torch.ones(1, 1, 2))
    assert trace["applied_prefill_scalars"] == 2
    assert trace["applied_decode_scalars"] == 0


def test_iti_artifact_adds_std_scaled_direction_at_attention_head(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "iti.pt"
    artifact = _write_iti_artifact(torch, path)
    loaded, metadata = load_iti_artifact(
        path,
        model_id="fake",
        d_model=4,
        top_k=1,
    )
    torch.testing.assert_close(loaded["head_directions"], artifact["head_directions"])
    assert metadata["requested_top_k"] == 1

    class FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = torch.nn.Identity()

        def forward(self, hidden):
            return self.o_proj(hidden)

    class FakeBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = FakeAttention()

        def forward(self, hidden):
            return self.self_attn(hidden)

    class FakeLensModel:
        n_layers = 1
        d_model = 4
        layers = torch.nn.ModuleList([FakeBlock()])

    bundle = _ModelBundle(
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(encode=lambda *_args, **_kwargs: []),
        lens_model=FakeLensModel(),
    )
    backend = InstrumentedHFBackend(
        HFBackendConfig(
            model_name_or_path="fake",
            mode=InstrumentationMode.INTERVENE,
            intervention=InterventionConfig(
                kind="steer",
                method="iti",
                layer=0,
                vector_path=path,
                iti_top_k=1,
                strength=2.0,
            ),
        ),
        bundle=bundle,
    )
    with backend._intervention_hook(active=True, selection_reason="selected") as trace:
        prompt = bundle.lens_model.layers[0](torch.zeros(1, 3, 4))
        decode = bundle.lens_model.layers[0](torch.zeros(1, 1, 4))

    torch.testing.assert_close(prompt[0, -1], torch.tensor([0.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(decode[0, -1], torch.tensor([0.0, 0.0, 1.0, 0.0]))
    assert trace["applied_prefill_heads"] == 1
    assert trace["applied_decode_heads"] == 1


def test_austeer_artifact_multiplies_all_attention_aus(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "austeer.pt"
    artifact = _write_austeer_artifact(torch, path)
    loaded, metadata = load_austeer_artifact(
        path,
        model_id="fake",
        d_model=4,
        top_k=2,
    )
    torch.testing.assert_close(loaded["selected_betas"], artifact["selected_betas"])
    assert metadata["requested_top_k"] == 2

    class FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = torch.nn.Identity()

        def forward(self, hidden):
            return self.o_proj(hidden)

    class FakeBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = FakeAttention()

        def forward(self, hidden):
            return self.self_attn(hidden)

    class FakeLensModel:
        n_layers = 1
        d_model = 4
        layers = torch.nn.ModuleList([FakeBlock()])

    bundle = _ModelBundle(
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(encode=lambda *_args, **_kwargs: []),
        lens_model=FakeLensModel(),
    )
    backend = InstrumentedHFBackend(
        HFBackendConfig(
            model_name_or_path="fake",
            mode=InstrumentationMode.INTERVENE,
            intervention=InterventionConfig(
                kind="steer",
                method="austeer",
                layer=0,
                vector_path=path,
                austeer_top_k=2,
                strength=2.0,
            ),
        ),
        bundle=bundle,
    )
    prompt_input = torch.ones(1, 3, 4)
    with backend._intervention_hook(active=True, selection_reason="selected") as trace:
        prompt = bundle.lens_model.layers[0](prompt_input)
        decode = bundle.lens_model.layers[0](torch.ones(1, 1, 4))

    expected = torch.tensor([1.0, 2.5, 1.0, 0.0])
    torch.testing.assert_close(prompt[0, 0], expected)
    torch.testing.assert_close(prompt[0, -1], expected)
    torch.testing.assert_close(decode[0, -1], expected)
    assert trace["applied_prefill_scalars"] == 6
    assert trace["applied_decode_scalars"] == 2


def test_loreft_artifact_changes_only_last_prompt_position(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "loreft.pt"
    artifact = _write_loreft_artifact(torch, path)
    loaded, metadata = load_loreft_artifact(path, model_id="fake", d_model=4)
    torch.testing.assert_close(loaded["rotations"], artifact["rotations"])
    assert metadata["rank"] == 1

    class FakeLensModel:
        n_layers = 1
        d_model = 4
        layers = torch.nn.ModuleList([torch.nn.Identity()])

    bundle = _ModelBundle(
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(encode=lambda *_args, **_kwargs: []),
        lens_model=FakeLensModel(),
    )
    backend = InstrumentedHFBackend(
        HFBackendConfig(
            model_name_or_path="fake",
            mode=InstrumentationMode.INTERVENE,
            intervention=InterventionConfig(
                kind="steer",
                method="loreft",
                layer=0,
                vector_path=path,
                strength=1.0,
                apply_decode=False,
            ),
        ),
        bundle=bundle,
    )
    prompt_input = torch.tensor(
        [[[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]]]
    )
    with backend._intervention_hook(active=True, selection_reason="selected") as trace:
        prompt = bundle.lens_model.layers[0](prompt_input)
        decode = bundle.lens_model.layers[0](torch.ones(1, 1, 4))

    torch.testing.assert_close(prompt[0, 0], prompt_input[0, 0])
    torch.testing.assert_close(prompt[0, 1], torch.tensor([4.5, 4.0, 0.0, 0.0]))
    torch.testing.assert_close(decode, torch.ones(1, 1, 4))
    assert trace["applied_prefill_positions"] == 1
    assert trace["applied_decode_positions"] == 0


def test_caa_config_parses_exact_turn_and_boundary_gates(tmp_path):
    intervention = InterventionConfig.from_dict(
        {
            "kind": "steer",
            "method": "caa",
            "layer": 1,
            "vector_path": str(tmp_path / "caa.pt"),
            "turn_indices": [9],
            "boundaries": ["after_tool_result"],
            "apply_decode": False,
        }
    )

    assert intervention.vector_path == tmp_path / "caa.pt"
    assert intervention.turn_indices == (9,)
    assert intervention.boundaries == ("after_tool_result",)
    assert InstrumentedHFBackend._intervention_is_active(
        intervention,
        turn_index=9,
        boundaries=["after_tool_result"],
    ) == (True, "selected")
    assert InstrumentedHFBackend._intervention_is_active(
        intervention,
        turn_index=8,
        boundaries=["after_tool_result"],
    ) == (False, "turn_not_selected")
    assert InstrumentedHFBackend._intervention_is_active(
        intervention,
        turn_index=9,
        boundaries=["after_user_message"],
    ) == (False, "boundary_not_selected")
    with pytest.raises(ValueError, match="boundaries must be a sequence"):
        InterventionConfig.from_dict(
            {
                "kind": "steer",
                "method": "caa",
                "layer": 1,
                "vector_path": str(tmp_path / "caa.pt"),
                "boundaries": "after_user_message",
            }
        )


def test_intervention_hook_applies_exactly_once_per_generation_position():
    torch = pytest.importorskip("torch")

    class FakeLensModel:
        n_layers = 1
        d_model = 2
        layers = torch.nn.ModuleList([torch.nn.Identity()])

    bundle = _ModelBundle(
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(encode=lambda *_args, **_kwargs: []),
        lens_model=FakeLensModel(),
    )
    backend = InstrumentedHFBackend(
        HFBackendConfig(
            model_name_or_path="fake",
            mode=InstrumentationMode.INTERVENE,
            intervention=InterventionConfig(
                kind="steer",
                layer=0,
                strength=2.0,
                vector=(1.0, 2.0),
            ),
        ),
        bundle=bundle,
    )

    with backend._intervention_hook(active=True, selection_reason="selected") as trace:
        prompt = bundle.lens_model.layers[0](torch.zeros(1, 3, 2))
        decode = bundle.lens_model.layers[0](torch.zeros(1, 1, 2))

    torch.testing.assert_close(prompt[0, :-1], torch.zeros(2, 2))
    torch.testing.assert_close(prompt[0, -1], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(decode[0, -1], torch.tensor([2.0, 4.0]))
    assert trace["applied_prefill_positions"] == 1
    assert trace["applied_decode_positions"] == 1
    assert trace["direction"]["source"] == "inline_vector"


def test_inactive_intervention_has_zero_dose():
    torch = pytest.importorskip("torch")

    class FakeLensModel:
        n_layers = 1
        d_model = 2
        layers = torch.nn.ModuleList([torch.nn.Identity()])

    bundle = _ModelBundle(
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(encode=lambda *_args, **_kwargs: []),
        lens_model=FakeLensModel(),
    )
    backend = InstrumentedHFBackend(
        HFBackendConfig(
            model_name_or_path="fake",
            mode=InstrumentationMode.INTERVENE,
            intervention=InterventionConfig(
                kind="steer",
                layer=0,
                vector=(1.0, 2.0),
                turn_indices=(9,),
            ),
        ),
        bundle=bundle,
    )

    with backend._intervention_hook(
        active=False, selection_reason="turn_not_selected"
    ) as trace:
        output = bundle.lens_model.layers[0](torch.zeros(1, 2, 2))

    torch.testing.assert_close(output, torch.zeros(1, 2, 2))
    assert trace["active"] is False
    assert trace["reason"] == "turn_not_selected"
    assert trace["applied_prefill_positions"] == 0


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
    assert observe.telemetry_record["schema_version"] == "tau2-jlens-v2"
    assert observe.telemetry_record["intervention"] == {
        "requested": False,
        "active": False,
        "reason": "not_configured",
    }
    assert observe.telemetry_record["semantic_positions"]["initial_decision"] == [2]
    assert observe.telemetry_record["full_ids_sha256"] == token_ids_sha256(
        [*observe.prompt_input_ids, *observe.generated_ids]
    )
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
