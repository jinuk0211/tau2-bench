"""Vocabulary-trigger gating, J-lens steering and chunked prefill on a tiny Qwen3.5.

Chunked prefill must not change what is generated, and the hooks must edit the
same absolute positions whether or not the prompt prefix was cached.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
jlens = pytest.importorskip("jlens")
pytest.importorskip("jlens_causal.vocab_trigger")

from tau2.agent.jlens_backend import (  # noqa: E402
    HFBackendConfig,
    InstrumentationMode,
    InstrumentedHFBackend,
    InterventionConfig,
    _ModelBundle,
)
from tau2.data_model.message import SystemMessage, UserMessage  # noqa: E402

LAYERS = 6
WIDTH = 64


class _IdentityLens:
    def __init__(self) -> None:
        self.jacobians = {layer: torch.eye(WIDTH) for layer in range(LAYERS)}
        self.d_model = WIDTH

    def transport(self, residual, layer):
        return residual @ self.jacobians[layer].to(residual.device, residual.dtype).T


@pytest.fixture(scope="module")
def bundle():
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        )
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"Qwen3.5 tokenizer unavailable: {error}")
    config = transformers.Qwen3_5TextConfig(
        vocab_size=max(len(tokenizer), tokenizer.vocab_size),
        hidden_size=WIDTH,
        intermediate_size=128,
        num_hidden_layers=LAYERS,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        + ["linear_attention", "full_attention"],
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3_5ForCausalLM(config).eval()
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    return _ModelBundle(model=model, tokenizer=tokenizer, lens_model=lens_model, lens=_IdentityLens())


def _spec(tmp_path, threshold: float):
    path = tmp_path / f"spec_{threshold}.json"
    path.write_text(json.dumps({
        "observe_layer": 2, "site": "initial_decision", "risk_ids": [1000, 1001],
        "safe_ids": [2000, 2001], "threshold": threshold, "risk_tokens": ["r1", "r2"],
        "safe_tokens": ["s1", "s2"],
    }))
    return path


def _run(bundle, intervention=None, chunk=0):
    config = HFBackendConfig(
        model_name_or_path="tiny-qwen3.5",
        mode=InstrumentationMode.INTERVENE if intervention else InstrumentationMode.OFF,
        intervention=InterventionConfig.from_dict(intervention),
        chat_template_kwargs={"enable_thinking": False},
        generation_kwargs={"max_new_tokens": 5, "do_sample": False},
        measure=False,
        prefill_chunk=chunk,
    )
    backend = InstrumentedHFBackend(config, bundle=bundle)
    messages = [
        SystemMessage(role="system", content="Follow the airline policy. " * 12),
        UserMessage(role="user", content="Please cancel reservation ABC123 for me. " * 8),
    ]
    result = backend.generate(messages=messages, tools=[], task_id="0", turn_index=0)
    return result.generated_ids, result.telemetry_record["intervention"], len(result.prompt_input_ids)


def test_chunked_prefill_and_detect_only_do_not_change_generation(bundle, tmp_path):
    off, _, _ = _run(bundle)
    off_chunked, trace, _ = _run(bundle, chunk=16)
    detect, detect_trace, _ = _run(bundle, {
        "kind": "steer", "method": "jlens_vocab", "layer": 4, "strength": 0.0,
        "trigger_spec_path": str(_spec(tmp_path, -1e9)),
    }, chunk=16)
    assert off == off_chunked == detect
    assert trace["prefix_cache_reused"] is True
    assert detect_trace["vocab_trigger"]["triggered"] is True
    assert detect_trace["applied_prefill_positions"] == 0


@pytest.mark.parametrize("chunk", [0, 16])
def test_jlens_steering_edits_decision_and_every_decode_position(bundle, tmp_path, chunk):
    off, _, _ = _run(bundle)
    steered, trace, _ = _run(bundle, {
        "kind": "steer", "method": "jlens_vocab", "layer": 4, "strength": 2.0,
        "trigger_spec_path": str(_spec(tmp_path, -1e9)),
    }, chunk=chunk)
    assert trace["active"] and trace["vocab_trigger"]["triggered"]
    assert trace["applied_prefill_positions"] == 1
    assert trace["applied_decode_positions"] == len(steered) - 1
    assert steered != off


def test_chunked_and_unchunked_steering_agree(bundle, tmp_path):
    spec = str(_spec(tmp_path, -1e9))
    intervention = {"kind": "steer", "method": "jlens_vocab", "layer": 4, "strength": 2.0,
                    "trigger_spec_path": spec}
    assert _run(bundle, intervention, chunk=0)[0] == _run(bundle, intervention, chunk=16)[0]


def test_trigger_below_threshold_leaves_generation_alone(bundle, tmp_path):
    off, _, _ = _run(bundle)
    quiet, trace, _ = _run(bundle, {
        "kind": "steer", "method": "jlens_vocab", "layer": 4, "strength": 2.0,
        "trigger_spec_path": str(_spec(tmp_path, 1e9)),
    }, chunk=16)
    assert quiet == off
    assert not trace["active"] and trace["reason"] == "trigger_below_threshold"


def test_local_user_simulator_reuses_the_loaded_bundle(bundle):
    from tau2.agent import jlens_backend
    from tau2.utils.llm_utils import generate

    with jlens_backend._BUNDLE_CACHE_LOCK:
        saved = dict(jlens_backend._BUNDLE_CACHE)
        jlens_backend._BUNDLE_CACHE.clear()
        jlens_backend._BUNDLE_CACHE[("tiny",)] = bundle
    try:
        message = generate(
            model="localhf/tiny",
            messages=[
                SystemMessage(role="system", content="You are a customer."),
                UserMessage(role="user", content="Hi! How can I help you today?"),
            ],
            max_tokens=4,
        )
    finally:
        with jlens_backend._BUNDLE_CACHE_LOCK:
            jlens_backend._BUNDLE_CACHE.clear()
            jlens_backend._BUNDLE_CACHE.update(saved)
    assert isinstance(message.content, str)
    assert message.cost == 0.0 and message.usage["completion_tokens"] <= 4


def test_official_caa_positions_survive_chunked_prefill(bundle, tmp_path):
    from jlens_causal.official import caa, upstream
    from jlens_causal.official.artifact import build_artifact, save_artifact

    try:
        provenance = upstream.provenance("caa", caa.UPSTREAM_FILES)
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"CAA upstream unavailable: {error}")
    torch.manual_seed(1)
    artifact = build_artifact(
        method="caa", model_id="tiny-qwen3.5", model_revision="none", upstream=provenance,
        fit={"instruction_end": caa.INSTRUCTION_END, "layers": [3]}, deviations=[],
        state={"vectors": {"3": torch.randn(WIDTH)}},
    )
    path = save_artifact(torch, artifact, tmp_path / "caa.pt")
    intervention = {
        "kind": "steer", "method": "caa", "implementation": "official", "layer": 3,
        "strength": 4.0, "vector_path": str(path), "official_options": {"vector_layer": 3},
        "trigger_spec_path": str(_spec(tmp_path, -1e9)),
    }
    plain, plain_trace, prompt_len = _run(bundle, intervention, chunk=0)
    chunked, chunked_trace, _ = _run(bundle, intervention, chunk=16)
    assert chunked_trace["prefix_cache_reused"] is False
    assert plain == chunked
    assert plain_trace["applied_official_edits"] == chunked_trace["applied_official_edits"]
    start = plain_trace["official"]["from_position"]
    # Edits from the instruction end through the prompt, then every decode step.
    assert plain_trace["applied_official_edits"] == (prompt_len - start) + len(plain) - 1
