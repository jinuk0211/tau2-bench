"""The backend routes `implementation: official` interventions to jlens_causal.official."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tau2.agent.jlens_backend import (
    HFBackendConfig,
    InstrumentationMode,
    InstrumentedHFBackend,
    InterventionConfig,
)
from tau2.agent.jlens_remote_backend import hf_config_from_wire, hf_config_to_wire


def _official(**overrides):
    value = {
        "kind": "steer",
        "method": "iti",
        "implementation": "official",
        "layer": 3,
        "strength": 10.0,
        "vector_path": "/artifacts/iti-official.pt",
        "official_options": {"intervene_on_prompt": False},
        "boundaries": ["after_tool_error"],
    }
    value.update(overrides)
    return InterventionConfig.from_dict(value)


def test_official_intervention_round_trips_over_the_wire():
    config = HFBackendConfig(
        model_name_or_path="Qwen/Qwen3.5-4B",
        mode=InstrumentationMode.INTERVENE,
        intervention=_official(),
    )
    restored = hf_config_from_wire(hf_config_to_wire(config))
    assert restored.intervention.implementation == "official"
    assert restored.intervention.official_options == {"intervene_on_prompt": False}


def test_official_intervention_validation():
    with pytest.raises(ValueError):
        _official(implementation="vendored")
    with pytest.raises(ValueError):
        _official(vector_path=None)
    with pytest.raises(ValueError):
        _official(method="legacy")
    assert (
        InterventionConfig.from_dict(
            {"kind": "steer", "method": "caa", "layer": 1, "vector_path": "/a.pt"}
        ).implementation
        == "reimplementation"
    )


def test_hook_dispatches_to_official_runtime(monkeypatch):
    runtime = pytest.importorskip("jlens_causal.official.runtime")
    calls = {}

    @contextmanager
    def fake_official_steering(**kwargs):
        calls.update(kwargs)
        trace = {"applied_edits": 0}
        yield trace
        trace["applied_edits"] = 7

    monkeypatch.setattr(runtime, "official_steering", fake_official_steering)
    config = HFBackendConfig(
        model_name_or_path="Qwen/Qwen3.5-4B",
        mode=InstrumentationMode.INTERVENE,
        intervention=_official(),
    )
    bundle = SimpleNamespace(
        model=object(),
        tokenizer=object(),
        lens_model=SimpleNamespace(n_layers=32),
        lens=None,
    )
    backend = InstrumentedHFBackend(config, bundle=bundle)
    with backend._intervention_hook(
        active=True, selection_reason="selected", prompt_ids=[1, 2, 3]
    ) as trace:
        assert trace["implementation"] == "official"
    assert trace["applied_official_edits"] == 7
    assert calls["method"] == "iti" and calls["prompt_ids"] == [1, 2, 3]
    assert calls["options"] == {"intervene_on_prompt": False}
