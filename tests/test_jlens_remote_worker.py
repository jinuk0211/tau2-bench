import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "jlens_remote_worker.py"
    spec = importlib.util.spec_from_file_location("jlens_remote_worker", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_worker_preflight_checks_cuda_remote_hf_and_artifacts(tmp_path):
    script = _load_script()
    artifact = tmp_path / "caa.pt"
    artifact.write_bytes(b"artifact")

    ready = script.worker_preflight_report(
        [str(artifact)],
        {"HF_TOKEN": "secret"},
        cuda_available=True,
        cuda_device_count=1,
        loaded_backends=0,
    )
    assert ready["status"] == "ok"
    assert ready["huggingface_token_present"]
    assert ready["missing_artifacts"] == []

    missing = script.worker_preflight_report(
        [str(tmp_path / "missing.pt")],
        {},
        cuda_available=True,
        cuda_device_count=1,
        loaded_backends=0,
    )
    assert missing["status"] == "not_ready"
    assert len(missing["missing_artifacts"]) == 1


def test_remote_preflight_endpoint_is_authenticated_and_does_not_load_model(
    tmp_path, monkeypatch
):
    script = _load_script()
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"artifact")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
            )
        ),
    )
    client = TestClient(script.create_app(token="worker-secret"))

    unauthorized = client.post(
        "/v1/preflight",
        json={"artifact_paths": [str(artifact)]},
    )
    assert unauthorized.status_code == 401

    response = client.post(
        "/v1/preflight",
        headers={"Authorization": "Bearer worker-secret"},
        json={"artifact_paths": [str(artifact)]},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["loaded_backends"] == 0


def test_generation_heartbeat_reports_task_turn_and_elapsed(monkeypatch):
    script = _load_script()
    messages = []

    class StopAfterOneHeartbeat:
        calls = 0

        def wait(self, _interval):
            self.calls += 1
            return self.calls > 1

    monkeypatch.setattr(
        script.LOGGER,
        "info",
        lambda message, *args: messages.append(message % args),
    )
    script._log_generation_heartbeat(
        StopAfterOneHeartbeat(),
        task_id="33",
        turn_index=4,
        started_at=script.time.perf_counter() - 5,
        interval_seconds=0,
    )

    assert len(messages) == 1
    assert "task=33" in messages[0]
    assert "turn=4" in messages[0]
    assert "elapsed=" in messages[0]


def test_cuda_cache_release_collects_and_empties_cache(monkeypatch):
    script = _load_script()
    calls = []
    monkeypatch.setattr(script.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                empty_cache=lambda: calls.append("cuda"),
            )
        ),
    )

    script._release_cuda_cache()

    assert calls == ["gc", "cuda"]
