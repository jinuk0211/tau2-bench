import importlib.util
from pathlib import Path


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "preflight_failure_steering.py"
    spec = importlib.util.spec_from_file_location("preflight_failure_steering", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_preflight_reports_presence_without_secret_values():
    script = _load_script()
    matrix = {
        "matrix_fingerprint": "fingerprint",
        "execution": {
            "endpoint_env": "JLENS_REMOTE_ENDPOINT",
            "token_env": "JLENS_REMOTE_TOKEN",
        },
        "conditions": [
            {"method": method}
            for method in ["caa", "mera", "sadi", "iti", "austeer"]
        ],
    }
    environment = {
        "JLENS_REMOTE_ENDPOINT": "https://secret-host.example",
        "JLENS_REMOTE_TOKEN": "secret-token",
        "OPENAI_API_KEY": "secret-review",
        "HF_TOKEN": "secret-hf",
    }

    report = script.preflight_report(
        matrix,
        environment,
        health={
            "status": "ok",
            "cuda_available": True,
            "missing_artifacts": [],
            "huggingface_token_present": True,
        },
        method="caa",
        require_hf_token=True,
    )

    assert report["ready"]
    rendered = str(report)
    assert "secret-token" not in rendered
    assert "secret-review" not in rendered
    assert "secret-hf" not in rendered


def test_artifact_selection_is_method_scoped_and_baseline_is_empty():
    script = _load_script()
    matrix = {
        "conditions": [
            {
                "method": "caa",
                "agent_llm_args": {
                    "jlens_intervention": {"vector_path": "/artifacts/caa.pt"}
                },
            },
            {
                "method": "cast",
                "agent_llm_args": {
                    "jlens_intervention": {"vector_path": "/artifacts/cast.pt"}
                },
            },
        ]
    }

    assert script.artifact_paths_for_method(matrix, "baseline") == []
    assert script.artifact_paths_for_method(matrix, "caa") == ["/artifacts/caa.pt"]
    assert script.artifact_paths_for_method(matrix, "all") == [
        "/artifacts/caa.pt",
        "/artifacts/cast.pt",
    ]


def test_five_baselines_plus_jservo_are_supported():
    script = _load_script()
    methods = ["caa", "mera", "sadi", "iti", "austeer", "jservo"]
    matrix = {
        "matrix_fingerprint": "fingerprint",
        "execution": {
            "endpoint_env": "JLENS_REMOTE_ENDPOINT",
            "token_env": "JLENS_REMOTE_TOKEN",
        },
        "conditions": [{"method": method} for method in methods],
    }

    report = script.preflight_report(
        matrix,
        {
            "JLENS_REMOTE_ENDPOINT": "http://127.0.0.1:8877",
            "JLENS_REMOTE_TOKEN": "token",
            "OPENAI_API_KEY": "review-key",
        },
        health={
            "status": "ok",
            "cuda_available": True,
            "missing_artifacts": [],
        },
        method="caa",
        require_hf_token=False,
    )

    assert report["ready"]
    assert report["checks"]["matrix_methods_supported"]


def test_skipped_remote_check_does_not_claim_artifacts_are_ready():
    script = _load_script()
    matrix = {
        "matrix_fingerprint": "fingerprint",
        "execution": {
            "endpoint_env": "JLENS_REMOTE_ENDPOINT",
            "token_env": "JLENS_REMOTE_TOKEN",
        },
        "conditions": [
            {"method": method}
            for method in ["caa", "mera", "sadi", "iti", "austeer"]
        ],
    }

    report = script.preflight_report(
        matrix,
        {},
        health={"status": "skipped"},
        method="baseline",
        require_hf_token=False,
    )

    assert not report["checks"]["remote_artifacts_ready"]
    assert not report["ready"]
