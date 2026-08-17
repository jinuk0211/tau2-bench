import importlib.util
import json
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "run_airline_failure_steering.py"
    spec = importlib.util.spec_from_file_location("run_airline_failure_steering", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generic_runner_forces_remote_backend_and_official_full_review(monkeypatch):
    script = _load_script()
    monkeypatch.setenv("JLENS_ENDPOINT", "https://gpu.example")
    matrix = {
        "model": {"model_id": "org/model"},
        "execution": {
            "mode": "remote",
            "endpoint_env": "JLENS_ENDPOINT",
            "token_env": "JLENS_TOKEN",
        },
        "generation": {"seed": 300},
    }
    condition = {
        "name": "retry-caa",
        "agent_llm_args": {
            "jlens_mode": "intervene",
            "jlens_require_remote": True,
            "jlens_intervention": {
                "method": "caa",
                "kind": "steer",
                "layer": 20,
                "strength": 1.0,
                "vector_path": "/remote/caa.pt",
                "boundaries": ["after_tool_error"],
            },
        },
    }
    command = script.build_run_command(
        matrix,
        condition,
        split="evaluation",
        task_ids=["2", "6"],
        user_llm="review/user",
        user_llm_args={"temperature": 0},
        save_to="failure/eval/retry-caa",
        telemetry_path=Path("trace-{task_id}.jsonl"),
        num_trials=1,
        max_concurrency=1,
    )
    encoded = json.loads(command[command.index("--agent-llm-args") + 1])
    assert encoded["jlens_remote_endpoint"] == "https://gpu.example"
    assert encoded["jlens_remote_token_env"] == "JLENS_TOKEN"
    assert encoded["jlens_require_remote"]
    assert "hf_device" not in encoded
    assert command[command.index("--task-ids") + 1 : command.index("--num-trials")] == [
        "2",
        "6",
    ]
    assert "--verbose-logs" in command
    assert command[command.index("--llm-log-mode") + 1] == "all"

    review = script.build_review_command(
        Path("results.json"), review_model="review/model"
    )
    assert review[review.index("--mode") + 1] == "full"
    assert "--show-details" in review


def test_review_completion_requires_matching_keys_and_embedded_reviews(tmp_path):
    script = _load_script()
    results = tmp_path / "results.json"
    reviewed = tmp_path / "results_reviewed.json"
    results.write_text(
        json.dumps({"simulations": [{"task_id": "2", "trial": 0}]}), encoding="utf-8"
    )
    reviewed.write_text(
        json.dumps(
            {"simulations": [{"task_id": "2", "trial": 0, "review": {"errors": []}}]}
        ),
        encoding="utf-8",
    )

    assert script.reviewed_results_complete(results)

    reviewed.write_text(
        json.dumps({"simulations": [{"task_id": "6", "trial": 0, "review": {}}]}),
        encoding="utf-8",
    )
    assert not script.reviewed_results_complete(results)


def test_simulation_completion_requires_exact_task_trial_cartesian_product(tmp_path):
    script = _load_script()
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "simulations": [
                    {"task_id": task_id, "trial": trial}
                    for task_id in ("2", "6")
                    for trial in range(2)
                ]
            }
        ),
        encoding="utf-8",
    )

    assert script.simulation_results_complete(
        results,
        task_ids=["2", "6"],
        num_trials=2,
    )

    value = json.loads(results.read_text(encoding="utf-8"))
    value["simulations"].pop()
    results.write_text(json.dumps(value), encoding="utf-8")
    assert not script.simulation_results_complete(
        results,
        task_ids=["2", "6"],
        num_trials=2,
    )


def test_method_selection_runs_one_core_baseline_family_at_a_time():
    script = _load_script()
    conditions = [
        {"name": "baseline", "method": "none"},
        {"name": "retry-caa-target", "method": "caa"},
        {"name": "retry-caa-control", "method": "caa"},
        {"name": "retry-cast-target", "method": "cast"},
    ]

    selected = script.select_conditions(
        conditions,
        condition_name=None,
        method="caa",
    )

    assert [item["name"] for item in selected] == [
        "retry-caa-target",
        "retry-caa-control",
    ]
    assert script.select_conditions(
        conditions,
        condition_name=None,
        method=None,
    ) == [{"name": "baseline", "method": "none"}]
    with pytest.raises(ValueError, match="either --condition or --method"):
        script.select_conditions(
            conditions,
            condition_name="baseline",
            method="caa",
        )


def test_defaults_match_original_jlens_user_simulator():
    script = _load_script()

    args = script._parser().parse_args(["matrix.json"])

    assert args.user_llm == "gpt-5.2-2025-12-11"
    assert json.loads(args.user_llm_args) == {"reasoning_effort": "low"}
    assert args.review_model == "gpt-4.1-2025-04-14"
