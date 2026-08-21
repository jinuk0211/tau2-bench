import importlib.util
import json
from pathlib import Path


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "analyze_airline_failure_steering.py"
    spec = importlib.util.spec_from_file_location(
        "analyze_airline_failure_steering", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _simulation(reward, errors, repeated=False, user_errors=0):
    calls = [
        {"id": "a", "name": "lookup", "arguments": {"x": 1}},
    ]
    if repeated:
        calls.append({"id": "b", "name": "lookup", "arguments": {"x": 1}})
    return {
        "id": "sim",
        "task_id": "2",
        "trial": 0,
        "reward_info": {"reward": reward},
        "messages": [{"role": "assistant", "tool_calls": [call]} for call in calls],
        "review": {
            "errors": [
                {"source": "agent", "error_tags": ["wrong_sequence"]}
                for _ in range(errors)
            ]
            + [
                {"source": "user", "error_tags": ["incorrect_information"]}
                for _ in range(user_errors)
            ]
        },
    }


def test_analysis_pairs_conditions_to_baseline_and_keeps_user_agent_metrics(tmp_path):
    script = _load_script()
    matrix = {
        "manifest_fingerprint": "fingerprint",
        "conditions": [
            {
                "name": "baseline",
                "method": "none",
                "failure_category": "none",
                "control_type": "no_steer",
            },
            {
                "name": "retry-caa-s1",
                "method": "caa",
                "failure_category": "retry_without_state_change",
                "control_type": "targeted",
                "agent_llm_args": {
                    "jlens_intervention": {
                        "strength": 1.0,
                        "layer": 20,
                        "boundaries": ["after_tool_error"],
                    }
                },
            },
            {
                "name": "retry-caa-zero",
                "method": "caa",
                "failure_category": "retry_without_state_change",
                "control_type": "zero_dose",
                "agent_llm_args": {
                    "jlens_intervention": {
                        "strength": 0.0,
                        "layer": 20,
                        "boundaries": ["after_tool_error"],
                    }
                },
            },
        ],
    }
    root = tmp_path / "results"
    for condition, simulation in (
        ("baseline", _simulation(0, 1, repeated=True, user_errors=1)),
        ("retry-caa-s1", _simulation(1, 0, repeated=False, user_errors=1)),
        ("retry-caa-zero", _simulation(0, 1, repeated=True, user_errors=1)),
    ):
        directory = root / "evaluation" / condition
        directory.mkdir(parents=True)
        (directory / "results_reviewed.json").write_text(
            json.dumps({"simulations": [simulation]}), encoding="utf-8"
        )
    telemetry = tmp_path / "telemetry" / "evaluation"
    telemetry.mkdir(parents=True)
    (telemetry / "retry-caa-s1-task.jsonl").write_text(
        json.dumps(
            {
                "boundaries": ["after_tool_error"],
                "intervention": {
                    "active": True,
                    "strength": 1.0,
                    "applied_prefill_positions": 2,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (telemetry / "retry-caa-zero-task.jsonl").write_text(
        json.dumps(
            {
                "boundaries": ["after_tool_error"],
                "intervention": {
                    "active": True,
                    "strength": 0.0,
                    "applied_prefill_positions": 20,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    analysis = script.analyze(
        matrix,
        results_root=root,
        telemetry_root=tmp_path / "telemetry",
        split="evaluation",
    )
    row = next(
        item for item in analysis["paired_rows"] if item["condition"] == "retry-caa-s1"
    )
    assert row["reward_gain_vs_baseline"] == 1
    assert row["success_flip_vs_baseline"]
    assert not row["success_regression_vs_baseline"]
    assert row["success_delta_vs_baseline"] == 1
    assert row["repeat_reduction_vs_baseline"] == 1
    assert row["agent_review_error_reduction_vs_baseline"] == 1
    assert row["user_review_error_reduction_vs_baseline"] == 0
    summaries = {item["condition"]: item for item in analysis["summary"]}
    assert summaries["retry-caa-s1"]["strength"] == 1.0
    assert summaries["retry-caa-zero"]["strength"] == 0.0
    assert summaries["retry-caa-s1"]["mean_task_success"] == 1.0
    assert summaries["retry-caa-s1"]["paired_success_improved"] == 1
    assert summaries["retry-caa-s1"]["paired_success_worsened"] == 0
    assert summaries["retry-caa-s1"]["mcnemar_exact_p"] == 1.0
    assert summaries["retry-caa-zero"]["mean_task_success"] == 0.0
    assert summaries["retry-caa-s1"]["verified_total_dose"] == 2
    assert summaries["retry-caa-zero"]["verified_total_dose"] == 0


def test_exact_mcnemar_p_uses_only_discordant_pairs():
    script = _load_script()
    assert script.exact_mcnemar_p(5, 0) == 0.0625
    assert script.exact_mcnemar_p(0, 0) is None
