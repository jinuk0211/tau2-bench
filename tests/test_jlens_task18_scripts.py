import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tau2.agent.jlens_task18_metrics import (
    add_tool_metric_baseline_deltas,
    trajectory_tool_metrics,
)


def _load_script(name: str):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_task18_trajectory_metrics_use_explicit_errors_and_detect_short_cycles():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "1", "name": "lookup", "arguments": {"b": 2, "a": 1}},
                {"id": "2", "name": "update", "arguments": {"x": "A"}},
                {"id": "3", "name": "lookup", "arguments": {"a": 1, "b": 2}},
                {"id": "4", "name": "update", "arguments": {"x": "A"}},
            ],
        },
        {"role": "tool", "id": "1", "requestor": "assistant", "error": True},
        {"role": "tool", "id": "2", "requestor": "assistant", "error": False},
        {
            "role": "tool",
            "tool_messages": [
                {"role": "tool", "id": "3", "requestor": "assistant", "error": False},
                {"role": "tool", "id": "user-1", "requestor": "user", "error": True},
            ],
        },
    ]

    metrics = trajectory_tool_metrics(messages, "max_steps")

    assert metrics["total_tool_call_count"] == 4
    assert metrics["unique_tool_call_count"] == 2
    assert metrics["repeated_tool_call_count"] == 2
    assert metrics["immediate_repeat_tool_call_count"] == 0
    assert metrics["loop_pattern_count"] == 1
    assert metrics["tool_loop_detected"] is True
    assert metrics["tool_result_count"] == 3
    assert metrics["tool_call_error_count"] == 1
    assert metrics["unresolved_tool_call_count"] == 1
    assert metrics["max_steps_termination"] is True
    assert metrics["error_termination"] is True


def test_task18_tool_metric_deltas_are_positive_when_steering_reduces_failures():
    baseline = {
        "tool_call_error_count": 2,
        "repeated_tool_call_count": 3,
        "immediate_repeat_tool_call_count": 2,
        "tool_loop_detected": True,
        "max_steps_termination": True,
    }
    steered = {
        "tool_call_error_count": 0,
        "repeated_tool_call_count": 1,
        "immediate_repeat_tool_call_count": 0,
        "tool_loop_detected": False,
        "max_steps_termination": False,
    }
    rows = [baseline, steered]

    add_tool_metric_baseline_deltas(rows, baseline)

    assert steered["tool_error_reduction_vs_baseline"] == 2
    assert steered["repeat_reduction_vs_baseline"] == 2
    assert steered["immediate_repeat_reduction_vs_baseline"] == 2
    assert steered["loop_eliminated_vs_baseline"] is True
    assert steered["max_steps_eliminated_vs_baseline"] is True


@pytest.mark.parametrize(
    "method",
    ["caa", "cast", "mera", "sadi", "iti", "austeer", "loreft"],
)
def test_all_task18_analyzers_report_tool_failure_reductions(tmp_path, method):
    script = _load_script(f"analyze_airline_task18_{method}.py")
    output_dir = tmp_path / method / "output"
    results_root = tmp_path / method / "results"
    config_path = tmp_path / method / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "schema_version": f"taubench-failure-{method}-v1",
                "output_dir": str(output_dir),
                "causal_turn_index": 9,
                "sweep": {
                    "primary_metric": "reward",
                    "expected_payment_mapping": {"A": "gift_a"},
                },
            }
        ),
        encoding="utf-8",
    )

    baseline_messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "b1",
                    "name": "update_reservation_flights",
                    "arguments": {"reservation_id": "A", "payment_id": "wrong"},
                },
                {
                    "id": "b2",
                    "name": "update_reservation_flights",
                    "arguments": {"reservation_id": "A", "payment_id": "wrong"},
                },
            ],
        },
        {"role": "tool", "id": "b1", "error": True, "requestor": "assistant"},
        {"role": "tool", "id": "b2", "error": False, "requestor": "assistant"},
    ]
    steered_messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "s1",
                    "name": "update_reservation_flights",
                    "arguments": {"reservation_id": "A", "payment_id": "gift_a"},
                }
            ],
        },
        {"role": "tool", "id": "s1", "error": False, "requestor": "assistant"},
    ]
    for condition, reward, termination, messages in [
        ("baseline", 0.0, "max_steps", baseline_messages),
        ("steered", 1.0, "agent_stop", steered_messages),
    ]:
        condition_dir = results_root / condition
        condition_dir.mkdir(parents=True)
        (condition_dir / "results.json").write_text(
            json.dumps(
                {
                    "simulations": [
                        {
                            "termination_reason": termination,
                            "reward_info": {
                                "reward": reward,
                                "db_check": {"db_reward": reward},
                            },
                            "messages": messages,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    result = script.analyze(config_path, results_root)
    rows = {row["condition"]: row for row in result["rows"]}

    assert rows["baseline"]["tool_call_error_count"] == 1
    assert rows["baseline"]["repeated_tool_call_count"] == 1
    assert rows["steered"]["tool_error_reduction_vs_baseline"] == 1
    assert rows["steered"]["repeat_reduction_vs_baseline"] == 1
    assert rows["steered"]["max_steps_eliminated_vs_baseline"] is True
    csv_text = Path(result["csv_path"]).read_text(encoding="utf-8")
    assert "tool_call_error_count" in csv_text
    assert "repeat_reduction_vs_baseline" in csv_text


def test_task18_payment_metric_counts_per_reservation_binding():
    script = _load_script("analyze_airline_task18_caa.py")
    expected = {
        "A": "gift_a",
        "B": "card_b",
        "C": "gift_c",
    }
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "update_reservation_flights",
                    "arguments": {"reservation_id": "A", "payment_id": "gift_a"},
                },
                {
                    "name": "update_reservation_flights",
                    "arguments": {"reservation_id": "B", "payment_id": "gift_a"},
                },
                {
                    "name": "update_reservation_flights",
                    "arguments": {"reservation_id": "C", "payment_id": "gift_c"},
                },
            ],
        }
    ]

    score = script._payment_score(messages, expected)

    assert score["correct_payment_mapping_count"] == 2
    assert score["correct_by_reservation"] == {"A": True, "B": False, "C": True}


def test_task18_condition_builder_allows_baseline_before_vector_extraction(tmp_path):
    script = _load_script("run_airline_task18_caa.py")
    config = {
        "model": {
            "model_id": "Qwen/Qwen3.5-4B",
            "model_revision": "a" * 40,
            "dtype": "bfloat16",
        },
        "causal_turn_index": 9,
        "causal_boundary": "after_user_message",
        "sweep": {
            "layers": [20],
            "alphas": [0.0, 1.0],
            "random_seeds": [11, 23, 37],
            "wrong_layer": 4,
        },
    }

    conditions = script.build_conditions(config, tmp_path)

    assert [condition.name for condition in conditions] == ["baseline"]
    assert conditions[0].agent_args["jlens_mode"] == "observe"


def test_task18_cast_builder_includes_gate_and_site_controls(tmp_path):
    script = _load_script("run_airline_task18_cast.py")
    config = {
        "model": {
            "model_id": "Qwen/Qwen3.5-4B",
            "model_revision": "a" * 40,
            "dtype": "bfloat16",
        },
        "generation": {"seed": 626729, "do_sample": True},
        "extraction": {
            "behavior_layers": [20],
            "condition_layers": [4, 8],
        },
        "causal_turn_index": 9,
        "causal_boundary": "after_user_message",
        "sweep": {
            "layers": [20],
            "alphas": [0.0, 1.0],
            "include_ungated_control": True,
            "include_complement_gate_control": True,
        },
    }

    conditions = script.build_conditions(config, tmp_path)

    assert [condition.name for condition in conditions] == [
        "baseline",
        "cast-positive-l20-a1",
        "cast-negative-l20-a1",
        "cast-decision-only-l20-a1",
        "cast-ungated-l20-a1",
        "cast-complement-gate-l20-a1",
    ]
    positive = conditions[1].agent_args["jlens_intervention"]
    assert positive["turn_indices"] == [9]
    assert positive["boundaries"] == ["after_user_message"]
    assert conditions[3].agent_args["jlens_intervention"]["cast_prefill_mode"] == (
        "decision_only"
    )
    assert conditions[4].agent_args["jlens_intervention"]["cast_gate_override"] is True
    assert conditions[5].agent_args["jlens_intervention"]["cast_invert_comparator"] is True


def test_task18_cast_payment_metric_counts_per_reservation_binding():
    script = _load_script("analyze_airline_task18_cast.py")
    expected = {"A": "gift_a", "B": "card_b"}
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "update_reservation_flights",
                    "arguments": {"reservation_id": "A", "payment_id": "gift_a"},
                },
                {
                    "name": "update_reservation_flights",
                    "arguments": {"reservation_id": "B", "payment_id": "gift_a"},
                },
            ],
        }
    ]

    score = script._payment_score(messages, expected)

    assert score["correct_payment_mapping_count"] == 1
    assert score["correct_by_reservation"] == {"A": True, "B": False}


def test_task18_mera_builder_allows_baseline_before_probe_extraction(tmp_path):
    script = _load_script("run_airline_task18_mera.py")
    config = {
        "model": {
            "model_id": "Qwen/Qwen3.5-4B",
            "model_revision": "a" * 40,
            "dtype": "bfloat16",
        },
        "generation": {"seed": 626729, "do_sample": True},
        "extraction": {"layers": [20]},
        "causal_turn_index": 9,
        "causal_boundary": "after_user_message",
        "sweep": {
            "layers": [20],
            "uncalibrated_alphas": [0.3, 0.5, 0.7],
            "random_seeds": [11, 23, 37],
        },
    }

    conditions = script.build_conditions(config, tmp_path)

    assert [condition.name for condition in conditions] == ["baseline"]
    assert conditions[0].agent_args["jlens_mode"] == "observe"


def test_task18_mera_payment_metric_counts_per_reservation_binding():
    script = _load_script("analyze_airline_task18_mera.py")
    expected = {"A": "gift_a", "B": "card_b"}
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "update_reservation_flights",
                    "arguments": {"reservation_id": "A", "payment_id": "gift_a"},
                },
                {
                    "name": "update_reservation_flights",
                    "arguments": {"reservation_id": "B", "payment_id": "gift_a"},
                },
            ],
        }
    ]

    score = script._payment_score(messages, expected)

    assert score["correct_payment_mapping_count"] == 1


def test_task18_sadi_builder_allows_baseline_before_unit_extraction(tmp_path):
    script = _load_script("run_airline_task18_sadi.py")
    config = {
        "model": {
            "model_id": "Qwen/Qwen3.5-4B",
            "model_revision": "a" * 40,
            "dtype": "bfloat16",
        },
        "generation": {"seed": 626729, "do_sample": True},
        "extraction": {"layers": [20, 24]},
        "causal_turn_index": 9,
        "causal_boundary": "after_user_message",
        "sweep": {
            "top_k_values": [4, 10, 20],
            "strengths": [5.0, 10.0, 20.0],
            "primary_top_k": 10,
            "primary_strength": 10.0,
            "random_seeds": [11, 23, 37],
        },
    }

    conditions = script.build_conditions(config, tmp_path)

    assert [condition.name for condition in conditions] == ["baseline"]
    assert conditions[0].agent_args["jlens_mode"] == "observe"


def test_task18_sadi_payment_metric_counts_per_reservation_binding():
    script = _load_script("analyze_airline_task18_sadi.py")
    expected = {"A": "gift_a", "B": "card_b"}
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "update_reservation_flights",
                    "arguments": {"reservation_id": "A", "payment_id": "gift_a"},
                },
                {
                    "name": "update_reservation_flights",
                    "arguments": {"reservation_id": "B", "payment_id": "gift_a"},
                },
            ],
        }
    ]

    score = script._payment_score(messages, expected)

    assert score["correct_payment_mapping_count"] == 1
    assert score["correct_by_reservation"] == {"A": True, "B": False}


def test_task18_iti_builder_allows_baseline_before_head_extraction(tmp_path):
    script = _load_script("run_airline_task18_iti.py")
    config = {
        "model": {
            "model_id": "Qwen/Qwen3.5-4B",
            "model_revision": "a" * 40,
            "dtype": "bfloat16",
        },
        "generation": {"seed": 626729, "do_sample": True},
        "extraction": {"layers": [20, 24]},
        "causal_turn_index": 9,
        "causal_boundary": "after_user_message",
        "sweep": {
            "top_k_values": [2, 4, 8],
            "alphas": [5.0, 10.0, 15.0],
            "primary_top_k": 4,
            "primary_alpha": 15.0,
            "random_seeds": [11, 23, 37],
        },
    }

    conditions = script.build_conditions(config, tmp_path)

    assert [condition.name for condition in conditions] == ["baseline"]
    assert conditions[0].agent_args["jlens_mode"] == "observe"


def test_task18_iti_payment_metric_counts_per_reservation_binding():
    script = _load_script("analyze_airline_task18_iti.py")
    expected = {"A": "gift_a", "B": "card_b"}
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "update_reservation_flights",
                    "arguments": {"reservation_id": "A", "payment_id": "gift_a"},
                },
                {
                    "name": "update_reservation_flights",
                    "arguments": {"reservation_id": "B", "payment_id": "gift_a"},
                },
            ],
        }
    ]

    score = script._payment_score(messages, expected)

    assert score["correct_payment_mapping_count"] == 1
    assert score["correct_by_reservation"] == {"A": True, "B": False}


def test_task18_austeer_builder_allows_baseline_before_au_extraction(tmp_path):
    script = _load_script("run_airline_task18_austeer.py")
    config = {
        "model": {
            "model_id": "Qwen/Qwen3.5-4B",
            "model_revision": "a" * 40,
            "dtype": "bfloat16",
        },
        "generation": {"seed": 626729, "do_sample": True},
        "extraction": {"layers": [20, 24]},
        "causal_turn_index": 9,
        "causal_boundary": "after_user_message",
        "sweep": {
            "top_k_values": [25, 50, 100],
            "alphas": [5.0, 10.0, 15.0],
            "primary_top_k": 100,
            "primary_alpha": 15.0,
            "random_seeds": [11, 23, 37],
        },
    }

    conditions = script.build_conditions(config, tmp_path)

    assert [condition.name for condition in conditions] == ["baseline"]
    assert conditions[0].agent_args["jlens_mode"] == "observe"


def test_task18_austeer_payment_metric_counts_per_reservation_binding():
    script = _load_script("analyze_airline_task18_austeer.py")
    expected = {"A": "gift_a", "B": "card_b"}
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "update_reservation_flights",
                    "arguments": {"reservation_id": "A", "payment_id": "gift_a"},
                },
                {
                    "name": "update_reservation_flights",
                    "arguments": {"reservation_id": "B", "payment_id": "gift_a"},
                },
            ],
        }
    ]

    score = script._payment_score(messages, expected)

    assert score["correct_payment_mapping_count"] == 1
    assert score["correct_by_reservation"] == {"A": True, "B": False}


def test_task18_loreft_builder_allows_baseline_before_training(tmp_path):
    script = _load_script("run_airline_task18_loreft.py")
    config = {
        "model": {
            "model_id": "Qwen/Qwen3.5-4B",
            "model_revision": "a" * 40,
            "dtype": "bfloat16",
        },
        "generation": {"seed": 626729, "do_sample": True},
        "training": {
            "layers": [20, 24],
            "ranks": [1, 4, 8],
            "primary_rank": 4,
            "random_seeds": [11, 23, 37],
        },
        "causal_turn_index": 9,
        "causal_boundary": "after_user_message",
    }

    conditions = script.build_conditions(config, tmp_path)

    assert [condition.name for condition in conditions] == ["baseline"]
    assert conditions[0].agent_args["jlens_mode"] == "observe"


def test_task18_loreft_payment_metric_counts_per_reservation_binding():
    script = _load_script("analyze_airline_task18_loreft.py")
    expected = {"A": "gift_a", "B": "card_b"}
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "update_reservation_flights",
                    "arguments": {"reservation_id": "A", "payment_id": "gift_a"},
                },
                {
                    "name": "update_reservation_flights",
                    "arguments": {"reservation_id": "B", "payment_id": "gift_a"},
                },
            ],
        }
    ]

    score = script._payment_score(messages, expected)

    assert score["correct_payment_mapping_count"] == 1
    assert score["correct_by_reservation"] == {"A": True, "B": False}
