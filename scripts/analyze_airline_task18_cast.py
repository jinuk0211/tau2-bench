"""Score Task-18 reward, payment binding, CAST gate, and verified dose."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tau2.agent.jlens_task18_metrics import (
    TOOL_METRIC_CSV_KEYS,
    add_tool_metric_baseline_deltas,
    trajectory_tool_metrics,
)


def _load_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = path.expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "taubench-failure-cast-v1":
        raise ValueError("unsupported Task-18 CAST config")
    output = Path(config["output_dir"]).expanduser()
    if not output.is_absolute():
        output = (path.parent / output).resolve()
    return config, output


def _tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        call
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    ]


def _payment_score(
    messages: list[dict[str, Any]], expected: dict[str, str]
) -> dict[str, Any]:
    observed: dict[str, list[str]] = {}
    for call in _tool_calls(messages):
        if call.get("name") != "update_reservation_flights":
            continue
        arguments = call.get("arguments") or {}
        reservation = str(arguments.get("reservation_id", ""))
        payment = str(arguments.get("payment_id", ""))
        if reservation in expected:
            observed.setdefault(reservation, []).append(payment)
    correct = {
        reservation: bool(observed.get(reservation))
        and observed[reservation][-1] == payment
        for reservation, payment in expected.items()
    }
    return {
        "correct_payment_mapping_count": sum(correct.values()),
        "expected_payment_mapping_count": len(expected),
        "correct_by_reservation": correct,
        "observed_payment_ids": observed,
    }


def _telemetry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "telemetry_present": False,
            "active_turns": [],
            "natural_gate_triggered": None,
            "effective_gate_triggered": None,
            "condition_score": None,
            "applied_prefill_positions": 0,
            "applied_decode_positions": 0,
        }
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
    ]
    active = [record for record in records if record.get("intervention", {}).get("active")]
    selected = active[-1]["intervention"] if active else {}
    return {
        "telemetry_present": True,
        "telemetry_turns": len(records),
        "active_turns": [int(record["turn_index"]) for record in active],
        "natural_gate_triggered": selected.get("natural_gate_triggered"),
        "effective_gate_triggered": selected.get("gate_triggered"),
        "condition_score": selected.get("condition_score"),
        "condition_threshold": selected.get("condition_threshold"),
        "condition_comparator": selected.get("condition_comparator"),
        "condition_layer": selected.get("condition_layer"),
        "behavior_layer": selected.get("behavior_layer"),
        "applied_prefill_positions": sum(
            int(record["intervention"].get("applied_prefill_positions", 0))
            for record in active
        ),
        "applied_decode_positions": sum(
            int(record["intervention"].get("applied_decode_positions", 0))
            for record in active
        ),
    }


def analyze(config_path: Path, results_root: Path) -> dict[str, Any]:
    config, output = _load_config(config_path)
    expected = config["sweep"]["expected_payment_mapping"]
    rows: list[dict[str, Any]] = []
    for result_path in sorted(results_root.glob("*/results.json")):
        condition = result_path.parent.name
        result = json.loads(result_path.read_text(encoding="utf-8"))
        simulations = result.get("simulations") or []
        if len(simulations) != 1:
            raise ValueError(f"expected one simulation in {result_path}")
        simulation = simulations[0]
        reward_info = simulation.get("reward_info") or {}
        rows.append(
            {
                "condition": condition,
                "reward": reward_info.get("reward"),
                "db_reward": (reward_info.get("db_check") or {}).get("db_reward"),
                "termination_reason": simulation.get("termination_reason"),
                **_payment_score(simulation.get("messages") or [], expected),
                **trajectory_tool_metrics(
                    simulation.get("messages") or [],
                    simulation.get("termination_reason"),
                ),
                **_telemetry(output / "telemetry" / f"{condition}.jsonl"),
            }
        )
    baseline = next((row for row in rows if row["condition"] == "baseline"), None)
    for row in rows:
        row["mapping_gain_vs_baseline"] = (
            row["correct_payment_mapping_count"]
            - baseline["correct_payment_mapping_count"]
            if baseline is not None
            else None
        )
        row["reward_gain_vs_baseline"] = (
            float(row["reward"] or 0) - float(baseline["reward"] or 0)
            if baseline is not None
            else None
        )
    add_tool_metric_baseline_deltas(rows, baseline)
    analysis = {
        "schema_version": "taubench-task18-cast-analysis-v1",
        "task_id": "18",
        "primary_metric": config["sweep"]["primary_metric"],
        "causal_turn_index": int(config["causal_turn_index"]),
        "baseline_reproduces_failure": (
            baseline is not None
            and float(baseline["reward"] or 0) == 0.0
            and baseline["correct_payment_mapping_count"] < len(expected)
        ),
        "rows": rows,
    }
    analysis_dir = output / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    json_path = analysis_dir / "task18-cast.json"
    json_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    csv_path = analysis_dir / "task18-cast.csv"
    scalar_keys = [
        "condition",
        "reward",
        "db_reward",
        "correct_payment_mapping_count",
        "expected_payment_mapping_count",
        "mapping_gain_vs_baseline",
        "reward_gain_vs_baseline",
        "termination_reason",
        *TOOL_METRIC_CSV_KEYS,
        "telemetry_present",
        "active_turns",
        "natural_gate_triggered",
        "effective_gate_triggered",
        "condition_score",
        "condition_threshold",
        "condition_comparator",
        "condition_layer",
        "behavior_layer",
        "applied_prefill_positions",
        "applied_decode_positions",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in scalar_keys})
    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "baseline_reproduces_failure": analysis["baseline_reproduces_failure"],
        "rows": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("data/simulations/task18-cast"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    print(
        json.dumps(
            analyze(args.config, args.results_root.expanduser().resolve()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
