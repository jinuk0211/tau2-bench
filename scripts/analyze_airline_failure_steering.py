"""Analyze paired TauBench failure-mode steering outcomes and verified dose."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from tau2.agent.jlens_failure_metrics import simulation_failure_metrics
from tau2.agent.jlens_failure_protocol import load_failure_steering_matrix


def exact_mcnemar_p(improved: int, worsened: int) -> float | None:
    """Two-sided exact McNemar p-value for paired task-success changes."""
    discordant = int(improved) + int(worsened)
    if discordant == 0:
        return None
    tail = sum(
        math.comb(discordant, index) for index in range(min(improved, worsened) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _results_path(root: Path, split: str, condition: str) -> tuple[Path, bool]:
    directory = root / split / condition
    reviewed = directory / "results_reviewed.json"
    if reviewed.is_file():
        return reviewed, True
    return directory / "results.json", False


def _simulation_key(simulation: dict[str, Any]) -> tuple[str, int]:
    return str(simulation.get("task_id", "")), int(simulation.get("trial", 0) or 0)


def _telemetry_dose(path: Path) -> dict[str, Any]:
    records = 0
    active_records = 0
    nonzero_dose_records = 0
    total_dose = 0
    weighted_dose = 0.0
    active_boundaries: set[str] = set()
    for telemetry in path.parent.glob(path.name):
        with telemetry.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                records += 1
                trace = record.get("controller") or record.get("intervention") or {}
                if trace.get("active"):
                    active_records += 1
                    active_boundaries.update(record.get("boundaries") or [])
                applied = sum(
                    int(value)
                    for key, value in trace.items()
                    if key.startswith("applied_")
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                )
                actual_dose = float(trace.get("cumulative_dose", 0.0) or 0.0)
                strength = float(trace.get("strength", 0.0) or 0.0)
                verified_dose = actual_dose if actual_dose > 0 else applied * abs(strength)
                if applied and verified_dose > 0.0:
                    nonzero_dose_records += 1
                    total_dose += applied
                    weighted_dose += verified_dose
    return {
        "telemetry_records": records,
        "active_intervention_records": active_records,
        "nonzero_dose_records": nonzero_dose_records,
        "verified_total_dose": total_dose,
        "verified_absolute_weighted_dose": weighted_dose,
        "active_boundaries": sorted(active_boundaries),
    }


def analyze(
    matrix: dict[str, Any],
    *,
    results_root: Path,
    telemetry_root: Path,
    split: str,
) -> dict[str, Any]:
    conditions = {item["name"]: item for item in matrix["conditions"]}
    baseline_path, baseline_reviewed = _results_path(results_root, split, "baseline")
    if not baseline_path.is_file():
        raise FileNotFoundError(f"baseline results are missing: {baseline_path}")
    baseline_results = _read_json(baseline_path)
    baselines: dict[tuple[str, int], dict[str, Any]] = {}
    for simulation in baseline_results.get("simulations") or []:
        key = _simulation_key(simulation)
        if key in baselines:
            raise ValueError(f"duplicate baseline simulation key {key}")
        baselines[key] = simulation_failure_metrics(simulation)
    rows: list[dict[str, Any]] = []
    condition_dose: dict[str, dict[str, Any]] = {}
    missing_condition_results: list[str] = []
    unpaired_simulations: dict[str, int] = {}
    for name, condition in conditions.items():
        path, reviewed = _results_path(results_root, split, name)
        if not path.is_file():
            missing_condition_results.append(name)
            continue
        result = _read_json(path)
        telemetry_pattern = telemetry_root / split / f"{name}-*.jsonl"
        condition_dose[name] = _telemetry_dose(telemetry_pattern)
        intervention = (
            condition.get("agent_llm_args", {}).get("jlens_intervention") or {}
        )
        controller = condition.get("agent_llm_args", {}).get("jlens_controller") or {}
        seen_condition_keys: set[tuple[str, int]] = set()
        for simulation in result.get("simulations") or []:
            key = _simulation_key(simulation)
            if key in seen_condition_keys:
                raise ValueError(f"duplicate simulation key {key} in condition {name}")
            seen_condition_keys.add(key)
            baseline = baselines.get(key)
            if baseline is None:
                unpaired_simulations[name] = unpaired_simulations.get(name, 0) + 1
                continue
            metrics = simulation_failure_metrics(simulation)
            row = {
                "condition": name,
                "method": condition["method"],
                "failure_category": condition["failure_category"],
                "control_type": condition["control_type"],
                "strength": intervention.get("strength", controller.get("fixed_strength")),
                "layer": intervention.get("layer", controller.get("layer_override")),
                "boundaries": intervention.get("boundaries") or [],
                "task_id": key[0],
                "trial": key[1],
                "review_available": reviewed,
                "baseline_review_available": baseline_reviewed,
                **metrics,
                "reward_gain_vs_baseline": (
                    None
                    if metrics["task_reward"] is None or baseline["task_reward"] is None
                    else metrics["task_reward"] - baseline["task_reward"]
                ),
                "success_flip_vs_baseline": bool(
                    metrics["task_success"] and not baseline["task_success"]
                ),
                "success_regression_vs_baseline": bool(
                    baseline["task_success"] and not metrics["task_success"]
                ),
                "success_delta_vs_baseline": int(metrics["task_success"])
                - int(baseline["task_success"]),
                "tool_error_reduction_vs_baseline": (
                    baseline["tool_call_error_count"] - metrics["tool_call_error_count"]
                ),
                "repeat_reduction_vs_baseline": (
                    baseline["repeated_tool_call_count"]
                    - metrics["repeated_tool_call_count"]
                ),
                "immediate_repeat_reduction_vs_baseline": (
                    baseline["immediate_repeat_tool_call_count"]
                    - metrics["immediate_repeat_tool_call_count"]
                ),
                "loop_pattern_reduction_vs_baseline": (
                    baseline["loop_pattern_count"] - metrics["loop_pattern_count"]
                ),
                "tool_call_reduction_vs_baseline": (
                    baseline["total_tool_call_count"] - metrics["total_tool_call_count"]
                ),
                "normal_tool_suppression_vs_baseline": int(
                    baseline["total_tool_call_count"] > 0
                    and metrics["total_tool_call_count"] == 0
                ),
                "loop_eliminated_vs_baseline": bool(
                    baseline["tool_loop_detected"] and not metrics["tool_loop_detected"]
                ),
                "max_steps_eliminated_vs_baseline": bool(
                    baseline["max_steps_termination"]
                    and not metrics["max_steps_termination"]
                ),
                "agent_review_error_reduction_vs_baseline": (
                    baseline["agent_review_error_count"]
                    - metrics["agent_review_error_count"]
                    if baseline_reviewed and reviewed
                    else None
                ),
                "user_review_error_reduction_vs_baseline": (
                    baseline["user_review_error_count"]
                    - metrics["user_review_error_count"]
                    if baseline_reviewed and reviewed
                    else None
                ),
            }
            rows.append(row)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    metrics = (
        "task_success",
        "task_reward",
        "agent_review_error_count",
        "user_review_error_count",
        "tool_call_error_count",
        "total_tool_call_count",
        "repeated_tool_call_count",
        "immediate_repeat_tool_call_count",
        "loop_pattern_count",
        "abstention_count",
        "candidate_validation_failure_count",
        "reward_gain_vs_baseline",
        "success_flip_vs_baseline",
        "success_regression_vs_baseline",
        "success_delta_vs_baseline",
        "tool_error_reduction_vs_baseline",
        "repeat_reduction_vs_baseline",
        "immediate_repeat_reduction_vs_baseline",
        "loop_pattern_reduction_vs_baseline",
        "tool_call_reduction_vs_baseline",
        "normal_tool_suppression_vs_baseline",
        "loop_eliminated_vs_baseline",
        "max_steps_eliminated_vs_baseline",
        "agent_review_error_reduction_vs_baseline",
        "user_review_error_reduction_vs_baseline",
    )
    summary: list[dict[str, Any]] = []
    for condition_name, items in sorted(grouped.items()):
        first = items[0]
        dose = condition_dose[condition_name]
        entry: dict[str, Any] = {
            "condition": condition_name,
            "method": first["method"],
            "failure_category": first["failure_category"],
            "control_type": first["control_type"],
            "strength": first["strength"],
            "layer": first["layer"],
            "boundaries": first["boundaries"],
            "n": len(items),
            "paired_coverage": len(items) / len(baselines) if baselines else 0.0,
            "review_coverage": sum(item["review_available"] for item in items)
            / len(items),
            "verified_total_dose": dose["verified_total_dose"],
            "verified_absolute_weighted_dose": dose["verified_absolute_weighted_dose"],
            "nonzero_dose_records": dose["nonzero_dose_records"],
            "active_boundaries": dose["active_boundaries"],
        }
        success_improved = sum(item["success_flip_vs_baseline"] for item in items)
        success_worsened = sum(item["success_regression_vs_baseline"] for item in items)
        entry.update(
            {
                "paired_success_improved": success_improved,
                "paired_success_worsened": success_worsened,
                "paired_success_unchanged": len(items)
                - success_improved
                - success_worsened,
                "mcnemar_exact_p": exact_mcnemar_p(
                    success_improved,
                    success_worsened,
                ),
            }
        )
        for metric in metrics:
            values = [item[metric] for item in items if item[metric] is not None]
            entry[f"mean_{metric}"] = (
                sum(float(value) for value in values) / len(values) if values else None
            )
        summary.append(entry)
    incomplete_condition_pairing = [
        entry["condition"] for entry in summary if entry["paired_coverage"] != 1.0
    ]
    return {
        "schema_version": "taubench-failure-steering-analysis-v1",
        "matrix_fingerprint": matrix.get("matrix_fingerprint"),
        "manifest_fingerprint": matrix["manifest_fingerprint"],
        "split": split,
        "baseline_review_available": baseline_reviewed,
        "baseline_simulations": len(baselines),
        "paired_rows": rows,
        "summary": summary,
        "condition_dose": condition_dose,
        "missing_condition_results": missing_condition_results,
        "unpaired_simulations_by_condition": unpaired_simulations,
        "incomplete_condition_pairing": incomplete_condition_pairing,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument(
        "--results-root", type=Path, default=Path("data/simulations/failure-steering")
    )
    parser.add_argument(
        "--telemetry-root",
        type=Path,
        default=Path("data/jlens-telemetry/failure-steering"),
    )
    parser.add_argument(
        "--split", choices=["train", "validation", "evaluation"], default="evaluation"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/analysis/failure-steering")
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail unless every compiled condition covers the baseline task/trial keys",
    )
    args = parser.parse_args()
    analysis = analyze(
        load_failure_steering_matrix(args.matrix),
        results_root=args.results_root,
        telemetry_root=args.telemetry_root,
        split=args.split,
    )
    if args.require_complete and (
        analysis["missing_condition_results"]
        or analysis["unpaired_simulations_by_condition"]
        or analysis["incomplete_condition_pairing"]
    ):
        raise RuntimeError(
            "failure steering analysis is incomplete; inspect missing_condition_results, "
            "unpaired_simulations_by_condition, and incomplete_condition_pairing"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.split}.json"
    output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_dir / f"{args.split}-paired.csv", analysis["paired_rows"])
    _write_csv(args.output_dir / f"{args.split}-summary.csv", analysis["summary"])
    print(
        json.dumps(
            {"output": str(output), "paired_rows": len(analysis["paired_rows"])},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
