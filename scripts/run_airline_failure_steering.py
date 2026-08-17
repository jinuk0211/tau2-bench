"""Run a generic remote-only failure-mode steering matrix on TauBench airline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from tau2.agent.jlens_failure_protocol import load_failure_steering_matrix


def _load_matrix(path: Path) -> dict[str, Any]:
    return load_failure_steering_matrix(path)


def _condition_args(
    matrix: dict[str, Any],
    condition: dict[str, Any],
    *,
    telemetry_path: Path,
    allow_missing_endpoint: bool,
) -> dict[str, Any]:
    execution = matrix["execution"]
    endpoint_env = str(execution["endpoint_env"])
    endpoint = os.environ.get(endpoint_env)
    if not endpoint and not allow_missing_endpoint:
        raise RuntimeError(f"missing remote endpoint in {endpoint_env}")
    args = dict(condition["agent_llm_args"])
    args.update(
        {
            "jlens_remote_endpoint": endpoint or f"https://<{endpoint_env}>",
            "jlens_remote_token_env": str(execution["token_env"]),
            "jlens_remote_timeout_seconds": float(
                execution.get("timeout_seconds", 600.0)
            ),
            "jlens_require_remote": True,
            "jlens_telemetry_path": str(telemetry_path),
        }
    )
    for forbidden in ("hf_device", "local_path"):
        if args.get(forbidden) in {"cpu", "cuda", "auto"}:
            args.pop(forbidden, None)
    return args


def build_run_command(
    matrix: dict[str, Any],
    condition: dict[str, Any],
    *,
    split: str,
    task_ids: list[str],
    user_llm: str,
    user_llm_args: dict[str, Any],
    save_to: str,
    telemetry_path: Path,
    num_trials: int,
    max_concurrency: int,
    allow_missing_endpoint: bool = False,
) -> list[str]:
    agent_args = _condition_args(
        matrix,
        condition,
        telemetry_path=telemetry_path,
        allow_missing_endpoint=allow_missing_endpoint,
    )
    return [
        sys.executable,
        "-m",
        "tau2.cli",
        "run",
        "--domain",
        "airline",
        "--task-set-name",
        "airline",
        "--task-split-name",
        "base",
        "--task-ids",
        *task_ids,
        "--num-trials",
        str(num_trials),
        "--agent",
        "jlens_hf_agent",
        "--agent-llm",
        str(matrix["model"]["model_id"]),
        "--agent-llm-args",
        json.dumps(agent_args, ensure_ascii=False, separators=(",", ":")),
        "--user",
        "user_simulator",
        "--user-llm",
        user_llm,
        "--user-llm-args",
        json.dumps(user_llm_args, ensure_ascii=False, separators=(",", ":")),
        "--max-steps",
        "200",
        "--max-concurrency",
        str(max_concurrency),
        "--seed",
        str(matrix.get("generation", {}).get("seed", 300)),
        "--save-to",
        save_to,
        "--auto-resume",
    ]


def build_review_command(results_path: Path, *, review_model: str) -> list[str]:
    """Use Tau2's official full reviewer so agent/user errors stay separate."""
    return [
        sys.executable,
        "-m",
        "tau2.cli",
        "review",
        str(results_path),
        "--mode",
        "full",
        "--show-details",
        "--review-model",
        review_model,
    ]


def reviewed_results_complete(results_path: Path) -> bool:
    """Return true only when every current simulation has an embedded full review."""
    reviewed_path = results_path.with_name("results_reviewed.json")
    if not results_path.is_file() or not reviewed_path.is_file():
        return False
    raw = json.loads(results_path.read_text(encoding="utf-8"))
    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
    raw_keys = {
        (str(item.get("task_id", "")), int(item.get("trial", 0) or 0))
        for item in raw.get("simulations") or []
    }
    reviewed_items = reviewed.get("simulations") or []
    reviewed_keys = {
        (str(item.get("task_id", "")), int(item.get("trial", 0) or 0))
        for item in reviewed_items
    }
    return (
        bool(raw_keys)
        and raw_keys == reviewed_keys
        and all(isinstance(item.get("review"), dict) for item in reviewed_items)
    )


def simulation_results_complete(
    results_path: Path,
    *,
    task_ids: list[str],
    num_trials: int,
) -> bool:
    """Require exactly one result for every requested task/trial pair."""
    if not results_path.is_file():
        return False
    value = json.loads(results_path.read_text(encoding="utf-8"))
    keys = [
        (str(item.get("task_id", "")), int(item.get("trial", 0) or 0))
        for item in value.get("simulations") or []
    ]
    expected = {
        (str(task_id), trial)
        for task_id in task_ids
        for trial in range(int(num_trials))
    }
    return len(keys) == len(set(keys)) and set(keys) == expected


def select_conditions(
    conditions: list[dict[str, Any]],
    *,
    condition_name: str | None,
    method: str | None,
) -> list[dict[str, Any]]:
    """Select baseline, one named condition, one complete method family, or all."""
    if condition_name is not None and method is not None:
        raise ValueError("choose either --condition or --method, not both")
    if method is not None:
        selected = [item for item in conditions if item.get("method") == method]
    elif condition_name in {None, "baseline"}:
        selected = [item for item in conditions if item.get("name") == "baseline"]
    elif condition_name == "all":
        selected = conditions
    else:
        selected = [item for item in conditions if item.get("name") == condition_name]
    if not selected:
        selector = (
            f"method {method!r}"
            if method is not None
            else f"condition {condition_name!r}"
        )
        raise ValueError(f"unknown {selector}; use --condition list")
    return selected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--condition")
    parser.add_argument(
        "--method",
        choices=[
            "caa",
            "cast",
            "mera",
            "sadi",
            "iti",
            "austeer",
            "loreft",
            "jservo",
        ],
        help="run every target/control condition for one steering method",
    )
    parser.add_argument(
        "--split", choices=["train", "validation", "evaluation"], default="evaluation"
    )
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--user-llm", default="gpt-4.1-2025-04-14")
    parser.add_argument("--user-llm-args", default='{"temperature":0.0,"seed":300}')
    parser.add_argument("--save-prefix", default="failure-steering")
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--review-model", default="gpt-4.1-2025-04-14")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    matrix = _load_matrix(args.matrix)
    conditions = matrix["conditions"]
    if args.condition == "list":
        print("\n".join(item["name"] for item in conditions))
        return 0
    selected = select_conditions(
        conditions,
        condition_name=args.condition,
        method=args.method,
    )
    task_ids = [str(item) for item in matrix["splits"][f"{args.split}_task_ids"]]
    user_llm_args = json.loads(args.user_llm_args)
    simulations_root = Path("data/simulations")
    for condition in selected:
        relative_save = f"{args.save_prefix}/{args.split}/{condition['name']}"
        telemetry = (
            Path("data/jlens-telemetry")
            / args.save_prefix
            / args.split
            / f"{condition['name']}-{{task_id}}-{{simulation_id}}.jsonl"
        )
        command = build_run_command(
            matrix,
            condition,
            split=args.split,
            task_ids=task_ids,
            user_llm=args.user_llm,
            user_llm_args=user_llm_args,
            save_to=relative_save,
            telemetry_path=telemetry,
            num_trials=args.num_trials,
            max_concurrency=args.max_concurrency,
            allow_missing_endpoint=args.dry_run,
        )
        print(
            json.dumps(
                {"condition": condition["name"], "command": command},
                ensure_ascii=False,
            )
        )
        if args.dry_run:
            continue
        subprocess.run(command, check=True)
        results = simulations_root / relative_save / "results.json"
        if not simulation_results_complete(
            results,
            task_ids=task_ids,
            num_trials=args.num_trials,
        ):
            raise RuntimeError(
                f"incomplete simulation coverage at {results}; refusing review/analysis"
            )
        if args.review:
            reviewed = results.with_name("results_reviewed.json")
            if reviewed_results_complete(results):
                print(json.dumps({"review": "already_complete", "path": str(reviewed)}))
                continue
            if reviewed.exists():
                raise RuntimeError(
                    f"stale or partial reviewed results at {reviewed}; preserve or move that "
                    "file before explicitly regenerating the full review"
                )
            review_command = build_review_command(
                results,
                review_model=args.review_model,
            )
            print(json.dumps({"review_command": review_command}, ensure_ascii=False))
            subprocess.run(review_command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
