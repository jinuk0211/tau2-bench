"""Run the three-task τ²-Telecom No-User J-Lens plumbing gate.

The script runs the same deterministic local HF model in ``off`` and
``observe`` modes. It fails unless:

* exact prompt IDs match turn by turn,
* generated actions match exactly,
* observer telemetry was produced,
* strict trajectory replay reproduces the original reward.

This is a plumbing test, not a benchmark score or a calibration run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import TextRunConfig
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from tau2.orchestrator.modes import CommunicationMode
from tau2.registry import registry
from tau2.runner.build import build_text_orchestrator
from tau2.runner.helpers import load_tasks
from tau2.runner.simulation import run_simulation

PINNED_REVISIONS = {
    "jacobian-lens": "581d398",
    "tau2-bench": "1d244f5",
    "agents-last-exam": "d332c1a",
}
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


@dataclass
class ConditionResult:
    simulation: Any
    records: list[dict[str, Any]]


def _safe_task_dir_name(task_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", task_id).strip("._")
    digest = hashlib.sha256(task_id.encode()).hexdigest()[:10]
    return f"{slug[:80]}-{digest}"


def _git_revision(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True
        ).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--short")),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _runtime_info() -> dict[str, Any]:
    import torch

    cuda_available = torch.cuda.is_available()
    gpus = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": cuda_available,
        "pytorch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda_available else None,
        "gpus": gpus,
    }


def _action_signature(simulation: Any) -> list[Any]:
    signature: list[Any] = []
    for message in simulation.get_messages():
        if not isinstance(message, AssistantMessage):
            continue
        if message.is_tool_call():
            signature.append(
                [
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                    for call in message.tool_calls
                ]
            )
        else:
            signature.append({"content": message.content})
    return signature


def _run_condition(
    *,
    task: Any,
    mode: str,
    model: str,
    revision: str | None,
    output_dir: Path,
    seed: int,
    max_steps: int,
    max_new_tokens: int,
    max_input_tokens: int,
    lens_path: str | None,
    device: str,
    dtype: str,
    sdpa_backend: str,
) -> ConditionResult:
    task_dir = output_dir / mode / _safe_task_dir_name(str(task.id))
    task_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = task_dir / "jlens.jsonl"
    if telemetry_path.exists():
        telemetry_path.unlink()
    llm_args: dict[str, Any] = {
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "jlens_mode": mode,
        "jlens_telemetry_path": str(telemetry_path),
        "hf_revision": revision,
        "hf_max_input_tokens": max_input_tokens,
        "hf_device": device,
        "hf_dtype": dtype,
        "hf_sdpa_backend": sdpa_backend,
        "hf_chat_template_kwargs": {"enable_thinking": False},
    }
    if lens_path:
        llm_args["jlens_path"] = lens_path
    config = TextRunConfig(
        domain="telecom",
        task_set_name="telecom",
        task_split_name="small",
        agent="jlens_direct_solo",
        user="dummy_user",
        llm_agent=model,
        llm_args_agent=llm_args,
        llm_user="unused",
        llm_args_user={},
        num_trials=1,
        max_steps=max_steps,
        max_errors=1,
        seed=seed,
    )
    orchestrator = build_text_orchestrator(config, task, seed=seed)
    simulation = run_simulation(orchestrator)
    (task_dir / "simulation.json").write_text(
        simulation.model_dump_json(indent=2), encoding="utf-8"
    )
    return ConditionResult(
        simulation=simulation,
        records=_read_jsonl(telemetry_path),
    )


def _replay_reward(simulation: Any, task: Any) -> Any:
    return evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=True,
        domain="telecom",
        mode=CommunicationMode.HALF_DUPLEX,
        strict_replay=True,
    )


def _strict_environment_replay(simulation: Any, task: Any) -> Any:
    return EnvironmentEvaluator.calculate_reward(
        environment_constructor=registry.get_env_constructor("telecom"),
        task=task,
        full_trajectory=simulation.get_messages(),
        solo_mode=True,
        strict_replay=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--lens-path")
    parser.add_argument("--output-dir", default="data/jlens_smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-tasks", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--sdpa-backend",
        choices=["auto", "efficient", "flash", "math"],
        default="auto",
    )
    args = parser.parse_args()
    if args.num_tasks != 3:
        parser.error("the milestone gate is defined on exactly three tasks")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks("telecom", "small")[:3]
    root = Path(__file__).resolve().parents[2]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "plumbing_only",
        "claims": "No benchmark or calibration claim may be made from this run.",
        "model": args.model,
        "model_revision": args.revision,
        "seed": args.seed,
        "task_ids": [str(task.id) for task in tasks],
        "pinned_revisions": PINNED_REVISIONS,
        "source_state": {name: _git_revision(root / name) for name in PINNED_REVISIONS},
        "runtime": _runtime_info(),
        "run_config": {
            "device": args.device,
            "dtype": args.dtype,
            "sdpa_backend": args.sdpa_backend,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "max_steps": args.max_steps,
            "lens_path": args.lens_path,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    summaries: list[dict[str, Any]] = []
    all_passed = True
    resolved_model_revision = None
    resolved_tokenizer_revision = None
    for task in tasks:
        off = _run_condition(
            task=task,
            mode="off",
            model=args.model,
            revision=args.revision,
            output_dir=output_dir,
            seed=args.seed,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            lens_path=args.lens_path,
            device=args.device,
            dtype=args.dtype,
            sdpa_backend=args.sdpa_backend,
        )
        observe = _run_condition(
            task=task,
            mode="observe",
            model=args.model,
            revision=args.revision,
            output_dir=output_dir,
            seed=args.seed,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            lens_path=args.lens_path,
            device=args.device,
            dtype=args.dtype,
            sdpa_backend=args.sdpa_backend,
        )
        off_inputs = [record["input_ids"] for record in off.records]
        observe_inputs = [record["input_ids"] for record in observe.records]
        if off.records:
            resolved_model_revision = off.records[0]["model"]["resolved_revision"]
            resolved_tokenizer_revision = off.records[0]["model"][
                "resolved_tokenizer_revision"
            ]
        actions_equal = _action_signature(off.simulation) == _action_signature(
            observe.simulation
        )
        termination_equal = (
            off.simulation.termination_reason == observe.simulation.termination_reason
        )
        prompt_ids_equal = off_inputs == observe_inputs and bool(off_inputs)
        tool_parsing_verified = any(
            record["tool_calls"] for record in off.records
        ) and any(record["tool_calls"] for record in observe.records)
        observer_has_measurements = bool(observe.records) and all(
            record["measurement"].get("residuals") for record in observe.records
        )
        off_replay = _replay_reward(off.simulation, task)
        observe_replay = _replay_reward(observe.simulation, task)
        off_env_replay = _strict_environment_replay(off.simulation, task)
        observe_env_replay = _strict_environment_replay(observe.simulation, task)
        off_reward_equal = (
            off.simulation.reward_info is not None
            and off_replay.reward == off.simulation.reward_info.reward
        )
        observe_reward_equal = (
            observe.simulation.reward_info is not None
            and observe_replay.reward == observe.simulation.reward_info.reward
        )
        off_env_reward_equal = (
            off.simulation.reward_info is not None
            and off_env_replay.reward == off.simulation.reward_info.reward
        )
        observe_env_reward_equal = (
            observe.simulation.reward_info is not None
            and observe_env_replay.reward == observe.simulation.reward_info.reward
        )
        passed = all(
            [
                actions_equal,
                termination_equal,
                prompt_ids_equal,
                tool_parsing_verified,
                observer_has_measurements,
                off_reward_equal,
                observe_reward_equal,
                off_env_reward_equal,
                observe_env_reward_equal,
            ]
        )
        all_passed = all_passed and passed
        summaries.append(
            {
                "task_id": str(task.id),
                "passed": passed,
                "actions_equal": actions_equal,
                "termination_equal": termination_equal,
                "prompt_ids_equal": prompt_ids_equal,
                "tool_parsing_verified": tool_parsing_verified,
                "observer_has_measurements": observer_has_measurements,
                "off_reward_reproduced": off_reward_equal,
                "observe_reward_reproduced": observe_reward_equal,
                "off_strict_env_replay_reproduced": off_env_reward_equal,
                "observe_strict_env_replay_reproduced": observe_env_reward_equal,
                "off_strict_env_replay_reward": off_env_replay.reward,
                "observe_strict_env_replay_reward": observe_env_replay.reward,
                "off_termination": off.simulation.termination_reason.value,
                "observe_termination": observe.simulation.termination_reason.value,
                "off_reward": off.simulation.reward_info.reward
                if off.simulation.reward_info
                else None,
                "observe_reward": observe.simulation.reward_info.reward
                if observe.simulation.reward_info
                else None,
            }
        )
        print(json.dumps(summaries[-1], ensure_ascii=False))

    report = {
        "passed": all_passed,
        "gates": summaries,
    }
    manifest["resolved_model_revision"] = resolved_model_revision
    manifest["resolved_tokenizer_revision"] = resolved_tokenizer_revision
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "smoke_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
