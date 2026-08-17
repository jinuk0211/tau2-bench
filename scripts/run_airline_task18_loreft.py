"""Run the TauBench Task-18 LoReFT pilot one condition at a time."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Condition:
    name: str
    agent_args: dict[str, Any]


def _load_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = path.expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "taubench-failure-loreft-v1":
        raise ValueError("unsupported Task-18 LoReFT config")
    output = Path(config["output_dir"]).expanduser()
    return config, (output if output.is_absolute() else path.parent / output).resolve()


def _base_agent_args(config: dict[str, Any], telemetry: Path) -> dict[str, Any]:
    generation = config.get("generation", {})
    args = {
        "seed": int(generation.get("seed", 626729)),
        "max_new_tokens": int(generation.get("max_new_tokens", 4096)),
        "do_sample": bool(generation.get("do_sample", True)),
        "use_cache": bool(generation.get("use_cache", True)),
        "jlens_telemetry_path": str(telemetry),
        "jlens_selected_layers": config["training"]["layers"],
        "hf_revision": config["model"]["model_revision"],
        "hf_max_input_tokens": 32768,
        "hf_chat_template_kwargs": {"enable_thinking": False},
        "hf_device": "auto",
        "hf_dtype": config["model"].get("dtype", "bfloat16"),
        "hf_sdpa_backend": "auto",
        "hf_trust_remote_code": config["model"].get("trust_remote_code", False),
    }
    for key in ("temperature", "top_p", "top_k"):
        if key in generation:
            args[key] = generation[key]
    return args


def build_conditions(config: dict[str, Any], output: Path) -> list[Condition]:
    telemetry_root = output / "telemetry"
    baseline_args = _base_agent_args(config, telemetry_root / "baseline.jsonl")
    baseline_args["jlens_mode"] = "observe"
    conditions = [Condition("baseline", baseline_args)]
    layers = [int(value) for value in config["training"]["layers"]]
    turn = int(config["causal_turn_index"])
    boundary = str(config["causal_boundary"])

    def intervention(path: Path, *, scale: float = 1.0, apply_decode: bool = False):
        return {
            "kind": "steer",
            "method": "loreft",
            "layer": min(layers),
            "strength": float(scale),
            "vector_path": str(path),
            "turn_indices": [turn],
            "boundaries": [boundary],
            "apply_prefill_decision": True,
            "apply_decode": bool(apply_decode),
        }

    variants: list[tuple[str, dict[str, Any]]] = []
    for rank_value in config["training"]["ranks"]:
        rank = int(rank_value)
        path = output / "artifacts" / f"loreft-rank-{rank}.pt"
        if path.is_file():
            variants.append((f"loreft-r{rank}", intervention(path)))
    primary_rank = int(config["training"]["primary_rank"])
    primary_path = output / "artifacts" / f"loreft-rank-{primary_rank}.pt"
    if primary_path.is_file():
        variants.extend(
            [
                (
                    f"loreft-identity-r{primary_rank}",
                    intervention(primary_path, scale=0.0),
                ),
                (
                    f"loreft-decode-extension-r{primary_rank}",
                    intervention(primary_path, apply_decode=True),
                ),
            ]
        )
    for seed_value in config["training"]["random_seeds"]:
        seed = int(seed_value)
        path = output / "artifacts" / f"loreft-random-seed-{seed}.pt"
        if path.is_file():
            variants.append(
                (f"loreft-random-r{primary_rank}-s{seed}", intervention(path))
            )
    for name, value in variants:
        args = _base_agent_args(config, telemetry_root / f"{name}.jsonl")
        args.update({"jlens_mode": "intervene", "jlens_intervention": value})
        conditions.append(Condition(name, args))
    return conditions


def _command(
    config: dict[str, Any],
    condition: Condition,
    *,
    user_llm: str,
    user_llm_args: dict[str, Any],
    save_prefix: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tau2.cli",
        "run",
        "--domain",
        "airline",
        "--task-set-name",
        "airline",
        "--task-ids",
        "18",
        "--num-trials",
        "1",
        "--agent",
        "jlens_hf_agent",
        "--agent-llm",
        config["model"]["model_id"],
        "--agent-llm-args",
        json.dumps(condition.agent_args, ensure_ascii=False, separators=(",", ":")),
        "--user",
        "user_simulator",
        "--user-llm",
        user_llm,
        "--user-llm-args",
        json.dumps(user_llm_args, ensure_ascii=False, separators=(",", ":")),
        "--max-steps",
        "200",
        "--max-concurrency",
        "1",
        "--seed",
        "300",
        "--save-to",
        f"{save_prefix}/{condition.name}",
        "--auto-resume",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--user-llm", default="gpt-4.1-2025-04-14")
    parser.add_argument("--user-llm-args", default='{"temperature":0.0,"seed":300}')
    parser.add_argument("--save-prefix", default="task18-loreft")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config, output = _load_config(args.config)
    conditions = build_conditions(config, output)
    if args.condition == "list":
        print("\n".join(condition.name for condition in conditions))
        return 0
    selected = (
        conditions
        if args.condition == "all"
        else [condition for condition in conditions if condition.name == args.condition]
    )
    if not selected:
        raise ValueError(f"unknown condition {args.condition!r}; train artifacts, then use list")
    user_llm_args = json.loads(args.user_llm_args)
    for condition in selected:
        command = _command(
            config,
            condition,
            user_llm=args.user_llm,
            user_llm_args=user_llm_args,
            save_prefix=args.save_prefix,
        )
        print(json.dumps({"condition": condition.name, "command": command}, ensure_ascii=False))
        if not args.dry_run:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
