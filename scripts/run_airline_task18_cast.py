"""Run the TauBench Task-18 CAST pilot one reproducible condition at a time."""

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
    if config.get("schema_version") != "taubench-failure-cast-v1":
        raise ValueError("unsupported Task-18 CAST config")
    output = Path(config["output_dir"]).expanduser()
    if not output.is_absolute():
        output = (path.parent / output).resolve()
    return config, output


def _artifact_path(output: Path, layer: int) -> Path:
    return output / "artifacts" / f"cast-layer-{layer}.pt"


def _base_agent_args(config: dict[str, Any], telemetry: Path) -> dict[str, Any]:
    generation = config.get("generation", {})
    selected_layers = sorted(
        set(config["extraction"]["behavior_layers"] + config["extraction"]["condition_layers"])
    )
    args = {
        "seed": int(generation.get("seed", 626729)),
        "max_new_tokens": int(generation.get("max_new_tokens", 4096)),
        "do_sample": bool(generation.get("do_sample", True)),
        "use_cache": bool(generation.get("use_cache", True)),
        "jlens_telemetry_path": str(telemetry),
        "jlens_selected_layers": selected_layers,
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
    turn = int(config["causal_turn_index"])
    boundary = str(config["causal_boundary"])
    sweep = config["sweep"]
    for layer in sweep["layers"]:
        artifact = _artifact_path(output, int(layer))
        for alpha_value in sweep["alphas"]:
            alpha = float(alpha_value)
            if alpha == 0.0:
                continue

            def cast_intervention(
                *,
                strength: float,
                prefill_mode: str = "all_tokens",
                gate_override: bool | None = None,
                invert_comparator: bool = False,
            ) -> dict[str, Any]:
                value: dict[str, Any] = {
                    "kind": "steer",
                    "method": "cast",
                    "layer": int(layer),
                    "artifact_layer": int(layer),
                    "strength": strength,
                    "vector_path": str(artifact),
                    "turn_indices": [turn],
                    "boundaries": [boundary],
                    "apply_decode": True,
                    "cast_prefill_mode": prefill_mode,
                }
                if gate_override is not None:
                    value["cast_gate_override"] = gate_override
                if invert_comparator:
                    value["cast_invert_comparator"] = True
                return value

            variants = [
                ("cast-positive", cast_intervention(strength=alpha)),
                ("cast-negative", cast_intervention(strength=-alpha)),
                (
                    "cast-decision-only",
                    cast_intervention(strength=alpha, prefill_mode="decision_only"),
                ),
            ]
            if sweep.get("include_ungated_control", False):
                variants.append(
                    (
                        "cast-ungated",
                        cast_intervention(strength=alpha, gate_override=True),
                    )
                )
            if sweep.get("include_complement_gate_control", False):
                variants.append(
                    (
                        "cast-complement-gate",
                        cast_intervention(strength=alpha, invert_comparator=True),
                    )
                )
            for label, intervention in variants:
                name = f"{label}-l{layer}-a{alpha:g}"
                args = _base_agent_args(config, telemetry_root / f"{name}.jsonl")
                args.update(
                    {
                        "jlens_mode": "intervene",
                        "jlens_intervention": intervention,
                    }
                )
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--condition",
        default="baseline",
        help="exact condition name, 'list', or 'all' (one-at-a-time is recommended)",
    )
    parser.add_argument("--user-llm", default="gpt-4.1-2025-04-14")
    parser.add_argument("--user-llm-args", default='{"temperature":0.0,"seed":300}')
    parser.add_argument("--save-prefix", default="task18-cast")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
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
        raise ValueError(f"unknown condition {args.condition!r}; use --condition list")
    user_llm_args = json.loads(args.user_llm_args)
    for condition in selected:
        intervention = condition.agent_args.get("jlens_intervention")
        if intervention is not None and not Path(intervention["vector_path"]).is_file():
            raise FileNotFoundError(
                f"extract the CAST artifact before running {condition.name}: "
                f"{intervention['vector_path']}"
            )
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
