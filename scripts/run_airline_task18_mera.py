"""Run the TauBench Task-18 MERA pilot one condition at a time."""

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
    if config.get("schema_version") != "taubench-failure-mera-v1":
        raise ValueError("unsupported Task-18 MERA config")
    output = Path(config["output_dir"]).expanduser()
    if not output.is_absolute():
        output = (path.parent / output).resolve()
    return config, output


def _artifact_path(output: Path, layer: int) -> Path:
    return output / "artifacts" / f"mera-layer-{layer}.pt"


def _base_agent_args(config: dict[str, Any], telemetry: Path) -> dict[str, Any]:
    generation = config.get("generation", {})
    args = {
        "seed": int(generation.get("seed", 626729)),
        "max_new_tokens": int(generation.get("max_new_tokens", 4096)),
        "do_sample": bool(generation.get("do_sample", True)),
        "use_cache": bool(generation.get("use_cache", True)),
        "jlens_telemetry_path": str(telemetry),
        "jlens_selected_layers": config["extraction"]["layers"],
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


def _load_probe(path: Path) -> tuple[Any, float]:
    import torch

    artifact = torch.load(path, map_location="cpu", weights_only=True)
    return artifact["probe_vector"].detach().float().cpu(), float(
        artifact["selected_alpha"]
    )


def _random_probe(vector: Any, *, seed: int) -> list[float]:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random = torch.randn(vector.shape, generator=generator, dtype=torch.float32)
    random = random / random.norm() * vector.norm()
    return random.tolist()


def build_conditions(config: dict[str, Any], output: Path) -> list[Condition]:
    telemetry_root = output / "telemetry"
    baseline_args = _base_agent_args(config, telemetry_root / "baseline.jsonl")
    baseline_args["jlens_mode"] = "observe"
    conditions = [Condition("baseline", baseline_args)]
    turn = int(config["causal_turn_index"])
    boundary = str(config["causal_boundary"])
    for layer in config["sweep"]["layers"]:
        artifact_path = _artifact_path(output, int(layer))
        if not artifact_path.is_file():
            continue
        probe, selected_alpha = _load_probe(artifact_path)

        def intervention(
            *,
            method: str,
            alpha: float,
            prefill_mode: str = "all_tokens",
            vector: list[float] | None = None,
        ) -> dict[str, Any]:
            value: dict[str, Any] = {
                "kind": "steer",
                "method": "mera",
                "layer": int(layer),
                "strength": 1.0,
                "turn_indices": [turn],
                "boundaries": [boundary],
                "apply_decode": True,
                "mera_prefill_mode": prefill_mode,
            }
            if vector is None:
                value["artifact_layer"] = int(layer)
                value["vector_path"] = str(artifact_path)
                if alpha != selected_alpha:
                    value["mera_alpha_override"] = alpha
            else:
                value["vector"] = vector
                value["mera_alpha_override"] = alpha
            return value

        variants = [
            (
                "mera",
                intervention(method="mera", alpha=selected_alpha),
            ),
            (
                "mera-decision-only",
                intervention(
                    method="mera_decision_only",
                    alpha=selected_alpha,
                    prefill_mode="decision_only",
                ),
            ),
            (
                "mera-abstain",
                intervention(method="mera_abstain", alpha=1.0),
            ),
        ]
        for alpha in config["sweep"]["uncalibrated_alphas"]:
            alpha = float(alpha)
            if alpha != selected_alpha:
                variants.append(
                    (
                        f"mera-uncalibrated-{alpha:g}",
                        intervention(method="mera_uncalibrated", alpha=alpha),
                    )
                )
        for seed in config["sweep"]["random_seeds"]:
            variants.append(
                (
                    f"mera-random-s{seed}",
                    intervention(
                        method="mera_random_probe",
                        alpha=selected_alpha,
                        vector=_random_probe(probe, seed=int(seed)),
                    ),
                )
            )
        for label, value in variants:
            name = f"{label}-l{layer}"
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--user-llm", default="gpt-4.1-2025-04-14")
    parser.add_argument("--user-llm-args", default='{"temperature":0.0,"seed":300}')
    parser.add_argument("--save-prefix", default="task18-mera")
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
        raise ValueError(
            f"unknown condition {args.condition!r}; extract artifacts, then use list"
        )
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
