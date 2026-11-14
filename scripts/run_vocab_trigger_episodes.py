"""Full Tau2 airline episodes with vocabulary-triggered steering.

Conditions (all gated by the same layer-16 J-lens vocabulary trigger, so they
differ only in what happens on triggered turns):

* ``detect``  trigger scored and logged, no edit (the unsteered baseline);
* ``jlens``   J-lens direction, h += alpha * ||h|| * d at layer 20;
* ``random``  random unit direction at the same dose (control);
* ``caa``     nrimsky/CAA vector through its own add_vector_from_position code.

The user simulator is the same Qwen3.5-4B weights served in-process
(``--user-llm localhf/...``, greedy, thinking off), so episodes are
deterministic and conditions are paired per task.  Pass ``--conditions`` and
``--task-ids`` to split the work across GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
HELD_OUT = [
    "33", "34", "43", "46", "47", "49",  # validation
    "2", "6", "8", "13", "16", "18", "19", "22", "24", "25", "26", "29",
    "30", "31", "32", "35", "37", "44", "45", "48",  # evaluation
]


def intervention(name: str, args: argparse.Namespace) -> dict:
    gate = {"trigger_spec_path": str(args.trigger_spec)}
    if name == "detect":
        return {"kind": "steer", "method": "jlens_vocab", "layer": 20, "strength": 0.0, **gate}
    if name == "jlens":
        return {"kind": "steer", "method": "jlens_vocab", "layer": 20, "strength": args.alpha, **gate}
    if name == "random":
        return {"kind": "steer", "method": "jlens_vocab", "layer": 20, "strength": args.alpha,
                "jlens_direction": "random", **gate}
    if name == "caa":
        dose = json.loads(Path(args.caa_artifact).with_suffix(".dose.json").read_text())
        multiplier = args.caa_multiplier or dose["layers"][str(args.caa_layer)]["matched_multiplier"]
        return {"kind": "steer", "method": "caa", "implementation": "official",
                "layer": args.caa_layer, "strength": multiplier,
                "vector_path": str(args.caa_artifact),
                "official_options": {"vector_layer": args.caa_layer}, **gate}
    raise ValueError(f"unknown condition {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B",
                        help="Hub id (with --revision) or a local model directory")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--trigger-spec", type=Path, required=True)
    parser.add_argument("--caa-artifact", type=Path)
    parser.add_argument("--caa-layer", type=int, default=13)
    parser.add_argument("--caa-multiplier", type=float, help="default: the matched multiplier")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--conditions", nargs="+", default=["detect", "jlens", "caa"])
    parser.add_argument("--task-ids", nargs="+", default=HELD_OUT)
    parser.add_argument("--save-prefix", default="vocab-trigger")
    parser.add_argument("--telemetry-root", type=Path, default=Path("data/jlens-telemetry"))
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for name in args.conditions:
        agent_args = {
            "hf_dtype": args.dtype,
            "hf_chat_template_kwargs": {"enable_thinking": False},
            "hf_max_input_tokens": 32768,
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "max_time": 300,
            "seed": 300,
            "use_cache": True,
            "jlens_path": str(args.lens),
            "jlens_mode": "intervene",
            "jlens_measure": False,
            "jlens_telemetry_path": str(
                args.telemetry_root / args.save_prefix / name / "{task_id}-{simulation_id}.jsonl"
            ),
            "jlens_intervention": intervention(name, args),
        }
        if args.revision and not Path(args.model).is_dir():
            agent_args["hf_revision"] = args.revision
        command = [
            sys.executable, "-m", "tau2.cli", "run",
            "--domain", "airline", "--task-set-name", "airline", "--task-split-name", "base",
            "--task-ids", *args.task_ids,
            "--num-trials", "1",
            "--agent", "jlens_hf_agent",
            "--agent-llm", args.model,
            "--agent-llm-args", json.dumps(agent_args, separators=(",", ":")),
            "--user", "user_simulator",
            "--user-llm", "localhf/qwen3.5-4b",
            "--user-llm-args", "{}",
            "--max-steps", "200",
            "--timeout", "3600",
            "--max-retries", "0",
            "--max-concurrency", "1",
            "--seed", "300",
            "--save-to", f"{args.save_prefix}/{name}",
            "--auto-resume",
        ]
        print(f"== {name}: {len(args.task_ids)} tasks", flush=True)
        if args.dry_run:
            print(" ".join(command))
            continue
        result = subprocess.run(command, env=os.environ.copy())
        if result.returncode != 0:
            print(f"condition {name} exited with {result.returncode}", flush=True)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
