"""Full Tau2 airline episodes with vocabulary-triggered steering.

Conditions (all gated by the same layer-16 J-lens vocabulary trigger, so they
differ only in what happens on triggered turns):

* ``detect``  trigger scored and logged, no edit (the unsteered baseline);
* ``jlens``   J-lens direction, h += alpha * ||h|| * d at layer 20;
* ``random``  random unit direction at the same dose (control);
* ``caa``     nrimsky/CAA vector through its own add_vector_from_position code.

The user simulator is the same Qwen3.5-4B weights served in-process
(``--user-llm localhf/...``, greedy, thinking off), so episodes are
deterministic and conditions are paired per task.

Tasks run one at a time, every condition back to back, inside this process
(the model bundle is loaded once and shared).  Each (condition, task) is saved
to its own results file and skipped when it already exists, so a run can be
restarted and split across GPUs with ``--task-ids``.
"""

from __future__ import annotations

import argparse
import json
import shutil
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


def tau2_argv(name: str, task_id: str, args: argparse.Namespace) -> list[str]:
    agent_args = {
        "hf_dtype": args.dtype,
        "hf_chat_template_kwargs": {"enable_thinking": False},
        "hf_max_input_tokens": 32768,
        "hf_prefill_chunk": args.prefill_chunk,
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
    return [
        "tau2", "run",
        "--domain", "airline", "--task-set-name", "airline", "--task-split-name", "base",
        "--task-ids", task_id,
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
        "--save-to", f"{args.save_prefix}/{name}/task_{task_id}",
    ]


def finished(save_to: str) -> bool:
    path = Path("data/simulations") / save_to / "results.json"
    if not path.is_file():
        return False
    try:
        sims = json.loads(path.read_text(encoding="utf-8")).get("simulations") or []
    except json.JSONDecodeError:
        return False
    return any((sim.get("reward_info") or {}).get("reward") is not None for sim in sims)


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
    parser.add_argument("--prefill-chunk", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from tau2.cli import main as tau2_main

    for task_id in args.task_ids:
        for name in args.conditions:
            argv = tau2_argv(name, task_id, args)
            save_to = argv[argv.index("--save-to") + 1]
            if finished(save_to):
                print(f"== task {task_id} {name}: done, skipping", flush=True)
                continue
            print(f"== task {task_id} {name}", flush=True)
            if args.dry_run:
                print(" ".join(argv))
                continue
            partial = Path("data/simulations") / save_to
            if partial.exists():
                # An interrupted episode: start it over (tau2 would prompt to resume).
                shutil.rmtree(partial)
            sys.argv = argv
            try:
                tau2_main()
            except SystemExit as stop:
                if stop.code not in (None, 0):
                    print(f"task {task_id} {name} exited with {stop.code}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
