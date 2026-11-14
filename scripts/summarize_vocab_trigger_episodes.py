"""Summarise vocab-trigger Tau2 episodes: pass rates, paired task outcomes, trigger use.

Reads ``<root>/simulations/<prefix>/<condition>/task_<id>/results.json`` and the
matching agent telemetry ``<root>/jlens-telemetry/<prefix>/<condition>/*.jsonl``
(any number of roots, e.g. one per GPU host).  Conditions are compared on the
tasks every condition finished, so the comparison stays paired.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

VALIDATION = {"33", "34", "43", "46", "47", "49"}


def load(roots: list[Path], prefix: str) -> dict[str, dict[str, dict]]:
    runs: dict[str, dict[str, dict]] = defaultdict(dict)
    for root in roots:
        for path in sorted((root / "simulations" / prefix).glob("*/task_*/results.json")):
            condition = path.parent.parent.name
            data = json.loads(path.read_text(encoding="utf-8"))
            for sim in data.get("simulations") or []:
                reward = (sim.get("reward_info") or {}).get("reward")
                if reward is None:
                    continue
                info = sim.get("reward_info") or {}
                runs[condition][str(sim["task_id"])] = {
                    "reward": float(reward),
                    "db_match": (info.get("db_check") or {}).get("db_match"),
                    "termination": sim.get("termination_reason"),
                    "duration": sim.get("duration") or 0.0,
                    "messages": len(sim.get("messages") or []),
                    "writes": [call["name"] for message in sim.get("messages") or []
                               if message.get("role") == "assistant"
                               for call in message.get("tool_calls") or []
                               if not call["name"].startswith(("get_", "search_", "list_", "calculate", "think"))],
                    "simulation_id": sim.get("id"),
                    "host": root.name,
                }
        for path in sorted((root / "jlens-telemetry" / prefix).glob("*/*.jsonl")):
            condition = path.parent.name
            task_id = path.name.split("-", 1)[0]
            turns = [json.loads(line) for line in path.open(encoding="utf-8")]
            record = runs[condition].get(task_id)
            if record is None or not path.name.endswith(f"{record['simulation_id']}.jsonl"):
                continue
            gates = [turn["intervention"].get("vocab_trigger") or {} for turn in turns]
            record["agent_turns"] = len(turns)
            record["triggered_turns"] = sum(bool(g.get("triggered")) for g in gates)
            record["steered_turns"] = sum(bool(turn["intervention"].get("active")) for turn in turns)
            record["max_score"] = max((g.get("score", float("-inf")) for g in gates), default=None)
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+", help="tau2 data dirs (each holding simulations/)")
    parser.add_argument("--prefix", default="vt-heldout")
    parser.add_argument("--baseline", default="detect")
    parser.add_argument("--json", type=Path, help="also write the per-task table here")
    args = parser.parse_args()
    runs = load(args.roots, args.prefix)
    conditions = sorted(runs, key=lambda c: (c != args.baseline, c))
    common = sorted(set.intersection(*(set(runs[c]) for c in conditions)), key=int) if conditions else []
    print(f"tasks finished by every condition: {len(common)} {common}")
    for condition in conditions:
        done = runs[condition]
        print(f"  {condition}: {len(done)} tasks finished")

    print(f"\n{'condition':<10}{'pass':>8}{'val':>8}{'eval':>8}{'db_ok':>8}{'turns':>8}"
          f"{'fired':>8}{'steered':>9}{'min/ep':>8}")
    for condition in conditions:
        rows = [runs[condition][t] for t in common]
        if not rows:
            continue
        val = [r["reward"] for t, r in zip(common, rows) if t in VALIDATION]
        ev = [r["reward"] for t, r in zip(common, rows) if t not in VALIDATION]
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
        print(f"{condition:<10}{mean([r['reward'] for r in rows]):>8.3f}{mean(val):>8.3f}{mean(ev):>8.3f}"
              f"{mean([float(bool(r['db_match'])) for r in rows]):>8.3f}"
              f"{mean([r.get('agent_turns', 0) for r in rows]):>8.1f}"
              f"{sum(r.get('triggered_turns', 0) for r in rows):>8d}"
              f"{sum(r.get('steered_turns', 0) for r in rows):>9d}"
              f"{mean([r['duration'] / 60 for r in rows]):>8.1f}")

    print(f"\npaired against {args.baseline} (tasks finished by both):")
    base = runs.get(args.baseline, {})
    for condition in conditions:
        if condition == args.baseline:
            continue
        shared = sorted(set(base) & set(runs[condition]), key=int)
        fixed = [t for t in shared if runs[condition][t]["reward"] > base[t]["reward"]]
        broke = [t for t in shared if runs[condition][t]["reward"] < base[t]["reward"]]
        same_traj = sum(runs[condition][t]["messages"] == base[t]["messages"]
                        and runs[condition][t]["writes"] == base[t]["writes"] for t in shared)
        print(f"  {condition:<8} n={len(shared):<3} fixed {len(fixed)} {fixed}  broke {len(broke)} {broke}  "
              f"(identical-looking trajectories: {same_traj})")

    print("\nper task (reward / triggered turns / writes):")
    for task in sorted({t for c in conditions for t in runs[c]}, key=int):
        cells = []
        for condition in conditions:
            r = runs[condition].get(task)
            cells.append("-" if r is None else
                         f"{r['reward']:.0f}/{r.get('triggered_turns', '?')}/{','.join(Counter(r['writes']).keys()) or 'none'}")
        print(f"  {task:>3} {'val' if task in VALIDATION else 'eval'}  " + " | ".join(
            f"{c}: {cell}" for c, cell in zip(conditions, cells)))
    if args.json:
        args.json.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
