"""Run a global tau2 task range through a hosted vLLM agent.

The catalog is ordered as airline -> retail -> telecom. ``--start`` is
one-based and ``--count`` selects consecutive catalog entries.  Unlike the
offline J-Lens replay step, this command only generates normal tau2 results and
verbose LLM logs.  Those logs are later teacher-forced through Transformers so
trajectory generation stays comparable with the GPT-OSS benchmark runs.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.tasks import Task
from tau2.jlens.profiles import PROFILES
from tau2.run import get_tasks, run_domain
from tau2.utils.llm_utils import set_llm_log_mode

DOMAIN_ORDER = ("airline", "retail", "telecom")
MODEL_PROFILES = {
    name: {
        "model_id": profile.model_id,
        "revision": profile.model_revision,
        "served_model": profile.model_id.rsplit("/", 1)[-1],
        "diagnostic_only": name.endswith("-base"),
    }
    for name, profile in PROFILES.items()
}


@dataclass(frozen=True)
class CatalogEntry:
    """One task's stable position in the combined three-domain catalog."""

    index: int
    domain: str
    task: Task


def build_catalog(
    task_sets: dict[str, Sequence[Task]],
) -> list[CatalogEntry]:
    """Flatten tasks in the declared domain order using one-based indexes."""
    catalog: list[CatalogEntry] = []
    for domain in DOMAIN_ORDER:
        for task in task_sets[domain]:
            catalog.append(
                CatalogEntry(index=len(catalog) + 1, domain=domain, task=task)
            )
    return catalog


def select_range(
    catalog: Sequence[CatalogEntry], *, start: int, count: int
) -> list[CatalogEntry]:
    """Return a strict one-based slice and reject partially invalid requests."""
    if start < 1:
        raise ValueError("--start must be at least 1")
    if count < 1:
        raise ValueError("--count must be at least 1")
    if start > len(catalog):
        raise ValueError(
            f"--start={start} exceeds the combined catalog size {len(catalog)}"
        )
    end = start + count - 1
    if end > len(catalog):
        raise ValueError(
            f"requested range {start}..{end} exceeds the combined catalog size "
            f"{len(catalog)}"
        )
    return list(catalog[start - 1 : end])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="One-based position in the combined catalog (default: 1)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of consecutive combined-catalog tasks (default: 1)",
    )
    parser.add_argument(
        "--profile", choices=tuple(MODEL_PROFILES), default="qwen3.5-4b"
    )
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--max-tokens",
        "--max-new-tokens",
        dest="max_tokens",
        type=int,
        default=4096,
        help="Maximum generated agent tokens (default: 4096)",
    )
    parser.add_argument(
        "--agent-api-base",
        default=os.getenv("HOSTED_VLLM_API_BASE", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible vLLM endpoint",
    )
    parser.add_argument(
        "--user-model",
        default=os.getenv("TAU2_USER_MODEL", "gpt-5.2-2025-12-11"),
        help="LiteLLM model for the user simulator",
    )
    parser.add_argument(
        "--user-api-base",
        default=os.getenv("TAU2_USER_API_BASE", "https://api.openai.com/v1"),
        help="OpenAI-compatible endpoint used by the user simulator",
    )
    parser.add_argument(
        "--trajectory-root",
        "--trace-root",
        dest="trajectory_root",
        type=Path,
        default=Path("data/jlens_trajectories"),
        help="Root for standard tau2 results and verbose LLM logs",
    )
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print counts and the selected range without contacting a model",
    )
    args = parser.parse_args(argv)
    if args.num_trials < 1:
        parser.error("--num-trials must be at least 1")
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be at least 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    return args


def _selection_manifest(
    catalog: Sequence[CatalogEntry], selected: Sequence[CatalogEntry]
) -> dict[str, Any]:
    counts = {
        domain: sum(entry.domain == domain for entry in catalog)
        for domain in DOMAIN_ORDER
    }
    return {
        "domain_order": list(DOMAIN_ORDER),
        "total_tasks": len(catalog),
        "domain_counts": counts,
        "selection": {
            "start": selected[0].index,
            "count": len(selected),
            "end": selected[-1].index,
        },
        "tasks": [
            {
                "index": entry.index,
                "domain": entry.domain,
                "task_id": str(entry.task.id),
            }
            for entry in selected
        ],
    }


def _build_config(
    args: argparse.Namespace,
    *,
    domain: str,
    task_ids: list[str],
    run_label: str,
) -> TextRunConfig:
    """Build the GPT-OSS-compatible online trajectory configuration."""
    profile = MODEL_PROFILES[args.profile]
    run_dir = args.trajectory_root.resolve() / run_label / domain
    agent_args = {
        "api_base": args.agent_api_base.rstrip("/"),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    user_args = {
        "api_base": args.user_api_base.rstrip("/"),
        "reasoning_effort": "low",
    }
    return TextRunConfig(
        domain=domain,
        # The combined catalog uses the full domain task set, so execution must
        # not silently apply the default ``base`` split afterward.
        task_split_name=None,
        task_ids=task_ids,
        agent="llm_agent",
        llm_agent=f"hosted_vllm/{profile['served_model']}",
        llm_args_agent=agent_args,
        user="user_simulator",
        llm_user=args.user_model,
        llm_args_user=user_args,
        num_trials=args.num_trials,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        save_to=str(run_dir),
        verbose_logs=True,
        auto_resume=args.auto_resume,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    task_sets = {domain: get_tasks(domain) for domain in DOMAIN_ORDER}
    catalog = build_catalog(task_sets)
    try:
        selected = select_range(catalog, start=args.start, count=args.count)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    manifest = _selection_manifest(catalog, selected)
    manifest["generation"] = {
        "backend": "vllm",
        "profile": args.profile,
        "model": MODEL_PROFILES[args.profile]["model_id"],
        "model_revision": MODEL_PROFILES[args.profile]["revision"],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "user_model": args.user_model,
        "user_reasoning_effort": "low",
        "seed": args.seed,
        "max_concurrency": args.max_concurrency,
        "verbose_logs": True,
        "llm_log_mode": "all",
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.list_only:
        return 0

    if MODEL_PROFILES[args.profile]["diagnostic_only"]:
        raise SystemExit(
            f"{args.profile} is a base model and cannot generate a comparable "
            "tool-use trajectory through the standard tau2 agent"
        )

    os.environ["HOSTED_VLLM_API_BASE"] = args.agent_api_base.rstrip("/")
    os.environ.setdefault("HOSTED_VLLM_API_KEY", "dummy")
    set_llm_log_mode("all")

    run_label = f"{selected[0].index:04d}-{selected[-1].index:04d}"
    selection_dir = args.trajectory_root.resolve() / run_label
    selection_dir.mkdir(parents=True, exist_ok=True)
    (selection_dir / "selection.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    grouped: dict[str, list[str]] = defaultdict(list)
    for entry in selected:
        grouped[entry.domain].append(str(entry.task.id))
    for domain in DOMAIN_ORDER:
        task_ids = grouped.get(domain)
        if not task_ids:
            continue
        print(
            f"[tau2 range] {domain}: running {len(task_ids)} selected task(s) "
            "through vLLM",
            flush=True,
        )
        run_domain(
            _build_config(
                args,
                domain=domain,
                task_ids=task_ids,
                run_label=run_label,
            )
        )

    print(f"[tau2 range] trajectories: {selection_dir}")
    print(
        "[tau2 range] vLLM can now be stopped; replay the verbose logs with "
        "scripts/analyze_tau2.py from jacobian-lens"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
