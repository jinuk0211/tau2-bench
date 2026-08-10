"""Run one global task range across airline, retail, and telecom.

The catalog is always ordered as airline -> retail -> telecom. ``--start`` is
one-based and ``--count`` is the number of catalog entries to run. The runner
automatically uses the conversational J-Lens agent for airline/retail and the
direct solo J-Lens agent for telecom.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.tasks import Task
from tau2.run import get_tasks, run_domain

DOMAIN_ORDER = ("airline", "retail", "telecom")
MODEL_PROFILES = {
    "qwen3-8b": {
        "model_id": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
    },
    "qwen3.5-4b": {
        "model_id": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    },
    "qwen3.5-9b-base": {
        "model_id": "Qwen/Qwen3.5-9B-Base",
        "revision": "68c46c4b3498877f3ef123c856ecfde50c39f404",
        "diagnostic_only": True,
    },
    "qwen3.6-27b": {
        "model_id": "Qwen/Qwen3.6-27B",
        "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    },
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
        "--profile", choices=tuple(MODEL_PROFILES), default="qwen3-8b"
    )
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--user-model",
        default=os.getenv("TAU2_USER_MODEL", "openai/qwen3:8b"),
        help="LiteLLM user-simulator model for airline/retail",
    )
    parser.add_argument(
        "--user-api-base",
        default=os.getenv("TAU2_USER_API_BASE", "http://127.0.0.1:11434/v1"),
        help="OpenAI-compatible endpoint used by the airline/retail user simulator",
    )
    parser.add_argument("--trace-root", type=Path, default=Path("data/jlens_traces"))
    parser.add_argument("--save-prefix", default="jlens-range")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print counts and the selected range without loading a model",
    )
    args = parser.parse_args(argv)
    if args.num_trials < 1:
        parser.error("--num-trials must be at least 1")
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
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
    profile = MODEL_PROFILES[args.profile]
    trace_template = (
        args.trace_root.resolve()
        / run_label
        / domain
        / "{simulation_id}.jsonl"
    )
    agent_args = {
        "hf_revision": profile["revision"],
        "jlens_mode": "off",
        "jlens_telemetry_path": str(trace_template),
        "hf_dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
    }
    conversational = domain in {"airline", "retail"}
    return TextRunConfig(
        domain=domain,
        # The combined catalog is built from the complete domain task set.
        # Keep execution on that same unfiltered set instead of TextRunConfig's
        # default ``base`` split, otherwise persona-expanded telecom IDs that
        # appear in the catalog cannot be resolved by run_domain().
        task_split_name=None,
        task_ids=task_ids,
        agent="jlens_hf_agent" if conversational else "jlens_direct_solo",
        llm_agent=profile["model_id"],
        llm_args_agent=agent_args,
        user="user_simulator" if conversational else "dummy_user",
        llm_user=args.user_model,
        llm_args_user=(
            {"api_base": args.user_api_base.rstrip("/"), "temperature": 0.0}
            if conversational
            else {}
        ),
        num_trials=args.num_trials,
        max_concurrency=args.max_concurrency,
        save_to=f"{args.save_prefix}-{run_label}-{domain}",
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
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.list_only:
        return 0

    if MODEL_PROFILES[args.profile].get("diagnostic_only"):
        print(
            "[J-Lens range] WARNING: Qwen3.5-9B-Base is pre-trained-only; "
            "results are diagnostic and not an instruction/tool-use benchmark.",
            file=sys.stderr,
        )

    os.environ.setdefault("OPENAI_API_KEY", "not-needed")
    run_label = f"{selected[0].index:04d}-{selected[-1].index:04d}"
    selection_dir = args.trace_root.resolve() / run_label
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
            f"[J-Lens range] {domain}: running {len(task_ids)} selected task(s)",
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
    print(f"[J-Lens range] traces: {selection_dir}")
    print(
        "[J-Lens range] analyze with: "
        f"uv run tau2 jlens \"{selection_dir}\" --profile {args.profile} "
        "--output-dir data/jlens_analysis --layer-stride 1"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[J-Lens range] interrupted", file=sys.stderr)
        raise SystemExit(130) from None
