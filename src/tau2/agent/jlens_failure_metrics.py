"""Benchmark-wide outcome metrics for failure-mode steering experiments."""

from __future__ import annotations

from collections import Counter
from typing import Any

from tau2.agent.jlens_task18_metrics import (
    TOOL_METRIC_CSV_KEYS,
    add_tool_metric_baseline_deltas,
    trajectory_tool_metrics,
)

FAILURE_METRIC_CSV_KEYS = [
    "task_reward",
    "task_success",
    "agent_review_error_count",
    "user_review_error_count",
    "abstention_count",
    "candidate_validation_failure_count",
    *TOOL_METRIC_CSV_KEYS,
]


def _review_errors(simulation: dict[str, Any]) -> list[dict[str, Any]]:
    review = simulation.get("review") or simulation.get("llm_review") or {}
    if not isinstance(review, dict):
        return []
    errors = review.get("errors") or []
    return [item for item in errors if isinstance(item, dict)]


def simulation_failure_metrics(simulation: dict[str, Any]) -> dict[str, Any]:
    """Combine official reward, review labels, and structural tool metrics."""
    messages = simulation.get("messages") or simulation.get("trajectory") or []
    if isinstance(messages, dict):
        messages = messages.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    termination = simulation.get("termination_reason")
    if termination is None and isinstance(simulation.get("info"), dict):
        termination = simulation["info"].get("termination_reason")
    reward = simulation.get("reward")
    if reward is None and isinstance(simulation.get("reward_info"), dict):
        reward = simulation["reward_info"].get("reward")
    if isinstance(reward, dict):
        reward = reward.get("reward")
        if reward is None:
            reward = simulation["reward"].get("overall")
    try:
        numeric_reward = float(reward) if reward is not None else None
    except (TypeError, ValueError):
        numeric_reward = None

    errors = _review_errors(simulation)
    by_source = Counter(str(item.get("source", "unknown")) for item in errors)
    by_tag: Counter[str] = Counter()
    for error in errors:
        tags = error.get("error_tags") or error.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        by_tag.update(str(tag) for tag in tags)
    abstentions = 0
    validator_failures = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        raw_data = message.get("raw_data") or {}
        if not isinstance(raw_data, dict):
            continue
        abstentions += int(bool(raw_data.get("jlens_abstained")))
        validation = raw_data.get("jlens_candidate_validation") or {}
        validator_failures += int(
            isinstance(validation, dict) and validation.get("valid") is False
        )
    return {
        "task_reward": numeric_reward,
        "task_success": numeric_reward == 1.0,
        "agent_review_error_count": by_source["agent"],
        "user_review_error_count": by_source["user"],
        "review_error_counts_by_tag": dict(sorted(by_tag.items())),
        "abstention_count": abstentions,
        "candidate_validation_failure_count": validator_failures,
        **trajectory_tool_metrics(messages, termination),
    }


__all__ = [
    "FAILURE_METRIC_CSV_KEYS",
    "TOOL_METRIC_CSV_KEYS",
    "add_tool_metric_baseline_deltas",
    "simulation_failure_metrics",
    "trajectory_tool_metrics",
]
