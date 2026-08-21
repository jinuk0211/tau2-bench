"""Compatibility module for the original Task-18 steering experiments.

New benchmark-wide code should import :mod:`tau2.agent.jlens_failure_metrics`.
The implementation stays here so the original Task-18 scripts remain exactly
reproducible while the public interface is no longer tied to one task.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

TOOL_METRIC_CSV_KEYS = [
    "total_tool_call_count",
    "unique_tool_call_count",
    "repeated_tool_call_count",
    "immediate_repeat_tool_call_count",
    "loop_pattern_count",
    "tool_loop_detected",
    "tool_result_count",
    "tool_call_error_count",
    "unresolved_tool_call_count",
    "max_steps_termination",
    "error_termination",
    "tool_error_reduction_vs_baseline",
    "repeat_reduction_vs_baseline",
    "immediate_repeat_reduction_vs_baseline",
    "loop_eliminated_vs_baseline",
    "max_steps_eliminated_vs_baseline",
]

_ERROR_TERMINATIONS = {
    "agent_error",
    "user_error",
    "max_steps",
    "too_many_errors",
    "infrastructure_error",
    "timeout",
}


def _assistant_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        call
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
        if isinstance(call, dict)
    ]


def _assistant_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        nested = message.get("tool_messages")
        candidates = nested if isinstance(nested, list) else [message]
        results.extend(
            candidate
            for candidate in candidates
            if isinstance(candidate, dict)
            and candidate.get("requestor", "assistant") == "assistant"
        )
    return results


def _fingerprint(call: dict[str, Any]) -> str:
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {"__raw__": arguments}
    payload = json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return f"{call.get('name', '')}:{payload}"


def _loop_pattern_count(fingerprints: list[str], max_period: int = 3) -> int:
    """Count boundaries where an adjacent tool-call block repeats.

    This detects immediate retries (period 1) and short cycles such as A,B,A,B
    (period 2).  One boundary is counted once at its shortest matching period.
    """

    repeats = 0
    for end in range(2, len(fingerprints) + 1):
        for period in range(1, min(max_period, end // 2) + 1):
            if (
                fingerprints[end - 2 * period : end - period]
                == fingerprints[end - period : end]
            ):
                repeats += 1
                break
    return repeats


def trajectory_tool_metrics(
    messages: list[dict[str, Any]], termination_reason: Any
) -> dict[str, Any]:
    """Compute agent tool-use errors and repetition from one simulation."""

    calls = _assistant_tool_calls(messages)
    results = _assistant_tool_results(messages)
    fingerprints = [_fingerprint(call) for call in calls]
    counts = Counter(fingerprints)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    immediate = sum(
        left == right for left, right in zip(fingerprints, fingerprints[1:])
    )
    loop_patterns = _loop_pattern_count(fingerprints)

    result_ids = {
        str(result.get("id"))
        for result in results
        if result.get("id") not in (None, "")
    }
    call_ids = [
        str(call.get("id")) for call in calls if call.get("id") not in (None, "")
    ]
    reason = str(termination_reason or "")
    return {
        "total_tool_call_count": len(calls),
        "unique_tool_call_count": len(counts),
        "repeated_tool_call_count": repeated,
        "immediate_repeat_tool_call_count": immediate,
        "loop_pattern_count": loop_patterns,
        "tool_loop_detected": loop_patterns > 0,
        "tool_result_count": len(results),
        "tool_call_error_count": sum(bool(result.get("error")) for result in results),
        "unresolved_tool_call_count": sum(
            call_id not in result_ids for call_id in call_ids
        ),
        "max_steps_termination": reason == "max_steps",
        "error_termination": reason in _ERROR_TERMINATIONS,
        "repeated_tool_fingerprints": {
            fingerprint: count for fingerprint, count in counts.items() if count > 1
        },
    }


def add_tool_metric_baseline_deltas(
    rows: list[dict[str, Any]], baseline: dict[str, Any] | None
) -> None:
    """Add positive-is-better reductions relative to the baseline condition."""

    for row in rows:
        if baseline is None:
            row.update(
                {
                    "tool_error_reduction_vs_baseline": None,
                    "repeat_reduction_vs_baseline": None,
                    "immediate_repeat_reduction_vs_baseline": None,
                    "loop_eliminated_vs_baseline": None,
                    "max_steps_eliminated_vs_baseline": None,
                }
            )
            continue
        row["tool_error_reduction_vs_baseline"] = (
            baseline["tool_call_error_count"] - row["tool_call_error_count"]
        )
        row["repeat_reduction_vs_baseline"] = (
            baseline["repeated_tool_call_count"] - row["repeated_tool_call_count"]
        )
        row["immediate_repeat_reduction_vs_baseline"] = (
            baseline["immediate_repeat_tool_call_count"]
            - row["immediate_repeat_tool_call_count"]
        )
        row["loop_eliminated_vs_baseline"] = bool(
            baseline["tool_loop_detected"] and not row["tool_loop_detected"]
        )
        row["max_steps_eliminated_vs_baseline"] = bool(
            baseline["max_steps_termination"] and not row["max_steps_termination"]
        )
