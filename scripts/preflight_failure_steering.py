"""Validate remote credentials and the compiled baseline matrix without loading a model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from tau2.agent.jlens_failure_protocol import load_failure_steering_matrix

SUPPORTED_METHODS = {"caa", "cast", "mera", "sadi", "iti", "austeer", "jservo"}
REVIEW_KEY_ENVS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")


def _load_matrix(path: Path) -> dict[str, Any]:
    return load_failure_steering_matrix(path)


def preflight_report(
    matrix: dict[str, Any],
    environment: Mapping[str, str],
    *,
    health: dict[str, Any] | None,
    method: str,
    require_hf_token: bool,
) -> dict[str, Any]:
    execution = matrix["execution"]
    endpoint_env = str(execution["endpoint_env"])
    token_env = str(execution["token_env"])
    methods = {
        str(item.get("method"))
        for item in matrix.get("conditions") or []
        if item.get("method") != "none"
    }
    selected_method_present = (
        method in {"baseline", "all"} or method in methods
    )
    checks = {
        "matrix_fingerprint_verified": True,
        "matrix_methods_supported": bool(methods)
        and methods.issubset(SUPPORTED_METHODS),
        "selected_method_present": selected_method_present,
        "remote_endpoint_present": bool(environment.get(endpoint_env)),
        "remote_token_present": bool(environment.get(token_env)),
        "review_provider_key_present": any(
            environment.get(name) for name in REVIEW_KEY_ENVS
        ),
        "remote_health_ok": bool(health and health.get("status") == "ok"),
        "remote_cuda_available": bool(health and health.get("cuda_available")),
        "remote_artifacts_ready": bool(
            health
            and health.get("status") in {"ok", "not_ready"}
            and health.get("missing_artifacts") == []
        ),
    }
    if require_hf_token:
        checks["remote_huggingface_token_present"] = bool(
            health and health.get("huggingface_token_present")
        )
    return {
        "schema_version": "failure-steering-preflight-v1",
        "ready": all(checks.values()),
        "checks": checks,
        "matrix_fingerprint": matrix["matrix_fingerprint"],
        "conditions": len(matrix.get("conditions") or []),
        "methods": sorted(methods),
        "selected_method": method,
        "remote_health": health,
        "notes": {
            "secrets": "Only presence booleans are reported; values are never serialized.",
            "huggingface": (
                "Checked on the remote worker only when --require-hf-token is set."
            ),
            "review": "Required for the Tau2 user simulator and full agent/user review.",
        },
    }


def artifact_paths_for_method(matrix: dict[str, Any], method: str) -> list[str]:
    """Return unique remote artifacts required by one method or the full sweep."""
    if method == "baseline":
        return []
    selected_methods = methods_in_matrix(matrix) if method == "all" else {method}
    paths = {
        str(intervention["vector_path"])
        for condition in matrix.get("conditions") or []
        if condition.get("method") in selected_methods
        for intervention in [
            condition.get("agent_llm_args", {}).get("jlens_intervention") or {}
        ]
        if intervention.get("vector_path")
    }
    paths.update(
        str(controller["artifact_path"])
        for condition in matrix.get("conditions") or []
        if condition.get("method") in selected_methods
        for controller in [
            condition.get("agent_llm_args", {}).get("jlens_controller") or {}
        ]
        if controller.get("artifact_path")
    )
    return sorted(paths)


def methods_in_matrix(matrix: dict[str, Any]) -> set[str]:
    return {
        str(condition.get("method"))
        for condition in matrix.get("conditions") or []
        if condition.get("method") != "none"
    }


def request_remote_preflight(
    endpoint: str,
    token: str,
    *,
    artifact_paths: list[str],
    timeout: float,
) -> dict[str, Any]:
    import requests

    response = requests.post(
        f"{endpoint.rstrip('/')}/v1/preflight",
        headers={"Authorization": f"Bearer {token}"},
        json={"artifact_paths": artifact_paths},
        timeout=float(timeout),
    )
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError("remote preflight returned a non-object response")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument(
        "--method",
        choices=[
            "baseline",
            "caa",
            "cast",
            "mera",
            "sadi",
            "iti",
            "austeer",
            "jservo",
            "all",
        ],
        default="baseline",
    )
    parser.add_argument("--require-hf-token", action="store_true")
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    matrix = _load_matrix(args.matrix)
    endpoint_env = str(matrix["execution"]["endpoint_env"])
    token_env = str(matrix["execution"]["token_env"])
    endpoint = os.environ.get(endpoint_env)
    token = os.environ.get(token_env)
    health = None
    if endpoint and token and not args.skip_health:
        health = request_remote_preflight(
            endpoint,
            token,
            artifact_paths=artifact_paths_for_method(matrix, args.method),
            timeout=args.timeout,
        )
    elif args.skip_health:
        health = {"status": "skipped"}
    report = preflight_report(
        matrix,
        os.environ,
        health=health,
        method=args.method,
        require_hf_token=args.require_hf_token,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
