"""Serve Tau2 J-Lens generation on a remote GPU host."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException

from tau2.agent.jlens_backend import HFBackendConfig, InstrumentedHFBackend
from tau2.agent.jlens_remote_backend import (
    backend_cache_key,
    execute_remote_payload,
    hf_config_from_wire,
)

LOGGER = logging.getLogger("uvicorn.error")
GENERATION_HEARTBEAT_SECONDS = 30.0


def _release_cuda_cache() -> None:
    """Release completed/failed generation allocations before the next request."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        LOGGER.exception("Failed to release the CUDA cache")


def _log_generation_heartbeat(
    stop_event: threading.Event,
    *,
    task_id: str,
    turn_index: int,
    started_at: float,
    interval_seconds: float = GENERATION_HEARTBEAT_SECONDS,
) -> None:
    while not stop_event.wait(interval_seconds):
        LOGGER.info(
            "Qwen generation running task=%s turn=%s elapsed=%.1fs",
            task_id,
            turn_index,
            time.perf_counter() - started_at,
        )


def worker_preflight_report(
    artifact_paths: list[str],
    environment: dict[str, str],
    *,
    cuda_available: bool,
    cuda_device_count: int,
    loaded_backends: int,
) -> dict[str, Any]:
    """Report remote execution readiness without loading a model or a lens."""
    normalized = sorted({str(Path(path).expanduser()) for path in artifact_paths})
    missing = [path for path in normalized if not Path(path).is_file()]
    ready = bool(cuda_available) and not missing
    return {
        "status": "ok" if ready else "not_ready",
        "cuda_available": bool(cuda_available),
        "cuda_device_count": int(cuda_device_count),
        "loaded_backends": int(loaded_backends),
        "artifacts_checked": len(normalized),
        "missing_artifacts": missing,
        "huggingface_token_present": bool(
            environment.get("HF_TOKEN") or environment.get("HUGGING_FACE_HUB_TOKEN")
        ),
    }


def create_app(*, token: str) -> FastAPI:
    if not token:
        raise ValueError("the remote worker requires a non-empty bearer token")
    app = FastAPI(title="Tau2 J-Lens remote worker")
    cache: dict[str, InstrumentedHFBackend] = {}
    cache_lock = threading.Lock()
    generation_lock = threading.Lock()

    def load_backend(config: HFBackendConfig) -> InstrumentedHFBackend:
        key = backend_cache_key(config)
        with cache_lock:
            backend = cache.get(key)
            if backend is None:
                backend = InstrumentedHFBackend.from_pretrained(config)
                cache[key] = backend
        return backend

    def require_authorization(authorization: str | None) -> None:
        if not secrets.compare_digest(authorization or "", f"Bearer {token}"):
            raise HTTPException(status_code=401, detail="invalid bearer token")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "loaded_backends": len(cache)}

    @app.post("/v1/preflight")
    def preflight(
        payload: dict[str, Any],
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_authorization(authorization)
        raw_paths = payload.get("artifact_paths") or []
        if not isinstance(raw_paths, list) or not all(
            isinstance(path, str) and path for path in raw_paths
        ):
            raise HTTPException(
                status_code=422,
                detail="artifact_paths must be a list of non-empty strings",
            )
        import torch

        return worker_preflight_report(
            raw_paths,
            os.environ,
            cuda_available=torch.cuda.is_available(),
            cuda_device_count=torch.cuda.device_count(),
            loaded_backends=len(cache),
        )

    @app.post("/v1/generate")
    def generate(
        payload: dict[str, Any],
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_authorization(authorization)
        try:
            # Validate before entering the model cache so malformed requests fail fast.
            hf_config_from_wire(dict(payload.get("backend_config") or {}))
            # Hooks mutate shared module registration state. Serialize generations
            # until the backend gains a process-per-GPU scheduler.
            with generation_lock:
                _release_cuda_cache()
                task_id = str(payload.get("task_id", "unknown"))
                turn_index = int(payload.get("turn_index", -1))
                started_at = time.perf_counter()
                stop_heartbeat = threading.Event()
                heartbeat = threading.Thread(
                    target=_log_generation_heartbeat,
                    kwargs={
                        "stop_event": stop_heartbeat,
                        "task_id": task_id,
                        "turn_index": turn_index,
                        "started_at": started_at,
                    },
                    name=f"jlens-heartbeat-{task_id}-{turn_index}",
                    daemon=True,
                )
                LOGGER.info(
                    "Qwen generation started task=%s turn=%s",
                    task_id,
                    turn_index,
                )
                heartbeat.start()
                try:
                    response = execute_remote_payload(
                        payload, backend_loader=load_backend
                    )
                except Exception:
                    LOGGER.exception(
                        "Qwen generation failed task=%s turn=%s elapsed=%.1fs",
                        task_id,
                        turn_index,
                        time.perf_counter() - started_at,
                    )
                    raise
                finally:
                    stop_heartbeat.set()
                    heartbeat.join(timeout=1.0)
                    _release_cuda_cache()
                LOGGER.info(
                    "Qwen generation completed task=%s turn=%s elapsed=%.1fs tokens=%s",
                    task_id,
                    turn_index,
                    time.perf_counter() - started_at,
                    len(response.get("generated_ids") or []),
                )
                return response
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8877, type=int)
    parser.add_argument("--token-env", default="JLENS_REMOTE_TOKEN")
    args = parser.parse_args()
    token = os.environ.get(args.token_env, "")
    if not token:
        raise SystemExit(f"missing remote worker token in {args.token_env}")
    import uvicorn

    uvicorn.run(create_app(token=token), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
