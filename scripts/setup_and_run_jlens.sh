#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_and_run_jlens.sh (--pilot|--full) [--start N] [--count N]

  --pilot  Run catalog positions 50..51 (one airline + one retail task).
  --full   Run positions 1..164 (all 50 airline + all 114 retail tasks).
  --start  Override the mode's one-based catalog start position.
  --count  Override the mode's number of consecutive tasks.

Trajectory generation matches the GPT-OSS benchmark path: vLLM agent,
GPT-5.2 user simulator, temperature 1.0, 4096 output tokens, seed 300, verbose
LLM logs, and llm-log-mode=all. J-Lens runs offline only after vLLM stops.

Optional environment variables:
  OPENAI_API_KEY          OpenAI key for the user simulator; prompted if unset.
  TAU2_USER_MODEL         User model (default: gpt-5.2-2025-12-11).
  TAU2_USER_API_BASE      OpenAI-compatible user endpoint.
  TAU2_JLENS_PROFILE      Model/lens profile (default: qwen3.5-4b).
  TAU2_MAX_CONCURRENCY    Concurrent simulations (default: 4).
  TAU2_VLLM_PORT          Local vLLM port (default: 8000).
  TAU2_HEARTBEAT_SEC      Progress heartbeat interval (default: 15).
EOF
}

MODE=""
START_OVERRIDE=""
COUNT_OVERRIDE=""
while (( $# > 0 )); do
  case "$1" in
    --pilot|--full)
      if [[ -n "${MODE}" ]]; then
        echo "ERROR: choose exactly one of --pilot or --full." >&2
        exit 2
      fi
      MODE="${1#--}"
      shift
      ;;
    --start|--count)
      option="$1"
      if (( $# < 2 )); then
        echo "ERROR: ${option} requires a positive integer." >&2
        exit 2
      fi
      if [[ "${option}" == "--start" ]]; then
        START_OVERRIDE="$2"
      else
        COUNT_OVERRIDE="$2"
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${MODE}" ]]; then
  usage >&2
  exit 2
fi
if [[ -n "${START_OVERRIDE}" && ! "${START_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --start must be a positive integer." >&2
  exit 2
fi
if [[ -n "${COUNT_OVERRIDE}" && ! "${COUNT_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --count must be a positive integer." >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(dirname "${PROJECT_DIR}")"
JACOBIAN_DIR="${WORKSPACE_DIR}/jacobian-lens"
RESULTS_ROOT="${TAU2_RESULTS_ROOT:-${WORKSPACE_DIR}/results}"
LOG_ROOT="${TAU2_LOG_ROOT:-${WORKSPACE_DIR}/logs}"
STAMP="$(date +%Y%m%d-%H%M%S)"
HEARTBEAT_SECONDS="${TAU2_HEARTBEAT_SEC:-15}"
PROFILE="${TAU2_JLENS_PROFILE:-qwen3.5-4b}"
USER_MODEL="${TAU2_USER_MODEL:-gpt-5.2-2025-12-11}"
USER_API_BASE="${TAU2_USER_API_BASE:-https://api.openai.com/v1}"
MAX_CONCURRENCY="${TAU2_MAX_CONCURRENCY:-4}"
VLLM_PORT="${TAU2_VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${TAU2_VLLM_MAX_MODEL_LEN:-32768}"
TP_SIZE="${TAU2_TP_SIZE:-1}"
AGENT_TEMPERATURE="${TAU2_AGENT_TEMPERATURE:-1.0}"
AGENT_MAX_TOKENS="${TAU2_AGENT_MAX_TOKENS:-4096}"
SEED="${TAU2_SEED:-300}"

for integer_setting in \
  "${HEARTBEAT_SECONDS}" \
  "${MAX_CONCURRENCY}" \
  "${VLLM_PORT}" \
  "${VLLM_MAX_MODEL_LEN}" \
  "${TP_SIZE}" \
  "${AGENT_MAX_TOKENS}" \
  "${SEED}"; do
  if [[ ! "${integer_setting}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: numeric runner settings must be positive integers." >&2
    exit 2
  fi
done

if [[ "${MODE}" == "pilot" ]]; then
  START=50
  COUNT=2
  RUN_NAME="tau2-airline-retail-pilot"
  RUN_DESCRIPTION="Airline 50 + Retail 1"
else
  START=1
  COUNT=164
  RUN_NAME="tau2-airline-retail-full"
  RUN_DESCRIPTION="Airline 1..50 + Retail 1..114"
fi

CUSTOM_RANGE=0
if [[ -n "${START_OVERRIDE}" ]]; then
  START="${START_OVERRIDE}"
  CUSTOM_RANGE=1
fi
if [[ -n "${COUNT_OVERRIDE}" ]]; then
  COUNT="${COUNT_OVERRIDE}"
  CUSTOM_RANGE=1
fi

END=$((START + COUNT - 1))
RUN_LABEL="$(printf "%04d-%04d" "${START}" "${END}")"
if (( CUSTOM_RANGE )); then
  RUN_NAME="tau2-range"
  RUN_DESCRIPTION="Catalog positions ${START}..${END}"
fi

TRAJECTORY_ROOT="${RESULTS_ROOT}/${RUN_NAME}-${PROFILE}-trajectories-vllm"
TRAJECTORY_DIR="${TRAJECTORY_ROOT}/${RUN_LABEL}"
RESULT_DIR="${RESULTS_ROOT}/${RUN_NAME}-jlens-${STAMP}"
INSPECT_DIR="${RESULTS_ROOT}/${RUN_NAME}-inspect-${STAMP}"
SETUP_LOG="${LOG_ROOT}/${RUN_NAME}-setup-${STAMP}.log"
VLLM_LOG="${LOG_ROOT}/${RUN_NAME}-vllm-${STAMP}.log"
RUN_LOG="${LOG_ROOT}/${RUN_NAME}-run-${STAMP}.log"
INSPECT_LOG="${LOG_ROOT}/${RUN_NAME}-inspect-${STAMP}.log"
ANALYSIS_LOG="${LOG_ROOT}/${RUN_NAME}-jlens-${STAMP}.log"

export PATH="/root/.local/bin:${PATH}"
export HF_HOME="${HF_HOME:-${WORKSPACE_DIR}/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKSPACE_DIR}/.cache/uv}"
export PYTHONUNBUFFERED=1

mkdir -p \
  "${RESULTS_ROOT}" \
  "${LOG_ROOT}" \
  "${HF_HOME}" \
  "${UV_CACHE_DIR}" \
  "${TRAJECTORY_DIR}" \
  "${RESULT_DIR}" \
  "${INSPECT_DIR}"

HEARTBEAT_PID=""
VLLM_PID=""

format_elapsed() {
  local total="$1"
  printf "%02d:%02d:%02d" \
    "$((total / 3600))" \
    "$(((total % 3600) / 60))" \
    "$((total % 60))"
}

heartbeat() {
  local label="$1"
  local started="$2"
  local elapsed gpu result_count llm_log_count view_count

  while true; do
    sleep "${HEARTBEAT_SECONDS}"
    elapsed=$(( $(date +%s) - started ))
    gpu="unavailable"
    if command -v nvidia-smi >/dev/null 2>&1; then
      gpu="$(
        nvidia-smi \
          --query-gpu=utilization.gpu,memory.used,memory.free \
          --format=csv,noheader,nounits 2>/dev/null |
          head -n 1 |
          tr -d '\r' || true
      )"
      gpu="${gpu:-unavailable}"
    fi

    result_count="$(find "${TRAJECTORY_DIR}" -type f -name results.json 2>/dev/null | wc -l | tr -d ' ')"
    llm_log_count="$(find "${TRAJECTORY_DIR}" -type f -path '*/llm_debug/*' 2>/dev/null | wc -l | tr -d ' ')"
    view_count="$(find "${RESULT_DIR}" -type f -name index.html 2>/dev/null | wc -l | tr -d ' ')"
    echo "[WORKING $(format_elapsed "${elapsed}")] ${label} | GPU util/used/free MB: ${gpu} | result files: ${result_count} | LLM logs: ${llm_log_count} | views: ${view_count}"
  done
}

stop_heartbeat() {
  if [[ -n "${HEARTBEAT_PID}" ]]; then
    kill "${HEARTBEAT_PID}" 2>/dev/null || true
    wait "${HEARTBEAT_PID}" 2>/dev/null || true
    HEARTBEAT_PID=""
  fi
}

stop_vllm() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "Stopping vLLM (PID ${VLLM_PID})..."
    kill "${VLLM_PID}" >/dev/null 2>&1 || true
    wait "${VLLM_PID}" >/dev/null 2>&1 || true
  fi
  VLLM_PID=""
}

cleanup() {
  stop_heartbeat
  stop_vllm
}

run_step() {
  local label="$1"
  local log_path="$2"
  shift 2
  local started command_status tee_status
  local -a pipeline_status

  started="$(date +%s)"
  echo
  echo "============================================================"
  echo "START: ${label}"
  echo "Time:  $(date --iso-8601=seconds)"
  echo "Log:   ${log_path}"
  echo "A progress heartbeat will print every ${HEARTBEAT_SECONDS}s."
  echo "============================================================"

  heartbeat "${label}" "${started}" &
  HEARTBEAT_PID=$!

  set +e
  "$@" 2>&1 | tee "${log_path}"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  stop_heartbeat

  command_status="${pipeline_status[0]}"
  tee_status="${pipeline_status[1]}"
  if (( command_status != 0 || tee_status != 0 )); then
    echo "FAILED: ${label} (command=${command_status}, tee=${tee_status})" >&2
    echo "Inspect: ${log_path}" >&2
    if (( command_status != 0 )); then
      return "${command_status}"
    fi
    return "${tee_status}"
  fi
  echo "DONE: ${label} in $(format_elapsed "$(( $(date +%s) - started ))")"
}

trap cleanup EXIT INT TERM

echo "============================================================"
echo "tau2 vLLM trajectory + offline J-Lens"
echo "Mode:       ${MODE}"
echo "Selection:  ${START}..${END} (${COUNT} tasks)"
echo "Workload:   ${RUN_DESCRIPTION}"
echo "Profile:    ${PROFILE}"
echo "User model: ${USER_MODEL}"
echo "Generation: vLLM, temperature=${AGENT_TEMPERATURE}, max_tokens=${AGENT_MAX_TOKENS}"
echo "Analysis:   offline Transformers replay after vLLM shutdown"
echo "============================================================"

if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  echo "Installing git and curl..."
  apt-get update
  apt-get install -y git curl ca-certificates
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="/root/.local/bin:${PATH}"
fi

if [[ -f /venv/main/bin/activate ]]; then
  # Use the Vast.ai CUDA/vLLM runtime for serving. ``uv run`` below still uses
  # tau2-bench's project environment.
  source /venv/main/bin/activate
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "vLLM is missing; installing the same stable runtime used by the benchmark..."
  uv pip install --python "$(command -v python)" "vllm==0.19.1" --torch-backend=cu129
fi
python -c 'import vllm; print("vLLM:", vllm.__version__)'

if [[ ! -d "${JACOBIAN_DIR}/.git" ]]; then
  echo "Cloning Jacobian Lens into ${JACOBIAN_DIR}..."
  git clone \
    --depth 1 \
    --single-branch \
    --branch agent/full-trajectory-viewer \
    https://github.com/jinuk0211/visual-jlens.git \
    "${JACOBIAN_DIR}"
else
  echo "Updating Jacobian Lens..."
  git -C "${JACOBIAN_DIR}" pull \
    --ff-only origin agent/full-trajectory-viewer
fi

cd "${PROJECT_DIR}"
uv python install 3.12
run_step \
  "Install Python dependencies" \
  "${SETUP_LOG}" \
  uv sync --python 3.12 --extra jlens --extra dev

ANALYZER_HELP="$(
  uv run --no-sync python "${JACOBIAN_DIR}/scripts/analyze_tau2.py" --help
)"
for required_option in --top-k --position-chunk-size --max-tracked; do
  if [[ "${ANALYZER_HELP}" != *"${required_option}"* ]]; then
    echo "ERROR: ${JACOBIAN_DIR}/scripts/analyze_tau2.py is missing ${required_option}." >&2
    echo "Update visual-jlens/agent/full-trajectory-viewer before generation." >&2
    exit 1
  fi
done
echo "Jacobian Lens analyzer compatibility check passed."

mapfile -t PROFILE_CONFIG < <(
  uv run --no-sync python - "${PROFILE}" <<'PY'
import sys

from tau2.jlens.profiles import get_profile

profile = get_profile(sys.argv[1])
print(profile.model_id)
print(profile.model_revision)
print(profile.lens_repo)
print(profile.lens_revision)
print(profile.lens_file)
PY
)
if (( ${#PROFILE_CONFIG[@]} != 5 )); then
  echo "ERROR: could not resolve pinned model/lens profile ${PROFILE}." >&2
  exit 1
fi
MODEL_ID="${PROFILE_CONFIG[0]}"
MODEL_REVISION="${PROFILE_CONFIG[1]}"
LENS_REPO="${PROFILE_CONFIG[2]}"
LENS_REVISION="${PROFILE_CONFIG[3]}"
LENS_FILE="${PROFILE_CONFIG[4]}"
SERVED_MODEL="${MODEL_ID##*/}"

case "${PROFILE}" in
  qwen3-8b)
    VLLM_EXTRA_ARGS=(
      --dtype bfloat16
      --enable-auto-tool-choice
      --tool-call-parser hermes
      --reasoning-parser qwen3
    )
    ;;
  qwen3.5-4b|qwen3.6-27b)
    VLLM_EXTRA_ARGS=(
      --dtype bfloat16
      --language-model-only
      --enable-auto-tool-choice
      --tool-call-parser qwen3_coder
      --reasoning-parser qwen3
    )
    ;;
  qwen3.5-9b-base)
    echo "ERROR: ${PROFILE} is a base model and cannot run the comparable tau2 tool-use trajectory." >&2
    exit 2
    ;;
  *)
    echo "ERROR: unsupported vLLM profile: ${PROFILE}" >&2
    exit 2
    ;;
esac

printf '%s\n' "${MODEL_REVISION}" > "${TRAJECTORY_DIR}/model_revision.txt"

if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == "not-needed" ]]; then
  echo
  echo "Enter OPENAI_API_KEY for the GPT-5.2 user simulator. Input is hidden."
  IFS= read -r -s OPENAI_API_KEY </dev/tty
  echo
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is empty." >&2
  exit 1
fi
export OPENAI_API_KEY

if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
  echo "ERROR: port ${VLLM_PORT} already has a model server; refusing to stop an unrelated process." >&2
  exit 1
fi

echo
echo "Starting vLLM: ${MODEL_ID}@${MODEL_REVISION}"
vllm serve "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --served-model-name "${SERVED_MODEL}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization 0.90 \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_CONCURRENCY}" \
  --port "${VLLM_PORT}" \
  "${VLLM_EXTRA_ARGS[@]}" \
  > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

VLLM_READY=0
for attempt in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    VLLM_READY=1
    break
  fi
  if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "ERROR: vLLM exited before becoming ready." >&2
    tail -n 200 "${VLLM_LOG}"
    exit 1
  fi
  if (( attempt % 6 == 0 )); then
    echo "Waiting for vLLM... $((attempt * 5))s"
  fi
  sleep 5
done
if (( VLLM_READY != 1 )); then
  echo "ERROR: vLLM was not ready within 15 minutes." >&2
  tail -n 200 "${VLLM_LOG}"
  exit 1
fi

export HOSTED_VLLM_API_BASE="http://127.0.0.1:${VLLM_PORT}/v1"
export HOSTED_VLLM_API_KEY="dummy"
export OPENAI_API_BASE="${USER_API_BASE}"
export OPENAI_BASE_URL="${USER_API_BASE}"
unset OPENAI_ORGANIZATION OPENAI_ORG_ID OPENAI_PROJECT OPENAI_PROJECT_ID

run_step \
  "Generate ${RUN_DESCRIPTION} with GPT-OSS-compatible settings" \
  "${RUN_LOG}" \
  uv run --no-sync python scripts/run_jlens_range.py \
    --start "${START}" \
    --count "${COUNT}" \
    --profile "${PROFILE}" \
    --user-model "${USER_MODEL}" \
    --user-api-base "${USER_API_BASE}" \
    --agent-api-base "${HOSTED_VLLM_API_BASE}" \
    --trajectory-root "${TRAJECTORY_ROOT}" \
    --temperature "${AGENT_TEMPERATURE}" \
    --max-tokens "${AGENT_MAX_TOKENS}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --seed "${SEED}" \
    --auto-resume

uv run --no-sync python - "${TRAJECTORY_DIR}" "${COUNT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_count = int(sys.argv[2])
selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
expected = {(item["domain"], str(item["task_id"])) for item in selection["tasks"]}
completed = set()
infrastructure_errors = []

for results_path in sorted(root.glob("*/results.json")):
    domain = results_path.parent.name
    results = json.loads(results_path.read_text(encoding="utf-8"))
    for simulation in results.get("simulations", []):
        key = (domain, str(simulation.get("task_id")))
        if simulation.get("termination_reason") == "infrastructure_error":
            infrastructure_errors.append(key)
        elif key in expected:
            completed.add(key)

missing = expected - completed
print(
    f"trajectory status: completed {len(completed)}/{expected_count}, "
    f"infrastructure_error {len(infrastructure_errors)}"
)
if missing or infrastructure_errors or len(expected) != expected_count:
    raise SystemExit(
        f"trajectory generation incomplete; missing={sorted(missing)}, "
        f"infrastructure_errors={infrastructure_errors}"
    )
PY

stop_vllm
echo "vLLM stopped. Waiting 15 seconds for GPU memory release before Transformers replay..."
sleep 15

inspect_trajectories() {
  local run_dir domain found=0
  for run_dir in "${TRAJECTORY_DIR}"/*; do
    [[ -f "${run_dir}/results.json" ]] || continue
    found=1
    domain="$(basename "${run_dir}")"
    uv run --no-sync python "${JACOBIAN_DIR}/scripts/analyze_tau2.py" \
      --run-dir "${run_dir}" \
      --output-dir "${INSPECT_DIR}/${domain}" \
      --model "${MODEL_ID}" \
      --model-revision "${MODEL_REVISION}" \
      --lens-repo "${LENS_REPO}" \
      --lens-revision "${LENS_REVISION}" \
      --lens-file "${LENS_FILE}" \
      --include-successes \
      --call-selection all \
      --inspect-only || return $?
  done
  (( found == 1 ))
}

run_step \
  "Validate standard tau2 results and verbose agent logs" \
  "${INSPECT_LOG}" \
  inspect_trajectories

uv run --no-sync python - "${INSPECT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

manifests = sorted(Path(sys.argv[1]).glob("*/manifest.json"))
if not manifests:
    raise SystemExit("no inspect manifests were produced")
for path in manifests:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    summary = manifest.get("summary", {})
    if summary.get("selected_calls", 0) < 1:
        raise SystemExit(f"no replayable agent calls in {path}")
    if summary.get("cases_without_agent_logs", 0):
        raise SystemExit(f"missing verbose agent logs in {path}")
    print(path.parent.name, summary)
PY

analyze_trajectories() {
  local run_dir domain found=0
  for run_dir in "${TRAJECTORY_DIR}"/*; do
    [[ -f "${run_dir}/results.json" ]] || continue
    found=1
    domain="$(basename "${run_dir}")"
    uv run --no-sync python "${JACOBIAN_DIR}/scripts/analyze_tau2.py" \
      --run-dir "${run_dir}" \
      --output-dir "${RESULT_DIR}/${domain}" \
      --model "${MODEL_ID}" \
      --model-revision "${MODEL_REVISION}" \
      --lens-repo "${LENS_REPO}" \
      --lens-revision "${LENS_REVISION}" \
      --lens-file "${LENS_FILE}" \
      --include-successes \
      --call-selection all \
      --top-k 10 \
      --layer-stride 1 \
      --last-n-tokens 0 \
      --position-chunk-size 32 \
      --max-tracked 128 \
      --max-seq-len 32768 || return $?
  done
  (( found == 1 ))
}

run_step \
  "Teacher-force every saved agent call and analyze every token position" \
  "${ANALYSIS_LOG}" \
  analyze_trajectories

uv run --no-sync python - "${RESULT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

manifests = sorted(Path(sys.argv[1]).glob("*/manifest.json"))
if not manifests:
    raise SystemExit("no J-Lens manifests were produced")
for path in manifests:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    summary = manifest.get("output_summary", {})
    print(path.parent.name, manifest.get("status"), summary)
    if manifest.get("status") != "complete" or summary.get("errors"):
        raise SystemExit(f"J-Lens analysis did not complete cleanly: {path}")
PY

echo "Compressing result directory..."
tar \
  -C "$(dirname "${RESULT_DIR}")" \
  -czf "${RESULT_DIR}.tar.gz" \
  "$(basename "${RESULT_DIR}")"

printf '%s\n' "${TRAJECTORY_DIR}" > "${WORKSPACE_DIR}/latest_tau2_trajectory_dir.txt"
printf '%s\n' "${TRAJECTORY_DIR}" > "${WORKSPACE_DIR}/latest_tau2_trace_dir.txt"
printf '%s\n' "${RESULT_DIR}" > "${WORKSPACE_DIR}/latest_tau2_jlens_result_dir.txt"
printf '%s\n' "${RUN_LOG}" > "${WORKSPACE_DIR}/latest_tau2_run_log.txt"
printf '%s\n' "${ANALYSIS_LOG}" > "${WORKSPACE_DIR}/latest_tau2_analysis_log.txt"

echo
echo "============================================================"
echo "COMPLETE"
echo "Mode:        ${MODE}"
echo "Trajectory:  ${TRAJECTORY_DIR}"
echo "Result:      ${RESULT_DIR}"
echo "Archive:     ${RESULT_DIR}.tar.gz"
echo "vLLM log:    ${VLLM_LOG}"
echo "Run log:     ${RUN_LOG}"
echo "J-Lens log:  ${ANALYSIS_LOG}"
echo "No viewer port was started."
echo "============================================================"
du -sh "${TRAJECTORY_DIR}" "${RESULT_DIR}" "${RESULT_DIR}.tar.gz"

trap - EXIT INT TERM
