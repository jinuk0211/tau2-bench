#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_and_run_jlens.sh --pilot|--full

  --pilot  Analyze catalog positions 50..51 (one airline + one retail task).
  --full   Analyze positions 1..164 (all 50 airline + all 114 retail tasks).

Optional environment variables:
  OPENAI_API_KEY       OpenAI key for the user simulator; prompted if unset.
  TAU2_USER_MODEL      LiteLLM user model (default: openai/gpt-4.1-mini).
  TAU2_USER_API_BASE   OpenAI-compatible base URL.
  TAU2_JLENS_PROFILE   J-Lens profile (default: qwen3.5-4b).
  TAU2_HEARTBEAT_SEC   Progress heartbeat interval (default: 15).
EOF
}

MODE=""
case "${1:-}" in
  --pilot) MODE="pilot" ;;
  --full) MODE="full" ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(dirname "${PROJECT_DIR}")"
JACOBIAN_DIR="${WORKSPACE_DIR}/jacobian-lens"
RESULTS_ROOT="${TAU2_RESULTS_ROOT:-${WORKSPACE_DIR}/results}"
LOG_ROOT="${TAU2_LOG_ROOT:-${WORKSPACE_DIR}/logs}"
STAMP="$(date +%Y%m%d-%H%M%S)"
HEARTBEAT_SECONDS="${TAU2_HEARTBEAT_SEC:-15}"
PROFILE="${TAU2_JLENS_PROFILE:-qwen3.5-4b}"
USER_MODEL="${TAU2_USER_MODEL:-openai/gpt-4.1-mini}"
USER_API_BASE="${TAU2_USER_API_BASE:-https://api.openai.com/v1}"

if [[ ! "${HEARTBEAT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: TAU2_HEARTBEAT_SEC must be a positive integer." >&2
  exit 2
fi

if [[ "${MODE}" == "pilot" ]]; then
  START=50
  COUNT=2
  RUN_NAME="tau2-airline-retail-pilot"
  RUN_DESCRIPTION="Airline 1 task + Retail 1 task"
else
  START=1
  COUNT=164
  RUN_NAME="tau2-airline-retail-full"
  RUN_DESCRIPTION="Airline 50 tasks + Retail 114 tasks"
fi

END=$((START + COUNT - 1))
RUN_LABEL="$(printf "%04d-%04d" "${START}" "${END}")"
TRACE_ROOT="${RESULTS_ROOT}/${RUN_NAME}-traces"
TRACE_DIR="${TRACE_ROOT}/${RUN_LABEL}"
RESULT_DIR="${RESULTS_ROOT}/${RUN_NAME}-jlens-${STAMP}"
INSPECT_DIR="${RESULTS_ROOT}/${RUN_NAME}-inspect-${STAMP}"
SETUP_LOG="${LOG_ROOT}/${RUN_NAME}-setup-${STAMP}.log"
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
  "${TRACE_ROOT}" \
  "${RESULT_DIR}"

HEARTBEAT_PID=""

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
  local elapsed gpu trace_count view_count

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

    trace_count=0
    if [[ -d "${TRACE_DIR}" ]]; then
      trace_count="$(find "${TRACE_DIR}" -type f -name '*.jsonl' 2>/dev/null | wc -l | tr -d ' ')"
    fi

    view_count=0
    if [[ -d "${RESULT_DIR}/views" ]]; then
      view_count="$(find "${RESULT_DIR}/views" -type f -name analysis.json 2>/dev/null | wc -l | tr -d ' ')"
    fi

    echo "[WORKING $(format_elapsed "${elapsed}")] ${label} | GPU util/used/free MB: ${gpu} | traces: ${trace_count} | completed views: ${view_count}"
  done
}

stop_heartbeat() {
  if [[ -n "${HEARTBEAT_PID}" ]]; then
    kill "${HEARTBEAT_PID}" 2>/dev/null || true
    wait "${HEARTBEAT_PID}" 2>/dev/null || true
    HEARTBEAT_PID=""
  fi
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

trap stop_heartbeat EXIT INT TERM

echo "============================================================"
echo "tau2 full-position J-Lens"
echo "Mode:      ${MODE}"
echo "Selection: ${START}..${END} (${COUNT} tasks)"
echo "Workload:  ${RUN_DESCRIPTION}"
echo "Profile:   ${PROFILE}"
echo "No HTTP server or viewer port will be started."
echo "============================================================"

if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  echo "Installing git and curl..."
  apt-get update
  apt-get install -y git curl
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="/root/.local/bin:${PATH}"
fi

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

if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == "not-needed" ]]; then
  echo
  echo "OpenAI API key를 붙여넣고 Enter를 누르세요."
  echo "보안을 위해 입력 내용은 화면에 표시되지 않습니다."
  IFS= read -r -s OPENAI_API_KEY </dev/tty
  echo
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OpenAI API key가 비어 있습니다." >&2
  exit 1
fi
export OPENAI_API_KEY
echo "OpenAI API key accepted. Starting the benchmark."

run_step \
  "Generate ${RUN_DESCRIPTION}" \
  "${RUN_LOG}" \
  uv run --no-sync python scripts/run_jlens_range.py \
    --start "${START}" \
    --count "${COUNT}" \
    --profile "${PROFILE}" \
    --user-model "${USER_MODEL}" \
    --user-api-base "${USER_API_BASE}" \
    --trace-root "${TRACE_ROOT}" \
    --dtype bfloat16 \
    --max-new-tokens 256 \
    --max-concurrency 1 \
    --auto-resume

TRACE_COUNT="$(find "${TRACE_DIR}" -type f -name '*.jsonl' | wc -l | tr -d ' ')"
echo "Trace files present: ${TRACE_COUNT} (expected at least ${COUNT})"
if (( TRACE_COUNT < COUNT )); then
  echo "ERROR: not every selected task produced a trace file." >&2
  echo "Re-run the same mode to continue with auto-resume." >&2
  exit 1
fi

run_step \
  "Validate exact token IDs and hashes" \
  "${INSPECT_LOG}" \
  uv run --no-sync tau2 jlens "${TRACE_DIR}" \
    --profile "${PROFILE}" \
    --output-dir "${INSPECT_DIR}" \
    --inspect-only

run_step \
  "Analyze every token position and fitted layer" \
  "${ANALYSIS_LOG}" \
  uv run --no-sync tau2 jlens "${TRACE_DIR}" \
    --profile "${PROFILE}" \
    --output-dir "${RESULT_DIR}" \
    --top-k 10 \
    --layer-stride 1 \
    --position-chunk-size 32 \
    --max-tracked 128 \
    --max-seq-len 32768

uv run --no-sync python - "${RESULT_DIR}/manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
entries = manifest.get("entries", [])
ok = sum(entry.get("status") == "ok" for entry in entries)
errors = sum(entry.get("status") == "error" for entry in entries)

print("status:", manifest.get("status"))
print("records:", len(entries))
print("ok:", ok)
print("errors:", errors)
print("all positions:", manifest.get("all_positions"))
print("layer stride:", manifest.get("layer_stride"))

if manifest.get("status") != "complete" or errors:
    raise SystemExit("J-Lens analysis did not complete without errors")
PY

echo "Compressing result directory..."
tar \
  -C "$(dirname "${RESULT_DIR}")" \
  -czf "${RESULT_DIR}.tar.gz" \
  "$(basename "${RESULT_DIR}")"

printf '%s\n' "${TRACE_DIR}" > "${WORKSPACE_DIR}/latest_tau2_trace_dir.txt"
printf '%s\n' "${RESULT_DIR}" > "${WORKSPACE_DIR}/latest_tau2_jlens_result_dir.txt"
printf '%s\n' "${RUN_LOG}" > "${WORKSPACE_DIR}/latest_tau2_run_log.txt"
printf '%s\n' "${ANALYSIS_LOG}" > "${WORKSPACE_DIR}/latest_tau2_analysis_log.txt"

echo
echo "============================================================"
echo "COMPLETE"
echo "Mode:     ${MODE}"
echo "Trace:    ${TRACE_DIR}"
echo "Result:   ${RESULT_DIR}"
echo "Archive:  ${RESULT_DIR}.tar.gz"
echo "Run log:  ${RUN_LOG}"
echo "JLens log:${ANALYSIS_LOG}"
echo "No port was used."
echo "============================================================"
du -sh "${TRACE_DIR}" "${RESULT_DIR}" "${RESULT_DIR}.tar.gz"

trap - EXIT
