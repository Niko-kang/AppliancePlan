#!/usr/bin/env bash
# Single-node (8-GPU) one-shot evaluation runner:
#   open-loop(plan) + close-loop + bbox + keypage
#
# It runs each test script into its own subdirectory under OUT_DIR, then
# runs aggregate_eval_results.py to produce merged files / summary reports.
#
# Example:
#   bash run_all_evals_single_node.sh \
#     --model_path /abs/path/to/checkpoint \
#     --root_path  /path/to/workspace \
#     --plan_data  /abs/path/to/plan_test_dir \
#     --close_loop_data /abs/path/to/close_loop_test_dir \
#     --bbox_data  /abs/path/to/bbox_test_dir \
#     --keypage_data /abs/path/to/keypage_test_dir \
#     --out_dir    /abs/path/to/eval_out
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage:
  bash run_all_evals_single_node.sh \
    --model_path PATH \
    --root_path  PATH \
    --out_dir    PATH \
    [--plan_data PATH] \
    [--close_loop_data PATH] \
    [--bbox_data PATH] \
    [--keypage_data PATH]

Notes:
  - Each *_data must be a directory containing test files expected by the corresponding test_*.py.
  - Missing/empty data dirs are skipped (with a warning).
  - Output layout:
      OUT_DIR/
        plan/
        close_loop/
        bbox/
        keypage/
EOF
}

MODEL_PATH=""
ROOT_PATH="${ROOT_PATH:-}"
OUT_DIR=""
PLAN_DATA=""
CLOSE_LOOP_DATA=""
BBOX_DATA=""
KEYPAGE_DATA=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) MODEL_PATH="${2:-}"; shift 2 ;;
    --root_path) ROOT_PATH="${2:-}"; shift 2 ;;
    --out_dir) OUT_DIR="${2:-}"; shift 2 ;;
    --plan_data) PLAN_DATA="${2:-}"; shift 2 ;;
    --close_loop_data) CLOSE_LOOP_DATA="${2:-}"; shift 2 ;;
    --bbox_data) BBOX_DATA="${2:-}"; shift 2 ;;
    --keypage_data) KEYPAGE_DATA="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${MODEL_PATH}" || -z "${ROOT_PATH}" || -z "${OUT_DIR}" ]]; then
  echo "[ERROR] --model_path, --root_path, --out_dir are required" >&2
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_DIR="${REPO_ROOT}/eval"
mkdir -p "${OUT_DIR}"

is_dir_nonempty() {
  local d="$1"
  [[ -n "${d}" && -d "${d}" && -n "$(ls -A "${d}" 2>/dev/null)" ]]
}

run_task() {
  local task_name="$1"
  local test_script="$2"
  local data_path="$3"
  local task_out_dir="$4"

  if ! is_dir_nonempty "${data_path}"; then
    echo "[WARN] Skip ${task_name}: data dir missing/empty: ${data_path:-<unset>}"
    return 0
  fi

  mkdir -p "${task_out_dir}"
  local log_file="${task_out_dir}/${task_name}_test_$(date +%Y%m%d_%H%M%S).log"

  echo "========================================" | tee -a "${log_file}"
  echo "Task      : ${task_name}" | tee -a "${log_file}"
  echo "Model     : ${MODEL_PATH}" | tee -a "${log_file}"
  echo "Data      : ${data_path}" | tee -a "${log_file}"
  echo "Out dir   : ${task_out_dir}" | tee -a "${log_file}"
  echo "Script    : ${SCRIPT_DIR}/${test_script}" | tee -a "${log_file}"
  echo "Python    : ${PYTHON_BIN}" | tee -a "${log_file}"
  echo "Start     : $(date)" | tee -a "${log_file}"
  echo "========================================" | tee -a "${log_file}"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/${test_script}" \
    --model_name "${MODEL_PATH}" \
    --data_path "${data_path}" \
    --root_path "${ROOT_PATH}" \
    --result_dir "${task_out_dir}" \
    2>&1 | tee -a "${log_file}"
}

aggregate_task() {
  local task_name="$1"
  local test_script="$2"
  local task_out_dir="$3"
  local data_path_for_report="${4:-}"

  local log_file="${task_out_dir}/${task_name}_aggregate_$(date +%Y%m%d_%H%M%S).log"

  # Only aggregate if we have inference outputs
  local jsonl_count
  jsonl_count="$(find "${task_out_dir}" -maxdepth 1 -type f -name "*_inference_*.jsonl" 2>/dev/null | wc -l || true)"
  if [[ "${jsonl_count}" -eq 0 ]]; then
    echo "[WARN] Skip aggregation for ${task_name}: no *_inference_*.jsonl under ${task_out_dir}" | tee -a "${log_file}"
    return 0
  fi

  echo "========================================" | tee -a "${log_file}"
  echo "Aggregate : ${task_name}" | tee -a "${log_file}"
  echo "Out dir   : ${task_out_dir}" | tee -a "${log_file}"
  echo "Count     : ${jsonl_count} jsonl(s)" | tee -a "${log_file}"
  echo "Start     : $(date)" | tee -a "${log_file}"
  echo "========================================" | tee -a "${log_file}"

  "${PYTHON_BIN}" "${EVAL_DIR}/aggregate_eval_results.py" \
    --test_script "${test_script}" \
    --result_dir "${task_out_dir}" \
    --test_data_path "${data_path_for_report}" \
    --model_name "${MODEL_PATH}" \
    --root_path "${ROOT_PATH}" \
    --eval_name "${task_name}" \
    2>&1 | tee -a "${log_file}"
}

# Run + aggregate each task in sequence (simple + robust)
# Default order follows the typical pipeline:
#   keypage -> open-loop(plan) -> bbox -> close-loop
PLAN_OUT="${OUT_DIR}/plan"
CLOSE_LOOP_OUT="${OUT_DIR}/close_loop"
BBOX_OUT="${OUT_DIR}/bbox"
KEYPAGE_OUT="${OUT_DIR}/keypage"

run_task "keypage" "test_keypage.py" "${KEYPAGE_DATA}" "${KEYPAGE_OUT}"
aggregate_task "keypage" "test_keypage.py" "${KEYPAGE_OUT}" "${KEYPAGE_DATA}"

run_task "plan" "test.py" "${PLAN_DATA}" "${PLAN_OUT}"
aggregate_task "plan" "test.py" "${PLAN_OUT}" "${PLAN_DATA}"

run_task "bbox" "test_bbox.py" "${BBOX_DATA}" "${BBOX_OUT}"
aggregate_task "bbox" "test_bbox.py" "${BBOX_OUT}" "${BBOX_DATA}"

run_task "close_loop" "test_close_loop.py" "${CLOSE_LOOP_DATA}" "${CLOSE_LOOP_OUT}"
aggregate_task "close_loop" "test_close_loop.py" "${CLOSE_LOOP_OUT}" "${CLOSE_LOOP_DATA}"

echo "========================================"
echo "All done."
echo "OUT_DIR: ${OUT_DIR}"
echo "Subdirs:"
echo "  - ${PLAN_OUT}"
echo "  - ${CLOSE_LOOP_OUT}"
echo "  - ${BBOX_OUT}"
echo "  - ${KEYPAGE_OUT}"
echo "========================================"


