#!/usr/bin/env bash
# Real bbox evaluation
# Edit the configuration block below before running.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_DIR="${REPO_ROOT}/eval"

########################################
# User configuration
########################################
MODEL_PATHS=(
  "/path/to/checkpoint"
)

REAL_TEST_DATA_PATH="/path/to/real_test_data"

ROOT_PATH="${ROOT_PATH:-/path/to/workspace}"

RESULT_DIRS=(
  "/path/to/output_dir"
)
########################################

if [ ! -d "${REAL_TEST_DATA_PATH}" ]; then
  echo "[ERROR] REAL_TEST_DATA_PATH not found: ${REAL_TEST_DATA_PATH}"
  exit 1
fi

if [ ${#MODEL_PATHS[@]} -ne ${#RESULT_DIRS[@]} ]; then
  echo "[ERROR] MODEL_PATHS and RESULT_DIRS must have the same length"
  exit 1
fi

if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
  echo "[ERROR] MODEL_PATHS is empty"
  exit 1
fi


OVERALL_STATUS=0
for idx in "${!MODEL_PATHS[@]}"; do
  MODEL_PATH="${MODEL_PATHS[$idx]}"
  RESULT_DIR="${RESULT_DIRS[$idx]}"
  mkdir -p "${RESULT_DIR}"
  LOG_FILE="${RESULT_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"

  echo "Running test_bbox.py -> ${RESULT_DIR}" | tee -a "${LOG_FILE}"
  python "${EVAL_DIR}/test_bbox.py" \
    --model_name "${MODEL_PATH}" \
    --data_path "${REAL_TEST_DATA_PATH}" \
    --root_path "${ROOT_PATH}" \
    --result_dir "${RESULT_DIR}" \
    2>&1 | tee -a "${LOG_FILE}"
  TEST_STATUS=$?

  if [ $TEST_STATUS -eq 0 ]; then
    python3 "${EVAL_DIR}/aggregate_eval_results.py" \
      --test_script "test_bbox.py" \
      --result_dir "${RESULT_DIR}" \
      --test_data_path "${REAL_TEST_DATA_PATH}" \
      --model_name "${MODEL_PATH}" \
      --root_path "${ROOT_PATH}" \
      --eval_name "real_bbox" \
      2>&1 | tee -a "${LOG_FILE}" || true
  else
    OVERALL_STATUS=1
  fi
done
exit $OVERALL_STATUS
