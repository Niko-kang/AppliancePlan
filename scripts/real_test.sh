#!/usr/bin/env bash
# Real open-loop (plan) evaluation
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
  LOG_FILE="${RESULT_DIR}/real_test_$(date +%Y%m%d_%H%M%S).log"

  if [ ! -d "${RESULT_DIR}" ]; then
    echo "[ERROR] Cannot create or access RESULT_DIR: ${RESULT_DIR}" | tee -a "${LOG_FILE}"
    OVERALL_STATUS=1
    continue
  fi

  if [ ! -e "${MODEL_PATH}" ]; then
    echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}" | tee -a "${LOG_FILE}"
    OVERALL_STATUS=1
    continue
  fi

  {
    echo "========================================"
    echo " Real Plan Test - Start"
    echo "========================================"
    echo "MODEL_PATH           : ${MODEL_PATH}"
    echo "REAL_TEST_DATA_PATH  : ${REAL_TEST_DATA_PATH}"
    echo "ROOT_PATH            : ${ROOT_PATH}"
    echo "RESULT_DIR           : ${RESULT_DIR}"
    echo "Start Time           : $(date)"
    echo "========================================"
  } | tee -a "${LOG_FILE}"

  python "${EVAL_DIR}/test.py" \
    --model_name "${MODEL_PATH}" \
    --data_path "${REAL_TEST_DATA_PATH}" \
    --root_path "${ROOT_PATH}" \
    --result_dir "${RESULT_DIR}" \
    2>&1 | tee -a "${LOG_FILE}"

  TEST_STATUS=$?

  if [ ${TEST_STATUS} -eq 0 ]; then
    {
      echo "========================================"
      echo " Real Plan Test - Done"
      echo "========================================"
      echo "Starting aggregation of results..."
    } | tee -a "${LOG_FILE}"

    JSONL_COUNT=$(find "${RESULT_DIR}" -maxdepth 1 -type f -name "*_inference_*.jsonl" 2>/dev/null | wc -l)
    if [ ${JSONL_COUNT} -eq 0 ]; then
      echo "[WARNING] No inference result files found (*_inference_*.jsonl), skipping aggregation" | tee -a "${LOG_FILE}"
    else
      echo "Found ${JSONL_COUNT} inference result file(s), starting aggregation..." | tee -a "${LOG_FILE}"

      python3 "${EVAL_DIR}/aggregate_eval_results.py" \
        --test_script "test.py" \
        --result_dir "${RESULT_DIR}" \
        --test_data_path "${REAL_TEST_DATA_PATH}" \
        --model_name "${MODEL_PATH}" \
        --root_path "${ROOT_PATH}" \
        --eval_name "真实测试集" \
        2>&1 | tee -a "${LOG_FILE}"

      AGG_STATUS=$?

      if [ ${AGG_STATUS} -eq 0 ]; then
        echo "✓ Plan aggregation completed successfully" | tee -a "${LOG_FILE}"

        SUMMARY_FILES=$(find "${RESULT_DIR}" -maxdepth 1 -type f \( -name "*.txt" -o -name "*summary*.txt" \) 2>/dev/null | wc -l)
        if [ ${SUMMARY_FILES} -gt 0 ]; then
          echo "Summary files found: ${SUMMARY_FILES}" | tee -a "${LOG_FILE}"
          echo "Summary files:" | tee -a "${LOG_FILE}"
          find "${RESULT_DIR}" -maxdepth 1 -type f \( -name "*.txt" -o -name "*summary*.txt" \) -exec basename {} \; 2>/dev/null | tee -a "${LOG_FILE}"
        else
          echo "[WARNING] No summary files found after aggregation" | tee -a "${LOG_FILE}"
        fi
      else
        echo "[ERROR] Plan aggregation failed (exit ${AGG_STATUS})" | tee -a "${LOG_FILE}"
        OVERALL_STATUS=1
      fi
    fi
  else
    {
      echo "========================================"
      echo " Real Plan Test - Failed (exit ${TEST_STATUS})"
      echo "Skipping aggregation due to test failure"
    } | tee -a "${LOG_FILE}"
    OVERALL_STATUS=1
  fi

  {
    echo "========================================"
    echo "End Time: $(date)"
    echo "Logs saved to: ${LOG_FILE}"
    echo "Results dir:  ${RESULT_DIR}"
    echo "========================================"
  } | tee -a "${LOG_FILE}"
done

exit ${OVERALL_STATUS}


