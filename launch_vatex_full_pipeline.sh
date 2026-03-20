#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${1:-/data2/vatex_experiments}"
PCME_EPOCHS="${PCME_EPOCHS:-40}"
CHEB_EPOCHS="${CHEB_EPOCHS:-20}"

mkdir -p "${ROOT_DIR}/logs"

echo "Launching VATEX full pipeline in background..."
echo "  root: ${ROOT_DIR}"
echo "  log:  ${ROOT_DIR}/logs/pipeline.log"

CONDA_BIN="${CONDA_BIN:-/mnt/data/miniconda3/bin/conda}"
nohup "${CONDA_BIN}" run --no-capture-output -n cross python "${REPO_DIR}/run_vatex_full_pipeline.py" \
  --root "${ROOT_DIR}" \
  --repo_dir "${REPO_DIR}" \
  --pcme_epochs "${PCME_EPOCHS}" \
  --cheb_epochs "${CHEB_EPOCHS}" \
  > "${ROOT_DIR}/logs/pipeline.log" 2>&1 &

echo $! > "${ROOT_DIR}/logs/pipeline.pid"
echo "Started PID: $(cat "${ROOT_DIR}/logs/pipeline.pid")"
echo "Monitor: tail -f ${ROOT_DIR}/logs/pipeline.log"
