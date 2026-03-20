#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_BIN="${CONDA_BIN:-/mnt/data/miniconda3/bin/conda}"
PY_ENV="${PY_ENV:-cross}"

PAIR_DIR="${PAIR_DIR:-${REPO_DIR}/dataset_splits_msvd_multi}"
TRAIN_PAIRS="${TRAIN_PAIRS:-${PAIR_DIR}/msvd_train_pairs.json}"
TEST_PAIRS="${TEST_PAIRS:-${PAIR_DIR}/msvd_test_pairs.json}"

TRAIN_EMB="${TRAIN_EMB:-${REPO_DIR}/msvd_multi_imagebind_train_embeddings}"
TEST_EMB="${TEST_EMB:-${REPO_DIR}/msvd_multi_imagebind_test_embeddings}"

RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/sweep_runs_msvd_multi}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
mkdir -p "${RUN_ROOT}" "${LOG_DIR}"

PIPELINE_LOG="${LOG_DIR}/pipeline.log"
PIPELINE_PID="${LOG_DIR}/pipeline.pid"

cat > "${LOG_DIR}/run_pipeline_inner.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$1"
CONDA_BIN="$2"
PY_ENV="$3"
TRAIN_PAIRS="$4"
TEST_PAIRS="$5"
TRAIN_EMB="$6"
TEST_EMB="$7"
RUN_ROOT="$8"

run_py() {
  "${CONDA_BIN}" run --no-capture-output -n "${PY_ENV}" python "$@"
}

echo "[info] waiting for active VATEX ImageBind jobs..."
while pgrep -af "generate_imagebind_embeddings_generic.py.*vatex_experiments" >/dev/null; do
  sleep 30
done
echo "[info] no active VATEX embedding job, start MSVD multi pipeline."

mkdir -p "${RUN_ROOT}"

if [ ! -f "${TRAIN_EMB}/emb_text.pt" ]; then
  echo "[stage] msvd train embeddings"
  run_py "${REPO_DIR}/generate_imagebind_embeddings_generic.py" \
    --pairs_json "${TRAIN_PAIRS}" \
    --output_dir "${TRAIN_EMB}" \
    --num_frames 16 --image_size 224 --text_batch_size 512 --use_fp16
else
  echo "[skip] train embeddings already exist"
fi

if [ ! -f "${TEST_EMB}/emb_text.pt" ]; then
  echo "[stage] msvd test embeddings"
  run_py "${REPO_DIR}/generate_imagebind_embeddings_generic.py" \
    --pairs_json "${TEST_PAIRS}" \
    --output_dir "${TEST_EMB}" \
    --num_frames 16 --image_size 224 --text_batch_size 512 --use_fp16
else
  echo "[skip] test embeddings already exist"
fi

echo "[stage] pcme train+eval"
mkdir -p "${RUN_ROOT}/run_pcme_msvd_multi"
run_py "${REPO_DIR}/train_pcme_projector.py" \
  --emb_dir "${TRAIN_EMB}" \
  --save_dir "${RUN_ROOT}/run_pcme_msvd_multi" \
  --epochs 40
run_py "${REPO_DIR}/measure_latency_memory_variance.py" \
  --emb_dir "${TEST_EMB}" \
  --ckpt "${RUN_ROOT}/run_pcme_msvd_multi/best_projectors.pth" \
  --save "${RUN_ROOT}/run_pcme_msvd_multi/metrics.json"

for DEG in 3 4 5 6; do
  OUT_DIR="${RUN_ROOT}/run_gaussian_cheb_msvd_multi_deg${DEG}"
  mkdir -p "${OUT_DIR}"
  echo "[stage] cheb train+eval deg=${DEG}"
  run_py "${REPO_DIR}/train_cheb_projector_v2.py" \
    --emb_dir "${TRAIN_EMB}" \
    --save_dir "${OUT_DIR}" \
    --save_name "best_gaussian_cheb_bilinear.pth" \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --loss_mode asymmetric \
    --t2v_weight 1.0 \
    --v2t_weight 2.5 \
    --mu_on_sphere \
    --kernel_use_mu_residual \
    --cheb_order "${DEG}"
  run_py "${REPO_DIR}/measure_cheb_v2.py" \
    --emb_dir "${TEST_EMB}" \
    --cheb_ckpt "${OUT_DIR}/best_gaussian_cheb_bilinear.pth" \
    --baseline_name "ImageBind" \
    --save "${OUT_DIR}/metrics.json"
done

echo "[stage] summary table"
export RUN_ROOT
run_py - <<'PY'
import json, os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
rows = []

pcme = json.load(open(run_root / "run_pcme_msvd_multi" / "metrics.json"))
ret = pcme["summary"]["retrieval"]
rows.append(["ImageBind","-",
             ret["imagebind"]["t2v"]["R@k"]["1"], ret["imagebind"]["t2v"]["R@k"]["5"], ret["imagebind"]["t2v"]["R@k"]["10"],
             ret["imagebind"]["v2t"]["R@k"]["1"], ret["imagebind"]["v2t"]["R@k"]["5"], ret["imagebind"]["v2t"]["R@k"]["10"]])
rows.append(["PCME","-",
             ret["projector"]["t2v"]["R@k"]["1"], ret["projector"]["t2v"]["R@k"]["5"], ret["projector"]["t2v"]["R@k"]["10"],
             ret["projector"]["v2t"]["R@k"]["1"], ret["projector"]["v2t"]["R@k"]["5"], ret["projector"]["v2t"]["R@k"]["10"]])

for deg in [3,4,5,6]:
    m = json.load(open(run_root / f"run_gaussian_cheb_msvd_multi_deg{deg}" / "metrics.json"))
    r = m["summary"]["retrieval"]["gaussian_cheb_v2"]
    rows.append([f"Cheb-{deg}",str(deg),
                 r["t2v"]["R@k"]["1"], r["t2v"]["R@k"]["5"], r["t2v"]["R@k"]["10"],
                 r["v2t"]["R@k"]["1"], r["v2t"]["R@k"]["5"], r["v2t"]["R@k"]["10"]])

headers = ["model","degree","t2v_r1","t2v_r5","t2v_r10","v2t_r1","v2t_r5","v2t_r10"]
md = run_root / "msvd_multi_summary.md"
csv = run_root / "msvd_multi_summary.csv"

with open(csv, "w", encoding="utf-8") as f:
    f.write(",".join(headers) + "\n")
    for r in rows:
        f.write(",".join(str(x) for x in r) + "\n")

with open(md, "w", encoding="utf-8") as f:
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
    for r in rows:
        f.write("| " + " | ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in r) + " |\n")

print("saved", csv)
print("saved", md)
PY

echo "[done] msvd multi pipeline complete"
EOF

chmod +x "${LOG_DIR}/run_pipeline_inner.sh"

echo "Launching MSVD multi pipeline in background..."
echo "  log: ${PIPELINE_LOG}"

nohup bash "${LOG_DIR}/run_pipeline_inner.sh" \
  "${REPO_DIR}" "${CONDA_BIN}" "${PY_ENV}" \
  "${TRAIN_PAIRS}" "${TEST_PAIRS}" \
  "${TRAIN_EMB}" "${TEST_EMB}" "${RUN_ROOT}" \
  > "${PIPELINE_LOG}" 2>&1 &

echo $! > "${PIPELINE_PID}"
echo "Started PID: $(cat "${PIPELINE_PID}")"
echo "Monitor: tail -f ${PIPELINE_LOG}"
