#!/bin/sh
set -e
cd "$(dirname "$0")"

# Compare scalar vs vector gate_mode, T1+T3 vs T1+T3+T5.
# Uncomment one run_one below and run: sh run_gate_mode_compare.sh
#
# 1) scalar T1+T3:
#   python train_cheb_projector.py --emb_dir /mnt/data/pes/ImageBind/msrvtt_train_embeddings --save_dir ./sweep_runs/run_scalar_t1t3 --epochs 40 --batch_size 256 --lr 1e-4 --temperature 0.07 --loss_mode asymmetric --v2t_weight 2.5 --t2v_weight 1.0 --distill_weight 0.0 --no_t2 --infer_vid_ids --caps_per_video 20 --save_name best_cheb_gated_asym.pth
# 2) scalar T1+T3+T5: add --include_t5
# 3) vector T1+T3: add --gate_mode vector
# 4) vector T1+T3+T5: add --gate_mode vector --include_t5
# Measure: python measure_latency_memory_variance_cheb.py --cheb_ckpt SAVE_DIR/best_cheb_gated_asym.pth --emb_dir /mnt/data/pes/ImageBind/msrvtt_results --save SAVE_DIR/metrics.json

EMB_DIR="${EMB_DIR:-/mnt/data/pes/ImageBind/msrvtt_train_embeddings}"
EVAL_EMB_DIR="${EVAL_EMB_DIR:-/mnt/data/pes/ImageBind/msrvtt_results}"
CKPT_NAME="best_cheb_gated_asym.pth"

# 公共参数
EPOCHS=40
BATCH=256
LR=1e-4
TEMP=0.07
LOSS=asymmetric
V2T_W=2.5
DISTILL=0.0

run_one() {
  _name="$1"
  _save="$2"
  shift 2
  echo "========== $_name -> $_save =========="
  python train_cheb_projector.py \
    --emb_dir "$EMB_DIR" \
    --save_dir "$_save" \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --lr $LR \
    --temperature $TEMP \
    --loss_mode $LOSS \
    --v2t_weight $V2T_W \
    --t2v_weight 1.0 \
    --distill_weight $DISTILL \
    --no_t2 \
    --infer_vid_ids \
    --caps_per_video 20 \
    --save_name "$CKPT_NAME" \
    "$@"
  python measure_latency_memory_variance_cheb.py \
    --cheb_ckpt "$_save/$CKPT_NAME" \
    --emb_dir "$EVAL_EMB_DIR" \
    --save "$_save/metrics.json"
  echo "Done: $_save/metrics.json"
}

# ----- 1. scalar + T1+T3（基线，和之前 38% 一致） -----
# run_one "scalar T1+T3" "./sweep_runs/run_scalar_t1t3"

# ----- 2. scalar + T1+T3+T5 -----
# run_one "scalar T1+T3+T5" "./sweep_runs/run_scalar_t1t3t5" --include_t5

# ----- 3. vector + T1+T3 (per-dim gates) -----
run_one "vector T1+T3" "./sweep_runs/run_vector_t1t3" --gate_mode vector

# ----- 4. vector + T1+T3+T5 -----
run_one "vector T1+T3+T5" "./sweep_runs/run_vector_t1t3t5" --gate_mode vector --include_t5
