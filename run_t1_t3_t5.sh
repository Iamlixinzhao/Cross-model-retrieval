#!/bin/sh
# Single run: train T1+T3+T5 (odd-only), no sweep, fixed set of parameters
set -e
cd "$(dirname "$0")"

EMB_DIR="${EMB_DIR:-/mnt/data/pes/ImageBind/msrvtt_train_embeddings}"
EVAL_EMB_DIR="${EVAL_EMB_DIR:-/mnt/data/pes/ImageBind/msrvtt_results}"
SAVE_DIR="./sweep_runs/run_t1_t3_t5"
CKPT_NAME="best_cheb_gated_asym.pth"

echo "Training T1+T3+T5 (no_t2, include_t5) -> $SAVE_DIR"
python train_cheb_projector.py \
  --emb_dir "$EMB_DIR" \
  --save_dir "$SAVE_DIR" \
  --epochs 20 \
  --batch_size 256 \
  --lr 1e-4 \
  --temperature 0.07 \
  --loss_mode asymmetric \
  --v2t_weight 2.5 \
  --t2v_weight 1.0 \
  --distill_weight 0.0 \
  --no_t2 \
  --include_t5 \
  --infer_vid_ids \
  --caps_per_video 20 \
  --save_name "$CKPT_NAME"

echo "Measuring..."
python measure_latency_memory_variance_cheb.py \
  --cheb_ckpt "$SAVE_DIR/$CKPT_NAME" \
  --emb_dir "$EVAL_EMB_DIR" \
  --save "$SAVE_DIR/metrics.json"

echo "Done. Metrics: $SAVE_DIR/metrics.json"
