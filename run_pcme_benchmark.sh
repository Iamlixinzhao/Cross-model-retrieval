#!/bin/bash

#$ -M email address
#$ -m abe
#$ -pe smp 4
#$ -q gpu
#$ -l gpu_card=1
#$ -N PCME_Benchmark

set -e

conda activate imagebind
export PYTHONPATH="/scratch365/jzheng7/ImageBind:${PYTHONPATH}"
cd /scratch365/jzheng7/ImageBind

ROOT="/scratch365/jzheng7/ImageBind"
TRAIN_EMB_DIR="$ROOT/msrvtt_train_embeddings"
TEST_EMB_DIR="$ROOT/msrvtt_results"
CKPT_DIR="$ROOT/pcme_checkpoints_correct"

echo "========================================"
echo "  PCME Benchmark Pipeline (CORRECTED)"
echo "========================================"
echo "Start: $(date)"
echo ""

# ============================================
# Step 0: Verify no data leakage
# ============================================
echo "[0/4] Data Leakage Check..."
echo ""

if [[ -d "$TRAIN_EMB_DIR" ]]; then
    train_size=$(python -c "import torch; print(torch.load('$TRAIN_EMB_DIR/emb_text.pt').shape[0])")
    echo "  ✓ Training set size: $train_size samples"

    if [[ "$train_size" != "6513" ]]; then
        echo "  ⚠️  WARNING: Expected 6513 training samples, got $train_size"
    fi
else
    echo "  ✗ Training embeddings not found"
    train_size=0
fi

if [[ -d "$TEST_EMB_DIR" ]]; then
    test_size=$(python -c "import torch; print(torch.load('$TEST_EMB_DIR/emb_text.pt').shape[0])")
    echo "  ✓ Test set size: $test_size samples"

    if [[ "$test_size" != "1000" ]]; then
        echo "  ⚠️  WARNING: Expected 1000 test samples (MSR-VTT 1kA), got $test_size"
    fi
else
    echo "  ✗ Test embeddings not found"
    test_size=0
fi

echo ""

if [[ "$train_size" == "$test_size" ]]; then
    echo "  ❌ CRITICAL ERROR: Train and test sets have the same size!"
    echo "     This indicates you're training and testing on the same data."
    echo "     This WILL cause data leakage and unrealistic results."
    echo ""
    echo "  Please run: python generate_train_embeddings.py"
    exit 1
fi

echo "  ✅ Data splits look correct (train=$train_size, test=$test_size)"
echo ""

# ============================================
# Step 1: Generate TRAINING embeddings (if needed)
# ============================================
echo "[1/4] Checking training embeddings..."

if [[ -f "$TRAIN_EMB_DIR/emb_text.pt" ]] && [[ -f "$TRAIN_EMB_DIR/emb_video.pt" ]]; then
    echo "  ✓ Found existing training embeddings"
else
    echo "  → Generating TRAINING set embeddings (this takes ~1 hour)..."
    python generate_train_embeddings.py
    echo "  ✓ Training embeddings generated"
fi

# ============================================
# Step 2: Generate TEST embeddings (if needed)
# ============================================
echo ""
echo "[2/4] Checking test embeddings..."

if [[ -f "$TEST_EMB_DIR/emb_text.pt" ]] && [[ -f "$TEST_EMB_DIR/emb_video.pt" ]]; then
    echo "  ✓ Found existing test embeddings (MSR-VTT 1kA)"
else
    echo "  → Generating TEST set embeddings..."
    python eval_msrvtt_1kA.py
    echo "  ✓ Test embeddings generated"
fi

# ============================================
# Step 3: Train PCME on TRAINING set
# ============================================
echo ""
echo "[3/4] Training PCME projectors on TRAINING set..."

if [[ -f "$CKPT_DIR/best_projectors.pth" ]]; then
    echo "  ✓ Found existing checkpoint"
    echo "  → Checking if it was trained on correct data..."

    # Verify checkpoint was trained on training set
    ckpt_samples=$(python -c "
import torch
ckpt = torch.load('$CKPT_DIR/best_projectors.pth', map_location='cpu')
if 'config' in ckpt and 'n_samples' in ckpt['config']:
    print(ckpt['config']['n_samples'])
else:
    print('unknown')
" 2>/dev/null || echo "unknown")

    if [[ "$ckpt_samples" == "1000" ]]; then
        echo "  ⚠️  Checkpoint was trained on 1000 samples (TEST SET!)"
        echo "  → Retraining on correct TRAINING set..."
        rm -f "$CKPT_DIR/best_projectors.pth"
    elif [[ "$ckpt_samples" == "6513" ]]; then
        echo "  ✓ Checkpoint looks correct (trained on 6513 samples)"
    else
        echo "  ⚠️  Cannot verify checkpoint training data"
        echo "  → Retraining to be safe..."
        rm -f "$CKPT_DIR/best_projectors.pth"
    fi
fi

if [[ ! -f "$CKPT_DIR/best_projectors.pth" ]]; then
    mkdir -p "$CKPT_DIR"

    echo "  → Training PCME on TRAINING set (6513 samples)..."
    python /scratch365/jzheng7/ImageBind/train_pcme_projector.py \
      --emb_dir /scratch365/jzheng7/ImageBind/msrvtt_train_embeddings \
      --save_dir /scratch365/jzheng7/ImageBind/pcme_checkpoints_correct \
      --epochs 40 --batch_size 64 --lr 1e-5 --temperature 0.07 \
      --loss_type pcme_mc --n_samples 5 \
      --var_reg_type upper_bound --max_var 0.09 --var_reg_weight 0.05

    echo "  ✓ Training completed"
fi

# ============================================
# Step 4: Evaluate on TEST set
# ============================================
echo ""
echo "[4/4] Evaluating on TEST set (1000 samples, 10 runs)..."
echo ""

python measure_latency_memory_variance.py \
    --emb_dir /scratch365/jzheng7/ImageBind/msrvtt_results \
    --ckpt /scratch365/jzheng7/ImageBind/pcme_checkpoints_correct/best_projectors.pth \
    --runs 10 --warmup 5 --num_samples 15 --eval_sigma_scale 0.0

# ============================================
# Summary
# ============================================
echo ""
echo "========================================"
echo "  ✓ All Done!"
echo "========================================"
echo "End: $(date)"
echo ""
echo "Results saved to:"
echo "  → $TEST_EMB_DIR/variance_analysis.json"
echo ""

# Display results with sanity checks
if [[ -f "$TEST_EMB_DIR/variance_analysis.json" ]]; then
    echo "Quick Preview:"
    echo "=============="

    python -c "
import json
with open('$TEST_EMB_DIR/variance_analysis.json') as f:
    data = json.load(f)

ib_r1 = data['imagebind']['retrieval_scores']['Text2Video']['R@1']['mean']
pcme_r1 = data['pcme']['retrieval_scores']['Text2Video']['R@1']['mean']
improvement = pcme_r1 - ib_r1

print(f'ImageBind T2V R@1: {ib_r1:.2f}%')
print(f'PCME T2V R@1:      {pcme_r1:.2f}%')
print(f'Improvement:       +{improvement:.2f}%')
print()

# Sanity checks
if improvement > 20:
    print('⚠️  WARNING: Improvement > 20% is suspicious!')
    print('   Typical PCME improvements are 2-5% on MSR-VTT.')
    print('   Please verify:')
    print('   1. No data leakage (train != test)')
    print('   2. Embeddings were computed correctly')
    print('   3. Training converged properly')
elif improvement > 5:
    print('✅ Good improvement! Within expected range.')
elif improvement < 0:
    print('❌ Performance degraded. Check training logs.')
else:
    print('✓ Modest improvement, typical for PCME.')

print()
print(f'Latency overhead: {data[\"comparison\"][\"slowdown_factor\"]:.2f}x')
"

fi

echo ""
