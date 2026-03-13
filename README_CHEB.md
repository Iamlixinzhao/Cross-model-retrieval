# Chebyshev Projector: Training and Evaluation

This guide explains how to run the **Chebyshev (Cheb) projector** pipeline: training with `train_cheb_projector.py` and benchmarking with `measure_latency_memory_variance_cheb.py`. It includes the prerequisites (getting ImageBind embeddings from MSR-VTT) and step-by-step commands.

---

## Overview

- **`train_cheb_projector.py`** – Trains a gated Chebyshev projector on top of ImageBind embeddings (projector + T₁, T₃, optionally T₅). Uses the **training split** (e.g. 6513 text/video pairs).
- **`measure_latency_memory_variance_cheb.py`** – Evaluates retrieval (R@1, R@5, R@10, MedR), latency, and memory using a trained Cheb checkpoint on the **test split** (e.g. 1000 pairs). Can also run ImageBind baseline and PCME if checkpoints are provided.

**Data split convention:**

| Split   | Purpose              | Typical size | Directory (example)                |
|---------|----------------------|--------------|-----------------------------------|
| **Train** | Train Cheb projector | 6513         | `msrvtt_train_embeddings`         |
| **Test**  | Evaluate retrieval   | 1000         | `msrvtt_results` (or test folder) |

Both scripts expect **ImageBind-style embeddings**: `emb_text.pt` and `emb_video.pt` (PyTorch tensors, shape `[N, 1024]`, float32).

---

## Prerequisites

- **Python 3.9+**, **PyTorch 2.0+**, **CUDA** (for GPU training/eval).
- **ImageBind** repository and model (for generating embeddings).
- **MSR-VTT**: videos + annotations (see below).

---

## Step 1: MSR-VTT dataset

You need:

1. **Videos** – e.g. under `msrvtt_videos/` (or similar), one `.mp4` per video ID (`video0.mp4`, …).
2. **Annotations** – e.g. under `msrvtt_annotation/`:
   - `MSRVTT_data.json` (captions per video),
   - `MSRVTT_train.9k.csv` (optional, for split definition),
   - For test: e.g. `MSRVTT_JSFUSION_test.csv` (for 1k test set).

**Option A – Use the full setup script (if available on your cluster):**

```bash
# Adjust paths inside the script for your environment, then run
./setup_msrvtt_complete.sh
```

This typically downloads annotations, downloads/extracts videos, then runs embedding generation (Steps 2a and 2b below). If you already have videos and annotations, you can do Steps 2a and 2b manually.

**Option B – Manual download**

- Annotations: e.g. from [CLIP4Clip releases](https://github.com/ArrowLuo/CLIP4Clip/releases) (`msrvtt_data.zip`).
- Videos: e.g. from [Frozen-in-Time](https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip).

Place them so that your **embedding scripts** (and ImageBind) can find:
- Video files: e.g. `msrvtt_videos/video0.mp4`, …
- Annotation files: e.g. `msrvtt_annotation/MSRVTT_data.json`, etc.

---

## Step 2: Generate ImageBind embeddings

Embeddings must be **1024‑dim** ImageBind vectors, saved as `emb_text.pt` and `emb_video.pt` in the directories used for training and for evaluation.

### 2a. Training embeddings (for `train_cheb_projector.py`)

Use the **training split** (e.g. 6513 videos). From the **ImageBind** repo (or the project that contains the encoding code):

```bash
# From Cross-model-retrieval or the directory that has generate_train_embeddings.py
# Ensure PYTHONPATH or sys.path includes ImageBind and that paths inside the script match your layout
python generate_train_embeddings.py \
  --output_dir /path/to/msrvtt_train_embeddings
```

**Required layout for `generate_train_embeddings.py` (typical):**

- ImageBind: e.g. `/mnt/pes/ImageBind` (or set `sys.path` in the script).
- Annotations: e.g. `msrvtt_annotation/MSRVTT_data.json`, `MSRVTT_train.9k.csv`.
- Videos: e.g. `msrvtt_videos/video{i}.mp4`.

**Output:**

- `msrvtt_train_embeddings/emb_text.pt` – `[6513, 1024]`
- `msrvtt_train_embeddings/emb_video.pt` – `[6513, 1024]`
- `msrvtt_train_embeddings/metadata.json` (optional)

If your script uses different paths, edit the paths at the top of `generate_train_embeddings.py` or pass the correct `--output_dir`.

### 2b. Test embeddings (for `measure_latency_memory_variance_cheb.py`)

Use the **test split** (e.g. 1kA: 1000 videos). Often this is done by a script like `eval_msrvtt_1kA.py` in the ImageBind repo:

```bash
# From ImageBind repo (or wherever eval_msrvtt_1kA.py lives)
python eval_msrvtt_1kA.py
# Or with explicit output directory if supported:
# python eval_msrvtt_1kA.py --output_dir /path/to/msrvtt_results
```

**Output (convention):**

- `msrvtt_results/emb_text.pt` – `[1000, 1024]`
- `msrvtt_results/emb_video.pt` – `[1000, 1024]`

If you use a different test folder (e.g. `msrvtt_test_embeddings`), use that path as `--emb_dir` in the measure script.

**Summary:**

| Output directory            | Contents          | Used by                    |
|----------------------------|-------------------|----------------------------|
| `msrvtt_train_embeddings/` | Train embeddings  | `train_cheb_projector.py`  |
| `msrvtt_results/`          | Test embeddings   | `measure_latency_memory_variance_cheb.py` |

---

## Step 3: Train the Chebyshev projector

```bash
cd /path/to/Cross-model-retrieval

python train_cheb_projector.py \
  --emb_dir /path/to/msrvtt_train_embeddings \
  --save_dir ./ckpt_cheb_gated \
  --epochs 20 \
  --batch_size 256 \
  --lr 1e-4 \
  --temperature 0.07 \
  --loss_mode asymmetric \
  --v2t_weight 2.5 \
  --t2v_weight 1.0 \
  --distill_weight 0.0 \
  --no_t2 \
  --infer_vid_ids \
  --caps_per_video 20 \
  --save_name best_cheb_gated_asym.pth
```

**Important flags:**

- `--emb_dir` – Directory with `emb_text.pt` and `emb_video.pt` (training set).
- `--save_dir` – Where to write the checkpoint.
- `--no_t2` – Use only T₁ and T₃ (odd-only); add `--include_t5` for T₁+T₃+T₅.
- `--loss_mode asymmetric` – Asymmetric retrieval loss (often better V2T).
- `--infer_vid_ids` – Infer video IDs from order (needed when IDs are not stored in the .pt files).
- `--save_name` – Checkpoint filename (e.g. `best_cheb_gated_asym.pth`).

**Output:**

- `save_dir/best_cheb_gated_asym.pth` (or the name you set) – use this as `--cheb_ckpt` in the measure script.

---

## Step 4: Run measurement / retrieval benchmark

```bash
python measure_latency_memory_variance_cheb.py \
  --emb_dir /path/to/msrvtt_results \
  --cheb_ckpt ./ckpt_cheb_gated/best_cheb_gated_asym.pth \
  --cheb_odd_only \
  --k_list 1 5 10 \
  --runs 10 \
  --save ./results/cheb_metrics.json
```

**Important flags:**

- `--emb_dir` – Directory with **test** `emb_text.pt` and `emb_video.pt` (e.g. 1000 samples).
- `--cheb_ckpt` – Path to the trained Cheb checkpoint from Step 3.
- `--cheb_odd_only` – Use only odd-order Chebyshev (match training with `--no_t2` or `--include_t5`).
- `--k_list` – Recall@k values to report.
- `--save` – Output JSON path.

**Optional:**

- Omit `--ckpt` or leave PCME checkpoint missing to skip PCME and only run ImageBind + Cheb.
- Use `--cheb_no_projector` to evaluate Cheb expansion only (no projector); then the checkpoint is not used for a projector.

**Output:**

- JSON with retrieval metrics (R@1, R@5, R@10, MedR, MeanR), latency, and memory for ImageBind, Cheb, and (if provided) PCME.

---

## One-shot script (T1+T3+T5)

To train once with T₁+T₃+T₅ and then run the measure script:

```bash
./run_t1_t3_t5.sh
```

Edit the script to set `EMB_DIR`, `EVAL_EMB_DIR`, and paths if needed. It will:

1. Train with `--no_t2 --include_t5`, saving to `./sweep_runs/run_t1_t3_t5/`.
2. Run `measure_latency_memory_variance_cheb.py` with the resulting checkpoint and write metrics to `./sweep_runs/run_t1_t3_t5/metrics.json`.

---

## Hyperparameter sweep (optional)

`sweep_cheb.py` runs a grid over `loss_mode`, `v2t_weight`, and `distill_weight`, trains each config, and runs the measure script for each:

```bash
# Edit EMB_DIR, EVAL_EMB_DIR, SAVE_ROOT at the top of sweep_cheb.py if needed
python sweep_cheb.py
```

Results are collected in `sweep_runs/summary.json`.

---

## Troubleshooting

- **“Expected checkpoint not found”**  
  Ensure `--save_name` in training matches the filename you pass to `--cheb_ckpt` (e.g. `best_cheb_gated_asym.pth`).

- **“Looks like TEST split (N=1000)” in training**  
  You passed the test embedding dir to `--emb_dir`. Use the **training** embedding dir (e.g. 6513 samples) for `train_cheb_projector.py`.

- **Missing `emb_text.pt` or `emb_video.pt`**  
  Run Step 2a for training and Step 2b for test, and point `--emb_dir` to the correct folder.

- **ImageBind import errors when generating embeddings**  
  Install ImageBind and set `PYTHONPATH` (or `sys.path` in the script) so that `imagebind.models` and the eval script (e.g. `eval_msrvtt_1kA`) can be imported.

---

## File reference

| File                                  | Role                                      |
|---------------------------------------|-------------------------------------------|
| `train_cheb_projector.py`             | Train gated Chebyshev projector           |
| `measure_latency_memory_variance_cheb.py` | Benchmark retrieval + latency + memory |
| `generate_train_embeddings.py`        | Build train ImageBind embeddings          |
| `eval_msrvtt_1kA.py`                  | Build test ImageBind embeddings (1k)      |
| `run_t1_t3_t5.sh`                     | Single train+measure run (T1+T3+T5)       |
| `sweep_cheb.py`                       | Grid sweep over training hyperparameters  |
| `setup_msrvtt_complete.sh`            | Full dataset + embedding setup (if used)  |

For more context on the project and other baselines (PCME, Poly), see the main [README.md](README.md).
