# Polynomial Approximation for Probabilistic Cross-Modal Retrieval

**CIM-Friendly Probabilistic Retrieval via Polynomial Surrogate**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📂 Project Structure

```
Cross-modal-retrieval/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
# MSR-VTT (ImageBind 1k eval + embeddings)
├── download_msrvtt.py
├── setup_msrvtt_complete.sh
├── eval_msrvtt_1kA.py
├── generate_train_embeddings.py
├── generate_clip_embeddings.py
├── run_pcme_benchmark.sh
│
# Dataset prep + VATEX full benchmark
├── prepare_video_text_dataset.py
├── run_vatex_full_pipeline.py
├── launch_vatex_full_pipeline.sh
├── launch_msvd_multi_pipeline.sh
├── generate_imagebind_embeddings_generic.py
├── aggregate_vatex_results.py
│
# Models / training & eval
├── train_pcme_projector.py
├── train_cheb_projector.py
├── train_cheb_projector_v2.py
├── measure_latency_memory_variance.py
├── measure_cheb_v2.py
│
# Export (CIM)
├── export_cheb_for_cim.py
│
└── results_summary/
    ├── FINAL_COMPARISON_RESULTS.md
    └── sweep_summary.csv
```

---

## 🚀 Quick Start

### **1. Setup Environment**

```bash
# Clone the repository
git clone https://github.com/Iamlixinzhao/Cross-model-retrieval.git
cd Cross-model-retrieval

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**System Requirements:**
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU training)
- ~50GB disk space for dataset

---

### **2. Download Dataset**

Download the MSR-VTT dataset (videos and annotations):

```bash
python download_msrvtt.py
```

**Expected output:**
```
msrvtt/
├── videos/              # 10,000 videos
├── train_val_videodatainfo.json
├── train_val_annotation/
└── test_videodatainfo.json
```

---

### **3. Generate Embeddings**

Generate ImageBind embeddings for both training and testing:

#### **Generate Training Embeddings (6513 samples)**

```bash
python generate_train_embeddings.py \
  --data_dir ./msrvtt \
  --output_dir ./msrvtt_train_embeddings
```


#### **Generate Test Embeddings (1000 samples)**

```bash
python eval_msrvtt_1kA.py \
  --data_dir ./msrvtt \
  --output_dir ./msrvtt_test_embeddings
```


**Output structure:**
```
msrvtt_train_embeddings/
├── video_emb.npy
├── text_emb.npy
└── video_ids.json

msrvtt_test_embeddings/
├── video_emb.npy
├── text_emb.npy
└── video_ids.json
```

---

### **4. Train Probabilistic Projectors**

#### **Option A: Train Poly (Polynomial Surrogate) - Recommended for CIM**

**Step 1: Build Teacher Dataset (Monte Carlo Sampling)**

```bash
python train_poly_projector.py build_teacher \
  --emb_dir /mnt/pes/ImageBind/msrvtt_train_embeddings \
  --out teacher.npz \
  --n_pairs 300000 \
  --K 10
```

**Step 2: Fit Polynomial Coefficients**

```bash
python train_poly_projector.py fit_poly \
  --teacher teacher.npz \
  --degree 4 \
  --alpha 1e-3 \
  --out poly_deg4.pth
```

**Step 3: Train Projectors**

```bash
python train_poly_projector.py train_poly \
  --emb_dir /mnt/pes/ImageBind/msrvtt_train_embeddings \
  --poly poly_deg4.pth \
  --epochs 10 \
  --lr 5e-6 \
  --save best_poly_projector.pth \
  --batch_size 64
```

**Expected output:**
```
poly_model/
├── best_projectors_eq4_poly.pth
├── training_log.txt
└── config.json
```

#### **Option B: Train PCME (Monte Carlo Baseline)**

```bash
python train_pcme_projector.py \
  --emb_dir ./msrvtt_train_embeddings \
  --save_dir ./pcme_model \
  --loss pcme_mc \
  --K_train 5 \
  --K_test 15 \
  --epochs 40 \
  --lr 1e-5 \
  --var_reg_weight 0.001 \
  --batch_size 64
```

**Expected output:**
```
pcme_model/
├── best_projectors_pcme_mc.pth
├── training_log.txt
└── config.json
```

---

### **5. Evaluate Performance**

#### **Evaluate Poly Model**

```bash
python measure_latency_memory_variance.py \
  --emb_dir ./msrvtt_test_embeddings \
  --ckpt ./poly_model/best_projectors_eq4_poly.pth \
  --poly_path poly_coeffs_deg6.pt \
  --k_list 1 5 10 \
  --runs 5
```

#### **Evaluate PCME Model**

```bash
python measure_latency_memory_variance.py \
  --emb_dir ./msrvtt_test_embeddings \
  --ckpt ./pcme_model/best_projectors_pcme_mc.pth \
  --K_test 15 \
  --k_list 1 5 10 \
  --runs 5
```

#### **Evaluate ImageBind Baseline**

```bash
python measure_latency_memory_variance.py \
  --emb_dir ./msrvtt_test_embeddings \
  --k_list 1 5 10 \
  --runs 5
```

**Expected output:**
```json
{
  "text_to_video": {
    "R@1": 37.70,
    "R@5": 61.90,
    "R@10": 71.70,
    "MedR": 3.0
  },
  "video_to_text": {
    "R@1": 33.20,
    "R@5": 59.60,
    "R@10": 70.30,
    "MedR": 3.0
  },
  "latency_ms": 50.64,
  "memory_mb": 1234.56
}
```

---

### **6. (Optional) Hyperparameter Sweeps**

#### **Sweep Polynomial Degrees**

Test different polynomial degrees (3, 4, 5, 6) with full comparison:

```bash
python sweep_poly_degree.py \
  --emb_dir ./msrvtt_train_embeddings \
  --test_emb_dir ./msrvtt_test_embeddings \
  --output_dir ./sweep_degree_results \
  --degree_list 3 4 5 6 \
  --epochs 10 \
  --lr 5e-6 \
  --include_imagebind \
  --include_pcme
```

**Output:**
```
sweep_degree_results/
├── sweep_summary.csv
├── poly_ckpt_deg3/
├── poly_ckpt_deg4/
├── poly_ckpt_deg5/
└── poly_ckpt_deg6/
```

#### **Sweep Training Epochs**

Find optimal epochs to avoid overfitting:

```bash
python sweep_epochs.py \
  --poly_path poly_coeffs_deg6.pt \
  --emb_dir ./msrvtt_train_embeddings \
  --test_emb_dir ./msrvtt_test_embeddings \
  --output_dir ./sweep_epochs_results \
  --epochs_list 5 10 15 20 25 30
```

#### **Sweep Hyperparameters (Learning Rate & Variance Reg)**

Explore learning rate and variance regularization weight:

```bash
python sweep_hyperparams.py \
  --poly_path poly_coeffs_deg6.pt \
  --emb_dir ./msrvtt_train_embeddings \
  --test_emb_dir ./msrvtt_test_embeddings \
  --output_dir ./sweep_hp_results \
  --lr_list 1e-6 5e-6 1e-5 5e-5 \
  --var_reg_weight_list 0.0001 0.001 0.01 \
  --epochs 10
```

**Analysis:**
- Check `sweep_summary.csv` for performance comparison
- Identify best configuration based on T2V R@1 and V2T R@1

---

## 📊 Expected Results

本仓库的性能（`R@1/5/10`、`MedR`、`MeanR`）会随以下因素变化：
- 数据集与 split（MSR-VTT / MSVD / VATEX）
- Chebyshev projector 的阶数（`--cheb_order=3/4/5/6`）
- loss 权重（`--loss_mode`、`--v2t_weight`、`--t2v_weight`）
- 以及（PCME）Monte Carlo 采样参数

建议以你运行得到的结果为准，查看：
- `${ROOT_DIR}/summary/`（VATEX/MSVD pipeline 汇总）
- `results_summary/FINAL_COMPARISON_RESULTS.md`（仓库内汇总示例）

---

## 🎯 Recommended Configuration

如果你想用于 **CIM 部署/部署端加速**，建议直接使用仓库里的 **Chebyshev (Cheb) projector**（而不是旧的 Poly）。

- 训练：`train_cheb_projector_v2.py`（用 `--cheb_order 3/4/5/6`）
- 完整 benchmark：`launch_vatex_full_pipeline.sh` / `run_vatex_full_pipeline.py`
- 选择最优：查看 `${ROOT_DIR}/summary/` 里的汇总表（不同数据/损失权重下最佳超参可能会变化）

---

## VATEX Full Pipeline

推荐直接在服务器上后台运行，脚本会分 stage 执行并写入：
- marker：`${ROOT_DIR}/state/*.done`
- 日志：`${ROOT_DIR}/logs/pipeline.log`
- 汇总表：`${ROOT_DIR}/summary/`

```bash
# 推荐：使用 /data2 作为根目录
bash launch_vatex_full_pipeline.sh /data2/vatex_experiments

# 查看进度
tail -f /data2/vatex_experiments/logs/pipeline.log

# 运行完成后查看汇总
ls -1 /data2/vatex_experiments/summary
```

你也可以在 `launch_vatex_full_pipeline.sh` 里改 `PCME_EPOCHS` / `CHEB_EPOCHS`。

---

## 🔧 Troubleshooting

### **Out of Memory (OOM)**

If you encounter OOM during training:

```bash
# Reduce batch size
--batch_size 32  # or 16

# Reduce number of MC samples
--K_train 3  # for PCME
--K 5  # for Poly teacher building
```

### **Slow Training**

If training is too slow:

```bash
# Use fewer teacher samples
--n_pairs 100000  # instead of 300000

# Use fewer epochs for quick testing
--epochs 5
```

### **Poor Performance**

If performance is lower than expected:

1. **Check embeddings**: Ensure embeddings are generated correctly
2. **Verify learning rate**: Use `5e-6` for Poly, not `1e-5`
3. **Check epochs**: Use 10 epochs for Poly, not 20-30
4. **Run sweep**: Use hyperparameter sweeps to find optimal config

---

## 📚 Additional Documentation

- **[MATHEMATICAL_DERIVATION.md](MATHEMATICAL_DERIVATION.md)** - Detailed mathematical derivation
- **[results_summary/FINAL_COMPARISON_RESULTS.md](results_summary/FINAL_COMPARISON_RESULTS.md)** - Full comparison results

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

