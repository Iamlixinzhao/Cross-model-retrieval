# Probabilistic Cross-Modal Embeddings (PCME) for Video-Text Retrieval

This repository contains the implementation of Probabilistic Cross-Modal Embeddings (PCME) for video-text retrieval on MSR-VTT dataset, comparing against ImageBind baseline.

## Overview

This project explores **probabilistic embeddings** as an alternative to deterministic approaches for cross-modal retrieval. Instead of mapping inputs to point embeddings, PCME learns Gaussian distributions that capture uncertainty in the embedding space.

**Key Findings:**
- Video→Text R@1: **+13.6%** improvement (30.9% → 44.5%)
- Text→Video R@1: Modest decrease (-0.6%)
- Latency overhead: **22x** slower due to Monte Carlo sampling
- Demonstrates asymmetric cross-modal matching behavior

## Features

- ✅ **No Data Leakage**: Proper train/test split with diagnostic tools
- ✅ **Variance Regularization**: Multiple strategies to prevent collapse
- ✅ **Comprehensive Benchmarking**: Latency, memory, and retrieval metrics with confidence intervals
- ✅ **Automated Pipeline**: One-click setup and evaluation scripts

## Project Structure

```
.
├── README.md
├── requirements.txt
│
├── setup/
│   ├── download_msrvtt.py          # Download MSR-VTT dataset
│   ├── setup_msrvtt_complete.sh    # Complete setup pipeline (qsub)
│   └── diagnose_data_leakage.py    # Verify no train/test overlap
│
├── embeddings/
│   ├── eval_msrvtt_1kA.py          # Generate test embeddings (1000 samples)
│   └── generate_train_embeddings.py # Generate train embeddings (6513 samples)
│
├── training/
│   └── train_pcme_projector.py     # Train PCME probabilistic projectors
│
├── evaluation/
│   └── measure_latency_memory_variance.py  # Comprehensive benchmarking
│
└── pipelines/
    └── run_pcme_benchmark.sh       # End-to-end pipeline (qsub)
```

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/yourusername/pcme-video-text-retrieval.git
cd pcme-video-text-retrieval

# Create conda environment
conda create -n imagebind python=3.10
conda activate imagebind

# Install dependencies
pip install -r requirements.txt
conda install -c conda-forge ffmpeg

# Clone ImageBind (required for feature extraction)
git clone https://github.com/facebookresearch/ImageBind.git
export PYTHONPATH="${PWD}/ImageBind:${PYTHONPATH}"
```

### 2. Dataset Setup

**Option A: Interactive Download (Recommended for first-time users)**

```bash
python setup/download_msrvtt.py
```

This script will:
- Download MSR-VTT annotations (~3MB)
- Download MSR-VTT videos (~40GB)
- Organize files into correct directory structure
- Verify dataset integrity

**Option B: Automated Pipeline (for cluster environments)**

```bash
# Modify paths in setup_msrvtt_complete.sh to match your environment
qsub setup/setup_msrvtt_complete.sh
```

### 3. Generate Embeddings

```bash
# Generate training set embeddings (6513 samples)
python embeddings/generate_train_embeddings.py

# Generate test set embeddings (1000 samples - MSR-VTT 1kA)
python embeddings/eval_msrvtt_1kA.py
```

**Expected output locations:**
- Training: `./msrvtt_train_embeddings/`
- Test: `./msrvtt_results/`

### 4. Verify No Data Leakage

```bash
python setup/diagnose_data_leakage.py
```

**Expected output:**
```
✅ No data leakage issues detected!

Your setup looks correct:
  - Training set: 6513 samples
  - Test set: 1000 samples
  - Checkpoint trained on correct data
```

### 5. Train PCME Projectors

```bash
python training/train_pcme_projector.py \
  --emb_dir ./msrvtt_train_embeddings \
  --save_dir ./pcme_checkpoints \
  --epochs 40 \
  --batch_size 64 \
  --lr 1e-5 \
  --temperature 0.07 \
  --loss_type pcme_mc \
  --n_samples 5 \
  --var_reg_type upper_bound \
  --max_var 0.09 \
  --var_reg_weight 0.05
```

**Training configuration options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--loss_type` | `pcme_mc` | Loss function: `pcme_mc` (Monte Carlo) or `deterministic` |
| `--n_samples` | 5 | Monte Carlo samples during training |
| `--var_reg_type` | `upper_bound` | Variance regularization: `kl`, `lower_bound`, `upper_bound`, `target` |
| `--var_reg_weight` | 0.001 | Weight for variance regularization |
| `--max_var` | 0.09 | Maximum variance threshold |

### 6. Evaluate on Test Set

```bash
python evaluation/measure_latency_memory_variance.py \
  --emb_dir ./msrvtt_results \
  --ckpt ./pcme_checkpoints/best_projectors.pth \
  --runs 10 \
  --warmup 5 \
  --num_samples 15
```

**This will measure:**
- Retrieval scores (R@1, R@5, R@10, MedR, MeanR)
- Latency (mean, std, CV%, 95% CI)
- GPU memory usage
- Variance statistics

### 7. Complete Pipeline (Recommended)

For cluster environments, use the automated pipeline:

```bash
# Modify paths in run_pcme_benchmark.sh
qsub pipelines/run_pcme_benchmark.sh
```

This runs steps 3-6 automatically with proper train/test separation checks.

## Results

### Retrieval Performance (MSR-VTT 1kA)

| Metric | ImageBind | PCME | Δ |
|--------|-----------|------|---|
| **Text→Video R@1** | 38.8% | 38.2% | -0.6% |
| **Text→Video R@5** | 63.7% | 65.0% | +1.3% |
| **Video→Text R@1** | 30.9% | **44.5%** | **+13.6%** |
| **Video→Text R@5** | 54.3% | 71.5% | +17.2% |

### Computational Overhead

| Metric | ImageBind | PCME | Overhead |
|--------|-----------|------|----------|
| **Latency** | 0.46 ms | 10.28 ms | **22.2x** |
| **GPU Memory** | 199.5 MB | 367.6 MB | **1.84x** |

### Key Insights

1. **Asymmetric Performance**: PCME excels at Video→Text (+13.6%) but shows modest decrease in Text→Video (-0.6%)
   - **Explanation**: Videos naturally admit multiple textual descriptions (one-to-many), where uncertainty helps
   - Text queries are typically more specific (deterministic matching preferred)

2. **Variance Collapse Challenge**: Training requires careful regularization to maintain meaningful uncertainty
   - Solution: Upper-bound constraints (max_var=0.09) prevent excessive variance while avoiding collapse

3. **Compute-Accuracy Tradeoff**: 22x latency increase may be acceptable for applications where accuracy is critical

## Architecture

### Probabilistic Projector

```python
class ProbabilisticProjector(nn.Module):
    def __init__(self, dim=1024, hidden=2048):
        super().__init__()
        # Mean projection (with residual connection)
        self.mu_proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, dim)
        )
        # Variance projection (clamped to prevent instability)
        self.logvar_proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, dim)
        )
    
    def forward(self, x):
        mu = x + self.mu_proj(x)  # Residual connection
        mu = F.normalize(mu, dim=-1)
        logvar = torch.clamp(self.logvar_proj(x), -5, 2)
        return mu, logvar
```

### Monte Carlo Similarity

```python
def pcme_similarity(mu_t, logvar_t, mu_v, logvar_v, num_samples=15):
    """
    E[cos(z_t, z_v)] where z_t ~ N(μ_t, Σ_t), z_v ~ N(μ_v, Σ_v)
    """
    t_samples = sample_from_gaussian(mu_t, logvar_t, num_samples)
    v_samples = sample_from_gaussian(mu_v, logvar_v, num_samples)
    
    # Average cosine similarity across samples
    sims = torch.einsum('snd,smd->snm', t_samples, v_samples)
    return sims.mean(dim=0)
```

## Common Issues

### Issue 1: Data Leakage Warning

**Symptom:**
```
⚠️  WARNING: Improvement > 20% is suspicious!
```

**Solution:**
```bash
python setup/diagnose_data_leakage.py
```

Ensure:
- Training set: 6513 samples
- Test set: 1000 samples
- No overlap between sets

### Issue 2: Variance Collapse

**Symptom:**
```
Variance: text=0.000001, video=0.000002
```

**Solution:**

1. Increase variance regularization weight:
   ```bash
   --var_reg_weight 0.1
   ```

2. Use upper-bound regularization:
   ```bash
   --var_reg_type upper_bound --max_var 0.09
   ```

3. Add KL divergence term:
   ```bash
   --var_reg_type kl --var_reg_weight 0.05
   ```

### Issue 3: Out of Memory

**Solution:**

1. Reduce batch size:
   ```bash
   --batch_size 32
   ```

2. Reduce Monte Carlo samples:
   ```bash
   --n_samples 3
   ```

3. Use gradient checkpointing (modify code)

## Citation

If you use this code for your research, please cite:

```bibtex
@inproceedings{pcme2025,
  title={Probabilistic Cross-Modal Embeddings for Video-Text Retrieval},
  author={Your Name},
  booktitle={Design Automation Conference (DAC)},
  year={2025}
}
```

## Related Work

- **ImageBind**: [Girdhar et al., CVPR 2023](https://arxiv.org/abs/2305.05665)
- **PCME**: [Chun et al., CVPR 2021](https://arxiv.org/abs/2101.05068)
- **MSR-VTT**: [Xu et al., CVPR 2016](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ImageBind authors for the pretrained multimodal model
- MSR-VTT dataset creators
- Notre Dame CRC for computational resources

## Contact

- **Author**: Jiahao Zheng
- **Email**: jzheng7@nd.edu
- **Project Link**: https://github.com/yourusername/pcme-video-text-retrieval

---

**Note**: This is research code. For production use, consider optimizing Monte Carlo sampling and implementing quantization for deployment.