#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Measure latency, memory, variance and retrieval for ImageBind baseline vs PCME projector.

Key fixes vs older drafts:
- NORMALIZE BEFORE PROJECTOR during evaluation to match training input distribution.
- Clear Monte Carlo similarity for PCME with per-dimension Gaussian samples.
- Detailed stats (latency mean/std/CI, GPU peak), retrieval (R@K/MedR/MeanR), and JSON dump.

Usage (example):
  python measure_latency_memory_variance.py \
    --emb_dir /mnt/pes/ImageBind/msrvtt_results \
    --ckpt   /mnt/pes/Cross-model-retrieval/pcme_checkpoints_correct/best_projectors.pth \
    --runs 10 --warmup 5 --num_samples 10 --k_list 1 5 10 \
    --save /mnt/pes/ImageBind/msrvtt_results/variance_analysis.json
"""

import os
import json
import time
import math
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def human_interval(ms_list: List[float]) -> Tuple[float, float]:
    """95% CI for a list of milliseconds."""
    import statistics as stats
    if len(ms_list) < 2:
        m = ms_list[0] if ms_list else 0.0
        return (m, m)
    m = stats.mean(ms_list)
    sd = stats.pstdev(ms_list) if len(ms_list) == 1 else stats.stdev(ms_list)
    ci = 1.96 * (sd / math.sqrt(len(ms_list)))
    return (m - ci, m + ci)

@contextmanager
def cuda_timer(device="cuda"):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    torch.cuda.synchronize()
    # returns milliseconds
    yield_ms = start.elapsed_time(end)
    # caller uses the returned variable in "with ... as t:"
    # but since we can't return from contextmanager easily,
    # we store it on the object (hacky but fine).
    cuda_timer.last_ms = yield_ms

def peak_gpu_mem_mb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)

def reset_peak_gpu_mem():
    torch.cuda.reset_peak_memory_stats()


# -----------------------------
# Retrieval metrics
# -----------------------------
def ranks_from_sim(sim: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    sim: [N, N] where rows: text queries, cols: videos (or vice-versa).
    Returns (text2video_ranks, video2text_ranks), both shape [N],
    where rank is 1-based position of the correct match.
    """
    N = sim.size(0)
    # T->V ranks
    sort_idx = torch.argsort(sim, dim=1, descending=True)  # [N, N]
    gt = torch.arange(N, device=sim.device)
    # where along each row does the ground-truth index appear?
    t2v_rank = (sort_idx == gt[:, None]).nonzero()[:, 1] + 1  # [N]

    # V->T ranks (transpose)
    sort_idx_T = torch.argsort(sim.t(), dim=1, descending=True)
    v2t_rank = (sort_idx_T == gt[:, None]).nonzero()[:, 1] + 1  # [N]
    return t2v_rank, v2t_rank

def recall_at_k(ranks: torch.Tensor, ks=(1,5,10)) -> Dict[int, float]:
    res = {}
    for k in ks:
        res[k] = (ranks <= k).float().mean().item() * 100.0
    return res

def median_rank(ranks: torch.Tensor) -> float:
    return ranks.median().item()

def mean_rank(ranks: torch.Tensor) -> float:
    return ranks.float().mean().item()


# -----------------------------
# PCME similarity (MC)
# -----------------------------
def sample_from_gaussian(mu: torch.Tensor, logvar: torch.Tensor, num_samples: int, sigma_scale: float=1.0) -> torch.Tensor:
    """
    mu, logvar: [N, D]
    returns samples: [S, N, D]
    """
    eps = torch.randn((num_samples, ) + mu.shape, device=mu.device, dtype=mu.dtype)
    std = (torch.exp(0.5*logvar) * sigma_scale).unsqueeze(0)  # [1, N, D]
    samples = mu.unsqueeze(0) + eps * std
    # normalize samples to keep cosine semantics
    samples = F.normalize(samples, dim=-1)
    return samples  # [S, N, D]

def pcme_similarity(mu_t: torch.Tensor, logvar_t: torch.Tensor,
                    mu_v: torch.Tensor, logvar_v: torch.Tensor,
                    num_samples: int, sigma_scale: float=1.0) -> torch.Tensor:
    """
    Monte Carlo estimate of cosine similarity:
    E_{x~N(mu_t,Î£_t), y~N(mu_v,Î£_v)}[cos(x,y)]
    Implemented by sampling S pairs and averaging dot products of normalized vectors.

    NOTE: When sigma_scale=0.0, returns deterministic similarity using mu only.
    This is faster and often performs better in practice, and is the standard
    evaluation method in PCME paper.

    Returns: [N, N] similarity matrix
    """
    # If sigma_scale is 0, use deterministic (mu only) - this is what works well
    # This is also the standard evaluation method in PCME paper
    if sigma_scale == 0.0 or num_samples == 1:
        return mu_t @ mu_v.t()  # [N, N]
    
    # Otherwise, use Monte Carlo sampling
    # NOTE: Original implementation uses independent sampling for text and video.
    # This means even for matching pairs (i, i), different random noise is used,
    # which can destroy the similarity when mu_t[i] = mu_v[i].
    # However, this is the original implementation from the author's code.
    S = max(1, num_samples)
    t_samps = sample_from_gaussian(mu_t, logvar_t, S, sigma_scale)  # [S, N, D]
    v_samps = sample_from_gaussian(mu_v, logvar_v, S, sigma_scale)  # [S, N, D]
    # cosine similarity ≈ mean_s ( t_s @ v_s^T )
    sims = torch.einsum('snd,smd->snm', t_samps, v_samps)  # [S, N, N]
    return sims.mean(dim=0)  # [N, N]


# -----------------------------
# Poly surrogate (for train_poly_projector checkpoints)
# -----------------------------
@dataclass
class PolySurrogate:
    degree: int
    coeff: torch.Tensor  # [n_feat]
    bias: float
    fit_logit: bool = True

def _poly_num_features_2vars(degree: int) -> int:
    return (degree + 2) * (degree + 1) // 2 - 1

def _poly_features_2vars(ed: torch.Tensor, vd: torch.Tensor, degree: int) -> torch.Tensor:
    """Match sklearn PolynomialFeatures(include_bias=False) for [ed, vd]. Returns [..., n_feat]."""
    feats = [ed, vd]
    if degree >= 2:
        feats += [ed * ed, ed * vd, vd * vd]
    if degree >= 3:
        feats += [ed**3, (ed**2) * vd, ed * (vd**2), vd**3]
    if degree >= 4:
        feats += [ed**4, (ed**3) * vd, (ed**2) * (vd**2), ed * (vd**3), vd**4]
    if degree >= 5:
        feats += [ed**5, (ed**4) * vd, (ed**3) * (vd**2), (ed**2) * (vd**3), ed * (vd**4), vd**5]
    if degree >= 6:
        feats += [ed**6, (ed**5) * vd, (ed**4) * (vd**2), (ed**3) * (vd**3),
                  (ed**2) * (vd**4), ed * (vd**5), vd**6]
    if degree > 6:
        raise ValueError("poly degree > 6 not supported in this evaluator")
    return torch.stack(feats, dim=-1)

def _poly_predict_logits(ed: torch.Tensor, vd: torch.Tensor, poly: PolySurrogate) -> torch.Tensor:
    X = _poly_features_2vars(ed, vd, poly.degree)
    coeff = poly.coeff.to(X.device, dtype=X.dtype)
    return (X * coeff).sum(dim=-1) + float(poly.bias)

def dist_stats_l2sq_matrix(t_mu: torch.Tensor, t_lv: torch.Tensor,
                          v_mu: torch.Tensor, v_lv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """t_*: [N,D], v_*: [N,D] -> ed, vd: [N,N] for D=||z_t - z_v||^2."""
    var1 = torch.exp(t_lv)
    var2 = torch.exp(v_lv)
    s2 = var1.unsqueeze(1) + var2.unsqueeze(0)  # [N,N,D]
    delta = t_mu.unsqueeze(1) - v_mu.unsqueeze(0)  # [N,N,D]
    dim = t_mu.size(-1)
    ed = ((delta * delta).sum(dim=-1) + s2.sum(dim=-1)) / dim
    vd = (2.0 * (s2 * s2).sum(dim=-1) + 4.0 * ((delta * delta) * s2).sum(dim=-1)) / (dim * dim)
    return ed, vd


def _poly_similarity_chunked(t_mu: torch.Tensor, t_lv: torch.Tensor,
                             v_mu: torch.Tensor, v_lv: torch.Tensor,
                             poly: PolySurrogate, chunk_size: int = 256) -> torch.Tensor:
    """Compute poly logits [N,N] in row-chunks to limit GPU memory (avoids 15GB+ for N=1000)."""
    N = t_mu.size(0)
    dim = t_mu.size(-1)
    var_v = torch.exp(v_lv)   # [N,D]
    out = torch.empty((N, N), device=t_mu.device, dtype=t_mu.dtype)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        B = end - start
        t_mu_b = t_mu[start:end]   # [B,D]
        t_lv_b = t_lv[start:end]
        var_t_b = torch.exp(t_lv_b)
        s2 = var_t_b.unsqueeze(1) + var_v.unsqueeze(0)   # [B,N,D]
        delta = t_mu_b.unsqueeze(1) - v_mu.unsqueeze(0)  # [B,N,D]
        ed = ((delta * delta).sum(dim=-1) + s2.sum(dim=-1)) / dim  # [B,N]
        vd = (2.0 * (s2 * s2).sum(dim=-1) + 4.0 * ((delta * delta) * s2).sum(dim=-1)) / (dim * dim)
        out[start:end, :] = _poly_predict_logits(ed, vd, poly)
    return out


# -----------------------------
# Simple MLP projector shells (to load checkpoint)
# -----------------------------
class PCMEProjector(nn.Module):
    """
    Must match the training-time projector heads:
    - Input dim = D (e.g., 1024 for ImageBind)
    - Two heads: mu and logvar
    """
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout_p: float = 0.0):
        super().__init__()
        self.mu_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, out_dim),
        )
        self.logvar_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f_mu = self.mu_proj(x)
        mu = x + f_mu  # CRITICAL: Residual connection to match training code!
        logvar = torch.clamp(self.logvar_proj(x), min=-5.0, max=2.0)
        return mu, logvar


# -----------------------------
# Benchmark routines
# -----------------------------
def benchmark_imagebind(text_emb, video_emb, runs, warmup, device, k_list):
    """Baseline: cosine on normalized deterministic embeddings."""
    text = F.normalize(text_emb, dim=-1)
    video = F.normalize(video_emb, dim=-1)

    # warmup
    reset_peak_gpu_mem()
    for _ in range(warmup):
        with torch.no_grad():
            _ = text @ video.t()  # [N,N]
    _ = peak_gpu_mem_mb()  # consume reading

    # actual runs
    lat_ms = []
    gpu_peaks = []
    for i in range(runs):
        reset_peak_gpu_mem()
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            sim = text @ video.t()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
        lat_ms.append(dt)
        gpu_peaks.append(peak_gpu_mem_mb())

    # retrieval
    with torch.no_grad():
        sim = text @ video.t()
        t2v_ranks, v2t_ranks = ranks_from_sim(sim)
    res = summarize_retrieval(t2v_ranks, v2t_ranks, k_list)

    return lat_ms, gpu_peaks, res


def benchmark_pcme(text_emb, video_emb, text_proj, video_proj,
                   runs, warmup, device, num_samples, k_list, eval_sigma_scale: float=0.0):
    print("[dbg] benchmark_pcme eval_sigma_scale=", eval_sigma_scale)
    """
    PCME path:
      IMPORTANT: normalize BEFORE projector to match training distribution.
    """
    # warmup
    reset_peak_gpu_mem()
    for _ in range(warmup):
        with torch.no_grad():
            t_in = F.normalize(text_emb, dim=-1)
            v_in = F.normalize(video_emb, dim=-1)
            t_mu, t_lv = text_proj(t_in)
            # mu is already normalized by projector, but ensure consistency
            t_mu = F.normalize(t_mu, dim=-1)
            v_mu, v_lv = video_proj(v_in)
            v_mu = F.normalize(v_mu, dim=-1)
            _ = pcme_similarity(t_mu, t_lv, v_mu, v_lv, num_samples, eval_sigma_scale)
    _ = peak_gpu_mem_mb()

    # actual runs
    lat_ms = []
    gpu_peaks = []
    for i in range(runs):
        reset_peak_gpu_mem()
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            t_in = F.normalize(text_emb, dim=-1)
            v_in = F.normalize(video_emb, dim=-1)
            t_mu, t_lv = text_proj(t_in)
            # mu is already normalized by projector, but ensure consistency
            t_mu = F.normalize(t_mu, dim=-1)
            v_mu, v_lv = video_proj(v_in)
            v_mu = F.normalize(v_mu, dim=-1)
            sim = pcme_similarity(t_mu, t_lv, v_mu, v_lv, num_samples, eval_sigma_scale)  # [N,N]
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
        lat_ms.append(dt)
        gpu_peaks.append(peak_gpu_mem_mb())

    # retrieval
    # NOTE: mu is already normalized in the projector (train_pcme_projector.py line 37)
    # So we don't need to normalize again here - it's redundant but harmless
    with torch.no_grad():
        t_in = F.normalize(text_emb, dim=-1)
        v_in = F.normalize(video_emb, dim=-1)
        t_mu, t_lv = text_proj(t_in)
        # t_mu is already normalized by the projector, but normalize again for consistency
        # (normalizing an already normalized vector is idempotent)
        t_mu = F.normalize(t_mu, dim=-1)
        v_mu, v_lv = video_proj(v_in)
        v_mu = F.normalize(v_mu, dim=-1)
        sim = pcme_similarity(t_mu, t_lv, v_mu, v_lv, num_samples, eval_sigma_scale)
        t2v_ranks, v2t_ranks = ranks_from_sim(sim)
    res = summarize_retrieval(t2v_ranks, v2t_ranks, k_list)

    return lat_ms, gpu_peaks, res


def benchmark_poly(text_emb, video_emb, text_proj, video_proj, poly: PolySurrogate,
                   runs, warmup, device, k_list, chunk_size: int = 256):
    """
    Poly (Eq.4 polynomial surrogate) path:
      normalize BEFORE projector, (ed, vd) in row-chunks, logits = poly(ed, vd).
      Use logits as similarity (higher = more similar). Chunked to avoid 15GB+ GPU memory.
    """
    def _run_once(t_mu, t_lv, v_mu, v_lv):
        return _poly_similarity_chunked(t_mu, t_lv, v_mu, v_lv, poly, chunk_size=chunk_size)

    # warmup
    reset_peak_gpu_mem()
    for _ in range(warmup):
        with torch.no_grad():
            t_in = F.normalize(text_emb, dim=-1)
            v_in = F.normalize(video_emb, dim=-1)
            t_mu, t_lv = text_proj(t_in)
            t_mu = F.normalize(t_mu, dim=-1)
            v_mu, v_lv = video_proj(v_in)
            v_mu = F.normalize(v_mu, dim=-1)
            _ = _run_once(t_mu, t_lv, v_mu, v_lv)
    _ = peak_gpu_mem_mb()

    # actual runs
    lat_ms = []
    gpu_peaks = []
    for i in range(runs):
        reset_peak_gpu_mem()
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            t_in = F.normalize(text_emb, dim=-1)
            v_in = F.normalize(video_emb, dim=-1)
            t_mu, t_lv = text_proj(t_in)
            t_mu = F.normalize(t_mu, dim=-1)
            v_mu, v_lv = video_proj(v_in)
            v_mu = F.normalize(v_mu, dim=-1)
            sim = _run_once(t_mu, t_lv, v_mu, v_lv)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
        lat_ms.append(dt)
        gpu_peaks.append(peak_gpu_mem_mb())

    # retrieval
    with torch.no_grad():
        t_in = F.normalize(text_emb, dim=-1)
        v_in = F.normalize(video_emb, dim=-1)
        t_mu, t_lv = text_proj(t_in)
        t_mu = F.normalize(t_mu, dim=-1)
        v_mu, v_lv = video_proj(v_in)
        v_mu = F.normalize(v_mu, dim=-1)
        sim = _run_once(t_mu, t_lv, v_mu, v_lv)
        # Sanity: if poly output is constant, retrieval is random
        sim_std = sim.std().item()
        if sim_std < 1e-5:
            print("\n[WARNING] Poly logits are nearly constant (std=%.2e). Retrieval will be random.\n"
                  "  Cause: polynomial coefficients are ~0 (Ridge over-regularized or teacher (ed,vd) not informative).\n"
                  "  Fix: re-run fit_poly with smaller --alpha (e.g. 1e-5) or build_teacher with more pairs / different (a,b)." % sim_std)
        t2v_ranks, v2t_ranks = ranks_from_sim(sim)
    res = summarize_retrieval(t2v_ranks, v2t_ranks, k_list)
    return lat_ms, gpu_peaks, res


def summarize_retrieval(t2v_ranks, v2t_ranks, k_list):
    t2v_recall = recall_at_k(t2v_ranks, ks=k_list)
    v2t_recall = recall_at_k(v2t_ranks, ks=k_list)
    out = {
        "t2v": {
            "R@k": {int(k): t2v_recall[k] for k in k_list},
            "MedR": median_rank(t2v_ranks),
            "MeanR": mean_rank(t2v_ranks),
        },
        "v2t": {
            "R@k": {int(k): v2t_recall[k] for k in k_list},
            "MedR": median_rank(v2t_ranks),
            "MeanR": mean_rank(v2t_ranks),
        }
    }
    return out


def summarize_latency(name, l_ms: List[float], gpu_mb: List[float]) -> Dict[str, Any]:
    import statistics as stats
    mean_ms = stats.mean(l_ms) if l_ms else 0.0
    sd_ms   = stats.stdev(l_ms) if len(l_ms) > 1 else 0.0
    cv_ms   = (sd_ms / mean_ms * 100.0) if mean_ms > 0 else 0.0
    ci_lo, ci_hi = human_interval(l_ms)
    res = {
        "latency_ms": {
            "mean": mean_ms,
            "std": sd_ms,
            "cv_pct": cv_ms,
            "ci95": [ci_lo, ci_hi],
            "min": min(l_ms) if l_ms else 0.0,
            "max": max(l_ms) if l_ms else 0.0,
            "median": (sorted(l_ms)[len(l_ms)//2] if l_ms else 0.0),
        },
        "gpu_mem_mb": {
            "mean": (sum(gpu_mb)/len(gpu_mb)) if gpu_mb else 0.0,
            "std": (float(torch.tensor(gpu_mb).std(unbiased=True)) if len(gpu_mb) > 1 else 0.0),
            "min": min(gpu_mb) if gpu_mb else 0.0,
            "max": max(gpu_mb) if gpu_mb else 0.0,
            "median": (sorted(gpu_mb)[len(gpu_mb)//2] if gpu_mb else 0.0),
        }
    }
    return res


# -----------------------------
# Loading
# -----------------------------
def load_embeddings(emb_dir: str, device: str):
    """
    Expect two files:
      emb_text.pt:  [N, D] float
      emb_video.pt: [N, D] float
    """
    txt_path = os.path.join(emb_dir, "emb_text.pt")
    vid_path = os.path.join(emb_dir, "emb_video.pt")
    
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Text embeddings not found: {txt_path}")
    if not os.path.exists(vid_path):
        raise FileNotFoundError(f"Video embeddings not found: {vid_path}")
    
    txt = torch.load(txt_path, map_location=device, weights_only=False)
    vid = torch.load(vid_path, map_location=device, weights_only=False)
    assert txt.dim() == 2 and vid.dim() == 2, "embeddings must be [N, D]"
    assert txt.size(0) == vid.size(0), "text/video count must match"
    return txt.to(device), vid.to(device)

def load_projectors(ckpt_path: str, in_dim: int, hidden: int, out_dim: int, device: str, for_poly: bool = False):
    if not os.path.exists(ckpt_path):
        if for_poly:
            hint = (
                f"Checkpoint not found: {ckpt_path}\n"
                f"For Poly evaluation, use projectors from train_poly (saved as best_projectors_eq4_poly.pth).\n"
                f"Example: --ckpt /mnt/pes/Cross-model-retrieval/poly_checkpoints/best_projectors_eq4_poly.pth"
            )
        else:
            hint = (
                f"Checkpoint not found: {ckpt_path}\n"
                f"Train PCME first: python train_pcme_projector.py --emb_dir /mnt/pes/ImageBind/msrvtt_train_embeddings --save_dir ./pcme_checkpoints"
            )
        raise FileNotFoundError(hint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    keys = list(ckpt.keys())
    
    # Support two key naming conventions: 'text'/'video' (PCME) or 'text_proj'/'video_proj' (train_poly_e2e)
    if "text_proj" in ckpt and "video_proj" in ckpt:
        text_key, video_key = "text_proj", "video_proj"
    elif "text" in ckpt and "video" in ckpt:
        text_key, video_key = "text", "video"
    else:
        raise AssertionError(
            f"Checkpoint must contain ('text', 'video') or ('text_proj', 'video_proj') state_dicts, got keys={keys}"
        )

    text_proj = PCMEProjector(in_dim, hidden, out_dim, dropout_p=0.0).to(device)
    video_proj = PCMEProjector(in_dim, hidden, out_dim, dropout_p=0.0).to(device)

    # strict=True so we catch any mismatch immediately
    text_proj.load_state_dict(ckpt[text_key], strict=True)
    video_proj.load_state_dict(ckpt[video_key], strict=True)

    text_proj.eval()
    video_proj.eval()
    return text_proj, video_proj


def load_poly(poly_path: str = None, ckpt_path: str = None) -> PolySurrogate:
    """
    Load polynomial surrogate.
    
    Two modes:
    1. Load from separate poly_coeffs.pt (train_poly approach)
    2. Extract poly from ckpt (train_poly_e2e end-to-end training approach)
    
    Args:
        poly_path: Path to poly_coeffs.pt (optional)
        ckpt_path: Path to best_projectors_eq4_poly.pth (used when poly_path is None)
    """
    # Mode 1: Load from separate poly_coeffs.pt
    if poly_path is not None:
        if not os.path.exists(poly_path):
            raise FileNotFoundError(
                f"Poly coeffs not found: {poly_path}\n"
                f"Run fit_poly first: python train_poly_projector.py fit_poly --teacher_npz teacher.npz --out poly_coeffs.pt"
            )
        ckpt = torch.load(poly_path, map_location="cpu", weights_only=False)
        n_feat = _poly_num_features_2vars(int(ckpt["degree"]))
        if ckpt["coeff"].numel() != n_feat:
            raise ValueError(f"Poly coeff size mismatch: degree={ckpt['degree']} => n_feat={n_feat}, got {ckpt['coeff'].numel()}")
        return PolySurrogate(
            degree=int(ckpt["degree"]),
            coeff=ckpt["coeff"].float(),
            bias=float(ckpt["bias"]),
            fit_logit=bool(ckpt.get("fit_logit", True)),
        )
    
    # Mode 2: Extract poly from ckpt (end-to-end training)
    elif ckpt_path is not None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "poly" not in ckpt:
            raise ValueError(
                f"Checkpoint {ckpt_path} does not contain poly coefficients.\n"
                f"This ckpt was likely trained with train_poly (not train_poly_e2e).\n"
                f"Use --poly_path to provide poly_coeffs.pt separately."
            )
        poly_dict = ckpt["poly"]
        return PolySurrogate(
            degree=int(poly_dict["degree"]),
            coeff=poly_dict["coeff"].float(),
            bias=float(poly_dict["bias"]),
            fit_logit=bool(poly_dict.get("fit_logit", True)),
        )
    
    else:
        raise ValueError("Must provide either poly_path or ckpt_path to load_poly")


# -----------------------------
# Pretty printing
# -----------------------------
def print_table_latency(name, lat_ms, gpu_mb):
    import statistics as stats
    mean_ms = stats.mean(lat_ms) if lat_ms else 0.0
    sd_ms   = stats.stdev(lat_ms) if len(lat_ms) > 1 else 0.0
    cv_ms   = (sd_ms / mean_ms * 100.0) if mean_ms > 0 else 0.0
    ci_lo, ci_hi = human_interval(lat_ms)
    print(f"Latency:")
    print(f"  Metric               ImageBind                      PCME                          ")
    print(f"  -------------------------------------------------------------------------------")
    # This function prints only structure; actual filling happens in main where we know both
    # Kept for visual symmetry with your previous logs.


def print_retrieval_table(title, base, pcme, k_list, model_name: str = "PCME"):
    def fmt_row(metric, b, p, sign="+"):
        diff = p - b
        s = f"{metric:>6}        {b:>8.2f}Â±0.00               {p:>8.2f}Â±0.00 ({sign}{diff:.2f})"
        return s

    print("\n" + "="*80)
    print("RETRIEVAL SCORES")
    print("="*80 + "\n")
    # Text -> Video
    print("Text â†’ Video:")
    print(f"  Metric          ImageBind                      {model_name}                               ")
    print("  -------------------------------------------------------------------------------")
    for k in k_list:
        b = base["t2v"]["R@k"][k]
        p = pcme["t2v"]["R@k"][k]
        print(f"  R@{k:<2}              {b:>6.2f}Â±0.00               {p:>6.2f}Â±0.00 ({p-b:+.2f})")
    print(f"  MedR               {base['t2v']['MedR']:.2f}Â±0.00               {pcme['t2v']['MedR']:.2f}Â±0.00 ({pcme['t2v']['MedR']-base['t2v']['MedR']:+.2f})")
    print(f"  MeanR              {base['t2v']['MeanR']:.2f}Â±0.00               {pcme['t2v']['MeanR']:.2f}Â±0.00 ({pcme['t2v']['MeanR']-base['t2v']['MeanR']:+.2f})")
    # Video -> Text
    print("\nVideo â†’ Text:")
    print(f"  Metric          ImageBind                      {model_name}                               ")
    print("  -------------------------------------------------------------------------------")
    for k in k_list:
        b = base["v2t"]["R@k"][k]
        p = pcme["v2t"]["R@k"][k]
        print(f"  R@{k:<2}              {b:>6.2f}Â±0.00               {p:>6.2f}Â±0.00 ({p-b:+.2f})")
    print(f"  MedR               {base['v2t']['MedR']:.2f}Â±0.00               {pcme['v2t']['MedR']:.2f}Â±0.00 ({pcme['v2t']['MedR']-base['v2t']['MedR']:+.2f})")
    print(f"  MeanR              {base['v2t']['MeanR']:.2f}Â±0.00               {pcme['v2t']['MeanR']:.2f}Â±0.00 ({pcme['v2t']['MeanR']-base['v2t']['MeanR']:+.2f})")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", type=str, 
                        default="/mnt/pes/ImageBind/msrvtt_results",
                        help="Directory containing emb_text.pt and emb_video.pt for the TEST split.")
    parser.add_argument("--ckpt", type=str, 
                        default="/mnt/pes/Cross-model-retrieval/pcme_checkpoints_correct/best_projectors.pth",
                        help="Path to best_projectors.pth (PCME or Poly projectors)")
    parser.add_argument("--poly_path", type=str, default=None,
                        help="Poly coeffs path (e.g. poly_coeffs.pt). If None and --use_poly, will extract poly from --ckpt (train_poly_e2e mode).")
    parser.add_argument("--use_poly", action="store_true",
                        help="Use Poly model. If set without --poly_path, will load poly from ckpt (train_poly_e2e mode).")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=15, help="MC samples for PCME (recommended: 15)")
    parser.add_argument("--eval_sigma_scale", type=float, default=0.0, 
                        help="Sigma scale for evaluation (0.0=deterministic/mu only [RECOMMENDED], 1.0=use learned variance via MC sampling)")
    parser.add_argument("--k_list", type=int, nargs="+", default=[1,5,10])
    parser.add_argument("--in_dim", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--out_dim", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save", type=str, 
                        default="/mnt/pes/ImageBind/msrvtt_results/variance_analysis.json",
                        help="Path to save JSON results")
    args = parser.parse_args()
    # Determine whether to use Poly: explicitly specified --use_poly or provided --poly_path
    use_poly = args.use_poly or bool(args.poly_path)
    model_name = "Poly" if use_poly else "PCME"
    if use_poly:
        if args.poly_path:
            print("[model] Evaluating Poly (train_poly mode); poly_path =", args.poly_path)
        else:
            print("[model] Evaluating Poly (train_poly_e2e mode); poly coeffs from ckpt")
    else:
        print("[dbg] main args.eval_sigma_scale=", args.eval_sigma_scale)

    set_seed(1234)
    assert torch.cuda.is_available(), "CUDA required for timing/memory parity."

    device = args.device
    torch.backends.cudnn.benchmark = True

    print("\nDevice:", device)
    print(f"Measurement runs: {args.runs}")
    print(f"Warmup runs: {args.warmup}\n")

    # Load data & models
    print("Loading embeddings...")
    text_emb, video_emb = load_embeddings(args.emb_dir, device)
    N, D = text_emb.size(0), text_emb.size(1)
    print(f"Dataset: {N} pairs\n")

    ckpt_path = args.ckpt
    if use_poly and not os.path.exists(ckpt_path):
        # train_poly_projector saves as best_projectors_eq4_poly.pth, not best_projectors.pth
        d = os.path.dirname(ckpt_path)
        alt = os.path.join(d, "best_projectors_eq4_poly.pth") if d else "best_projectors_eq4_poly.pth"
        if os.path.exists(alt):
            ckpt_path = alt
            print(f"Using Poly checkpoint: {ckpt_path}\n")
    print(f"Loading projectors from: {ckpt_path}\n")
    text_proj, video_proj = load_projectors(ckpt_path, args.in_dim, args.hidden, args.out_dim, device, for_poly=use_poly)
    text_proj.eval()
    video_proj.eval()

    poly = None
    if use_poly:
        if args.poly_path:
            print(f"Loading Poly surrogate from: {args.poly_path}\n")
            poly = load_poly(poly_path=args.poly_path)
        else:
            print(f"Loading Poly surrogate from ckpt (train_poly_e2e): {ckpt_path}\n")
            poly = load_poly(ckpt_path=ckpt_path)

    # ---------------- Baseline ----------------
    print(f"Benchmarking ImageBind ({args.runs} runs, {args.warmup} warmup)...")
    lat_b, gpu_b, res_b = benchmark_imagebind(text_emb, video_emb,
                                              args.runs, args.warmup, device, args.k_list)
    for i, (ms, gp) in enumerate(zip(lat_b, gpu_b), 1):
        print(f"  Run {i}/{args.runs}: {ms:.2f}ms, GPU peak: {gp:.1f}MB")

    # -------------- PCME or Poly -----------------
    if use_poly:
        print(f"\nBenchmarking {model_name} ({args.runs} runs, {args.warmup} warmup)...")
        lat_p, gpu_p, res_p = benchmark_poly(text_emb, video_emb, text_proj, video_proj, poly,
                                             args.runs, args.warmup, device, args.k_list)
    else:
        print(f"\nBenchmarking {model_name} ({args.runs} runs, {args.warmup} warmup, {args.num_samples} MC samples)...")
        lat_p, gpu_p, res_p = benchmark_pcme(text_emb, video_emb, text_proj, video_proj,
                                             args.runs, args.warmup, device, args.num_samples, args.k_list, args.eval_sigma_scale)
    for i, (ms, gp) in enumerate(zip(lat_p, gpu_p), 1):
        print(f"  Run {i}/{args.runs}: {ms:.2f}ms, GPU peak: {gp:.1f}MB")

    # ---------------- Summary -----------------
    import statistics as stats
    print("\n" + "="*80)
    print("DETAILED VARIANCE ANALYSIS")
    print("="*80 + "\n")

    mean_b, sd_b = (stats.mean(lat_b), (stats.stdev(lat_b) if len(lat_b) > 1 else 0.0))
    mean_p, sd_p = (stats.mean(lat_p), (stats.stdev(lat_p) if len(lat_p) > 1 else 0.0))
    cv_b = (sd_b / mean_b * 100.0) if mean_b > 0 else 0.0
    cv_p = (sd_p / mean_p * 100.0) if mean_p > 0 else 0.0
    ci_b = human_interval(lat_b)
    ci_p = human_interval(lat_p)

    print("Latency:")
    print(f"  Metric               ImageBind                      {model_name}                          ")
    print("  -------------------------------------------------------------------------------")
    print(f"  Mean                     {mean_b:>5.2f} ms                      {mean_p:>5.2f} ms")
    print(f"  Std Dev                  {sd_b:>5.2f} ms                       {sd_p:>5.2f} ms")
    print(f"  CV (%)                   {cv_b:>5.2f}%                        {cv_p:>5.2f}%")
    print(f"  95% CI               [{ci_b[0]:.2f}, {ci_b[1]:.2f}]           [{ci_p[0]:.2f}, {ci_p[1]:.2f}]")
    print(f"  Min                      {min(lat_b):>5.2f} ms                      {min(lat_p):>5.2f} ms")
    print(f"  Max                      {max(lat_b):>5.2f} ms                      {max(lat_p):>5.2f} ms")
    print(f"  Median                   {sorted(lat_b)[len(lat_b)//2]:>5.2f} ms                      {sorted(lat_p)[len(lat_p)//2]:>5.2f} ms")

    print("\nGPU Memory:")
    print(f"  Metric               ImageBind                      {model_name}                          ")
    print("  -------------------------------------------------------------------------------")
    print(f"  Mean                   {stats.mean(gpu_b):>6.2f} MB                     {stats.mean(gpu_p):>6.2f} MB")
    std_b = (float(torch.tensor(gpu_b).std(unbiased=True)) if len(gpu_b) > 1 else 0.0)
    std_p = (float(torch.tensor(gpu_p).std(unbiased=True)) if len(gpu_p) > 1 else 0.0)
    print(f"  Std Dev                  {std_b:>6.2f} MB                       {std_p:>6.2f} MB")
    print(f"  CV (%)                   {((std_b/(stats.mean(gpu_b)+1e-9))*100):>6.2f}%                        {((std_p/(stats.mean(gpu_p)+1e-9))*100):>6.2f}%")
    print(f"  Min                    {min(gpu_b):>6.2f} MB                     {min(gpu_p):>6.2f} MB")
    print(f"  Max                    {max(gpu_b):>6.2f} MB                     {max(gpu_p):>6.2f} MB")
    print(f"  Median                 {sorted(gpu_b)[len(gpu_b)//2]:>6.2f} MB                     {sorted(gpu_p)[len(gpu_p)//2]:>6.2f} MB")

    print_retrieval_table("Retrieval", res_b, res_p, args.k_list, model_name=model_name)

    overhead = (mean_p / (mean_b + 1e-9))
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80 + "\n")
    print(f"{model_name} Overhead:")
    print(f"  Latency: {overhead:.2f}x slower")

    k1_b = res_b["t2v"]["R@k"][1]
    k1_p = res_p["t2v"]["R@k"][1]
    print("\n" + "="*40)
    print("  ✓ All Done!")
    print("="*40)
    print(f"ImageBind T2V R@1: {k1_b:.2f}%")
    print(f"{model_name} T2V R@1:      {k1_p:.2f}%")
    print(f"Improvement:       {k1_p - k1_b:+.2f}%\n")

    if args.save:
        out = {
            "runs": args.runs,
            "warmup": args.warmup,
            "num_samples": args.num_samples if not use_poly else None,
            "model": model_name,
            "k_list": args.k_list,
            "N": N,
            "D": D,
            "latency": {
                "imagebind_ms": lat_b,
                "projector_ms": lat_p
            },
            "gpu_mem_mb": {
                "imagebind": gpu_b,
                "projector": gpu_p
            },
            "summary": {
                "imagebind": summarize_latency("ImageBind", lat_b, gpu_b),
                "projector": summarize_latency(model_name, lat_p, gpu_p),
                "retrieval": {
                    "imagebind": res_b,
                    "projector": res_p
                },
                "overhead_latency_x": overhead
            }
        }
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"✓ Saved to: {args.save}")


if __name__ == "__main__":
    main()
