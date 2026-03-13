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
from pathlib import Path
from contextlib import contextmanager
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
# Chebyshev embedding (deterministic, CIM-friendly)
# -----------------------------
def _cheb_degrees(degree: int, odd_only: bool) -> list:
    """Orders to compute. odd_only: only T_1, T_3, T_5, ... (sign-sensitive)."""
    if odd_only:
        return [n for n in range(1, degree + 1) if n % 2 == 1]
    return list(range(1, degree + 1))


def chebyshev_features(x: torch.Tensor, degree: int, include_t0: bool = False, odd_only: bool = False) -> torch.Tensor:
    """
    Build Chebyshev features for x in [-1,1]:
        T0 = 1, T1 = x, Tn = 2 x T_{n-1} - T_{n-2}
    odd_only: use only T_1, T_3, T_5, ... (preserves sign-sensitive shape).
    """
    assert degree >= 1, "degree must be >= 1"
    x = torch.clamp(x, -1.0, 1.0)

    degrees = _cheb_degrees(degree, odd_only)
    if not degrees:
        raise ValueError("empty degrees")
    max_n = max(degrees)

    feats = []
    if include_t0:
        feats.append(torch.ones_like(x))
    T_nm2 = torch.ones_like(x)
    T_nm1 = x
    all_T = [T_nm1]
    for n in range(2, max_n + 1):
        T_n = 2.0 * x * T_nm1 - T_nm2
        all_T.append(T_n)
        T_nm2, T_nm1 = T_nm1, T_n
    for n in degrees:
        feats.append(all_T[n - 1])
    return torch.cat(feats, dim=-1)


class ChebProjector(nn.Module):
    """Lightweight projector that maps x -> x' in [-1,1] before Chebyshev expansion."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout_p: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual helps preserve semantic alignment (similar to PCME)
        y = x + self.net(x)
        # squash to [-1,1] for Chebyshev stability
        return torch.tanh(y)


# Match train_cheb_projector.py for loading best_cheb_gated.pth (projector + gated Cheb)
class TrainChebProjector(nn.Module):
    """Adapter from train_cheb_projector.py: output h in [-1,1]."""
    def __init__(self, dim, hidden=None, residual=True):
        super().__init__()
        self.residual = residual
        if hidden is None:
            hidden = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        out = self.net(x)
        if self.residual:
            out = x + out
        return torch.tanh(out)


class GatedChebyshevLayer(nn.Module):
    """From train_cheb_projector.py: Phi(h) = [g1*T1, g2*T2?, g3*T3, g5*T5?]."""
    def __init__(self, dim, gate_mode="scalar", include_t2=True, include_t5=False,
                 init_gates=(1.0, 0.1, 0.1), init_g5=0.1):
        super().__init__()
        self.dim = dim
        self.gate_mode = gate_mode
        self.include_t2 = include_t2
        self.include_t5 = include_t5
        if gate_mode == "scalar":
            self.g1 = nn.Parameter(torch.tensor(float(init_gates[0])))
            self.g2 = nn.Parameter(torch.tensor(float(init_gates[1])))
            self.g3 = nn.Parameter(torch.tensor(float(init_gates[2])))
            if include_t5:
                self.g5 = nn.Parameter(torch.tensor(float(init_g5)))
        else:
            self.g1 = nn.Parameter(torch.full((dim,), float(init_gates[0])))
            self.g2 = nn.Parameter(torch.full((dim,), float(init_gates[1])))
            self.g3 = nn.Parameter(torch.full((dim,), float(init_gates[2])))
            if include_t5:
                self.g5 = nn.Parameter(torch.full((dim,), float(init_g5)))

    def forward(self, h):
        T1 = h
        T2 = 2.0 * h * h - 1.0
        T3 = 4.0 * h * h * h - 3.0 * h
        if self.include_t5:
            T5 = 16.0 * h ** 5 - 20.0 * h * h * h + 5.0 * h
        if self.gate_mode == "scalar":
            g1, g2, g3 = self.g1, self.g2, self.g3
            g5 = self.g5 if self.include_t5 else None
        else:
            g1 = self.g1.view(1, -1)
            g2 = self.g2.view(1, -1)
            g3 = self.g3.view(1, -1)
            g5 = self.g5.view(1, -1) if self.include_t5 else None
        parts = [g1 * T1]
        if self.include_t2:
            parts.append(g2 * T2)
        parts.append(g3 * T3)
        if self.include_t5:
            parts.append(g5 * T5)
        phi = torch.cat(parts, dim=-1)
        return F.normalize(phi, dim=-1)

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


def benchmark_cheb(text_emb, video_emb, text_cheb_proj, video_cheb_proj,
                   cheb_degree: int, include_t0: bool,
                   runs, warmup, device, k_list,
                   cheb_odd_only: bool = False, cheb_gates: list = None,
                   temperature: float = 0.07, normalize_phi: bool = True):
    """
    Chebyshev path: Phi(x) = [g_1*T_1, g_3*T_3, g_5*T_5, ...], s = Phi(q)^T Phi(c).
    cheb_gates: per-order scalars (g_1 fixed 1.0). None = no gating (all 1).
    """

    def build_phi(x, proj):
        x_in = F.normalize(x, dim=-1)
        if proj is not None:
            x_in = proj(x_in)  # already tanh'd to [-1,1]
        else:
            x_in = torch.clamp(x_in, -1.0, 1.0)
        phi = chebyshev_features(x_in, degree=cheb_degree, include_t0=include_t0, odd_only=cheb_odd_only)
        if cheb_gates is not None and len(cheb_gates) > 0:
            K = len(cheb_gates)
            D = phi.shape[1] // K
            g = torch.tensor(cheb_gates, device=phi.device, dtype=phi.dtype).view(1, K, 1)
            phi = (phi.view(phi.shape[0], K, D) * g).view(phi.shape[0], -1)
        if normalize_phi:
            phi = F.normalize(phi, dim=-1)
        return phi

    # warmup
    reset_peak_gpu_mem()
    for _ in range(warmup):
        with torch.no_grad():
            t_phi = build_phi(text_emb, text_cheb_proj)
            v_phi = build_phi(video_emb, video_cheb_proj)
            _ = (t_phi @ v_phi.t()) / temperature
    _ = peak_gpu_mem_mb()

    # actual runs
    lat_ms = []
    gpu_peaks = []
    for _ in range(runs):
        reset_peak_gpu_mem()
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            t_phi = build_phi(text_emb, text_cheb_proj)
            v_phi = build_phi(video_emb, video_cheb_proj)
            sim = (t_phi @ v_phi.t()) / temperature
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
        lat_ms.append(dt)
        gpu_peaks.append(peak_gpu_mem_mb())

    # retrieval
    with torch.no_grad():
        t_phi = build_phi(text_emb, text_cheb_proj)
        v_phi = build_phi(video_emb, video_cheb_proj)
        sim = (t_phi @ v_phi.t()) / temperature
        t2v_ranks, v2t_ranks = ranks_from_sim(sim)
    res = summarize_retrieval(t2v_ranks, v2t_ranks, k_list)

    return lat_ms, gpu_peaks, res


def benchmark_cheb_gated(text_emb, video_emb, projector, cheb_layer,
                         runs, warmup, device, k_list, temperature=0.07):
    """Full gated model: x -> normalize -> (projector or clamp) -> cheb_layer -> similarity."""
    def build_phi(x):
        x_in = F.normalize(x, dim=-1)
        if projector is not None:
            h = projector(x_in)
        else:
            h = torch.clamp(x_in, -1.0, 1.0)
        return cheb_layer(h)

    reset_peak_gpu_mem()
    for _ in range(warmup):
        with torch.no_grad():
            t_phi = build_phi(text_emb)
            v_phi = build_phi(video_emb)
            _ = t_phi @ v_phi.t() / temperature
    _ = peak_gpu_mem_mb()

    lat_ms, gpu_peaks = [], []
    for _ in range(runs):
        reset_peak_gpu_mem()
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            t_phi = build_phi(text_emb)
            v_phi = build_phi(video_emb)
            sim = (t_phi @ v_phi.t()) / temperature
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
        lat_ms.append(dt)
        gpu_peaks.append(peak_gpu_mem_mb())

    with torch.no_grad():
        t_phi = build_phi(text_emb)
        v_phi = build_phi(video_emb)
        sim = (t_phi @ v_phi.t()) / temperature
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

def load_projectors(ckpt_path: str, in_dim: int, hidden: int, out_dim: int, device: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Please train the PCME projector first using:\n"
            f"  python train_pcme_projector.py --emb_dir /mnt/pes/ImageBind/msrvtt_train_embeddings --save_dir /mnt/pes/Cross-model-retrieval/pcme_checkpoints_correct"
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    keys = list(ckpt.keys())
    assert "text" in ckpt and "video" in ckpt, \
        f"Checkpoint must contain 'text' and 'video' state_dicts, got keys={keys}"

    text_proj = PCMEProjector(in_dim, hidden, out_dim, dropout_p=0.0).to(device)
    video_proj = PCMEProjector(in_dim, hidden, out_dim, dropout_p=0.0).to(device)

    # strict=True so we catch any mismatch immediately
    text_proj.load_state_dict(ckpt["text"], strict=True)
    video_proj.load_state_dict(ckpt["video"], strict=True)

    text_proj.eval()
    video_proj.eval()
    return text_proj, video_proj



def load_gates_from_gated_ckpt(ckpt_path: str):
    """
    Load gate values from best_cheb_gated.pth (train_cheb_projector.py with GatedChebyshevLayer).
    Returns (gates_list, no_t2) e.g. ([g1, g3], True) for no_t2, or (None, None) if not this format.
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None, None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "cheb_layer" not in ckpt:
        return None, None
    sd = ckpt["cheb_layer"]
    # scalar: g1, g2, g3 are 0-dim tensors
    g1 = float(sd["g1"].cpu()) if "g1" in sd else 1.0
    g2 = float(sd["g2"].cpu()) if "g2" in sd else 0.1
    g3 = float(sd["g3"].cpu()) if "g3" in sd else 0.1
    config = ckpt.get("config") or {}
    no_t2 = config.get("no_t2", True)
    if no_t2:
        return [g1, g3], True
    return [g1, g2, g3], False


def load_gated_full_ckpt(ckpt_path: str, dim: int, device: str):
    """
    Load full gated model (projector + GatedChebyshevLayer) from best_cheb_gated.pth.
    Returns (projector, cheb_layer). projector is None if ckpt was trained with --no_projector.
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None, None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "cheb_layer" not in ckpt:
        return None, None
    config = ckpt.get("config") or {}
    no_t2 = config.get("no_t2", True)
    include_t5 = config.get("include_t5", False)
    gate_mode = config.get("gate_mode", "scalar")
    init_g1 = config.get("init_g1", 1.0)
    init_g2 = config.get("init_g2", 0.1)
    init_g3 = config.get("init_g3", 0.1)
    init_g5 = config.get("init_g5", 0.1)
    # include_t5 => T1+T3+T5 (no T2); else no_t2 => T1+T3 only
    cheb_layer = GatedChebyshevLayer(
        dim, gate_mode=gate_mode,
        include_t2=not no_t2 and not include_t5,
        include_t5=include_t5,
        init_gates=(init_g1, init_g2, init_g3),
        init_g5=init_g5,
    ).to(device)
    cheb_layer.load_state_dict(ckpt["cheb_layer"], strict=True)
    cheb_layer.eval()

    projector = None
    if "projector" in ckpt:
        hidden = config.get("hidden_dim")
        residual = not config.get("no_residual", False)
        projector = TrainChebProjector(dim=dim, hidden=hidden, residual=residual).to(device)
        projector.load_state_dict(ckpt["projector"], strict=True)
        projector.eval()
    return projector, cheb_layer


def load_cheb_projectors(ckpt_path: str, in_dim: int, hidden: int, out_dim: int, device: str):
    """Load Chebyshev projectors saved by train_cheb_projector.py (keys: 'text','video')."""
    if ckpt_path is None:
        return None, None
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Cheb checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "cheb_layer" in ckpt:
        raise ValueError("This is a gated checkpoint (best_cheb_gated.pth). Use --cheb_no_projector and pass --cheb_ckpt to load gates only.")
    if "text" not in ckpt or "video" not in ckpt:
        raise ValueError(f"Cheb checkpoint must have 'text' and 'video' state_dicts, got keys={list(ckpt.keys())}")
    text_proj = ChebProjector(in_dim, hidden, out_dim, dropout_p=0.0).to(device)
    video_proj = ChebProjector(in_dim, hidden, out_dim, dropout_p=0.0).to(device)
    text_proj.load_state_dict(ckpt["text"], strict=True)
    video_proj.load_state_dict(ckpt["video"], strict=True)
    text_proj.eval()
    video_proj.eval()
    return text_proj, video_proj

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


def print_retrieval_table(title, base, pcme, cheb, k_list):
    print("\n" + "="*80)
    print("RETRIEVAL SCORES")
    print("="*80 + "\n")

    def print_block(header, key):
        print(header + ":")
        print("  Metric          ImageBind                      PCME                          Cheb")
        print("  -------------------------------------------------------------------------------")
        for k in k_list:
            b = base[key]["R@k"][k]
            c = cheb[key]["R@k"][k]
            dc = c - b
            if pcme is not None:
                p = pcme[key]["R@k"][k]
                dp = p - b
                print(f"  R@{k:<2}              {b:>6.2f}±0.00               {p:>6.2f}±0.00 ({dp:+.2f})   {c:>6.2f}±0.00 ({dc:+.2f})")
            else:
                print(f"  R@{k:<2}              {b:>6.2f}±0.00                 n/a                 {c:>6.2f}±0.00 ({dc:+.2f})")

        b_med, b_mean = base[key]["MedR"], base[key]["MeanR"]
        c_med, c_mean = cheb[key]["MedR"], cheb[key]["MeanR"]
        if pcme is not None:
            p_med, p_mean = pcme[key]["MedR"], pcme[key]["MeanR"]
            print(f"  MedR               {b_med:>6.2f}±0.00               {p_med:>6.2f}±0.00 ({p_med-b_med:+.2f})   {c_med:>6.2f}±0.00 ({c_med-b_med:+.2f})")
            print(f"  MeanR              {b_mean:>6.2f}±0.00               {p_mean:>6.2f}±0.00 ({p_mean-b_mean:+.2f})   {c_mean:>6.2f}±0.00 ({c_mean-b_mean:+.2f})")
        else:
            print(f"  MedR               {b_med:>6.2f}±0.00                  n/a                 {c_med:>6.2f}±0.00 ({c_med-b_med:+.2f})")
            print(f"  MeanR              {b_mean:>6.2f}±0.00                  n/a                 {c_mean:>6.2f}±0.00 ({c_mean-b_mean:+.2f})")
        print()

    print_block("Text → Video", "t2v")
    print_block("Video → Text", "v2t")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", type=str, 
                        default="/mnt/pes/ImageBind/msrvtt_results",
                        help="Directory containing emb_text.pt and emb_video.pt for the TEST split.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to PCME best_projectors.pth. If omitted or file missing, PCME benchmark is skipped.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=15, help="MC samples for PCME (recommended: 15)")
    parser.add_argument("--eval_sigma_scale", type=float, default=0.0, 
                        help="Sigma scale for evaluation (0.0=deterministic/mu only [RECOMMENDED], 1.0=use learned variance via MC sampling)")
    
    # Chebyshev projector (deterministic polynomial embedding)
    _script_dir = Path(__file__).resolve().parent
    _default_cheb_ckpt = str(_script_dir / "ckpt_cheb_gated" / "best_cheb_gated_asym.pth")
    parser.add_argument("--cheb_ckpt", type=str, default=_default_cheb_ckpt,
                        help="Path to Chebyshev gated checkpoint (e.g. best_cheb_gated_asym.pth). Use --cheb_no_projector to skip loading.")
    parser.add_argument("--cheb_no_projector", action="store_true",
                        help="Do not use a Chebyshev projector; directly clip normalized embeddings to [-1,1] before Chebyshev expansion.")
    parser.add_argument("--cheb_degree", type=int, default=3,
                        help="Max Cheb degree. With --cheb_odd_only use only T_1, T_3, ... up to this.")
    parser.add_argument("--cheb_odd_only", action="store_true",
                        help="Use only odd-order Chebyshev (T_1, T_3, T_5, ...). Match training setting.")
    parser.add_argument("--cheb_include_t0", action="store_true",
                        help="Include T0(x)=1 in Chebyshev features (adds D dims). Usually unnecessary.")
    parser.add_argument("--cheb_temperature", type=float, default=0.07,
                        help="Temperature used for Chebyshev similarity (for consistency with training).")
    parser.add_argument("--cheb_g3", type=float, default=1.0,
                        help="Gate for T_3 when odd_only (g_1=1 fixed). Sweep e.g. 0.0~0.5 to damp higher order.")
    parser.add_argument("--cheb_g5", type=float, default=1.0,
                        help="Gate for T_5 when degree>=5 odd_only. Try 0.0~0.2.")
    parser.add_argument("--k_list", type=int, nargs="+", default=[1,5,10])
    parser.add_argument("--in_dim", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--out_dim", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save", type=str, 
                        default="/mnt/pes/ImageBind/msrvtt_results/variance_analysis.json",
                        help="Path to save JSON results")
    args = parser.parse_args()
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

    run_pcme = args.ckpt and os.path.exists(args.ckpt)
    if run_pcme:
        print(f"Loading PCME checkpoint: {args.ckpt}\n")
        text_proj, video_proj = load_projectors(args.ckpt, args.in_dim, args.hidden, args.out_dim, device)
        text_proj.eval()
        video_proj.eval()
    else:
        text_proj, video_proj = None, None
        if args.ckpt:
            print(f"PCME checkpoint not found: {args.ckpt} — skipping PCME benchmark.\n")
        else:
            print("No PCME checkpoint (--ckpt) provided — skipping PCME benchmark.\n")

    # ---------------- Baseline ----------------
    print(f"Benchmarking ImageBind ({args.runs} runs, {args.warmup} warmup)...")
    lat_b, gpu_b, res_b = benchmark_imagebind(text_emb, video_emb,
                                              args.runs, args.warmup, device, args.k_list)
    for i, (ms, gp) in enumerate(zip(lat_b, gpu_b), 1):
        print(f"  Run {i}/{args.runs}: {ms:.2f}ms, GPU peak: {gp:.1f}MB")

    # -------------- PCME (MC) -----------------
    lat_p, gpu_p, res_p = None, None, None
    if run_pcme:
        print(f"\nBenchmarking PCME ({args.runs} runs, {args.warmup} warmup, {args.num_samples} MC samples)...")
        lat_p, gpu_p, res_p = benchmark_pcme(text_emb, video_emb, text_proj, video_proj,
                                             args.runs, args.warmup, device, args.num_samples, args.k_list, args.eval_sigma_scale)
        for i, (ms, gp) in enumerate(zip(lat_p, gpu_p), 1):
            print(f"  Run {i}/{args.runs}: {ms:.2f}ms, GPU peak: {gp:.1f}MB")

    
    # -------------- Chebyshev: full gated model (projector + GatedCheb) or gates-only / old format -----
    cheb_ckpt_path = args.cheb_ckpt
    if cheb_ckpt_path and not os.path.exists(cheb_ckpt_path) and not os.path.isabs(cheb_ckpt_path):
        alt = Path(__file__).resolve().parent / cheb_ckpt_path
        if alt.exists():
            cheb_ckpt_path = str(alt)
    gated_proj, gated_cheb = load_gated_full_ckpt(cheb_ckpt_path, D, device)
    if gated_cheb is not None:
        print(f"\nLoaded gated checkpoint: {cheb_ckpt_path}")
        print(f"  Projector: {'ON' if gated_proj is not None else 'OFF (gates only)'}")
        print(f"  Benchmarking Chebyshev (full gated model)...")
        lat_c, gpu_c, res_c = benchmark_cheb_gated(
            text_emb, video_emb,
            gated_proj, gated_cheb,
            runs=args.runs, warmup=args.warmup,
            device=device, k_list=args.k_list,
            temperature=args.cheb_temperature,
        )
    else:
        gates_from_ckpt, no_t2_from_ckpt = load_gates_from_gated_ckpt(cheb_ckpt_path)
        if gates_from_ckpt is not None:
            cheb_degrees = [1, 3] if no_t2_from_ckpt else [1, 2, 3]
            cheb_gates = gates_from_ckpt
            cheb_odd_only = no_t2_from_ckpt
            cheb_degree_eff = 3
            print(f"\nLoaded gates from {cheb_ckpt_path}: {[round(g, 4) for g in cheb_gates]}, no_t2={no_t2_from_ckpt}")
        else:
            cheb_degree_eff = args.cheb_degree
            cheb_odd_only = args.cheb_odd_only
            cheb_degrees = _cheb_degrees(args.cheb_degree, args.cheb_odd_only)
            cheb_gates = [1.0]
            if len(cheb_degrees) >= 2:
                cheb_gates.append(args.cheb_g3)
            if len(cheb_degrees) >= 3:
                cheb_gates.append(args.cheb_g5)
        if any(g != 1.0 for g in cheb_gates):
            print(f"\nBenchmarking Chebyshev (degree={cheb_degree_eff}, odd_only={cheb_odd_only}, gates={[round(g,4) for g in cheb_gates]}, "
                  f"projector={'OFF' if args.cheb_no_projector else 'ON'})...")
        else:
            print(f"\nBenchmarking Chebyshev (degree={cheb_degree_eff}, odd_only={cheb_odd_only}, "
                  f"projector={'OFF' if args.cheb_no_projector else ('ON' if args.cheb_ckpt else 'OFF')} )...")

        if args.cheb_no_projector or gates_from_ckpt is not None:
            cheb_text_proj, cheb_video_proj = None, None
            print("  Using NO projector (clip normalized embeddings to [-1,1]).")
        else:
            if not cheb_ckpt_path:
                print("  [warn] --cheb_ckpt not provided, falling back to --cheb_no_projector behavior.")
                cheb_text_proj, cheb_video_proj = None, None
            elif not os.path.exists(cheb_ckpt_path):
                print(f"  [warn] Cheb checkpoint not found: {cheb_ckpt_path}")
                cheb_text_proj, cheb_video_proj = None, None
            else:
                print(f"  Loading Chebyshev checkpoint: {cheb_ckpt_path}")
                cheb_text_proj, cheb_video_proj = load_cheb_projectors(
                    cheb_ckpt_path, args.in_dim, args.hidden, args.out_dim, device
                )

        lat_c, gpu_c, res_c = benchmark_cheb(
            text_emb, video_emb,
            cheb_text_proj, cheb_video_proj,
            cheb_degree=cheb_degree_eff,
            include_t0=args.cheb_include_t0,
            runs=args.runs, warmup=args.warmup,
            device=device, k_list=args.k_list,
            cheb_odd_only=cheb_odd_only,
            cheb_gates=cheb_gates,
            temperature=args.cheb_temperature,
            normalize_phi=True
        )
    for i, (ms, gp) in enumerate(zip(lat_c, gpu_c), 1):
        print(f"  Run {i}/{args.runs}: {ms:.2f}ms, GPU peak: {gp:.1f}MB")

# ---------------- Summary -----------------
    import statistics as stats
    print("\n" + "="*80)
    print("DETAILED VARIANCE ANALYSIS")
    print("="*80 + "\n")

    # Latency summary table
    mean_b, sd_b = (stats.mean(lat_b), (stats.stdev(lat_b) if len(lat_b) > 1 else 0.0))
    cv_b = (sd_b / mean_b * 100.0) if mean_b > 0 else 0.0
    ci_b = human_interval(lat_b)
    mean_p = (stats.mean(lat_p) if lat_p else 0.0)
    sd_p = (stats.stdev(lat_p) if lat_p and len(lat_p) > 1 else 0.0)
    cv_p = (sd_p / mean_p * 100.0) if mean_p > 0 else 0.0
    ci_p = human_interval(lat_p) if lat_p else (0.0, 0.0)

    print("Latency:")
    print("  Metric               ImageBind                      PCME                          ")
    print("  -------------------------------------------------------------------------------")
    print(f"  Mean                     {mean_b:>5.2f} ms                      {(f'{mean_p:>5.2f} ms' if lat_p else 'n/a'):>12}")
    print(f"  Std Dev                  {sd_b:>5.2f} ms                       {(f'{sd_p:>5.2f} ms' if lat_p else 'n/a'):>12}")
    print(f"  CV (%)                   {cv_b:>5.2f}%                        {(f'{cv_p:>5.2f}%' if lat_p else 'n/a'):>12}")
    print(f"  95% CI               [{ci_b[0]:.2f}, {ci_b[1]:.2f}]           {(f'[{ci_p[0]:.2f}, {ci_p[1]:.2f}]' if lat_p else 'n/a'):>20}")
    print(f"  Min                      {min(lat_b):>5.2f} ms                      {(f'{min(lat_p):>5.2f} ms' if lat_p else 'n/a'):>12}")
    print(f"  Max                      {max(lat_b):>5.2f} ms                      {(f'{max(lat_p):>5.2f} ms' if lat_p else 'n/a'):>12}")
    print(f"  Median                   {sorted(lat_b)[len(lat_b)//2]:>5.2f} ms                      {(f'{sorted(lat_p)[len(lat_p)//2]:>5.2f} ms' if lat_p else 'n/a'):>12}")

    print("\nGPU Memory:")
    print("  Metric               ImageBind                      PCME                          ")
    print("  -------------------------------------------------------------------------------")
    std_b = (float(torch.tensor(gpu_b).std(unbiased=True)) if len(gpu_b) > 1 else 0.0)
    std_p = (float(torch.tensor(gpu_p).std(unbiased=True)) if gpu_p and len(gpu_p) > 1 else 0.0)
    pcme_mean_s = f"{stats.mean(gpu_p):>6.2f} MB" if gpu_p else "n/a"
    pcme_std_s = f"{std_p:>6.2f} MB" if gpu_p else "n/a"
    pcme_cv_s = f"{(std_p/(stats.mean(gpu_p)+1e-9))*100:>6.2f}%" if gpu_p else "n/a"
    print(f"  Mean                   {stats.mean(gpu_b):>6.2f} MB                     {pcme_mean_s:>12}")
    print(f"  Std Dev                  {std_b:>6.2f} MB                       {pcme_std_s:>12}")
    print(f"  CV (%)                   {((std_b/(stats.mean(gpu_b)+1e-9))*100):>6.2f}%                        {pcme_cv_s:>12}")
    print(f"  Min                    {min(gpu_b):>6.2f} MB                     {(f'{min(gpu_p):>6.2f} MB' if gpu_p else 'n/a'):>12}")
    print(f"  Max                    {max(gpu_b):>6.2f} MB                     {(f'{max(gpu_p):>6.2f} MB' if gpu_p else 'n/a'):>12}")
    print(f"  Median                 {sorted(gpu_b)[len(gpu_b)//2]:>6.2f} MB                     {(f'{sorted(gpu_p)[len(gpu_p)//2]:>6.2f} MB' if gpu_p else 'n/a'):>12}")

    print_retrieval_table("Retrieval", res_b, res_p, res_c, args.k_list)

    # Overhead
    overhead = (mean_p / (mean_b + 1e-9)) if lat_p else None
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80 + "\n")
    if lat_p:
        print(f"PCME Overhead:")
        print(f"  Latency: {overhead:.2f}x slower")
    else:
        print("PCME Overhead: n/a (PCME not run)")

    # Quick preview
    k1_b = res_b["t2v"]["R@k"][1]
    print("\n" + "="*40)
    print("  ✓ All Done!")
    print("="*40)
    print(f"ImageBind T2V R@1: {k1_b:.2f}%")
    if res_p is not None:
        k1_p = res_p["t2v"]["R@k"][1]
        print(f"PCME T2V R@1:      {k1_p:.2f}%")
        print(f"Improvement:       {k1_p - k1_b:+.2f}%")
    k1_c = res_c["t2v"]["R@k"][1]
    print(f"Cheb T2V R@1:      {k1_c:.2f}%")
    print(f"Cheb vs ImageBind: {k1_c - k1_b:+.2f}%\n")

    # Save JSON
    if args.save:
        out = {
            "runs": args.runs,
            "warmup": args.warmup,
            "num_samples": args.num_samples,
            "k_list": args.k_list,
            "N": N,
            "D": D,
            "latency": {"imagebind_ms": lat_b, "cheb_ms": lat_c},
            "gpu_mem_mb": {"imagebind": gpu_b, "cheb": gpu_c},
            "summary": {
                "imagebind": summarize_latency("ImageBind", lat_b, gpu_b),
                "cheb": summarize_latency("Cheb", lat_c, gpu_c),
                "retrieval": {"imagebind": res_b, "cheb": res_c},
            }
        }
        if lat_p is not None:
            out["latency"]["pcme_ms"] = lat_p
            out["gpu_mem_mb"]["pcme"] = gpu_p
            out["summary"]["pcme"] = summarize_latency("PCME", lat_p, gpu_p)
            out["summary"]["retrieval"]["pcme"] = res_p
            out["summary"]["overhead_latency_x"] = overhead
        if os.path.dirname(args.save):
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"✓ Saved to: {args.save}")


if __name__ == "__main__":
    main()
