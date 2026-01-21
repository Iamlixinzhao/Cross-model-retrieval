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
    --emb_dir /scratch365/jzheng7/ImageBind/msrvtt_results \
    --ckpt   /scratch365/jzheng7/ImageBind/pcme_checkpoints_correct/best_projectors.pth \
    --runs 10 --warmup 5 --num_samples 10 --k_list 1 5 10 \
    --save /scratch365/jzheng7/ImageBind/msrvtt_results/variance_analysis.json
"""

import os
import json
import time
import math
import argparse
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

    Returns: [N, N] similarity matrix
    """
    S = max(1, num_samples)
    t_samps = sample_from_gaussian(mu_t, logvar_t, S, sigma_scale)  # [S, N, D]
    v_samps = sample_from_gaussian(mu_v, logvar_v, S, sigma_scale)  # [S, N, D]
    # cosine similarity â‰ˆ mean_s ( t_s @ v_s^T )
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
            v_mu, v_lv = video_proj(v_in)
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
            v_mu, v_lv = video_proj(v_in)
            sim = pcme_similarity(t_mu, t_lv, v_mu, v_lv, num_samples, eval_sigma_scale)  # [N,N]
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
        lat_ms.append(dt)
        gpu_peaks.append(peak_gpu_mem_mb())

    # retrieval
    with torch.no_grad():
        t_in = F.normalize(text_emb, dim=-1)
        v_in = F.normalize(video_emb, dim=-1)
        t_mu, t_lv = text_proj(t_in)
        t_mu = torch.nn.functional.normalize(t_mu, dim=-1)
        v_mu, v_lv = video_proj(v_in)
        v_mu = torch.nn.functional.normalize(v_mu, dim=-1)
        sim = pcme_similarity(t_mu, t_lv, v_mu, v_lv, num_samples, eval_sigma_scale)
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
    txt = torch.load(os.path.join(emb_dir, "emb_text.pt"), map_location=device)
    vid = torch.load(os.path.join(emb_dir, "emb_video.pt"), map_location=device)
    assert txt.dim() == 2 and vid.dim() == 2, "embeddings must be [N, D]"
    assert txt.size(0) == vid.size(0), "text/video count must match"
    return txt.to(device), vid.to(device)

def load_projectors(ckpt_path: str, in_dim: int, hidden: int, out_dim: int, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
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


def print_retrieval_table(title, base, pcme, k_list):
    def fmt_row(metric, b, p, sign="+"):
        diff = p - b
        s = f"{metric:>6}        {b:>8.2f}Â±0.00               {p:>8.2f}Â±0.00 ({sign}{diff:.2f})"
        return s

    print("\n" + "="*80)
    print("RETRIEVAL SCORES")
    print("="*80 + "\n")
    # Text -> Video
    print("Text â†’ Video:")
    print("  Metric          ImageBind                      PCME                               ")
    print("  -------------------------------------------------------------------------------")
    for k in k_list:
        b = base["t2v"]["R@k"][k]
        p = pcme["t2v"]["R@k"][k]
        print(f"  R@{k:<2}              {b:>6.2f}Â±0.00               {p:>6.2f}Â±0.00 ({p-b:+.2f})")
    print(f"  MedR               {base['t2v']['MedR']:.2f}Â±0.00               {pcme['t2v']['MedR']:.2f}Â±0.00 ({pcme['t2v']['MedR']-base['t2v']['MedR']:+.2f})")
    print(f"  MeanR              {base['t2v']['MeanR']:.2f}Â±0.00               {pcme['t2v']['MeanR']:.2f}Â±0.00 ({pcme['t2v']['MeanR']-base['t2v']['MeanR']:+.2f})")
    # Video -> Text
    print("\nVideo â†’ Text:")
    print("  Metric          ImageBind                      PCME                               ")
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
    parser.add_argument("--emb_dir", type=str, required=True, help="Directory containing emb_text.pt and emb_video.pt for the TEST split.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best_projectors.pth")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=10, help="MC samples for PCME")
    parser.add_argument("--eval_sigma_scale", type=float, default=0.0)
    parser.add_argument("--k_list", type=int, nargs="+", default=[1,5,10])
    parser.add_argument("--in_dim", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--out_dim", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save", type=str, default="")
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

    print(f"Loading PCME checkpoint: {args.ckpt}\n")
    text_proj, video_proj = load_projectors(args.ckpt, args.in_dim, args.hidden, args.out_dim, device)
    text_proj.eval()
    video_proj.eval()

    # ---------------- Baseline ----------------
    print(f"Benchmarking ImageBind ({args.runs} runs, {args.warmup} warmup)...")
    lat_b, gpu_b, res_b = benchmark_imagebind(text_emb, video_emb,
                                              args.runs, args.warmup, device, args.k_list)
    for i, (ms, gp) in enumerate(zip(lat_b, gpu_b), 1):
        print(f"  Run {i}/{args.runs}: {ms:.2f}ms, GPU peak: {gp:.1f}MB")

    # -------------- PCME (MC) -----------------
    print(f"\nBenchmarking PCME ({args.runs} runs, {args.warmup} warmup, {args.num_samples} MC samples)...")
    lat_p, gpu_p, res_p = benchmark_pcme(text_emb, video_emb, text_proj, video_proj,
                                         args.runs, args.warmup, device, args.num_samples, args.k_list, args.eval_sigma_scale)
    for i, (ms, gp) in enumerate(zip(lat_p, gpu_p), 1):
        print(f"  Run {i}/{args.runs}: {ms:.2f}ms, GPU peak: {gp:.1f}MB")

    # ---------------- Summary -----------------
    import statistics as stats
    print("\n" + "="*80)
    print("DETAILED VARIANCE ANALYSIS")
    print("="*80 + "\n")

    # Latency summary table
    mean_b, sd_b = (stats.mean(lat_b), (stats.stdev(lat_b) if len(lat_b) > 1 else 0.0))
    mean_p, sd_p = (stats.mean(lat_p), (stats.stdev(lat_p) if len(lat_p) > 1 else 0.0))
    cv_b = (sd_b / mean_b * 100.0) if mean_b > 0 else 0.0
    cv_p = (sd_p / mean_p * 100.0) if mean_p > 0 else 0.0
    ci_b = human_interval(lat_b)
    ci_p = human_interval(lat_p)

    print("Latency:")
    print("  Metric               ImageBind                      PCME                          ")
    print("  -------------------------------------------------------------------------------")
    print(f"  Mean                     {mean_b:>5.2f} ms                      {mean_p:>5.2f} ms")
    print(f"  Std Dev                  {sd_b:>5.2f} ms                       {sd_p:>5.2f} ms")
    print(f"  CV (%)                   {cv_b:>5.2f}%                        {cv_p:>5.2f}%")
    print(f"  95% CI               [{ci_b[0]:.2f}, {ci_b[1]:.2f}]           [{ci_p[0]:.2f}, {ci_p[1]:.2f}]")
    print(f"  Min                      {min(lat_b):>5.2f} ms                      {min(lat_p):>5.2f} ms")
    print(f"  Max                      {max(lat_b):>5.2f} ms                      {max(lat_p):>5.2f} ms")
    print(f"  Median                   {sorted(lat_b)[len(lat_b)//2]:>5.2f} ms                      {sorted(lat_p)[len(lat_p)//2]:>5.2f} ms")

    print("\nGPU Memory:")
    print("  Metric               ImageBind                      PCME                          ")
    print("  -------------------------------------------------------------------------------")
    print(f"  Mean                   {stats.mean(gpu_b):>6.2f} MB                     {stats.mean(gpu_p):>6.2f} MB")
    std_b = (float(torch.tensor(gpu_b).std(unbiased=True)) if len(gpu_b) > 1 else 0.0)
    std_p = (float(torch.tensor(gpu_p).std(unbiased=True)) if len(gpu_p) > 1 else 0.0)
    print(f"  Std Dev                  {std_b:>6.2f} MB                       {std_p:>6.2f} MB")
    print(f"  CV (%)                   {((std_b/(stats.mean(gpu_b)+1e-9))*100):>6.2f}%                        {((std_p/(stats.mean(gpu_p)+1e-9))*100):>6.2f}%")
    print(f"  Min                    {min(gpu_b):>6.2f} MB                     {min(gpu_p):>6.2f} MB")
    print(f"  Max                    {max(gpu_b):>6.2f} MB                     {max(gpu_p):>6.2f} MB")
    print(f"  Median                 {sorted(gpu_b)[len(gpu_b)//2]:>6.2f} MB                     {sorted(gpu_p)[len(gpu_p)//2]:>6.2f} MB")

    print_retrieval_table("Retrieval", res_b, res_p, args.k_list)

    # Overhead
    overhead = (mean_p / (mean_b + 1e-9))
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80 + "\n")
    print(f"PCME Overhead:")
    print(f"  Latency: {overhead:.2f}x slower")

    # Quick preview
    k1_b = res_b["t2v"]["R@k"][1]
    k1_p = res_p["t2v"]["R@k"][1]
    print("\n" + "="*40)
    print("  âœ“ All Done!")
    print("="*40)
    print(f"ImageBind T2V R@1: {k1_b:.2f}%")
    print(f"PCME T2V R@1:      {k1_p:.2f}%")
    print(f"Improvement:       {k1_p - k1_b:+.2f}%\n")

    # Save JSON
    if args.save:
        out = {
            "runs": args.runs,
            "warmup": args.warmup,
            "num_samples": args.num_samples,
            "k_list": args.k_list,
            "N": N,
            "D": D,
            "latency": {
                "imagebind_ms": lat_b,
                "pcme_ms": lat_p
            },
            "gpu_mem_mb": {
                "imagebind": gpu_b,
                "pcme": gpu_p
            },
            "summary": {
                "imagebind": summarize_latency("ImageBind", lat_b, gpu_b),
                "pcme": summarize_latency("PCME", lat_p, gpu_p),
                "retrieval": {
                    "imagebind": res_b,
                    "pcme": res_p
                },
                "overhead_latency_x": overhead
            }
        }
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"âœ“ Saved to: {args.save}")


if __name__ == "__main__":
    main()
