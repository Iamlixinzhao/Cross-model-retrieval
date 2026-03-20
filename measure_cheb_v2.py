#!/usr/bin/env python3
"""
评测 train_cheb_projector_v2.py 保存的 checkpoint（GaussianProjector + GaussianChebCoeffLayer）。
即：用 Chebyshev 系数表示 Gaussian embedding 的那一版。

用法:
  python measure_cheb_v2.py --emb_dir /path/to/msrvtt_results \\
    --cheb_ckpt ./sweep_runs/run_gaussian_cheb/best_gaussian_cheb_coeff.pth \\
    --save ./sweep_runs/run_gaussian_cheb/metrics.json
"""
import os
import json
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from measure_latency_memory_variance import (
    load_embeddings,
    benchmark_imagebind,
    summarize_retrieval,
    summarize_latency,
    reset_peak_gpu_mem,
    peak_gpu_mem_mb,
)


# -----------------------------
# 与 train_cheb_projector_v2 一致的模型（Gaussian + Cheb 系数）
# -----------------------------

class GaussianProjector(nn.Module):
    """x -> (mu, logvar). mu_on_sphere 时 mu 做 L2 归一化（PCME），否则 tanh。"""
    def __init__(self, dim, hidden=None, residual=True, dropout=0.1, num_layers=1, use_ln=False, mu_on_sphere=False):
        super().__init__()
        self.residual = residual
        self.use_ln = use_ln
        self.mu_on_sphere = mu_on_sphere
        if hidden is None:
            hidden = dim
        def build_mlp():
            layers = [
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            for _ in range(num_layers - 1):
                layers.extend([
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
            layers.append(nn.Linear(hidden, dim))
            return nn.Sequential(*layers)
        self.mu_net = build_mlp()
        self.logvar_net = build_mlp()
        self.mu_ln = nn.LayerNorm(dim) if use_ln else None
        self.lv_ln = nn.LayerNorm(dim) if use_ln else None

    def forward(self, x):
        mu = self.mu_net(x)
        if self.residual:
            mu = x + mu
        if self.mu_ln is not None:
            mu = self.mu_ln(mu)
        if self.mu_on_sphere:
            mu = F.normalize(mu, dim=-1)
        else:
            mu = torch.tanh(mu)
        logvar = self.logvar_net(x)
        if self.lv_ln is not None:
            logvar = self.lv_ln(logvar)
        logvar = torch.clamp(logvar, -5.0, 2.0)
        return mu, logvar


class GaussianChebCoeffLayer(nn.Module):
    """(mu, logvar) -> flattened Chebyshev coefficients."""
    def __init__(
        self,
        dim,
        order=3,
        num_nodes=16,
        u_min=-3.0,
        u_max=3.0,
        learnable_order_gates=True,
    ):
        super().__init__()
        self.dim = dim
        self.order = order
        self.num_nodes = num_nodes
        self.u_min = u_min
        self.u_max = u_max

        m = torch.arange(num_nodes).float()
        theta = math.pi * (m + 0.5) / num_nodes
        x_nodes = torch.cos(theta)
        u_nodes = 0.5 * (u_max - u_min) * x_nodes + 0.5 * (u_max + u_min)
        basis = [torch.ones_like(x_nodes)]
        if order >= 1:
            basis.append(x_nodes)
        for _ in range(2, order + 1):
            basis.append(2.0 * x_nodes * basis[-1] - basis[-2])
        basis_table = torch.stack(basis[: order + 1], dim=0)

        self.register_buffer("theta", theta)
        self.register_buffer("x_nodes", x_nodes)
        self.register_buffer("u_nodes", u_nodes)
        self.register_buffer("basis_table", basis_table)

        if learnable_order_gates:
            self.order_gates = nn.Parameter(torch.ones(order + 1))
        else:
            self.register_buffer("order_gates", torch.ones(order + 1))

    def forward(self, mu, logvar):
        coeff = self.coefficients(mu, logvar)
        B, D, K = coeff.shape
        return coeff.reshape(B, D * K)

    def coefficients(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar).clamp_min(1e-4)
        B, D = mu.shape
        M = self.num_nodes
        K = self.order + 1

        u = self.u_nodes.view(1, 1, M)
        mu_e = mu.unsqueeze(-1)
        sigma_e = sigma.unsqueeze(-1)
        pdf = torch.exp(-0.5 * ((u - mu_e) / sigma_e) ** 2) / (
            sigma_e * math.sqrt(2.0 * math.pi)
        )
        basis_table = self.basis_table.view(1, 1, K, M)
        pdf_e = pdf.unsqueeze(2)
        coeff = (2.0 / M) * torch.sum(pdf_e * basis_table, dim=-1)
        coeff[:, :, 0] = 0.5 * coeff[:, :, 0]
        coeff = coeff * self.order_gates.view(1, 1, K)
        return coeff


class OrderBilinearSimilarity(nn.Module):
    def __init__(self, order, normalize=True, init="identity", use_mu_residual=False, init_mu_weight=1.0):
        super().__init__()
        self.order = order
        self.normalize = normalize
        self.use_mu_residual = use_mu_residual
        K = order + 1
        if init == "identity":
            A0 = torch.eye(K)
        elif init == "ones":
            A0 = torch.ones(K, K) / float(K)
        else:
            raise ValueError(f"Unknown kernel init: {init}")
        self.order_matrix = nn.Parameter(A0)
        if use_mu_residual:
            self.mu_weight = nn.Parameter(torch.tensor(float(init_mu_weight)))

    def effective_matrix(self):
        return 0.5 * (self.order_matrix + self.order_matrix.t())

    def forward(self, coeff_t, coeff_v, mu_t=None, mu_v=None):
        A = self.effective_matrix()
        t_a = torch.einsum("bdk,kl->bdl", coeff_t, A)
        sim = torch.einsum("bdl,mdl->bm", t_a, coeff_v)
        if self.normalize:
            v_a = torch.einsum("mdk,kl->mdl", coeff_v, A)
            t_norm = torch.sqrt(torch.einsum("bdl,bdl->b", t_a, coeff_t).clamp_min(1e-8))
            v_norm = torch.sqrt(torch.einsum("mdl,mdl->m", v_a, coeff_v).clamp_min(1e-8))
            sim = sim / (t_norm[:, None] * v_norm[None, :]).clamp_min(1e-8)
        if self.use_mu_residual:
            if mu_t is None or mu_v is None:
                raise ValueError("mu_t and mu_v are required when use_mu_residual=True")
            sim = sim + self.mu_weight * (F.normalize(mu_t, dim=-1) @ F.normalize(mu_v, dim=-1).t())
        return sim

# -----------------------------
# 加载与 benchmark
# -----------------------------


def load_gaussian_cheb_coeff_ckpt(ckpt_path: str, dim: int, device: str):
    """
    加载 train_cheb_projector_v2 的 Gaussian + Cheb 系数 checkpoint。
    若 ckpt 含 "text"/"video" 则双 projector（与 PCME 一致），否则用 "projector" 单头。
    返回 (text_proj, video_proj, coeff_layer)，单头时 text_proj=video_proj。
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None, None, None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "coeff_layer" not in ckpt:
        return None, None, None
    config = ckpt.get("config") or {}

    hidden = config.get("hidden_dim")
    residual = not config.get("no_residual", False)
    dropout = config.get("dropout", 0.1)
    num_layers = config.get("num_layers", 1)
    use_ln = config.get("use_ln", False)
    mu_on_sphere = config.get("mu_on_sphere", False)

    def make_projector():
        return GaussianProjector(
            dim=dim,
            hidden=hidden,
            residual=residual,
            dropout=0.0,
            num_layers=num_layers,
            use_ln=use_ln,
            mu_on_sphere=mu_on_sphere,
        ).to(device)

    if "text" in ckpt and "video" in ckpt:
        text_proj = make_projector()
        video_proj = make_projector()
        text_proj.load_state_dict(ckpt["text"], strict=True)
        video_proj.load_state_dict(ckpt["video"], strict=True)
        text_proj.eval()
        video_proj.eval()
    else:
        text_proj = make_projector()
        text_proj.load_state_dict(ckpt["projector"], strict=True)
        text_proj.eval()
        video_proj = text_proj

    order = config.get("cheb_order", 3)
    num_nodes = config.get("num_nodes", 16)
    u_min = config.get("u_min", -3.0)
    u_max = config.get("u_max", 3.0)
    learnable_order_gates = not config.get("fixed_order_gates", False)
    coeff_layer = GaussianChebCoeffLayer(
        dim=dim,
        order=order,
        num_nodes=num_nodes,
        u_min=u_min,
        u_max=u_max,
        learnable_order_gates=learnable_order_gates,
    ).to(device)
    coeff_layer.load_state_dict(ckpt["coeff_layer"], strict=True)
    coeff_layer.eval()

    sim_kernel = OrderBilinearSimilarity(
        order=order,
        normalize=not config.get("no_kernel_norm", False),
        init=config.get("kernel_init", "identity"),
        use_mu_residual=config.get("kernel_use_mu_residual", False),
        init_mu_weight=config.get("init_mu_weight", 1.0),
    ).to(device)
    if "sim_kernel" not in ckpt:
        raise KeyError("Checkpoint missing 'sim_kernel'; this script now only supports bilinear-kernel checkpoints.")
    sim_kernel.load_state_dict(ckpt["sim_kernel"], strict=True)
    sim_kernel.eval()

    return text_proj, video_proj, coeff_layer, sim_kernel


def benchmark_gaussian_cheb_coeff(text_emb, video_emb, text_proj, video_proj, coeff_layer, sim_kernel,
                                  runs, warmup, device, k_list):
    """双线性 kernel: sum_d c_t[d]^T A c_v[d] + optional mu residual."""
    def build_gaussian_t(x):
        x_in = F.normalize(x, dim=-1)
        return text_proj(x_in)

    def build_gaussian_v(x):
        x_in = F.normalize(x, dim=-1)
        return video_proj(x_in)

    def build_sim():
        mu_t, lv_t = build_gaussian_t(text_emb)
        mu_v, lv_v = build_gaussian_v(video_emb)
        coeff_t = coeff_layer.coefficients(mu_t, lv_t)
        coeff_v = coeff_layer.coefficients(mu_v, lv_v)
        return sim_kernel(coeff_t, coeff_v, mu_t=mu_t, mu_v=mu_v)

    if not torch.cuda.is_available():
        def reset_peak_gpu_mem(): pass
        def peak_gpu_mem_mb(): return 0.0

    reset_peak_gpu_mem()
    for _ in range(warmup):
        with torch.no_grad():
            _ = build_sim()
    _ = peak_gpu_mem_mb()

    import time
    lat_ms, gpu_peaks = [], []
    for _ in range(runs):
        reset_peak_gpu_mem()
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            sim = build_sim()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
        lat_ms.append(dt)
        gpu_peaks.append(peak_gpu_mem_mb())

    with torch.no_grad():
        sim = build_sim()
    t2v_ranks, v2t_ranks = ranks_from_sim(sim)
    res = summarize_retrieval(t2v_ranks, v2t_ranks, k_list)
    return lat_ms, gpu_peaks, res


def ranks_from_sim(sim: torch.Tensor):
    """从相似度矩阵得到 rank（与 measure 主脚本一致）。"""
    N = sim.size(0)
    sort_idx = torch.argsort(sim, dim=1, descending=True)
    gt = torch.arange(N, device=sim.device)
    t2v_rank = (sort_idx == gt[:, None]).nonzero()[:, 1] + 1
    sort_idx_T = torch.argsort(sim.t(), dim=1, descending=True)
    v2t_rank = (sort_idx_T == gt[:, None]).nonzero()[:, 1] + 1
    return t2v_rank, v2t_rank


def main():
    parser = argparse.ArgumentParser(description="Measure Gaussian+ChebCoeff V2 checkpoint")
    parser.add_argument("--emb_dir", type=str, required=True, help="emb_text.pt / emb_video.pt 目录")
    parser.add_argument("--cheb_ckpt", type=str, required=True,
                       help="best_gaussian_cheb_coeff.pth 路径")
    parser.add_argument("--baseline_name", type=str, default="ImageBind",
                       help="原始 embedding baseline 的显示名称，例如 ImageBind / CLIP")
    parser.add_argument("--save", type=str, default=None, help="输出 JSON 路径")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--k_list", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CUDA not available, using CPU.")
    else:
        torch.backends.cudnn.benchmark = True

    print("Loading embeddings...")
    text_emb, video_emb = load_embeddings(args.emb_dir, device)
    N, D = text_emb.size(0), text_emb.size(1)
    print(f"N={N}, D={D}\n")

    print("Loading Gaussian+ChebCoeff V2 checkpoint...")
    text_proj, video_proj, coeff, sim_kernel = load_gaussian_cheb_coeff_ckpt(args.cheb_ckpt, D, device)
    if coeff is None:
        raise FileNotFoundError(
            f"Not a Gaussian+ChebCoeff V2 checkpoint (missing 'coeff_layer'): {args.cheb_ckpt}"
        )
    two_heads = text_proj is not video_proj
    print("  Text + Video projectors (PCME-style)" if two_heads else "  Single projector (legacy)")
    print("  Coeff layer loaded.\n")

    print(f"Benchmarking {args.baseline_name} ({args.runs} runs, {args.warmup} warmup)...")
    lat_b, gpu_b, res_b = benchmark_imagebind(
        text_emb, video_emb, args.runs, args.warmup, device, args.k_list)
    print("Benchmarking Gaussian+ChebCoeff V2 (order_bilinear kernel)...")
    lat_c, gpu_c, res_c = benchmark_gaussian_cheb_coeff(
        text_emb, video_emb, text_proj, video_proj, coeff, sim_kernel,
        runs=args.runs, warmup=args.warmup,
        device=device, k_list=args.k_list,
    )

    import statistics as stats
    print("\n" + "=" * 60)
    print("Retrieval (Text → Video)")
    for k in args.k_list:
        b, c = res_b["t2v"]["R@k"][k], res_c["t2v"]["R@k"][k]
        print(f"  R@{k}   {args.baseline_name}: {b:.2f}%   GaussChebV2: {c:.2f}%   ({c - b:+.2f})")
    print("\nRetrieval (Video → Text)")
    for k in args.k_list:
        b, c = res_b["v2t"]["R@k"][k], res_c["v2t"]["R@k"][k]
        print(f"  R@{k}   {args.baseline_name}: {b:.2f}%   GaussChebV2: {c:.2f}%   ({c - b:+.2f})")
    print(f"\nLatency: {args.baseline_name} {stats.mean(lat_b):.2f} ms   GaussChebV2 {stats.mean(lat_c):.2f} ms")

    if args.save:
        out = {
            "runs": args.runs, "warmup": args.warmup, "k_list": args.k_list, "sim_mode": "order_bilinear",
            "baseline_name": args.baseline_name,
            "N": N, "D": D,
            "latency": {"imagebind_ms": lat_b, "gaussian_cheb_v2_ms": lat_c},
            "gpu_mem_mb": {"imagebind": gpu_b, "gaussian_cheb_v2": gpu_c},
            "summary": {
                "imagebind": summarize_latency(args.baseline_name, lat_b, gpu_b),
                "gaussian_cheb_v2": summarize_latency("GaussChebV2", lat_c, gpu_c),
                "retrieval": {"imagebind": res_b, "gaussian_cheb_v2": res_c},
            },
        }
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved: {args.save}")


if __name__ == "__main__":
    main()
