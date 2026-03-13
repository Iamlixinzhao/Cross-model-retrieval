#!/usr/bin/env python3
"""
Demo: PCME Monte-Carlo matching vs Chebyshev polynomial embedding (deterministic)

- Monte Carlo side: z ~ N(mu, sigma^2 I), p_MC = E[sigmoid(-a||z_t-z_v||^2 + b)] via K^2 samples
- Cheb side: Phi(x) = [T1(x), T3(x), T5(x), ...] (odd only), score s = <Phi(mu_t), Phi(mu_v)>
- Calibration: fit logit(p_MC) ≈ alpha * s + beta (linear regression), then p_cheb = sigmoid(alpha*s+beta)
- Prints p_MC distribution stats to detect saturation/near-constant regime.
- Structure intentionally mirrors train_cheb_projector.py style: single file, argparse, clear functions.
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# Chebyshev feature expansion (odd degrees only)
# ============================================================

def cheb_expand_odd(x: torch.Tensor, max_odd: int, gates=None, block_norm=False):
    """
    x: [B, D], expected roughly in [-1,1] (we clamp)
    max_odd: 1,3,5,7,...
    gates: optional dict {1:g1, 3:g3, 5:g5, ...}
    block_norm: if True, L2-normalize each odd-degree block before concatenation
    """
    assert max_odd >= 1 and (max_odd % 2 == 1), "max_odd must be odd and >= 1"
    x = torch.clamp(x, -1.0, 1.0)

    if gates is None:
        gates = {}

    def apply_gate(t, k):
        g = gates.get(k, 1.0)
        return t * float(g)

    # recursion:
    # T0 = 1, T1 = x
    T_prev = torch.ones_like(x)  # T0
    T_curr = x                  # T1

    blocks = []
    # T1
    t = apply_gate(T_curr, 1)
    if block_norm:
        t = F.normalize(t, dim=-1)
    blocks.append(t)

    # build T2..Tmax
    for n in range(2, max_odd + 1):
        T_next = 2.0 * x * T_curr - T_prev
        T_prev, T_curr = T_curr, T_next
        if n % 2 == 1:
            t = apply_gate(T_curr, n)
            if block_norm:
                t = F.normalize(t, dim=-1)
            blocks.append(t)

    phi = torch.cat(blocks, dim=-1)
    phi = F.normalize(phi, dim=-1)
    return phi


# ============================================================
# PCME-style Monte Carlo matching probability
# ============================================================

@torch.no_grad()
def mc_match_prob(mu_t, sigma_t, mu_v, sigma_v, K=8, a=1.0, b=0.0, normalize_samples=True):
    """
    mu_t, mu_v: [B, D]
    sigma_t, sigma_v: [B, 1] or [B, D] (std, diagonal)
    returns p_mc: [B]
    """
    B, D = mu_t.shape
    device = mu_t.device

    eps_t = torch.randn(B, K, D, device=device)
    eps_v = torch.randn(B, K, D, device=device)

    # broadcast sigma
    sig_t = sigma_t
    sig_v = sigma_v
    if sig_t.ndim == 2 and sig_t.shape[1] == 1:
        sig_t = sig_t.view(B, 1, 1)
    else:
        sig_t = sig_t.view(B, 1, D)
    if sig_v.ndim == 2 and sig_v.shape[1] == 1:
        sig_v = sig_v.view(B, 1, 1)
    else:
        sig_v = sig_v.view(B, 1, D)

    zt = mu_t.unsqueeze(1) + eps_t * sig_t  # [B,K,D]
    zv = mu_v.unsqueeze(1) + eps_v * sig_v  # [B,K,D]

    if normalize_samples:
        zt = F.normalize(zt, dim=-1)
        zv = F.normalize(zv, dim=-1)

    # pairwise within each pair: [B,K,K,D]
    diff = zt.unsqueeze(2) - zv.unsqueeze(1)
    dist2 = (diff * diff).sum(dim=-1)  # [B,K,K]

    logits = (-a * dist2 + b)
    p = torch.sigmoid(logits).mean(dim=(1, 2))  # [B]
    return p


# ============================================================
# Calibration: fit logit(p_mc) ~ alpha*s + beta
# ============================================================

def logit(p: np.ndarray, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

def fit_logit_calibration(sim: np.ndarray, p_mc: np.ndarray):
    """
    Closed-form linear regression:
      y = alpha * sim + beta, where y = logit(p_mc)
    """
    y = logit(p_mc)
    x = sim

    x_mean = x.mean()
    y_mean = y.mean()
    cov = np.mean((x - x_mean) * (y - y_mean))
    var = np.mean((x - x_mean) ** 2) + 1e-12

    alpha = cov / var
    beta = y_mean - alpha * x_mean
    return float(alpha), float(beta)

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================
# Utilities: stats + plotting
# ============================================================

def print_stats(name: str, arr: np.ndarray):
    q = np.quantile(arr, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    print(f"{name} stats:")
    print(f"  mean={arr.mean():.6f}, std={arr.std():.6f}, min={arr.min():.6f}, max={arr.max():.6f}")
    print(f"  quantiles [0,1,5,50,95,99,100]% = {', '.join([f'{v:.6f}' for v in q])}")

def scatter_plot(x, y, title, xlabel, ylabel):
    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, s=10)
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def hist_plot(err, title, xlabel):
    plt.figure(figsize=(7, 4))
    plt.hist(err, bins=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main demo (train_cheb_projector-like structure)
# ============================================================

def run_demo(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, D = args.num_pairs, args.dim

    # 1) generate deterministic embeddings (ImageBind-like) and normalize
    x_t = F.normalize(torch.randn(B, D, device=device), dim=-1)
    x_v = F.normalize(torch.randn(B, D, device=device), dim=-1)

    # 2) define probabilistic embeddings around x (toy PCME-like)
    mu_t = x_t
    mu_v = x_v

    if args.sigma_mode == "constant":
        sigma_t = torch.full((B, 1), args.sigma, device=device)
        sigma_v = torch.full((B, 1), args.sigma, device=device)
    else:  # random
        sigma_t = torch.empty(B, 1, device=device).uniform_(args.sigma * 0.5, args.sigma * 1.5)
        sigma_v = torch.empty(B, 1, device=device).uniform_(args.sigma * 0.5, args.sigma * 1.5)

    # 3) MC probability (K^2)
    p_mc = mc_match_prob(
        mu_t, sigma_t, mu_v, sigma_v,
        K=args.K, a=args.a, b=args.b,
        normalize_samples=not args.no_norm_samples
    ).detach().cpu().numpy()

    print("======================================================")
    print("PCME Monte-Carlo vs Chebyshev Deterministic Demo")
    print("======================================================")
    print(f"Pairs B={B}, Dim D={D}, MC samples K={args.K}")
    print(f"MC score: sigmoid(-a||z_t-z_v||^2 + b), a={args.a}, b={args.b}")
    print(f"sigma_mode={args.sigma_mode}, sigma_base={args.sigma}, normalize_samples={not args.no_norm_samples}")
    print(f"Cheb: odd degrees up to {args.max_odd}, block_norm={args.block_norm}")

    # 4) Cheb deterministic similarity s = <Phi(mu_t), Phi(mu_v)>
    gates = {1: 1.0}
    if args.max_odd >= 3: gates[3] = args.g3
    if args.max_odd >= 5: gates[5] = args.g5
    if args.max_odd >= 7: gates[7] = args.g7

    phi_t = cheb_expand_odd(mu_t, max_odd=args.max_odd, gates=gates, block_norm=args.block_norm)
    phi_v = cheb_expand_odd(mu_v, max_odd=args.max_odd, gates=gates, block_norm=args.block_norm)
    sim = (phi_t * phi_v).sum(dim=-1).detach().cpu().numpy()

    # 5) Print distribution stats (NEW)
    print_stats("p_MC", p_mc)
    print_stats("sim_Cheb", sim)

    # 6) Calibration on split: fit logit(p_mc) ~ alpha*sim + beta (NEW)
    n_cal = int(args.calib_ratio * B)
    sim_cal, sim_test = sim[:n_cal], sim[n_cal:]
    p_cal, p_test = p_mc[:n_cal], p_mc[n_cal:]

    alpha, beta = fit_logit_calibration(sim_cal, p_cal)
    p_cheb = sigmoid_np(alpha * sim + beta)

    # evaluate on test
    p_cheb_test = p_cheb[n_cal:]
    mae = float(np.mean(np.abs(p_cheb_test - p_test)))
    mse = float(np.mean((p_cheb_test - p_test) ** 2))
    corr = float(np.corrcoef(p_cheb_test, p_test)[0, 1])

    print("------------------------------------------------------")
    print(f"Calibration split: {n_cal}/{B} ({args.calib_ratio:.2f})")
    print(f"Fitted (logit regression): alpha={alpha:.6f}, beta={beta:.6f}")
    print(f"Test: MAE={mae:.6f}, MSE={mse:.6f}, Corr={corr:.4f}")

    # 7) Plots
    if not args.no_plots:
        scatter_plot(
            p_test, p_cheb_test,
            title="Match Probability: MC vs Chebyshev (calibrated via logit)",
            xlabel="p_MC (Monte Carlo)", ylabel="p_Cheb (Deterministic)"
        )
        err = p_cheb_test - p_test
        hist_plot(err, "Approximation Error Distribution", "p_Cheb - p_MC")


def main():
    ap = argparse.ArgumentParser()

    # data size
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_pairs", type=int, default=2000)
    ap.add_argument("--dim", type=int, default=256)

    # MC / PCME proxy
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=0.0)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--sigma_mode", type=str, default="constant", choices=["constant", "random"])
    ap.add_argument("--no_norm_samples", action="store_true",
                    help="Disable L2-normalization on sampled z (less ImageBind-like).")

    # Cheb
    ap.add_argument("--max_odd", type=int, default=1, help="1,3,5,7,...")
    ap.add_argument("--g3", type=float, default=1.0)
    ap.add_argument("--g5", type=float, default=1.0)
    ap.add_argument("--g7", type=float, default=1.0)
    ap.add_argument("--block_norm", action="store_true",
                    help="Normalize each odd-degree block before concatenation.")

    # calibration split
    ap.add_argument("--calib_ratio", type=float, default=0.3)

    # plotting
    ap.add_argument("--no_plots", action="store_true")

    args = ap.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()