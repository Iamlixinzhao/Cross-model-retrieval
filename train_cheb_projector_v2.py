#!/usr/bin/env python3

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =========================================================
# Dataset
# =========================================================

class EmbeddingDataset(Dataset):
    def __init__(self, text, video, vid_ids):
        self.text = text
        self.video = video
        self.vid_ids = vid_ids

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        return self.text[i], self.video[i], self.vid_ids[i]


# =========================================================
# Gaussian projector
# x -> (mu, logvar)
# =========================================================

class GaussianProjector(nn.Module):
    """x -> (mu, logvar). 可选 mu_on_sphere=True 时 mu 做 L2 归一化（与 PCME 一致），否则 tanh 限制在 [-1,1]。"""
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

        # PCME 用 normalize(mu) 使 mu 在单位球上，检索直接用 mu；否则 tanh 限制在 [-1,1] 供 Cheb PDF 用
        if self.mu_on_sphere:
            mu = F.normalize(mu, dim=-1)
        else:
            mu = torch.tanh(mu)

        logvar = self.logvar_net(x)
        if self.lv_ln is not None:
            logvar = self.lv_ln(logvar)

        # keep sigma in a reasonable range
        logvar = torch.clamp(logvar, -5.0, 2.0)
        return mu, logvar


# =========================================================
# Gaussian -> Chebyshev coefficient embedding
#
# 用 Chebyshev 拟合 PCME 式 Gaussian embedding：
# 对每个维度 j，该样本得到 N(mu_j, sigma_j^2)。将其 PDF
#   g_j(u) = (1/(sigma_j*sqrt(2*pi))) * exp(-(u-mu_j)^2/(2*sigma_j^2))
# 在区间 [u_min, u_max] 上按 Chebyshev 节点采样，再做离散投影得到系数 c_{j,0..K}。
# 最终 embedding = 所有维度的系数拼接，再乘 order_gates、可选 L2 归一化。
# 这样 (mu, sigma) 被表示为确定性向量 phi，可用于检索相似度。
# =========================================================

class GaussianChebCoeffLayer(nn.Module):
    """(mu, logvar) -> Chebyshev coefficients c with shape [B, D, K]."""
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

        # Chebyshev nodes on [-1,1]
        m = torch.arange(num_nodes).float()
        theta = math.pi * (m + 0.5) / num_nodes
        x_nodes = torch.cos(theta)  # [-1,1]

        # scale to [u_min, u_max]
        u_nodes = 0.5 * (u_max - u_min) * x_nodes + 0.5 * (u_max + u_min)

        # 在节点上用递推定义构造 Chebyshev 基:
        # T0(x)=1, T1(x)=x, T_{n+1}(x)=2xT_n(x)-T_{n-1}(x)
        basis = [torch.ones_like(x_nodes)]
        if order >= 1:
            basis.append(x_nodes)
        for _ in range(2, order + 1):
            basis.append(2.0 * x_nodes * basis[-1] - basis[-2])
        basis_table = torch.stack(basis[: order + 1], dim=0)  # [K, M]

        self.register_buffer("theta", theta)         # [M]
        self.register_buffer("x_nodes", x_nodes)     # [M]
        self.register_buffer("u_nodes", u_nodes)     # [M]
        self.register_buffer("basis_table", basis_table) # [K,M]

        if learnable_order_gates:
            self.order_gates = nn.Parameter(torch.ones(order + 1))
        else:
            self.register_buffer("order_gates", torch.ones(order + 1))

    def coefficients(self, mu, logvar):
        """Return Chebyshev coefficients with shape [B, D, K]."""
        sigma = torch.exp(0.5 * logvar).clamp_min(1e-4)  # [B,D]

        B, D = mu.shape
        M = self.num_nodes
        K = self.order + 1

        # [1,1,M]
        u = self.u_nodes.view(1, 1, M)

        # [B,D,1]
        mu_e = mu.unsqueeze(-1)
        sigma_e = sigma.unsqueeze(-1)

        # Gaussian PDF evaluated on nodes
        # g(u; mu, sigma) = exp(-(u-mu)^2/(2 sigma^2)) / (sigma sqrt(2pi))
        pdf = torch.exp(-0.5 * ((u - mu_e) / sigma_e) ** 2) / (
            sigma_e * math.sqrt(2.0 * math.pi)
        )  # [B,D,M]

        # Chebyshev coefficients via discrete projection
        # c_k ≈ (2/M) sum_m g(u_m) T_k(x_m)
        # c_0 usually has half weight in classical convention; here we keep a
        # consistent learned scale and let order_gates absorb global scaling.
        basis_table = self.basis_table.view(1, 1, K, M)   # [1,1,K,M]
        pdf_e = pdf.unsqueeze(2)                       # [B,D,1,M]

        coeff = (2.0 / M) * torch.sum(pdf_e * basis_table, dim=-1)  # [B,D,K]

        # 经典 Chebyshev 展开常数项为 c0/2，这里对 c0 乘 0.5 以保持一致
        coeff[:, :, 0] = 0.5 * coeff[:, :, 0]

        # apply learnable per-order gates
        coeff = coeff * self.order_gates.view(1, 1, K)
        return coeff

    def forward(self, mu, logvar):
        """Return flattened coefficients with shape [B, D*(K)]."""
        coeff = self.coefficients(mu, logvar)
        B, D, K = coeff.shape
        return coeff.reshape(B, D * K)

    def gate_summary(self):
        return {
            f"g{k}": float(self.order_gates[k].detach().cpu())
            for k in range(len(self.order_gates))
        }


class OrderBilinearSimilarity(nn.Module):
    """
    Similarity in coefficient space:
      sim(i,j) = sum_d c_i[d]^T A c_j[d] + mu_weight * <mu_i, mu_j>
    where A is a learnable KxK order-interaction matrix.
    """
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
        # Symmetrize so the kernel stays well-behaved.
        return 0.5 * (self.order_matrix + self.order_matrix.t())

    def forward(self, coeff_t, coeff_v, mu_t=None, mu_v=None):
        """
        coeff_t: [B, D, K], coeff_v: [B, D, K]
        returns sim: [B, B]
        """
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

    def summary(self):
        out = {"A_diag_mean": float(self.effective_matrix().diag().mean().detach().cpu())}
        if self.use_mu_residual:
            out["mu_weight"] = float(self.mu_weight.detach().cpu())
        return out


# =========================================================
# Losses
# =========================================================

def multi_positive_nce(sim, vid_ids, temperature=0.07):
    sim = sim / temperature
    vid_ids = vid_ids.view(-1, 1)
    pos_mask = (vid_ids == vid_ids.T).float()

    log_denom = torch.logsumexp(sim, dim=1)
    sim_pos = sim.masked_fill(pos_mask == 0, float("-inf"))
    log_num = torch.logsumexp(sim_pos, dim=1)

    return -(log_num - log_denom).mean()


def asymmetric_multi_positive_nce(sim, vid_ids, temperature=0.07, t2v_weight=1.0, v2t_weight=2.5):
    loss_t2v = multi_positive_nce(sim, vid_ids, temperature)
    loss_v2t = multi_positive_nce(sim.T, vid_ids, temperature)
    loss = t2v_weight * loss_t2v + v2t_weight * loss_v2t
    return loss, loss_t2v, loss_v2t


def distill_mu_loss(mu, x):
    # keep mu not too far from original ImageBind embedding
    x_n = F.normalize(x, dim=-1)
    mu_n = F.normalize(mu, dim=-1)
    return ((mu_n - x_n) ** 2).mean()


def variance_reg_loss(logvar, target_sigma=0.3):
    sigma = torch.exp(0.5 * logvar)
    target = torch.tensor(target_sigma, device=sigma.device, dtype=sigma.dtype)
    return ((sigma - target) ** 2).mean()


# =========================================================
# Utilities
# =========================================================

def load_vid_ids(emb_dir, infer_vid_ids=False, caps_per_video=20):
    candidates = [
        "vid_ids.pt",
        "video_ids.pt",
        "cap_vid_ids.pt",
        "text_vid_ids.pt",
        "vid_ids_text.pt",
    ]
    emb_dir = Path(emb_dir)
    for name in candidates:
        p = emb_dir / name
        if p.exists():
            print(f"Loaded video ids from {p}")
            return torch.load(p)

    if infer_vid_ids:
        text = torch.load(emb_dir / "emb_text.pt", weights_only=False)
        n = len(text)
        print(f"Infer video ids with caps_per_video={caps_per_video}")
        return torch.arange(n) // caps_per_video

    raise RuntimeError("No vid_ids file found. Use --infer_vid_ids --caps_per_video N")


# =========================================================
# Training
# =========================================================

def train(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emb_dir = Path(args.emb_dir)
    text = torch.load(emb_dir / "emb_text.pt", weights_only=False)
    video = torch.load(emb_dir / "emb_video.pt", weights_only=False)
    vid_ids = load_vid_ids(args.emb_dir, args.infer_vid_ids, args.caps_per_video)

    dataset = EmbeddingDataset(text, video, vid_ids)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    dim = text.size(-1)

    # 与 PCME 一致：text / video 各用一个 projector，模态特异
    text_proj = GaussianProjector(
        dim=dim,
        hidden=args.hidden_dim,
        residual=not args.no_residual,
        dropout=args.dropout,
        num_layers=args.num_layers,
        use_ln=args.use_ln,
        mu_on_sphere=args.mu_on_sphere,
    ).to(device)
    video_proj = GaussianProjector(
        dim=dim,
        hidden=args.hidden_dim,
        residual=not args.no_residual,
        dropout=args.dropout,
        num_layers=args.num_layers,
        use_ln=args.use_ln,
        mu_on_sphere=args.mu_on_sphere,
    ).to(device)

    coeff_layer = GaussianChebCoeffLayer(
        dim=dim,
        order=args.cheb_order,
        num_nodes=args.num_nodes,
        u_min=args.u_min,
        u_max=args.u_max,
        learnable_order_gates=not args.fixed_order_gates,
    ).to(device)

    sim_kernel = OrderBilinearSimilarity(
        order=args.cheb_order,
        normalize=not args.no_kernel_norm,
        init=args.kernel_init,
        use_mu_residual=args.kernel_use_mu_residual,
        init_mu_weight=args.init_mu_weight,
    ).to(device)

    params = list(text_proj.parameters()) + list(video_proj.parameters()) + list(coeff_layer.parameters())
    params += list(sim_kernel.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = 1e9
    best_result = None

    for epoch in range(args.epochs):
        text_proj.train()
        video_proj.train()
        coeff_layer.train()
        sim_kernel.train()

        total_loss = 0.0
        total_nce = 0.0
        total_t2v = 0.0
        total_v2t = 0.0
        total_distill = 0.0
        total_varreg = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for text_b, video_b, vid_b in pbar:
            text_b = F.normalize(text_b.to(device), dim=-1)
            video_b = F.normalize(video_b.to(device), dim=-1)
            vid_b = vid_b.to(device)

            mu_t, lv_t = text_proj(text_b)
            mu_v, lv_v = video_proj(video_b)

            coeff_t = coeff_layer.coefficients(mu_t, lv_t)
            coeff_v = coeff_layer.coefficients(mu_v, lv_v)
            sim = sim_kernel(coeff_t, coeff_v, mu_t=mu_t, mu_v=mu_v)

            if args.loss_mode == "symmetric":
                loss_t2v = multi_positive_nce(sim, vid_b, temperature=args.temperature)
                loss_v2t = multi_positive_nce(sim.T, vid_b, temperature=args.temperature)
                nce = loss_t2v + loss_v2t
            else:
                nce, loss_t2v, loss_v2t = asymmetric_multi_positive_nce(
                    sim,
                    vid_b,
                    temperature=args.temperature,
                    t2v_weight=args.t2v_weight,
                    v2t_weight=args.v2t_weight,
                )

            dloss = distill_mu_loss(mu_t, text_b) + distill_mu_loss(mu_v, video_b)
            vreg = variance_reg_loss(lv_t, args.target_sigma) + variance_reg_loss(lv_v, args.target_sigma)

            loss = nce + args.distill_weight * dloss + args.var_reg_weight * vreg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_nce += nce.item()
            total_t2v += loss_t2v.item()
            total_v2t += loss_v2t.item()
            total_distill += dloss.item()
            total_varreg += vreg.item()

            gate_info = coeff_layer.gate_summary()
            kernel_info = sim_kernel.summary()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "t2v": f"{loss_t2v.item():.4f}",
                "v2t": f"{loss_v2t.item():.4f}",
                "g0": f"{gate_info.get('g0', 0):.3f}",
                "g1": f"{gate_info.get('g1', 0):.3f}",
                "g2": f"{gate_info.get('g2', 0):.3f}",
                "mu_w": f"{kernel_info.get('mu_weight', 0):.3f}" if "mu_weight" in kernel_info else "n/a",
            })

        avg_loss = total_loss / len(loader)
        avg_nce = total_nce / len(loader)
        avg_t2v = total_t2v / len(loader)
        avg_v2t = total_v2t / len(loader)
        avg_distill = total_distill / len(loader)
        avg_varreg = total_varreg / len(loader)

        print(
            f"Epoch {epoch+1}: loss={avg_loss:.4f}, "
            f"nce={avg_nce:.4f}, t2v={avg_t2v:.4f}, v2t={avg_v2t:.4f}, "
            f"distill={avg_distill:.4f}, varreg={avg_varreg:.4f}"
        )
        print("Order gates:", coeff_layer.gate_summary())
        print("Kernel summary:", sim_kernel.summary())

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_result = {
                "epoch": epoch,
                "loss": best_loss,
                "config": vars(args),
                "order_gates": coeff_layer.gate_summary(),
                "kernel_summary": sim_kernel.summary(),
            }

            ckpt = {
                "epoch": epoch,
                "loss": best_loss,
                "config": vars(args),
                "text": text_proj.state_dict(),
                "video": video_proj.state_dict(),
                "coeff_layer": coeff_layer.state_dict(),
                "sim_kernel": sim_kernel.state_dict(),
            }
            torch.save(ckpt, Path(args.save_dir) / args.save_name)
            print(f"Saved best checkpoint: loss={best_loss:.4f}")

    if best_result is not None:
        with open(Path(args.save_dir) / (Path(args.save_name).stem + "_result.json"), "w") as f:
            json.dump(best_result, f, indent=2)

    print("Training done.")


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--emb_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_name", type=str, default="best_gaussian_cheb_coeff.pth")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--loss_mode", choices=["symmetric", "asymmetric"], default="asymmetric")
    parser.add_argument("--t2v_weight", type=float, default=1.0)
    parser.add_argument("--v2t_weight", type=float, default=2.5)

    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--no_residual", action="store_true")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--use_ln", action="store_true")

    parser.add_argument("--cheb_order", type=int, default=3)
    parser.add_argument("--num_nodes", type=int, default=16)
    parser.add_argument("--u_min", type=float, default=-3.0)
    parser.add_argument("--u_max", type=float, default=3.0)
    parser.add_argument("--fixed_order_gates", action="store_true")
    parser.add_argument("--mu_on_sphere", action="store_true",
                        help="mu = normalize(mu) 如 PCME，否则 tanh(mu)")
    parser.add_argument("--kernel_init", choices=["identity", "ones"], default="identity",
                        help="order_bilinear 的 KxK 初始矩阵")
    parser.add_argument("--no_kernel_norm", action="store_true",
                        help="关闭双线性核的 G-like cosine 归一化")
    parser.add_argument("--kernel_use_mu_residual", action="store_true",
                        help="在双线性核外额外加 mu_t @ mu_v^T 残差项")
    parser.add_argument("--init_mu_weight", type=float, default=1.0,
                        help="mu 残差项的初始权重")

    parser.add_argument("--distill_weight", type=float, default=0.02)
    parser.add_argument("--var_reg_weight", type=float, default=0.01)
    parser.add_argument("--target_sigma", type=float, default=0.3)

    parser.add_argument("--infer_vid_ids", action="store_true")
    parser.add_argument("--caps_per_video", type=int, default=20)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    train(args)