#!/usr/bin/env python3

import argparse
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
# Projector / Adapter
# =========================================================

class ChebProjector(nn.Module):
    """
    Adapter after ImageBind embedding.
    Output h is constrained to [-1,1] for Chebyshev basis.
    """
    def __init__(self, dim, hidden=None, residual=True, dropout=0.1):
        super().__init__()
        self.residual = residual

        if hidden is None:
            hidden = dim

        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        out = self.net(x)
        if self.residual:
            out = x + out
        h = torch.tanh(out)
        return h


# =========================================================
# Gated Chebyshev basis
# Phi(h) = [g1*T1(h), g2*T2(h)?, g3*T3(h), g5*T5(h)?]
# --no_t2: no T2; --include_t5: add T5 (odd-only T1+T3+T5)
# =========================================================

class GatedChebyshevLayer(nn.Module):
    def __init__(self, dim, gate_mode="scalar", include_t2=True, include_t3=True, include_t5=False,
                 init_gates=(1.0, 0.1, 0.1), init_g5=0.1):
        super().__init__()
        self.dim = dim
        self.gate_mode = gate_mode
        self.include_t2 = include_t2
        self.include_t3 = include_t3
        self.include_t5 = include_t5

        if gate_mode == "scalar":
            self.g1 = nn.Parameter(torch.tensor(float(init_gates[0])))
            if include_t2:
                self.g2 = nn.Parameter(torch.tensor(float(init_gates[1])))
            if include_t3:
                self.g3 = nn.Parameter(torch.tensor(float(init_gates[2])))
            if include_t5:
                self.g5 = nn.Parameter(torch.tensor(float(init_g5)))
        elif gate_mode == "vector":
            self.g1 = nn.Parameter(torch.full((dim,), float(init_gates[0])))
            if include_t2:
                self.g2 = nn.Parameter(torch.full((dim,), float(init_gates[1])))
            if include_t3:
                self.g3 = nn.Parameter(torch.full((dim,), float(init_gates[2])))
            if include_t5:
                self.g5 = nn.Parameter(torch.full((dim,), float(init_g5)))
        else:
            raise ValueError("gate_mode must be 'scalar' or 'vector'")

    def forward(self, h):
        T1 = h
        T2 = 2.0 * h * h - 1.0
        T3 = 4.0 * h * h * h - 3.0 * h
        if self.include_t5:
            T5 = 16.0 * h ** 5 - 20.0 * h * h * h + 5.0 * h

        if self.gate_mode == "scalar":
            g1 = self.g1
            g2 = self.g2 if self.include_t2 else None
            g3 = self.g3 if self.include_t3 else None
            g5 = self.g5 if self.include_t5 else None
        else:
            g1 = self.g1.view(1, -1)
            g2 = self.g2.view(1, -1) if self.include_t2 else None
            g3 = self.g3.view(1, -1) if self.include_t3 else None
            g5 = self.g5.view(1, -1) if self.include_t5 else None

        parts = [g1 * T1]
        if self.include_t2:
            parts.append(g2 * T2)
        if self.include_t3:
            parts.append(g3 * T3)
        if self.include_t5:
            parts.append(g5 * T5)

        phi = torch.cat(parts, dim=-1)
        return phi

    def gate_summary(self):
        out = {}
        if self.gate_mode == "scalar":
            out["g1"] = float(self.g1.detach().cpu())
            if self.include_t2:
                out["g2"] = float(self.g2.detach().cpu())
            if self.include_t3:
                out["g3"] = float(self.g3.detach().cpu())
            if self.include_t5:
                out["g5"] = float(self.g5.detach().cpu())
        else:
            out["g1_mean"] = float(self.g1.detach().mean().cpu())
            if self.include_t2:
                out["g2_mean"] = float(self.g2.detach().mean().cpu())
            if self.include_t3:
                out["g3_mean"] = float(self.g3.detach().mean().cpu())
            if self.include_t5:
                out["g5_mean"] = float(self.g5.detach().mean().cpu())
        return out


# =========================================================
# Multi-positive InfoNCE
# =========================================================

def multi_positive_nce(sim, vid_ids, temperature=0.07):
    """
    sim: [B,B]
    vid_ids: [B]
    positives = same vid_id
    """
    sim = sim / temperature
    vid_ids = vid_ids.view(-1, 1)
    pos_mask = (vid_ids == vid_ids.T).float()

    log_denom = torch.logsumexp(sim, dim=1)
    sim_pos = sim.masked_fill(pos_mask == 0, float("-inf"))
    log_num = torch.logsumexp(sim_pos, dim=1)

    loss = -(log_num - log_denom).mean()
    return loss


def asymmetric_multi_positive_nce(
    sim,
    vid_ids,
    temperature=0.07,
    t2v_weight=1.0,
    v2t_weight=2.0,
):
    """
    T2V = text query, retrieve videos
    V2T = video query, retrieve texts
    """
    loss_t2v = multi_positive_nce(sim, vid_ids, temperature)
    loss_v2t = multi_positive_nce(sim.T, vid_ids, temperature)
    loss = t2v_weight * loss_t2v + v2t_weight * loss_v2t
    return loss, loss_t2v, loss_v2t


# =========================================================
# Distillation / keep close to original ImageBind space
# =========================================================

def distill_loss(h, x):
    return ((F.normalize(h, dim=-1) - F.normalize(x, dim=-1)) ** 2).mean()


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
        print(f"Infer video ids with caps_per_video={caps_per_video}")
        text = torch.load(emb_dir / "emb_text.pt", weights_only=False)
        N = len(text)
        return torch.arange(N) // caps_per_video

    raise RuntimeError(
        "No vid_ids file found. Provide vid_ids.pt or use --infer_vid_ids --caps_per_video N"
    )


# =========================================================
# Training
# =========================================================

def train(args):
    # seed for reproducibility
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
        num_workers=4,
        drop_last=True,
    )

    dim = text.size(-1)

    projector = None
    if not args.no_projector:
        projector = ChebProjector(
            dim=dim,
            hidden=args.hidden_dim,
            residual=not args.no_residual,
            dropout=args.dropout,
        ).to(device)

    # t1_only => 仅 T1 (phi=g1*h); include_t5 => T1+T3+T5; no_t2 => T1+T3 only
    include_t5 = getattr(args, "include_t5", False)
    t1_only = getattr(args, "t1_only", False)
    include_t2 = not args.no_t2 and not include_t5 and not t1_only
    include_t3 = not include_t5 and not t1_only
    cheb_layer = GatedChebyshevLayer(
        dim=dim,
        gate_mode=args.gate_mode,
        include_t2=include_t2,
        include_t3=include_t3,
        include_t5=include_t5,
        init_gates=(args.init_g1, args.init_g2, args.init_g3),
        init_g5=getattr(args, "init_g5", 0.1),
    ).to(device)

    n_cheb = sum(p.numel() for p in cheb_layer.parameters())
    n_proj = sum(p.numel() for p in projector.parameters()) if projector is not None else 0
    print(f"gate_mode={args.gate_mode}, Cheb params={n_cheb}, Projector params={n_proj}")

    params = list(cheb_layer.parameters())
    if projector is not None:
        params += list(projector.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = 1e9

    for epoch in range(args.epochs):
        if projector is not None:
            projector.train()
        cheb_layer.train()

        total_loss = 0.0
        total_nce = 0.0
        total_t2v = 0.0
        total_v2t = 0.0
        total_distill = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for text_b, video_b, vid_b in pbar:
            text_b = F.normalize(text_b.to(device), dim=-1)
            video_b = F.normalize(video_b.to(device), dim=-1)
            vid_b = vid_b.to(device)

            if projector is not None:
                h_t = projector(text_b)
                h_v = projector(video_b)
            else:
                h_t = torch.clamp(text_b, -1.0, 1.0)
                h_v = torch.clamp(video_b, -1.0, 1.0)

            phi_t = cheb_layer(h_t)
            phi_v = cheb_layer(h_v)

            sim = phi_t @ phi_v.T

            if args.loss_mode == "symmetric":
                nce = multi_positive_nce(sim, vid_b, temperature=args.temperature) + \
                    multi_positive_nce(sim.T, vid_b, temperature=args.temperature)
                loss_t2v = multi_positive_nce(sim, vid_b, temperature=args.temperature)
                loss_v2t = multi_positive_nce(sim.T, vid_b, temperature=args.temperature)
            else:
                nce, loss_t2v, loss_v2t = asymmetric_multi_positive_nce(
                    sim,
                    vid_b,
                    temperature=args.temperature,
                    t2v_weight=args.t2v_weight,
                    v2t_weight=args.v2t_weight,
                )

            if projector is not None and args.distill_weight > 0:
                dloss = distill_loss(h_t, text_b) + distill_loss(h_v, video_b)
            else:
                dloss = torch.tensor(0.0, device=device)

            loss = nce + args.distill_weight * dloss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_nce += nce.item()
            total_t2v += loss_t2v.item()
            total_v2t += loss_v2t.item()
            total_distill += dloss.item()

            gate_info = cheb_layer.gate_summary()
            postfix = {
                "loss": f"{loss.item():.4f}",
                "t2v": f"{loss_t2v.item():.4f}",
                "v2t": f"{loss_v2t.item():.4f}",
                "g1": f"{gate_info.get('g1', gate_info.get('g1_mean', 0)):.3f}",
                "g2": f"{gate_info.get('g2', gate_info.get('g2_mean', 0)):.3f}",
                "g3": f"{gate_info.get('g3', gate_info.get('g3_mean', 0)):.3f}",
            }
            if "g5" in gate_info or "g5_mean" in gate_info:
                postfix["g5"] = f"{gate_info.get('g5', gate_info.get('g5_mean', 0)):.3f}"
            pbar.set_postfix(postfix)

        avg_loss = total_loss / len(loader)
        avg_nce = total_nce / len(loader)
        avg_t2v = total_t2v / len(loader)
        avg_v2t = total_v2t / len(loader)
        avg_distill = total_distill / len(loader)

        print(
            f"Epoch {epoch+1}: loss={avg_loss:.4f}, "
            f"nce={avg_nce:.4f}, t2v={avg_t2v:.4f}, v2t={avg_v2t:.4f}, "
            f"distill={avg_distill:.4f}"
        )
        print("Gate summary:", cheb_layer.gate_summary())

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = {
                "epoch": epoch,
                "loss": best_loss,
                "config": vars(args),
                "cheb_layer": cheb_layer.state_dict(),
            }
            if projector is not None:
                ckpt["projector"] = projector.state_dict()

            torch.save(ckpt, Path(args.save_dir) / args.save_name)
            print(f"Saved best checkpoint: loss={best_loss:.4f}")

    print("Training done.")


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--emb_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)

    # projector
    parser.add_argument("--no_projector", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--no_residual", action="store_true")

    # gated chebyshev
    parser.add_argument("--gate_mode", choices=["scalar", "vector"], default="scalar")
    parser.add_argument("--no_t2", action="store_true",
                        help="Use only [g1*T1(h), g3*T3(h)] (no T2)")
    parser.add_argument("--t1_only", action="store_true",
                        help="Use only T1: phi=g1*h (no Cheb higher-order terms, to ablate)")
    parser.add_argument("--include_t5", action="store_true",
                        help="Add T5: [g1*T1, g3*T3, g5*T5] (implies odd-only, no T2)")
    parser.add_argument("--init_g1", type=float, default=1.0)
    parser.add_argument("--init_g2", type=float, default=0.1)
    parser.add_argument("--init_g3", type=float, default=0.1)
    parser.add_argument("--init_g5", type=float, default=0.1)

    # regularization
    parser.add_argument("--distill_weight", type=float, default=0.02)
    # loss mode
    parser.add_argument("--loss_mode", choices=["symmetric", "asymmetric"], default="asymmetric")
    # dropout rate
    parser.add_argument("--dropout", type=float, default=0.1)
    #seed   
    parser.add_argument("--seed", type=int, default=42)
    #save
    parser.add_argument("--save_name", type=str, default="best_cheb_gated_asym.pth")
    
    # asymmetric retrieval objective
    parser.add_argument("--t2v_weight", type=float, default=1.0)
    parser.add_argument("--v2t_weight", type=float, default=2.5)

    # ids
    parser.add_argument("--infer_vid_ids", action="store_true")
    parser.add_argument("--caps_per_video", type=int, default=20)

    args = parser.parse_args()
    train(args)