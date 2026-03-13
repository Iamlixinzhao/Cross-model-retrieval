#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Cheb-projected embeddings for CrossSim (CIM simulator) or other use.
Output: query vectors, candidate vectors, and ground-truth target indices (diagonal pairing).
Formats: JSON, CSV, or NPZ + meta JSON.

Usage:
  cd /mnt/pes/Cross-model-retrieval
  python export_cheb_for_cim.py \
    --emb_dir /mnt/pes/ImageBind/msrvtt_results \
    --cheb_ckpt ./ckpt_cheb_gated/best_cheb_gated_asym.pth \
    --out_dir ./cheb_export_cim \
    --format csv
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Reuse loader and model from measure script
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from measure_latency_memory_variance_cheb import (
    load_embeddings,
    load_gated_full_ckpt,
)


def build_phi(text_emb, video_emb, projector, cheb_layer, device):
    """Return (text_phi, video_phi) after normalize -> projector/clamp -> cheb_layer."""
    def _phi(x):
        x_in = F.normalize(x, dim=-1)
        if projector is not None:
            h = projector(x_in)
        else:
            h = torch.clamp(x_in, -1.0, 1.0)
        return cheb_layer(h)
    with torch.no_grad():
        text_emb = text_emb.to(device)
        video_emb = video_emb.to(device)
        text_phi = _phi(text_emb).cpu().float().numpy()
        video_phi = _phi(video_emb).cpu().float().numpy()
    return text_phi, video_phi


def export_json(out_dir: Path, text_phi, video_phi, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    # Full JSON: meta + vectors (can be large)
    data = {
        "meta": meta,
        "t2v": {
            "query_vectors": text_phi.tolist(),
            "candidate_vectors": video_phi.tolist(),
            "targets": [{"query_idx": i, "target_idx": i} for i in range(meta["n_samples"])],
        },
        "v2t": {
            "query_vectors": video_phi.tolist(),
            "candidate_vectors": text_phi.tolist(),
            "targets": [{"query_idx": i, "target_idx": i} for i in range(meta["n_samples"])],
        },
    }
    with open(out_dir / "embeddings_and_meta.json", "w") as f:
        json.dump(data, f, indent=2)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote {out_dir / 'embeddings_and_meta.json'} (meta + vectors)")
    print(f"  Wrote {meta_path}")


def export_csv(out_dir: Path, text_phi, video_phi, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    N, D = text_phi.shape
    # T2V: query = text, candidate = video
    with open(out_dir / "query_t2v.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_idx"] + [f"dim{i}" for i in range(D)])
        for i in range(N):
            w.writerow([i] + text_phi[i].tolist())
    with open(out_dir / "candidate_t2v.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_idx"] + [f"dim{i}" for i in range(D)])
        for i in range(N):
            w.writerow([i] + video_phi[i].tolist())
    with open(out_dir / "targets_t2v.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_idx", "target_idx"])
        for i in range(N):
            w.writerow([i, i])
    # V2T: query = video, candidate = text
    with open(out_dir / "query_v2t.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_idx"] + [f"dim{i}" for i in range(D)])
        for i in range(N):
            w.writerow([i] + video_phi[i].tolist())
    with open(out_dir / "candidate_v2t.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_idx"] + [f"dim{i}" for i in range(D)])
        for i in range(N):
            w.writerow([i] + text_phi[i].tolist())
    with open(out_dir / "targets_v2t.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_idx", "target_idx"])
        for i in range(N):
            w.writerow([i, i])
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote query_t2v.csv, candidate_t2v.csv, targets_t2v.csv (T2V)")
    print(f"  Wrote query_v2t.csv, candidate_v2t.csv, targets_v2t.csv (V2T)")
    print(f"  Wrote meta.json")


def export_npz(out_dir: Path, text_phi, video_phi, meta: dict):
    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "embeddings.npz",
        text_phi=text_phi,
        video_phi=video_phi,
    )
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    # Targets: diagonal
    targets = np.arange(meta["n_samples"], dtype=np.int64)
    np.save(out_dir / "targets.npy", targets)
    print(f"  Wrote embeddings.npz (text_phi, video_phi)")
    print(f"  Wrote targets.npy (target[i]=i for diagonal pairing)")
    print(f"  Wrote meta.json")
    print(f"  Usage: T2V query=text_phi, candidate=video_phi; V2T query=video_phi, candidate=text_phi.")


def main():
    ap = argparse.ArgumentParser(description="Export Cheb embeddings for CIM (CrossSim)")
    ap.add_argument("--emb_dir", type=str, default="/mnt/pes/ImageBind/msrvtt_results",
                    help="Directory with emb_text.pt, emb_video.pt")
    ap.add_argument("--cheb_ckpt", type=str,
                    default=str(SCRIPT_DIR / "ckpt_cheb_gated" / "best_cheb_gated_asym.pth"),
                    help="Path to gated Cheb checkpoint")
    ap.add_argument("--out_dir", type=str, default="./cheb_export_cim",
                    help="Output directory")
    ap.add_argument("--format", choices=["json", "csv", "npz"], default="csv",
                    help="Export format: json (one file), csv (per-vector files), npz (numpy)")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)

    # Resolve checkpoint path
    ckpt_path = args.cheb_ckpt
    if not os.path.exists(ckpt_path) and not os.path.isabs(ckpt_path):
        alt = SCRIPT_DIR / ckpt_path
        if alt.exists():
            ckpt_path = str(alt)

    print("Loading embeddings...")
    text_emb, video_emb = load_embeddings(args.emb_dir, device)
    N, D_input = text_emb.size(0), text_emb.size(1)
    print(f"  N={N}, D_input={D_input}")

    print(f"Loading gated Cheb model: {ckpt_path}")
    projector, cheb_layer = load_gated_full_ckpt(ckpt_path, D_input, device)
    if cheb_layer is None:
        raise FileNotFoundError(f"Not a gated checkpoint or missing: {ckpt_path}")
    print(f"  Projector: {'ON' if projector is not None else 'OFF'}")

    print("Forward pass (normalize -> projector/clamp -> Cheb layer)...")
    text_phi, video_phi = build_phi(text_emb, video_emb, projector, cheb_layer, device)
    N, D_phi = text_phi.shape
    print(f"  text_phi: {text_phi.shape}, video_phi: {video_phi.shape}")

    meta = {
        "n_samples": int(N),
        "d_input": int(D_input),
        "d_phi": int(D_phi),
        "pairing": "diagonal",
        "description": "query_i is matched to candidate_i (text_i <-> video_i). T2V: query=text, candidate=video. V2T: query=video, candidate=text.",
        "cheb_ckpt": ckpt_path,
        "emb_dir": args.emb_dir,
    }

    print(f"Exporting to {out_dir} ({args.format})...")
    if args.format == "json":
        export_json(out_dir, text_phi, video_phi, meta)
    elif args.format == "csv":
        export_csv(out_dir, text_phi, video_phi, meta)
    else:
        export_npz(out_dir, text_phi, video_phi, meta)

    print("Done.")


if __name__ == "__main__":
    main()
