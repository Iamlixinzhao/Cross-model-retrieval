#!/usr/bin/env python3
"""Quick check: print gate_mode and param count from a gated checkpoint."""
import sys
import torch

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "sweep_runs/run_vector_t1t3/best_cheb_gated_asym.pth"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt.get("config") or {}
    gate_mode = config.get("gate_mode", "scalar")
    no_t2 = config.get("no_t2", True)
    include_t5 = config.get("include_t5", False)
    sd = ckpt.get("cheb_layer") or {}
    n_cheb = sum(v.numel() for v in sd.values())
    print(f"Path: {path}")
    print(f"  gate_mode = {gate_mode}")
    print(f"  no_t2 = {no_t2}, include_t5 = {include_t5}")
    print(f"  cheb_layer param count = {n_cheb} (scalar=>3 or 4, vector=>3*dim or 4*dim)")

if __name__ == "__main__":
    main()
