#!/usr/bin/env python3
"""
sweep hyperparameters
-learning rate
- variance regularization weight
- batch size
"""

import argparse
import subprocess
import json
import csv
from pathlib import Path
import time
import itertools


def run_command(cmd, description):
    """run command and show progress"""
    print(f"\n{'='*80}")
    print(f"[{description}]")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        return False
    
    print(f"\n✅ Done: {description} (took {elapsed:.1f}s)")
    return True


def sweep_hyperparams(args):
    """sweep different hyperparameters"""
    
    results = []
    
    # check poly file
    poly_path = Path(args.poly_path)
    if not poly_path.exists():
        print(f"❌ Error: poly_path not found: {poly_path}")
        return
    
    print(f"Using poly: {poly_path} (degree={args.degree})")
    print(f"Fixed epochs: {args.epochs}")
    
    # generate all hyperparameter combinations
    param_combinations = list(itertools.product(
        args.lr_list,
        args.var_reg_weight_list,
        args.batch_size_list if args.batch_size_list else [args.batch_size]
    ))
    
    total_runs = len(param_combinations)
    print(f"\nTotal combinations to test: {total_runs}")
    print(f"Estimated time: {total_runs * 8 / 60:.1f} hours (assuming ~8 min per run)\n")
    
    for idx, (lr, var_reg_weight, batch_size) in enumerate(param_combinations, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Run {idx}/{total_runs}: lr={lr}, var_reg_weight={var_reg_weight}, batch_size={batch_size}")
        print(f"{'#'*80}\n")
        
        # define output path
        run_name = f"lr{lr}_vreg{var_reg_weight}_bs{batch_size}"
        ckpt_dir = Path(args.output_dir) / run_name
        result_json = Path(args.output_dir) / f"results_{run_name}.json"
        
        # step 1: train projector
        if not (ckpt_dir / "best_projectors_eq4_poly.pth").exists() or args.retrain:
            train_cmd = [
                "python", "train_poly_projector.py", "train_poly",
                "--poly_path", str(poly_path),
                "--save_dir", str(ckpt_dir),
                "--epochs", str(args.epochs),
                "--batch_size", str(batch_size),
                "--lr", str(lr),
                "--var_reg_type", args.var_reg_type,
                "--var_reg_weight", str(var_reg_weight),
            ]
            
            if not run_command(train_cmd, f"Train ({run_name})"):
                print(f"⚠️  Skipping {run_name} due to training failure")
                continue
        else:
            print(f"✓ Using existing checkpoint: {ckpt_dir}")
        
        # step 2: evaluate
        eval_cmd = [
            "python", "measure_latency_memory_variance.py",
            "--emb_dir", args.test_emb_dir,
            "--ckpt", str(ckpt_dir / "best_projectors_eq4_poly.pth"),
            "--poly_path", str(poly_path),
            "--k_list", "1", "5", "10",
            "--runs", str(args.runs),
            "--warmup", str(args.warmup),
            "--save", str(result_json),
        ]
        
        if not run_command(eval_cmd, f"Evaluate ({run_name})"):
            print(f"⚠️  Skipping {run_name} due to evaluation failure")
            continue
        
        # parse results
        try:
            with open(result_json) as f:
                res = json.load(f)
            
            summary = res.get("summary", {})
            retrieval = summary.get("retrieval", {}).get("projector", {})
            projector_perf = summary.get("projector", {})
            
            result = {
                "lr": lr,
                "var_reg_weight": var_reg_weight,
                "batch_size": batch_size,
                "t2v_r1": retrieval["t2v"]["R@k"]["1"],
                "t2v_r5": retrieval["t2v"]["R@k"]["5"],
                "t2v_r10": retrieval["t2v"]["R@k"]["10"],
                "t2v_medr": retrieval["t2v"]["MedR"],
                "v2t_r1": retrieval["v2t"]["R@k"]["1"],
                "v2t_r5": retrieval["v2t"]["R@k"]["5"],
                "v2t_r10": retrieval["v2t"]["R@k"]["10"],
                "v2t_medr": retrieval["v2t"]["MedR"],
                "latency_ms": projector_perf.get("latency_mean_ms", 0),
                "gpu_mb": projector_perf.get("gpu_mean_mb", 0),
            }
            results.append(result)
            
            print(f"\n📊 Results:")
            print(f"  T2V R@1: {result['t2v_r1']:.2f}%")
            print(f"  V2T R@1: {result['v2t_r1']:.2f}%")
            
        except Exception as e:
            print(f"⚠️  Failed to parse results for {run_name}: {e}")
            continue
    
    # save summary CSV
    if results:
        csv_path = Path(args.output_dir) / "hyperparam_sweep_summary.csv"
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ["lr", "var_reg_weight", "batch_size", 
                          "t2v_r1", "v2t_r1", "t2v_r5", "v2t_r5", "t2v_r10", "v2t_r10",
                          "t2v_medr", "v2t_medr", "latency_ms", "gpu_mb"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n\n{'='*80}")
        print(f"✅ Hyperparam Sweep Complete! Results saved to: {csv_path}")
        print(f"{'='*80}\n")
        
        # print summary table (top 10)
        print("\nTop 10 Configurations (by T2V R@1):")
        print(f"{'Rank':<5} {'LR':<10} {'VarReg':<10} {'BS':<5} {'T2V R@1':<10} {'V2T R@1':<10}")
        print('-' * 70)
        
        sorted_results = sorted(results, key=lambda x: x['t2v_r1'], reverse=True)
        for rank, r in enumerate(sorted_results[:10], 1):
            print(f"{rank:<5} {r['lr']:<10.1e} {r['var_reg_weight']:<10.1e} "
                  f"{r['batch_size']:<5} {r['t2v_r1']:<10.2f} {r['v2t_r1']:<10.2f}")
        
        # best overall
        best = sorted_results[0]
        print(f"\n🏆 Best configuration:")
        print(f"   LR: {best['lr']}")
        print(f"   Var Reg Weight: {best['var_reg_weight']}")
        print(f"   Batch Size: {best['batch_size']}")
        print(f"   T2V R@1: {best['t2v_r1']:.2f}%")
        print(f"   V2T R@1: {best['v2t_r1']:.2f}%")
        
        # analysis: learning rate sensitivity
        print(f"\n📈 Learning Rate Analysis:")
        lr_groups = {}
        for r in results:
            lr = r['lr']
            if lr not in lr_groups:
                lr_groups[lr] = []
            lr_groups[lr].append(r['t2v_r1'])
        
        for lr in sorted(lr_groups.keys()):
            avg_r1 = sum(lr_groups[lr]) / len(lr_groups[lr])
            print(f"   LR {lr:.1e}: avg T2V R@1 = {avg_r1:.2f}%")
        
        # analysis: variance regularization sensitivity
        print(f"\n📉 Variance Reg Analysis:")
        vreg_groups = {}
        for r in results:
            vreg = r['var_reg_weight']
            if vreg not in vreg_groups:
                vreg_groups[vreg] = []
            vreg_groups[vreg].append(r['t2v_r1'])
        
        for vreg in sorted(vreg_groups.keys()):
            avg_r1 = sum(vreg_groups[vreg]) / len(vreg_groups[vreg])
            print(f"   Var Reg {vreg:.1e}: avg T2V R@1 = {avg_r1:.2f}%")
        
    else:
        print("\n❌ No successful runs!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep hyperparameters for Poly")
    
    # Required
    parser.add_argument("--poly_path", type=str, required=True,
                        help="Path to poly_coeffs.pt")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save all outputs")
    
    # Fixed params
    parser.add_argument("--epochs", type=int, default=10,
                        help="Fixed epochs (default: 10, fast and avoid overfitting)")
    parser.add_argument("--degree", type=int, default=5,
                        help="Poly degree (for display only)")
    
    # Sweep ranges
    parser.add_argument("--lr_list", type=float, nargs="+", 
                        default=[5e-6, 1e-5, 2e-5, 5e-5],
                        help="List of learning rates to test")
    parser.add_argument("--var_reg_weight_list", type=float, nargs="+",
                        default=[0.0005, 0.001, 0.002, 0.005],
                        help="List of var reg weights to test")
    parser.add_argument("--batch_size_list", type=int, nargs="+",
                        default=None,
                        help="List of batch sizes to test (default: only use --batch_size)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Fixed batch size if --batch_size_list not provided")
    
    # Train params
    parser.add_argument("--emb_dir", type=str, 
                        default="/mnt/pes/ImageBind/msrvtt_train_embeddings",
                        help="Training embeddings directory")
    parser.add_argument("--var_reg_type", type=str, default="kl",
                        choices=["none", "kl", "upper_bound"])
    parser.add_argument("--retrain", action="store_true",
                        help="Re-train even if checkpoint exists")
    
    # Evaluation params
    parser.add_argument("--test_emb_dir", type=str,
                        default="/mnt/pes/ImageBind/msrvtt_results",
                        help="Test embeddings directory")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    
    # optional: generate teacher NPZ if not exists
    parser.add_argument("--teacher_npz", type=str, default="teacher.npz",
                        help="Teacher NPZ (for reference)")
    
    args = parser.parse_args()
    
    # create output directory if not exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Hyperparameter Sweep Configuration")
    print(f"{'='*80}")
    print(f"Poly path:       {args.poly_path}")
    print(f"Poly degree:     {args.degree}")
    print(f"Output Dir:      {args.output_dir}")
    print(f"Fixed epochs:    {args.epochs}")
    print(f"LR range:        {args.lr_list}")
    print(f"Var Reg range:   {args.var_reg_weight_list}")
    print(f"Batch size(s):   {args.batch_size_list if args.batch_size_list else [args.batch_size]}")
    print(f"Re-train:        {args.retrain}")
    print(f"{'='*80}\n")
    
    sweep_hyperparams(args)
