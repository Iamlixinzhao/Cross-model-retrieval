#!/usr/bin/env python3
"""
Batch testing the impact of training epochs on Poly retrieval performance

Fix poly degree (recommend 5), test different epochs (e.g., 10, 15, 20, 25, 30).
Find the optimal epochs to avoid underfitting or overfitting.
"""

import argparse
import subprocess
import json
import csv
from pathlib import Path
import time


def run_command(cmd, description):
    """Run command and display progress"""
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


def sweep_epochs(args):
    """Batch test different epochs"""
    
    results = []
    
    # Use fixed poly degree
    poly_path = Path(args.poly_path)
    if not poly_path.exists():
        print(f"❌ Error: poly_path not found: {poly_path}")
        print("Generate it first:")
        print(f"  python train_poly_projector.py fit_poly --teacher_npz {args.teacher_npz} --out {poly_path} --degree {args.degree}")
        return
    
    print(f"Using poly: {poly_path} (degree={args.degree})")
    
    for epochs in args.epochs_list:
        print(f"\n\n{'#'*80}")
        print(f"# Testing Epochs: {epochs}")
        print(f"{'#'*80}\n")
        
        # Define output paths
        ckpt_dir = Path(args.output_dir) / f"poly_ep{epochs}"
        result_json = Path(args.output_dir) / f"results_ep{epochs}.json"
        
        # Step 1: Train projector
        if not (ckpt_dir / "best_projectors_eq4_poly.pth").exists() or args.retrain:
            train_cmd = [
                "python", "train_poly_projector.py", "train_poly",
                "--poly_path", str(poly_path),
                "--save_dir", str(ckpt_dir),
                "--epochs", str(epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--var_reg_type", args.var_reg_type,
                "--var_reg_weight", str(args.var_reg_weight),
            ]
            
            if not run_command(train_cmd, f"Train projector (epochs={epochs})"):
                print(f"⚠️  Skipping epochs {epochs} due to training failure")
                continue
        else:
            print(f"✓ Using existing checkpoint: {ckpt_dir}")
        
        # Step 2: Evaluate
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
        
        if not run_command(eval_cmd, f"Evaluate (epochs={epochs})"):
            print(f"⚠️  Skipping epochs {epochs} due to evaluation failure")
            continue
        
        # Parse results
        try:
            with open(result_json) as f:
                res = json.load(f)
            
            # Extract from the actual JSON structure
            summary = res.get("summary", {})
            retrieval = summary.get("retrieval", {}).get("projector", {})
            projector_perf = summary.get("projector", {})
            
            result = {
                "epochs": epochs,
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
            
            print(f"\n📊 Results for epochs={epochs}:")
            print(f"  T2V R@1: {result['t2v_r1']:.2f}%")
            print(f"  V2T R@1: {result['v2t_r1']:.2f}%")
            
        except Exception as e:
            print(f"⚠️  Failed to parse results for epochs {epochs}: {e}")
            continue
    
    # Save summary CSV
    if results:
        csv_path = Path(args.output_dir) / "epochs_sweep_summary.csv"
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ["epochs", "t2v_r1", "v2t_r1", "t2v_r5", "v2t_r5", "t2v_r10", "v2t_r10",
                          "t2v_medr", "v2t_medr", "latency_ms", "gpu_mb"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n\n{'='*80}")
        print(f"✅ Epochs Sweep Complete! Results saved to: {csv_path}")
        print(f"{'='*80}\n")
        
        # Print summary table
        print("\nSummary Table:")
        print(f"{'Epochs':<8} {'T2V R@1':<10} {'V2T R@1':<10} {'T2V R@5':<10} {'T2V MedR':<10}")
        print('-' * 60)
        for r in results:
            print(f"{r['epochs']:<8} {r['t2v_r1']:<10.2f} {r['v2t_r1']:<10.2f} "
                  f"{r['t2v_r5']:<10.2f} {r['t2v_medr']:<10.1f}")
        
        # Find best
        best = max(results, key=lambda x: x['t2v_r1'])
        print(f"\n🏆 Best epochs: {best['epochs']} (T2V R@1 = {best['t2v_r1']:.2f}%)")
        
        # Check for overfitting
        if len(results) >= 2:
            sorted_by_epochs = sorted(results, key=lambda x: x['epochs'])
            last = sorted_by_epochs[-1]
            second_last = sorted_by_epochs[-2] if len(sorted_by_epochs) > 1 else None
            
            if second_last and last['t2v_r1'] < second_last['t2v_r1'] - 0.5:
                print(f"\n⚠️  Warning: Performance dropped at {last['epochs']} epochs")
                print(f"   (from {second_last['t2v_r1']:.2f}% to {last['t2v_r1']:.2f}%)")
                print(f"   This suggests overfitting. Consider using {second_last['epochs']} epochs.")
    else:
        print("\n❌ No successful runs!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep epochs hyperparameter for Poly")
    
    # Required
    parser.add_argument("--poly_path", type=str, required=True,
                        help="Path to poly_coeffs.pt (fixed poly)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save all outputs")
    
    # Sweep range
    parser.add_argument("--epochs_list", type=int, nargs="+", default=[10, 15, 20, 25, 30],
                        help="List of epochs to test (default: 10 15 20 25 30)")
    parser.add_argument("--degree", type=int, default=5,
                        help="Poly degree (for display only, should match poly_path)")
    
    # Optional: generate poly if not exists
    parser.add_argument("--teacher_npz", type=str, default="teacher.npz",
                        help="Teacher NPZ (for reference in error messages)")
    
    # Train params
    parser.add_argument("--emb_dir", type=str, 
                        default="/mnt/pes/ImageBind/msrvtt_train_embeddings",
                        help="Training embeddings directory")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--var_reg_type", type=str, default="kl",
                        choices=["none", "kl", "upper_bound"])
    parser.add_argument("--var_reg_weight", type=float, default=0.001)
    parser.add_argument("--retrain", action="store_true",
                        help="Re-train even if checkpoint exists")
    
    # Evaluation params
    parser.add_argument("--test_emb_dir", type=str,
                        default="/mnt/pes/ImageBind/msrvtt_results",
                        help="Test embeddings directory")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of evaluation runs")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup runs")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Epochs Sweep Configuration")
    print(f"{'='*80}")
    print(f"Poly path:       {args.poly_path}")
    print(f"Poly degree:     {args.degree}")
    print(f"Output Dir:      {args.output_dir}")
    print(f"Epochs to test:  {args.epochs_list}")
    print(f"Re-train:        {args.retrain}")
    print(f"{'='*80}\n")
    
    sweep_epochs(args)
