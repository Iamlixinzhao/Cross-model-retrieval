#!/usr/bin/env python3
"""
Batch testing the impact of polynomial degree on retrieval performance

Using an existing teacher.npz, test different poly degrees.
For each degree:
  1. fit_poly: Fit polynomial coefficients
  2. train_poly: Train projector
  3. measure: Evaluate retrieval performance

Results are saved to a CSV file for easy comparison.
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


def eval_imagebind(args, output_dir):
    """Evaluate ImageBind baseline (for comparison)"""
    print(f"\n\n{'#'*80}")
    print(f"# Evaluating ImageBind Baseline (for comparison)")
    print(f"{'#'*80}\n")
    
    imagebind_result_json = Path(output_dir) / "results_imagebind.json"
    
    # Evaluate ImageBind (no training needed)
    eval_cmd = [
        "python", "measure_latency_memory_variance.py",
        "--emb_dir", args.test_emb_dir,
        "--k_list", "1", "5", "10",
        "--runs", str(args.runs),
        "--warmup", str(args.warmup),
        "--save", str(imagebind_result_json),
    ]
    
    if not run_command(eval_cmd, "Evaluate ImageBind baseline"):
        print("⚠️  ImageBind evaluation failed")
        return None
    
    # Parse ImageBind results
    try:
        with open(imagebind_result_json) as f:
            res = json.load(f)
        
        summary = res.get("summary", {})
        retrieval = summary.get("retrieval", {}).get("imagebind", {})
        imagebind_perf = summary.get("imagebind", {})
        
        imagebind_result = {
            "model": "ImageBind",
            "degree": "N/A",
            "t2v_r1": retrieval["t2v"]["R@k"]["1"],
            "t2v_r5": retrieval["t2v"]["R@k"]["5"],
            "t2v_r10": retrieval["t2v"]["R@k"]["10"],
            "t2v_medr": retrieval["t2v"]["MedR"],
            "v2t_r1": retrieval["v2t"]["R@k"]["1"],
            "v2t_r5": retrieval["v2t"]["R@k"]["5"],
            "v2t_r10": retrieval["v2t"]["R@k"]["10"],
            "v2t_medr": retrieval["v2t"]["MedR"],
            "latency_ms": imagebind_perf.get("latency_mean_ms", 0),
            "gpu_mb": imagebind_perf.get("gpu_mean_mb", 0),
        }
        
        print(f"\n✅ ImageBind Baseline Results:")
        print(f"  T2V R@1: {imagebind_result['t2v_r1']:.2f}%")
        print(f"  V2T R@1: {imagebind_result['v2t_r1']:.2f}%")
        
        return imagebind_result
    
    except Exception as e:
        print(f"⚠️  Failed to parse ImageBind results: {e}")
        return None


def train_and_eval_pcme(args, output_dir):
    """Train and evaluate PCME baseline (for comparison)"""
    print(f"\n\n{'#'*80}")
    print(f"# Training PCME Baseline (for comparison)")
    print(f"# Using PCME-specific hyperparameters")
    print(f"{'#'*80}\n")
    
    pcme_ckpt_dir = Path(output_dir) / "pcme_baseline"
    pcme_result_json = Path(output_dir) / "results_pcme.json"
    
    # Train PCME with its own optimal hyperparameters
    if not (pcme_ckpt_dir / "best_projectors.pth").exists() or args.retrain:
        train_cmd = [
            "python", "train_pcme_projector.py",
            "--emb_dir", args.emb_dir,
            "--save_dir", str(pcme_ckpt_dir),
            "--epochs", str(args.pcme_epochs),  # Use PCME-specific epochs
            "--batch_size", str(args.batch_size),
            "--lr", str(args.pcme_lr),  # Use PCME-specific learning rate
            "--loss_type", "pcme_mc",
            "--n_samples", "5",
            "--var_reg_type", args.var_reg_type,
            "--var_reg_weight", str(args.var_reg_weight),
        ]
        
        if not run_command(train_cmd, "Train PCME baseline"):
            print("⚠️  PCME training failed, but continuing with Poly sweep...")
            return None
    else:
        print(f"✓ Using existing PCME checkpoint: {pcme_ckpt_dir}")
    
    # Evaluate PCME
    eval_cmd = [
        "python", "measure_latency_memory_variance.py",
        "--emb_dir", args.test_emb_dir,
        "--ckpt", str(pcme_ckpt_dir / "best_projectors.pth"),
        "--num_samples", "15",
        "--k_list", "1", "5", "10",
        "--runs", str(args.runs),
        "--warmup", str(args.warmup),
        "--save", str(pcme_result_json),
    ]
    
    if not run_command(eval_cmd, "Evaluate PCME baseline"):
        print("⚠️  PCME evaluation failed")
        return None
    
    # Parse PCME results
    try:
        with open(pcme_result_json) as f:
            res = json.load(f)
        
        summary = res.get("summary", {})
        retrieval = summary.get("retrieval", {}).get("projector", {})
        projector_perf = summary.get("projector", {})
        
        pcme_result = {
            "model": "PCME",
            "degree": "N/A",
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
        
        print(f"\n📊 PCME Baseline Results:")
        print(f"  T2V R@1: {pcme_result['t2v_r1']:.2f}%")
        print(f"  V2T R@1: {pcme_result['v2t_r1']:.2f}%")
        
        return pcme_result
    
    except Exception as e:
        print(f"⚠️  Failed to parse PCME results: {e}")
        return None


def sweep_poly_degree(args):
    """Batch test different poly degrees"""
    
    results = []
    
    # First, evaluate ImageBind baseline (if requested)
    if args.include_imagebind:
        imagebind_result = eval_imagebind(args, args.output_dir)
        if imagebind_result:
            results.append(imagebind_result)
    
    # Then, train and evaluate PCME baseline (if requested)
    if args.include_pcme:
        pcme_result = train_and_eval_pcme(args, args.output_dir)
        if pcme_result:
            results.append(pcme_result)
    
    for degree in args.degrees:
        print(f"\n\n{'#'*80}")
        print(f"# Testing Poly Degree: {degree}")
        print(f"{'#'*80}\n")
        
        # Define output paths
        poly_path = Path(args.output_dir) / f"poly_coeffs_deg{degree}.pt"
        ckpt_dir = Path(args.output_dir) / f"poly_ckpt_deg{degree}"
        result_json = Path(args.output_dir) / f"results_deg{degree}.json"
        
        # Step 1: Fit polynomial
        if not poly_path.exists() or args.refit:
            fit_cmd = [
                "python", "train_poly_projector.py", "fit_poly",
                "--teacher_npz", args.teacher_npz,
                "--out", str(poly_path),
                "--degree", str(degree),
                "--alpha", str(args.alpha),
            ]
            
            if not run_command(fit_cmd, f"Fit poly (degree={degree})"):
                print(f"⚠️  Skipping degree {degree} due to fit_poly failure")
                continue
        else:
            print(f"✓ Using existing poly: {poly_path}")
        
        # Step 2: Train projector
        if not (ckpt_dir / "best_projectors_eq4_poly.pth").exists() or args.retrain:
            train_cmd = [
                "python", "train_poly_projector.py", "train_poly",
                "--poly_path", str(poly_path),
                "--save_dir", str(ckpt_dir),
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--var_reg_type", args.var_reg_type,
                "--var_reg_weight", str(args.var_reg_weight),
            ]
            
            if not run_command(train_cmd, f"Train projector (degree={degree})"):
                print(f"⚠️  Skipping degree {degree} due to train_poly failure")
                continue
        else:
            print(f"✓ Using existing checkpoint: {ckpt_dir}")
        
        # Step 3: Evaluate
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
        
        if not run_command(eval_cmd, f"Evaluate (degree={degree})"):
            print(f"⚠️  Skipping degree {degree} due to evaluation failure")
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
                "model": "Poly",
                "degree": degree,
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
            
            print(f"\n📊 Results for degree={degree}:")
            print(f"  T2V R@1: {result['t2v_r1']:.2f}%")
            print(f"  V2T R@1: {result['v2t_r1']:.2f}%")
            print(f"  Latency: {result['latency_ms']:.2f}ms")
            
        except Exception as e:
            print(f"⚠️  Failed to parse results for degree {degree}: {e}")
            continue
    
    # Save summary CSV
    if results:
        csv_path = Path(args.output_dir) / "sweep_summary.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n\n{'='*80}")
        print(f"✅ Sweep Complete! Results saved to: {csv_path}")
        print(f"{'='*80}\n")
        
        # Sort results: ImageBind first, then PCME, then Poly by degree
        sorted_results = []
        for r in results:
            if r['model'] == 'ImageBind':
                sorted_results.insert(0, r)
            elif r['model'] == 'PCME':
                if sorted_results and sorted_results[0]['model'] == 'ImageBind':
                    sorted_results.insert(1, r)
                else:
                    sorted_results.insert(0, r)
            else:
                sorted_results.append(r)
        
        # Print summary table
        print("\nSummary Table:")
        print(f"{'Model':<12} {'Degree':<8} {'T2V R@1':<10} {'V2T R@1':<10} {'T2V R@5':<10} {'V2T R@5':<10}")
        print('-' * 70)
        for r in sorted_results:
            degree_str = str(r['degree']) if r['degree'] != 'N/A' else 'N/A'
            print(f"{r['model']:<12} {degree_str:<8} {r['t2v_r1']:<10.2f} {r['v2t_r1']:<10.2f} "
                  f"{r['t2v_r5']:<10.2f} {r['v2t_r5']:<10.2f}")
        
        # Find best Poly
        poly_results = [r for r in results if r['model'] == 'Poly']
        if poly_results:
            best_poly = max(poly_results, key=lambda x: x['t2v_r1'])
            print(f"\n🏆 Best Poly degree: {best_poly['degree']} (T2V R@1 = {best_poly['t2v_r1']:.2f}%)")
            
            # Compare with baselines
            imagebind_results = [r for r in results if r['model'] == 'ImageBind']
            pcme_results = [r for r in results if r['model'] == 'PCME']
            
            if imagebind_results:
                ib = imagebind_results[0]
                gap_ib = best_poly['t2v_r1'] - ib['t2v_r1']
                print(f"\n📊 vs ImageBind: T2V R@1 {best_poly['t2v_r1']:.2f}% vs {ib['t2v_r1']:.2f}%")
                if gap_ib >= 0:
                    print(f"   Poly is {gap_ib:.2f}% better ✅")
                else:
                    print(f"   Poly is {-gap_ib:.2f}% worse ⚠️")
            
            if pcme_results:
                pcme = pcme_results[0]
                gap_pcme = best_poly['t2v_r1'] - pcme['t2v_r1']
                print(f"\n📊 vs PCME: T2V R@1 {best_poly['t2v_r1']:.2f}% vs {pcme['t2v_r1']:.2f}%")
                if gap_pcme >= 0:
                    print(f"   Poly is {gap_pcme:.2f}% better ✅")
                else:
                    print(f"   Poly is {-gap_pcme:.2f}% worse ⚠️")
                    pct = (best_poly['t2v_r1'] / pcme['t2v_r1']) * 100
                    print(f"   Poly achieves {pct:.1f}% of PCME performance")
    else:
        print("\n❌ No successful runs!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep poly degree hyperparameter")
    
    # Required
    parser.add_argument("--teacher_npz", type=str, required=True,
                        help="Path to teacher.npz (pre-generated)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save all outputs")
    
    # Sweep range
    parser.add_argument("--degrees", type=int, nargs="+", default=[3, 4, 5, 6],
                        help="List of poly degrees to test (default: 3 4 5 6)")
    parser.add_argument("--include_imagebind", action="store_true",
                        help="Also evaluate ImageBind baseline for comparison")
    parser.add_argument("--include_pcme", action="store_true",
                        help="Also train and evaluate PCME baseline for comparison")
    
    # Fit poly params
    parser.add_argument("--alpha", type=float, default=1e-3,
                        help="Ridge alpha for fit_poly")
    parser.add_argument("--refit", action="store_true",
                        help="Re-fit poly even if coeffs exist")
    
    # Train poly params
    parser.add_argument("--emb_dir", type=str, 
                        default="/mnt/pes/ImageBind/msrvtt_train_embeddings",
                        help="Training embeddings directory")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs for Poly (default: 10)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate for Poly (default: 5e-6)")
    parser.add_argument("--var_reg_type", type=str, default="kl",
                        choices=["none", "kl", "upper_bound"])
    parser.add_argument("--var_reg_weight", type=float, default=0.001)
    parser.add_argument("--retrain", action="store_true",
                        help="Re-train projector even if checkpoint exists")
    
    # PCME-specific params (independent of Poly)
    parser.add_argument("--pcme_epochs", type=int, default=40,
                        help="Training epochs for PCME baseline (default: 40)")
    parser.add_argument("--pcme_lr", type=float, default=1e-5,
                        help="Learning rate for PCME baseline (default: 1e-5)")
    
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
    print(f"Poly Degree Sweep Configuration")
    print(f"{'='*80}")
    print(f"Teacher NPZ:       {args.teacher_npz}")
    print(f"Output Dir:        {args.output_dir}")
    print(f"Degrees to test:   {args.degrees}")
    print(f"Include ImageBind: {args.include_imagebind}")
    print(f"Include PCME:      {args.include_pcme}")
    print(f"Re-fit poly:       {args.refit}")
    print(f"Re-train:          {args.retrain}")
    print(f"\n--- Poly Training Params ---")
    print(f"Epochs:            {args.epochs}")
    print(f"Learning Rate:     {args.lr}")
    print(f"Var Reg Weight:    {args.var_reg_weight}")
    if args.include_pcme:
        print(f"\n--- PCME Training Params ---")
        print(f"Epochs:            {args.pcme_epochs}")
        print(f"Learning Rate:     {args.pcme_lr}")
        print(f"Var Reg Weight:    {args.var_reg_weight}")
    print(f"{'='*80}\n")
    
    # Check teacher.npz exists
    if not Path(args.teacher_npz).exists():
        print(f"❌ Error: teacher.npz not found at {args.teacher_npz}")
        print("Run build_teacher first:")
        print("  python train_poly_projector.py build_teacher --emb_dir ... --out teacher.npz")
        exit(1)
    
    sweep_poly_degree(args)
