import itertools
import json
import subprocess
from pathlib import Path

# =============================
# paths
# =============================

TRAIN_SCRIPT = "train_cheb_projector.py"
MEASURE_SCRIPT = "measure_latency_memory_variance_cheb.py"

EMB_DIR = "/mnt/data/pes/ImageBind/msrvtt_train_embeddings"
# For evaluation: use msrvtt_results (contains emb_text.pt / emb_video.pt) when msrvtt_test_embeddings is not available
EVAL_EMB_DIR = "/mnt/data/pes/ImageBind/msrvtt_results"

SAVE_ROOT = Path("./sweep_runs")
SAVE_ROOT.mkdir(exist_ok=True)

# =============================
# parameter grid
# =============================

# Only sweep configs with projector (no_projector fixed to False)
param_grid = {
    "loss_mode": ["symmetric", "asymmetric"],
    "v2t_weight": [1.0, 2.0, 2.5],
    "distill_weight": [0.0, 0.02, 0.05],
}

# Fixed args (aligned with train_cheb_projector)
# Aligned with ckpt_cheb_gated (V2T R@1≈38%): no_t2=True (T1+T3 only), projector, asymmetric
fixed_args = {
    "epochs": 20,
    "batch_size": 256,
    "lr": 1e-4,
    "temperature": 0.07,
    "infer_vid_ids": True,
    "caps_per_video": 20,
    "no_projector": False,
    "no_t2": True,   # Match 38% baseline: T1+T3 only, no T2
}
# Checkpoint filename saved by training; must match train script --save_name
CKPT_FILENAME = "best_cheb_gated_asym.pth"

# =============================
# helper
# =============================

def build_train_cmd(config, save_dir):

    cmd = [
        "python",
        TRAIN_SCRIPT,
        "--emb_dir", EMB_DIR,
        "--save_dir", str(save_dir),
        "--epochs", str(config["epochs"]),
        "--batch_size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--temperature", str(config["temperature"]),
        "--loss_mode", config["loss_mode"],
        "--v2t_weight", str(config["v2t_weight"]),
        "--distill_weight", str(config["distill_weight"]),
        "--caps_per_video", str(config["caps_per_video"]),
    ]
    if config["no_projector"]:
        cmd.append("--no_projector")
    if config.get("infer_vid_ids", False):
        cmd.append("--infer_vid_ids")
    if config.get("no_t2", False):
        cmd.append("--no_t2")

    return cmd


def build_measure_cmd(cheb_ckpt_path, output_json):
    """measure_latency_memory_variance_cheb.py uses --cheb_ckpt and --save."""
    cmd = [
        "python",
        MEASURE_SCRIPT,
        "--cheb_ckpt", str(cheb_ckpt_path),
        "--emb_dir", EVAL_EMB_DIR,
        "--save", str(output_json),
    ]
    return cmd


# =============================
# sweep loop
# =============================

keys = list(param_grid.keys())
values = list(param_grid.values())

all_runs = list(itertools.product(*values))
total = len(all_runs)
print(f"Total sweep runs: {total}")

results_summary = []

for run_id, combo in enumerate(all_runs):

    config = dict(zip(keys, combo))
    config.update(fixed_args)

    tag = "_".join(f"{k}{v}" for k,v in config.items() if k in param_grid)

    save_dir = SAVE_ROOT / f"run_{run_id}_{tag}"
    save_dir.mkdir(exist_ok=True)

    print("\n=================================")
    print(f"RUN {run_id + 1}/{total}")
    print(config)
    print("=================================")

    # -------------------------
    # train
    # -------------------------

    train_cmd = build_train_cmd(config, save_dir)

    print("TRAIN CMD:")
    print(" ".join(train_cmd))

    subprocess.run(train_cmd, check=True)

    # -------------------------
    # find checkpoint (training saves only one best ckpt)
    # -------------------------
    ckpt = save_dir / CKPT_FILENAME
    if not ckpt.exists():
        raise FileNotFoundError(f"Expected checkpoint {ckpt} not found after training.")

    # -------------------------
    # measure
    # -------------------------

    output_json = save_dir / "metrics.json"

    measure_cmd = build_measure_cmd(ckpt, output_json)

    print("MEASURE CMD:")
    print(" ".join(measure_cmd))

    subprocess.run(measure_cmd, check=True)

    # -------------------------
    # collect results
    # -------------------------

    with open(output_json) as f:
        metrics = json.load(f)

    results_summary.append({
        "config": config,
        "metrics": metrics
    })

# =============================
# save global summary
# =============================

summary_path = SAVE_ROOT / "summary.json"

with open(summary_path, "w") as f:
    json.dump(results_summary, f, indent=2)

print("\nSaved summary to", summary_path)