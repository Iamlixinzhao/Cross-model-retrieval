#!/usr/bin/env python3
"""
Run full VATEX pipeline on /data2 (mapped to /mnt/data on this server):
1) download VATEX videos to data disk using KaggleHub
2) build HF dataset layout and run prepare_video_text_dataset.py
3) generate ImageBind embeddings (train/test)
4) baseline ImageBind + PCME train/eval
5) Chebyshev degree sweep (3/4/5/6) train/eval
6) aggregate results table

This script is resumable: each stage writes a marker in <root>/state.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path):
    print("\n[cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def mark_done(state_dir: Path, stage: str):
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / f"{stage}.done").write_text("ok\n", encoding="utf-8")


def is_done(state_dir: Path, stage: str) -> bool:
    return (state_dir / f"{stage}.done").exists()


def find_videos_dir(root: Path) -> Path:
    candidates = [
        root / "vatex" / "videos",
        root / "videos",
    ]
    for c in candidates:
        if c.exists():
            return c
    for c in root.rglob("videos"):
        if c.is_dir():
            return c
    raise FileNotFoundError(f"Could not find videos directory under {root}")


def build_combined_videos_dir(dst_dir: Path, src_dirs: list[Path]) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    # Clean old symlinks/files if rerun.
    for p in dst_dir.glob("*.mp4"):
        p.unlink()
    seen = set()
    for src in src_dirs:
        for vid in src.glob("*.mp4"):
            if vid.name in seen:
                continue
            (dst_dir / vid.name).symlink_to(vid)
            seen.add(vid.name)
    return dst_dir


def build_hf_json_from_repo_pairs(repo_dir: Path, vatex_root: Path, videos_dir: Path):
    from datasets import Dataset, DatasetDict

    src = repo_dir / "dataset_splits" / "vatex_full"
    train_path = src / "vatex_train_pairs.json"
    val_path = src / "vatex_val_pairs.json"
    test_path = src / "vatex_test_pairs.json"
    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing source pair file: {p}")

    def load_pairs(path: Path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)["pairs"]

    def to_rows(pairs):
        rows = []
        miss = 0
        for p in pairs:
            vid = str(p["video_id"])
            cap = str(p["caption"]).strip()
            if not cap:
                continue
            if not (videos_dir / f"{vid}.mp4").exists():
                miss += 1
                continue
            rows.append({"videoID": vid, "enCap": [cap]})
        return rows, miss

    tr_rows, tr_miss = to_rows(load_pairs(train_path))
    va_rows, va_miss = to_rows(load_pairs(val_path))
    te_rows, te_miss = to_rows(load_pairs(test_path))

    print(f"[hf-json] rows train={len(tr_rows)} (miss={tr_miss})")
    print(f"[hf-json] rows val={len(va_rows)} (miss={va_miss})")
    print(f"[hf-json] rows test={len(te_rows)} (miss={te_miss})")
    if len(tr_rows) == 0 or len(te_rows) == 0:
        raise RuntimeError(
            "Insufficient VATEX rows after video filtering. "
            "Need non-empty train/test. Check downloaded datasets."
        )
    if len(va_rows) == 0:
        # Some VATEX Kaggle mirrors do not provide the official val videos.
        # Keep one valid row so HF load_from_disk works without empty-shard errors.
        va_rows = tr_rows[:1]
        print("[hf-json] validation split empty; using 1 train row as placeholder.")

    ds = DatasetDict(
        {
            "train": Dataset.from_list(tr_rows),
            "validation": Dataset.from_list(va_rows),
            "public_test": Dataset.from_list(te_rows),
        }
    )

    json_root = vatex_root / "json"
    if json_root.exists():
        # keep old if present but ensure overwritten with latest schema
        import shutil

        shutil.rmtree(json_root)
    ds.save_to_disk(str(json_root))
    print(f"[hf-json] saved to {json_root}")


def main():
    ap = argparse.ArgumentParser(description="Run full VATEX benchmark pipeline")
    ap.add_argument("--root", type=str, default="/data2/vatex_experiments")
    ap.add_argument("--repo_dir", type=str, default=str(Path(__file__).resolve().parent))
    ap.add_argument("--download_dataset", type=str, default="khaledatef1/vatex0110")
    ap.add_argument("--download_dataset_test", type=str, default="khaledatef1/vatex011011")
    ap.add_argument("--pcme_epochs", type=int, default=40)
    ap.add_argument("--cheb_epochs", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    repo_dir = Path(args.repo_dir).resolve()
    state = root / "state"
    logs = root / "logs"
    datasets_dir = root / "datasets"
    splits_dir = root / "splits"
    embeddings_dir = root / "embeddings"
    ckpt_dir = root / "checkpoints"
    metrics_dir = root / "metrics"
    summary_dir = root / "summary"
    for d in [logs, datasets_dir, splits_dir, embeddings_dir, ckpt_dir, metrics_dir, summary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Download VATEX and normalize layout
    # ------------------------------------------------------------------
    stage = "01_vatex_download_layout"
    if not is_done(state, stage):
        print(f"[stage] {stage}")
        os.environ["KAGGLEHUB_CACHE"] = str((root / "cache" / "kagglehub").resolve())
        import kagglehub

        raw_train = Path(kagglehub.dataset_download(args.download_dataset)).resolve()
        raw_test = Path(kagglehub.dataset_download(args.download_dataset_test)).resolve()
        print(f"[download] dataset path train: {raw_train}")
        print(f"[download] dataset path test:  {raw_test}")
        videos_train = find_videos_dir(raw_train)
        videos_test = find_videos_dir(raw_test)
        print(f"[download] videos dir train: {videos_train}")
        print(f"[download] videos dir test:  {videos_test}")
        videos_dir = build_combined_videos_dir(
            datasets_dir / "vatex_combined_videos",
            [videos_train, videos_test],
        )
        print(f"[download] videos dir merged: {videos_dir}")

        vatex_root = datasets_dir / "vatex"
        vatex_root.mkdir(parents=True, exist_ok=True)

        videos_link = vatex_root / "videos"
        if videos_link.exists() or videos_link.is_symlink():
            videos_link.unlink()
        videos_link.symlink_to(videos_dir)

        build_hf_json_from_repo_pairs(repo_dir, vatex_root, videos_dir)
        mark_done(state, stage)
    else:
        print(f"[skip] {stage}")

    # ------------------------------------------------------------------
    # 2) Prepare split JSON with prepare_video_text_dataset.py
    # ------------------------------------------------------------------
    stage = "02_split_prep"
    if not is_done(state, stage):
        print(f"[stage] {stage}")
        vatex_root = datasets_dir / "vatex"
        cmd = [
            sys.executable,
            str(repo_dir / "prepare_video_text_dataset.py"),
            "--dataset",
            "vatex",
            "--vatex_root",
            str(vatex_root),
            "--output_dir",
            str(splits_dir),
        ]
        run_cmd(cmd, repo_dir)
        # validate
        for sp in ["train", "val", "test"]:
            p = splits_dir / f"vatex_{sp}_pairs.json"
            if not p.exists():
                raise FileNotFoundError(f"Missing split output: {p}")
            payload = json.loads(p.read_text(encoding="utf-8"))
            if int(payload.get("num_pairs", 0)) <= 0:
                raise RuntimeError(f"Empty split output: {p}")
        mark_done(state, stage)
    else:
        print(f"[skip] {stage}")

    # ------------------------------------------------------------------
    # 3) Generate ImageBind embeddings (train + test)
    # ------------------------------------------------------------------
    stage = "03_imagebind_embeddings"
    if not is_done(state, stage):
        print(f"[stage] {stage}")
        train_pairs = splits_dir / "vatex_train_pairs.json"
        test_pairs = splits_dir / "vatex_test_pairs.json"
        emb_train = embeddings_dir / "vatex_train_imagebind"
        emb_test = embeddings_dir / "vatex_test_imagebind"

        if not (emb_train / "emb_text.pt").exists():
            run_cmd(
                [
                    sys.executable,
                    str(repo_dir / "generate_imagebind_embeddings_generic.py"),
                    "--pairs_json",
                    str(train_pairs),
                    "--output_dir",
                    str(emb_train),
                    "--num_frames",
                    "16",
                    "--image_size",
                    "224",
                    "--text_batch_size",
                    "512",
                    "--use_fp16",
                ],
                repo_dir,
            )
        else:
            print("[skip] train embeddings already exist")

        if not (emb_test / "emb_text.pt").exists():
            run_cmd(
                [
                    sys.executable,
                    str(repo_dir / "generate_imagebind_embeddings_generic.py"),
                    "--pairs_json",
                    str(test_pairs),
                    "--output_dir",
                    str(emb_test),
                    "--num_frames",
                    "16",
                    "--image_size",
                    "224",
                    "--text_batch_size",
                    "512",
                    "--use_fp16",
                ],
                repo_dir,
            )
        else:
            print("[skip] test embeddings already exist")
        mark_done(state, stage)
    else:
        print(f"[skip] {stage}")

    # ------------------------------------------------------------------
    # 4) Baseline ImageBind + PCME
    # ------------------------------------------------------------------
    stage = "04_baseline_imagebind_pcme"
    if not is_done(state, stage):
        print(f"[stage] {stage}")
        emb_train = embeddings_dir / "vatex_train_imagebind"
        emb_test = embeddings_dir / "vatex_test_imagebind"
        pcme_ckpt_dir = ckpt_dir / "pcme"
        pcme_ckpt_dir.mkdir(parents=True, exist_ok=True)
        pcme_ckpt = pcme_ckpt_dir / "best_projectors.pth"

        if not pcme_ckpt.exists():
            run_cmd(
                [
                    sys.executable,
                    str(repo_dir / "train_pcme_projector.py"),
                    "--emb_dir",
                    str(emb_train),
                    "--save_dir",
                    str(pcme_ckpt_dir),
                    "--epochs",
                    str(args.pcme_epochs),
                ],
                repo_dir,
            )
        else:
            print("[skip] PCME checkpoint exists")

        run_cmd(
            [
                sys.executable,
                str(repo_dir / "measure_latency_memory_variance.py"),
                "--emb_dir",
                str(emb_test),
                "--ckpt",
                str(pcme_ckpt),
                "--save",
                str(metrics_dir / "pcme_metrics.json"),
            ],
            repo_dir,
        )
        mark_done(state, stage)
    else:
        print(f"[skip] {stage}")

    # ------------------------------------------------------------------
    # 5) Chebyshev degree sweep 3/4/5/6
    # ------------------------------------------------------------------
    stage = "05_cheb_degree_sweep"
    if not is_done(state, stage):
        print(f"[stage] {stage}")
        emb_train = embeddings_dir / "vatex_train_imagebind"
        emb_test = embeddings_dir / "vatex_test_imagebind"

        for deg in [3, 4, 5, 6]:
            ddir = ckpt_dir / f"cheb_deg{deg}"
            ddir.mkdir(parents=True, exist_ok=True)
            ckpt = ddir / "best_gaussian_cheb_coeff.pth"
            if not ckpt.exists():
                run_cmd(
                    [
                        sys.executable,
                        str(repo_dir / "train_cheb_projector_v2.py"),
                        "--emb_dir",
                        str(emb_train),
                        "--save_dir",
                        str(ddir),
                        "--save_name",
                        "best_gaussian_cheb_coeff.pth",
                        "--epochs",
                        str(args.cheb_epochs),
                        "--loss_mode",
                        "asymmetric",
                        "--t2v_weight",
                        "1.0",
                        "--v2t_weight",
                        "2.5",
                        "--mu_on_sphere",
                        "--kernel_use_mu_residual",
                        "--infer_vid_ids",
                        "--caps_per_video",
                        "20",
                        "--cheb_order",
                        str(deg),
                    ],
                    repo_dir,
                )
            else:
                print(f"[skip] Cheb deg{deg} checkpoint exists")

            run_cmd(
                [
                    sys.executable,
                    str(repo_dir / "measure_cheb_v2.py"),
                    "--emb_dir",
                    str(emb_test),
                    "--cheb_ckpt",
                    str(ckpt),
                    "--baseline_name",
                    "ImageBind",
                    "--save",
                    str(metrics_dir / f"cheb_deg{deg}_metrics.json"),
                ],
                repo_dir,
            )

        mark_done(state, stage)
    else:
        print(f"[skip] {stage}")

    # ------------------------------------------------------------------
    # 6) Aggregate table
    # ------------------------------------------------------------------
    stage = "06_aggregate_table"
    if not is_done(state, stage):
        print(f"[stage] {stage}")
        run_cmd(
            [
                sys.executable,
                str(repo_dir / "aggregate_vatex_results.py"),
                "--root",
                str(root),
            ],
            repo_dir,
        )
        mark_done(state, stage)
    else:
        print(f"[skip] {stage}")

    print("\n[done] VATEX full pipeline completed.")
    print(f"[summary] {summary_dir / 'vatex_results.csv'}")
    print(f"[summary] {summary_dir / 'vatex_results.md'}")


if __name__ == "__main__":
    main()
