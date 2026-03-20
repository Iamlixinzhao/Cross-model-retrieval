#!/usr/bin/env python3
"""
Aggregate VATEX benchmark metrics into CSV/Markdown tables.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_get(dct, *keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _lat_mean(summary_obj: dict) -> float | None:
    return _safe_get(summary_obj, "mean_ms", default=None)


def _gpu_mean(summary_obj: dict) -> float | None:
    return _safe_get(summary_obj, "gpu_mean_mb", default=None)


def _retrieval_row(tag: str, degree: str, ret: dict, lat: float | None, gpu: float | None, ckpt: str):
    return {
        "model": tag,
        "degree": degree,
        "t2v_r1": _safe_get(ret, "t2v", "R@k", 1, default=None),
        "t2v_r5": _safe_get(ret, "t2v", "R@k", 5, default=None),
        "t2v_r10": _safe_get(ret, "t2v", "R@k", 10, default=None),
        "t2v_medr": _safe_get(ret, "t2v", "MedR", default=None),
        "t2v_meanr": _safe_get(ret, "t2v", "MeanR", default=None),
        "v2t_r1": _safe_get(ret, "v2t", "R@k", 1, default=None),
        "v2t_r5": _safe_get(ret, "v2t", "R@k", 5, default=None),
        "v2t_r10": _safe_get(ret, "v2t", "R@k", 10, default=None),
        "v2t_medr": _safe_get(ret, "v2t", "MedR", default=None),
        "v2t_meanr": _safe_get(ret, "v2t", "MeanR", default=None),
        "latency_ms": lat,
        "gpu_mb": gpu,
        "checkpoint": ckpt,
    }


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser(description="Aggregate VATEX benchmark metrics")
    ap.add_argument("--root", type=str, required=True, help="Root path, e.g. /data2/vatex_experiments")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    metrics_dir = root / "metrics"
    ckpt_dir = root / "checkpoints"
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    pcme_metrics_path = metrics_dir / "pcme_metrics.json"
    if not pcme_metrics_path.exists():
        raise FileNotFoundError(f"Missing {pcme_metrics_path}")

    rows = []

    # Baseline + PCME
    pcme = _read_json(pcme_metrics_path)
    s = _safe_get(pcme, "summary", default={}) or {}
    ret = _safe_get(s, "retrieval", default={}) or {}

    rows.append(
        _retrieval_row(
            "ImageBind",
            "-",
            _safe_get(ret, "imagebind", default={}) or {},
            _lat_mean(_safe_get(s, "imagebind", default={}) or {}),
            _gpu_mean(_safe_get(s, "imagebind", default={}) or {}),
            "-",
        )
    )

    rows.append(
        _retrieval_row(
            "PCME",
            "-",
            _safe_get(ret, "projector", default={}) or {},
            _lat_mean(_safe_get(s, "projector", default={}) or {}),
            _gpu_mean(_safe_get(s, "projector", default={}) or {}),
            str((ckpt_dir / "pcme" / "best_projectors.pth").resolve()),
        )
    )

    # Cheb degree sweep
    for deg in [3, 4, 5, 6]:
        mpath = metrics_dir / f"cheb_deg{deg}_metrics.json"
        if not mpath.exists():
            raise FileNotFoundError(f"Missing {mpath}")
        data = _read_json(mpath)
        s2 = _safe_get(data, "summary", default={}) or {}
        ret2 = _safe_get(s2, "retrieval", "gaussian_cheb_v2", default={}) or {}
        rows.append(
            _retrieval_row(
                "Chebyshev",
                str(deg),
                ret2,
                _lat_mean(_safe_get(s2, "gaussian_cheb_v2", default={}) or {}),
                _gpu_mean(_safe_get(s2, "gaussian_cheb_v2", default={}) or {}),
                str((ckpt_dir / f"cheb_deg{deg}" / "best_gaussian_cheb_coeff.pth").resolve()),
            )
        )

    headers = [
        "model",
        "degree",
        "t2v_r1",
        "t2v_r5",
        "t2v_r10",
        "t2v_medr",
        "t2v_meanr",
        "v2t_r1",
        "v2t_r5",
        "v2t_r10",
        "v2t_medr",
        "v2t_meanr",
        "latency_ms",
        "gpu_mb",
        "checkpoint",
    ]

    csv_path = summary_dir / "vatex_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    md_path = summary_dir / "vatex_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# VATEX Results Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(_fmt(r[h]) for h in headers) + " |\n")

    print(f"Saved CSV: {csv_path}")
    print(f"Saved MD:  {md_path}")


if __name__ == "__main__":
    main()
