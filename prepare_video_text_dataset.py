#!/usr/bin/env python3
"""
Prepare unified video-text split files for retrieval experiments.

Output format (JSON):
{
  "dataset": "msvd",
  "split": "train",
  "num_pairs": N,
  "pairs": [
    {
      "video_id": "...",
      "caption": "...",
      "video_path": "/abs/path/to/video.ext"
    }
  ]
}
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def save_pairs(dataset: str, split: str, pairs: List[Dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "dataset": dataset,
        "split": split,
        "num_pairs": len(pairs),
        "pairs": pairs,
    }
    out_path = output_dir / f"{dataset}_{split}_pairs.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {split}: {len(pairs)} pairs -> {out_path}")


def _choose_one_caption(captions: List[str]) -> str:
    for c in captions:
        c = str(c).strip()
        if c:
            return c
    return ""


def prepare_msvd(
    corpus_dir: Path,
    videos_dir: Path,
    output_dir: Path,
    keep_all_captions: bool = True,
):
    ann_path = corpus_dir / "annotations.txt"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing MSVD annotations: {ann_path}")

    clips_dir = videos_dir / "YouTubeClips"
    if not clips_dir.exists():
        raise FileNotFoundError(f"Missing MSVD video directory: {clips_dir}")

    cap_map: Dict[str, List[str]] = {}
    with open(ann_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: "<video_id_start_end> <caption...>"
            parts = line.split(" ", 1)
            if len(parts) < 2:
                continue
            vid, cap = parts[0].strip(), parts[1].strip()
            cap_map.setdefault(vid, []).append(cap)

    valid_ids = []
    for vid in sorted(cap_map.keys()):
        vpath = clips_dir / f"{vid}.avi"
        if vpath.exists():
            valid_ids.append(vid)

    # Standard public split: 1200 / 100 / 670 on 1970 clips.
    train_ids = valid_ids[:1200]
    val_ids = valid_ids[1200:1300]
    test_ids = valid_ids[1300:1970]

    def build(ids: List[str]) -> List[Dict]:
        out = []
        for vid in ids:
            caps = cap_map.get(vid, [])
            if keep_all_captions:
                for caption in caps:
                    caption = str(caption).strip()
                    if not caption:
                        continue
                    out.append(
                        {
                            "video_id": vid,
                            "caption": caption,
                            "video_path": str((clips_dir / f"{vid}.avi").resolve()),
                        }
                    )
            else:
                caption = _choose_one_caption(caps)
                if not caption:
                    continue
                out.append(
                    {
                        "video_id": vid,
                        "caption": caption,
                        "video_path": str((clips_dir / f"{vid}.avi").resolve()),
                    }
                )
        return out

    save_pairs("msvd", "train", build(train_ids), output_dir)
    save_pairs("msvd", "val", build(val_ids), output_dir)
    save_pairs("msvd", "test", build(test_ids), output_dir)


def prepare_vatex(vatex_root: Path, output_dir: Path):
    """
    Expect a HuggingFace datasets disk layout:
      <vatex_root>/json/dataset_dict.json + split folders
    """
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise ImportError(
            "datasets package is required for VATEX split parsing. "
            "Install with: pip install datasets"
        ) from exc

    ds_root = vatex_root / "json"
    if not (ds_root / "dataset_dict.json").exists():
        raise FileNotFoundError(f"Missing VATEX dataset_dict.json under: {ds_root}")

    ds_dict = load_from_disk(str(ds_root))
    split_map: List[Tuple[str, str]] = [
        ("train", "train"),
        ("validation", "val"),
        ("public_test", "test"),
    ]

    def extract_split(split_name: str) -> List[Dict]:
        if split_name not in ds_dict:
            return []
        ds = ds_dict[split_name]
        pairs = []
        for row in ds:
            video_id = (
                row.get("videoID")
                or row.get("video_id")
                or row.get("video")
                or row.get("id")
            )
            if not video_id:
                continue
            video_id = str(video_id)

            caps = (
                row.get("enCap")
                or row.get("captions")
                or row.get("caption")
                or row.get("en_captions")
                or row.get("en")
            )
            if isinstance(caps, list):
                caption = _choose_one_caption(caps)
            else:
                caption = str(caps).strip() if caps is not None else ""
            if not caption:
                continue

            vpath = vatex_root / "videos" / f"{video_id}.mp4"
            if not vpath.exists():
                continue

            pairs.append(
                {
                    "video_id": video_id,
                    "caption": caption,
                    "video_path": str(vpath.resolve()),
                }
            )
        return pairs

    for src_split, dst_split in split_map:
        save_pairs("vatex", dst_split, extract_split(src_split), output_dir)


def main():
    parser = argparse.ArgumentParser(description="Prepare unified split files for MSVD/VATEX")
    parser.add_argument("--dataset", choices=["msvd", "vatex"], required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--corpus_dir", type=str, default=None, help="MSVD corpus path")
    parser.add_argument("--videos_dir", type=str, default=None, help="MSVD videos root path")
    parser.add_argument("--vatex_root", type=str, default=None, help="VATEX root path")
    parser.add_argument(
        "--msvd_single_caption",
        action="store_true",
        help="Use one caption per MSVD video (legacy behavior). Default keeps all captions.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()

    if args.dataset == "msvd":
        corpus_dir = Path(args.corpus_dir or "/mnt/data/datasets/msvd/corpus").resolve()
        videos_dir = Path(args.videos_dir or "/mnt/data/datasets/msvd/videos").resolve()
        prepare_msvd(
            corpus_dir,
            videos_dir,
            output_dir,
            keep_all_captions=not args.msvd_single_caption,
        )
    else:
        vatex_root = Path(args.vatex_root or "/mnt/data/datasets/vatex").resolve()
        prepare_vatex(vatex_root, output_dir)


if __name__ == "__main__":
    main()

