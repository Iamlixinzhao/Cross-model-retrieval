#!/usr/bin/env python3
"""
Generate ImageBind text/video embeddings from prepared pair JSON.
"""

import argparse
import json
from pathlib import Path
import sys
import importlib

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path("/mnt/data/pes/ImageBind")))
# pytorchvideo expects this legacy torchvision module path.
try:
    importlib.import_module("torchvision.transforms.functional_tensor")
except ModuleNotFoundError:
    ft = importlib.import_module("torchvision.transforms._functional_tensor")
    sys.modules["torchvision.transforms.functional_tensor"] = ft
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data as ib_data

import eval_msrvtt_1kA as eval_script


def load_pairs(path: Path):
    with open(path, "r") as f:
        payload = json.load(f)
    pairs = payload["pairs"]
    texts = [p["caption"] for p in pairs]
    video_paths = [Path(p["video_path"]) for p in pairs]
    video_ids = [str(p.get("video_id", i)) for i, p in enumerate(pairs)]
    return payload, texts, video_paths, video_ids


@torch.no_grad()
def encode_text_batched(model, device, texts, batch_size=512, use_fp16=True):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encode text", ncols=100):
        batch = texts[i:i + batch_size]
        td = ib_data.load_and_transform_text(batch, device=device)
        if use_fp16 and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model({ModalityType.TEXT: td})[ModalityType.TEXT]
        else:
            out = model({ModalityType.TEXT: td})[ModalityType.TEXT]
        embs.append(out.detach().cpu())
    return torch.cat(embs, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Generate ImageBind embeddings from pair JSON")
    parser.add_argument("--pairs_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--text_batch_size", type=int, default=512)
    parser.add_argument("--use_fp16", action="store_true")
    args = parser.parse_args()

    pairs_json = Path(args.pairs_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, texts, video_paths, video_ids = load_pairs(pairs_json)
    # Fail fast: eval_msrvtt_1kA.encode_video uses zero vectors for missing/failed videos,
    # which silently destroys retrieval metrics.
    missing = [p for p in video_paths if not p.exists()]
    if missing:
        n = len(missing)
        examples = [str(p) for p in missing[:5]]
        raise FileNotFoundError(
            f"Missing {n}/{len(video_paths)} video files. "
            f"Restore clips or re-run prepare_video_text_dataset.py with valid --corpus_dir/--videos_dir. "
            f"Examples: {examples}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Pairs: {len(texts)}")

    print("Loading ImageBind model...")
    model = imagebind_model.imagebind_huge(pretrained=True).to(device).eval()

    print("Encoding text...")
    text_emb = encode_text_batched(
        model,
        device,
        texts,
        batch_size=args.text_batch_size,
        use_fp16=args.use_fp16,
    )

    print("Encoding video...")
    video_emb = eval_script.encode_video(
        model,
        device,
        video_paths,
        num_frames=args.num_frames,
        image_size=args.image_size,
        use_fp16=args.use_fp16,
    )

    text_emb = F.normalize(text_emb, dim=-1)
    video_emb = F.normalize(video_emb, dim=-1)

    torch.save(text_emb, output_dir / "emb_text.pt")
    torch.save(video_emb, output_dir / "emb_video.pt")
    # Build retrieval group ids from video_id so multiple captions of same video
    # share the same positive group id for multi-positive contrastive training.
    vid_to_idx = {}
    vid_ids_tensor = torch.empty(len(video_ids), dtype=torch.long)
    next_idx = 0
    for i, vid in enumerate(video_ids):
        if vid not in vid_to_idx:
            vid_to_idx[vid] = next_idx
            next_idx += 1
        vid_ids_tensor[i] = vid_to_idx[vid]
    torch.save(vid_ids_tensor, output_dir / "vid_ids.pt")

    meta = {
        "source_pairs": str(pairs_json),
        "dataset": payload.get("dataset"),
        "split": payload.get("split"),
        "num_pairs": len(texts),
        "num_unique_videos": int(len(vid_to_idx)),
        "embedding_dim": int(text_emb.shape[-1]),
        "num_frames": args.num_frames,
        "image_size": args.image_size,
        "use_fp16": bool(args.use_fp16),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved to {output_dir}")
    print(f"text={tuple(text_emb.shape)}, video={tuple(video_emb.shape)}")


if __name__ == "__main__":
    main()

