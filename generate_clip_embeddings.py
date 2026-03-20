#!/usr/bin/env python3
"""
Generate OpenAI CLIP or OpenCLIP embeddings for MSR-VTT train/test splits.

Outputs:
  emb_text.pt   [N, D]
  emb_video.pt  [N, D]
  vid_ids.pt    [N]
  metadata.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

import clip as openai_clip

try:
    import open_clip
except ImportError:
    open_clip = None


ROOT = Path("/mnt/data/pes/ImageBind")
VID_DIR = ROOT / "msrvtt_videos"
ANN_DIR = ROOT / "msrvtt_annotation"


def get_train_pairs():
    json_path = ANN_DIR / "MSRVTT_data.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    standard_train_ids = {f"video{i}" for i in range(6513)}
    video_to_captions = {}
    for sent_info in data["sentences"]:
        vid = sent_info["video_id"]
        caption = sent_info["caption"]
        video_to_captions.setdefault(vid, []).append(caption)

    video_ids, captions = [], []
    for vid in sorted(standard_train_ids):
        caps = video_to_captions.get(vid, [])
        if caps:
            video_ids.append(vid)
            captions.append(caps[0])

    return video_ids, captions


def get_test_pairs():
    csv_path = ANN_DIR / "MSRVTT_JSFUSION_test.csv"
    list_1ka = ANN_DIR / "msrvtt1kA.txt"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {csv_path}")

    df = pd.read_csv(csv_path)

    if list_1ka.exists():
        lines = [ln.strip() for ln in list_1ka.read_text().splitlines() if ln.strip()]
        caps_by_vid = {}
        for _, row in df.iterrows():
            caps_by_vid.setdefault(row["video_id"], []).append(row["sentence"])

        video_ids, captions = [], []
        for line in lines:
            parts = line.split()
            vid = parts[0]
            cap_idx = int(parts[1]) if len(parts) > 1 else 0
            candidates = caps_by_vid.get(vid, [])
            if not candidates:
                continue
            cap_idx = max(0, min(cap_idx, len(candidates) - 1))
            video_ids.append(vid)
            captions.append(candidates[cap_idx])
        return video_ids, captions

    seen = set()
    video_ids, captions = [], []
    for _, row in df.iterrows():
        vid = row["video_id"]
        if vid in seen:
            continue
        seen.add(vid)
        video_ids.append(vid)
        captions.append(row["sentence"])
    return video_ids, captions


def uniform_sample_frames(video_path: Path, num_frames: int):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    num_total = len(vr)
    if num_total == 0:
        return []
    indices = torch.linspace(0, num_total - 1, steps=num_frames).long().tolist()
    return [Image.fromarray(vr[idx].asnumpy()) for idx in indices]


@torch.no_grad()
def encode_texts(model, tokenizer, device, texts, batch_size):
    embs = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encode text", ncols=100):
        batch = texts[start:start + batch_size]
        tokens = tokenizer(batch).to(device)
        feats = model.encode_text(tokens)
        feats = F.normalize(feats.float(), dim=-1)
        embs.append(feats.cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def encode_videos(model, preprocess, device, video_paths, num_frames):
    embs = []
    feature_dim = model.text_projection.shape[1]
    for video_path in tqdm(video_paths, desc="Encode video", ncols=100):
        try:
            frames = uniform_sample_frames(video_path, num_frames)
        except Exception:
            frames = []

        if not frames:
            embs.append(torch.zeros(feature_dim))
            continue

        images = torch.stack([preprocess(frame) for frame in frames]).to(device)
        feats = model.encode_image(images)
        feats = F.normalize(feats.float(), dim=-1)
        video_feat = F.normalize(feats.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        embs.append(video_feat.cpu())
    return torch.stack(embs, dim=0)


def load_model_and_utils(args, device):
    if args.backend == "openai":
        model, preprocess = openai_clip.load(args.model_name, device=device, jit=False)

        def tokenizer(texts):
            return openai_clip.tokenize(texts, truncate=True)

        meta = {
            "backbone": "openai_clip",
            "model_name": args.model_name,
            "pretrained": None,
        }
        return model.eval(), preprocess, tokenizer, meta

    if open_clip is None:
        raise ImportError(
            "open_clip_torch is not installed. Install it with: pip install open_clip_torch"
        )
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=device,
    )
    tokenizer_fn = open_clip.get_tokenizer(args.model_name)

    def tokenizer(texts):
        return tokenizer_fn(texts)

    meta = {
        "backbone": "openclip",
        "model_name": args.model_name,
        "pretrained": args.pretrained,
    }
    return model.eval(), preprocess, tokenizer, meta


def main():
    parser = argparse.ArgumentParser(description="Generate CLIP/OpenCLIP embeddings for MSR-VTT")
    parser.add_argument("--split", choices=["train", "test"], default=None,
                        help="MSR-VTT split mode. Ignored when --pairs_json is provided.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pairs_json", type=str, default=None,
                        help="Generic dataset mode: path to *_pairs.json from prepare_video_text_dataset.py")
    parser.add_argument("--backend", choices=["openai", "openclip"], default="openai")
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k",
                        help="Used only when --backend=openclip")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--text_batch_size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading {args.backend} model: {args.model_name}")
    model, preprocess, tokenizer, model_meta = load_model_and_utils(args, device)

    if args.pairs_json:
        p = Path(args.pairs_json).resolve()
        with open(p, "r") as f:
            payload = json.load(f)
        pairs = payload.get("pairs", [])
        video_ids = [str(x["video_id"]) for x in pairs]
        captions = [str(x["caption"]) for x in pairs]
        video_paths = [Path(x["video_path"]).resolve() for x in pairs]
    else:
        if args.split is None:
            raise ValueError("Either --split (MSR-VTT mode) or --pairs_json (generic mode) must be set.")
        if args.split == "train":
            video_ids, captions = get_train_pairs()
        else:
            video_ids, captions = get_test_pairs()
        video_paths = [VID_DIR / f"{vid}.mp4" for vid in video_ids]
    existing = sum(1 for p in video_paths if p.exists())
    print(f"Split={args.split}, pairs={len(video_ids)}, existing_videos={existing}")

    text_emb = encode_texts(model, tokenizer, device, captions, args.text_batch_size)
    video_emb = encode_videos(model, preprocess, device, video_paths, args.frames)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(text_emb, output_dir / "emb_text.pt")
    torch.save(video_emb, output_dir / "emb_video.pt")
    torch.save(torch.arange(len(video_ids), dtype=torch.long), output_dir / "vid_ids.pt")

    metadata = {
        **model_meta,
        "split": args.split,
        "pairs_json": str(Path(args.pairs_json).resolve()) if args.pairs_json else None,
        "num_pairs": len(video_ids),
        "embedding_dim": int(text_emb.shape[-1]),
        "frames": args.frames,
        "text_batch_size": args.text_batch_size,
        "sample_video_ids": video_ids[:10],
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved embeddings to: {output_dir}")
    print(f"  text:  {tuple(text_emb.shape)}")
    print(f"  video: {tuple(video_emb.shape)}")


if __name__ == "__main__":
    main()
