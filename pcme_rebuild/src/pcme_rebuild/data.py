"""Split-safe image/text manifests and tensor caches with explicit group IDs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    seen, images, paths, captions = {}, {}, {}, set()
    for row in rows:
        if not all(k in row for k in ("id", "image", "caption", "split")):
            raise ValueError("Manifest requires id, image, caption, split")
        if not all(isinstance(row[k], str) and row[k] for k in row):
            raise ValueError("Manifest fields must be nonempty strings")
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        key, split = row["id"], row["split"]
        resolved = str(Path(row["image"]).expanduser().resolve())
        if key in seen and (seen[key] != split or images[key] != resolved):
            raise ValueError(f"Group {key} spans splits or multiple image files")
        if resolved in paths and paths[resolved] != (key, split):
            raise ValueError("Same image path appears under different IDs/splits")
        if (key, row["caption"]) in captions:
            raise ValueError(f"Duplicate caption for {key}")
        captions.add((key, row["caption"]))
        seen[key], images[key], paths[resolved] = split, resolved, (key, split)
    if not rows:
        raise ValueError("Empty manifest")
    return rows


def load_cache(path):
    cache = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if cache.get("schema_version") != 1:
        raise ValueError("Expected cache schema_version=1")
    for kind in ("text", "media"):
        ids = cache[kind + "_ids"]
        if not ids or not all(isinstance(x, str) for x in ids):
            raise ValueError("Cache IDs must be nonempty lists of strings")
        for suffix, ndim in (("tokens", 3), ("pool", 2), ("mask", 2)):
            t = cache[kind + "_" + suffix]
            if t.ndim != ndim or len(t) != len(ids):
                raise ValueError(f"Invalid {kind}_{suffix} shape")
            if not torch.isfinite(t).all():
                raise ValueError("Cache contains NaN/Inf")
        mask = cache[kind + "_mask"]
        if mask.dtype != torch.bool or mask.shape != cache[kind + "_tokens"].shape[:2] or not mask.any(1).all():
            raise ValueError("Invalid attention mask")
    if len(set(cache["media_ids"])) != len(cache["media_ids"]):
        raise ValueError("Media gallery must have unique IDs")
    if set(cache["text_ids"]) != set(cache["media_ids"]):
        raise ValueError("Every media item and caption needs a labeled match")
    if cache.get("split") not in {"train", "val", "test"}:
        raise ValueError("Cache needs an explicit train/val/test split")
    return cache


def require_disjoint(a, b):
    if set(a.group_ids) & set(b.group_ids):
        raise ValueError("Data leakage: media IDs overlap between splits")
    if a.split == b.split:
        raise ValueError("Training and validation must be different splits")


class GroupDataset(Dataset):
    def __init__(self, mode, path, split, captions_per_media=2, seed=17):
        self.mode, self.path, self.split = mode, str(path), split
        self.captions_per_media, self.seed, self.epoch = captions_per_media, seed, 0
        self.cache = None
        self.source_hash = fingerprint(path)
        if mode == "cache":
            self.cache = load_cache(path)
            if self.cache["split"] != split:
                raise ValueError(f"Expected {split} cache")
            self.group_ids = self.cache["media_ids"]
            text_ids = self.cache["text_ids"]
        elif mode == "clip":
            self.rows = [r for r in read_manifest(path) if r["split"] == split]
            self.group_ids = list(dict.fromkeys(r["id"] for r in self.rows))
            text_ids = [r["id"] for r in self.rows]
        else:
            raise ValueError(mode)
        if not self.group_ids:
            raise ValueError(f"No data for {split}")
        lookup = {key: i for i, key in enumerate(self.group_ids)}
        self.caption_indices = [[] for _ in self.group_ids]
        for i, key in enumerate(text_ids):
            self.caption_indices[lookup[key]].append(i)
        self.text_ids = text_ids

    @property
    def dimensions(self):
        if self.cache is None:
            from transformers import CLIPConfig
            c = CLIPConfig.from_pretrained(self.clip_name)
            return {"text": [c.text_config.hidden_size, c.projection_dim],
                    "media": [c.vision_config.hidden_size, c.projection_dim]}
        return {k: [self.cache[k + "_tokens"].shape[-1], self.cache[k + "_pool"].shape[-1]]
                for k in ("text", "media")}

    def __len__(self):
        return len(self.group_ids)

    def __getitem__(self, idx):
        indices = self.caption_indices[idx]
        if self.captions_per_media:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, idx]))
            indices = rng.choice(indices, self.captions_per_media,
                                 replace=len(indices) < self.captions_per_media).tolist()
        out = {"media_ids": torch.tensor([idx]), "text_ids": torch.full((len(indices),), idx),
               "media_index": torch.tensor([idx]), "text_index": torch.tensor(indices)}
        if self.cache is not None:
            for kind, ix in (("media", [idx]), ("text", indices)):
                for suffix in ("tokens", "pool", "mask"):
                    out[kind + "_" + suffix] = self.cache[kind + "_" + suffix][ix]
        else:
            row = self.rows[self.caption_indices[idx][0]]
            with Image.open(Path(row["image"]).expanduser()) as image:
                out["images"] = [image.convert("RGB")]
            out["captions"] = [self.rows[i]["caption"] for i in indices]
        return out


class Collator:
    def __init__(self, clip_name=None):
        self.processor = None
        if clip_name:
            from transformers import CLIPProcessor
            self.processor = CLIPProcessor.from_pretrained(clip_name)

    def __call__(self, rows):
        output = {k: torch.cat([row[k] for row in rows]) for k in rows[0]
                  if k not in ("images", "captions")}
        if self.processor:
            output.update(self.processor(images=sum([r["images"] for r in rows], []),
                                         text=sum([r["captions"] for r in rows], []),
                                         padding="max_length", max_length=77,
                                         truncation=True, return_tensors="pt"))
        return output


def make_dataset(config, split, training=False):
    mode = config.get("mode", "cache")
    path = config[split + "_cache"] if mode == "cache" else config["manifest"]
    ds = GroupDataset(mode, path, split, config.get("captions_per_media", 2) if training else 0,
                      config.get("seed", 17))
    ds.clip_name = config.get("clip_name")
    return ds


def to_device(batch, device):
    return {k: v.to(device=device, dtype=torch.float32 if v.is_floating_point() else v.dtype,
                    non_blocking=True) for k, v in batch.items()}
