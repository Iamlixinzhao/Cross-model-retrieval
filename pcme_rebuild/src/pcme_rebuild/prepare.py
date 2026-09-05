"""Prepare local data. Never downloads a dataset or chooses a benchmark split silently."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import Collator, GroupDataset, fingerprint, read_manifest, to_device


def karpathy(args):
    source = json.loads(Path(args.annotations).read_text())
    rows = []
    for image in source["images"]:
        split = image["split"]
        if split == "restval":
            if not args.include_restval:
                continue
            split = "train"
        path = Path(args.images_root) / image.get("filepath", "") / image["filename"]
        key = args.namespace + ":" + str(image.get("cocoid", image.get("imgid", image["filename"])))
        for caption in image["sentences"]:
            rows.append({"id": key, "image": str(path.resolve()), "caption": caption["raw"], "split": split})
    write_manifest(args.output, rows)


def write_manifest(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    read_manifest(path)


def subset(args):
    rows = read_manifest(args.manifest)
    g = torch.Generator().manual_seed(args.seed)
    output = []
    for split, count in (("train", args.train), ("val", args.val), ("test", args.test)):
        ids = sorted(set(r["id"] for r in rows if r["split"] == split))
        selected = {ids[i] for i in torch.randperm(len(ids), generator=g)[:count].tolist()}
        output += [r for r in rows if r["split"] == split and r["id"] in selected]
    write_manifest(args.output, output)


@torch.no_grad()
def cache_clip(args):
    from transformers import CLIPModel
    from .model import RetrievalModel
    ds = GroupDataset("clip", args.manifest, args.split, captions_per_media=0)
    model = CLIPModel.from_pretrained(args.clip_name).to(args.device).eval()
    # Reuse the exact feature path of the online model without constructing new heads.
    class Holder:
        backbone = model
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=Collator(args.clip_name), num_workers=0)
    chunks = {}
    ordered_text_ids = []
    for batch in loader:
        ordered_text_ids += [ds.group_ids[i] for i in batch["text_ids"].tolist()]
        features = RetrievalModel.clip_features(Holder(), to_device(batch, args.device))
        for k, v in features.items():
            if args.pooled_only and k.endswith("tokens"):
                v = features[k.replace("tokens", "pool")][:, None]
            if args.pooled_only and k.endswith("mask"):
                v = torch.ones((len(v), 1), device=v.device, dtype=torch.bool)
            chunks.setdefault(k, []).append(v.cpu().to(torch.bool if k.endswith("mask") else torch.float16))
    cache = {k: torch.cat(v) for k, v in chunks.items()}
    cache.update(schema_version=1, split=args.split, media_ids=ds.group_ids, text_ids=ordered_text_ids,
                 feature_kind="pooled_ablation" if args.pooled_only else "local_tokens",
                 source_sha256=fingerprint(args.manifest), backbone=args.clip_name)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.output)


def import_legacy(args):
    # IDs are mandatory: NEVER infer one-to-one matches from row position.
    text = torch.load(args.text, weights_only=True, map_location="cpu").float()
    media = torch.load(args.media, weights_only=True, map_location="cpu").float()
    ti = json.loads(Path(args.text_ids).read_text())
    mi = json.loads(Path(args.media_ids).read_text())
    if text.ndim != 2 or media.ndim != 2 or len(ti) != len(text) or len(mi) != len(media):
        raise ValueError("Feature/ID sizes disagree")
    lookup, unique, indices = {}, [], []
    for i, key in enumerate(mi):
        if not isinstance(key, str):
            raise ValueError("Use globally namespaced string IDs")
        if key in lookup:
            if not torch.allclose(media[i], media[lookup[key]], atol=1e-5, rtol=1e-4):
                raise ValueError("Duplicate ID has different media features; resolve views explicitly")
        else:
            lookup[key] = i
            unique.append(key)
            indices.append(i)
    media = media[indices]
    out = dict(schema_version=1, split=args.split, text_ids=ti, media_ids=unique,
               feature_kind="pooled_ablation", backbone="legacy-user-features")
    for name, values in (("text", text), ("media", media)):
        values = torch.nn.functional.normalize(values, dim=-1)
        out[name + "_pool"], out[name + "_tokens"] = values, values[:, None]
        out[name + "_mask"] = torch.ones(len(values), 1, dtype=torch.bool)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    from .data import load_cache
    load_cache(args.output)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    k = sub.add_parser("karpathy")
    k.add_argument("--annotations", required=True)
    k.add_argument("--images-root", required=True)
    k.add_argument("--namespace", required=True, help="e.g. flickr30k or coco")
    k.add_argument("--include-restval", action="store_true")
    k.add_argument("--output", required=True)
    k.set_defaults(fn=karpathy)
    s = sub.add_parser("subset")
    s.add_argument("--manifest", required=True)
    for split, n in (("train", 1000), ("val", 200), ("test", 200)):
        s.add_argument("--" + split, type=int, default=n)
    s.add_argument("--seed", type=int, default=17)
    s.add_argument("--output", required=True)
    s.set_defaults(fn=subset)
    c = sub.add_parser("cache-clip")
    c.add_argument("--manifest", required=True)
    c.add_argument("--split", choices=["train", "val", "test"], required=True)
    c.add_argument("--clip-name", default="openai/clip-vit-base-patch32")
    c.add_argument("--device", default="cuda")
    c.add_argument("--batch-size", type=int, default=8)
    c.add_argument("--pooled-only", action="store_true")
    c.add_argument("--output", required=True)
    c.set_defaults(fn=cache_clip)
    l = sub.add_parser("import-legacy")
    for arg in ("text", "media", "text-ids", "media-ids", "output"):
        l.add_argument("--" + arg, required=True)
    l.add_argument("--split", choices=["train", "val", "test"], required=True)
    l.set_defaults(fn=import_legacy)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
