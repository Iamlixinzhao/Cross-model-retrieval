"""Exact labeled retrieval on a declared gallery, with uncertainty interventions."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import Collator, make_dataset, to_device
from .model import RetrievalModel
from .probability import matrix_log_probs, sample


@torch.no_grad()
def encode(model, dataset, device, batch_size=16):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=Collator(dataset.clip_name if dataset.mode == "clip" else None), num_workers=0)
    out = {}
    for batch in loader:
        output = model(to_device(batch, device))
        for k in ("text_mu", "text_logvar", "media_mu", "media_logvar", "text_base", "media_base"):
            out.setdefault(k, []).append(output[k].cpu())
        for k in ("text_ids", "media_ids", "text_index", "media_index"):
            out.setdefault(k, []).append(batch[k])
    result = {k: torch.cat(v) for k, v in out.items()}
    # IDs are in dataset-local coordinates, preserved with their original names.
    result.update(group_ids=dataset.group_ids, split=dataset.split, source_sha256=dataset.source_hash,
                  scale=float(output["scale"]), shift=float(output["shift"]),
                  sigmoid_factor=model.config.get("sigmoid_factor", 1.0))
    return result


def rank_metrics(scores, query_ids, candidate_ids):
    ranks, hits, ap = [], [], []
    for row, key in zip(scores, query_ids):
        relevant = candidate_ids.eq(key)
        if not relevant.any():
            raise ValueError("Query without labeled match in gallery")
        order = row.argsort(descending=True, stable=True)
        positions = relevant[order].nonzero().flatten() + 1
        rank = int(positions[0])
        ranks.append(rank)
        hits.append(rank == 1)
        ap.append(float((torch.arange(1, len(positions) + 1) / positions).mean()))
    r = torch.tensor(ranks, dtype=torch.float32)
    return {"R@1": float((r <= 1).float().mean() * 100),
            "R@5": float((r <= 5).float().mean() * 100),
            "R@10": float((r <= 10).float().mean() * 100),
            "median_rank": float(torch.quantile(r, .5)), "mAP": float(np.mean(ap)),
            "queries": len(ranks)}, torch.tensor(hits)


def bidirectional_metrics(scores, encoded):
    t, th = rank_metrics(scores, encoded["text_ids"], encoded["media_ids"])
    m, mh = rank_metrics(scores.T, encoded["media_ids"], encoded["text_ids"])
    return {"text_to_media": t, "media_to_text": m,
            "mean_R@1": (t["R@1"] + m["R@1"]) / 2}, (th, mh)


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def average_ranks(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=float)
    _, start, counts = np.unique(x[order], return_index=True, return_counts=True)
    for first, n in zip(start, counts):
        ranks[order[first:first+n]] = first + (n - 1) / 2
    return ranks


def score_comparison(pred, teacher, encoded):
    p, t = pred.flatten().numpy(), teacher.flatten().numpy()
    pos = encoded["text_ids"][:, None].eq(encoded["media_ids"][None, :]).numpy().ravel()
    delta = np.abs(p - t)
    return {"mae": float(delta.mean()), "positive_mae": float(delta[pos].mean()),
            "negative_mae": float(delta[~pos].mean()),
            "balanced_mae": float(.5 * (delta[pos].mean() + delta[~pos].mean())),
            "rmse": float(np.sqrt(((p-t)**2).mean())), "pearson": correlation(p, t),
            "spearman": correlation(average_ranks(p), average_ranks(t)),
            "top1_agreement_text": float((pred.argmax(1) == teacher.argmax(1)).float().mean()),
            "top1_agreement_media": float((pred.argmax(0) == teacher.argmax(0)).float().mean())}


def sigma_diagnostics(mu, logvar, hits=None):
    s = (.5 * logvar).exp()
    radius = logvar.exp().sum(-1).sqrt()
    out = {"sigma_quantiles": {str(q): float(torch.quantile(s.flatten(), q)) for q in (0., .01, .1, .5, .9, .99, 1.)},
           "mean_sigma": float(s.mean()), "between_item_sigma_std": float(s.mean(-1).std(unbiased=False)),
           "mean_noise_radius": float(radius.mean()),
           "mean_mu_norm": float(mu.norm(dim=-1).mean()),
           "fraction_sigma_below_1e-3": float((s < 1e-3).float().mean()),
           "lower_guard_fraction": float((logvar <= -29.99).float().mean()),
           "upper_guard_fraction": float((logvar >= 9.99).float().mean())}
    if hits is not None:
        out["uncertainty_error_spearman"] = correlation(average_ranks(radius.numpy()), average_ranks((~hits).numpy()))
        order = radius.argsort()
        out["selective_R@1"] = {str(c): float(hits[order[:max(1, math.ceil(len(order)*c))]].float().mean()*100)
                                for c in (.25, .5, .75, 1.)}
    return out


@torch.no_grad()
def mc_scores(encoded, count=16, seed=123, device="cpu", block=16, pair_chunk=256,
              intervention="learned"):
    g = torch.Generator(device=device).manual_seed(seed)
    draws = []
    for kind in ("text", "media"):
        mu, lv = encoded[kind + "_mu"].to(device), encoded[kind + "_logvar"].to(device)
        if intervention == "constant":
            lv = lv.exp().mean(0, keepdim=True).log().expand_as(lv)
        elif intervention == "shuffled":
            # Dedicated generator preserves identical MC epsilon for fair interventions.
            perm = torch.randperm(len(lv), generator=torch.Generator().manual_seed(seed + 999))
            lv = lv[perm.to(device)]
        elif intervention == "zero":
            lv = torch.full_like(lv, -torch.inf)
        elif intervention != "learned":
            raise ValueError(intervention)
        draws.append(sample(mu, lv, count, g))
    scores = torch.empty(len(draws[0]), len(draws[1]))
    scale = torch.tensor(encoded["scale"], device=device)
    shift = torch.tensor(encoded["shift"], device=device)
    for i in range(0, len(draws[0]), block):
        for j in range(0, len(draws[1]), block):
            lp, _ = matrix_log_probs(draws[0][i:i+block], draws[1][j:j+block], scale, shift,
                                     pair_chunk, encoded.get("sigmoid_factor", 1.))
            scores[i:i+block, j:j+block] = lp.exp().cpu()
    return scores


def evaluate_encoded(encoded, count=16, seeds=(123, 124), device="cpu", interventions=True,
                     max_pairs=10_000_000, pair_chunk=256):
    n = len(encoded["text_mu"]) * len(encoded["media_mu"])
    if n > max_pairs:
        raise ValueError(f"Gallery has {n} pairs; set --max-pairs explicitly or use a declared subset")
    if not seeds:
        raise ValueError("At least one MC seed is required")
    out = {"split": encoded["split"], "media_count": len(encoded["media_mu"]),
           "caption_count": len(encoded["text_mu"]), "mc_samples": count, "seeds": list(seeds)}
    if encoded["text_base"].shape[-1] == encoded["media_base"].shape[-1]:
        base = encoded["text_base"] @ encoded["media_base"].T
        out["base_cosine"] = bidirectional_metrics(base, encoded)[0]
    mean_scores = encoded["text_mu"] @ encoded["media_mu"].T
    out["mean_only"] = bidirectional_metrics(mean_scores, encoded)[0]
    all_scores = [mc_scores(encoded, count, s, device, pair_chunk=pair_chunk) for s in seeds]
    out["pcme_per_seed"] = [bidirectional_metrics(s, encoded)[0] for s in all_scores]
    teacher = torch.stack(all_scores).mean(0)
    out["pcme_ensemble"], hits = bidirectional_metrics(teacher, encoded)
    if len(all_scores) > 1:
        out["mc_repeatability"] = score_comparison(all_scores[0], all_scores[1], encoded)
    for kind, h in zip(("text", "media"), hits):
        out[kind + "_uncertainty"] = sigma_diagnostics(encoded[kind + "_mu"], encoded[kind + "_logvar"], h)
    # These calibration metrics refer to observed dataset labels, not semantic truth.
    pos = encoded["text_ids"][:, None].eq(encoded["media_ids"][None])
    error = (teacher - pos.float()).square()
    out["observed_label_brier"] = float(error.mean())
    out["balanced_observed_label_brier"] = float(.5 * (error[pos].mean() + error[~pos].mean()))
    if interventions:
        for variant in ("zero", "constant", "shuffled"):
            values = [mc_scores(encoded, count, s, device, pair_chunk=pair_chunk, intervention=variant) for s in seeds]
            avg = torch.stack(values).mean(0)
            out["sigma_" + variant] = bidirectional_metrics(avg, encoded)[0]
            out["sigma_" + variant + "_effect"] = score_comparison(avg, teacher, encoded)
    return out, teacher


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = RetrievalModel(checkpoint["config"], checkpoint["dimensions"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--data", help="Override split cache or manifest path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=[123, 124, 125])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-pairs", type=int, default=10_000_000)
    p.add_argument("--export", help="Export frozen Gaussian representations for PolySim")
    p.add_argument("--export-only", action="store_true", help="Encode without expensive all-pairs MC evaluation")
    p.add_argument("--pair-chunk", type=int, default=128)
    p.add_argument("--output")
    p.add_argument("--no-interventions", action="store_true")
    args = p.parse_args()
    if args.export_only and not args.export:
        p.error("--export-only requires --export")
    if not args.export_only and not args.output:
        p.error("Evaluation requires --output")
    if min(args.samples, args.pair_chunk, args.batch_size, args.max_pairs) < 1:
        p.error("Sample and chunk sizes must be positive")
    model, checkpoint = load_model(args.checkpoint, args.device)
    config = checkpoint["config"].copy()
    if args.data:
        config[args.split + "_cache" if config.get("mode", "cache") == "cache" else "manifest"] = args.data
    ds = make_dataset(config, args.split)
    if args.split != "train" and set(ds.group_ids) & set(checkpoint["train_group_ids"]):
        raise ValueError("Evaluation gallery overlaps training IDs")
    encoded = encode(model, ds, args.device, args.batch_size)
    from .data import fingerprint
    encoded.update(checkpoint_sha256=fingerprint(args.checkpoint),
                   objective=config.get("objective", "pcme"))
    if args.export:
        Path(args.export).parent.mkdir(parents=True, exist_ok=True)
        torch.save(encoded, args.export)
    if args.export_only:
        print(f"Exported {len(encoded['media_mu'])} media and {len(encoded['text_mu'])} captions")
        return
    report, _ = evaluate_encoded(encoded, args.samples, args.seeds, args.device,
                                 not args.no_interventions, args.max_pairs, args.pair_chunk)
    report.update(checkpoint_sha256=encoded["checkpoint_sha256"], source_sha256=ds.source_hash,
                  objective=encoded["objective"], feature_kind=ds.cache.get("feature_kind") if ds.cache else "online_clip")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report["pcme_ensemble"], indent=2))


if __name__ == "__main__":
    main()
