"""One-GPU or torchrun DDP trainer; global differentiable candidates in both directions."""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .data import Collator, make_dataset, require_disjoint, to_device
from .evaluate import bidirectional_metrics, encode, mc_scores, sigma_diagnostics
from .model import RetrievalModel
from .probability import (gaussian_kl, matching_nll, matrix_log_probs,
                          multi_positive_nce, sample, uniformity)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gather_features(t):
    return torch.cat(all_gather(t.contiguous())) if dist.is_initialized() else t


def gather_ids(t):
    if not dist.is_initialized():
        return t
    result = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(result, t)
    return torch.cat(result)


def objective(output, batch, config):
    tm, tv = output["text_mu"], output["text_logvar"]
    mm, mv = output["media_mu"], output["media_logvar"]
    tid, mid = batch["text_ids"], batch["media_ids"]
    gt, gm = gather_ids(tid), gather_ids(mid)
    mode = config.get("objective", "pcme")
    if mode == "deterministic":
        mt, mmall = gather_features(tm), gather_features(mm)
        temperature = config.get("temperature", .07)
        primary = .5 * (multi_positive_nce(tm @ mmall.T / temperature, tid[:, None].eq(gm)) +
                        multi_positive_nce(mm @ mt.T / temperature, mid[:, None].eq(gt)))
        return primary, {"match_loss": primary.detach(), "kl": primary.detach()*0,
                         "uniformity": primary.detach()*0}
    tx = sample(tm, tv, config.get("train_samples", 4))
    mx = sample(mm, mv, config.get("train_samples", 4))
    # Gather the SAME draws, not a second random sample on each rank.
    all_tx, all_mx = gather_features(tx), gather_features(mx)
    kwargs = {"pair_chunk": config.get("pair_chunk", 128), "factor": config.get("sigmoid_factor", 1.)}
    lp, ln = matrix_log_probs(tx, all_mx, output["scale"], output["shift"], **kwargs)
    rp, rn = matrix_log_probs(mx, all_tx, output["scale"], output["shift"], **kwargs)
    primary = .5 * (matching_nll(lp, ln, tid[:, None].eq(gm), config.get("pair_reduction", "balanced")) +
                    matching_nll(rp, rn, mid[:, None].eq(gt), config.get("pair_reduction", "balanced")))
    kl = .5 * (gaussian_kl(tm, tv) + gaussian_kl(mm, mv))
    ul = uniformity(torch.cat((tx, mx))) if config.get("uniform_weight", 0) else kl * 0
    loss = primary + config.get("kl_beta", 1e-4) * kl + config.get("uniform_weight", 0) * ul
    return loss, {"match_loss": primary.detach(), "kl": kl.detach(), "uniformity": ul.detach()}


def rng_state():
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


def atomic_save(obj, path):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(obj, temp)
    temp.replace(path)


def validate_config(config):
    if config.get("objective", "pcme") not in ("pcme", "deterministic"):
        raise ValueError("objective must be pcme or deterministic")
    for key in ("epochs", "batch_size", "captions_per_media", "train_samples", "pair_chunk", "dim", "hidden"):
        if config.get(key, 1) < 1:
            raise ValueError(f"{key} must be positive")
    if config.get("kl_beta", 1e-4) < 0 or config.get("uniform_weight", 0) < 0:
        raise ValueError("Regularization weights must be nonnegative")
    if config.get("amp", "none") not in ("none", "bf16"):
        raise ValueError("amp supports none/bf16; probability arithmetic always uses fp32")
    if config.get("sigmoid_factor", 1.) not in (1., 2.):
        raise ValueError("sigmoid_factor must be 1 (paper notation) or 2 (official logit convention)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", help="Epoch-boundary checkpoint; same configuration and world size required")
    p.add_argument("--init-from", help="Weights only, e.g. token-cache heads -> online CLIP finetune")
    p.add_argument("--set", action="append", default=[], help='Override key with JSON value, e.g. --set kl_beta=0')
    args = p.parse_args()
    if args.resume and args.init_from:
        p.error("Use only one of --resume and --init-from")
    config = json.loads(Path(args.config).read_text())
    for item in args.set:
        key, value = item.split("=", 1)
        config[key] = json.loads(value)
    validate_config(config)
    world, rank, local = (int(os.environ.get(k, d)) for k, d in (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0)))
    use_cuda = torch.cuda.is_available() and config.get("device", "auto") != "cpu"
    device = torch.device("cuda", local) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl" if use_cuda else "gloo")
    seed = config.get("seed", 17)
    seed_everything(seed)  # identical initialization, then rank-specific sampling below
    train = make_dataset(config, "train", True)
    val = make_dataset(config, "val")
    require_disjoint(train, val)
    if train.dimensions != val.dimensions:
        raise ValueError("Train and validation feature dimensions differ")
    if train.cache is not None and val.cache is not None:
        for key in ("backbone", "feature_kind"):
            if train.cache.get(key) != val.cache.get(key):
                raise ValueError(f"Train/val cache {key} differs")
    if len(train) < config["batch_size"] * world or config["batch_size"] * world < 2:
        raise ValueError("Need >= one complete global batch and >=2 distinct media per global batch")
    model = RetrievalModel(config, train.dimensions).to(device)
    if args.init_from:
        ck = torch.load(args.init_from, weights_only=True, map_location="cpu")
        if ck["dimensions"] != train.dimensions or ck["config"].get("dim", 128) != config.get("dim", 128):
            raise ValueError("Warm-start feature and embedding dimensions must agree")
        if ck["config"].get("mode", "cache") == "clip" and config.get("mode", "cache") == "clip":
            if ck["config"]["clip_name"] != config["clip_name"]:
                raise ValueError("Online warm start requires the same CLIP backbone")
            model.load_state_dict(ck["model"])  # Retain already finetuned backbone weights.
        else:
            if ck["config"].get("mode", "cache") == "clip":
                raise ValueError("Online -> cache warm start needs fresh features from the finetuned backbone")
            if config.get("mode") == "clip" and ck.get("backbone_source") != config["clip_name"]:
                raise ValueError("Token cache must originate from the same CLIP backbone")
            state = {k: v for k, v in ck["model"].items() if not k.startswith("backbone.")}
            missing, unexpected = model.load_state_dict(state, strict=False)
            if unexpected or any(not k.startswith("backbone.") for k in missing):
                raise ValueError("Warm start requires matching head architecture/objective")
    net = DDP(model, device_ids=[local] if use_cuda else None) if world > 1 else model
    groups = [dict(params=[p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")],
                   lr=config.get("lr", 2e-4)),
              dict(params=[p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")],
                   lr=config.get("backbone_lr", 1e-6))]
    optimizer = torch.optim.AdamW(groups, weight_decay=config.get("weight_decay", 0.))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True, seed=seed, drop_last=True)
    loader = DataLoader(train, batch_size=config["batch_size"], sampler=sampler, drop_last=True,
                        num_workers=config.get("workers", 0), pin_memory=use_cuda, persistent_workers=False,
                        collate_fn=Collator(config.get("clip_name") if train.mode == "clip" else None))
    output_dir = Path(args.output)
    start, best = 0, -float("inf")
    seed_everything(seed + rank)
    if args.resume:
        ck = torch.load(args.resume, weights_only=True, map_location="cpu")
        if ck["config"] != config or ck["world_size"] != world:
            raise ValueError("Resume needs identical config/world size; use --init-from for a new phase")
        if ck["train_sha256"] != train.source_hash or ck["val_sha256"] != val.source_hash:
            raise ValueError("Data changed since checkpoint")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        restore_rng(ck["rng_by_rank"][rank])
        start, best = ck["epoch"] + 1, ck["best"]
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / "last.pt").exists() and not args.resume:
            raise ValueError("Output already contains a run; choose another directory")
        (output_dir / "config.json").write_text(json.dumps(config, indent=2))
        metadata = {"torch": str(torch.__version__), "device": str(device), "world_size": world,
                    "global_media_batch": world * config["batch_size"],
                    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "train_sha256": train.source_hash, "val_sha256": val.source_hash,
                    "feature_kind": train.cache.get("feature_kind") if train.cache else "online_clip"}
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    for epoch in range(start, config["epochs"]):
        model.train()
        train.epoch = epoch
        sampler.set_epoch(epoch)
        begin = time.monotonic()
        totals = torch.zeros(5, device=device)
        for batch in loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=config.get("amp", "none") == "bf16"):
                output = net(batch)
            # Keep distances, exponentials, KL, all reductions in fp32.
            loss, terms = objective(output, batch, config)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; inspect variance guards and scale")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 2.), error_if_nonfinite=True)
            optimizer.step()
            totals += torch.stack([loss.detach(), terms["match_loss"], terms["kl"], terms["uniformity"], grad_norm.detach()])
        scheduler.step()
        if world > 1:
            dist.all_reduce(totals)
        totals /= len(loader) * world
        # Save training RNG BEFORE validation; validation cannot perturb the next epoch.
        state = rng_state()
        states = [None for _ in range(world)]
        if world > 1:
            dist.all_gather_object(states, state)
        else:
            states = [state]
        if rank == 0:
            encoded = encode(model, val, device, config.get("eval_batch_size", 16))
            n_pairs = len(encoded["text_mu"]) * len(encoded["media_mu"])
            if n_pairs > config.get("max_eval_pairs", 10_000_000):
                raise ValueError("Validation gallery exceeds max_eval_pairs")
            if config.get("objective", "pcme") == "deterministic":
                scores = encoded["text_mu"] @ encoded["media_mu"].T
            else:
                scores = mc_scores(encoded, config.get("eval_samples", 8), seed=731, device=device,
                                   pair_chunk=config.get("pair_chunk", 128))
            metrics, _ = bidirectional_metrics(scores, encoded)
            score = metrics["mean_R@1"]
            improved = score > best
            best = max(score, best)
            log = dict(epoch=epoch, seconds=time.monotonic()-begin,
                       **dict(zip(("loss", "match_loss", "kl", "uniformity", "grad_norm"), totals.cpu().tolist())),
                       validation=metrics, scale=encoded["scale"], shift=encoded["shift"])
            for kind in ("text", "media"):
                log[kind + "_uncertainty"] = sigma_diagnostics(encoded[kind+"_mu"], encoded[kind+"_logvar"])
            with (output_dir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(log, allow_nan=False) + "\n")
            ck = dict(schema_version=1, config=config, dimensions=train.dimensions,
                      model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                      epoch=epoch, best=best, world_size=world, rng_by_rank=states,
                      backbone_source=train.cache.get("backbone") if train.cache else config.get("clip_name"),
                      train_group_ids=train.group_ids, train_sha256=train.source_hash, val_sha256=val.source_hash)
            atomic_save(ck, output_dir / "last.pt")
            if improved:
                atomic_save(ck, output_dir / "best.pt")
            print(f"epoch={epoch+1} loss={float(totals[0]):.5f} val_mean_R1={score:.2f}", flush=True)
        restore_rng(state)
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
