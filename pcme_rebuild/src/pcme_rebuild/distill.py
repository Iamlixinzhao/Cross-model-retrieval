"""Freeze PCME completely; fit only order-bilinear parameters to its MC probabilities."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .evaluate import bidirectional_metrics, mc_scores, score_comparison
from .polynomial import ChebyshevDensity, OrderBilinear, density_diagnostics
from .probability import pair_log_probs, sample
from .train import atomic_save


def read_export(path):
    e = torch.load(path, weights_only=True, map_location="cpu")
    if "checkpoint_sha256" not in e or e.get("objective") != "pcme":
        raise ValueError("Use evaluate --export from a probabilistic checkpoint")
    for kind in ("text", "media"):
        mu, lv = e[kind+"_mu"], e[kind+"_logvar"]
        if mu.shape != lv.shape or mu.ndim != 2 or not torch.isfinite(mu).all() or not torch.isfinite(lv).all():
            raise ValueError("Invalid Gaussian export")
    return e


def pair_indices(e, count, generator):
    if len(e["media_ids"]) < 2:
        raise ValueError("Need at least two gallery items")
    ti = torch.randint(len(e["text_mu"]), (count,), generator=generator)
    # Encoded media rows are unique; IDs need not equal their row positions.
    lookup = {int(k): i for i, k in enumerate(e["media_ids"])}
    mi = torch.tensor([lookup[int(e["text_ids"][i])] for i in ti])
    negative = torch.arange(count) >= count//2
    offset = torch.randint(1, len(e["media_ids"]), (int(negative.sum()),), generator=generator)
    mi[negative] = (mi[negative]+offset) % len(e["media_ids"])
    return ti, mi


@torch.no_grad()
def pair_targets(e, ti, mi, count, device, generator, chunk=64):
    out = []
    scale, shift = (torch.tensor(e[k], device=device) for k in ("scale", "shift"))
    for i in range(0, len(ti), chunk):
        t, m = ti[i:i+chunk], mi[i:i+chunk]
        tx = sample(e["text_mu"][t].to(device), e["text_logvar"][t].to(device), count, generator)
        mx = sample(e["media_mu"][m].to(device), e["media_logvar"][m].to(device), count, generator)
        lp, _ = pair_log_probs(tx, mx, scale, shift, e.get("sigmoid_factor", 1.))
        out.append(lp.exp())
    return torch.cat(out)


@torch.no_grad()
def features(layer, e, chunk=64):
    out = []
    for kind in ("text", "media"):
        out.append(torch.cat([layer(e[kind+"_mu"][i:i+chunk], e[kind+"_logvar"][i:i+chunk])
                              for i in range(0, len(e[kind+"_mu"]), chunk)]))
    return out


def check_splits(train, val):
    if train["split"] != "train" or val["split"] != "val":
        raise ValueError("Fit uses train only; model selection uses val only")
    if set(train["group_ids"]) & set(val["group_ids"]):
        raise ValueError("Train/val media ID leakage")
    if train["checkpoint_sha256"] != val["checkpoint_sha256"]:
        raise ValueError("Exports must use the same frozen PCME checkpoint")


def fit(args):
    train, val = read_export(args.train), read_export(args.val)
    check_splits(train, val)
    torch.manual_seed(args.seed)
    layer = ChebyshevDensity.from_training(train, args.degree, args.nodes)
    tc, mc = features(layer, train)
    vtc, vmc = features(layer, val)
    model = OrderBilinear(args.degree, args.mu_residual).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(args.seed)
    mg = torch.Generator(device=args.device).manual_seed(args.seed+1)
    vti, vmi = pair_indices(val, args.validation_pairs, torch.Generator().manual_seed(739))
    target = pair_targets(val, vti, vmi, args.samples, args.device,
                          torch.Generator(device=args.device).manual_seed(740))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "best.pt").exists():
        raise ValueError("Choose a new distillation output directory")
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total = 0.
        for _ in range(args.steps_per_epoch):
            ti, mi = pair_indices(train, args.batch_pairs, g)
            # MC remains a fixed teacher. No gradient can flow into exported mu/logvar.
            y = pair_targets(train, ti, mi, args.samples, args.device, mg)
            logits = model.aligned_logits(tc[ti].to(args.device), mc[mi].to(args.device),
                                          train["text_mu"][ti].to(args.device), train["media_mu"][mi].to(args.device))
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            opt.step()
            total += float(loss)
        model.eval()
        with torch.no_grad():
            pred = model.aligned_logits(vtc[vti].to(args.device), vmc[vmi].to(args.device),
                                        val["text_mu"][vti].to(args.device), val["media_mu"][vmi].to(args.device)).sigmoid()
            mse = float((pred-target).square().mean())
        log = {"epoch": epoch, "train_soft_BCE": total/args.steps_per_epoch,
               "val_balanced_teacher_MSE": mse}
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(log, allow_nan=False)+"\n")
        if mse < best:
            best = mse
            atomic_save(dict(layer=layer.state_dict(), model=model.state_dict(), degree=args.degree,
                             nodes=args.nodes, mu_residual=args.mu_residual, checkpoint_sha256=train["checkpoint_sha256"],
                             train_group_ids=train["group_ids"], val_group_ids=val["group_ids"],
                             train_source_sha256=train["source_sha256"], val_source_sha256=val["source_sha256"],
                             config=vars(args), epoch=epoch, best=mse), out / "best.pt")
        print(json.dumps(log), flush=True)
    (out / "density_train.json").write_text(json.dumps(density_diagnostics(layer, train), indent=2, allow_nan=False))
    (out / "density_val.json").write_text(json.dumps(density_diagnostics(layer, val), indent=2, allow_nan=False))


@torch.no_grad()
def evaluate(args):
    e = read_export(args.embeddings)
    ck = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    if e["checkpoint_sha256"] != ck["checkpoint_sha256"]:
        raise ValueError("Wrong PCME teacher checkpoint")
    if e["split"] != "train" and set(e["group_ids"]) & set(ck["train_group_ids"]):
        raise ValueError("Evaluation media overlap distillation training")
    if e["split"] == "test" and set(e["group_ids"]) & set(ck["val_group_ids"]):
        raise ValueError("Test media overlap model-selection validation")
    if len(e["text_mu"])*len(e["media_mu"]) > args.max_pairs:
        raise ValueError("Gallery exceeds --max-pairs; use a declared subset or explicitly raise limit")
    layer = ChebyshevDensity(ck["layer"]["center"], ck["layer"]["halfwidth"], ck["degree"], ck["nodes"])
    layer.load_state_dict(ck["layer"])
    model = OrderBilinear(ck["degree"], ck["mu_residual"])
    model.load_state_dict(ck["model"])
    tc, mc = features(layer, e)
    query, database = model.mvm_features(tc, mc, e["text_mu"], e["media_mu"])
    logits = query @ database.T
    pred = logits.sigmoid()
    refs = [mc_scores(e, args.samples, s, args.device) for s in args.seeds]
    teacher = torch.stack(refs).mean(0)
    report = {"split": e["split"], "mc_samples": args.samples, "mc_seeds": args.seeds,
              "density": density_diagnostics(layer, e),
              "surrogate_vs_teacher": score_comparison(pred, teacher, e),
              "polysim": bidirectional_metrics(logits, e)[0],
              "pcme": bidirectional_metrics(teacher, e)[0],
              "mean_only": bidirectional_metrics(e["text_mu"] @ e["media_mu"].T, e)[0],
              "mu_residual": ck["mu_residual"], "gamma": float(model.gamma),
              "order_matrix_eigenvalues": torch.linalg.eigvalsh(model.matrix).tolist(),
              "checkpoint_sha256": e["checkpoint_sha256"]}
    if len(refs) > 1:
        report["teacher_MC_repeatability"] = score_comparison(refs[0], refs[1], e)
    # Verify whether sigma contributes after approximation; use same fixed intervals/A.
    for intervention in ("constant", "shuffled"):
        modified = e.copy()
        for kind in ("text", "media"):
            lv = e[kind+"_logvar"]
            if intervention == "constant":
                lv = lv.exp().mean(0, keepdim=True).log().expand_as(lv)
            else:
                lv = lv[torch.randperm(len(lv), generator=torch.Generator().manual_seed(917))]
            modified[kind+"_logvar"] = lv
        a, b = features(layer, modified)
        scores = model.score_matrix(a, b, e["text_mu"], e["media_mu"])
        report["sigma_"+intervention] = bidirectional_metrics(scores, e)[0]
        report["sigma_"+intervention+"_vs_teacher"] = score_comparison(scores.sigmoid(), teacher, e)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
    if args.export_mvm:
        Path(args.export_mvm).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(query=query, database=database, text_ids=e["text_ids"], media_ids=e["media_ids"],
                        group_ids=e["group_ids"], split=e["split"], checkpoint_sha256=e["checkpoint_sha256"],
                        score="query @ database.T (logits); sigmoid only for probabilities",
                        hardware_claim="Algebraic MVM mapping only; no hardware latency/energy claim"), args.export_mvm)
    print(json.dumps(report["surrogate_vs_teacher"], indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit")
    for arg in ("train", "val", "output"):
        f.add_argument("--"+arg, required=True)
    f.add_argument("--degree", type=int, default=5)
    f.add_argument("--nodes", type=int, default=128)
    f.add_argument("--mu-residual", action="store_true")
    f.add_argument("--epochs", type=int, default=30)
    f.add_argument("--steps-per-epoch", type=int, default=20)
    f.add_argument("--batch-pairs", type=int, default=128)
    f.add_argument("--validation-pairs", type=int, default=2048)
    f.add_argument("--lr", type=float, default=.01)
    f.add_argument("--seed", type=int, default=17)
    e = sub.add_parser("evaluate")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--embeddings", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--export-mvm")
    e.add_argument("--seeds", nargs="+", type=int, default=[123, 124, 125])
    e.add_argument("--max-pairs", type=int, default=10_000_000)
    for sp in (e, f):
        sp.add_argument("--samples", type=int, default=16)
        sp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    for key in ("samples", "epochs", "steps_per_epoch", "batch_pairs", "validation_pairs"):
        if hasattr(args, key) and getattr(args, key) < 1:
            p.error(f"{key} must be positive")
    if args.command == "fit":
        fit(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
