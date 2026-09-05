"""Explicit convention everywhere: logvar = log(sigma ** 2)."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def sample(mu, logvar, count, generator=None):
    if count < 1:
        raise ValueError("MC sample count must be positive")
    eps = torch.randn((len(mu), count, mu.shape[-1]), device=mu.device,
                      dtype=torch.float32, generator=generator)
    # Do NOT normalize draws: normalized Gaussian draws are not Gaussian.
    return mu.float()[:, None] + (0.5 * logvar.float()).exp()[:, None] * eps


def pair_log_probs(x, y, scale, shift, factor=1.0):
    """Aligned pairs [P,K,D], all KxK combinations, stable log E[sigmoid]."""
    d2 = (x[:, :, None].float() - y[:, None, :].float()).square().sum(-1)
    logits = factor * (shift.float() - scale.float() * (d2 + 1e-8).sqrt())
    z = math.log(x.shape[1] * y.shape[1])
    return (F.logsigmoid(logits).flatten(1).logsumexp(1) - z,
            F.logsigmoid(-logits).flatten(1).logsumexp(1) - z)


def matrix_log_probs(x, y, scale, shift, pair_chunk=256, factor=1.0):
    """No [B,B,K,K,D] allocation; checkpoint each chunk during backward."""
    positive, negative = [], []
    for start in range(0, len(x) * len(y), pair_chunk):
        idx = torch.arange(start, min(start + pair_chunk, len(x) * len(y)), device=x.device)
        # Index inside checkpoint so expanded pair tensors are not retained.
        def block(a, b, s, t, indices):
            return pair_log_probs(a[indices // len(b)], b[indices % len(b)], s, t, factor)
        args = (x, y, scale, shift, idx)
        if torch.is_grad_enabled() and any(t.requires_grad for t in args[:4]):
            lp, ln = checkpoint(block, *args, use_reentrant=False)
        else:
            lp, ln = block(*args)
        positive.append(lp)
        negative.append(ln)
    return torch.cat(positive).view(len(x), len(y)), torch.cat(negative).view(len(x), len(y))


def matching_nll(logp, logn, positives, reduction="balanced"):
    if not positives.any() or not (~positives).any():
        raise ValueError("Each training batch needs both matched and unmatched pairs")
    if reduction == "balanced":
        # Deliberate small-batch variant: fixed 50/50 class prior across batch sizes.
        return -0.5 * (logp[positives].mean() + logn[~positives].mean())
    if reduction == "all_pairs":
        return -torch.where(positives, logp, logn).mean()
    raise ValueError(reduction)


def gaussian_kl(mu, logvar):
    """Mean over examples, SUM over dimensions; analytic KL to N(0,I)."""
    mu, logvar = mu.float(), logvar.float()
    return 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1).mean()


def uniformity(draws, max_points=256):
    x = draws.flatten(0, 1).float()
    if len(x) > max_points:
        x = x[torch.randperm(len(x), device=x.device)[:max_points]]
    if len(x) < 2:
        return x.sum() * 0
    values = -2 * torch.pdist(x).square()
    return values.logsumexp(0) - math.log(len(values))


def multi_positive_nce(scores, positive):
    if not positive.any(1).all():
        raise ValueError("Every anchor needs a labeled positive")
    return (scores.logsumexp(1) - scores.masked_fill(~positive, -torch.inf).logsumexp(1)).mean()


class MatchParameters(nn.Module):
    def __init__(self, scale=15., shift=15.):
        super().__init__()
        self.raw_scale = nn.Parameter(torch.tensor(math.log(math.expm1(scale))))
        self.shift = nn.Parameter(torch.tensor(shift))

    def forward(self):
        return F.softplus(self.raw_scale) + 1e-6, self.shift
