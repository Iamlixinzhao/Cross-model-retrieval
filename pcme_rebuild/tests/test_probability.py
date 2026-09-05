import torch
from torch.nn import functional as F

from pcme_rebuild.probability import (gaussian_kl, matching_nll, matrix_log_probs,
                                     pair_log_probs, sample)


def test_probability_is_log_of_mean_and_not_mean_of_losses():
    x = torch.tensor([[[0.], [4.]]])
    y = torch.tensor([[[0.], [1.]]])
    scale, shift = torch.tensor(2.), torch.tensor(1.)
    lp, ln = pair_log_probs(x, y, scale, shift)
    logits = 1 - 2*((x[:, :, None]-y[:, None]).square().sum(-1)+1e-8).sqrt()
    assert torch.allclose(lp.exp(), logits.sigmoid().mean((1, 2)), atol=1e-7)
    assert torch.allclose(ln.exp(), (-logits).sigmoid().mean((1, 2)), atol=1e-7)
    assert not torch.allclose(-lp, F.softplus(-logits).mean((1, 2)))


def test_chunked_values_and_gradients_match_independent_broadcast_reference():
    torch.manual_seed(2)
    x = torch.randn(3, 2, 4, requires_grad=True)
    y = torch.randn(4, 2, 4, requires_grad=True)
    a = torch.tensor(1.3, requires_grad=True)
    b = torch.tensor(.7, requires_grad=True)
    lp, ln = matrix_log_probs(x, y, a, b, pair_chunk=5, factor=2.)
    d = ((x[:, None, :, None]-y[None, :, None]).square().sum(-1)+1e-8).sqrt()
    logits = 2*(b-a*d)
    rp = logits.sigmoid().mean((2, 3)).log()
    rn = (-logits).sigmoid().mean((2, 3)).log()
    assert torch.allclose(lp, rp, atol=2e-6)
    assert torch.allclose(ln, rn, atol=2e-6)
    gs = torch.autograd.grad((lp+ln).sum(), (x, y, a, b), retain_graph=True)
    refs = torch.autograd.grad((rp+rn).sum(), (x, y, a, b))
    for actual, expected in zip(gs, refs):
        assert torch.allclose(actual, expected, atol=1e-5)


def test_extreme_logits_finite_and_multi_positive_labels():
    x = torch.zeros(2, 2, 4, requires_grad=True)
    y = torch.ones(3, 2, 4, requires_grad=True)
    lp, ln = matrix_log_probs(x, y, torch.tensor(10000.), torch.tensor(-10000.), 2)
    positive = torch.tensor([[False, True, True], [True, False, False]])
    loss = matching_nll(lp, ln, positive)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_kl_uses_log_variance_and_opposes_collapse():
    mu = torch.zeros(2, 3)
    lv = torch.full((2, 3), -12., requires_grad=True)
    loss = gaussian_kl(mu, lv)
    normal = torch.distributions.Normal(mu, (.5*lv).exp())
    prior = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
    expected = torch.distributions.kl_divergence(normal, prior).sum(-1).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert (lv.grad < 0).all()


def test_draws_have_correct_variance_and_are_not_renormalized():
    mu = torch.tensor([[2., 0.]])
    lv = torch.tensor([[.25, 1.]]).log()
    x = sample(mu, lv, 20000, torch.Generator().manual_seed(1))[0]
    assert torch.allclose(x.mean(0), mu[0], atol=.025)
    assert torch.allclose(x.var(0), lv.exp()[0], atol=.03)
    assert not torch.allclose(x.norm(dim=-1), torch.ones(len(x)))
