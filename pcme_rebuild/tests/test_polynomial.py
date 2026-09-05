import torch

from pcme_rebuild.polynomial import ChebyshevDensity, OrderBilinear, density_diagnostics


def test_density_reconstruction_and_change_of_variable():
    # A broad, resolved PDF should integrate to the retained mass and converge.
    layer = ChebyshevDensity(torch.tensor([1.]), torch.tensor([2.]), degree=24, nodes=128)
    mu, lv = torch.tensor([[1.2]]), torch.tensor([[.7**2]]).log()
    e = dict(text_mu=mu, media_mu=mu, text_logvar=lv, media_logvar=lv)
    report = density_diagnostics(layer, e)
    assert report["mean_marginal_L1_on_grid"] < 1e-5
    assert report["relative_coefficient_change_double_nodes"] < 1e-5
    assert 0 < report["mean_marginal_tail_mass"] < .02


def test_indefinite_symmetric_bilinear_has_exact_mvm_mapping():
    torch.manual_seed(4)
    layer = OrderBilinear(2, mu_residual=True)
    with torch.no_grad():
        layer.raw_matrix.copy_(torch.diag(torch.tensor([-2., 1., 3.])))
        layer.gamma.fill_(-.5)
        layer.bias.fill_(.25)
    tc, mc = torch.randn(4, 5, 3), torch.randn(6, 5, 3)
    tm, mm = torch.randn(4, 5), torch.randn(6, 5)
    actual = layer.score_matrix(tc, mc, tm, mm)
    expected = torch.stack([torch.stack([sum(tc[i, d] @ layer.matrix @ mc[j, d] for d in range(5))
                                        -.5*(tm[i]*mm[j]).sum()+.25 for j in range(6)]) for i in range(4)])
    assert torch.allclose(actual, expected, atol=1e-5)


def test_interval_is_fit_on_training_only_and_reports_unseen_tails():
    e = dict(text_mu=torch.zeros(2, 2), media_mu=torch.zeros(2, 2),
             text_logvar=torch.full((2, 2), -4.), media_logvar=torch.full((2, 2), -4.))
    layer = ChebyshevDensity.from_training(e)
    tail = layer.tail_mass(torch.full((1, 2), 100.), torch.zeros(1, 2))
    assert (tail > .99).all()
