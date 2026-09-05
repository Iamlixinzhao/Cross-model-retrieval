"""Chebyshev marginal densities and an explicitly asymmetric MVM mapping."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ChebyshevDensity(nn.Module):
    def __init__(self, center, halfwidth, degree=5, nodes=128):
        super().__init__()
        if degree < 0 or nodes <= degree or (halfwidth <= 0).any():
            raise ValueError("Need degree>=0, nodes>degree and positive interval halfwidth")
        self.degree, self.nodes = degree, nodes
        self.register_buffer("center", center.float())
        self.register_buffer("halfwidth", halfwidth.float())
        theta = (torch.arange(nodes, dtype=torch.float64) + .5) * math.pi / nodes
        self.register_buffer("u", theta.cos().float())
        basis = torch.cos(torch.arange(degree+1)[:, None] * theta[None]) * (2./nodes)
        basis[0] *= .5  # f(u) = c0 + c1 T1(u) + ...; c0 is already halved.
        self.register_buffer("projection", basis.float())

    @classmethod
    def from_training(cls, encoded, degree=5, nodes=128, tail_sigmas=4.):
        mu = torch.cat([encoded[k+"_mu"] for k in ("text", "media")])
        sigma = torch.cat([(.5*encoded[k+"_logvar"]).exp() for k in ("text", "media")])
        lower = (mu-tail_sigmas*sigma).amin(0)
        upper = (mu+tail_sigmas*sigma).amax(0)
        return cls((lower+upper)/2, ((upper-lower)/2).clamp_min(1e-5), degree, nodes)

    def coefficients(self, mu, logvar):
        mean = (mu.float()-self.center)/self.halfwidth
        std = (.5*logvar.float()).exp()/self.halfwidth
        # Density in u=(x-center)/halfwidth, INCLUDING change-of-variable Jacobian.
        pdf = torch.exp(-.5*((self.u-mean[..., None])/std[..., None]).square()) / (std[..., None]*math.sqrt(2*math.pi))
        return pdf @ self.projection.T

    def forward(self, mu, logvar):
        c = self.coefficients(mu, logvar)
        # Explicit feature normalization; raw coefficients are used for fidelity diagnostics.
        return F.normalize(c.flatten(1), dim=-1).view_as(c)

    def tail_mass(self, mu, logvar):
        std = (.5*logvar).exp()
        lo, hi = self.center-self.halfwidth, self.center+self.halfwidth
        cdf = lambda x: .5*(1+torch.erf(x/math.sqrt(2)))
        return (cdf((lo-mu)/std) + 1-cdf((hi-mu)/std)).clamp(0, 1)


class OrderBilinear(nn.Module):
    def __init__(self, degree, mu_residual=False):
        super().__init__()
        self.raw_matrix = nn.Parameter(torch.eye(degree+1))
        self.bias = nn.Parameter(torch.tensor(-1.))
        self.mu_residual = mu_residual
        if mu_residual:
            self.gamma = nn.Parameter(torch.tensor(0.))
        else:
            self.register_buffer("gamma", torch.tensor(0.))

    @property
    def matrix(self):
        return (self.raw_matrix+self.raw_matrix.T)/2

    def aligned_logits(self, tc, mc, tm, mm):
        return ((tc @ self.matrix)*mc).sum((1, 2)) + self.gamma*(tm*mm).sum(-1) + self.bias

    def mvm_features(self, tc, mc, tm, mm):
        # Exact for ANY real A, including indefinite matrices. No sqrt(A).
        # Include bias as a constant feature; sigmoid is optional for ranking.
        query = torch.cat(((tc @ self.matrix).flatten(1), self.gamma*tm,
                           self.bias.expand(len(tc), 1)), dim=1)
        database = torch.cat((mc.flatten(1), mm, torch.ones(len(mc), 1, device=mc.device)), dim=1)
        return query, database

    def score_matrix(self, tc, mc, tm, mm):
        q, d = self.mvm_features(tc, mc, tm, mm)
        return q @ d.T


@torch.no_grad()
def density_diagnostics(layer, encoded, max_examples=32, max_dimensions=32):
    # Explicit sub-sampling: report scope; don't silently claim full-density certification.
    mu = torch.cat((encoded["text_mu"], encoded["media_mu"]))
    lv = torch.cat((encoded["text_logvar"], encoded["media_logvar"]))
    tails = layer.tail_mass(mu, lv)
    retention = (1-tails.double()).clamp_min(1e-300).log().sum(-1).exp()
    ix = torch.linspace(0, len(mu)-1, min(max_examples, len(mu))).long()
    dims = torch.linspace(0, mu.shape[1]-1, min(max_dimensions, mu.shape[1])).long()
    means, logvars = mu[ix], lv[ix]
    c = layer.coefficients(means, logvars)
    dense = ChebyshevDensity(layer.center, layer.halfwidth, layer.degree, layer.nodes*2)
    c2 = dense.coefficients(means, logvars)
    u = torch.linspace(-1, 1, 1025)
    basis = [torch.ones_like(u)]
    if layer.degree:
        basis.append(u)
    for k in range(2, layer.degree+1):
        basis.append(2*u*basis[-1]-basis[-2])
    approx = c[:, dims] @ torch.stack(basis)
    m = ((means-layer.center)/layer.halfwidth)[:, dims, None]
    s = ((.5*logvars).exp()/layer.halfwidth)[:, dims, None]
    pdf = torch.exp(-.5*((u-m)/s).square())/(s*math.sqrt(2*math.pi))
    l1 = torch.trapezoid((pdf-approx).abs(), u, dim=-1)
    true_grid_mass = torch.trapezoid(pdf, u, dim=-1)
    retained = (1-tails[ix][:, dims])
    min_std_u = float(((.5*lv).exp()/layer.halfwidth).min())
    return {"diagnostic_examples": len(ix), "diagnostic_dimensions": len(dims),
            "degree": layer.degree, "nodes": layer.nodes,
            "mean_marginal_tail_mass": float(tails.mean()), "max_marginal_tail_mass": float(tails.max()),
            "mean_joint_box_retained_mass": float(retention.mean()),
            "minimum_joint_box_retained_mass": float(retention.min()),
            "mean_marginal_L1_on_grid": float(l1.mean()), "max_marginal_L1_on_grid": float(l1.max()),
            "negative_density_grid_fraction": float((approx<0).float().mean()),
            "max_true_pdf_grid_mass_error": float((true_grid_mass-retained).abs().max()),
            "minimum_sigma_in_scaled_coordinate": min_std_u,
            "max_node_gap_over_min_sigma": (math.pi/layer.nodes)/max(min_std_u, 1e-30),
            "zero_coefficient_vector_fraction": float((c.flatten(1).norm(dim=-1)<1e-12).float().mean()),
            "relative_coefficient_change_double_nodes": float((c-c2).norm()/c2.norm().clamp_min(1e-12)),
            "scope": "Marginal PDF check on a grid, not a joint-density or PCME-score error bound"}
