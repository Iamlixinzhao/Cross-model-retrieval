"""
ChebProjectorV2: 增强版 projector，用于提升检索。
- num_layers=1: dim -> hidden -> dim（与原始一致）
- num_layers=2: dim -> hidden -> hidden -> dim（更深）
- use_ln: 在 tanh 前加 LayerNorm，提高稳定性
"""
import torch
import torch.nn as nn


class ChebProjectorV2(nn.Module):
    """
    Adapter after ImageBind embedding.
    Output h in [-1, 1] for Chebyshev basis.
    num_layers=1: dim -> hidden -> dim
    num_layers=2: dim -> hidden -> hidden -> dim
    use_ln: LayerNorm before tanh.
    """
    def __init__(self, dim, hidden=None, residual=True, dropout=0.1, num_layers=1, use_ln=False):
        super().__init__()
        self.residual = residual
        self.use_ln = use_ln
        if hidden is None:
            hidden = dim

        layers = []
        layers.extend([
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(dim) if use_ln else None

    def forward(self, x):
        out = self.net(x)
        if self.residual:
            out = x + out
        if self.ln is not None:
            out = self.ln(out)
        h = torch.tanh(out)
        return h
