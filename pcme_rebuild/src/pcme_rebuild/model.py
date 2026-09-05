import math

import torch
from torch import nn
from torch.nn import functional as F

from .probability import MatchParameters


class AttentionPool(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, tokens, mask):
        logits = self.score(tokens).squeeze(-1).float().masked_fill(~mask.bool(), -torch.inf)
        weights = logits.softmax(-1).to(tokens.dtype)
        return (weights[..., None] * tokens).sum(1)


class GaussianEncoder(nn.Module):
    """Separate local attention for mean and uncertainty; pooled cache is an ablation."""
    def __init__(self, token_dim, pool_dim, dim=128, hidden=128, initial_sigma=.05,
                 deterministic=False):
        super().__init__()
        self.mu_attention = AttentionPool(token_dim, hidden)
        self.mu_local = nn.Linear(token_dim, dim)
        self.mu_global = nn.Linear(pool_dim, dim)
        self.mu_norm = nn.LayerNorm(dim)
        self.deterministic = deterministic
        if not deterministic:
            self.var_attention = AttentionPool(token_dim, hidden)
            self.var_local = nn.Linear(token_dim, dim)
            self.var_global = nn.Linear(pool_dim, dim)
            # Small nonzero weights let attention receive gradients immediately.
            nn.init.normal_(self.var_local.weight, std=.001)
            nn.init.normal_(self.var_global.weight, std=.001)
            nn.init.constant_(self.var_local.bias, math.log(initial_sigma ** 2))
            nn.init.zeros_(self.var_global.bias)

    def forward(self, tokens, pool, mask):
        mean = self.mu_global(pool) + torch.sigmoid(self.mu_local(self.mu_attention(tokens, mask)))
        mu = F.normalize(self.mu_norm(mean).float(), dim=-1)
        if self.deterministic:
            return mu, torch.full_like(mu, -30.)
        # No LN, sigmoid, L2 norm, target sigma, or enforced floor on the variance branch.
        raw = self.var_global(pool) + self.var_local(self.var_attention(tokens, mask))
        # Numerical guard only; diagnostics report occupancy of both bounds.
        return mu, raw.float().clamp(-30., 10.)


class RetrievalModel(nn.Module):
    def __init__(self, config, dimensions):
        super().__init__()
        self.config = config
        self.dimensions = dimensions
        self.mode = config.get("mode", "cache")
        self.backbone = None
        if self.mode == "clip":
            from transformers import CLIPModel
            self.backbone = CLIPModel.from_pretrained(config["clip_name"])
            self.backbone.requires_grad_(False)
            self.backbone.logit_scale.requires_grad_(False)
            for tower_name in ("vision_model", "text_model"):
                tower = getattr(self.backbone, tower_name)
                n = config.get("unfreeze_last_n", 0)
                if n > len(tower.encoder.layers) or n < 0:
                    raise ValueError("Invalid unfreeze_last_n")
                if n:
                    for layer in tower.encoder.layers[-n:]:
                        layer.requires_grad_(True)
            if config.get("gradient_checkpointing", False):
                self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        args = dict(dim=config.get("dim", 128), hidden=config.get("hidden", 128),
                    initial_sigma=config.get("initial_sigma", .05),
                    deterministic=config.get("objective") == "deterministic")
        self.text = GaussianEncoder(*dimensions["text"], **args)
        self.media = GaussianEncoder(*dimensions["media"], **args)
        self.match = MatchParameters(config.get("init_scale", 15.), config.get("init_shift", 15.))
        if args["deterministic"]:
            self.match.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        # Train mode is needed for activation checkpointing in unfrozen blocks.
        if self.backbone is not None:
            self.backbone.train(mode and self.config.get("unfreeze_last_n", 0) > 0)
        return self

    def clip_features(self, batch):
        vision = self.backbone.vision_model(pixel_values=batch["pixel_values"])
        text = self.backbone.text_model(input_ids=batch["input_ids"],
                                       attention_mask=batch["attention_mask"])
        vp = self.backbone.visual_projection(vision.pooler_output)
        tp = self.backbone.text_projection(text.pooler_output)
        return {"media_tokens": vision.last_hidden_state,
                "media_pool": vp,
                "media_mask": torch.ones(vision.last_hidden_state.shape[:2], device=vp.device, dtype=torch.bool),
                "text_tokens": text.last_hidden_state, "text_pool": tp,
                "text_mask": batch["attention_mask"].bool()}

    def forward(self, batch):
        f = self.clip_features(batch) if self.backbone is not None else batch
        tm, tv = self.text(f["text_tokens"], f["text_pool"], f["text_mask"])
        mm, mv = self.media(f["media_tokens"], f["media_pool"], f["media_mask"])
        scale, shift = self.match()
        return {"text_mu": tm, "text_logvar": tv, "media_mu": mm, "media_logvar": mv,
                "scale": scale, "shift": shift,
                "text_base": F.normalize(f["text_pool"].float(), dim=-1),
                "media_base": F.normalize(f["media_pool"].float(), dim=-1)}
