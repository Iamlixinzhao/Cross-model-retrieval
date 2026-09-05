"""Random tiny CLIP, offline: verify cache/online equivalence and selected-layer gradients."""
import pytest
import torch


def test_tiny_clip_cache_equivalence_and_unfrozen_gradients(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig
    from pcme_rebuild.model import RetrievalModel
    from pcme_rebuild.probability import gaussian_kl
    torch.set_num_threads(1)
    text = CLIPTextConfig(vocab_size=20, hidden_size=8, intermediate_size=16,
                          num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=5,
                          bos_token_id=0, eos_token_id=2, pad_token_id=1)
    vision = CLIPVisionConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                             num_attention_heads=2, image_size=8, patch_size=4)
    config = CLIPConfig(text_config=text.to_dict(), vision_config=vision.to_dict(), projection_dim=4)
    tiny = CLIPModel(config)
    monkeypatch.setattr(CLIPModel, "from_pretrained", lambda *a, **kw: tiny)
    cfg = dict(mode="clip", clip_name="offline-test", dim=4, hidden=4,
               unfreeze_last_n=1, gradient_checkpointing=True)
    model = RetrievalModel(cfg, {"text": [8, 4], "media": [8, 4]})
    batch = {"pixel_values": torch.randn(2, 3, 8, 8),
             "input_ids": torch.tensor([[0, 4, 5, 2, 1], [0, 6, 2, 1, 1]]),
             "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])}
    model.eval()
    with torch.no_grad():
        out = model(batch)
        features = model.clip_features(batch)
        cache_model = RetrievalModel({**cfg, "mode": "cache"}, {"text": [8, 4], "media": [8, 4]})
        cache_model.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")})
        cached = cache_model(features)
        for k in ("text_mu", "text_logvar", "media_mu", "media_logvar"):
            assert torch.allclose(out[k], cached[k], atol=1e-6)
    model.train()
    out = model(batch)
    loss = sum(gaussian_kl(out[k+"_mu"], out[k+"_logvar"]) + out[k+"_mu"][:, 0].sum()
               for k in ("text", "media"))
    loss.backward()
    for name in ("vision_model", "text_model"):
        tower = getattr(model.backbone, name)
        assert all(p.grad is None for p in tower.encoder.layers[0].parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tower.encoder.layers[-1].parameters())
