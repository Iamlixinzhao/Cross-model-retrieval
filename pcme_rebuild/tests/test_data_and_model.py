import json

import pytest
import torch

from pcme_rebuild.data import Collator, GroupDataset, load_cache, read_manifest, require_disjoint
from pcme_rebuild.evaluate import bidirectional_metrics
from pcme_rebuild.model import GaussianEncoder


def write_cache(path, split="train", ids=("a", "b")):
    out = dict(schema_version=1, split=split, media_ids=list(ids), text_ids=[ids[0], ids[0], ids[1], ids[1]])
    for kind, count in (("media", 2), ("text", 4)):
        out[kind+"_tokens"] = torch.randn(count, 3, 4)
        out[kind+"_pool"] = torch.randn(count, 4)
        out[kind+"_mask"] = torch.ones(count, 3, dtype=torch.bool)
    torch.save(out, path)


def test_grouped_captions_and_split_leakage(tmp_path):
    a, b = tmp_path/"train.pt", tmp_path/"val.pt"
    write_cache(a)
    write_cache(b, "val")
    ds = GroupDataset("cache", a, "train", 2)
    batch = Collator()([ds[0], ds[1]])
    assert len(batch["media_ids"]) == 2 and len(batch["text_ids"]) == 4
    assert batch["text_ids"].tolist() == [0, 0, 1, 1]
    assert ds[0]["text_index"].unique().numel() == 2
    with pytest.raises(ValueError, match="leakage"):
        require_disjoint(ds, GroupDataset("cache", b, "val"))
    invalid = torch.load(a, weights_only=True)
    invalid["media_ids"] = ["a", "a"]
    torch.save(invalid, a)
    with pytest.raises(ValueError, match="unique"):
        load_cache(a)


def test_manifest_rejects_cross_split_image_alias(tmp_path):
    rows = [dict(id="a", image="same.jpg", caption="one", split="train"),
            dict(id="b", image="same.jpg", caption="two", split="test")]
    path = tmp_path/"manifest.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="Same image"):
        read_manifest(path)


def test_attention_mask_and_variance_branch_gradients():
    torch.manual_seed(7)
    model = GaussianEncoder(4, 4, 4, 4)
    tokens = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    pool = torch.randn(2, 4)
    mu, lv = model(tokens, pool, mask)
    altered = tokens.clone()
    altered[~mask] = 10000
    other_mu, other_lv = model(altered, pool, mask)
    assert torch.allclose(mu, other_mu)
    assert torch.allclose(lv, other_lv)
    (lv.exp().sum()+mu[:, 0].sum()).backward()
    assert model.var_attention.score[0].weight.grad.abs().sum() > 0
    assert model.mu_attention.score[0].weight.grad.abs().sum() > 0


def test_multiple_caption_retrieval_uses_all_positives():
    e = dict(text_ids=torch.tensor([7, 7, 9]), media_ids=torch.tensor([9, 7]))
    scores = torch.tensor([[0., 1.], [0., 2.], [2., 0.]])
    metrics, _ = bidirectional_metrics(scores, e)
    assert metrics["mean_R@1"] == 100.
    assert metrics["text_to_media"]["mAP"] == 1.
    assert metrics["media_to_text"]["mAP"] == 1.
