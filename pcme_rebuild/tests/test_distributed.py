"""CPU/Gloo gradient equivalence, no optimizer steps or training runs."""
from pathlib import Path
import os

import pytest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import pcme_rebuild.train as trainer
from pcme_rebuild.model import RetrievalModel


def fixed_draws(mu, lv, count):
    # Fixed reparameterization draws isolate distributed gradients from RNG ordering.
    eps = torch.linspace(-1., 1., count, device=mu.device)[None, :, None]
    return mu[:, None]+(.5*lv).exp()[:, None]*eps


def make_batch():
    torch.manual_seed(234)
    batch = {"text_ids": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]), "media_ids": torch.arange(4)}
    for kind, n in (("text", 8), ("media", 4)):
        batch[kind+"_tokens"] = torch.randn(n, 3, 4)
        batch[kind+"_pool"] = torch.randn(n, 4)
        batch[kind+"_mask"] = torch.ones(n, 3, dtype=torch.bool)
    return batch


def worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://"+rendezvous, rank=rank, world_size=2)
    torch.manual_seed(123)
    config = dict(dim=4, hidden=4, train_samples=2, pair_chunk=5, kl_beta=1e-4)
    model = RetrievalModel(config, {"text": [4, 4], "media": [4, 4]})
    net = DistributedDataParallel(model)
    batch = make_batch()
    local = {k: v[rank*(4 if k.startswith("text") else 2):(rank+1)*(4 if k.startswith("text") else 2)]
             for k, v in batch.items()}
    trainer.sample = fixed_draws
    loss, _ = trainer.objective(net(local), local, config)
    loss.backward()
    if rank == 0:
        torch.save({n: p.grad for n, p in model.named_parameters()}, output)
    dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("PCME_TEST_DDP") != "1", reason="Set PCME_TEST_DDP=1 on a host allowing Gloo sockets")
def test_two_process_pcme_gradients_equal_global_batch(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    config = dict(dim=4, hidden=4, train_samples=2, pair_chunk=5, kl_beta=1e-4)
    reference = RetrievalModel(config, {"text": [4, 4], "media": [4, 4]})
    batch = make_batch()
    monkeypatch.setattr(trainer, "sample", fixed_draws)
    loss, _ = trainer.objective(reference(batch), batch, config)
    loss.backward()
    path = str(tmp_path/"gradients.pt")
    mp.spawn(worker, args=(str(tmp_path/"rendezvous"), path), nprocs=2, join=True)
    actual = torch.load(path, weights_only=True)
    for name, p in reference.named_parameters():
        assert p.grad is not None, name
        assert actual[name] is not None, name
        assert torch.allclose(actual[name], p.grad, atol=3e-5, rtol=5e-4), name
