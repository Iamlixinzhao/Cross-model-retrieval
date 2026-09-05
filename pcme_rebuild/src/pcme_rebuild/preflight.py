"""Read-only server/data checks. No training, no external dataset downloads."""
import argparse
import json
import platform
from collections import Counter
from pathlib import Path

import torch

from .data import make_dataset, require_disjoint
from .train import validate_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    validate_config(config)
    train, val = make_dataset(config, "train", True), make_dataset(config, "val")
    require_disjoint(train, val)
    report = {"python": platform.python_version(), "torch": str(torch.__version__),
              "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
              "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
              "dimensions": train.dimensions, "splits": {}}
    if train.dimensions != val.dimensions:
        raise ValueError("Feature dimensions differ between splits")
    for ds in (train, val):
        report["splits"][ds.split] = {"media": len(ds), "captions": len(ds.text_ids),
                                      "captions_per_media_histogram": dict(Counter(map(len, ds.caption_indices))),
                                      "sha256": ds.source_hash,
                                      "feature_kind": ds.cache.get("feature_kind") if ds.cache else "online_clip"}
    if config.get("amp") == "bf16" and torch.cuda.is_available():
        report["bf16_supported"] = torch.cuda.is_bf16_supported()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
