#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ramp.dinov2.branch import SMALL_CHECKPOINT_SHA256, verify_checkpoint
from ramp.third_party.depth_anything_v2 import DepthAnythingV2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sha256", default=SMALL_CHECKPOINT_SHA256)
    args = parser.parse_args()

    actual_hash = verify_checkpoint(args.checkpoint, args.sha256)
    model = DepthAnythingV2(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    result = {
        "checkpoint": str(Path(args.checkpoint)),
        "sha256": actual_hash,
        "state_tensors": len(state),
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
        "strict_load": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
