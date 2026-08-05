import json
from pathlib import Path

import torch

from ramp.net import VONet
from train import normalize_checkpoint_state


ROOT = Path(__file__).resolve().parents[1]


def test_phase_zero_has_no_dino_parameters_and_strict_loads_legacy_checkpoint():
    config = json.loads(
        (ROOT / "config_net/MultiScale_TartanEvent.json").read_text()
    )
    train_config = config["data_loader"]["train"]["args"]
    network = VONet(cfg=train_config)
    checkpoint = torch.load(
        ROOT / "checkpoints/RAMPVO_MultiScale.pth",
        map_location="cpu",
        weights_only=False,
    )
    state = normalize_checkpoint_state(checkpoint)

    assert not any("dino" in name.lower() for name in network.state_dict())
    incompatible = network.load_state_dict(state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
