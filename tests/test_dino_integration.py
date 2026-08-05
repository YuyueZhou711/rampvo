import json
from pathlib import Path

import pytest
import torch

from ramp.checkpoints import load_ramp_state_dict
from ramp.net import VONet
from train import normalize_checkpoint_state


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA extensions")
def test_phase_one_side_branch_preserves_ramp_outputs():
    baseline_config = json.loads(
        (ROOT / "config_net/MultiScale_TartanEvent.json").read_text()
    )["data_loader"]["train"]["args"]
    dino_config = json.loads(
        (ROOT / "config_net/MultiScale_TartanEvent_DINOv2.json").read_text()
    )["data_loader"]["train"]["args"]
    dino_config["dino"]["checkpoint"] = str(
        ROOT / dino_config["dino"]["checkpoint"]
    )
    checkpoint = torch.load(
        ROOT / "checkpoints/RAMPVO_MultiScale.pth",
        map_location="cpu",
        weights_only=False,
    )
    state = normalize_checkpoint_state(checkpoint)

    torch.manual_seed(1234)
    baseline = VONet(baseline_config).cuda().eval()
    baseline.load_state_dict(state, strict=True)
    torch.manual_seed(1234)
    with_dino = VONet(dino_config).cuda().eval()
    incompatible = load_ramp_state_dict(with_dino, state)
    assert incompatible.missing_keys
    assert all(
        key.startswith("patchify.dino_branch.")
        for key in incompatible.missing_keys
    )
    assert incompatible.unexpected_keys == []

    events = torch.linspace(-1, 1, 1 * 2 * 5 * 64 * 64, device="cuda").reshape(
        1, 2, 5, 64, 64
    )
    images = torch.linspace(
        -0.5, 1.5, 1 * 2 * 3 * 64 * 64, device="cuda"
    ).reshape(1, 2, 3, 64, 64)
    mask = torch.ones(1, 2, dtype=torch.bool, device="cuda")
    inputs = (events, images, mask)

    with torch.no_grad():
        baseline_output = baseline.patchify(
            inputs,
            patches_per_image=8,
            selector_mode="event",
            reinit_hidden=True,
        )
        dino_output = with_dino.patchify(
            inputs,
            patches_per_image=8,
            selector_mode="event",
            reinit_hidden=True,
        )

    for baseline_tensor, dino_tensor in zip(
        baseline_output.as_legacy_tuple(), dino_output.as_legacy_tuple()
    ):
        torch.testing.assert_close(baseline_tensor, dino_tensor, rtol=0, atol=0)
    assert dino_output.context_dino.shape == (1, 2, 384, 16, 16)
    assert dino_output.prior_logits.shape == (1, 2, 1, 16, 16)
    assert dino_output.inv_depth.shape == (1, 2, 1, 16, 16)

    with torch.no_grad():
        online_output = with_dino.patchify(
            (events[:, :1], images[:, :1], mask[0, :1]),
            patches_per_image=8,
            selector_mode="event",
            reinit_hidden=True,
        )
    assert online_output.context_dino.shape == (1, 1, 384, 16, 16)
