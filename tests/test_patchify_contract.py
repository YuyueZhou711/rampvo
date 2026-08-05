import json
from pathlib import Path

import torch
import yaml

from ramp.model_types import PatchifyOutput, SelectorMode, resolve_selector_mode
from ramp.utils import get_coords_from_topk_events


ROOT = Path(__file__).resolve().parents[1]


def test_patchify_output_has_stable_named_contract():
    values = tuple(object() for _ in range(7))
    output = PatchifyOutput(*values)

    assert output.as_legacy_tuple() == values[:6]
    assert output.coords is values[6]
    assert output.prior_logits is None
    assert output.inv_depth is None
    assert output.raw_patch_inv_depth is None


def test_selector_mode_preserves_legacy_event_bias():
    assert resolve_selector_mode(event_bias=True) is SelectorMode.EVENT
    assert resolve_selector_mode(gradient_bias=True) is SelectorMode.GRADIENT
    assert resolve_selector_mode() is SelectorMode.RANDOM
    assert (
        resolve_selector_mode("event", event_bias=True)
        is SelectorMode.EVENT
    )


def test_selector_mode_rejects_conflicting_configuration():
    try:
        resolve_selector_mode("random", event_bias=True)
    except ValueError as error:
        assert "conflicts" in str(error)
    else:
        raise AssertionError("conflicting selector configuration was accepted")


def test_train_and_online_configs_use_the_same_explicit_k():
    network_config = json.loads(
        (ROOT / "config_net/MultiScale_TartanEvent.json").read_text()
    )
    train_config = network_config["data_loader"]["train"]["args"]
    online_config = yaml.safe_load((ROOT / "config_vo/default.yaml").read_text())

    assert train_config["patches_per_frame"] == 80
    assert online_config["PATCHES_PER_FRAME"] == 80
    assert train_config["train_selector_mode"] == "event"
    assert train_config["eval_selector_mode"] == "event"


def test_event_selector_keeps_single_frame_dimension():
    generator = torch.Generator().manual_seed(1234)
    events = torch.randn(1, 1, 5, 64, 64, generator=generator)

    coords = get_coords_from_topk_events(events, patches_per_image=8)

    assert coords.shape == (1, 8, 2)
    assert torch.isfinite(coords).all()
