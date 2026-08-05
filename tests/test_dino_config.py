import pytest

from ramp.model_types import validate_dino_configuration


def test_dino_peak_requires_enabled_prior():
    with pytest.raises(ValueError, match="requires DINO and prior"):
        validate_dino_configuration(
            {
                "eval_selector_mode": "dino_peak",
                "dino": {"enabled": True, "prior": {"enabled": False}},
            }
        )


def test_scaled_depth_requires_enabled_depth_path():
    with pytest.raises(ValueError, match="requires depth.enabled"):
        validate_dino_configuration(
            {
                "dino": {
                    "enabled": True,
                    "depth": {"enabled": False, "mode": "scaled"},
                }
            }
        )


def test_mainline_rejects_unfrozen_pretrained_path():
    with pytest.raises(ValueError, match="freeze_pretrained"):
        validate_dino_configuration(
            {"dino": {"enabled": True, "freeze_pretrained": False}}
        )
