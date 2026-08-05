from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class SelectorMode(str, Enum):
    RANDOM = "random"
    GRADIENT = "gradient"
    EVENT = "event"
    GRID_RANDOM = "grid_random"
    DINO_PEAK = "dino_peak"


def resolve_selector_mode(
    selector_mode: Optional[str] = None,
    *,
    event_bias: Optional[bool] = None,
    gradient_bias: Optional[bool] = None,
) -> SelectorMode:
    """Resolve the new selector enum while preserving legacy boolean configs."""
    if event_bias and gradient_bias:
        raise ValueError("event_bias and gradient_bias are mutually exclusive")

    legacy_mode = None
    if event_bias:
        legacy_mode = SelectorMode.EVENT
    elif gradient_bias:
        legacy_mode = SelectorMode.GRADIENT

    if selector_mode is None:
        return legacy_mode or SelectorMode.RANDOM

    try:
        resolved = SelectorMode(selector_mode)
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in SelectorMode)
        raise ValueError(
            f"Unknown selector_mode={selector_mode!r}; expected one of: {choices}"
        ) from exc

    if legacy_mode is not None and resolved != legacy_mode:
        raise ValueError(
            f"selector_mode={resolved.value!r} conflicts with legacy "
            f"{legacy_mode.value}_bias=True"
        )

    return resolved


def validate_dino_configuration(cfg):
    dino = cfg.get("dino") or {}
    enabled = bool(dino.get("enabled", False))
    prior_enabled = bool(dino.get("prior", {}).get("enabled", False))
    depth_enabled = bool(dino.get("depth", {}).get("enabled", False))
    matching_enabled = bool(dino.get("matching", {}).get("enabled", False))
    selectors = {
        cfg.get("train_selector_mode"),
        cfg.get("eval_selector_mode"),
    }
    if "dino_peak" in selectors and not (enabled and prior_enabled):
        raise ValueError("dino_peak requires DINO and prior to be enabled")
    depth_mode = dino.get("depth", {}).get("mode", "constant")
    if depth_mode in {"mask", "scaled"} and not depth_enabled:
        raise ValueError(f"depth mode {depth_mode!r} requires depth.enabled=true")
    if matching_enabled and not enabled:
        raise ValueError("DINO matching residual requires dino.enabled=true")
    if enabled and not dino.get("freeze_pretrained", True):
        raise ValueError("The mainline requires dino.freeze_pretrained=true")



@dataclass(frozen=True)
class PatchifyOutput:
    fmap: Any
    gmap: Any
    imap: Any
    patches: Any
    frame_index: Any
    colors: Any
    coords: Any
    prior_logits: Optional[Any] = None
    inv_depth: Optional[Any] = None
    raw_patch_inv_depth: Optional[Any] = None
    context_dino: Optional[Any] = None
    matching_residual: Optional[Any] = None

    def as_legacy_tuple(self):
        """Return the pre-Phase-0 tuple for numerical regression checks."""
        return (
            self.fmap,
            self.gmap,
            self.imap,
            self.patches,
            self.frame_index,
            self.colors,
        )
