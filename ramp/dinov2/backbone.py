from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DINOInputMetadata:
    original_size: Tuple[int, int]
    padded_size: Tuple[int, int]
    padding: Tuple[int, int, int, int]
    token_grid: Tuple[int, int]

    @property
    def quarter_size(self) -> Tuple[int, int]:
        return self.original_size[0] // 4, self.original_size[1] // 4


def _restore_zero_one(images: torch.Tensor, input_range: str) -> torch.Tensor:
    if input_range == "ramp_default":
        images = (images + 0.5) / 2.0
    elif input_range == "minus_one_one":
        images = (images + 1.0) / 2.0
    elif input_range == "zero_one":
        pass
    else:
        raise ValueError(
            "dino.input_range must be one of: ramp_default, minus_one_one, zero_one"
        )
    return images.clamp(0.0, 1.0)


def preprocess_dino_images(
    images: torch.Tensor,
    *,
    input_range: str = "ramp_default",
    patch_size: int = 14,
):
    """Restore RAMP images, replicate-pad to patch size, and normalize."""
    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError(
            "images must have shape [B,N,3,H,W], "
            f"got {tuple(images.shape)}"
        )
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")

    batch, frames, channels, height, width = images.shape
    flat = _restore_zero_one(images.float(), input_range).reshape(
        batch * frames, channels, height, width
    )
    pad_height = (-height) % patch_size
    pad_width = (-width) % patch_size
    flat = F.pad(flat, (0, pad_width, 0, pad_height), mode="replicate")

    mean = flat.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = flat.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    normalized = (flat - mean) / std
    padded_height, padded_width = normalized.shape[-2:]
    metadata = DINOInputMetadata(
        original_size=(height, width),
        padded_size=(padded_height, padded_width),
        padding=(0, pad_width, 0, pad_height),
        token_grid=(padded_height // patch_size, padded_width // patch_size),
    )
    return normalized, metadata
