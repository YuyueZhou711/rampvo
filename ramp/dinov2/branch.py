from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ramp.third_party.depth_anything_v2 import DepthAnythingV2

from .backbone import DINOInputMetadata, preprocess_dino_images
from .fusion_heads import FrozenDepthPath, TaskFusionHead
from .reassemble import SharedDINOReassemble


SMALL_CHECKPOINT_SHA256 = (
    "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"
)


@dataclass(frozen=True)
class DINOBranchOutput:
    context_dino: torch.Tensor
    prior_logits: torch.Tensor
    inv_depth: torch.Tensor
    matching_residual: Optional[torch.Tensor]
    metadata: DINOInputMetadata


def checkpoint_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path, expected_sha256=SMALL_CHECKPOINT_SHA256):
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DINO checkpoint not found: {checkpoint}")
    actual = checkpoint_sha256(checkpoint)
    if actual != expected_sha256:
        raise ValueError(
            f"DINO checkpoint SHA256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return actual


class FrozenDINOBranch(nn.Module):
    """Frozen Depth Anything V2-S shared/depth paths plus trainable task heads."""

    def __init__(self, config):
        super().__init__()
        if not config.get("freeze_pretrained", True):
            raise ValueError("Phase 1 requires dino.freeze_pretrained=true")
        if config.get("encoder", "vits") != "vits":
            raise ValueError("Phase 1 is fixed to encoder='vits'")
        if config.get("matching", {}).get("enabled", False):
            raise NotImplementedError(
                "DINO matching residual is intentionally deferred until Phase 7"
            )

        self.input_range = config.get("input_range", "ramp_default")
        self.micro_batch = int(config.get("micro_batch", 1))
        if self.micro_batch <= 0:
            raise ValueError("dino.micro_batch must be positive")
        self.amp_dtype = config.get("amp_dtype", "float16")
        if self.amp_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dino.amp_dtype must be float16, bfloat16, or float32")

        checkpoint = config.get("checkpoint")
        expected_hash = config.get("checkpoint_sha256", SMALL_CHECKPOINT_SHA256)
        self.checkpoint_sha256 = verify_checkpoint(checkpoint, expected_hash)

        official = DepthAnythingV2(
            encoder="vits",
            features=64,
            out_channels=[48, 96, 192, 384],
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        official.load_state_dict(state, strict=True)
        self.shared = SharedDINOReassemble(official)
        self.depth_path = FrozenDepthPath(official)
        self.context_head = TaskFusionHead(384, features=64)
        self.prior_head = TaskFusionHead(1, features=64)
        self.matching_head = None

        self._freeze_pretrained()

    @property
    def frozen_modules(self):
        return (self.shared, self.depth_path)

    def _freeze_pretrained(self):
        for module in self.frozen_modules:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_pretrained()
        return self

    def _autocast(self, device):
        if device.type != "cuda" or self.amp_dtype == "float32":
            return nullcontext()
        dtype = torch.float16 if self.amp_dtype == "float16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def forward(self, images):
        if images.shape[0] != 1:
            raise NotImplementedError("DINO branch currently supports B=1 only")
        normalized, metadata = preprocess_dino_images(
            images,
            input_range=self.input_range,
        )
        padded_quarter = (
            (metadata.padded_size[0] + 3) // 4,
            (metadata.padded_size[1] + 3) // 4,
        )

        contexts, priors, depths = [], [], []
        for start in range(0, normalized.shape[0], self.micro_batch):
            chunk = normalized[start : start + self.micro_batch]
            with self._autocast(chunk.device):
                with torch.no_grad():
                    levels, token_grid = self.shared(chunk)
                    depth_full = self.depth_path.forward_full(levels, token_grid)
                context = self.context_head(
                    levels, padded_quarter, metadata.quarter_size
                )
                prior = self.prior_head(
                    levels, padded_quarter, metadata.quarter_size
                )
                depth = depth_full[..., : metadata.original_size[0], : metadata.original_size[1]]
                depth = F.interpolate(
                    depth,
                    metadata.quarter_size,
                    mode="bilinear",
                    align_corners=True,
                )
            contexts.append(context)
            priors.append(prior)
            depths.append(depth)

        batch, frames = images.shape[:2]
        context = torch.cat(contexts).reshape(
            batch, frames, 384, *metadata.quarter_size
        )
        prior = torch.cat(priors).reshape(
            batch, frames, 1, *metadata.quarter_size
        )
        depth = torch.cat(depths).reshape(
            batch, frames, 1, *metadata.quarter_size
        )
        return DINOBranchOutput(
            context_dino=context,
            prior_logits=prior,
            inv_depth=depth,
            matching_residual=None,
            metadata=metadata,
        )

    def official_depth_from_preprocessed(self, normalized):
        """Expose the unchanged official path for numerical alignment tests."""
        with torch.no_grad(), self._autocast(normalized.device):
            levels, token_grid = self.shared(normalized)
            return self.depth_path.forward_full(levels, token_grid).squeeze(1)
