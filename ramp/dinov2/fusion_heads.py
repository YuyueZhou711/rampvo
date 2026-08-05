from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ramp.third_party.depth_anything_v2.dpt import _make_fusion_block


def _refine(levels, refinenets):
    layer_1, layer_2, layer_3, layer_4 = levels
    path_4 = refinenets[3](layer_4, size=layer_3.shape[2:])
    path_3 = refinenets[2](path_4, layer_3, size=layer_2.shape[2:])
    path_2 = refinenets[1](path_3, layer_2, size=layer_1.shape[2:])
    return refinenets[0](path_2, layer_1)


class FrozenDepthPath(nn.Module):
    """The pretrained official DPT refinement and relative-depth output path."""

    def __init__(self, official_model):
        super().__init__()
        scratch = official_model.depth_head.scratch
        self.refinenets = nn.ModuleList(
            [
                scratch.refinenet1,
                scratch.refinenet2,
                scratch.refinenet3,
                scratch.refinenet4,
            ]
        )
        self.output_conv1 = scratch.output_conv1
        self.output_conv2 = scratch.output_conv2

    def forward_full(self, levels, token_grid):
        path_1 = _refine(levels, self.refinenets)
        depth = self.output_conv1(path_1)
        depth = F.interpolate(
            depth,
            (token_grid[0] * 14, token_grid[1] * 14),
            mode="bilinear",
            align_corners=True,
        )
        return F.relu(self.output_conv2(depth))


class TaskFusionHead(nn.Module):
    """Independent trainable DPT fusion path for context or prior logits."""

    def __init__(self, output_channels: int, features: int = 64):
        super().__init__()
        self.refinenets = nn.ModuleList(
            [_make_fusion_block(features, use_bn=False) for _ in range(4)]
        )
        hidden = max(32, min(features, output_channels))
        self.output = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(features, hidden, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, output_channels, 1),
        )

    def forward(self, levels, padded_quarter_size, output_size):
        fused = _refine(levels, self.refinenets)
        fused = F.interpolate(
            fused,
            padded_quarter_size,
            mode="bilinear",
            align_corners=True,
        )
        output = self.output(fused)
        return output[..., : output_size[0], : output_size[1]]
