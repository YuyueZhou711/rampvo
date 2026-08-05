from typing import Sequence, Tuple

import torch
import torch.nn as nn


class SharedDINOReassemble(nn.Module):
    """Frozen DINOv2 backbone and the pretrained shared DPT reassemble path."""

    INTERMEDIATE_LAYERS = (2, 5, 8, 11)

    def __init__(self, official_model):
        super().__init__()
        head = official_model.depth_head
        self.backbone = official_model.pretrained
        self.projects = head.projects
        self.resize_layers = head.resize_layers
        self.reassemble_layers = nn.ModuleList(
            [
                head.scratch.layer1_rn,
                head.scratch.layer2_rn,
                head.scratch.layer3_rn,
                head.scratch.layer4_rn,
            ]
        )

    def forward(self, images: torch.Tensor):
        patch_height = images.shape[-2] // 14
        patch_width = images.shape[-1] // 14
        features = self.backbone.get_intermediate_layers(
            images,
            self.INTERMEDIATE_LAYERS,
            return_class_token=True,
        )

        levels = []
        for index, feature in enumerate(features):
            tokens = feature[0]
            tokens = tokens.permute(0, 2, 1).reshape(
                tokens.shape[0], tokens.shape[-1], patch_height, patch_width
            )
            level = self.projects[index](tokens)
            level = self.resize_layers[index](level)
            level = self.reassemble_layers[index](level)
            levels.append(level)

        return tuple(levels), (patch_height, patch_width)
