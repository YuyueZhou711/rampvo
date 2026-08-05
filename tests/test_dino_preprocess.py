import torch

from ramp.dinov2.backbone import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    preprocess_dino_images,
)


def test_ramp_default_restore_pad_and_normalize_contract():
    ramp_image = torch.full((1, 2, 3, 480, 640), 0.5)

    normalized, metadata = preprocess_dino_images(
        ramp_image, input_range="ramp_default"
    )

    assert normalized.shape == (2, 3, 490, 644)
    assert metadata.original_size == (480, 640)
    assert metadata.padded_size == (490, 644)
    assert metadata.padding == (0, 4, 0, 10)
    assert metadata.token_grid == (35, 46)
    assert metadata.quarter_size == (120, 160)
    expected = torch.tensor(
        [(0.5 - mean) / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)]
    )
    torch.testing.assert_close(normalized[0, :, 0, 0], expected)
    torch.testing.assert_close(normalized[0, :, -1, -1], expected)


def test_minus_one_one_restore_is_clamped():
    images = torch.tensor([-3.0, 0.0, 3.0]).view(1, 1, 3, 1, 1)
    normalized, metadata = preprocess_dino_images(
        images, input_range="minus_one_one"
    )

    recovered = normalized[:, :, :1, :1] * torch.tensor(IMAGENET_STD).view(
        1, 3, 1, 1
    ) + torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    torch.testing.assert_close(recovered.flatten(), torch.tensor([0.0, 0.5, 1.0]))
    assert metadata.padded_size == (14, 14)
