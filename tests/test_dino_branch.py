import json
from pathlib import Path

import pytest
import torch

from ramp.dinov2.backbone import preprocess_dino_images
from ramp.dinov2.branch import FrozenDINOBranch
from ramp.third_party.depth_anything_v2 import DepthAnythingV2


ROOT = Path(__file__).resolve().parents[1]


def dino_config(**overrides):
    config = json.loads(
        (ROOT / "config_net/MultiScale_TartanEvent_DINOv2.json").read_text()
    )["data_loader"]["train"]["args"]["dino"]
    config.update(overrides)
    config["checkpoint"] = str(ROOT / config["checkpoint"])
    return config


@pytest.fixture(scope="module")
def branch():
    if not torch.cuda.is_available():
        pytest.skip("Phase 1 branch tests require CUDA")
    model = FrozenDINOBranch(dino_config()).cuda().eval()
    yield model
    del model
    torch.cuda.empty_cache()


def test_branch_output_shapes_and_depth_validity(branch):
    images = torch.zeros(1, 2, 3, 480, 640, device="cuda")
    with torch.no_grad():
        output = branch(images)

    assert output.context_dino.shape == (1, 2, 384, 120, 160)
    assert output.prior_logits.shape == (1, 2, 1, 120, 160)
    assert output.inv_depth.shape == (1, 2, 1, 120, 160)
    assert output.matching_residual is None
    assert torch.isfinite(output.inv_depth).all()
    assert (output.inv_depth >= 0).all()


def test_micro_batch_matches_single_batch_with_amp_tolerance(branch):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    images = torch.rand(1, 3, 3, 224, 280, generator=generator, device="cuda")
    images = images * 2.0 - 0.5

    branch.micro_batch = 1
    with torch.no_grad():
        split = branch(images)
    branch.micro_batch = 3
    with torch.no_grad():
        together = branch(images)
    branch.micro_batch = 1

    for split_tensor, together_tensor in (
        (split.context_dino, together.context_dino),
        (split.prior_logits, together.prior_logits),
        (split.inv_depth, together.inv_depth),
    ):
        torch.testing.assert_close(
            split_tensor, together_tensor, rtol=5e-3, atol=5e-3
        )


def test_train_keeps_pretrained_eval_and_updates_only_new_heads(branch):
    branch.train()
    assert all(not module.training for module in branch.frozen_modules)
    assert all(
        not parameter.requires_grad
        for module in branch.frozen_modules
        for parameter in module.parameters()
    )
    assert branch.context_head.training and branch.prior_head.training

    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in branch.named_parameters()
        if name.startswith(("shared.", "depth_path."))
    }
    trainable_before = branch.context_head.output[-1].weight.detach().clone()
    optimizer = torch.optim.AdamW(
        list(branch.context_head.parameters()) + list(branch.prior_head.parameters()),
        lr=1e-4,
    )
    images = torch.zeros(1, 1, 3, 112, 140, device="cuda")
    optimizer.zero_grad(set_to_none=True)
    output = branch(images)
    loss = output.context_dino.float().square().mean() + output.prior_logits.float().square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    head_gradients = [
        parameter.grad
        for name, parameter in branch.named_parameters()
        if name.startswith(("context_head.", "prior_head."))
        and parameter.grad is not None
    ]
    assert head_gradients
    assert all(torch.isfinite(gradient).all() for gradient in head_gradients)
    assert all(
        parameter.grad is None
        for name, parameter in branch.named_parameters()
        if name.startswith(("shared.", "depth_path."))
    )
    optimizer.step()

    assert not torch.equal(
        trainable_before, branch.context_head.output[-1].weight.detach()
    )
    for name, before in frozen_before.items():
        torch.testing.assert_close(before, dict(branch.named_parameters())[name])
    branch.eval()


def test_wrapper_depth_is_official_depth_path(branch):
    reference = DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384]
    ).cuda().eval()
    state = torch.load(
        ROOT / "checkpoints/depth_anything_v2_vits.pth",
        map_location="cpu",
        weights_only=True,
    )
    reference.load_state_dict(state, strict=True)
    images = torch.zeros(1, 1, 3, 112, 140, device="cuda")
    normalized, _ = preprocess_dino_images(images, input_range="ramp_default")

    original_dtype = branch.amp_dtype
    branch.amp_dtype = "float32"
    with torch.no_grad():
        wrapped = branch.official_depth_from_preprocessed(normalized)
        official = reference(normalized)
    branch.amp_dtype = original_dtype

    torch.testing.assert_close(wrapped, official, rtol=0, atol=0)
    del reference
