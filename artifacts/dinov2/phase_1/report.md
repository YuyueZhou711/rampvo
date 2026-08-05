# DINOv2 Phase 1 report

Phase 1 passed. The official Depth Anything V2 Small source and Apache-2.0
license are vendored at the fixed upstream commit. The local checkpoint is
SHA256-verified and strict-loads all 239 tensors.

The branch restores the RAMP image range, replicate-pads `480x640` to
`490x644`, applies ImageNet normalization, and extracts a `35x46` DINO token
grid. For N=15 it returns context `[1,15,384,120,160]`, prior logits
`[1,15,1,120,160]`, and non-negative relative inverse depth
`[1,15,1,120,160]`.

The DINO backbone, shared reassemble and official depth path remain frozen and
in eval mode after a top-level `train()`. The independent context/prior fusion
heads receive finite gradients and update. The wrapper's float32 official
depth path is exactly equal to the vendored official forward path.

Micro-batch 1 and 15 are equivalent at `rtol=atol=5e-3`. Peak allocated memory
was 652.59 MiB for micro-batch 1 and 1838.33 MiB for the full N=15 batch, so
the default remains 1.

The branch is a side output only in this phase. Fixed-input RAMP outputs remain
bit-identical. On AquaticVision Easy/01 frames 0-127 it produced 128/128 poses,
zero hard failures, Sim(3) ATE RMSE 0.01246 m, and 100% valid trajectory. This
is 1.028x the Phase 0 run and remains within the 1.05 no-regression gate.
