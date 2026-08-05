def load_ramp_state_dict(model, state_dict):
    """Strictly load RAMP keys while allowing a newly enabled DINO side branch."""
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing_prefix = "patchify.dino_branch."
    illegal_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefix)
    ]
    if illegal_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint is incompatible: "
            f"missing={illegal_missing}, unexpected={incompatible.unexpected_keys}"
        )
    if model.patchify.dino_branch is None and incompatible.missing_keys:
        raise RuntimeError(
            "DINO-disabled model must strict-load every key; missing="
            f"{incompatible.missing_keys}"
        )
    return incompatible
