# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for geometry conditioning of the action head."""

import importlib.util
from pathlib import Path

import pytest
import torch


# Loaded by path rather than as ``gr00t.model.modules.geometry_conditioning``, because importing
# through the package executes ``gr00t/model/__init__.py`` and pulls in the whole training-time
# config stack. The module under test needs only torch, and asserting that here keeps it that way:
# geometry conditioning must stay importable wherever the policy runs, not only in a dev image.
_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "gr00t" / "model" / "modules" / "geometry_conditioning.py"
)
_spec = importlib.util.spec_from_file_location("geometry_conditioning", _MODULE_PATH)
_geometry_conditioning = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_geometry_conditioning)

FrozenGeometryEncoder = _geometry_conditioning.FrozenGeometryEncoder
GeometryConditioning = _geometry_conditioning.GeometryConditioning
GeometryConditioningConfig = _geometry_conditioning.GeometryConditioningConfig

BACKBONE_DIM = 2048
GEOMETRY_DIM = 384
# N1.7 preserves the camera's 4:3 aspect ratio, so the real token grid is rectangular.
GRID = (8, 11)
TOKENS_PER_IMAGE = GRID[0] * GRID[1]


def test_encoder_emits_resampled_token_grid():
    """The encoder resamples its own square patch grid onto the requested, possibly rectangular, grid."""
    encoder = FrozenGeometryEncoder(GeometryConditioningConfig(mode="align"))
    images = torch.randint(0, 256, (2, 3, 480, 640), dtype=torch.uint8)

    features = encoder(images, grid=GRID)

    assert features.shape == (2, TOKENS_PER_IMAGE, encoder.feature_dim)
    assert encoder.feature_dim == GEOMETRY_DIM
    assert torch.isfinite(features).all()


def test_encoder_feature_dim_requires_a_forward_pass():
    """The width is measured, never tabulated, so reading it before a forward must raise."""
    encoder = FrozenGeometryEncoder(GeometryConditioningConfig(mode="align"))

    with pytest.raises(AssertionError, match="only known after a forward pass"):
        _ = encoder.feature_dim

    assert encoder.probe_feature_dim() == GEOMETRY_DIM
    assert encoder.feature_dim == GEOMETRY_DIM


def test_encoder_stays_frozen():
    """No encoder parameter should require gradients, so it cannot be trained by accident."""
    encoder = FrozenGeometryEncoder(GeometryConditioningConfig(mode="align"))
    encoder(torch.randint(0, 256, (1, 3, 240, 320), dtype=torch.uint8), grid=(4, 5))

    assert not any(p.requires_grad for p in encoder._model.parameters())


def _identity_align_module(dim: int) -> GeometryConditioning:
    """An ``align`` module whose projector and target embedding are neutralised.

    The loss reduction is what these tests are about, so the projector is replaced by an identity
    and the positional embedding is zeroed. That isolates ``1 - cos`` and the masking from the
    learned parts, which have their own tests below.
    """
    module = GeometryConditioning(GeometryConditioningConfig(mode="align"), dim, dim)
    module.align_student_norm = torch.nn.Identity()
    module.align_projector = torch.nn.Identity()
    with torch.no_grad():
        module.align_position_embedding.zero_()
    return module


APACHE_LICENSED_GEOMETRY_ENCODERS = _geometry_conditioning.APACHE_LICENSED_GEOMETRY_ENCODERS
DA3_CHECKPOINT = Path("/models/isaaclab_arena/DA3METRIC-LARGE")
DA3_ANYVIEW_CHECKPOINT = Path("/models/isaaclab_arena/DA3-BASE")


def test_encoder_refuses_an_unlisted_checkpoint():
    """A non-Apache checkpoint must be refused by name, not by trusting a licence string.

    Hugging Face reports ``apache-2.0`` for ``DA3-LARGE-1.1`` in both its API and its model card,
    while upstream's own table says CC BY-NC 4.0. A licence check at download time would therefore
    pass a checkpoint that cannot be shipped, which is why the gate is an explicit allowlist.
    """
    encoder = FrozenGeometryEncoder(
        GeometryConditioningConfig(mode="align", encoder_id="depth-anything/DA3-LARGE-1.1")
    )

    with pytest.raises(AssertionError, match="not on the Apache allowlist"):
        encoder(torch.zeros(1, 3, 64, 64, dtype=torch.uint8), grid=(2, 2))

    assert "DA3-LARGE-1.1" not in APACHE_LICENSED_GEOMETRY_ENCODERS
    assert "DA3METRIC-LARGE" in APACHE_LICENSED_GEOMETRY_ENCODERS


def test_encoder_allowlist_can_be_waived_for_research():
    """The waiver exists, so an internal-only run is not blocked -- but it must be explicit."""
    config = GeometryConditioningConfig(
        mode="align", encoder_id="depth-anything/DA3-LARGE-1.1", allow_unlisted_encoder=True
    )

    FrozenGeometryEncoder(config)._check_licence()  # must not raise


@pytest.mark.skipif(
    not (DA3_CHECKPOINT / "model.safetensors").is_file(),
    reason=f"{DA3_CHECKPOINT} not fetched; see the plan's section 8.3",
)
def test_da3_teacher_emits_the_student_grid():
    """DA3METRIC-LARGE loads strictly and resamples onto the rectangular student grid.

    Also pins the two things that made the first draft's recipe wrong: the width is 1024 (not the
    doubled figure VGGT's aggregation would suggest), and the checkpoint loads without
    ``convert_metric_state_dict``, which on this safetensors release doubles the ``model.`` prefix
    and matches nothing.
    """
    pytest.importorskip("depth_anything_3")
    encoder = FrozenGeometryEncoder(
        GeometryConditioningConfig(mode="align", encoder_id=str(DA3_CHECKPOINT))
    )
    images = torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8)

    features = encoder(images, grid=GRID)

    assert encoder.feature_dim == 1024
    assert features.shape == (1, TOKENS_PER_IMAGE, 1024)
    assert torch.isfinite(features).all()
    assert not any(p.requires_grad for p in encoder._model.parameters())


@pytest.mark.skipif(
    not (DA3_ANYVIEW_CHECKPOINT / "model.safetensors").is_file(),
    reason=f"{DA3_ANYVIEW_CHECKPOINT} not fetched; see the plan's section 8.3",
)
def test_da3_anyview_teacher_loads_despite_an_incomplete_head():
    """DA3-BASE loads even though its release omits head tensors its own spec declares.

    Two regressions in one. The checkpoint is missing six
    ``head.scratch.output_conv2_aux.*`` tensors, so a whole-net strict load rejected the any-view
    teacher outright even though its backbone is complete and the head is discarded. And its width
    is 1536, not the 768 a ViT-B suggests, because ``cat_token`` is true -- the same doubling that
    made a hardcoded width table a trap.
    """
    pytest.importorskip("depth_anything_3")
    encoder = FrozenGeometryEncoder(
        GeometryConditioningConfig(mode="align", encoder_id=str(DA3_ANYVIEW_CHECKPOINT))
    )
    images = torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8)

    features = encoder(images, grid=GRID)

    assert encoder.feature_dim == 1536
    assert features.shape == (1, TOKENS_PER_IMAGE, 1536)
    assert torch.isfinite(features).all()


def test_alignment_loss_is_bounded_and_differentiable():
    """The alignment loss is a scalar in [0, 2] and reaches both the projector and the embedding."""
    module = GeometryConditioning(
        GeometryConditioningConfig(mode="align"), BACKBONE_DIM, GEOMETRY_DIM
    )
    tokens = torch.randn(2, TOKENS_PER_IMAGE, BACKBONE_DIM, requires_grad=True)
    geometry = torch.randn(2, TOKENS_PER_IMAGE, GEOMETRY_DIM)

    loss = module.alignment_loss(tokens, geometry)
    loss.backward()

    assert loss.ndim == 0
    assert 0.0 <= loss.item() <= 2.0
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in module.align_projector.parameters()
    )
    assert module.align_position_embedding.grad is not None, (
        "The target positional embedding must be trained; upstream's ablation loses ten points on"
        " LIBERO-Long without it, so a detached embedding would silently reproduce that."
    )


def test_alignment_loss_is_one_minus_cosine():
    """Identical vectors give 0 and anti-aligned vectors give 2, not -1 and +1.

    Upstream logs ``1 - cos``. Reporting ``-cos`` has the same gradient but is not comparable to
    those logs, which is the whole reason to pin the convention in a test.
    """
    module = _identity_align_module(GEOMETRY_DIM)
    geometry = torch.randn(2, TOKENS_PER_IMAGE, GEOMETRY_DIM)

    assert module.alignment_loss(geometry, geometry).item() == pytest.approx(0.0, abs=1e-5)
    assert module.alignment_loss(-geometry, geometry).item() == pytest.approx(2.0, abs=1e-5)


def test_alignment_mask_excludes_masked_tokens():
    """A masked loss equals the loss over just the valid tokens.

    Arena's G1 path is fixed-resolution today, so no padding arises -- but ``letter_box_transform``
    would introduce it, and an unmasked pad region trains the projector against blank pixels.
    """
    module = _identity_align_module(GEOMETRY_DIM)
    tokens = torch.randn(2, 6, GEOMETRY_DIM)
    geometry = torch.randn(2, 6, GEOMETRY_DIM)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, :3] = True

    masked = module.alignment_loss(tokens, geometry, align_mask=mask)
    subset = module.alignment_loss(tokens[:, :3], geometry[:, :3])

    torch.testing.assert_close(masked, subset)
    # And the masked positions genuinely do not matter.
    poisoned = geometry.clone()
    poisoned[:, 3:] = 1e3
    torch.testing.assert_close(module.alignment_loss(tokens, poisoned, align_mask=mask), masked)


def test_alignment_rejects_a_fully_masked_batch():
    """Masking every token means the grids disagree, which must raise rather than train on nothing."""
    module = _identity_align_module(GEOMETRY_DIM)

    with pytest.raises(AssertionError, match="nothing to"):
        module.alignment_loss(
            torch.randn(2, 4, GEOMETRY_DIM),
            torch.randn(2, 4, GEOMETRY_DIM),
            align_mask=torch.zeros(2, 4, dtype=torch.bool),
        )


def test_positional_embedding_is_wired_into_the_target():
    """The target embedding must reach the loss, and only through the target.

    Asserted at a magnitude comparable to the target's own scale rather than at the initialisation
    scale: the embedding is learnable, so what matters is that it participates, not how large it
    starts. See ``test_positional_embedding_is_negligible_at_the_default_scale`` for why the
    starting scale is nonetheless worth pinning down.
    """
    torch.manual_seed(0)
    module = GeometryConditioning(
        GeometryConditioningConfig(mode="align"), BACKBONE_DIM, GEOMETRY_DIM
    )
    tokens = torch.randn(2, TOKENS_PER_IMAGE, BACKBONE_DIM)
    geometry = torch.randn(2, TOKENS_PER_IMAGE, GEOMETRY_DIM)

    with torch.no_grad():
        module.align_position_embedding.normal_(std=1.0)
        with_embedding = module.alignment_loss(tokens, geometry).item()
        module.align_position_embedding.zero_()
        without_embedding = module.alignment_loss(tokens, geometry).item()

    assert abs(with_embedding - without_embedding) > 1e-3, (
        f"The positional embedding changed the loss by only {abs(with_embedding - without_embedding):.2e},"
        " so it is not reaching the target."
    )


def test_positional_embedding_is_negligible_at_the_default_scale():
    """Record that the default init is ~1e-5 against unit-scale targets, so it starts as a no-op.

    This is a characterisation test, not an endorsement. Spatial Forcing credits the target
    embedding with ten points on LIBERO-Long, but publishes no initialisation scale, and its target
    is VGGT's aggregated tokens whose magnitude we do not know. Against unit-variance features a
    std of 0.02 moves the cosine by ~2e-5, which the optimiser has to undo before the term can
    matter. The scale must be set against the real teacher's measured feature scale; until then this
    test documents the starting point rather than asserting it is right.
    """
    torch.manual_seed(0)
    config = GeometryConditioningConfig(mode="align")
    module = GeometryConditioning(config, BACKBONE_DIM, GEOMETRY_DIM)
    tokens = torch.randn(2, TOKENS_PER_IMAGE, BACKBONE_DIM)
    geometry = torch.randn(2, TOKENS_PER_IMAGE, GEOMETRY_DIM)

    with torch.no_grad():
        default = module.alignment_loss(tokens, geometry).item()
        module.align_position_embedding.zero_()
        zeroed = module.alignment_loss(tokens, geometry).item()

    assert config.align_position_embedding_std == 0.02
    assert abs(default - zeroed) < 1e-3, (
        "If this now differs materially, the default scale changed."
    )


def test_projector_widths_follow_the_measured_target():
    """Hidden and output widths both equal the target width, as in the reference implementation.

    Upstream writes ``2 * vggt_dim`` because VGGT's aggregated tokens are twice its configured
    width. That doubling is a VGGT quirk, so a projector sized from a tabulated width would be
    wrong for any other teacher -- DA3METRIC-LARGE included, whose target is a flat 1024.
    """
    target_dim = 1024
    module = GeometryConditioning(
        GeometryConditioningConfig(mode="align"), BACKBONE_DIM, target_dim
    )
    linears = [layer for layer in module.align_projector if isinstance(layer, torch.nn.Linear)]

    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (BACKBONE_DIM, target_dim),
        (target_dim, target_dim),
    ]
    assert isinstance(module.align_student_norm, torch.nn.LayerNorm)
    assert not any(isinstance(layer, torch.nn.BatchNorm1d) for layer in module.align_projector), (
        "Upstream normalises the student with LayerNorm; BatchNorm over a token axis is not what"
        " the reference implementation does."
    )
    assert module.align_position_embedding.shape[-1] == target_dim


def test_alignment_rejects_a_flattened_batch():
    """A pre-flattened batch loses the token axis the embedding and per-sample mean need."""
    module = GeometryConditioning(
        GeometryConditioningConfig(mode="align"), BACKBONE_DIM, GEOMETRY_DIM
    )

    with pytest.raises(AssertionError, match=r"Expected \(B, N, D\)"):
        module.alignment_loss(torch.randn(88, BACKBONE_DIM), torch.randn(88, GEOMETRY_DIM))


def test_alignment_rejects_token_count_mismatch():
    """A grid disagreement must raise rather than silently align mismatched positions."""
    module = GeometryConditioning(
        GeometryConditioningConfig(mode="align"), BACKBONE_DIM, GEOMETRY_DIM
    )

    with pytest.raises(AssertionError, match="Alignment is positional"):
        module.alignment_loss(torch.randn(2, 64, BACKBONE_DIM), torch.randn(2, 32, GEOMETRY_DIM))


def test_fuse_extends_sequence_and_masks_consistently():
    """Fused tokens are appended, and both masks grow to match."""
    module = GeometryConditioning(
        GeometryConditioningConfig(mode="mix"), BACKBONE_DIM, GEOMETRY_DIM
    )
    batch, length = 2, 40
    features = torch.randn(batch, length, BACKBONE_DIM)
    image_mask = torch.zeros(batch, length, dtype=torch.bool)
    image_mask[:, : TOKENS_PER_IMAGE // 2] = True
    attention_mask = torch.ones(batch, length, dtype=torch.bool)
    geometry = torch.randn(batch, TOKENS_PER_IMAGE, GEOMETRY_DIM)

    extended, extended_image_mask, extended_attention_mask = module.fuse(
        features, image_mask, attention_mask, geometry
    )

    assert extended.shape == (batch, length + TOKENS_PER_IMAGE, BACKBONE_DIM)
    assert extended_image_mask.shape == extended.shape[:2]
    assert extended_attention_mask.shape == extended.shape[:2]
    # The original conditioning is left untouched; only the appended block is new.
    torch.testing.assert_close(extended[:, :length], features)
    assert extended_image_mask[:, length:].all(), "geometry tokens default to the image branch"
    assert extended_attention_mask[:, length:].all()


def test_fuse_is_sensitive_to_geometry_content():
    """Zeroed geometry must change the fused output, or the gain is extra capacity, not geometry.

    This is the control 3D-Mix uses to separate real geometric information from the effect of simply
    conditioning on more tokens.
    """
    torch.manual_seed(0)
    module = GeometryConditioning(
        GeometryConditioningConfig(mode="mix"), BACKBONE_DIM, GEOMETRY_DIM
    ).eval()
    features = torch.randn(1, 24, BACKBONE_DIM)
    image_mask = torch.ones(1, 24, dtype=torch.bool)
    attention_mask = torch.ones(1, 24, dtype=torch.bool)

    with torch.no_grad():
        informative, _, _ = module.fuse(
            features, image_mask, attention_mask, torch.randn(1, TOKENS_PER_IMAGE, GEOMETRY_DIM)
        )
        zeroed, _, _ = module.fuse(
            features, image_mask, attention_mask, torch.zeros(1, TOKENS_PER_IMAGE, GEOMETRY_DIM)
        )

    difference = (informative[:, 24:] - zeroed[:, 24:]).abs().mean().item()
    assert difference > 1e-3, (
        f"Fused tokens barely responded to geometry content (delta {difference:.2e})"
    )


def test_off_mode_builds_no_parameters():
    """``off`` must add nothing, so the flag is genuinely free when disabled."""
    module = GeometryConditioning(
        GeometryConditioningConfig(mode="off"), BACKBONE_DIM, GEOMETRY_DIM
    )

    assert sum(p.numel() for p in module.parameters()) == 0


align_backbone_layer_from_site = _geometry_conditioning.align_backbone_layer_from_site


def test_align_site_names_resolve_to_layers():
    """The named sites resolve, and a shallow-layer sweep parses to its index."""
    assert align_backbone_layer_from_site("post_vl_self_attention") is None
    assert align_backbone_layer_from_site("backbone_output") is None
    assert [align_backbone_layer_from_site(f"backbone_layer_{k}") for k in (6, 9, 12)] == [6, 9, 12]


def test_align_site_rejects_unrecognised_names():
    """A typo must raise rather than silently fall back to a different depth.

    The two sites are four transformer layers apart, so a silent fallback would change what the run
    measures without changing anything visible in its logs.
    """
    for site in ("post_vlln", "backbone_layer_", "backbone_layer_x", "layer_9", ""):
        with pytest.raises(AssertionError, match="is not recognised"):
            align_backbone_layer_from_site(site)


token_grid_from_image_grid_thw = _geometry_conditioning.token_grid_from_image_grid_thw


def test_token_grid_is_read_from_the_processor_and_is_not_square():
    """A 4:3 frame gives an 8x11 grid of 88 tokens, which is what N1.7 actually produces."""
    # thw is in unmerged patch units; merge 2 halves each spatial dimension.
    assert token_grid_from_image_grid_thw([[1, 16, 22]], spatial_merge_size=2) == (8, 11)
    assert 8 * 11 == 88


def test_token_grid_handles_several_images_of_one_size():
    """The parallax pair and multi-view configs share one grid."""
    assert token_grid_from_image_grid_thw([[1, 16, 22], [1, 16, 22]], spatial_merge_size=2) == (
        8,
        11,
    )


def test_token_grid_rejects_mixed_resolutions():
    """Geometry features resample to a single grid, so a mixed batch must raise."""
    with pytest.raises(AssertionError, match="different patch grids"):
        token_grid_from_image_grid_thw([[1, 16, 22], [1, 16, 16]], spatial_merge_size=2)
