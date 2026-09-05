# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Condition the action head on geometry features from a frozen depth model.

The VLM backbone is pretrained on 2D data, so its visual embeddings carry accurate bearing to an
object and poor metric range. Two published remedies consume the same frozen geometry features in
different places, and this module implements both behind one flag so they can be compared on the
same task rather than adjudicated from their papers:

- ``align`` (Spatial Forcing, arXiv:2510.12276) adds a cosine alignment loss between the backbone's
  image tokens and the geometry features. The geometry encoder is dropped at inference, so the
  deployed policy pays nothing.
- ``mix`` (3D-Mix, arXiv:2603.24393) projects and gates the geometry features against the
  backbone's pooled semantic context, then concatenates them onto the conditioning sequence. The
  encoder runs at inference.

The ``align`` path follows Spatial Forcing's **reference implementation**
(``openpi-SF/src/openpi/models_pytorch/projectors.py``) rather than the paper prose, which disagree
with each other on the projector: the code uses an optional LayerNorm on the student side and no
BatchNorm. Three details are load-bearing and easy to get wrong, so they are named here:

1. The positional embedding is added to the **target**, not to the student tokens. Removing it costs
   ten points on LIBERO-Long in the paper's own ablation.
2. The projector's output width is the **measured** width of the teacher's features. Upstream writes
   ``2 * vggt_dim`` only because VGGT's aggregated tokens are twice its configured width; that
   doubling is a VGGT quirk, not a rule, so the width is probed rather than tabulated.
3. The loss is masked to valid image tokens and averaged per sample before the batch mean.

The encoder is Depth-Anything-V2-Small by default: Apache-2.0, 24.8M parameters, and already used
elsewhere in this stack. It is deliberately pluggable, because the published results use VGGT, whose
released checkpoint is not licensed for commercial use.
"""

from dataclasses import dataclass, field
import json
import os
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


GeometryMode = Literal["off", "align", "mix"]

DEFAULT_GEOMETRY_ENCODER = "depth-anything/Depth-Anything-V2-Small-hf"

GeometryEncoderKind = Literal["auto", "depth_anything_v2", "da3"]

APACHE_LICENSED_GEOMETRY_ENCODERS = frozenset(
    {
        "DA3METRIC-LARGE",
        "DA3MONO-LARGE",
        "DA3-BASE",
        "DA3-SMALL",
        "Depth-Anything-V2-Small-hf",
        "map-anything-apache",
    }
)
"""Encoder basenames cleared for use beyond internal research, as an explicit allowlist.

Hugging Face's own metadata cannot be trusted for this: ``DA3-LARGE-1.1`` reports
``license: apache-2.0`` in both the API response and its model card, while the upstream GitHub
README's licence table -- the authoritative one, maintainer-confirmed -- lists it as CC BY-NC 4.0,
along with ``DA3-LARGE``, the GIANT series and the NESTED series. The
``Depth-Anything-V2-Metric-*`` variants carry no licence tag at all. So the gate is a name list, not
a licence string read at download time, and a typo cannot select a non-commercial checkpoint.
"""


@dataclass
class GeometryConditioningConfig:
    """Configuration for geometry conditioning."""

    mode: GeometryMode = "off"
    """Which consumption mode to use. ``off`` disables the module entirely."""

    encoder_id: str = DEFAULT_GEOMETRY_ENCODER
    """Hugging Face id, or local directory, of the frozen geometry encoder."""

    encoder_kind: GeometryEncoderKind = "auto"
    """Which encoder family ``encoder_id`` names. ``auto`` reads it from the checkpoint's config."""

    encoder_input_size: int = 518
    """Shorter-side resolution the encoder is run at. Depth-Anything-V2's processor uses 518."""

    allow_unlisted_encoder: bool = False
    """Skip the Apache-only allowlist. For internal research runs that are never shipped."""

    da3_out_layer: int = -1
    """Which of DA3's own ``out_layers`` to align against, as an index into that list.

    ``-1`` selects the last (layer 23 for ``DA3METRIC-LARGE``), which is what the checkpoint's own
    DPT head consumes and therefore matches Spatial Forcing's "backbone latent before task heads".
    """

    align_loss_coeff: float = 0.5
    """Weight on the alignment loss in ``align`` mode.

    **Unknown upstream.** Spatial Forcing never states a number: its Appendix A is titled "Weight
    Factor" but the value is not recoverable from the published HTML, and it is absent from the
    reference implementation's committed configs. This default is a guess kept only so a run starts,
    and it must be swept rather than reported as the paper's value.
    """

    align_max_tokens: int = 4096
    """Rows in the target positional-embedding table, which is sliced to the batch's token count.

    Sized generously because it is a table of ``align_max_tokens x target_dim`` and the real grid is
    88 tokens per image; a fixed table keeps the parameter count independent of the batch.
    """

    align_position_embedding_std: float = 0.02
    """Initialisation scale of the target positional embedding.

    **Provisional.** Spatial Forcing credits this embedding with ten points on LIBERO-Long but
    publishes no initialisation scale, and its target is VGGT's aggregated tokens whose magnitude is
    unknown here. Against unit-variance features a std of 0.02 shifts the cosine by only ~2e-5, so
    the term starts as a near no-op and the optimiser has to grow it before it can matter. Set this
    against the teacher's measured feature scale once that measurement exists.
    """

    align_use_student_norm: bool = True
    """Apply LayerNorm to the student tokens before projecting, as the reference implementation does."""

    freeze_encoder: bool = True
    """Keep the geometry encoder frozen. 3D-Mix reports frozen is as good as or better than tuned."""

    encoder_dtype: str = "float32"
    """Dtype the encoder runs in. Kept separate from the policy dtype to bound its memory."""

    trainable_keys: list[str] = field(
        default_factory=lambda: ["projector", "gate", "semantic", "geometry"]
    )
    """Parameter-name fragments belonging to this module, so a caller can size or filter them."""

    mix_tokens_as: Literal["image", "text"] = "image"
    """Which cross-attention branch the fused tokens join in ``mix`` mode.

    N1.7's action head splits cross-attention into image and non-image streams on ``image_mask``,
    a split the 3D-Mix paper's backbones do not have. Geometry tokens carry spatial content, so they
    default to the image stream; the alternative is exposed because it is an untested design choice
    rather than a settled one.
    """


ALIGN_SITE_POST_VL_SELF_ATTENTION = "post_vl_self_attention"
"""Deepest reachable site: above the truncated backbone, after ``vlln`` and ``vl_self_attention``."""

ALIGN_SITE_BACKBONE_OUTPUT = "backbone_output"
"""The truncation point itself, i.e. the last surviving language-model layer."""

ALIGN_SITE_BACKBONE_LAYER_PREFIX = "backbone_layer_"
"""Prefix for a specific language-model layer, as in ``backbone_layer_9``."""


def align_backbone_layer_from_site(site: str) -> int | None:
    """Return the backbone layer index an alignment site names, or None if it names another site.

    Args:
        site: One of ``post_vl_self_attention``, ``backbone_output``, or ``backbone_layer_<k>``.

    Returns:
        The layer index for ``backbone_layer_<k>``, otherwise None.
    """
    if site in (ALIGN_SITE_POST_VL_SELF_ATTENTION, ALIGN_SITE_BACKBONE_OUTPUT):
        return None
    suffix = site[len(ALIGN_SITE_BACKBONE_LAYER_PREFIX) :]
    assert site.startswith(ALIGN_SITE_BACKBONE_LAYER_PREFIX) and suffix.isdigit(), (
        f"geometry_align_site={site!r} is not recognised. Use"
        f" {ALIGN_SITE_POST_VL_SELF_ATTENTION!r}, {ALIGN_SITE_BACKBONE_OUTPUT!r}, or"
        f" {ALIGN_SITE_BACKBONE_LAYER_PREFIX}<k>."
    )
    return int(suffix)


def token_grid_from_image_grid_thw(image_grid_thw, spatial_merge_size: int) -> tuple[int, int]:
    """Return the ``(rows, columns)`` of merged visual tokens for one image.

    Read from the processor's own ``image_grid_thw`` rather than inferred from the token count. The
    grid is **not** square in general: N1.7 preserves the camera's aspect ratio through
    ``shortest_image_edge``/``crop_fraction``, so a 4:3 frame yields an 8x11 grid of 88 tokens.
    Assuming a square grid silently misaligns geometry features against image tokens.

    Args:
        image_grid_thw: Processor output of shape ``(num_images, 3)`` holding ``(t, h, w)`` in
            unmerged patch units.
        spatial_merge_size: Side length of the token merge window.

    Returns:
        Tuple of merged-token rows and columns.
    """
    grid = torch.as_tensor(image_grid_thw).reshape(-1, 3)
    assert grid.shape[0] > 0, "image_grid_thw is empty, so there is no grid to align against."
    first = grid[0]
    assert bool((grid[:, 1:] == first[1:]).all()), (
        "Images in this batch have different patch grids"
        f" ({grid[:, 1:].tolist()}); geometry features are resampled to one grid, so a mixed batch"
        " would misalign them."
    )
    rows = int(first[1]) // spatial_merge_size
    columns = int(first[2]) // spatial_merge_size
    assert rows > 0 and columns > 0, (
        f"Degenerate token grid {rows}x{columns} from thw {first.tolist()}."
    )
    return rows, columns


class FrozenGeometryEncoder(nn.Module):
    """Frozen geometry encoder exposing a resampled patch-feature grid.

    The backbone's own patch grid (37x37 for Depth-Anything-V2 at 518 pixels) never matches the
    VLM's token grid, so features are bilinearly resampled to the requested grid. Resampling a
    feature map is what keeps geometry tokens in one-to-one correspondence with image tokens, which
    both consumption modes rely on.

    Args:
        config: Geometry conditioning configuration.
    """

    def __init__(self, config: GeometryConditioningConfig):
        super().__init__()
        self.config = config
        self._model = None
        self._kind: str | None = None
        self._feature_dim: int | None = None
        # ImageNet statistics, matching the encoder's own image processor. Applied on tensors so a
        # training step never round-trips through PIL.
        self.register_buffer(
            "_pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

    def _check_licence(self) -> None:
        """Refuse an encoder that is not on the Apache allowlist."""
        if self.config.allow_unlisted_encoder:
            return
        name = os.path.basename(str(self.config.encoder_id).rstrip("/"))
        assert name in APACHE_LICENSED_GEOMETRY_ENCODERS, (
            f"Geometry encoder {name!r} is not on the Apache allowlist"
            f" ({sorted(APACHE_LICENSED_GEOMETRY_ENCODERS)}). Hugging Face's licence metadata is"
            " unreliable here -- DA3-LARGE-1.1 advertises apache-2.0 while upstream's own table"
            " says CC BY-NC 4.0 -- so the gate is this list. Set allow_unlisted_encoder=True for a"
            " research-only run."
        )

    def _resolve_kind(self) -> str:
        """Return the encoder family, reading the checkpoint's own config when set to ``auto``."""
        if self.config.encoder_kind != "auto":
            return self.config.encoder_kind
        config_path = os.path.join(str(self.config.encoder_id), "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as handle:
                model_name = str(json.load(handle).get("model_name", ""))
            if model_name.startswith("da3"):
                return "da3"
        return "depth_anything_v2"

    def _ensure_loaded(self, device: torch.device) -> None:
        """Load the encoder on first use, so constructing the module stays cheap."""
        if self._model is not None:
            return
        self._check_licence()
        self._kind = self._resolve_kind()
        dtype = getattr(torch, self.config.encoder_dtype)

        if self._kind == "da3":
            model = self._load_da3(dtype)
        else:
            from transformers import AutoModelForDepthEstimation

            model = AutoModelForDepthEstimation.from_pretrained(
                self.config.encoder_id, torch_dtype=dtype
            )

        if self.config.freeze_encoder:
            model.requires_grad_(False)
        self._model = model.to(device).eval()

    def _load_da3(self, dtype: torch.dtype) -> nn.Module:
        """Return DA3's frozen ViT, built from the checkpoint's own construction spec.

        Deliberately avoids ``depth_anything_3.api``: importing it pulls
        ``depth_anything_3.utils.export``, whose ``__init__`` eagerly loads the COLMAP, GLB,
        Gaussian-splat and video exporters and so drags in seventeen third-party packages
        (``pycolmap``, ``trimesh``, ``moviepy==1.0.3`` among them) that a frozen feature extractor
        never touches. Building the net directly needs five, of which only ``addict`` is not
        already installed.

        The released safetensors were saved from that api wrapper, whose ``self.model`` is this
        net, so their keys need only the one prefix stripped. ``convert_metric_state_dict`` must
        **not** be applied: it targets the original ``torch.load`` research checkpoints and on this
        file it produces a doubled ``model.model.`` prefix that silently matches nothing.

        Args:
            dtype: Dtype to run the encoder in.

        Returns:
            DA3's ViT backbone, weights loaded, returning per-layer patch tokens.
        """
        from depth_anything_3.cfg import create_object
        from omegaconf import OmegaConf
        from safetensors.torch import load_file

        root = str(self.config.encoder_id)
        assert os.path.isdir(root), (
            f"DA3 encoders load from a local directory, got {root!r}. Fetch it first with"
            " `hf download <repo> --local-dir ${MODELS_DIR}<name>`."
        )
        with open(os.path.join(root, "config.json")) as handle:
            spec = json.load(handle)["config"]
        net = create_object(OmegaConf.create(spec))
        weights = load_file(os.path.join(root, "model.safetensors"))
        net.load_state_dict({k.removeprefix("model."): v for k, v in weights.items()}, strict=True)
        return net.backbone.to(dtype)

    def _forward_da3(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return DA3 patch tokens at the configured ``out_layers`` entry.

        No forward hook is needed and no token slicing either: ``DinoV2.forward`` already returns
        ``get_intermediate_layers(x, out_layers)``, which yields a ``(patch_tokens, cls_token)``
        pair per requested layer, and this monocular checkpoint emits no camera or register tokens
        to drop.

        Args:
            pixels: Normalised images as ``(N, 3, H, W)``.

        Returns:
            Patch tokens as ``(N, num_patches, feature_dim)``.
        """
        with torch.no_grad():
            # (N, S=1, 3, H, W): S is DA3's view axis, one view for a monocular teacher.
            layers, _aux = self._model(pixels.unsqueeze(1))
        patches, _cls = layers[self.config.da3_out_layer]
        return patches[:, 0]

    @property
    def feature_dim(self) -> int:
        """Channel width of the emitted geometry features, as measured by a forward pass."""
        assert self._feature_dim is not None, (
            "The encoder's feature width is only known after a forward pass. Call"
            " probe_feature_dim() at load time rather than reading a per-encoder table: VGGT's"
            " aggregated tokens are twice its configured width, so a tabulated width silently"
            " mis-sizes the projector."
        )
        return self._feature_dim

    def probe_feature_dim(self, device: torch.device | None = None) -> int:
        """Return the encoder's feature width, measured with one small forward pass.

        Args:
            device: Device to run the probe on. Defaults to the module's own device.

        Returns:
            Channel width of the emitted geometry features.
        """
        if self._feature_dim is not None:
            return self._feature_dim
        device = device if device is not None else self._pixel_mean.device
        probe = torch.zeros(1, 3, 64, 64, dtype=torch.uint8, device=device)
        with torch.no_grad():
            self(probe, grid=(1, 1))
        return self.feature_dim

    def _encoder_input_size(self, height: int, width: int) -> tuple[int, int]:
        """Return the ``(H, W)`` the encoder is run at, for the loaded encoder family.

        Depth-Anything-V2's own processor uses a square 518, so that path is unchanged. DA3 is a
        ViT/14 that accepts any multiple of its patch size, so its input keeps the camera's aspect
        ratio: the teacher's patch grid is then rectangular in the same sense the student's token
        grid is, which is what makes the resampling between them a rescale rather than a stretch.

        Args:
            height: Source image height.
            width: Source image width.

        Returns:
            Encoder input height and width.
        """
        if self._kind != "da3":
            return self.config.encoder_input_size, self.config.encoder_input_size
        patch = int(self._model.pretrained.patch_size)
        scale = self.config.encoder_input_size / min(height, width)
        return (
            max(patch, round(height * scale / patch) * patch),
            max(patch, round(width * scale / patch) * patch),
        )

    def forward(self, images_uint8: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        """Return geometry features for a batch of images, resampled to the given token grid.

        Args:
            images_uint8: Images as ``(N, 3, H, W)``, uint8.
            grid: Target token grid as ``(rows, columns)``. Need not be square.

        Returns:
            Features as ``(N, rows * columns, feature_dim)``.
        """
        self._ensure_loaded(images_uint8.device)
        dtype = getattr(torch, self.config.encoder_dtype)

        height, width = self._encoder_input_size(*images_uint8.shape[-2:])
        pixels = images_uint8.to(dtype) / 255.0
        pixels = F.interpolate(pixels, size=(height, width), mode="bilinear", align_corners=False)
        pixels = (pixels - self._pixel_mean.to(dtype)) / self._pixel_std.to(dtype)

        if self._kind == "da3":
            patches = self._forward_da3(pixels)
            patch = int(self._model.pretrained.patch_size)
            patch_rows, patch_columns = height // patch, width // patch
        else:
            with torch.no_grad():
                hidden = self._model(pixel_values=pixels, output_hidden_states=True).hidden_states[
                    -1
                ]
            # Drop the leading class token; the remainder is a square patch grid.
            patches = hidden[:, 1:, :]
            side = int(round(patches.shape[1] ** 0.5))
            assert side * side == patches.shape[1], (
                f"The encoder emitted {patches.shape[1]} patch tokens, which is not a square grid,"
                " so it cannot be resampled onto the VLM's token grid."
            )
            patch_rows = patch_columns = side

        assert patches.shape[1] == patch_rows * patch_columns, (
            f"The encoder emitted {patches.shape[1]} patch tokens against an expected"
            f" {patch_rows}x{patch_columns} grid. Reshaping on a wrong grid would rotate the"
            " features against the image and the misalignment would be invisible in the loss."
        )

        self._feature_dim = int(patches.shape[-1])
        feature_map = patches.transpose(1, 2).reshape(
            patches.shape[0], -1, patch_rows, patch_columns
        )
        resampled = F.interpolate(
            feature_map.to(dtype), size=grid, mode="bilinear", align_corners=False
        )
        return resampled.flatten(2).transpose(1, 2)


class GeometryConditioning(nn.Module):
    """Consume frozen geometry features either as an alignment target or as fused tokens.

    Args:
        config: Geometry conditioning configuration.
        backbone_dim: Hidden width of the VLM backbone features.
        geometry_dim: Channel width of the geometry features, measured from the encoder rather than
            tabulated.
    """

    def __init__(self, config: GeometryConditioningConfig, backbone_dim: int, geometry_dim: int):
        super().__init__()
        self.config = config
        self.backbone_dim = backbone_dim
        self.geometry_dim = geometry_dim

        if config.mode == "align":
            # Spatial Forcing projects the backbone's visual tokens into the geometry space and
            # aligns them there, so the geometry model is never needed at inference. Hidden and
            # output widths both equal the target width, following the reference implementation.
            self.align_student_norm = (
                nn.LayerNorm(backbone_dim) if config.align_use_student_norm else nn.Identity()
            )
            self.align_projector = nn.Sequential(
                nn.Linear(backbone_dim, geometry_dim),
                nn.GELU(),
                nn.Linear(geometry_dim, geometry_dim),
            )
            for layer in self.align_projector:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

            # Added to the *target*, so the supervised tokens keep their position order. Dropping
            # this term costs ten points on LIBERO-Long in Spatial Forcing's own ablation.
            self.align_position_embedding = nn.Parameter(
                torch.empty(1, config.align_max_tokens, geometry_dim)
            )
            nn.init.trunc_normal_(
                self.align_position_embedding, std=config.align_position_embedding_std
            )
        elif config.mode == "mix":
            # 3D-Mix GatedFusion: project geometry, summarise semantics, gate per position, blend.
            self.geometry_projector = nn.Linear(geometry_dim, backbone_dim)
            self.gate = nn.Linear(2 * backbone_dim, backbone_dim)
            self.semantic_value = nn.Linear(backbone_dim, backbone_dim)
            self.geometry_value = nn.Linear(backbone_dim, backbone_dim)

    def alignment_loss(
        self,
        image_tokens: torch.Tensor,
        geometry_features: torch.Tensor,
        align_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the cosine alignment loss between backbone image tokens and geometry features.

        The loss is ``1 - cos`` on explicitly normalised vectors, averaged over each sample's valid
        tokens and then over the batch, which is the reference implementation's reduction and keeps
        the value non-negative and comparable to its logs.

        Args:
            image_tokens: Backbone image tokens as ``(B, N, backbone_dim)``.
            geometry_features: Geometry features as ``(B, N, geometry_dim)``.
            align_mask: Boolean mask of valid image tokens as ``(B, N)``. Defaults to all valid.
                Padding introduced by a letterbox transform must be masked here, or the projector
                trains against blank pixels.

        Returns:
            Scalar loss in ``[0, 2]``.
        """
        assert image_tokens.ndim == 3 and geometry_features.ndim == 3, (
            f"Expected (B, N, D) tensors, got {tuple(image_tokens.shape)} and"
            f" {tuple(geometry_features.shape)}. The positional embedding and the per-sample mean"
            " are both defined over the token axis, so a pre-flattened batch cannot be used."
        )
        assert image_tokens.shape[:2] == geometry_features.shape[:2], (
            f"Got {tuple(image_tokens.shape[:2])} backbone image tokens against"
            f" {tuple(geometry_features.shape[:2])} geometry tokens. Alignment is positional, so a"
            " count mismatch means the grids disagree and the loss would be meaningless."
        )
        num_tokens = image_tokens.shape[1]
        assert num_tokens <= self.config.align_max_tokens, (
            f"{num_tokens} image tokens exceeds align_max_tokens={self.config.align_max_tokens},"
            " so the target positional embedding cannot cover the sequence."
        )

        projected = self.align_projector(self.align_student_norm(image_tokens))
        target = geometry_features.to(projected.dtype) + self.align_position_embedding[
            :, :num_tokens
        ].to(projected.dtype)

        cosine = (F.normalize(projected, dim=-1) * F.normalize(target, dim=-1)).sum(dim=-1)

        if align_mask is None:
            align_mask = torch.ones(cosine.shape, dtype=torch.bool, device=cosine.device)
        assert align_mask.shape == cosine.shape, (
            f"align_mask {tuple(align_mask.shape)} does not match the token grid"
            f" {tuple(cosine.shape)}; an unaligned mask would supervise the wrong positions."
        )
        weights = align_mask.to(cosine.dtype)
        per_sample = (cosine * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

        # A sample whose image tokens are entirely masked carries no signal, so it is dropped rather
        # than counted as a perfect mismatch.
        has_tokens = align_mask.any(dim=1)
        assert bool(has_tokens.any()), (
            "Every sample in this batch has all image tokens masked out, so there is nothing to"
            " align. This means the image mask and the geometry grid disagree."
        )
        return (1.0 - per_sample)[has_tokens].mean()

    def fuse(
        self,
        backbone_features: torch.Tensor,
        image_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        geometry_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gate geometry features against pooled semantics and append them to the conditioning.

        Args:
            backbone_features: Backbone output as ``(B, L, backbone_dim)``.
            image_mask: Boolean image-token mask as ``(B, L)``.
            attention_mask: Boolean attention mask as ``(B, L)``.
            geometry_features: Geometry features as ``(B, N, geometry_dim)``.

        Returns:
            The extended features ``(B, L + N, backbone_dim)`` and the correspondingly extended
            image and attention masks.
        """
        geometry = self.geometry_projector(geometry_features.to(backbone_features.dtype))

        # Pooled semantic context, broadcast to every geometry position. The sequences have
        # different lengths, so the gate blends context against geometry rather than token pairs.
        valid = attention_mask.unsqueeze(-1).to(backbone_features.dtype)
        pooled = (backbone_features * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        semantic = pooled.unsqueeze(1).expand(-1, geometry.shape[1], -1)

        gate = torch.sigmoid(self.gate(torch.cat([semantic, geometry], dim=-1)))
        fused = gate * self.semantic_value(semantic) + (1.0 - gate) * self.geometry_value(geometry)

        extended = torch.cat([backbone_features, fused], dim=1)
        is_image = self.config.mix_tokens_as == "image"
        geometry_image_mask = torch.full(
            fused.shape[:2], is_image, dtype=image_mask.dtype, device=image_mask.device
        )
        geometry_attention_mask = torch.ones(
            fused.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device
        )
        return (
            extended,
            torch.cat([image_mask, geometry_image_mask], dim=1),
            torch.cat([attention_mask, geometry_attention_mask], dim=1),
        )
