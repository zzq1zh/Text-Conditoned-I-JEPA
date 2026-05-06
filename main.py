"""
Hugging Face vision-backbone loading and text-conditioned vision model definitions.
Fine-tune with ``uv run python text_cond_train.py`` (defaults to CUDA when available;
use ``--cpu`` or ``--device cpu`` to force CPU). W&B optional. Copy ``.env.example`` to
``.env`` for ``WANDB_API_KEY`` (optional).
"""

from __future__ import annotations

import project_env

project_env.load_project_env()

import gc
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoVideoProcessor,
    CLIPModel,
    CLIPTextModelWithProjection,
    PreTrainedModel,
)

DEFAULT_PROMPT_TEMPLATE = "a photo of a {c}."


def load_clip_text_encoder_for_conditioning(clip_model_id: str) -> CLIPTextModelWithProjection:
    """
    Load only the CLIP text tower + text projection, without spurious "UNEXPECTED" load keys.

    :func:`CLIPTextModelWithProjection.from_pretrained` reads the *full* CLIP checkpoint
    (vision + ``logit_scale`` + etc.); keys that the text-only module does not use are
    reported as UNEXPECTED. Instead, load :class:`CLIPModel` (matches the file 1:1), copy
    ``text_model`` and ``text_projection`` into a :class:`CLIPTextModelWithProjection`, then
    drop the full model to avoid keeping vision weights in RAM.
    """
    full = CLIPModel.from_pretrained(clip_model_id)
    text = CLIPTextModelWithProjection._from_config(full.text_model.config)
    text.text_model.load_state_dict(full.text_model.state_dict())
    text.text_projection.load_state_dict(full.text_projection.state_dict())
    del full
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text

# Hub checkpoint ids for `--vision-backbone` presets. Default alias: ``dinov3``.
DEFAULT_IJEPA_ID = "facebook/ijepa_vith14_1k"
DEFAULT_VJEPA_ID = "facebook/vjepa2-vitl-fpc64-256"
DEFAULT_DINOV3_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
VISION_BACKBONE_PRESETS: dict[str, str] = {
    "ijepa": DEFAULT_IJEPA_ID,
    "vjepa": DEFAULT_VJEPA_ID,
    "dinov3": DEFAULT_DINOV3_ID,
}
# CLIP text tower for text conditioning (align with open_clip / CLIP paper)
DEFAULT_CLIP_TEXT_ID = "openai/clip-vit-base-patch32"


@dataclass
class VisionBackbonePipeline:
    """Holds a vision backbone :class:`~transformers.PreTrainedModel`, its processor, and device."""

    model: PreTrainedModel
    processor: Any
    device: torch.device
    model_id: str


def normalize_backbone_name(backbone: str) -> str:
    """Trim, lowercase, and normalize underscores to hyphens for preset lookup."""
    return backbone.strip().lower().replace("_", "-")


def resolve_vision_model_id(backbone: str = "dinov3", model_id: str = "") -> str:
    """
    Resolve the vision backbone id from an alias or explicit model id override.
    Explicit ``model_id`` wins to preserve backward compatibility.
    """
    mid = (model_id or "").strip()
    if mid:
        return mid
    key = normalize_backbone_name(backbone)
    if key not in VISION_BACKBONE_PRESETS:
        opts = ", ".join(sorted(VISION_BACKBONE_PRESETS.keys()))
        raise ValueError(f"Unsupported vision backbone {backbone!r}; expected one of: {opts}")
    return VISION_BACKBONE_PRESETS[key]


def _extract_model_pixel_values(processed: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Extract pixel values from an image/video processor output and normalize to one key.
    Supports image models (``pixel_values``) and video models (``pixel_values_videos``).
    """
    if "pixel_values" in processed:
        return processed["pixel_values"]
    if "pixel_values_videos" in processed:
        return processed["pixel_values_videos"]
    keys = ", ".join(sorted(processed.keys()))
    raise KeyError(
        "Processor output does not contain 'pixel_values' or 'pixel_values_videos'. "
        f"Found keys: {keys}"
    )


def load_vision_processor(model_id: str) -> Any:
    """
    Load image/video processor for a backbone id.
    Prefers ``AutoImageProcessor`` and falls back to ``AutoVideoProcessor``.
    """
    try:
        return AutoImageProcessor.from_pretrained(model_id)
    except (ValueError, OSError):
        return AutoVideoProcessor.from_pretrained(model_id)


def load_vision_backbone_pipeline(
    model_id: str = DEFAULT_DINOV3_ID,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> VisionBackbonePipeline:
    """
    Load a vision backbone and preprocessor from the Hugging Face Hub (``AutoModel`` + image/video processor).
    Typical ViTs expose patch tokens in ``last_hidden_state`` (mean-pool for a global vector for heads).
    """
    if device is not None:
        dev = torch.device(device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        # Match checkpoint weights; float32 is safest if unset
        dtype = torch.float32
    model = AutoModel.from_pretrained(model_id, dtype=dtype)
    model.to(dev)
    model.eval()
    processor = load_vision_processor(model_id)
    return VisionBackbonePipeline(model=model, processor=processor, device=dev, model_id=model_id)


def forward_vision_backbone(
    pipe: VisionBackbonePipeline, pixel_values: torch.Tensor, *, amp: bool = False
) -> torch.Tensor:
    """
    Run the loaded vision backbone and return a token sequence
    ``(batch, seq_len, hidden)``. Caller may mean-pool for classification.
    """
    pixel_values = pixel_values.to(device=pipe.device, dtype=next(pipe.model.parameters()).dtype)
    with torch.inference_mode(), torch.autocast(
        device_type=pipe.device.type, dtype=torch.float16, enabled=amp and pipe.device.type == "cuda"
    ):
        if pixel_values.ndim == 5:
            out = pipe.model(pixel_values_videos=pixel_values)
        else:
            out = pipe.model(pixel_values=pixel_values)
    seq = getattr(out, "last_hidden_state", None)
    if seq is None:
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            raise RuntimeError(
                "Backbone output is missing both last_hidden_state and pooler_output."
            )
        seq = pooled.unsqueeze(1) if pooled.ndim == 2 else pooled
    if seq.ndim == 2:
        seq = seq.unsqueeze(1)
    if seq.ndim > 3:
        seq = seq.reshape(seq.shape[0], -1, seq.shape[-1])
    return seq


class TextConditioningModule(nn.Module):
    """
    Encodes text with a frozen (by default) CLIP text+projection stack, then maps
    to a fixed conditioning size for the fusion head.
    """

    def __init__(
        self,
        clip_model_id: str = DEFAULT_CLIP_TEXT_ID,
        cond_dim: int = 256,
        freeze_text_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.text_encoder = load_clip_text_encoder_for_conditioning(clip_model_id)
        self._freeze_text = freeze_text_encoder
        p_dim: int = int(self.text_encoder.config.projection_dim)
        self.adapter = nn.Sequential(
            nn.LayerNorm(p_dim),
            nn.Linear(p_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self._freeze_text:
            self.text_encoder.eval()
        # No grad through CLIP text weights when frozen; adapter remains trainable
        if self._freeze_text:
            with torch.no_grad():
                out = self.text_encoder(
                    input_ids=input_ids, attention_mask=attention_mask, return_dict=True
                )
                text_embeds = out.text_embeds
        else:
            out = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            text_embeds = out.text_embeds
        if text_embeds is None:
            raise RuntimeError("CLIPTextModelWithProjection did not return text_embeds")
        return self.adapter(text_embeds)


class FusionHead(nn.Module):
    """
    Fuses visual features and a text conditioning vector into a scalar score.
    - ``cross_attention``: **sequence** of visual tokens ``(B, N, D_v)`` (or ``(B, D_v)`` as length-1)
      each attends to one text token, then per-position CLIP-style cosine similarities are **summed**
      over ``N`` (no mean-pool on the backbone; scalar is a sum of patch scores, scaled).
    - ``clip_similarity``: pooled ``(B, D_v)`` only — direct projections + normalized dot product.
    """

    def __init__(
        self,
        vis_dim: int,
        cond_dim: int,
        hidden: int = 512,
        fusion_type: str = "cross_attention",
    ) -> None:
        super().__init__()
        self.fusion_type = fusion_type
        if self.fusion_type == "linear":
            # Pairwise score head: (image, text) -> scalar score.
            self.score_head = nn.Linear(vis_dim + cond_dim, 1)
        elif self.fusion_type == "clip_similarity":
            # CLIP-style pair score: learned projections + normalized dot product with logit scale.
            self.visual_proj = nn.Linear(vis_dim, hidden, bias=False)
            self.text_proj = nn.Linear(cond_dim, hidden, bias=False)
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        elif self.fusion_type == "cross_attention":
            n_heads = 8
            while n_heads > 1 and hidden % n_heads != 0:
                n_heads //= 2
            self.visual_proj = nn.Linear(vis_dim, hidden)
            self.text_proj = nn.Linear(cond_dim, hidden)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden, num_heads=n_heads, dropout=0.1, batch_first=True
            )
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        else:
            raise ValueError(
                f"Unsupported fusion_type={self.fusion_type!r}; expected 'cross_attention', 'linear', or 'clip_similarity'"
            )

    def forward(self, visual: torch.Tensor, text_cond: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "linear":
            if visual.ndim != 2:
                raise ValueError(
                    f"fusion_type='linear' expects visual (B, D_v); got shape {tuple(visual.shape)}"
                )
            s = self.score_head(torch.cat([visual, text_cond], dim=-1))
            return s.squeeze(-1)
        if self.fusion_type == "clip_similarity":
            if visual.ndim != 2:
                raise ValueError(
                    f"fusion_type='clip_similarity' expects visual (B, D_v); got shape {tuple(visual.shape)}"
                )
            v = nn.functional.normalize(self.visual_proj(visual), dim=-1)
            t = nn.functional.normalize(self.text_proj(text_cond), dim=-1)
            # Same scaling form as CLIP: exp(logit_scale) * cosine_similarity.
            scale = self.logit_scale.exp().clamp(max=100.0)
            return (v * t).sum(dim=-1) * scale
        # cross_attention: (B, N, D_v) queries, text as single KV; sum of per-token cosine scores.
        if visual.ndim == 2:
            visual = visual.unsqueeze(1)
        q = self.text_proj(text_cond).unsqueeze(1)  # [B, 1, D]
        kv = self.visual_proj(visual)               # [B, N_img, D]

        attn_out, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            need_weights=False,
        )

        t = q.squeeze(1)
        v = attn_out.squeeze(1)

        t = nn.functional.normalize(t, dim=-1)
        v = nn.functional.normalize(v, dim=-1)

        scale = self.logit_scale.exp().clamp(max=100.0)
        return (v * t).sum(dim=-1) * scale


class TextConditionedVisionModel(nn.Module):
    """
    Vision backbone (I-JEPA / ViT / DINOv3) optionally frozen + text conditioning + fusion scorer.
    For ``fusion_type='cross_attention'``, image features are a **token sequence** (no pre-fusion mean-pool).
    When the backbone is frozen, visual features use inference mode and are detached.
    """

    def __init__(
        self,
        num_labels: int,
        ijepa_id: str = DEFAULT_DINOV3_ID,
        clip_id: str = DEFAULT_CLIP_TEXT_ID,
        cond_dim: int = 256,
        fusion_hidden: int = 512,
        fusion_type: str = "cross_attention",
        freeze_text_encoder: bool = True,
        freeze_vision_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.freeze_vision_backbone = bool(freeze_vision_backbone)
        self.backbone = AutoModel.from_pretrained(ijepa_id)
        if self.freeze_vision_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        vis_dim = int(self.backbone.config.hidden_size)

        self.text_cond = TextConditioningModule(
            clip_model_id=clip_id,
            cond_dim=cond_dim,
            freeze_text_encoder=freeze_text_encoder,
        )
        self.fusion = FusionHead(
            vis_dim=vis_dim,
            cond_dim=cond_dim,
            hidden=fusion_hidden,
            fusion_type=fusion_type,
        )
        self.num_labels = num_labels
        self.cond_dim = cond_dim

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Returns ``(B, N, D)`` token features for ``cross_attention``, else mean-pooled ``(B, D)``.
        """
        use_token_seq = self.fusion.fusion_type == "cross_attention"
        if self.freeze_vision_backbone:
            with torch.inference_mode():
                if pixel_values.ndim == 5:
                    enc = self.backbone(pixel_values_videos=pixel_values)
                else:
                    enc = self.backbone(pixel_values=pixel_values)
                seq = getattr(enc, "last_hidden_state", None)
                if seq is None:
                    pooled = getattr(enc, "pooler_output", None)
                    if pooled is None:
                        raise RuntimeError(
                            "Backbone output is missing both last_hidden_state and pooler_output."
                        )
                    seq = pooled.unsqueeze(1) if pooled.ndim == 2 else pooled
                if seq.ndim == 2:
                    seq = seq.unsqueeze(1)
                if seq.ndim > 3:
                    seq = seq.reshape(seq.shape[0], -1, seq.shape[-1])
                seq = seq.float()
                if use_token_seq:
                    return seq.detach()
                return seq.mean(dim=1).detach()
        if pixel_values.ndim == 5:
            enc = self.backbone(pixel_values_videos=pixel_values)
        else:
            enc = self.backbone(pixel_values=pixel_values)
        seq = getattr(enc, "last_hidden_state", None)
        if seq is None:
            pooled = getattr(enc, "pooler_output", None)
            if pooled is None:
                raise RuntimeError(
                    "Backbone output is missing both last_hidden_state and pooler_output."
                )
            seq = pooled.unsqueeze(1) if pooled.ndim == 2 else pooled
        if seq.ndim == 2:
            seq = seq.unsqueeze(1)
        if seq.ndim > 3:
            seq = seq.reshape(seq.shape[0], -1, seq.shape[-1])
        seq = seq.float()
        if use_token_seq:
            return seq
        return seq.mean(dim=1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.text_cond(input_ids, attention_mask)

    def score_pairs(self, image_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        """Pairwise score for aligned rows: ``(B, Dv)`` or ``(B, N, Dv)`` × ``(B, Dt)`` -> ``(B,)``."""
        return self.fusion(image_feats, text_feats)

    def score_candidates(
        self, image_feats: torch.Tensor, candidate_text_embeds: torch.Tensor, chunk_size: int = 256
    ) -> torch.Tensor:
        """
        Score all candidates for each image.
        image_feats: ``(B, D_v)`` or ``(B, N, D_v)`` (tokens for cross-attention),
        candidate_text_embeds: (C, Dt) -> scores: (B, C)
        """
        b = image_feats.size(0)
        c = candidate_text_embeds.size(0)
        out_chunks: list[torch.Tensor] = []
        for s in range(0, c, chunk_size):
            e = min(c, s + chunk_size)
            t = candidate_text_embeds[s:e]  # (Cc, Dt)
            cc = t.size(0)
            if image_feats.dim() == 2:
                img_rep = image_feats.unsqueeze(1).expand(-1, cc, -1).reshape(b * cc, -1)
            else:
                n_tok = image_feats.size(1)
                dv = image_feats.size(2)
                img_rep = (
                    image_feats.unsqueeze(1)
                    .expand(b, cc, n_tok, dv)
                    .reshape(b * cc, n_tok, dv)
                )
            txt_rep = t.unsqueeze(0).expand(b, -1, -1).reshape(b * cc, -1)
            sc = self.score_pairs(img_rep, txt_rep).view(b, cc)
            out_chunks.append(sc)
        return torch.cat(out_chunks, dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        neg_input_ids: torch.Tensor | None = None,
        neg_attention_mask: torch.Tensor | None = None,
        contrast_loss_weight: float = 0.0,
    ) -> dict[str, Any]:
        # Backward-compatible pairwise forward for a provided text batch.
        z = self.encode_image(pixel_values)
        c = self.encode_text(input_ids, attention_mask)
        pair_scores = self.score_pairs(z, c)
        loss: torch.Tensor | None = None
        loss_ce: torch.Tensor | None = None
        loss_contrast: torch.Tensor | None = None
        if labels is not None:
            # Keep a finite scalar for logging in legacy calls.
            loss_ce = pair_scores.new_zeros(())
            loss = loss_ce
        return {
            "loss": loss,
            "logits": pair_scores,
            "loss_ce": loss_ce,
            "loss_contrast": loss_contrast,
        }

