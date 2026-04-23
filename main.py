"""
Entry points and I-JEPA loading pipeline. Inline comments in English.
Fine-tune text-conditioned I-JEPA: ``uv run python text_cond_train.py`` (defaults to CUDA GPU when
available; use ``--cpu`` or ``--device cpu`` to force CPU). W&B optional. Copy ``.env.example`` to
``.env`` for ``HF_TOKEN`` / ``WANDB_API_KEY``.
"""

from __future__ import annotations

import project_env

project_env.load_project_env()

import argparse
import gc
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    CLIPModel,
    CLIPTextModelWithProjection,
    PreTrainedModel,
)

from vision_data import (
    list_vision_dataset_keys,
    load_vision_train_val_test_specs,
    prompts_for_label_indices,
)

# Same template as CLIP zero-shot in clip_pipeline / vision_data usage
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

# Default checkpoint: ViT-H/14 pretrained on IN1K (HuggingFace I-JEPA)
DEFAULT_IJEPA_ID = "facebook/ijepa_vith14_1k"
# CLIP text tower for text conditioning (align with open_clip / CLIP paper)
DEFAULT_CLIP_TEXT_ID = "openai/clip-vit-base-patch32"


@dataclass
class IJepaPipeline:
    """Holds a backbone IJepa model, its image processor, and the runtime device."""

    model: PreTrainedModel
    processor: Any
    device: torch.device
    model_id: str


def _resolve_device(device: str | torch.device | None) -> torch.device:
    """
    If ``device`` is ``None``, use the first available CUDA device (default for training/inference
    on GPU). Otherwise return that device, e.g. ``cpu`` to force the CPU.
    """
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def freeze_backbone_(module: nn.Module) -> None:
    """Set all parameters to frozen (no weight updates in the backbone)."""
    for p in module.parameters():
        p.requires_grad = False


def load_ijepa_pipeline(
    model_id: str = DEFAULT_IJEPA_ID,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> IJepaPipeline:
    """
    Load I-JEPA backbone + ViT image preprocessor from the Hugging Face Hub.
    The backbone outputs patch tokens in last_hidden_state (mean-pool for a global
    vector, as in MyIJepaClassifier).
    """
    dev = _resolve_device(device)
    if dtype is None:
        # Match checkpoint weights; float32 is safest if unset
        dtype = torch.float32
    model = AutoModel.from_pretrained(model_id, dtype=dtype)
    model.to(dev)
    model.eval()
    processor = AutoImageProcessor.from_pretrained(model_id)
    return IJepaPipeline(model=model, processor=processor, device=dev, model_id=model_id)


def forward_ijepa_backbone(
    pipe: IJepaPipeline, pixel_values: torch.Tensor, *, amp: bool = False
) -> torch.Tensor:
    """
    Run the backbone and return L2-friendly patch token sequence
    (batch, seq_len, hidden). Caller may mean-pool for classification.
    """
    pixel_values = pixel_values.to(device=pipe.device, dtype=next(pipe.model.parameters()).dtype)
    with torch.inference_mode(), torch.autocast(
        device_type=pipe.device.type, dtype=torch.float16, enabled=amp and pipe.device.type == "cuda"
    ):
        out = pipe.model(pixel_values=pixel_values)
    return out.last_hidden_state


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
    Fuses a pooled visual vector (I-JEPA) and a text conditioning vector.
    Only this module and TextConditioningModule (adapter) are trained when the
    backbones are frozen.
    """

    def __init__(
        self,
        vis_dim: int,
        cond_dim: int,
        num_labels: int,
        hidden: int = 512,
    ) -> None:
        super().__init__()
        in_dim = vis_dim + cond_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, visual: torch.Tensor, text_cond: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([visual, text_cond], dim=-1))


class TextConditionedIJepa(nn.Module):
    """
    Frozen I-JEPA trunk + trainable text conditioning + fusion classifier.
    Visual features are detached so gradients do not flow into the I-JEPA weights.
    """

    def __init__(
        self,
        num_labels: int,
        ijepa_id: str = DEFAULT_IJEPA_ID,
        clip_id: str = DEFAULT_CLIP_TEXT_ID,
        cond_dim: int = 256,
        fusion_hidden: int = 512,
        freeze_text_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(ijepa_id)
        freeze_backbone_(self.backbone)
        self.backbone.eval()
        vis_dim = int(self.backbone.config.hidden_size)

        self.text_cond = TextConditioningModule(
            clip_model_id=clip_id,
            cond_dim=cond_dim,
            freeze_text_encoder=freeze_text_encoder,
        )
        self.fusion = FusionHead(
            vis_dim=vis_dim, cond_dim=cond_dim, num_labels=num_labels, hidden=fusion_hidden
        )

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
        with torch.inference_mode():
            enc = self.backbone(pixel_values=pixel_values)
            # Detach: train fusion + text adapter only, not I-JEPA
            z = enc.last_hidden_state.float().mean(dim=1).detach()
        c = self.text_cond(input_ids, attention_mask)
        logits = self.fusion(z, c)

        loss: torch.Tensor | None = None
        loss_ce: torch.Tensor | None = None
        loss_contrast: torch.Tensor | None = None
        if labels is not None:
            loss_ce = nn.functional.cross_entropy(logits, labels)
            loss = loss_ce
        if (
            labels is not None
            and contrast_loss_weight > 0
            and neg_input_ids is not None
            and neg_attention_mask is not None
        ):
            b = z.size(0)
            c_neg = self.text_cond(neg_input_ids, neg_attention_mask)
            z_rep = z.repeat_interleave(4, dim=0)
            logits_neg = self.fusion(z_rep, c_neg)
            logits_neg = logits_neg.view(b, 4, -1)
            y = labels.view(b, 1, 1).expand(b, 4, 1)
            s_pos = logits.gather(1, labels.view(b, 1)).squeeze(1)
            s_neg = logits_neg.gather(2, y).squeeze(2)
            stack = torch.cat([s_pos.unsqueeze(1), s_neg], dim=1)
            loss_contrast = nn.functional.cross_entropy(
                stack, torch.zeros(b, device=stack.device, dtype=torch.long)
            )
            assert loss is not None
            loss = loss + contrast_loss_weight * loss_contrast
        return {
            "loss": loss,
            "logits": logits,
            "loss_ce": loss_ce,
            "loss_contrast": loss_contrast,
        }


class MyIJepaClassifier(nn.Module):
    def __init__(self, model_name: str = DEFAULT_IJEPA_ID, num_labels: int = 2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels),
        )

    def forward(self, pixel_values, labels=None):
        outputs = self.backbone(pixel_values=pixel_values)
        x = outputs.last_hidden_state.mean(dim=1)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


def _random_pil_image(size: int = 224) -> Image.Image:
    """Synthetic RGB image for a local smoke test (no disk I/O)."""
    import numpy as np

    arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def _run_smoke_test(model_id: str, cpu: bool) -> None:
    device = "cpu" if cpu else None
    pipe = load_ijepa_pipeline(model_id=model_id, device=device)
    print(f"Loaded {pipe.model_id} on {pipe.device}")

    im = _random_pil_image(224)
    inputs = pipe.processor(images=im, return_tensors="pt")
    inputs = {k: v.to(pipe.device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = pipe.model(**inputs)
    seq = out.last_hidden_state
    print(f"last_hidden_state shape: {tuple(seq.shape)}")

    clf = MyIJepaClassifier(model_name=model_id, num_labels=2)
    clf.to(pipe.device)
    clf.eval()
    with torch.inference_mode():
        z = clf(pixel_values=inputs["pixel_values"])
    print(f"custom head logits shape: {tuple(z['logits'].shape)}")


def _msamples(x: int) -> int | None:
    """0 or negative = use full split (per vision_data)."""
    return x if x and x > 0 else None


def _run_text_fusion_smoke(
    ijepa_id: str,
    cpu: bool,
    dataset_key: str = "cifar100",
    val_fraction: float = 0.1,
    split_seed: int = 0,
    max_train_samples: int = 8,
    max_val_samples: int = 4,
    max_test_samples: int = 4,
    text_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> None:
    """
    End-to-end check: :func:`vision_data.load_vision_train_val_test_specs` (train/val
    from Hub train, test = Hub test) + I-JEPA + text conditioning + fusion. One
    forward per split.
    """
    device = _resolve_device("cpu" if cpu else None)
    print(
        f"Text-conditioned I-JEPA on {device} | dataset={dataset_key} "
        f"val_fraction={val_fraction} split_seed={split_seed} "
        f"(max train/val/test = {max_train_samples}/{max_val_samples}/{max_test_samples}; "
        f"test split = val + official hub test)"
    )

    tvt = load_vision_train_val_test_specs(
        dataset_key,
        val_fraction=val_fraction,
        split_seed=split_seed,
        max_train_samples=_msamples(max_train_samples),
        max_val_samples=_msamples(max_val_samples),
        max_test_samples=_msamples(max_test_samples),
    )
    n_classes = len(tvt.train.class_names)
    for tag, spec in (
        ("train", tvt.train),
        ("val", tvt.val),
        ("test", tvt.test),
    ):
        n = len(spec.dataset)
        if n == 0:
            raise RuntimeError(f"Empty {tag} split after loading")

    pipe = load_ijepa_pipeline(model_id=ijepa_id, device=device)
    tok = AutoTokenizer.from_pretrained(DEFAULT_CLIP_TEXT_ID)
    model = TextConditionedIJepa(
        num_labels=n_classes, ijepa_id=ijepa_id, cond_dim=256
    )
    model.to(device)

    for tag, spec in (
        ("train", tvt.train),
        ("val", tvt.val),
        ("test", tvt.test),
    ):
        if tag == "train":
            model.train()
        else:
            model.eval()
        n = len(spec.dataset)
        batch_idx = list(range(n))
        images = [spec.dataset[i][spec.image_column] for i in batch_idx]
        labels_list = [int(spec.dataset[i][spec.label_key]) for i in batch_idx]
        texts = prompts_for_label_indices(spec.class_names, text_template, labels_list)
        enc_text = tok(texts, padding=True, return_tensors="pt")
        input_ids = enc_text["input_ids"].to(device)
        attention_mask = enc_text["attention_mask"].to(device)
        vis = pipe.processor(images=images, return_tensors="pt")
        pixel_values = vis["pixel_values"].to(device)
        labels_t = torch.tensor(labels_list, dtype=torch.long, device=device)
        with torch.set_grad_enabled(tag == "train"):
            out = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_t,
            )
        loss_v = out["loss"]
        loss_f = float(loss_v.detach()) if loss_v is not None else float("nan")
        print(
            f"[{tag}] n={n} | logits {tuple(out['logits'].shape)} | loss: {loss_f:.4f}"
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="I-JEPA load pipeline smoke test")
    p.add_argument("--model", default=DEFAULT_IJEPA_ID, help="HuggingFace model id (I-JEPA)")
    p.add_argument("--cpu", action="store_true", help="Force CPU (no CUDA)")
    p.add_argument(
        "--text-fusion",
        action="store_true",
        help="Run frozen I-JEPA + text conditioning + fusion on a real vision split (vision_data)",
    )
    p.add_argument(
        "--dataset",
        default="cifar100",
        choices=list_vision_dataset_keys(),
        help="Key from vision_data (same as clip_pipeline --dataset).",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of Hub train taken as validation; train/val from Hub train, test from Hub test",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Seed for train/val sub-split (stratified by label when possible)",
    )
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=8,
        help="Cap train split after sub-split; 0 = all train rows in that part",
    )
    p.add_argument(
        "--max-val-samples",
        type=int,
        default=4,
        help="Cap val split; 0 = all val rows",
    )
    p.add_argument(
        "--max-test-samples",
        type=int,
        default=4,
        help="Cap test split; 0 = full Hub test",
    )
    p.add_argument(
        "--text-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Class prompt; use {c} for class name (match CLIP eval).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.text_fusion:
        _run_text_fusion_smoke(
            ijepa_id=args.model,
            cpu=args.cpu,
            dataset_key=args.dataset,
            val_fraction=args.val_fraction,
            split_seed=args.split_seed,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            max_test_samples=args.max_test_samples,
            text_template=args.text_template,
        )
    else:
        _run_smoke_test(model_id=args.model, cpu=args.cpu)
