"""
CSP-oriented evaluation for vision–text models (text-cond training + CSP vocab post-train).

Computes optional mean contrastive loss, top-1 / top-5, seen vs unseen breakdown, and CSP-style AUC
when ``pair_seen_in_train`` is present in batches.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from main import TextConditionedIJepa


def clip_contrastive_loss(pair_scores: torch.Tensor) -> torch.Tensor:
    """
    CLIP-style symmetric InfoNCE from a pair score matrix.
    ``pair_scores``: (B, B), diagonal entries are positives.
    """
    if pair_scores.ndim != 2 or pair_scores.size(0) != pair_scores.size(1):
        raise ValueError(
            f"pair_scores must be square (B,B), got shape={tuple(pair_scores.shape)}"
        )
    target = torch.arange(pair_scores.size(0), device=pair_scores.device, dtype=torch.long)
    loss_i2t = F.cross_entropy(pair_scores, target)
    loss_t2i = F.cross_entropy(pair_scores.t(), target)
    return 0.5 * (loss_i2t + loss_t2i)


@torch.inference_mode()
def compute_auc_csp_style(
    logits_list: list[torch.Tensor],
    labels_list: list[torch.Tensor],
    seen_flags_list: list[torch.Tensor],
    num_classes: int,
) -> float:
    """
    CSP-style AUC: sweep an unseen-class bias and integrate seen/unseen top-1 trade-off.

    Returns NaN when seen/unseen supervision is unavailable or the split lacks both kinds.
    """
    if not logits_list or not labels_list or not seen_flags_list:
        return float("nan")
    logits = torch.cat(logits_list, dim=0).float()
    labels = torch.cat(labels_list, dim=0).long()
    seen_flags = torch.cat(seen_flags_list, dim=0).bool()
    if logits.ndim != 2 or labels.ndim != 1 or seen_flags.ndim != 1:
        return float("nan")
    if logits.size(0) != labels.numel() or labels.numel() != seen_flags.numel():
        return float("nan")
    if logits.size(1) != num_classes or num_classes <= 1:
        return float("nan")

    class_seen = torch.zeros(num_classes, dtype=torch.bool)
    for y, sf in zip(labels.tolist(), seen_flags.tolist(), strict=False):
        if 0 <= y < num_classes and bool(sf):
            class_seen[y] = True
    if (not class_seen.any()) or bool(class_seen.all()):
        return float("nan")

    unseen_sample_mask = ~class_seen[labels]
    if unseen_sample_mask.sum().item() == 0:
        return float("nan")
    seen_sample_mask = ~unseen_sample_mask
    if seen_sample_mask.sum().item() == 0:
        return float("nan")

    unseen_logits = logits[unseen_sample_mask]
    unseen_labels = labels[unseen_sample_mask]
    correct_scores = unseen_logits.gather(1, unseen_labels.view(-1, 1)).squeeze(1)
    max_seen_scores = unseen_logits[:, class_seen].max(dim=1).values
    unseen_score_diff = max_seen_scores - correct_scores

    pred0 = logits.argmax(dim=1)
    unseen_matches0 = pred0[unseen_sample_mask].eq(unseen_labels)
    correct_unseen_diff = unseen_score_diff[unseen_matches0] - 1e-4
    if correct_unseen_diff.numel() == 0:
        return float("nan")
    correct_unseen_diff = torch.sort(correct_unseen_diff).values
    magic_binsize = 20
    bias_skip = max(correct_unseen_diff.numel() // magic_binsize, 1)
    bias_list = correct_unseen_diff[::bias_skip]

    seen_acc: list[float] = []
    unseen_acc: list[float] = []
    unseen_class_mask = ~class_seen
    for bias in bias_list.tolist():
        s = logits.clone()
        s[:, unseen_class_mask] += float(bias)
        pred = s.argmax(dim=1)
        seen_acc.append(
            float(pred[seen_sample_mask].eq(labels[seen_sample_mask]).float().mean().item())
        )
        unseen_acc.append(
            float(
                pred[unseen_sample_mask].eq(labels[unseen_sample_mask]).float().mean().item()
            )
        )

    seen_acc.append(
        float(pred0[seen_sample_mask].eq(labels[seen_sample_mask]).float().mean().item())
    )
    unseen_acc.append(
        float(
            pred0[unseen_sample_mask]
            .eq(labels[unseen_sample_mask])
            .float()
            .mean()
            .item()
        )
    )
    try:
        import numpy as np

        x = np.asarray(unseen_acc)
        y = np.asarray(seen_acc)
        if hasattr(np, "trapezoid"):
            return float(np.trapezoid(y, x))
        return float(np.trapz(y, x))
    except (ValueError, TypeError):
        return float("nan")


def make_fixed_bank_forward(
    model: TextConditionedIJepa,
    device: torch.device,
    candidate_text_bank: torch.Tensor,
    *,
    use_amp: bool,
    use_bidirectional_infonce: bool = True,
    allowed_class_indices: list[int] | None = None,
) -> Callable[[dict[str, Any]], tuple[torch.Tensor, torch.Tensor | None]]:
    """Forward for eval when class text embeddings are precomputed in ``candidate_text_bank``."""

    c_full = int(candidate_text_bank.size(0))
    allowed_t: torch.Tensor | None
    bank_sub: torch.Tensor | None
    g2l: torch.Tensor | None
    if allowed_class_indices is None:
        allowed_t = None
        bank_sub = None
        g2l = None
    else:
        allowed_unique = sorted({int(i) for i in allowed_class_indices})
        if any(i < 0 or i >= c_full for i in allowed_unique):
            raise ValueError(
                f"allowed_class_indices out of range [0,{c_full}); got min/max "
                f"{min(allowed_unique)}/{max(allowed_unique)}"
            )
        allowed_t = torch.tensor(allowed_unique, dtype=torch.long, device=device)
        bank_sub = candidate_text_bank.index_select(0, allowed_t)
        g2l = torch.full((c_full,), -1, dtype=torch.long, device=device)
        g2l[allowed_t] = torch.arange(allowed_t.numel(), device=device, dtype=torch.long)

    def forward_batch(
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pv = batch["pixel_values"].to(device)
        y = batch["labels"].to(device)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda" and use_amp),
            dtype=torch.float16,
        ):
            z = model.encode_image(pv)
            if allowed_t is None:
                logits = model.score_candidates(z, candidate_text_bank)
                pos_text = candidate_text_bank.index_select(0, y)
            else:
                assert bank_sub is not None and g2l is not None
                y_loc = g2l[y]
                if not (y_loc >= 0).all():
                    raise RuntimeError(
                        "Label id not in allowed_class_indices for this eval split "
                        "(batch contains a class outside train∪val or train∪test)."
                    )
                logits_sub = model.score_candidates(z, bank_sub)
                finfo_min = torch.finfo(logits_sub.dtype).min
                mask_val = finfo_min / 2 if finfo_min > -3.4e38 else -3.4e38
                logits = torch.full(
                    (z.size(0), c_full),
                    mask_val,
                    device=z.device,
                    dtype=logits_sub.dtype,
                )
                logits[:, allowed_t] = logits_sub
                pos_text = bank_sub.index_select(0, y_loc)
            if not use_bidirectional_infonce:
                raise RuntimeError("Only bidirectional_infonce loss is supported.")
            pair_scores = model.score_candidates(z, pos_text)
            loss = clip_contrastive_loss(pair_scores)
        return logits, loss

    return forward_batch


@torch.inference_mode()
def eval_clip_style_classification(
    loader: DataLoader,
    device: torch.device,
    *,
    num_classes: int,
    use_amp: bool,
    forward_batch: Callable[[dict[str, Any]], tuple[torch.Tensor, torch.Tensor | None]],
    modules_to_eval: Iterable[nn.Module] | None = None,
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Mean loss (if ``forward_batch`` returns a non-None per-batch loss), top-1 / top-5,
    CSP AUC, and seen vs unseen top-k when ``pair_seen_in_train`` exists in batches.
    """
    if modules_to_eval is not None:
        for m in modules_to_eval:
            m.eval()

    n_ok1 = 0
    n_ok5 = 0
    n = 0
    seen_ok1 = 0
    seen_ok5 = 0
    seen_n = 0
    unseen_ok1 = 0
    unseen_ok5 = 0
    unseen_n = 0
    total_loss = 0.0
    loss_n = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_seen_flags: list[torch.Tensor] = []
    k_top = min(5, num_classes)

    for batch in loader:
        y = batch["labels"]
        bsz = int(y.size(0))
        logits, loss = forward_batch(batch)

        if loss is not None:
            total_loss += float(loss.item()) * bsz
            loss_n += bsz

        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())
        if "pair_seen_in_train" in batch:
            all_seen_flags.append(batch["pair_seen_in_train"].detach().cpu().bool())

        y_dev = y.to(logits.device)
        pred1 = logits.argmax(dim=-1)
        n_ok1 += int((pred1 == y_dev).sum().item())
        if k_top > 0:
            topk_idx = logits.topk(k_top, dim=-1).indices
            hit5 = (topk_idx == y_dev.unsqueeze(-1)).any(dim=-1)
            n_ok5 += int(hit5.sum().item())
        else:
            hit5 = torch.zeros_like(y_dev, dtype=torch.bool)

        if "pair_seen_in_train" in batch:
            seen_mask = batch["pair_seen_in_train"].to(device=logits.device, dtype=torch.bool)
            unseen_mask = ~seen_mask
            seen_n += int(seen_mask.sum().item())
            unseen_n += int(unseen_mask.sum().item())
            if seen_mask.any():
                seen_ok1 += int((pred1[seen_mask] == y_dev[seen_mask]).sum().item())
                if k_top > 0:
                    seen_ok5 += int(hit5[seen_mask].sum().item())
            if unseen_mask.any():
                unseen_ok1 += int((pred1[unseen_mask] == y_dev[unseen_mask]).sum().item())
                if k_top > 0:
                    unseen_ok5 += int(hit5[unseen_mask].sum().item())

        n += bsz

    if n == 0:
        return (
            0.0,
            0.0,
            0.0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    auc_csp_style = compute_auc_csp_style(all_logits, all_labels, all_seen_flags, num_classes)
    seen_top1 = float(seen_ok1 / seen_n) if seen_n > 0 else float("nan")
    seen_top5 = float(seen_ok5 / seen_n) if seen_n > 0 else float("nan")
    unseen_top1 = float(unseen_ok1 / unseen_n) if unseen_n > 0 else float("nan")
    unseen_top5 = float(unseen_ok5 / unseen_n) if unseen_n > 0 else float("nan")
    mean_loss = total_loss / loss_n if loss_n > 0 else float("nan")

    return (
        mean_loss,
        n_ok1 / n,
        n_ok5 / n,
        auc_csp_style,
        seen_top1,
        seen_top5,
        unseen_top1,
        unseen_top5,
    )
