"""
CLIP evaluation pipeline: load common vision benchmarks via Hugging Face Datasets
and run zero-shot image classification with a CLIP model from Transformers.
All inline comments are in English.
"""

from __future__ import annotations

import project_env

project_env.load_project_env()

import argparse

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from vision_data import (
    DATASET_CONFIG,
    build_text_prompts,
    get_image_column,
    limit_dataset_size,
    load_vision_dataset,
    set_seed,
)


def load_clip(
    model_id: str,
    device: torch.device,
) -> tuple[CLIPModel, CLIPProcessor]:
    """Load pretrained CLIP weights and matching preprocessor."""
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor


@torch.inference_mode()
def zero_shot_classify(
    model: CLIPModel,
    processor: CLIPProcessor,
    dataset: Dataset,
    class_names: list[str],
    label_key: str,
    image_column: str,
    text_template: str,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, float]:
    """
    Run zero-shot classification: similarity between image embeddings and
    one text prompt per class.

    Returns:
        metrics: 'accuracy' (top-1) and 'accuracy_top5' (fraction where true label
        is in the top-5 classes by score; for fewer than 5 classes, k is min(5, C))
    """
    prompts = build_text_prompts(class_names, text_template)
    text_inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    # In recent Transformers, CLIP feature helpers return BaseModelOutputWithPooling; use pooler_output.
    text_out = model.get_text_features(**text_inputs)
    text_features = text_out.pooler_output
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    n_classes = len(class_names)
    k_at5 = min(5, n_classes)

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    n = len(dataset)
    for start in tqdm(
        range(0, n, batch_size), desc="Zero-shot CLIP", unit="batch", leave=False
    ):
        batch = dataset.select(range(start, min(start + batch_size, n)))
        images = [row[image_column] for row in batch]
        labels = [int(row[label_key]) for row in batch]
        vis = processor(images=images, return_tensors="pt", padding=True)
        pixel_values = vis["pixel_values"].to(device)
        image_out = model.get_image_features(pixel_values=pixel_values)
        image_features = image_out.pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # Cosine similarity as logits (CLIP scaling is already in the model output space)
        logits = image_features @ text_features.T
        preds = logits.argmax(dim=-1).cpu().numpy().tolist()
        # Top-k class indices (descending score); k capped when C < 5
        topk_idx = logits.topk(k_at5, dim=-1).indices.cpu().numpy()
        for i, y in enumerate(labels):
            if preds[i] == y:
                correct_top1 += 1
            if y in set(topk_idx[i].tolist()):
                correct_top5 += 1
            total += 1

    denom = max(total, 1)
    return {
        "accuracy": correct_top1 / denom,
        "accuracy_top5": correct_top5 / denom,
    }


def run_clip_test(
    model_id: str = "openai/clip-vit-base-patch32",
    dataset_key: str = "cifar100",
    split: str = "test",
    max_samples: int | None = 500,
    batch_size: int = 32,
    seed: int = 0,
    text_template: str = "a photo of a {c}.",
) -> None:
    """
    End-to-end entry: load model + dataset, evaluate zero-shot accuracy, print result.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {model_id} | Dataset: {dataset_key} ({split})")

    ds, class_names, label_key = load_vision_dataset(dataset_key, split=split)
    ds = limit_dataset_size(ds, max_samples)
    col = get_image_column(ds)

    model, processor = load_clip(model_id, device)
    metrics = zero_shot_classify(
        model,
        processor,
        ds,
        class_names,
        label_key,
        col,
        text_template,
        device,
        batch_size=batch_size,
    )
    print(
        f"Zero-shot top-1 accuracy: {metrics['accuracy'] * 100:.2f}% | "
        f"top-5 accuracy: {metrics['accuracy_top5'] * 100:.2f}% "
        f"(n={len(ds)} images, k={min(5, len(class_names))})"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load CLIP + a HuggingFace vision dataset and run zero-shot testing."
    )
    p.add_argument(
        "--model",
        default="openai/clip-vit-base-patch32",
        help="Transformers model id (CLIP).",
    )
    p.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIG.keys()),
        default="cifar100",
        help="Benchmark to load (same families often used in CLIP evaluations).",
    )
    p.add_argument(
        "--split",
        default="test",
        help="Dataset split, e.g. 'test' for CIFAR-10/100.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Limit number of images (None = full split).",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--text-template",
        default="a photo of a {c}.",
        help="Use {c} for class name placeholder.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_clip_test(
        model_id=args.model,
        dataset_key=args.dataset,
        split=args.split,
        max_samples=None if args.max_samples < 0 else args.max_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        text_template=args.text_template,
    )
