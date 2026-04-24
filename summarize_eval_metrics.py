#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_float(x: object) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _is_finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def _derive_config_tag(exp_tag: str) -> str:
    # Expected patterns like:
    #   clip-sim-s0-default-lr... , cross-attn-s3-g2-lr...
    # Remove fusion prefix and seed prefix, keep hyperparameter/config part.
    s = exp_tag.strip()
    if not s:
        return "unknown"
    if s.startswith("clip-sim-"):
        s = s[len("clip-sim-") :]
    elif s.startswith("cross-attn-"):
        s = s[len("cross-attn-") :]
    if s.startswith("s"):
        # remove leading s{seed}-...
        pos = s.find("-")
        if pos > 0 and s[1:pos].isdigit():
            s = s[pos + 1 :]
    return s or "unknown"


def _read_metrics(metrics_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for p in sorted(metrics_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        data["file"] = str(p)
        data["config_tag"] = _derive_config_tag(str(data.get("experiment_tag", "")))
        rows.append(data)
    return rows


def _write_raw_csv(rows: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "split",
        "fusion_type",
        "seed",
        "experiment_tag",
        "config_tag",
        "loss",
        "top1",
        "top5",
        "auc_csp_style",
        "batch_size",
        "checkpoint",
        "from_hub",
        "file",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for r in rows:
        key = (
            str(r.get("config_tag", "unknown")),
            str(r.get("fusion_type", "unknown")),
            str(r.get("split", "unknown")),
        )
        groups.setdefault(key, []).append(r)

    metrics = ["loss", "top1", "top5", "auc_csp_style"]
    out: list[dict[str, object]] = []
    for (cfg, fusion, split), items in sorted(groups.items()):
        row: dict[str, object] = {
            "config_tag": cfg,
            "fusion_type": fusion,
            "split": split,
            "n_runs": len(items),
        }
        for m in metrics:
            vals = [_safe_float(it.get(m)) for it in items]
            vals = [v for v in vals if _is_finite(v)]
            if not vals:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_std"] = float("nan")
            else:
                row[f"{m}_mean"] = mean(vals)
                row[f"{m}_std"] = stdev(vals) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def _write_summary_csv(rows: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "config_tag",
        "fusion_type",
        "split",
        "n_runs",
        "loss_mean",
        "loss_std",
        "top1_mean",
        "top1_std",
        "top5_mean",
        "top5_std",
        "auc_csp_style_mean",
        "auc_csp_style_std",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _write_summary_json(rows: list[dict[str, object]], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _plot_metric(rows: list[dict[str, object]], metric: str, plot_path: Path) -> None:
    # Plot by split, grouped bars for fusion_type per config_tag.
    splits = sorted({str(r["split"]) for r in rows})
    fusions = sorted({str(r["fusion_type"]) for r in rows})
    cfgs = sorted({str(r["config_tag"]) for r in rows})
    if not splits or not fusions or not cfgs:
        return

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 5), squeeze=False)
    axes_row = axes[0]

    for ax, split in zip(axes_row, splits, strict=False):
        x = list(range(len(cfgs)))
        width = 0.8 / max(len(fusions), 1)
        for i, fusion in enumerate(fusions):
            y: list[float] = []
            yerr: list[float] = []
            for cfg in cfgs:
                matched = [
                    r
                    for r in rows
                    if str(r["split"]) == split
                    and str(r["fusion_type"]) == fusion
                    and str(r["config_tag"]) == cfg
                ]
                if matched:
                    y.append(float(matched[0].get(f"{metric}_mean", float("nan"))))
                    yerr.append(float(matched[0].get(f"{metric}_std", 0.0)))
                else:
                    y.append(float("nan"))
                    yerr.append(0.0)
            xs = [xx - 0.4 + (i + 0.5) * width for xx in x]
            ax.bar(xs, y, width=width, yerr=yerr, label=fusion, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(cfgs, rotation=30, ha="right")
        ax.set_title(f"{metric} ({split})")
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    axes_row[0].legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize eval metrics JSON files and plot results.")
    p.add_argument("--metrics-dir", required=True, help="Directory containing eval metrics JSON files.")
    p.add_argument("--raw-csv", required=True, help="Path to write raw merged CSV.")
    p.add_argument("--summary-csv", required=True, help="Path to write aggregated summary CSV.")
    p.add_argument("--summary-json", required=True, help="Path to write aggregated summary JSON.")
    p.add_argument("--plot-dir", required=True, help="Directory to save plots.")
    args = p.parse_args()

    metrics_dir = Path(args.metrics_dir)
    rows = _read_metrics(metrics_dir)
    if not rows:
        raise SystemExit(f"No metrics JSON found in: {metrics_dir}")

    raw_csv = Path(args.raw_csv)
    summary_csv = Path(args.summary_csv)
    summary_json = Path(args.summary_json)
    plot_dir = Path(args.plot_dir)

    _write_raw_csv(rows, raw_csv)
    agg = _aggregate(rows)
    _write_summary_csv(agg, summary_csv)
    _write_summary_json(agg, summary_json)

    _plot_metric(agg, "top1", plot_dir / "top1_summary.png")
    _plot_metric(agg, "top5", plot_dir / "top5_summary.png")
    _plot_metric(agg, "auc_csp_style", plot_dir / "auc_csp_style_summary.png")
    _plot_metric(agg, "loss", plot_dir / "loss_summary.png")

    print(f"Wrote raw CSV: {raw_csv}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote summary JSON: {summary_json}")
    print(f"Wrote plots to: {plot_dir}")


if __name__ == "__main__":
    main()
