"""Plotting helpers for adversarial examples and confusion matrices."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.data import CLASS_NAMES, denormalize


def save_confusion_matrix(
    cm: List[List[int]] | np.ndarray,
    path: str | Path,
    class_names: Sequence[str] = CLASS_NAMES,
    title: str = "Confusion Matrix",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(cm)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = matrix.max() / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                format(matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=7,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_adversarial_grid(
    clean: torch.Tensor,
    adv: torch.Tensor,
    labels: torch.Tensor,
    clean_preds: torch.Tensor,
    adv_preds: torch.Tensor,
    path: str | Path,
    n: int = 8,
    class_names: Sequence[str] = CLASS_NAMES,
    title: str = "Clean vs Adversarial",
) -> Path:
    """Save a grid of clean | perturbation | adversarial triples."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = min(n, clean.size(0))
    clean_vis = denormalize(clean[:n].detach().cpu())
    adv_vis = denormalize(adv[:n].detach().cpu())
    # Amplify perturbation for visibility
    delta = (adv_vis - clean_vis).abs()
    delta = delta / (delta.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)

    fig, axes = plt.subplots(n, 3, figsize=(7, 2.2 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = ["Clean", "|Perturbation| (scaled)", "Adversarial"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10)

    for i in range(n):
        true_y = class_names[int(labels[i])]
        c_pred = class_names[int(clean_preds[i])]
        a_pred = class_names[int(adv_preds[i])]

        axes[i, 0].imshow(clean_vis[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_ylabel(f"true:{true_y}\npred:{c_pred}", fontsize=8)
        axes[i, 1].imshow(delta[i].permute(1, 2, 0).numpy())
        axes[i, 2].imshow(adv_vis[i].permute(1, 2, 0).numpy())
        axes[i, 2].set_xlabel(f"pred:{a_pred}", fontsize=8)

        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_results_table_figure(
    rows: dict,
    path: str | Path,
    title: str = "Required Results Table",
) -> Path:
    """Render the course-required metrics table as an image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["Accuracy", "Robust Accuracy", "ASR", "Precision", "Recall", "F1-score"]
    keys = ["accuracy", "robust_accuracy", "asr", "precision", "recall", "f1"]
    columns = list(rows.keys())

    cell_text = []
    for metric, key in zip(metrics, keys):
        row = [metric]
        for col in columns:
            val = rows[col].get(key, float("nan"))
            row.append(f"{val:.3f}" if val == val else "—")
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=["Metric"] + columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    ax.set_title(title, pad=16)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
