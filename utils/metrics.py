"""Evaluation metrics for clean and adversarial settings."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    defense_fn=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) over a dataloader, optionally applying a defense."""
    model.eval()
    ys: List[int] = []
    preds: List[int] = []

    for images, labels in loader:
        images = images.to(device)
        if defense_fn is not None:
            images = defense_fn(images)
        logits = model(images)
        batch_preds = logits.argmax(dim=1).cpu().numpy()
        ys.extend(labels.numpy().tolist())
        preds.extend(batch_preds.tolist())

    return np.asarray(ys), np.asarray(preds)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> Dict[str, Any]:
    """Accuracy, precision, recall, F1, and confusion matrix."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }


def evaluate_clean(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    defense_fn=None,
) -> Dict[str, Any]:
    """Clean (non-adversarial) evaluation."""
    y_true, y_pred = predict(model, loader, device, defense_fn=defense_fn)
    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["robust_accuracy"] = metrics["accuracy"]
    metrics["asr"] = 0.0
    return metrics


def evaluate_attack(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    attack_fn,
    defense_fn=None,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate under an attack.

    - Clean accuracy: original labels vs clean predictions (for reference).
    - Robust accuracy: fraction of adversarial examples still correctly classified.
    - ASR: among examples that were correctly classified clean, fraction flipped.
    - Perturbation: mean L∞ and L2 over adversarial deltas (in normalized space).
    """
    model.eval()
    y_true_all: List[int] = []
    y_clean_all: List[int] = []
    y_adv_all: List[int] = []
    linf_vals: List[float] = []
    l2_vals: List[float] = []

    n_correct_clean = 0
    n_flipped = 0

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Attack eval", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            clean_logits = model(images)
            clean_preds = clean_logits.argmax(dim=1)

        adv_images = attack_fn(model, images, labels)

        eval_images = adv_images
        if defense_fn is not None:
            with torch.no_grad():
                eval_images = defense_fn(adv_images)

        with torch.no_grad():
            adv_logits = model(eval_images)
            adv_preds = adv_logits.argmax(dim=1)

        delta = (adv_images - images).detach()
        flat = delta.view(delta.size(0), -1)
        linf_vals.extend(flat.abs().max(dim=1).values.cpu().tolist())
        l2_vals.extend(flat.norm(p=2, dim=1).cpu().tolist())

        correctly_classified = clean_preds == labels
        flipped = correctly_classified & (adv_preds != labels)
        n_correct_clean += int(correctly_classified.sum().item())
        n_flipped += int(flipped.sum().item())

        y_true_all.extend(labels.cpu().numpy().tolist())
        y_clean_all.extend(clean_preds.cpu().numpy().tolist())
        y_adv_all.extend(adv_preds.cpu().numpy().tolist())

    y_true = np.asarray(y_true_all)
    y_adv = np.asarray(y_adv_all)
    cls = compute_classification_metrics(y_true, y_adv)

    asr = (n_flipped / n_correct_clean) if n_correct_clean > 0 else 0.0
    robust_acc = float((y_adv == y_true).mean())

    cls["robust_accuracy"] = robust_acc
    cls["asr"] = float(asr)
    cls["perturbation_linf_mean"] = float(np.mean(linf_vals)) if linf_vals else 0.0
    cls["perturbation_l2_mean"] = float(np.mean(l2_vals)) if l2_vals else 0.0
    cls["n_correct_clean"] = n_correct_clean
    cls["n_flipped"] = n_flipped
    return cls


def empty_metrics_row() -> Dict[str, float]:
    return {
        "accuracy": float("nan"),
        "robust_accuracy": float("nan"),
        "asr": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
    }


def metrics_to_row(m: Dict[str, Any]) -> Dict[str, float]:
    return {
        "accuracy": m.get("accuracy", float("nan")),
        "robust_accuracy": m.get("robust_accuracy", float("nan")),
        "asr": m.get("asr", float("nan")),
        "precision": m.get("precision", float("nan")),
        "recall": m.get("recall", float("nan")),
        "f1": m.get("f1", float("nan")),
    }
