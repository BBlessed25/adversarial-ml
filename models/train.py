"""Training utilities for the baseline and adversarially trained models."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    attack_fn: Optional[Callable] = None,
    adv_mix: float = 0.5,
) -> float:
    """
    Standard or adversarial-training epoch.

    If attack_fn is provided, a fraction `adv_mix` of each batch is replaced
    with adversarial examples (PGD/FGSM) before the loss is computed.
    """
    model.train()
    total_loss = 0.0
    n = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if attack_fn is not None and adv_mix > 0:
            model.eval()
            adv_images = attack_fn(model, images, labels)
            model.train()
            # Mix clean and adversarial samples in the same batch
            mix_mask = (torch.rand(images.size(0), device=device) < adv_mix).float()
            mix_mask = mix_mask.view(-1, 1, 1, 1)
            images = mix_mask * adv_images + (1.0 - mix_mask) * images

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    attack_fn: Optional[Callable] = None,
    adv_mix: float = 0.5,
    checkpoint_path: Optional[str | Path] = None,
) -> dict:
    """Train model; optionally with adversarial training. Saves best clean-acc checkpoint."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "test_acc": []}
    best_acc = -1.0

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            attack_fn=attack_fn,
            adv_mix=adv_mix,
        )
        acc = evaluate_accuracy(model, test_loader, device)
        scheduler.step()

        history["train_loss"].append(loss)
        history["test_acc"].append(acc)
        print(f"Epoch {epoch:02d}/{epochs}  loss={loss:.4f}  test_acc={acc:.4f}")

        if checkpoint_path is not None and acc > best_acc:
            best_acc = acc
            path = Path(checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "test_acc": acc,
                },
                path,
            )

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best checkpoint (acc={ckpt['test_acc']:.4f}) from {checkpoint_path}")

    return history


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    device: torch.device,
) -> nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model
