"""Adversarial training defense (Madry-style)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from attacks.pgd import make_pgd
from models.train import train_model


def adversarial_train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    epsilon: float = 8 / 255,
    alpha: float = 2 / 255,
    pgd_steps: int = 5,
    adv_mix: float = 0.5,
    checkpoint_path: Optional[str | Path] = None,
) -> dict:
    """
    Train with on-the-fly PGD adversaries mixed into each batch.

    Using fewer PGD steps during training keeps runtime practical for the
    course project while still improving robust accuracy.
    """
    attack_fn = make_pgd(epsilon=epsilon, alpha=alpha, steps=pgd_steps)
    return train_model(
        model,
        train_loader,
        test_loader,
        device,
        epochs=epochs,
        lr=lr,
        attack_fn=attack_fn,
        adv_mix=adv_mix,
        checkpoint_path=checkpoint_path,
    )
