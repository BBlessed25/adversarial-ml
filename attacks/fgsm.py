"""Fast Gradient Sign Method (FGSM) — white-box L∞ attack."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 8 / 255,
    targeted: bool = False,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> torch.Tensor:
    """
    FGSM: x_adv = x + ε * sign(∇_x L(θ, x, y))  (untargeted).

    For normalized CIFAR-10 inputs, epsilon is in normalized space unless
    you pass clip bounds matching your preprocessing.
    """
    was_training = model.training
    model.eval()

    images = images.clone().detach().requires_grad_(True)
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    model.zero_grad(set_to_none=True)
    loss.backward()

    grad_sign = images.grad.data.sign()
    if targeted:
        adv = images - epsilon * grad_sign
    else:
        adv = images + epsilon * grad_sign

    if clip_min is not None or clip_max is not None:
        cmin = clip_min if clip_min is not None else float("-inf")
        cmax = clip_max if clip_max is not None else float("inf")
        adv = torch.clamp(adv, cmin, cmax)

    if was_training:
        model.train()
    return adv.detach()


def make_fgsm(epsilon: float = 8 / 255):
    """Return a callable attack_fn(model, images, labels) -> adv_images."""

    def _attack(model, images, labels):
        return fgsm_attack(model, images, labels, epsilon=epsilon)

    return _attack
