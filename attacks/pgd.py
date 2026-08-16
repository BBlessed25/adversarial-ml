"""Projected Gradient Descent (PGD) — iterative white-box L∞ attack."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 10,
    random_start: bool = True,
    targeted: bool = False,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> torch.Tensor:
    """
    PGD / Madry attack under an L∞ ball of radius epsilon.

    x^{t+1} = Proj_{ε}( x^t + α * sign(∇_x L) )
    """
    was_training = model.training
    model.eval()

    original = images.detach()
    adv = original.clone()

    if random_start:
        adv = adv + torch.empty_like(adv).uniform_(-epsilon, epsilon)
        if clip_min is not None or clip_max is not None:
            cmin = clip_min if clip_min is not None else float("-inf")
            cmax = clip_max if clip_max is not None else float("inf")
            adv = torch.clamp(adv, cmin, cmax)

    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        logits = model(adv)
        loss = F.cross_entropy(logits, labels)
        model.zero_grad(set_to_none=True)
        loss.backward()

        grad_sign = adv.grad.data.sign()
        if targeted:
            adv = adv - alpha * grad_sign
        else:
            adv = adv + alpha * grad_sign

        # Project onto L∞ ball around the original image
        delta = torch.clamp(adv - original, min=-epsilon, max=epsilon)
        adv = original + delta

        if clip_min is not None or clip_max is not None:
            cmin = clip_min if clip_min is not None else float("-inf")
            cmax = clip_max if clip_max is not None else float("inf")
            adv = torch.clamp(adv, cmin, cmax)

    if was_training:
        model.train()
    return adv.detach()


def make_pgd(
    epsilon: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 10,
    random_start: bool = True,
):
    """Return a callable attack_fn(model, images, labels) -> adv_images."""

    def _attack(model, images, labels):
        return pgd_attack(
            model,
            images,
            labels,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            random_start=random_start,
        )

    return _attack
