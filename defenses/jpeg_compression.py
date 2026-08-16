"""JPEG compression input transformation defense."""

from __future__ import annotations

import io
from typing import Callable

import torch
from PIL import Image
from torchvision import transforms

from utils.data import CIFAR10_MEAN, CIFAR10_STD, denormalize


def _tensor_to_jpeg_tensor(img: torch.Tensor, quality: int) -> torch.Tensor:
    """
    Compress a single CHW float tensor in [0, 1] via JPEG and return [0, 1] tensor.
    """
    pil = transforms.ToPILImage()(img.cpu())
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    restored = Image.open(buf).convert("RGB")
    return transforms.ToTensor()(restored)


def jpeg_compress_batch(
    images: torch.Tensor,
    quality: int = 75,
) -> torch.Tensor:
    """
    Apply JPEG compression as a preprocessing defense.

    Pipeline: denormalize → JPEG → re-normalize (same CIFAR-10 stats).
    Operates per-image (CPU PIL) for correctness and simplicity.
    """
    device = images.device
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)

    visuals = denormalize(images)
    out = []
    for i in range(visuals.size(0)):
        compressed = _tensor_to_jpeg_tensor(visuals[i], quality=quality)
        out.append(compressed)
    batch = torch.stack(out, dim=0).to(device)
    return (batch - mean) / std


def make_jpeg_defense(quality: int = 75) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return defense_fn(images) -> defended_images."""

    def _defend(images: torch.Tensor) -> torch.Tensor:
        return jpeg_compress_batch(images, quality=quality)

    return _defend
