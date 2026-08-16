"""CIFAR-10 data loading and preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_transforms(train: bool = True) -> transforms.Compose:
    """Return train or eval transforms (normalize to CIFAR-10 stats)."""
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    num_workers: int = 2,
    subset_size: int | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Download CIFAR-10 (if needed) and return train/test loaders.

    Args:
        data_dir: Root directory for torchvision datasets.
        batch_size: Mini-batch size.
        num_workers: DataLoader workers.
        subset_size: If set, use only the first N samples (quick demos).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_set = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )
    test_set = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=get_transforms(train=False),
    )

    if subset_size is not None:
        n_train = min(subset_size, len(train_set))
        n_test = min(max(subset_size // 5, 200), len(test_set))
        train_set = Subset(train_set, list(range(n_train)))
        test_set = Subset(test_set, list(range(n_test)))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Convert normalized tensors back to [0, 1] RGB for visualization."""
    mean = torch.tensor(CIFAR10_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)
