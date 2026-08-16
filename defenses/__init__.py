"""Defense implementations."""

from defenses.adversarial_training import adversarial_train
from defenses.jpeg_compression import jpeg_compress_batch, make_jpeg_defense

__all__ = [
    "adversarial_train",
    "jpeg_compress_batch",
    "make_jpeg_defense",
]
