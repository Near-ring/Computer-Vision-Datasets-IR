"""Portable CV dataset IR: Rust transforms, NumPy arrays, optional PyTorch."""

from .api import (
    PackedReader,
    PackedWriter,
    crop,
    flip_horizontal,
    mask_crop,
    mask_crops,
    resize,
    rotate,
    write,
    zoom,
)
from .model import Annotations, Category, Mask, Metadata, Provenance, Sample, Split
from .portable import PurePackedReader

__all__ = [
    "Annotations",
    "Category",
    "Mask",
    "Metadata",
    "PackedReader",
    "PackedWriter",
    "Provenance",
    "PurePackedReader",
    "Sample",
    "Split",
    "crop",
    "flip_horizontal",
    "mask_crop",
    "mask_crops",
    "resize",
    "rotate",
    "write",
    "zoom",
]
