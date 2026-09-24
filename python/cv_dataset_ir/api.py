"""NumPy API backed by the Rust crate. Native imports happen only on first use."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from os import PathLike
from types import TracebackType
from typing import cast
from uuid import UUID

import msgpack
import numpy as np
from numpy.typing import NDArray

from .model import Category, Mask, Metadata, Sample

Path = str | PathLike[str]


def _record(sample: Sample) -> bytes:
    return cast(bytes, msgpack.packb(sample.to_wire(), use_bin_type=True))


class PackedReader:
    """Indexed reads by position or UUID. Each decoded sample owns its arrays.

    File reads/decompression release the GIL. Concurrent reads on one reader are
    serialized; use one reader per worker for parallel disk/decompression work.
    """

    def __init__(self, path: Path, *, max_record_bytes: int = 512 * 1024 * 1024) -> None:
        from . import _native

        self._reader = _native.PackedReader(str(path), max_record_bytes)
        header = json.loads(self._reader.header_json())
        self.version: int = header["version"]
        self.categories = [Category(**value) for value in header["categories"]]
        self.metadata: Metadata = header["metadata"]
        self.sample_ids = tuple(UUID(value) for value in header["sample_ids"])

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, key: int | str | UUID) -> Sample:
        if isinstance(key, int):
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(key)
            value = self._reader.read_index(key)
        else:
            value = self._reader.read_uid(str(key))
        return Sample.from_mapping(value)

    def __iter__(self) -> Iterator[Sample]:
        for index in range(len(self)):
            yield self[index]

    def close(self) -> None:
        self._reader.close()

    def __enter__(self) -> PackedReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class PackedWriter:
    """Write a new v2 file without overwriting existing data.

    The count is known up front. `finish()` (or successful context exit) checks it
    and flushes. An exception leaves an incomplete file; publish only after finish.
    Inputs are copied into Rust-owned buffers, so later array edits are harmless.
    """

    def __init__(
        self,
        path: Path,
        categories: Sequence[Category],
        samples: int,
        *,
        metadata: Metadata | None = None,
        compression_level: int = 3,
        max_record_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        from . import _native

        self._writer = _native.PackedWriter(
            str(path),
            json.dumps([asdict(c) for c in categories]),
            json.dumps(metadata or {}),
            samples,
            compression_level,
            max_record_bytes,
        )

    def push(self, sample: Sample) -> None:
        self._writer.push(_record(sample))

    def finish(self) -> None:
        self._writer.finish()

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> PackedWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.finish()
        else:
            self.close()


def write(
    path: Path,
    samples: Sequence[Sample],
    categories: Sequence[Category],
    *,
    metadata: Metadata | None = None,
) -> None:
    with PackedWriter(path, categories, len(samples), metadata=metadata) as writer:
        for sample in samples:
            writer.push(sample)


def mask_crops(
    sample: Sample,
    *,
    classes: list[int] | None = None,
    padding: int = 0,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> list[Sample]:
    """One sample per nonempty masked object, preserving metadata and every pose slot."""
    from . import _native

    return [
        Sample.from_mapping(value)
        for value in _native.mask_crops(_record(sample), classes, padding, fill)
    ]


def mask_crop(
    sample: Sample,
    object_index: int,
    *,
    mask: Mask | NDArray[np.uint8] | NDArray[np.bool_] | None = None,
    padding: int = 0,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Sample:
    """Crop one object using its stored mask or an explicit full-image mask."""
    from . import _native

    dense = mask.numpy() if isinstance(mask, Mask) else mask
    if dense is not None and dense.shape != sample.image.shape[:2]:
        raise ValueError("mask dimensions do not match source image")
    data = None if dense is None else np.ascontiguousarray(dense != 0, dtype=np.uint8).tobytes()
    values = _native.mask_crops(_record(sample), None, padding, fill, object_index, data)
    return Sample.from_mapping(values[0])


def _transform(sample: Sample, operation: str, parameters: Metadata) -> Sample:
    from . import _native

    return Sample.from_mapping(
        _native.transform(_record(sample), operation, json.dumps(parameters))
    )


def resize(sample: Sample, width: int, height: int) -> Sample:
    return _transform(sample, "resize", {"width": width, "height": height})


def crop(sample: Sample, x: int, y: int, width: int, height: int) -> Sample:
    return _transform(sample, "crop", {"x": x, "y": y, "width": width, "height": height})


def rotate(sample: Sample, degrees: float, *, fill: tuple[int, int, int] = (0, 0, 0)) -> Sample:
    return _transform(sample, "rotate", {"degrees": degrees, "fill": list(fill)})


def zoom(sample: Sample, factor: float, *, fill: tuple[int, int, int] = (0, 0, 0)) -> Sample:
    return _transform(sample, "zoom", {"factor": factor, "fill": list(fill)})


def flip_horizontal(sample: Sample, *, keypoint_permutation: list[int] | None = None) -> Sample:
    return _transform(
        sample,
        "flip",
        {
            "keypoint_permutation": None
            if keypoint_permutation is None
            else [int(v) for v in keypoint_permutation]
        },
    )
