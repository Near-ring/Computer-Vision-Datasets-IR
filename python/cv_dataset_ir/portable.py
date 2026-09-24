"""Independent reference reader: only stdlib + NumPy + MessagePack + zstandard.

It does NOT call the Rust extension. Framing scans skip image records; only requested
records are decompressed. This implementation favors clarity over optimization.
"""

from __future__ import annotations

import io
import math
import os
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO
from uuid import UUID

import msgpack
import numpy as np
import zstandard as zstd

from .model import ARRAY_DTYPES, Category, Sample, Wire


def _read(file: BinaryIO, size: int) -> bytes:
    data = file.read(size)
    if len(data) != size:
        raise ValueError("truncated packed file")
    return data


def _decode(frame: bytes, limit: int) -> Wire:
    with zstd.ZstdDecompressor(max_window_size=max(8 * 1024 * 1024, limit)).stream_reader(
        io.BytesIO(frame), read_across_frames=True
    ) as reader:
        raw = reader.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("decompressed record exceeds size limit")
    value = msgpack.unpackb(raw, raw=False, strict_map_key=True)
    if not isinstance(value, dict):
        raise ValueError("record must be a map")
    return value


def _array(value: Wire, dtype: str, copy: bool) -> Any:
    if value["dtype"] != dtype:
        raise ValueError(f"expected array dtype {dtype}")
    shape = value["shape"]
    if not isinstance(shape, list) or any(type(v) is not int or v < 0 for v in shape):
        raise ValueError("invalid array shape")
    expected = math.prod(shape) * np.dtype(dtype).itemsize
    if expected != len(value["data"]):
        raise ValueError("array byte count does not match shape")
    array = np.frombuffer(value["data"], dtype=dtype).reshape(shape)
    return array.copy() if copy else array


def decode_arrays(record: Wire, copy: bool = True) -> Wire:
    record["image"] = _array(record["image"], "|u1", copy)
    objects = record["objects"]
    for key, dtype in ARRAY_DTYPES.items():
        objects[key] = _array(objects[key], dtype, copy)
    for mask in objects["masks"]:
        if mask is not None:
            if mask["bitorder"] != "little":
                raise ValueError("unknown mask bit order")
            bits = np.frombuffer(mask["data"], dtype=np.uint8)
            mask["data"] = bits.copy() if copy else bits
    return record


def _legacy(record: Wire) -> Wire:
    """Translate the original Serde-shaped v1 records to the same public NumPy model."""
    image = record["image"]
    record["image"] = (
        np.frombuffer(image["pixels"], dtype=np.uint8)
        .reshape(image["height"], image["width"], 3)
        .copy()
    )
    record["uid"] = str(UUID(bytes=record["uid"]))
    split = record["split"]
    record["split"] = (
        {"kind": "named", "name": split["Named"]}
        if isinstance(split, dict)
        else {"kind": split.lower(), "name": None}
    )
    source = record["objects"]
    objects: Wire = {}
    for key, dtype in [("ids", np.uint64), ("class_ids", np.uint32)]:
        objects[key] = np.asarray(source[key], dtype=dtype)
    objects["boxes"] = np.asarray(
        [[r["x"], r["y"], r["width"], r["height"]] for r in source["boxes"]], dtype=np.float32
    ).reshape(-1, 4)
    types, params, polygon_objects = [], [], [0]
    for shape in source["shapes"]:
        kind, value = next(iter(shape.items()))
        if kind == "Rect":
            types.append(0)
            params.append([value[k] for k in ("x", "y", "width", "height")])
        elif kind == "Circle":
            types.append(1)
            params.append([value["center"]["x"], value["center"]["y"], value["radius"], 0])
        elif kind == "Polygons":
            types.append(2)
            params.append([0, 0, 0, 0])
        elif kind == "Point":
            types.append(3)
            params.append([value["x"], value["y"], 0, 0])
        else:
            raise ValueError("unknown v1 shape")
        polygon_objects.append(value["end"] if kind == "Polygons" else polygon_objects[-1])
    objects["shape_types"] = np.asarray(types, dtype=np.uint8)
    objects["shape_params"] = np.asarray(params, dtype=np.float32).reshape(-1, 4)
    objects["polygon_object_offsets"] = np.asarray(polygon_objects, dtype=np.uint64)
    objects["polygon_offsets"] = np.asarray(
        [0] + [r["end"] for r in source["polygons"]], dtype=np.uint64
    )
    objects["vertices"] = np.asarray(
        [[p["x"], p["y"]] for p in source["vertices"]], dtype=np.float32
    ).reshape(-1, 2)
    objects["keypoint_offsets"] = np.asarray(
        [0] + [r["end"] for r in source["keypoint_ranges"]], dtype=np.uint64
    )
    objects["keypoints"] = np.asarray(
        [[p["point"]["x"], p["point"]["y"]] for p in source["keypoints"]], dtype=np.float32
    ).reshape(-1, 2)
    objects["visibility"] = np.asarray(
        [{"Absent": 0, "Occluded": 1, "Visible": 2}[p["visibility"]] for p in source["keypoints"]],
        dtype=np.uint8,
    )
    objects["is_crowd"] = np.asarray(source["crowds"], dtype=np.uint8)
    objects["masks"] = [
        None
        if m is None
        else {
            "width": m["width"],
            "height": m["height"],
            "data": np.frombuffer(m["bits"], dtype=np.uint8).copy(),
        }
        for m in source["masks"]
    ]
    objects["metadata"] = source["metadata"]
    record["objects"] = objects
    record["provenance"] = [
        {**p, "parent_uid": str(UUID(bytes=p["parent_uid"]))} for p in record.get("provenance", [])
    ]
    return record


def validate_record(value: Wire) -> None:
    """Check lengths/ranges before exposing external arrays to ML code."""
    image = value["image"]
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape) <= 0:
        raise ValueError("image must be nonempty RGB8 HWC")
    if not value["id"] or UUID(value["uid"]).int == 0:
        raise ValueError("invalid sample identity")
    split = value["split"]
    if split["kind"] == "named":
        if not isinstance(split["name"], str):
            raise ValueError("named split needs a string name")
    elif split["kind"] not in ("train", "val", "test", "unassigned") or split["name"] is not None:
        raise ValueError("invalid split")
    o = value["objects"]
    n = len(o["ids"])
    for key in ("ids", "class_ids", "shape_types", "is_crowd"):
        if o[key].shape != (n,):
            raise ValueError("object column mismatch")
    for key in ("boxes", "shape_params"):
        if o[key].shape != (n, 4) or not np.isfinite(o[key]).all():
            raise ValueError("invalid object matrix")
    if np.any(o["boxes"][:, 2:] < 0) or np.any(o["shape_types"] > 3):
        raise ValueError("invalid boxes/shapes")
    for key in ("vertices", "keypoints"):
        if o[key].ndim != 2 or o[key].shape[1] != 2 or not np.isfinite(o[key]).all():
            raise ValueError("invalid point matrix")
    for key, rows, end in [
        ("polygon_object_offsets", n, len(o["polygon_offsets"]) - 1),
        ("polygon_offsets", len(o["polygon_offsets"]) - 1, len(o["vertices"])),
        ("keypoint_offsets", n, len(o["keypoints"])),
    ]:
        offsets = o[key]
        if offsets.shape != (rows + 1,) or rows < 0 or offsets[0] != 0 or offsets[-1] != end:
            raise ValueError("invalid ragged offsets")
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("offsets must be monotone")
    if o["visibility"].shape != (len(o["keypoints"]),) or np.any(o["visibility"] > 2):
        raise ValueError("invalid visibility")
    if np.any(o["is_crowd"] > 1) or len(o["metadata"]) != n or len(o["masks"]) != n:
        raise ValueError("invalid object columns")
    for i, kind in enumerate(o["shape_types"]):
        a, b = (int(v) for v in o["polygon_object_offsets"][i : i + 2])
        params = o["shape_params"][i]
        if kind != 2 and a != b:
            raise ValueError("non-polygon has polygon vertices")
        if kind == 0 and np.any(params[2:] < 0):
            raise ValueError("negative rectangle size")
        if kind == 1 and params[2] < 0:
            raise ValueError("negative circle radius")
        if kind == 2:
            if a == b or np.any(
                o["polygon_offsets"][a + 1 : b + 1] - o["polygon_offsets"][a:b] < 3
            ):
                raise ValueError("polygon requires at least three vertices")
    for mask in o["masks"]:
        if mask is None:
            continue
        count = mask["width"] * mask["height"]
        if (mask["height"], mask["width"]) != image.shape[:2] or len(mask["data"]) != (
            count + 7
        ) // 8:
            raise ValueError("invalid mask dimensions")
        if count % 8 and int(mask["data"][-1]) >> (count % 8):
            raise ValueError("nonzero mask padding bits")


class PurePackedReader:
    """Seekable independent reader for v1/v2. Not shared across threads/processes."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        max_record_bytes: int = 512 * 1024**2,
        copy_arrays: bool = True,
    ) -> None:
        if max_record_bytes <= 0:
            raise ValueError("max_record_bytes must be positive")
        self.path = Path(path)
        self._file = self.path.open("rb")
        self._limit = max_record_bytes
        self._copy = copy_arrays
        self._index: dict[UUID, tuple[int, int]] = {}
        try:
            magic = _read(self._file, 8)
            if magic not in (b"CVDSIR01", b"CVDSIR02"):
                raise ValueError("unknown packed version")
            self.version = int(magic[-1:])
            header = _decode(self._frame(), self._limit)
            if self.version == 2 and (
                header.get("version") != 2 or header.get("schema") != "cv-dataset-ir"
            ):
                raise ValueError("schema mismatch")
            self.categories = [Category(**c) for c in header["categories"]]
            names = [c.name for c in self.categories]
            if any(not name for name in names) or len(set(names)) != len(names):
                raise ValueError("invalid category names")
            for category in self.categories:
                if len(set(category.keypoints)) != len(category.keypoints):
                    raise ValueError("duplicate keypoint names")
                if any(
                    len(edge) != 2 or any(i < 0 or i >= len(category.keypoints) for i in edge)
                    for edge in category.skeleton
                ):
                    raise ValueError("invalid skeleton index")
            self.metadata = header["metadata"]
            end = self.path.stat().st_size
            count = header["samples"]
            if type(count) is not int or count < 0 or count > (end - self._file.tell()) // 24:
                raise ValueError("invalid sample count")
            for _ in range(count):
                uid = UUID(bytes=_read(self._file, 16))
                length = self._length()
                offset = self._file.tell()
                if uid.int == 0 or uid in self._index or offset + length > end:
                    raise ValueError("duplicate UUID or truncated sample")
                self._index[uid] = offset, length
                self._file.seek(length, 1)
            if self._file.tell() != end:
                raise ValueError("trailing packed data")
            self.sample_ids = tuple(self._index)
        except BaseException:
            self.close()
            raise

    def _length(self) -> int:
        length = struct.unpack("<Q", _read(self._file, 8))[0]
        if length > self._limit:
            raise ValueError("compressed record exceeds size limit")
        return length

    def _frame(self) -> bytes:
        return _read(self._file, self._length())

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, key: int | str | UUID) -> Sample:
        uid = self.sample_ids[key] if isinstance(key, int) else UUID(str(key))
        offset, length = self._index[uid]
        self._file.seek(offset)
        raw = _decode(_read(self._file, length), self._limit)
        value = decode_arrays(raw, self._copy) if self.version == 2 else _legacy(raw)
        if UUID(value["uid"]) != uid:
            raise ValueError("frame UUID mismatch")
        validate_record(value)
        objects = value["objects"]
        if np.any(objects["class_ids"] >= len(self.categories)):
            raise ValueError("unknown class index")
        for i, category_id in enumerate(objects["class_ids"]):
            count = int(objects["keypoint_offsets"][i + 1]) - int(objects["keypoint_offsets"][i])
            if count not in (0, len(self.categories[int(category_id)].keypoints)):
                raise ValueError("keypoint count disagrees with category")
        return Sample.from_mapping(value)

    def __iter__(self) -> Iterator[Sample]:
        for i in range(len(self)):
            yield self[i]

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> PurePackedReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
