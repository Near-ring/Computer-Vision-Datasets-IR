"""Typed NumPy-facing records; no Rust or PyTorch import is required here."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from torch import Tensor

Json: TypeAlias = None | bool | int | float | str | list["Json"] | dict[str, "Json"]
Metadata: TypeAlias = dict[str, Json]
# Only the dynamic MessagePack/PyO3 decoding boundary is untyped.
Wire: TypeAlias = dict[str, Any]
U8: TypeAlias = NDArray[np.uint8]
U32: TypeAlias = NDArray[np.uint32]
U64: TypeAlias = NDArray[np.uint64]
F32: TypeAlias = NDArray[np.float32]


@dataclass
class Category:
    name: str
    keypoints: list[str] = field(default_factory=list)
    skeleton: list[list[int]] = field(default_factory=list)
    metadata: Metadata = field(default_factory=dict)


@dataclass
class Split:
    kind: str = "unassigned"
    name: str | None = None


@dataclass
class Mask:
    width: int
    height: int
    bits: U8

    @classmethod
    def from_numpy(cls, mask: NDArray[np.uint8] | NDArray[np.bool_]) -> Mask:
        value = np.asarray(mask)
        if value.ndim != 2 or min(value.shape) <= 0 or value.dtype.kind not in "bu":
            raise ValueError("mask must be a nonempty 2D boolean/unsigned integer array")
        return cls(value.shape[1], value.shape[0], np.packbits(value != 0, bitorder="little"))

    def numpy(self) -> NDArray[np.bool_]:
        """Decode a writable boolean H×W array; holes and islands are exact."""
        return (
            np.unpackbits(self.bits, count=self.width * self.height, bitorder="little")
            .reshape(self.height, self.width)
            .view(np.bool_)
        )

    def to_wire(self) -> Wire:
        return {
            "width": self.width,
            "height": self.height,
            "bitorder": "little",
            "data": self.bits.tobytes(),
        }


@dataclass
class Provenance:
    parent_uid: UUID
    operation: str
    parameters: Metadata


@dataclass
class Annotations:
    ids: U64
    class_ids: U32
    boxes: F32
    shape_types: U8
    shape_params: F32
    polygon_object_offsets: U64
    polygon_offsets: U64
    vertices: F32
    keypoint_offsets: U64
    keypoints: F32
    visibility: U8
    is_crowd: U8
    masks: list[Mask | None]
    metadata: list[Metadata]

    def __len__(self) -> int:
        return len(self.ids)

    def points_for(self, index: int) -> F32:
        """Borrow the object's K×2 pose coordinate view (visibility stays separate)."""
        index = self._index(index)
        start, end = self.keypoint_offsets[index : index + 2]
        return self.keypoints[int(start) : int(end)]

    def polygons_for(self, index: int) -> list[F32]:
        index = self._index(index)
        start, end = self.polygon_object_offsets[index : index + 2]
        return [
            self.vertices[int(self.polygon_offsets[i]) : int(self.polygon_offsets[i + 1])]
            for i in range(int(start), int(end))
        ]

    def _index(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return index

    @classmethod
    def from_mapping(cls, value: Wire) -> Annotations:
        fields = value.copy()
        fields["masks"] = [
            None if m is None else Mask(m["width"], m["height"], m["data"]) for m in value["masks"]
        ]
        return cls(**fields)

    def to_wire(self) -> Wire:
        result: Wire = {name: array_to_wire(getattr(self, name)) for name in ARRAY_DTYPES}
        result["masks"] = [None if m is None else m.to_wire() for m in self.masks]
        result["metadata"] = self.metadata
        return result


ARRAY_DTYPES = {
    "ids": "<u8",
    "class_ids": "<u4",
    "boxes": "<f4",
    "shape_types": "|u1",
    "shape_params": "<f4",
    "polygon_object_offsets": "<u8",
    "polygon_offsets": "<u8",
    "vertices": "<f4",
    "keypoint_offsets": "<u8",
    "keypoints": "<f4",
    "visibility": "|u1",
    "is_crowd": "|u1",
}


def array_to_wire(array: NDArray[Any]) -> Wire:
    # Explicit little-endian encoding, including on a big-endian Python host.
    value = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    return {"dtype": value.dtype.str, "shape": list(value.shape), "data": value.tobytes()}


@dataclass
class Sample:
    uid: UUID
    id: str
    split: Split
    image: U8
    objects: Annotations
    metadata: Metadata = field(default_factory=dict)
    provenance: list[Provenance] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, value: Wire) -> Sample:
        return cls(
            UUID(value["uid"]),
            value["id"],
            Split(**value["split"]),
            value["image"],
            Annotations.from_mapping(value["objects"]),
            value["metadata"],
            [
                Provenance(UUID(p["parent_uid"]), p["operation"], p["parameters"])
                for p in value["provenance"]
            ],
        )

    def to_wire(self) -> Wire:
        return {
            "uid": str(self.uid),
            "id": self.id,
            "split": {"kind": self.split.kind, "name": self.split.name},
            "image": array_to_wire(self.image),
            "objects": self.objects.to_wire(),
            "metadata": self.metadata,
            "provenance": [
                {
                    "parent_uid": str(p.parent_uid),
                    "operation": p.operation,
                    "parameters": p.parameters,
                }
                for p in self.provenance
            ],
        }

    def to_torch(self) -> tuple[Tensor, dict[str, object]]:
        """CHW uint8 tensor + targets. Image shares NumPy storage when it is writable.

        Boxes are xyxy; pose is a ragged list of K×3 tensors with visibility in column 2.
        No scaling/device transfer happens implicitly. Metadata and provenance remain available.
        """
        import torch

        image = self.image if self.image.flags.writeable else self.image.copy()
        boxes = self.objects.boxes.copy()
        boxes[:, 2:] += boxes[:, :2]
        points = []
        for i in range(len(self.objects)):
            a, b = (int(v) for v in self.objects.keypoint_offsets[i : i + 2])
            points.append(
                torch.from_numpy(
                    np.column_stack((self.objects.keypoints[a:b], self.objects.visibility[a:b]))
                )
            )
        target: dict[str, object] = {
            "uid": str(self.uid),
            "id": self.id,
            "split": self.split,
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(self.objects.class_ids.astype(np.int64)),
            "object_ids": self.objects.ids.copy(),
            "keypoints": points,
            "masks": [
                None if m is None else torch.from_numpy(m.numpy()) for m in self.objects.masks
            ],
            "iscrowd": torch.from_numpy(self.objects.is_crowd.astype(np.int64)),
            "metadata": self.metadata,
            "object_metadata": self.objects.metadata,
            "provenance": self.provenance,
            "objects": self.objects,
        }
        return torch.from_numpy(image).permute(2, 0, 1), target
