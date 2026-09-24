"""Optional worker-local PyTorch dataset; importing the main package needs no torch."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypeAlias, cast

from torch import Tensor
from torch.utils.data import Dataset

from .api import PackedReader
from .api import Path as PathArg

Item: TypeAlias = tuple[Tensor, dict[str, object]]


class PackedTorchDataset(Dataset[Item]):
    """One native reader per process, opened lazily for DataLoader spawn/fork workers."""

    def __init__(self, path: PathArg, *, max_record_bytes: int = 512 * 1024 * 1024) -> None:
        self.path = str(Path(path).resolve())
        self.max_record_bytes = max_record_bytes
        with PackedReader(self.path, max_record_bytes=max_record_bytes) as reader:
            self.categories = reader.categories
            self.metadata = reader.metadata
            self.sample_ids = reader.sample_ids
        self._reader: PackedReader | None = None
        self._pid: int | None = None

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Item:
        if self._reader is None or self._pid != os.getpid():
            # A forked process must never reuse the parent's file cursor.
            self.close()
            self._reader = PackedReader(self.path, max_record_bytes=self.max_record_bytes)
            self._pid = os.getpid()
        return self._reader[index].to_torch()

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        self._pid = None

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_reader"] = None
        state["_pid"] = None
        return cast(dict[str, object], state)


def collate_samples(batch: list[Item]) -> tuple[list[Tensor], list[dict[str, object]]]:
    """Keep variable image sizes and ragged object/pose targets intact."""
    return [image for image, _ in batch], [target for _, target in batch]
