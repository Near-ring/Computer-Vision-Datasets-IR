from __future__ import annotations

import gc
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID

import msgpack
import numpy as np
import pytest
import zstandard as zstd
from cv_dataset_ir import (
    Mask,
    PackedReader,
    PackedWriter,
    PurePackedReader,
    crop,
    flip_horizontal,
    mask_crop,
    mask_crops,
    resize,
    rotate,
    write,
    zoom,
)
from cv_dataset_ir.model import ARRAY_DTYPES

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "tests/fixtures/legacy_v1.cvir"


@pytest.fixture
def packed(tmp_path):
    output = tmp_path / "portable.cvir"
    with PackedReader(LEGACY) as reader:
        write(output, list(reader), reader.categories, metadata=reader.metadata)
    return output


def equal_samples(a, b):
    assert a.uid == b.uid and a.id == b.id and a.split == b.split
    assert a.metadata == b.metadata and a.provenance == b.provenance
    np.testing.assert_array_equal(a.image, b.image)
    for name, dtype in ARRAY_DTYPES.items():
        x, y = getattr(a.objects, name), getattr(b.objects, name)
        assert x.dtype == y.dtype == np.dtype(dtype)
        assert x.flags.c_contiguous and y.flags.c_contiguous
        np.testing.assert_array_equal(x, y)
    assert a.objects.metadata == b.objects.metadata
    for x, y in zip(a.objects.masks, b.objects.masks, strict=True):
        if x is None or y is None:
            assert x is y
        else:
            np.testing.assert_array_equal(x.numpy(), y.numpy())


@pytest.mark.parametrize("legacy", [True, False])
def test_native_and_pure_readers(packed, legacy):
    path = LEGACY if legacy else packed
    with PackedReader(path) as native, PurePackedReader(path) as pure:
        assert len(native) == len(pure) == 3
        assert native.version == pure.version == (1 if legacy else 2)
        assert native.categories == pure.categories and native.metadata == pure.metadata
        for i in range(3):
            equal_samples(native[i], pure[i])
            equal_samples(native[native.sample_ids[i]], pure[str(pure.sample_ids[i])])
        assert native[-1].split.kind == "named" and native[-1].split.name == "train"
        assert native[0].image.dtype == np.uint8
        assert native[0].objects.points_for(-2).shape == (2, 2)
        assert native[0].objects.polygons_for(-1)[0].shape == (3, 2)
        with pytest.raises(IndexError):
            native[3]
        with pytest.raises(IndexError):
            native[-4]
        with pytest.raises(KeyError):
            native[UUID(int=999)]


def test_array_lifetime_thread_safety_and_no_native_requirement(packed):
    with PackedReader(packed) as reader:
        with ThreadPoolExecutor(4) as pool:
            samples = list(pool.map(lambda i: reader[i % 3], range(16)))
        sample = samples[0]
    gc.collect()
    sample.image[0, 0] = [11, 12, 13]
    assert samples[3].image[0, 0].tolist() == [0, 1, 2]
    assert sample.objects.keypoints[1].tolist() == [4, 3]
    code = """
import sys
sys.modules['cv_dataset_ir._native'] = None
from cv_dataset_ir import PurePackedReader
with PurePackedReader(sys.argv[1]) as reader:
    assert reader[0].image.shape == (6,8,3)
    assert reader[0].objects.masks[0].numpy().sum() == 15
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code, str(packed)], check=True)


def test_mask_crop_roundtrip_and_transform_api(packed, tmp_path):
    with PackedReader(packed) as reader:
        source = reader[0]
        categories = reader.categories
    result = mask_crops(source, fill=(200, 201, 202))[0]
    assert result.image.shape == (4, 4, 3)
    assert len(result.objects) == 1
    assert result.uid != source.uid and result.split == source.split
    assert result.metadata == source.metadata
    assert result.objects.metadata == source.objects.metadata[:1]
    assert result.objects.ids.tolist() == [19]
    assert result.objects.class_ids.tolist() == [0]
    assert result.objects.shape_types.tolist() == [1]
    np.testing.assert_array_equal(result.objects.shape_params, [[1, 2, 2.5, 0]])
    np.testing.assert_array_equal(result.objects.keypoints, [[-2, -1], [2, 2]])
    np.testing.assert_array_equal(result.objects.visibility, [1, 2])
    mask = result.objects.masks[0].numpy()
    assert mask.sum() == 15 and not mask[1, 1]
    np.testing.assert_array_equal(result.image[mask], source.image[1:5, 2:6][mask])
    np.testing.assert_array_equal(result.image[~mask], [[200, 201, 202]])
    assert result.provenance[0].parent_uid == source.uid
    assert result.provenance[0].parameters["origin"] == [2, 1]
    output = tmp_path / "mask.cvir"
    write(output, [result], categories, metadata={"source": "test"})
    with PackedReader(output) as a, PurePackedReader(output) as b:
        equal_samples(result, a[0])
        equal_samples(result, b[0])
        assert b.metadata == {"source": "test"}
    explicit = np.zeros(source.image.shape[:2], dtype=np.uint8)
    explicit[5, 7] = 255
    one = mask_crop(source, 1, mask=Mask.from_numpy(explicit))
    np.testing.assert_array_equal(one.image[0, 0], source.image[5, 7])
    assert len(one.objects.polygons_for(0)) == 1
    assert mask_crops(source, classes=[9]) == []
    with pytest.raises(ValueError, match="dimensions"):
        mask_crop(source, 0, mask=np.zeros((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="empty mask"):
        mask_crop(source, 0, mask=np.zeros((6, 8), dtype=bool))
    with pytest.raises(ValueError, match="no mask"):
        mask_crop(source, 1)
    for transformed in [
        resize(result, 8, 8),
        rotate(result, 90),
        zoom(result, 1.2),
        crop(result, 0, 0, 3, 3),
        flip_horizontal(result),
    ]:
        assert transformed.metadata == source.metadata
        assert transformed.provenance[-1].parent_uid == result.uid
        assert len(transformed.provenance) == 2


def test_writer_validation(packed, tmp_path):
    with PackedReader(packed) as reader:
        sample, categories = reader[0], reader.categories
    with pytest.raises(OSError):
        write(packed, [sample], categories)
    with pytest.raises(ValueError, match="incomplete"):
        with PackedWriter(tmp_path / "incomplete.cvir", categories, 2) as writer:
            writer.push(sample)
    with pytest.raises(ValueError, match="duplicate"):
        write(tmp_path / "duplicate.cvir", [sample, sample], categories)
    sample.objects.boxes = sample.objects.boxes.astype(np.float64)
    with pytest.raises(ValueError, match="wire array"):
        write(tmp_path / "bad_dtype.cvir", [sample], categories)


def test_direct_wire_layout_and_corruption(packed, tmp_path):
    data = packed.read_bytes()
    assert data[:8] == b"CVDSIR02"
    header_length = struct.unpack_from("<Q", data, 8)[0]
    start = 16 + header_length
    length = struct.unpack_from("<Q", data, start + 16)[0]
    record = msgpack.unpackb(
        zstd.ZstdDecompressor().stream_reader(data[start + 24 : start + 24 + length]).read(),
        raw=False,
    )
    image = record["image"]
    direct = np.frombuffer(image["data"], dtype=image["dtype"]).reshape(image["shape"])
    np.testing.assert_array_equal(direct, np.arange(144, dtype=np.uint8).reshape(6, 8, 3))
    assert record["objects"]["masks"][0]["bitorder"] == "little"
    bad = tmp_path / "bad.cvir"
    for corrupted in [data[:-1], data + b"extra", b"CVDSIR99" + data[8:]]:
        bad.write_bytes(corrupted)
        for cls in [PackedReader, PurePackedReader]:
            with pytest.raises((ValueError, OSError)):
                cls(bad)
    # Checksum corruption is detected when the affected record is read.
    corrupted = bytearray(data)
    corrupted[start + 24 + length - 1] ^= 1
    bad.write_bytes(corrupted)
    for cls in [PackedReader, PurePackedReader]:
        with cls(bad) as reader, pytest.raises((ValueError, OSError, zstd.ZstdError)):
            reader[0]
    for cls in [PackedReader, PurePackedReader]:
        with pytest.raises(ValueError):
            cls(packed, max_record_bytes=16)


def test_readonly_pure_arrays_stay_alive(packed):
    with PurePackedReader(packed, copy_arrays=False) as reader:
        sample = reader[0]
    assert not sample.image.flags.writeable
    gc.collect()
    assert int(sample.image.sum()) == sum(range(144))


def test_real_pytorch_training_and_spawn_workers(packed):
    torch = pytest.importorskip("torch")
    from cv_dataset_ir.torch import PackedTorchDataset, collate_samples
    from torch.utils.data import DataLoader

    with PackedReader(packed) as reader:
        sample = reader[0]
    image, target = sample.to_torch()
    assert image.dtype == torch.uint8 and image.shape == (3, 6, 8)
    assert image.data_ptr() == sample.image.ctypes.data
    sample.image[0, 0, 0] = 253
    assert image[0, 0, 0].item() == 253
    assert target["labels"].dtype == torch.int64
    assert target["masks"][0].sum().item() == 15
    np.testing.assert_array_equal(target["boxes"].numpy(), [[0.5, 0.5, 5.5, 5.5], [1, 1, 2, 2]])
    model = torch.nn.Conv2d(3, 2, 3, padding=1)
    loss = model(image.unsqueeze(0).float() / 255).square().mean()
    loss.backward()
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()
    dataset = PackedTorchDataset(packed)
    dataset[0]  # Force parent reader open before pickling to workers.
    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        multiprocessing_context="spawn",
        collate_fn=collate_samples,
        timeout=45,
    )
    uids = []
    for images, targets in loader:
        assert all(value.shape == (3, 6, 8) for value in images)
        uids.extend(value["uid"] for value in targets)
    assert uids == [str(uid) for uid in dataset.sample_ids]
    dataset.close()
