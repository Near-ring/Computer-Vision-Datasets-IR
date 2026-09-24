# Packed IR v2: language-neutral schema

Writers emit v2 (`CVDSIR02`). Rust, the PyO3 reader, and the independent Python reader also read v1 (`CVDSIR01`). V1 used Serde-shaped records; v2 makes numerical layout explicit. The file is **not Rust memory, pickle, or a Python object archive**. Reading needs a seekable file, MessagePack, zstd, and basic array handling. No source images or separate annotation files are needed.

## Framing

```text
ASCII "CVDSIR02"                         8 bytes
compressed_header_length                u64 little endian
zstd(MessagePack(header))                 that many bytes
repeat header.samples times:
    UUID                                16 bytes, standard UUID byte order
    compressed_record_length            u64 little endian
    zstd(MessagePack(record))            that many bytes
EOF
```

Each frame includes a zstd checksum. MessagePack maps have string keys; binary buffers use MessagePack `bin`. Header:

```json
{
  "schema": "cv-dataset-ir",
  "version": 2,
  "categories": [{"name": "plant", "keypoints": ["tip", "root"], "skeleton": [[0, 1]], "metadata": {}}],
  "metadata": {},
  "samples": 3
}
```

Categories are indexed from zero. Skeleton endpoints index their category's named keypoints from zero. Metadata at every level is a JSON-compatible map, including nested lists/maps. Large integer metadata values remain MessagePack integers; languages using only double-precision numbers must take care above 2^53.

Scan frame headers and seek over bodies to build UUID→offset/length. Only requested samples are decompressed. No global compression dictionary is required. Existing destinations are never overwritten. A streaming writer needs the final sample count; successful `finish()` validates it and flushes.

## Record and array layout

Each record contains `uid` (canonical UUID string), `id` (readable string), `split`, `image`, `objects`, `metadata`, and `provenance`. Frame UUID and record UUID must agree. A split is `{kind, name}`: `kind` is `train`, `val`, `test`, or `unassigned` with `name: null`, or `kind: named` with a string `name`. Thus a custom split called `train` remains distinguishable from the standard training split.

An array is `{dtype: string, shape: [dimensions], data: binary}`. All arrays are tightly packed, C order, without alignment padding or strides. Multibyte numbers are **little endian**. Dtypes use NumPy spelling: `|u1` is uint8, `<u4` uint32, `<u8` uint64, `<f4` IEEE float32. `len(data)` must equal the checked product of dimensions and element size.

| Field | Dtype | Shape | Meaning |
|---|---|---|---|
| `image` | `\|u1` | `[H,W,3]` | RGB8, decoded pixels |
| `objects.ids` | `<u8` | `[N]` | Original instance IDs, not required to be globally unique |
| `objects.class_ids` | `<u4` | `[N]` | Category indices |
| `objects.boxes` | `<f4` | `[N,4]` | x, y, width, height; distinct from shape bounds |
| `objects.shape_types` | `\|u1` | `[N]` | 0 rectangle, 1 circle, 2 polygons, 3 point |
| `objects.shape_params` | `<f4` | `[N,4]` | See below |
| `objects.polygon_object_offsets` | `<u8` | `[N+1]` | Object→polygon range |
| `objects.polygon_offsets` | `<u8` | `[R+1]` | Polygon→vertex range |
| `objects.vertices` | `<f4` | `[V,2]` | Polygon x,y coordinates |
| `objects.keypoint_offsets` | `<u8` | `[N+1]` | Object→pose slot range |
| `objects.keypoints` | `<f4` | `[K,2]` | Pose x,y coordinates |
| `objects.visibility` | `\|u1` | `[K]` | 0 absent, 1 occluded, 2 visible |
| `objects.is_crowd` | `\|u1` | `[N]` | 0 or 1 |

Shape parameters: rectangle `[x,y,width,height]`, circle `[center_x,center_y,radius,0]`, polygons `[0,0,0,0]`, point `[x,y,0,0]`. Polygon parts are filled exteriors; binary masks encode holes. Offsets start at zero, are monotone, and end at the referenced pool length. Non-polygons have empty polygon ranges. Objects may have no pose slots; otherwise slot count matches the category schema. Geometry is finite. Coordinates can be outside the image, notably after a mask crop preserving the full original shape and pose.

`objects.metadata` is a list of N maps. `objects.masks` is a list of N entries, each `null` or:

```text
{width: W, height: H, bitorder: "little", data: binary}
```

The bit for pixel `(x,y)` is `(data[(y*W+x)//8] >> ((y*W+x)%8)) & 1`. Rows are contiguous with **no per-row padding**; only the last byte may have padding bits, all zero. The byte count is `ceil(H*W/8)`. Decode using `np.unpackbits(bits, bitorder="little", count=H*W).reshape(H,W)`. A mask has exactly its sample's dimensions.

Provenance is a list of `{parent_uid: UUID string, operation: string, parameters: map}`. Each derived sample gets a fresh UUID and appends history without modifying user metadata. Mask-crop parameters include original pixel `origin`, `source_size`, `padding`, `object_index`, `object_id`, and `fill`. Bbox, shape, and all keypoints are translated by `-origin`; visibility and slot order are retained, including points outside the crop.

## Minimal independent NumPy read

This trusted-file illustration reads the first sample without this crate or extension. The included [`PurePackedReader`](../python/cv_dataset_ir/portable.py) adds indexing, limits, validation, v1 compatibility, and conversion of every column.

```python
import struct
import msgpack
import numpy as np
import zstandard as zstd

with open("prepared.cvir", "rb") as f:
    assert f.read(8) == b"CVDSIR02"

    def frame():
        size = struct.unpack("<Q", f.read(8))[0]
        with zstd.ZstdDecompressor().stream_reader(f.read(size)) as decoder:
            return msgpack.unpackb(decoder.read(), raw=False)

    header = frame()
    assert header["samples"] > 0
    uuid_bytes = f.read(16)
    record = frame()


def array(value):
    return np.frombuffer(value["data"], dtype=value["dtype"]).reshape(value["shape"])

image = array(record["image"])          # HWC uint8; read-only view into decoded bytes
boxes = array(record["objects"]["boxes"])  # N×4 float32 XYWH
# For a writable tensor, use torch.from_numpy(image.copy()).permute(2, 0, 1).
```

Compression requires decompression and allocation; this is not mmap or zero-copy disk access. The PyO3 reader moves its decoded RGB buffer into NumPy ownership without an additional full-image copy; other annotation buffers are materialized. `torch.from_numpy` can share the writable RGB array, while CHW permutation is a view. Normalization/device transfers are explicit. Python write/transform inputs are serialized and copied into Rust-owned buffers; they are not zero-copy calls.

## Limits and compatibility

Readers validate framing, UUIDs, dimensions, dtypes, ragged offsets, class indices, pose schema, and masks. Each compressed/decompressed record defaults to a 512 MiB limit. Decoder windows are also bounded, with an 8 MiB minimum to accommodate streaming zstd frames. This is not a total process memory quota. Checksums detect accidental corruption, not malicious tampering.

V1 remains readable, but new files are v2 and old readers cannot read them. Reading then writing migrates v1 automatically. V1 files without structured provenance load with an empty history; existing user metadata is retained verbatim. The frozen three-image fixture in `tests/fixtures/legacy_v1.cvir` predates this change and tests the compatibility path.
