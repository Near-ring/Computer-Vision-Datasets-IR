# cv-dataset-ir

An API-first Rust crate for detection, pose, and instance-segmentation datasets:

```text
COCO / Labelme / YOLO / packed IR
                 ↓
RGB8 + annotation columns + polygon/keypoint pools + bit-packed masks
                 ↓
parallel filter / crop / instance crop / mask crop / resize / rotate / zoom / affine
                 ↓
COCO / Labelme / YOLO / packed IR
```

No application CLI or GPU runtime is required. Stable Rust, edition 2024; dependencies require Rust 1.89 or newer. The checked-in lockfile makes development and tests reproducible. Examples are runnable demonstrations of the library API.

## Use the API

```rust,no_run
use cv_dataset_ir::{Result, packed};
use cv_dataset_ir::formats::{yolo, ExportOptions};
use cv_dataset_ir::transforms::{Filter, InstanceCrops, Pipeline, Resize};

fn prepare() -> Result<()> {
    let dataset = yolo::read("input/data.yaml", yolo::Task::Detection)?;
    let prepared = Pipeline::new()
        .then(Filter { min_area: 64.0, ..Default::default() })
        .then(InstanceCrops {
            padding: 0.1,
            keep_neighbors: false,
            ..Default::default()
        })
        .then(Resize { width: 224, height: 224 })
        .run(dataset)?;

    // Output paths must be new; existing files/directories are not overwritten.
    packed::write(&prepared, "prepared.cvir", Default::default())?;
    yolo::write(
        &prepared, "output", yolo::Task::Detection, ExportOptions::default(),
    )?;
    Ok(())
}
```

`formats::Frontend` and `formats::Backend` support generic adapter code; concrete `Reader` and `Writer` types exist in each standard-format module. `packed::{read, write, write_to, Reader}` provide native storage and random access. `Pass::apply` supports custom sample transforms with zero, one, or many outputs. `FilterObjects::new` accepts a custom predicate; `ObjectRef::area()` uses mask area when present, otherwise geometry area. The built-in `Filter` uses **bounding-box area in pixels²**.

Run the self-contained example with `cargo run --example pipeline`. See [examples/pipeline.rs](examples/pipeline.rs), [architecture](docs/architecture.md), and [validation](docs/validation.md).

## Native representation and identity

- `Raster`: decoded RGB8, contiguous row-major bytes. PNG/JPEG/WebP inputs; lossless PNG outputs. Grayscale expands to RGB, alpha is dropped, and EXIF orientation is not applied. Original encoded bytes are not retained.
- `ObjectTable`: contiguous class IDs, boxes, IDs, shape descriptors, crowd flags, and keypoint ranges. Polygon vertices and keypoints live in flat pools. `Annotation` is the ergonomic owned insertion type; `ObjectRef` reads columns without copying images. `polygons()` borrows vertex slices; `shape()` materializes an owned shape.
- `Shape`: rectangle, circle, multipart polygon, or standalone point. Polygon parts are filled exteriors; use `Mask` for holes.
- `Mask`: exact binary occupancy, one bit per pixel, row-major. COCO RLE is converted at the boundary. Semantic/panoptic class maps are not currently a separate IR modality.
- `Category`: dense class index, name, named pose slots, zero-based skeleton edges, and metadata.
- `Sample`: **128-bit UUID v4**, original/readable `id`, image, objects, split, metadata, structured provenance. A UUID is an identifier, not a content checksum. Independent imports without a saved identity generate fresh UUIDs.

```rust,no_run
# use cv_dataset_ir::{Dataset, Result, Uuid, packed};
# fn lookup(dataset: &Dataset, uid: Uuid) -> Result<()> {
let index = dataset.index()?; // O(samples), borrows the dataset
let sample = index.get(uid); // expected O(1), no image copy
let same = index.get_u128(uid.as_u128());
let mut file = packed::Reader::open("prepared.cvir", Default::default())?;
let one_sample = file.read_sample(uid)?; // seeks and decodes just this record
# Ok(()) }
```

UUIDs survive native and supported standard-format round-trips. COCO/Labelme store identity in `_cv_ir` JSON extensions; YOLO uses `_ir_manifest.json`. Keep extensions/sidecars when identity matters. Crops, warps, flips, resizes, and zooms issue fresh UUIDs and append `parent_uid`, `operation`, and parameters to `Sample.provenance` without overwriting user metadata. Filtering keeps the sample UUID. `Dataset::validate()` rejects duplicate source IDs and nil/duplicate UUIDs. IDs are not filenames; output filenames use UUIDs.

## Geometry and transforms

Coordinates are absolute `f32` pixel-edge coordinates. Boxes use `(x, y, width, height)` with half-open pixel extents. Points/keypoints may lie on image edges `[0, width] × [0, height]`. Images are sampled at `(x + 0.5, y + 0.5)` using the inverse affine map. Pose visibility is `Absent=0`, `Occluded=1`, `Visible=2`; general geometric crops/warps mark cropped-out slots `(0,0,Absent)` without changing slot order. `MaskCrops` preserves all slots and visibility instead.

`Crop` is an integer rectangle entirely inside the source. Polygons are clipped with `geo`, including concave shapes that split into several components. `min_visibility` is retained **shape area / original shape area**, not box IoU or mask-pixel fraction. Fully excluded objects and transformed masks with no foreground are dropped. Empty images otherwise remain unless a filter requests `drop_empty`.

`MaskCrops` copies foreground RGB exactly inside each mask's tight bounding rectangle, with optional integer padding and configurable background fill. It preserves holes/islands in the cropped binary mask. Each result keeps the selected object's ID, class, crowd flag, metadata, original shape/bbox, and all pose slots translated into crop coordinates; points outside the crop remain available. Sample metadata and split stay intact. Use `MaskCrops::crop(sample, object_index, mask)` for an explicit image-sized IR mask, `crop_instance` for its stored mask, or add the pass to a pipeline. Packed output retains the complete result. Unmasked/empty instances are skipped by the pass; an explicit empty-mask crop returns an error.

`InstanceCrops` creates one crop per selected bbox, supports fractional padding, and optionally keeps neighboring objects. `Resize` uses SIMD-enabled bilinear convolution. `Warp` supports arbitrary invertible affine maps, nearest/bilinear RGB sampling, and a fill color; masks always use nearest-neighbor occupancy. `Rotate` is clockwise in image coordinates and keeps the original canvas. `Zoom` scales around the canvas center. `FlipHorizontal` accepts a keypoint slot permutation for anatomical left/right swaps; no schema-dependent swap is guessed.

Circles stay analytic under similarity transforms while fully inside the output. Clipped circles and circles under nonuniform scale/shear use 64-vertex polygons. Rotated rectangles become polygons. Transform geometry approximations are fixed and documented; exporter approximations additionally require `LossPolicy::Allow` and return warnings.

## Format behavior

| Format | Input/output coverage | Boundaries |
|---|---|---|
| COCO | Boxes, multipart polygons, compressed or uncompressed RLE, crowd flags, keypoints/skeletons; sparse input class IDs | Supply split + image root per annotation file with `coco::Source`. `read_dir` reads this crate's generated split layout. Output remaps numeric image/category/annotation IDs. Circle output requires approximation permission. |
| Labelme | Rectangles, circles, polygons, points, bitmap masks, grouped pose points, embedded or referenced images, flags | Grouped pose points require a category keypoint schema or `_categories.json`. Multipart polygons share a group. Lines/linestrips and mixed non-polygon shapes in one group return errors. Referenced images must stay within the annotation directory. |
| YOLO | YAML names as list/map; image directories, lists of directories, and image-list files; detection, segmentation, 2D/3D pose, keypoint names and flip indices | Caller specifies task. Missing label files mean background images. Pose classes need equal slot counts. Standard segmentation has one polygon per row. |
| Packed IR | Exact native image pixels, all annotations, metadata, UUIDs, splits | Portable v2 typed byte arrays + MessagePack + checksummed zstd records; v1 reads supported; bounded record decoding; seekable UUID lookup. Not mmap/zero-copy and not an archive of source files. |

`LossPolicy::Error` is the export default. With `Allow`, each lossy object conversion is reported in `ExportReport::warnings`: geometry-to-box reduction, discarded pose slots, circle tessellation, mask contour conversion, lost holes, or selecting the largest disconnected polygon for YOLO. No invalid polygon is silently emitted. Standard formats are not universally lossless; use packed IR as the canonical checkpoint. Format-specific numeric IDs and grouping conventions may be regenerated.

Train/val/test/unassigned/custom splits live on samples and are retained through passes. Labelme infers standard split names from directories; YOLO reads YAML split membership. Set `ExportOptions { splits: SplitPolicy::Ignore, .. }` to flatten output, or call `Dataset::ignore_splits()` explicitly. Custom/unassigned YOLO splits are carried in the sidecar; training tools need an explicit train/val assignment before consuming them.

## Execution and storage

Rayon parallelizes image imports, exports, and sample passes; arbitrary image warps also parallelize rows. Results retain input order within each pass regardless of thread count. UUIDs are intentionally newly randomized for derived samples, so identities are not deterministic across independent runs.

Install `Pipeline::run` in a caller-owned `rayon::ThreadPool` to set thread count. `fast_image_resize` uses runtime SIMD dispatch (AVX2/SSE on x86, NEON on supported ARM) with scalar fallback. There is no hand-written unsafe code. This crate does not force CPU-specific flags on downstream applications. For known x86-64-v3 deployment targets, opt in with `RUSTFLAGS="-C target-cpu=x86-64-v3"`.

File readers/writers are buffered. Packed records compress independently and the reader builds a UUID→offset index by skipping compressed bodies. The default compressed/decompressed record limit is 512 MiB; configure `packed::Options` for larger images. Whole-dataset format adapters eagerly decode images into memory. Use `packed::Reader::read_sample` and `Pipeline::run_sample` for bounded per-sample processing, or shard large datasets. A streaming packed writer is also available (see architecture).

Writers reject existing destinations and preflight standard-format representability before creating output. An I/O failure can leave a **new, partial** output; directory export is not a filesystem transaction. Keep the source dataset until validation succeeds.

## Python / NumPy / PyTorch

Build the optional PyO3 package from this checkout (Rust is required for this step):

```sh
uv venv --python 3.12
uv pip install .
# Optional: choose the torch wheel index appropriate for your CPU/CUDA environment.
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```python
from cv_dataset_ir import PackedReader, mask_crops, write

with PackedReader("prepared.cvir") as reader:
    sample = reader[0]               # also reader[uuid] or reader[str(uuid)]
    categories = reader.categories
    dataset_metadata = reader.metadata

print(sample.image.shape)            # HWC uint8 NumPy array
print(sample.objects.boxes)          # contiguous N×4 float32 XYWH
crops = mask_crops(sample, padding=4, fill=(0, 0, 0))
write("instances.cvir", crops, categories, metadata=dataset_metadata)
image, target = sample.to_torch()    # CHW uint8; boxes become XYXY
# image.float().div(255).to(device) is an explicit caller choice.
```

Python also exposes `mask_crop` with an explicit NumPy/IR mask, resize/crop/rotate/zoom/flip, and a count-checked streaming `PackedWriter`. `Sample.to_torch()` retains metadata, provenance, and the full `objects` record alongside training targets. Foreground masks are boolean tensors. Pose targets remain ragged lists; no category-dependent padding is guessed.

`cv_dataset_ir.torch.PackedTorchDataset` supports worker-local readers and spawn/fork DataLoader workers. Use `collate_samples` for variable-size samples. The [PyTorch example](examples/pytorch_loader.py) performs a real CPU forward/backward pass:

```sh
cargo run --example interoperability_fixture -- target/example.cvir
.venv/bin/python examples/pytorch_loader.py target/example.cvir --workers 2
```

`PurePackedReader` implements the same read model using only NumPy, MessagePack, zstandard, and the standard library. It imports neither the Rust extension nor PyTorch. See the [full schema and short independent loader](docs/packed-format.md). Native arrays remain valid after closing the reader. Native reads release the GIL; one reader serializes concurrent requests, so use separate readers for parallel decompression. Disk reads are not zero-copy; Python write/transform inputs are also copied for ownership safety.

## Development

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo test --doc
cargo bench --bench throughput
```

Tests create small fixtures (three images per standard-format task), including malformed inputs and exact pixel/annotation assertions. The optional public smoke example checks eight images each from COCO8 detection, pose, and segmentation through all four formats. Downloaded datasets and generated reports belong under ignored `target/`; no large datasets are needed for normal tests.

Python development checks (activate the venv for `maturin develop`):

```sh
uv pip install maturin pytest ruff basedpyright
. .venv/bin/activate
maturin develop
ruff check python examples/pytorch_loader.py
ruff format --check python examples/pytorch_loader.py
basedpyright
pytest -q
cargo clippy --workspace --all-targets -- -D warnings
```

The [mask/Python validation record](docs/python-validation.md) documents cross-language round trips, legacy reads, preservation checks, and actual CPU PyTorch execution.
