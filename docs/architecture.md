# IR and pipeline contract

## Storage layout

`Dataset` owns the category vocabulary, sample array, and dataset metadata. Each `Sample` owns its decoded raster and `ObjectTable`. The hot object columns are independently contiguous: class indices, bounds, instance IDs, primitive descriptors, crowd flags, and keypoint ranges. Polygon descriptors reference a polygon-range pool, which references one flat vertex pool. Keypoints occupy one flat pool. Bitmap masks use packed bits and reside outside the hot box/class columns. Metadata is JSON-compatible and stays outside numerical geometry buffers.

`Annotation` is an owned interchange record for insertions and custom passes. `ObjectTable::push` validates it and packs its variable-sized data. Object filters evaluate borrowed `ObjectRef` views, move retained masks/metadata, and compact geometry pools; filters that retain everything do not copy these pools. Sample-level parallelism gives each worker independent ownership and avoids shared mutation locks.

Categories are dense and zero-based inside IR. Importers validate external IDs and map them to class indices. Skeleton edges are zero-based internally and converted at COCO boundaries. An object's bbox may differ from its segmentation bounds, so both are preserved in the IR and transformed separately; general affine transforms recalculate tight occupied bounds for masks. `MaskCrops` instead translates and preserves the original bbox and shape, separately from the tight pixel crop.

## Identity

`Sample.uid: Uuid` is a 16-byte UUID v4, not a hash of decoded pixels or labels. `Sample.id` retains a readable source identifier. Dataset indices borrow the dataset, so Rust prevents mutation that could invalidate their UUID→position mapping. Packed-file indices own UUID→(offset,length) entries. Derived samples append structured `Sample.provenance` records with the parent UUID, operation, and parameters. Previous history, sample/object metadata, and split stay intact; no user metadata key is reserved for transform history.

## Packed v2 layout and Python

The [packed format specification](packed-format.md) defines explicit little-endian numerical buffers, shape tags, ragged offsets, masks, metadata, and provenance. Records use independent checksummed zstd + MessagePack frames. Writers emit `CVDSIR02`; readers remain compatible with `CVDSIR01`. An independent NumPy reader verifies that the format does not require Rust. Opening scans frame headers only; each lookup decompresses one sample. A complete read additionally validates globally unique source IDs.

The separate `bindings/python` PyO3 crate leaves the Rust core independent of Python. Python `PackedReader` releases the GIL for file I/O, decompression, and native record validation, then transfers decoded RGB ownership into NumPy. Python `PackedTorchDataset` opens a reader per worker process; its custom collator retains variable image sizes and ragged object data. `Sample.to_torch()` exposes CHW uint8 images, XYXY boxes, integer labels, pose and mask targets, and the full original `objects` record. Device transfer and normalization are caller decisions.

`StreamWriter::new(output, categories, metadata, sample_count, options)` accepts a known record count. Call `push(&sample)` for each record and `finish()` to validate the count and flush. Use a `BufWriter<File>` for file output. Only one serialized/compressed record is buffered at a time; identity sets remain O(samples). If a pass expands/drops an unknown number of samples, compute the count first, collect a bounded shard, or write each shard independently. Dropping a writer before `finish` can leave an incomplete file.

Compressed checksums catch accidental corruption, not adversarial tampering. Defaults cap each compressed and decompressed frame at 512 MiB. This is not a global dataset memory quota; eager adapters still need RAM for all decoded samples.

## Transformation order

`Pipeline::run` validates input, applies each pass across samples in Rayon, flattens that pass's ordered outputs, then validates the result. A pass may drop or expand samples. Custom passes implement `Pass`; they are responsible for pixel/annotation consistency. Built-ins use one forward affine matrix for geometry and its inverse for RGB/mask sampling. Pixel centers and pixel edges are distinguished explicitly. RGB interpolation is in encoded RGB space, not linear light.

Cropping uses a row copy for RGB. Resizing uses `fast_image_resize` runtime SIMD dispatch. Arbitrary warps use inverse sampling with parallel rows. Polygon intersection uses `geo`. Masks use nearest-neighbor sampling to preserve binary values. Circle approximation is fixed at 64 vertices when exact analytic representation is unavailable. General crops/warps mark outside pose slots absent; flips only swap anatomical slots when the caller supplies a permutation. `MaskCrops` keeps every original slot and visibility, including coordinates outside its crop. It copies foreground RGB without interpolation, fills all background (including holes), preserves the exact cropped mask, and emits one selected object per sample. Padding is an integer pixel count. Empty/unmasked objects are skipped by the pass; explicitly requesting an empty mask is an error.

## Scope and fidelity

This release handles image samples and crops, not temporal video clips, camera calibration updates, 3D labels, semantic/panoptic rasters, or train/test rebalancing. `metadata` can carry extra information, but generic geometric passes do not reinterpret arbitrary spatial metadata. Self-intersecting polygons should be repaired upstream; polygon parts represent filled simple exteriors and their area is summed (overlapping parts are not unioned for area filtering).

Original compressed image bytes, alpha channels, EXIF, and arbitrary per-vertex Labelme fields are not a lossless archive. Packed IR exactly retains the normalized native representation. Standard-format adapters preserve supported geometry, masks, pose, split, sample identity, and applicable metadata; documented lossy projections require explicit permission.

## Primary format/library references

- [COCO reference mask codec](https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c)
- [Labelme primitives and JSON example](https://github.com/wkentaro/labelme/blob/main/examples/primitives/primitives.json)
- [Ultralytics detection](https://docs.ultralytics.com/datasets/detect/), [pose](https://docs.ultralytics.com/datasets/pose/), and [segmentation](https://docs.ultralytics.com/datasets/segment/)
- [fast_image_resize SIMD and runtime dispatch](https://docs.rs/fast_image_resize/6.1.0/fast_image_resize/)
