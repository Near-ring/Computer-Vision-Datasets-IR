> Historical baseline before packed v2 and the Python API. See [current mask/Python validation](python-validation.md) for the additions and current test counts.

# Validation — 2026-09-24

Tested on Linux x86-64, Intel Core i7-12700F. The library uses stable Rust features. Functional tests run with stable rustc 1.97.1; Clippy and release measurements use the installed rustc 1.99.0-nightly. The declared Rust 1.89 minimum follows the locked dependencies; that minimum compiler was not separately installed/tested.

## Automated checks

- `cargo fmt --check`
- `cargo +stable test --all-targets`: 31 unit/integration tests, plus benchmark smoke harnesses and example compilation.
- `cargo +stable test --doc`: two compiled README examples.
- `cargo clippy --all-targets -- -D warnings`
- `cargo +stable run --example pipeline`: self-contained import-free pipeline and packed UUID lookup.

The tests include three-image train/val/test fixtures for Labelme, COCO, and all three YOLO tasks; independently authored external JSON/YAML/label inputs; circles, polygons, points, pose visibility, and bit-packed masks; multipart clipping and holes; mask/pixel/keypoint alignment under crops; rotation/flip/resize and scale-sensitive circle conversion; identical numerical outputs with one/four threads; UUID round-trips and duplicate rejection; packed truncation, checksums, size limits, streaming counts, and random access; background and empty datasets; explicit lossy-export reporting; invalid paths and conflicting metadata; and fractional/clipped Labelme mask placement.

COCO RLE has eight independent golden cases generated with **pycocotools 2.0.11 / NumPy 2.5.3**, including all-zero masks, all-one masks, holes, nonsquare layouts, negative count deltas, and a one-pixel image. Tests assert exact encoded strings and decoded pixels. Regeneration instructions are in `tests/fixtures/README.md`.

## Public data smoke test

Eight images each from the official [COCO8 detection](https://docs.ultralytics.com/datasets/detect/coco8/), [COCO8-pose](https://docs.ultralytics.com/datasets/pose/coco8-pose/), and [COCO8-seg](https://docs.ultralytics.com/datasets/segment/coco8-seg/) subsets. Downloads total about 1.24 MB; images are kept under ignored `target/public-data`, not bundled with the crate. Upstream dataset/image licensing still applies.

| Source | Images | Objects | Split |
|---|---:|---:|---|
| COCO8 | 8 | 30 | 4 train, 4 val |
| COCO8-pose | 8 | 21 | 4 train, 4 val |
| COCO8-seg | 8 | 30 | 4 train, 4 val |

All three datasets passed packed, COCO, Labelme, and YOLO export/import checks. Assertions cover decoded pixel equality, UUID lookup, split membership, object counts, bbox tolerances, and pose slots/visibility. Each also passes a 320×320 resize followed by 12° rotation. The independent `pycocotools.COCO` reader loaded all six exported split JSON files (81 annotations); polygon masks rasterized successfully and pose schemas matched.

Archive SHA-256 values from the verified run:

```text
coco8.zip       54c67fe9ef88313e021ec0e92b73c200167bb0a86633e8df8658d832cca828c9
coco8-pose.zip  468e06839e06eaf317594386967d238af4bdfde670e33d9a2a287412b6a37de5
coco8-seg.zip   82c651abd01d556c77769ea834ebff1e77f76a291463e987fb69b63a87e8eb80
```

To reproduce in a new output directory (requires curl and Python's standard-library zipfile):

```sh
mkdir -p target/public-data
for name in coco8 coco8-pose coco8-seg; do
  curl --fail --location "https://github.com/ultralytics/assets/releases/download/v0.0.0/$name.zip" \
    --output "target/public-data/$name.zip"
  python3 -m zipfile -e "target/public-data/$name.zip" target/public-data
  curl --fail --location \
    "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/$name.yaml" \
    --output "target/public-data/$name.yaml"
done
cargo run --release --example public_smoke -- target/public-data target/public-smoke-reproduction
```

The example emits a JSON report in its output directory; the final captured run is [public-smoke-results.json](public-smoke-results.json). Independent checks also compared all 81 exported YOLO rows against the downloaded labels with a maximum normalized coordinate error below 1e-6; see [interop-results.json](interop-results.json). Release smoke timings are informational single runs; use Criterion for repeated timing. Packed file sizes exceed the source JPEG archive sizes because the pack stores decoded RGB pixels compressed with zstd, preserving the normalized IR exactly.

## Measured CPU throughput

Criterion, release profile with thin LTO, runtime AVX2 dispatch. No `target-cpu` override. Batch: **32 RGB images, 640×480 → 320×240, 100 rectangular annotations per image**. Timings include the pipeline's validation and annotation updates; input cloning is setup outside the timed portion. Output destruction is included. Filesystem I/O is excluded.

| Operation | Time per batch / lookup | Throughput |
|---|---:|---:|
| Resize, 1 Rayon thread | 7.996 ms / 32 images | 4,002 images/s |
| Resize, 4 Rayon threads | 3.516 ms / 32 images | 9,101 images/s |
| Filter 3,200 boxes (reject all) | 0.209 ms / 32 images | — |
| UUID hash-index lookup (32-entry index) | 10.10 ns | — |

Four-thread resize was **2.27×** faster in this workload. These are local synthetic measurements, not a guarantee for large images, mask-heavy datasets, storage performance, or larger hash indices. Sample counts: 10 for dataset benchmarks, 100 for UUID lookup. Full confidence intervals and environment are in [benchmark-results.json](benchmark-results.json); raw Criterion data remains under `target/criterion` and the captured log under `target/benchmark.log`.

The pack uses checksummed, independent zstd records and supports bounded per-record reading/writing. Standard-format frontends are eager; production datasets larger than RAM should be sharded. Standard-format conversions can have representational losses; strict export is the default, and allowed losses are reported explicitly.
