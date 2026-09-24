# Mask crops, packed v2, and Python validation

Verified 2026-09-24 on Linux x86-64. Core/workspace checks used stable rustc 1.97.1. Maturin release packaging used the installed rustc 1.99.0-nightly; the extension uses no nightly APIs. Python execution used CPython 3.12.13, PyO3 0.29.2, rust-numpy 0.29.0, NumPy 2.5.3, MessagePack 1.2.2, zstandard 0.25.0, and torch 2.14.0+cpu. This turn's training execution was **CPU**; CUDA, other Python versions, and other platforms were not exercised.

## Results

| Check | Result |
|---|---|
| `cargo +stable fmt --all --check` | Passed |
| `cargo +stable clippy --workspace --all-targets -- -D warnings` | Passed |
| `cargo +stable test --workspace --all-targets` | 35 Rust unit/integration tests passed; examples and benchmark smoke harnesses passed |
| `cargo +stable test --doc` | 2 README examples compiled |
| Ruff check and format check | Passed, 8 Python/stub files |
| BasedPyright | 0 errors, 0 warnings |
| `maturin build --release --interpreter .venv/bin/python --out target/wheels` | Built CPython 3.12 manylinux_2_35 x86-64 wheel |
| Install wheel with `uv pip install --force-reinstall --no-deps …whl` | Passed; imported from venv site-packages, not the source editable package |
| `pytest -q --junitxml=target/checks/python-tests.xml` against installed wheel | 8 tests passed in 2.01 s; torch test executed, not skipped |
| Rust-generated v2 fixture read through native and independent Python | All three sample pixels/pose/splits/metadata agreed |
| PyTorch example, two spawned workers | All three samples loaded; CPU forward/backward/optimizer steps completed |

No new throughput claim is inferred from these functional checks. Earlier baseline timing measurements remain in [validation.md](validation.md).

## Coverage

The frozen **v1** fixture contains three synthetic 8×6 images with train, val, and a custom split named `train`. Its circle object has a holey mask, two pose points (one outside the tight mask crop), and nested object metadata. A second object is a polygon without a mask. Sample metadata contains a user-owned `operation` key. This is deliberately independent of the current v2 writer.

Rust tests verify exact foreground RGB, background fill including mask holes, cropped mask occupancy, class/instance IDs, original circle and bbox translation, pose coordinates/visibility, metadata preservation, parent UUID history, padding saturation, explicit masks, unmasked objects, empty masks, and class selection. Pixel crop extents use integer bounds. The empty-mask test exposed and fixed an eager subtraction underflow in `Mask::bounds`.

Wire tests check old-file migration, all column types, ragged offsets, malformed dtype/dimensions/visibility/mask order, and non-finite values. Existing tests continue to exercise checksums, truncation, record limits, duplicate IDs, and all standard-format adapters. V2 output preserves the normalized IR exactly, including mask crops and structured provenance.

Python tests compare native versus independent reader arrays and metadata for v1 and v2, then write Python objects back through the Rust writer and read through both implementations. They test UUID/position lookup, concurrent native reads, array ownership after reader close, read-only reference-reader views, write counts, overwrite refusal, corruption, and native transformations. A subprocess explicitly blocks the native module and still reads v2 via `PurePackedReader`; importing the package does not import torch.

The PyTorch test verifies shared RGB storage using the NumPy pointer and a mutation, CHW uint8 layout, XYXY boxes, int64 labels, masks and pose, finite Conv2d gradients, and a two-worker spawned DataLoader after opening a parent-process reader. Targets include the complete original `Annotations` object alongside common training fields, preserving shape/polygon data and metadata.

## Reproduce

Use a fresh output path for the fixture and install the package first as shown in the README. For development builds activate the venv, then run `maturin develop`. The commands above run from the repository root.

```sh
cargo +stable run --example interoperability_fixture -- target/python-v2-fixture.cvir
.venv/bin/python examples/pytorch_loader.py target/python-v2-fixture.cvir --workers 2
```

Observed release-wheel example output (random model initialization means loss values vary):

```text
samples=2, loss=0.030602, device=cpu
samples=3, loss=0.028695, device=cpu
```

The wheel is under `target/wheels/`; JUnit output is under `target/checks/python-tests.xml`. These generated files are ignored. The checked-in schema, tests, fixture, and examples are the reproducible sources.

## Relevant upstream API contracts

- [PyO3 detach for work outside the GIL](https://pyo3.rs/main/parallelism)
- [rust-numpy owned array conversion](https://docs.rs/numpy/0.29.0/numpy/convert/trait.IntoPyArray.html)
- [Maturin mixed Rust/Python layout](https://www.maturin.rs/project_layout.html)
- [PyTorch DataLoader and multiprocessing](https://docs.pytorch.org/docs/stable/data.html)
