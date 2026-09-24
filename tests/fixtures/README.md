# COCO RLE golden cases

`coco_rle.json` was generated independently with pycocotools 2.0.11 and NumPy 2.5.3. It contains small binary masks, column-major compressed COCO RLE strings, and reference areas. Rust tests assert exact encoded strings and decoded row-major pixels. No Python dependency is needed to run the tests.

Regeneration (in a temporary uv environment with those package versions):

```python
import json
from pathlib import Path
import numpy as np
from pycocotools import mask

rng = np.random.default_rng(42)
cases = []
for i, (h, w) in enumerate([(2, 3), (5, 7), (7, 5), (17, 33),
                             (3, 3), (9, 11), (2, 2), (1, 1)]):
    pixels = (rng.random((h, w)) > 0.4).astype(np.uint8)
    if i == 4:
        pixels[:] = 0
    if i == 5:
        pixels[:] = 1
        pixels[2:7, 3:8] = 0
    if i == 7:
        pixels[:] = 1
    rle = mask.encode(np.asfortranarray(pixels))
    cases.append({
        "width": w, "height": h, "dense": pixels.ravel().tolist(),
        "counts": rle["counts"].decode(), "area": int(mask.area(rle)),
    })
Path("tests/fixtures/coco_rle.json").write_text(json.dumps(cases, indent=2) + "\n")
```

## Frozen packed v1 interoperability fixture

`legacy_v1.cvir` was written by the pre-v2 Rust writer, before the wire-format change. It contains three 8×6 RGB samples with deterministic UUIDs 1/2/3; train/val/custom-`train` splits; a circle with a holey binary mask and two pose points; a second polygon object; and nested dataset/object metadata. Sample metadata intentionally uses the key `operation` to detect accidental provenance overwrites.

SHA-256: `f02407f762930d34e0ec19aa3f57c9a2996fbf027077d325c19f1e674e15ce18`.

Do not regenerate this fixture with the current writer: `examples/interoperability_fixture.rs` now creates the equivalent **v2** data. Keeping the original v1 bytes tests actual backward compatibility independently of the current encoder. All content is synthetic; no external image downloads or licenses are involved.
