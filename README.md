# uo-tpu-adapters

Python adapters for running detection/classification/segmentation models on **Hailo** and
**Google Coral Edge TPU** hardware behind one shared result type, so application code can target
either accelerator without depending on their (very different) native SDKs directly.

Nothing here depends on a particular kind of imagery or application pipeline. Anything that
needs to run inference on a Hailo TPU or a Coral Edge TPU — a different embedded
vision project, a robotics application, any edge-inference workload — can depend on this package
and get the same `Object`/`BBox` result types and adapter functions regardless of which of the
two accelerators the target device has.

## Packages

- **`uo_tpu_adapters.common`** — hardware-agnostic result types shared by both backends:
  `Object` (`id`, `score`, `bbox`) and `BBox` (`xmin`, `ymin`, `xmax`, `ymax`, with `width`,
  `height`, `area`, `scale()`, `translate()`, `intersect()`, `union()`, `iou()`). Code written
  against these types works unchanged whether inference ran on Hailo or Coral.

- **`uo_tpu_adapters.hailo`** — Hailo TPU support:
  - `core.interpreter.Interpreter` — wraps the Hailo SDK (`hailo_platform`: `VDevice`, `HEF`,
    scheduling) behind a small `configure_model()` / `set_input()` / `invoke()` /
    `invoke_async()` / `get_output()` API, hiding HEF loading, binding creation, and
    sync/async output handling.
  - `adapters.detect` / `adapters.classify` / `adapters.segment` — turn raw interpreter output
    into lists of `Object`/`BBox` (or per-pixel masks, for segmentation).
  - `utils.hailotpu` — input-size/resize helpers used by the adapters.

- **`uo_tpu_adapters.pycoral`** — Coral Edge TPU support: `adapters.detect` /
  `adapters.classify` / `adapters.segment`, plus `utils` (dataset/EdgeTPU helpers), `learn`
  (on-device transfer learning: backprop softmax regression, weight imprinting), and `pipeline`
  (multi-model pipelined runner). This mirrors Google's `pycoral` adapter API but returns the
  same `common.Object`/`BBox` types as the Hailo side. **Not** importable as top-level `pycoral`
  — its code does `from pycoral.pybind import _pywrap_coral`, an absolute import of the real,
  separately installed `pycoral~=2.0` package's compiled extension. That real package stays a
  genuine runtime dependency (install with the `pycoral` extra below) rather than being replaced.

## Usage

```python
from uo_tpu_adapters.hailo.core.interpreter import Interpreter
from uo_tpu_adapters.hailo.adapters import detect

interpreter = Interpreter("model.hef")
interpreter.configure_model()
interpreter.set_input([image])
interpreter.invoke()

objects = detect.get_objects(interpreter, score_threshold=0.5)
# -> list of uo_tpu_adapters.common.Object(id, score, bbox)
```

Swapping to Coral Edge TPU means swapping the interpreter/adapter import (`uo_tpu_adapters.pycoral`
instead of `uo_tpu_adapters.hailo`) — calling code that only touches `Object`/`BBox` doesn't
change.

## Install

Requires the target device's own runtime for whichever backend you use (Hailo's `hailo_platform`
SDK for `hailo`; the real `pycoral` package, pulled in below, for `pycoral`) — both are
hardware/driver-dependent and installed separately from this package.

Base install (Hailo branch — no pycoral runtime needed):
```
pip install -e .
```

With the Coral Edge TPU `pycoral` dependency (Coral branch):
```
pip install -e .[pycoral]
```

The real `pycoral~=2.0` package is only published on Coral's own index, so add
`--extra-index-url https://google-coral.github.io/py-repo/` when installing the `pycoral` extra.

As a pinned dependency:
```
pip install "uo-tpu-adapters[pycoral] @ git+ssh://git@github.com/Mahomed2904/UOTPU.git@v0.1.0"
```
