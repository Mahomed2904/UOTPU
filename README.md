# uo-tpu-adapters

Python adapters for running detection/classification/segmentation models on **Hailo** and
**Google Coral Edge TPU** hardware behind one shared result type, so application code can target
either accelerator without depending on their (very different) native SDKs directly.

**This library is not specific to `UnderwaterObservserApp`.** It grew out of that project (a
reef-monitoring device that runs on-device fish detection on Raspberry Pi + Hailo/Coral hardware),
but nothing here depends on underwater imagery, fish detection, or that application's pipeline.
Anything that needs to run inference on a Hailo TPU or a Coral Edge TPU — a different embedded
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

  Used by `UnderwaterObservserApp`'s `HailoTPUObjectDetector`, but the `Interpreter` and adapters
  have no dependency on that class.

- **`uo_tpu_adapters.pycoral`** — Coral Edge TPU support: `adapters.detect` /
  `adapters.classify` / `adapters.segment`, plus `utils` (dataset/EdgeTPU helpers), `learn`
  (on-device transfer learning: backprop softmax regression, weight imprinting), and `pipeline`
  (multi-model pipelined runner). This mirrors Google's `pycoral` adapter API but returns the
  same `common.Object`/`BBox` types as the Hailo side. **Not** importable as top-level `pycoral`
  — its code does `from pycoral.pybind import _pywrap_coral`, an absolute import of the real,
  separately installed `pycoral~=2.0` package's compiled extension. That real package stays a
  genuine runtime dependency (install with the `pycoral` extra below) rather than being replaced.

  Used by `UnderwaterObservserApp`'s `EdgeTPUObjectDetector`, with the same independence from that
  class as the Hailo side.

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

## Origin

Originally three separate projects (`uo-common-types-lib`, `uo-hailo-adapters-lib`,
`uo-pycoral-adapters-lib`), merged after determining the split added repo/packaging overhead
without benefit — `hailo` and `pycoral` both hard-depend on `common` and are never versioned or
installed independently of it. Both, in turn, trace back to
`UnderwaterObservserApp/src/main/libs/{common,hailo,pycoral}/`, where they originated as vendored,
in-tree code, and were pulled out into their own package so they can be reused and versioned
independently of that application.
