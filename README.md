# uo-tpu-adapters

TPU detection adapters for UnderwaterObservserApp, consolidated into one package since the
pieces are always installed and used together:

- `uo_tpu_adapters.common` — shared `Object`/`BBox` detection types.
- `uo_tpu_adapters.hailo` — Hailo TPU interpreter and detection/classification/segmentation
  adapters. Used by `HailoTPUObjectDetector`.
- `uo_tpu_adapters.pycoral` — Coral Edge TPU detect/classify/segment adapters. Used by
  `EdgeTPUObjectDetector`. **Not** importable as top-level `pycoral` — its own code does
  `from pycoral.pybind import _pywrap_coral`, an absolute import of the real, separately
  installed `pycoral~=2.0` package's compiled extension, which stays a real dependency (install
  this project with the `pycoral` extra: `pip install -e .[pycoral]`) rather than being
  replaced.

Originally three separate projects (`uo-common-types-lib`, `uo-hailo-adapters-lib`,
`uo-pycoral-adapters-lib`), merged after determining the split added repo/packaging overhead
without benefit — `hailo` and `pycoral` both hard-depend on `common` and are never versioned or
installed independently of it. Both, in turn, trace back to
`UnderwaterObservserApp/src/main/libs/{common,hailo,pycoral}/`, where they originated as
vendored, in-tree code.

## Install

Base install (Hailo branch — no pycoral runtime needed):
```
pip install -e .
```

With the Coral Edge pycoral dependency (Coral branch):
```
pip install -e .[pycoral]
```
