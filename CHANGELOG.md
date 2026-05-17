# Changelog

All notable changes to **nx_tflite_mob** are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/spec/v2.0.0.html).

---

## [0.0.2]

### Added
- **iOS support.** `c_src/tflite_nif.c` now compiles for `ios_device`
  (arm64) and `ios_sim` (arm64) via xcrun, alongside the existing
  Android arm64 path. The C NIF picks the right delegate per platform:
  - `__ANDROID__` → NNAPI (`mtk-gpu_shim`, `mtk-neuron_shim`, etc.)
  - `__APPLE__` → Core ML (with optional `coreml_ane_only` for devices
    with an Apple Neural Engine)
  - `xnnpack` (default) on both
- `Makefile` targets: `make ios_device`, `make ios_sim`, `make android`,
  `make all_mobile`. Each produces a per-arch `libtflite_nif.{a,so}`
  under `priv/<target>/`.
- Framework-style includes on iOS (`<TensorFlowLiteC/c_api.h>` resolved
  via `-F`-flagged search paths) vs. flat-path includes on Android
  (`"tensorflow/lite/c/c_api.h"` resolved via `-I`).

### Measured on real hardware
| Device | Path | Inference |
|---|---|---|
| Moto G Power 5G (BXM-8-256) | NNAPI / `mtk-gpu_shim` INT8 | 75-117 ms |
| iPhone SE 3rd gen (A15) | Core ML → ANE (FP16 model) | **24 ms** |

### Notes
- iOS framework binaries (`TensorFlowLiteC.framework/TensorFlowLiteC` etc.)
  ship as Mach-O **MH_OBJECT** (`filetype=1`), not MH_DYLIB. The linker
  pulls them statically into the app's main binary at build time. Do
  NOT embed them as runtime `.framework` bundles in the `.app` — it
  trips iOS install twice (missing per-framework Info.plist, then
  "code signature version no longer supported" since codesign only
  produces v3 signatures for MH_EXECUTE/MH_DYLIB).
- Integration with the [Mob](https://github.com/GenericJam/mob)
  framework happens via `mix mob.enable tflite` in
  [mob_dev](https://hex.pm/packages/mob_dev) ≥ 0.5.8.

## [0.0.1] — 2026-05-16

Initial release.

### Added
- C NIF wrapping the TensorFlow Lite C API: `load_module/2`, `call/2`,
  `release_module/1`.
- Android NNAPI delegate support with accelerator selection
  (`mtk-gpu_shim`, `mtk-neuron_shim`, etc.).
- XNNPACK CPU path (default).
- Standalone Android `bench` CLI (`scripts/bench_android/bench.c`) that
  hit **155 ms YOLOv8n** via NNAPI `mtk-gpu_shim` on the Moto G Power
  5G (2024) — the headline that prompted this package's existence.
