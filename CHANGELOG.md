# Changelog

All notable changes to **nx_tflite_mob** are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/spec/v2.0.0.html).

---

## [0.0.4]

### Changed
- **Docs rewrite.** This release is docs-only — no source-level
  changes. Anyone landing on
  [hexdocs.pm/nx_tflite_mob](https://hexdocs.pm/nx_tflite_mob/0.0.4)
  can now follow end-to-end without having to ask "is this an Nx
  backend?" or "how do I decode the bytes?".

  - `@moduledoc` rewritten with a clear "this is NOT an Nx backend"
    lede + per-platform delegate sections (Core ML, NNAPI, XNNPACK,
    Metal-planned) + explicit input/output byte-layout tables for
    common models + an "optional Nx interop" section showing how to
    compose with `EMLX.Backend` / `Nx.from_binary/3` if desired.
  - `README.md` rewritten as the landing doc. Drops the stale "NIF
    loads inside Mob's running BEAM ⏸ blocked" status (resolved
    end-to-end in 0.0.3 via mob_dev's static-NIF integration). Adds
    "Two ways to use it" — `mix mob.enable tflite` for Mob apps vs
    `make {android,ios_device,ios_sim,mac}` for standalone Elixir
    apps. Per-platform perf table now includes the measured iPhone
    SE A15 / Moto BXM-8-256 numbers.
  - New guide: `guides/yolo_walkthrough.md` — complete YOLOv8n
    end-to-end walkthrough from model acquisition to bounding boxes,
    with the full per-stage timing breakdown that took our Android
    live-YOLO loop from 0.5 FPS to 3.9 FPS. Includes the pure-BEAM
    INT8 decoder (130× faster than the equivalent Nx.BinaryBackend
    decode) and the camera-format choice rationale per platform.
  - New guide: `guides/delegates.md` — picking a delegate per
    platform. Documents the "INT8 + Core ML doesn't work" trap
    (0/256 nodes delegate), how to discover NNAPI accelerators on
    Android, why `mtk-neuron_shim` NPU loses to `mtk-gpu_shim` GPU
    for YOLO-class models (post-processing CPU fallback dominates),
    and when to pick XNNPACK CPU even when GPU/NPU is available.
  - `docs()` config now groups extras into "Guides" and "Build
    recipes" sidebar sections.
  - Tightened function `@doc`s on `load_module/2`, `call/2`,
    `release_module/1` — explicit about input/output byte semantics +
    error conditions.

### Notes
- No code changes. The 0.0.3 NIF binary is bit-identical to the
  0.0.4 NIF. Upgrade is a no-op for `mix.lock`.

## [0.0.3]

### Added
- **Mac arm64 host build** for testing. `make mac` produces
  `priv/mac/libtflite_nif.so` linked against a locally-built
  `libtensorflowlite_c.dylib` (since TFLite has no Mac arm64 prebuilt
  distribution). Configurable via `MAC_TFLITE_DIR` (defaults to
  `~/.mob/cache/tflite-2.16.1-mac_arm64`).
- **Test suite**: `test/test_helper.exs` + `test/nx_tflite_mob_test.exs`
  with 16 tests covering module shape + package metadata (smoke tier,
  always runs) and load_module / call / release_module / opt
  normalisation (integration tier, auto-skipped when the host NIF
  isn't built — keeps `mix test` green for users who only deploy to
  phones).
- **Test fixture**: `test/fixtures/add.bin` — a 544-byte TFLite model
  (`output = 3*input`, 1×8×8×3 float32) lifted from the upstream
  `tensorflow/lite/testdata/` set. Used by the integration tests to
  prove the NIF executes a real model end-to-end on the host.
- **`docs/build_mac_tflite.md`** — reproducible recipe for building the
  Mac `libtensorflowlite_c.dylib` from TF v2.16.1 source via CMake.
  Documents the `std::abs<T>` libc++ patch and the
  `CMAKE_POLICY_VERSION_MINIMUM=3.5` env-var workaround for CMake-4
  compatibility with TF's older `cmake_minimum_required` declarations.

### Changed
- `c_src/tflite_nif.c` — `__APPLE__` branches now refined with
  `TARGET_OS_IPHONE || TARGET_OS_SIMULATOR` so Mac host builds skip the
  iOS framework-style headers + Core ML delegate. Mac builds use the
  same flat-path include layout as Android. No effect on iOS or
  Android binaries.

### Notes
- The Mac build is for **host-side testing only**. The dylib is not
  packaged into the Hex release; production phone builds use the
  prebuilt Android AAR + iOS xcframework as before.
- Tests run automatically in CI on macOS once the dylib is in
  `~/.mob/cache/`. The cache step is a per-CI-runner one-time setup;
  `make mac` reuses the cache afterwards.

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
