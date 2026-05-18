# nx_tflite_mob

**Call TensorFlow Lite models from Elixir / BEAM, with full vendor
accelerator access on phones — Apple Neural Engine on iOS, MediaTek /
Qualcomm GPU+NPU HALs on Android.** Same `.tflite` model file works on
all platforms.

```elixir
{:nx_tflite_mob, "~> 0.0.3"}
```

## Important: this is not an Nx backend

`NxTfliteMob` does **not** replace `Nx.BinaryBackend` / `EMLX.Backend`
/ `NxVulkan.Backend`. You can't do
`Nx.global_default_backend(NxTfliteMob.Backend)` — no such module
exists.

| | Nx backend (EMLX, NxVulkan, NxEigen, BinaryBackend) | `NxTfliteMob` |
|---|---|---|
| What you write | `Nx.dot`, `Nx.conv`, etc. (composable ops) | `load_module(model_bytes)` then `call(handle, inputs)` |
| What runs | Each op dispatches to backend | The whole pre-compiled model graph executes through a vendor delegate |
| Best for | Custom tensor math, arbitrary inference | Pre-trained models exported to `.tflite` — YOLO, MobileNet, MoveNet, etc. |
| Apple Neural Engine | Indirect via MLX | Direct via Core ML delegate ⭐ |
| Android vendor NPU/GPU | Not available | Direct via NNAPI delegate ⭐ |
| Compose with Nx code | n/a — it IS Nx | Yes — input prep + output decode can use `Nx.from_binary/3` |

Use `NxTfliteMob` when you have a pre-trained model. Use Nx backends
when you're writing arbitrary tensor code in Elixir.

## 30-second quickstart

```elixir
# 1. Load a .tflite model.
tflite = File.read!("priv/yolov8n_float16.tflite")
{:ok, handle} = NxTfliteMob.load_module(tflite,
  delegate: "coreml",          # or "nnapi", "xnnpack"
  coreml_ane_only: false
)

# 2. Inference — bytes in, bytes out (model-specific shape + dtype).
{:ok, [output_bytes]} = NxTfliteMob.call(handle, [input_bytes])

# 3. Free the model when done.
:ok = NxTfliteMob.release_module(handle)
```

The model file is the standard TFLite FlatBuffer format. Anything
exportable from Ultralytics, MediaPipe, TF, JAX (via the AI Edge
Toolkit), or PyTorch (via `ai-edge-torch`) works.

See **[the YOLO walkthrough](guides/yolo_walkthrough.md)** for a
complete worked example with input prep, NMS, and on-device perf
breakdown.

## Two ways to use it

### A. With Mob (mobile apps) — recommended

If you're building a Mob app, install via mob_dev's Igniter task:

```bash
mix mob.enable tflite
```

This runs once and:

1. Adds `{:nx_tflite_mob, "~> 0.0.3"}` + `{:nx, "~> 0.10"}` to deps.
2. Generates `lib/<your_app>/tflite_init.ex` with per-platform default
   delegate opts (`coreml` on iOS, `nnapi` + `mtk-gpu_shim` on Android).
3. Registers `:tflite_nif` in mob_dev's static-NIF table behind the
   `MOB_STATIC_TFLITE_NIF` guard.
4. Next `mix mob.deploy --native` automatically:
   - Downloads `tensorflow-lite-2.16.1.aar` (Android) or
     `TensorFlowLiteC-2.17.0.tar.gz` (iOS) into `~/.mob/cache/`
   - Cross-compiles `libtflite_nif.a` per arch
   - Links it into your app's main native binary
   - Drops `libtensorflowlite_jni.so` into `jniLibs/<abi>/` (Android)
     or links the framework statically into the app binary (iOS)

Then in code:

```elixir
{:ok, h} = NxTfliteMob.load_module(model_bytes, MyApp.TfliteInit.default_opts())
```

Requires `mob_dev >= 0.5.9` from
[hex](https://hex.pm/packages/mob_dev).

### B. Standalone (any Elixir app)

If you're not using Mob, build the NIF for your target host yourself:

```bash
git clone https://github.com/GenericJam/nx_tflite_mob.git
cd nx_tflite_mob

# Pick your target — see Makefile for options.
make android       # → priv/android_arm64/libtflite_nif.{so,a}
make ios_device    # → priv/ios_device/libtflite_nif.a
make ios_sim       # → priv/ios_sim/libtflite_nif.a
make mac           # → priv/mac/libtflite_nif.so (Mac-host tests)
```

The Mac build requires you to first build
`libtensorflowlite_c.dylib` from TF source — see
[docs/build_mac_tflite.md](docs/build_mac_tflite.md). TFLite has no
Mac arm64 prebuilt distribution (Android + iOS are prebuilt, the
Makefile points at known cache locations for those).

## Per-platform perf — measured on real hardware

| Device | Hardware | Model | Delegate | Latency |
|---|---|---|---|---|
| iPhone SE 3rd gen | A15 + ANE | YOLOv8n FP16 | Core ML → ANE | **24 ms** |
| iPhone SE 3rd gen | A15 | YOLOv8n INT8 | XNNPACK CPU+NEON | 37 ms |
| Moto G Power 5G | Dimensity 7020 + PowerVR BXM-8-256 | YOLOv8n INT8 | NNAPI / `mtk-gpu_shim` | **75-117 ms** |
| Moto G Power 5G | Dimensity 7020 + PowerVR BXM-8-256 | YOLOv8n INT8 | XNNPACK CPU+NEON | 77 ms |
| Moto G Power 5G | MediaTek APU/MDLA | YOLOv8n INT8 | NNAPI / `mtk-neuron_shim` | 355 ms (post-processing CPU fallback) |

Numbers above are inference-call latency (median of 5 runs, after
warmup). Live-camera screens with input prep + output decode in BEAM
add 30-80 ms of overhead per frame — see the walkthrough for the
per-stage timing breakdown that took our Android live-YOLO loop from
0.5 FPS to 3.9 FPS.

The headline:
**Apple Neural Engine via TFLite Core ML beats EMLX (~30 ms via MLX→ANE)
by ~20% on this model**, because Core ML's compiler is more aggressive
about ANE op coverage than the MLX→Metal→ANE path. Same `.tflite` model
file, both numbers from the same iPhone.

## What's in the package

| | |
|---|---|
| `lib/nx_tflite_mob.ex` | Elixir API — three public functions |
| `c_src/tflite_nif.c` | C NIF wrapping the TFLite C API |
| `Makefile` | Cross-compile per platform |
| `test/fixtures/add.bin` | 544-byte TFLite model (`output = 3*input`) for tests |
| `guides/` | YOLO walkthrough, delegate selection guide |
| `docs/` | Build recipes (Mac host build) |

The NIF (`libtflite_nif.{so,a}`) is **not** in the published Hex
release — the package ships source, and the consumer's `make` (or
mob_dev's auto-build) produces per-platform binaries against
platform-appropriate TFLite distributions.

## Architecture detail: how it works on each platform

### Android

The Maven Central `tensorflow-lite-2.16.1.aar` ships
`libtensorflowlite_jni.so` for arm64-v8a + armv7a. mob_dev's
`MobDev.TfliteDownloader` extracts it into `~/.mob/cache/` and
`MobDev.TfliteNif` cross-compiles `tflite_nif.c` against the AAR's
headers via the Android NDK. The resulting `libtflite_nif.a` is
statically linked into your app's main native lib alongside the BEAM,
so the NIF init function is resolvable at app launch (no `dlopen`,
which Bionic's `RTLD_LOCAL` would block).

NNAPI is part of Android. The delegate routes through whichever
vendor HAL is installed — MediaTek's `libmtk-gpu-shim.so`, Qualcomm's
`libqti-gpu.so`, etc. `accelerator_name` selects which one.

### iOS

CocoaPods CDN ships `TensorFlowLiteC-2.17.0.tar.gz` from `dl.google.com`
with `.xcframework` slices for ios-arm64 + ios-arm64_x86_64-simulator.
The framework binaries are unusual — they're MH_OBJECT (relocatable
object files, `filetype=1`), not MH_DYLIB. The linker pulls them
statically into your app's main Mach-O at build time. They do NOT need
to be embedded as runtime `.framework` bundles in the `.app` — trying
to embed them trips iOS install on missing `Info.plist` (CocoaPods
generates them) and then on "code signature version no longer
supported" (codesign only makes v3 sigs for MH_EXECUTE/MH_DYLIB).

Core ML delegate routes through Apple's Core ML framework, which
internally schedules to the Apple Neural Engine when ops are
supported.

### Mac (host tests only)

TFLite has no Mac arm64 prebuilt distribution — we tried every channel
(`pip install tflite-runtime`, `ai-edge-litert`'s wheel, MediaPipe's
wheel, TensorFlow's wheel, the iOS xcframework simulator slice) and
none yields a usable `libtensorflowlite_c.dylib`. Workaround: build it
from TF source via CMake (focused target, ~10-15 min one-time, cached
afterwards). See [docs/build_mac_tflite.md](docs/build_mac_tflite.md).

Mac is host-tests-only; the dylib is not packaged into the published
Hex release. Production phone builds use the prebuilt AAR + xcframework.

## Status

| Surface | State |
|---|---|
| Hex release | ✅ Published at [hex.pm/packages/nx_tflite_mob](https://hex.pm/packages/nx_tflite_mob) |
| HexDocs | ✅ [hexdocs.pm/nx_tflite_mob](https://hexdocs.pm/nx_tflite_mob) |
| Android arm64 | ✅ via `make android` or `mob_dev`'s `mix mob.enable tflite` |
| iOS arm64 (device) | ✅ via `make ios_device` or mob_dev |
| iOS arm64 (simulator) | ✅ via `make ios_sim` or mob_dev |
| Mac arm64 (host tests) | ✅ via `make mac` (build dylib first per `docs/build_mac_tflite.md`) |
| Tests (16 integration + smoke) | ✅ `mix test` against real `.tflite` |
| End-to-end in Mob's running BEAM | ✅ verified live: 24ms iPhone SE / 117ms Moto BXM |

## Versions pinned

| Distribution | Version | Source |
|---|---|---|
| Android AAR | `2.16.1` | Maven Central `org.tensorflow:tensorflow-lite` |
| iOS xcframework | `2.17.0` | `dl.google.com` (CocoaPods upstream) |
| Mac (CMake-built) | `2.16.1` | TensorFlow source v2.16.1 |

These ship under different version pins because of upstream packaging
differences (Android's last AAR was 2.16.1; iOS's CocoaPod is at
2.17.0). The TFLite C API is binary-stable across this range — same
`.tflite` model file loads + runs identically on either version.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Acknowledgements

Built on top of:

* **TensorFlow Lite** — Apache 2.0, Google
  [`tensorflow/lite`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite)
* **Mob** — the BEAM-on-device mobile framework this package was
  built for
  [`GenericJam/mob`](https://github.com/GenericJam/mob)
* **Nx** — interop is optional but the type system makes pre/post
  processing pleasant
  [`elixir-nx/nx`](https://github.com/elixir-nx/nx)
