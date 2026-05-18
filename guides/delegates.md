# Picking a delegate

A TFLite "delegate" is the runtime that actually executes the model
graph. TFLite ships several; picking the right one for your platform
+ model combination is the single biggest perf lever.

This guide lays out the decision matrix and explains the quirks we
discovered measuring real hardware.

## Quick decision tree

```
Are you on iOS (any iPhone/iPad)?
├── Model is FP16 or FP32?  → delegate: "coreml"        (24 ms iPhone SE A15)
└── Model is INT8?          → delegate: "xnnpack"       (27 ms iPhone SE A15)
                              (NOT coreml — 0/256 nodes delegate)

Are you on Android?
├── MediaTek Dimensity?     → delegate: "nnapi", accelerator: "mtk-gpu_shim"
├── Qualcomm Snapdragon?    → delegate: "nnapi", accelerator: "qti-gpu"
├── Pixel?                  → delegate: "nnapi", accelerator: "google-edgetpu"
├── Unknown OEM?            → delegate: "xnnpack"       (always works)
└── Want to discover?       → see "Discovering NNAPI accelerators" below

Mac / Linux dev host?       → delegate: "xnnpack"
```

## Per-delegate detail

### `xnnpack` — CPU+SIMD

Bundled into TFLite. Cross-platform. Default when no other delegate
is set explicitly.

Highly-optimised: tuned ARM NEON / Intel AVX kernels, INT8 + FP32 +
quantized-int8 paths.

| Pro | Con |
|---|---|
| Works everywhere TFLite runs | No GPU / NPU |
| Reproducible numbers (no thermal throttle) | Slower than accelerator paths when those work |
| No vendor-driver dependencies | |

Options:

* `num_threads:` — CPU thread count (default 6). Up to physical core
  count helps; oversubscription hurts.

On the Moto G Power 5G (Dimensity 7020), XNNPACK matched the GPU
delegate at **77 ms** for YOLOv8n. On modern phones with strong CPU
cores, XNNPACK is a competitive default.

### `nnapi` — Android Neural Networks API

Android's neural-net dispatch layer. Each device's vendor ships a
HAL driver (`libmtk-gpu-shim.so`, `libqti-gpu.so`, etc.); NNAPI picks
one based on the `accelerator` name (or by default, badly — see
below).

Options:

* `accelerator:` — vendor HAL name string. **Always pass this
  explicitly.** NNAPI's auto-selection on at least one MediaTek device
  picks the NPU which is 5× SLOWER than the GPU for YOLO-class models.
* `allow_fp16:` — let the HAL promote FP32 ops to FP16 (default
  `true`). Lossy but typically fine for inference.

#### Discovering NNAPI accelerators on a connected device

The standalone `bench` CLI (in `scripts/bench_android/`) has a
`list-nnapi` mode:

```bash
adb push scripts/bench_android/bench /data/local/tmp/
adb push ~/.mob/cache/tflite-2.16.1-android_arm64/jni/arm64-v8a/libtensorflowlite_jni.so /data/local/tmp/
adb shell 'cd /data/local/tmp && LD_LIBRARY_PATH=. ./bench list-nnapi'
```

Output is a list of accelerator names available on this device:

```
mtk-gpu_shim
mtk-neuron_shim
nnapi-reference  (CPU emulation — slow)
```

#### Known accelerator perf rankings (YOLOv8n on Moto G Power 5G)

| Accelerator | Median |
|---|---|
| `mtk-gpu_shim` | 75-117 ms (best — MediaTek's PowerVR HAL) |
| `xnnpack` CPU | 77-91 ms (tied with GPU; deterministic) |
| `mtk-neuron_shim` | 355 ms (NPU — slower because YOLO post-processing falls back to CPU) |
| `nnapi-reference` | 358 ms (CPU emulation — never use) |
| `nnapi` (no accelerator) | 358 ms (defaults to mtk-neuron_shim — never do this) |

The NPU loses despite being a "real" neural-net accelerator because
YOLO has `concat` + `reshape` ops in its post-processing that aren't
in the NPU's supported set. TFLite falls back to CPU for those ops
mid-graph, with cross-device buffer transfers between each fallback.
The transfer overhead swamps any per-op NPU speedup.

A model designed end-to-end for the APU (no reshape/concat in the
inference graph) would land much faster on `mtk-neuron_shim`. YOLOv8n
as exported doesn't fit.

### `coreml` — Apple Core ML

Routes the delegated portion of the graph through Apple's Core ML
framework, which internally schedules to the **Apple Neural Engine**
when ops are supported on devices that have one (A11+ iPhones).

Options:

* `coreml_ane_only:` — when `true`, `load_module/2` returns
  `{:error, _}` instead of falling back to CPU on devices without an
  ANE. Useful for "ANE-only or skip" logic. Default `false`.

#### Op-coverage caveat — the INT8 trap

**Don't use Core ML with INT8 models.** The Ultralytics
`yolov8n_full_integer_quant.tflite` export uses INT8 quantization ops
that Core ML's tooling doesn't translate to ANE primitives. The
result: **0 out of 256 nodes delegated**, and the whole model falls
back to CPU which is slower than just running XNNPACK directly.

For Core ML you want the **FP16 or FP32 model variant**:

| Model | Core ML delegation rate | Latency (iPhone SE A15) |
|---|---|---|
| INT8 | 0/256 (0%) — full CPU fallback | 45 ms (don't use) |
| FP16 | 214/385 (56%) | 23-25 ms |
| FP32 | 214/254 (84%) | 24-25 ms |

FP16 and FP32 hit the same wall-clock because the delegated portion
is the same (214 conv-shaped ops). FP16 wins on bundle size (~6 MB
vs ~12 MB).

The 30% of nodes that fall to CPU on FP16 are the post-processing
ops (concat / reshape / NMS-prep) — same shape as the Android NPU
problem. Core ML handles the boundary more gracefully than NNAPI
NPU does (cheap shared-memory transitions on Apple silicon), which
is why this works at all.

### `metal` — Apple Metal GPU (planned)

TFLite ships `TensorFlowLiteCMetal.xcframework` with a Metal GPU
delegate, but the current NIF doesn't expose it as a `delegate:`
option. PR welcome.

Core ML is usually faster than Metal on Apple Silicon since it can
pick ANE for supported ops + Metal as a fallback. Metal-only is
mainly useful for older devices without an ANE.

## Comparing the paths on the same device

Same iPhone SE 3rd gen A15, same `.tflite` model files, varying the
delegate:

| Variant | Delegate | Delegation | Min / Median / Max |
|---|---|---|---|
| INT8 | xnnpack | n/a (CPU+NEON) | 27 / 36 / 37 ms |
| INT8 | coreml | 0/256 (full fallback) | 36 / 39 / 42 ms |
| FP16 | xnnpack | n/a (CPU+NEON) | 86 / 98 / 265 ms |
| **FP16** | **coreml** | 214/385 (56%) | **23 / 25 / 26 ms** |
| FP32 | coreml | 214/254 (84%) | 24 / 24 / 25 ms |

The standout: **FP16 + Core ML wins** at 25 ms median. Half the
bundle of FP32 with identical wall-clock. The CPU+NEON XNNPACK path
is impressive at 36 ms — for context, our standalone bench
measurements show it consistently within 30 ms of the GPU/ANE paths
on modern phones.

## Composing with `Nx` backends

TFLite delegates handle the model graph. The pre/post-processing in
your Elixir code is separate compute that you can route to a
different backend:

```elixir
# Input prep on EMLX (Metal GPU on iOS) — useful for batch
# transformations, scaling, normalization.
input_bytes =
  camera_bytes
  |> Nx.from_binary(:f32, backend: {EMLX.Backend, device: :gpu})
  |> Nx.reshape({1, 640, 640, 3})
  |> Nx.divide(255.0)
  |> Nx.to_binary()

# Model inference on TFLite + Core ML → ANE
{:ok, [out]} = NxTfliteMob.call(handle, [input_bytes])

# Output decode on EMLX again
out
|> Nx.from_binary(:f32, backend: {EMLX.Backend, device: :gpu})
|> Nx.reshape({1, 84, 8400})
|> ...
```

Two distinct compute paths, one screen. The TFLite delegate doesn't
care what backend your Nx code uses — it sees only the bytes you
hand to `call/2`.

## When `xnnpack` is the right answer even when GPU/NPU is available

* Deterministic numbers (CPU paths don't thermal-throttle as
  aggressively)
* Cold-start (delegate init for Core ML / NNAPI is 100-500 ms)
* Tiny models (the delegate dispatch overhead dominates inference
  for sub-ms models)
* Cross-platform parity for tests
