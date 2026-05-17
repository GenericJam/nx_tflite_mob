# nx_tflite_mob

Cross-platform TensorFlow Lite NIF for Mob apps. Loads a `.tflite`
model, attaches the right per-platform delegate (NNAPI on Android,
CoreML on iOS), and runs inference. Same Elixir API, same `.tflite`
model file on both OSes.

On the Moto G Power 5G (2024) / Dimensity 7020 / IMG PowerVR BXM-8-256
this hits **155 ms YOLOv8n forward via the MediaTek `mtk-gpu_shim`
accelerator** — the headline that finally unlocks the chip after both
NxVulkan (3.9s) and IREE (1.7s) stopped short.

## Platforms

| Target        | Delegate path                                | Output                                  |
|---------------|----------------------------------------------|-----------------------------------------|
| android_arm64 | XNNPACK CPU / NNAPI (vendor GPU + NPU HALs)  | `priv/android_arm64/libtflite_nif.{so,a}` |
| ios_device    | XNNPACK CPU / CoreML (ANE) / Metal (GPU)     | `priv/ios_device/libtflite_nif.a`        |
| ios_sim       | XNNPACK CPU (no ANE on simulator)            | `priv/ios_sim/libtflite_nif.a`           |

## Status

| Surface | State |
|---|---|
| **Standalone Android `bench` CLI**: YOLOv8n via TFLite XNNPACK CPU INT8 | ✅ **273 ms** mean / 202 ms min |
| **Standalone Android `bench` CLI**: YOLOv8n via TFLite + NNAPI(`mtk-gpu_shim`) | ✅ **155 ms** mean / 118 ms min |
| **Standalone Android `bench` CLI**: YOLOv8n via TFLite + NNAPI(`mtk-neuron_shim`) NPU | 618 ms (partial NPU coverage, falls back) |
| **C NIF compiles + cross-compiles for Android arm64** | ✅ `priv/android/libtflite_nif.so` (~16 KB) |
| **NIF loads inside Mob's running BEAM** | ⏸ blocked: same `enif_*` namespace isolation we hit in [nx_iree_mob](https://github.com/GenericJam/nx_iree_mob) |

## The full perf ladder on a budget Moto BXM-8-256

| Stack | Median | vs original |
|---|---|---|
| Original NxVulkan (hand-rolled Vulkan compute) | 3.9 s | 1× |
| IREE CPU (f32, LLVM autovectorized) | 2.07 s | 1.9× |
| TFLite XNNPACK CPU (f32) | 525 ms | 7.4× |
| TFLite XNNPACK INT8 (QDQ, f32 input) | 411 ms | 9.5× |
| TFLite XNNPACK full_integer_quant (INT8 input) | 273 ms | 14× |
| **TFLite + NNAPI → `mtk-gpu_shim`** | **155 ms** | **25×** |
| TFLite + NNAPI → `mtk-neuron_shim` (APU NPU) | 618 ms | 6× — only partial op coverage |

**~155 ms = 6.5 FPS sustained on a budget Android, fully GPU-accelerated through the device's NNAPI HAL.** This is the number Mob+Android+YOLO has been chasing across three backends.

## Why TFLite + NNAPI wins where NxVulkan + IREE Vulkan didn't

The BXM-8-256's PowerVR driver caps at **Vulkan 1.1**. IREE's Vulkan
HAL requires **Vulkan 1.3 + timeline semaphores + scalarBlockLayout +
synchronization2**. So our direct Vulkan compute path was blocked at
the runtime baseline check, regardless of how good the shaders were.

NNAPI sidesteps that — it routes through MediaTek's own NN HAL driver
(`mtk-gpu_shim`), which talks to the PowerVR using vendor-specific code
paths that don't go through the public Vulkan 1.3 surface.

**Lesson:** for non-flagship Android phones, the cleanest "use the GPU"
path isn't to write Vulkan compute yourself — it's to compile to
TFLite, run with NNAPI, and let the vendor's HAL choose the kernel.

## Where the NPU went

`mtk-neuron_shim` exists and is the actual APU/MDLA, but TFLite + NNAPI
running YOLOv8n on it lands at 618 ms — slower than the GPU path. The
NPU only natively supports a subset of ops (mostly the conv-shaped
ones); YOLO's post-processing (concat / reshape / non-max suppression)
falls back to CPU with cross-device buffer transfers. The roundtrip
swamps any per-op speedup.

A model designed end-to-end for the APU (no reshape/concat in the
inference graph) would land much faster on `mtk-neuron_shim`. Doesn't
apply to YOLOv8n as exported.

## Standalone `bench` CLI

The Android benchmark used to produce all the numbers above lives in
`scripts/bench_android/` (a single C file `bench.c` + the TFLite AAR's
`.so` and headers). Recipe:

```bash
# Pull TFLite 2.16.1 AAR (smaller than building TFLite from source).
mkdir -p /tmp/tflite && cd /tmp/tflite
curl -sLO https://repo1.maven.org/maven2/org/tensorflow/tensorflow-lite/2.16.1/tensorflow-lite-2.16.1.aar
unzip -q tensorflow-lite-2.16.1.aar -d aar
# Patch in two missing headers (AAR bug):
bash -c '
for path in "tensorflow/lite/core/c/registration_external.h" \
            "tensorflow/lite/core/async/c/types.h"; do
  mkdir -p "aar/headers/$(dirname $path)"
  curl -sL -o "aar/headers/$path" \
    "https://raw.githubusercontent.com/tensorflow/tensorflow/v2.16.1/$path"
done
'

# Cross-compile bench.c (in this repo's scripts/bench_android/).
ANDROID_NDK=/path/to/ndk/27.2.12479018
$ANDROID_NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android29-clang \
  -O2 -I aar/headers \
  bench.c aar/jni/arm64-v8a/libtensorflowlite_jni.so \
  -ldl -llog \
  -o bench

# Push + run on phone.
adb push bench libtensorflowlite_jni.so yolov8n.tflite input_int8.bin /data/local/tmp/tflite/
adb shell "cd /data/local/tmp/tflite && LD_LIBRARY_PATH=. ./bench \
  yolov8n_full_integer_quant.tflite input_int8.bin nnapi:mtk-gpu_shim"

# List available NNAPI accelerators on the device:
adb shell "cd /data/local/tmp/tflite && LD_LIBRARY_PATH=. ./bench list-nnapi"
```

## The Mob NIF integration gap (same as nx_iree_mob)

Loading the NIF .so into Mob's running BEAM hits the same Bionic
linker-namespace isolation that NxIree did:

```
dlopen failed: cannot locate symbol "enif_open_resource_type"
   referenced by libtflite_nif.so in namespace clns-7
```

The launcher (`libnxeigen_probe.so` for our test app) does export all
176 `enif_*` symbols (`llvm-nm -D` confirms), but a NIF loaded
dynamically into the app's private namespace can't reach them. Same
fix patterns:

* **Mob static-NIF integration (recommended)** — extend `mob_dev`'s
  rustler/Zig pipeline to handle a C-NIF-with-extra-libs entry. The C
  source + `libtensorflowlite_jni.so` get linked into the app
  launcher binary, ERTS symbols resolve at link time. Same pattern as
  `nx_vulkan`.
* **Pre-load `libtensorflowlite_jni.so` from the app launcher's `JNI_OnLoad`**
  so it's already in the global namespace when our NIF tries to load.

Until either path lands, the standalone bench CLI is the way to measure
the perf number, and the Elixir-side wrapper (`lib/nx_tflite_mob.ex`)
is design-only for Mob integration.

## Layout

```
c_src/tflite_nif.c          — NIF: load_module, call, release_module
lib/nx_tflite_mob.ex        — Elixir API + the @on_load NIF loader stub
Makefile                    — Android arm64 cross-compile
priv/android/libtflite_nif.so — built artifact (after `make android`)
scripts/bxm_tflite_sweep.sh — reproduce the full perf table on a device
docs/perf_history.md        — the per-stack numbers + analysis
```

## License

Apache 2.0.
