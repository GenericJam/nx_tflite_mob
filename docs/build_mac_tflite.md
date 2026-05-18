# Building `libtensorflowlite_c.dylib` for Mac arm64

TFLite has no Mac arm64 prebuilt distribution. To run host-side
integration tests (`make mac && mix test`), build the C library from
TensorFlow's source via CMake — this is a one-time ~10-15 min step;
the output dylib is then cached at `~/.mob/cache/`.

## Prereqs

```bash
brew install cmake      # CMake 3.16+ required; tested with 4.2.1
# Apple Clang (any recent Xcode CLT version)
```

## Build

```bash
# 1. Shallow-clone TensorFlow at the version we pin everywhere else.
git clone --depth=1 --branch v2.16.1 \
  https://github.com/tensorflow/tensorflow.git ~/code/tensorflow_src

# 2. Patch one source file — Apple Clang 21+ libc++ dropped the
#    explicit-instantiation form of std::abs<T>. Replace
#    `std::abs<float>` with a lambda; same for std::abs<int32_t>.
#    The exact patch is documented inline below.
```

In `tensorflow/lite/kernels/elementwise.cc`, around the `AbsEval`
function (line ~290), change:

```cpp
case kTfLiteFloat32:
  return EvalImpl<float>(context, node, std::abs<float>, type);
// ...
case kTfLiteInt32:
  return EvalImpl<int32_t>(context, node, std::abs<int32_t>, type);
```

to:

```cpp
case kTfLiteFloat32:
  return EvalImpl<float>(
      context, node, [](float x) { return std::abs(x); }, type);
// ...
case kTfLiteInt32:
  return EvalImpl<int32_t>(
      context, node, [](int32_t x) { return std::abs(x); }, type);
```

```bash
# 3. Configure + build with the CMake-4 compat env var (propagates to
#    sub-projects' ExternalProject downloads, several of which use
#    `cmake_minimum_required(VERSION 2.x or 3.0)` and would otherwise
#    fail).
mkdir -p /tmp/tflite_build_mac && cd /tmp/tflite_build_mac
export CMAKE_POLICY_VERSION_MINIMUM=3.5
cmake ~/code/tensorflow_src/tensorflow/lite/c -DCMAKE_BUILD_TYPE=Release
cmake --build . --target tensorflowlite_c -j$(sysctl -n hw.ncpu)
```

Output: `/tmp/tflite_build_mac/libtensorflowlite_c.dylib` (~5 MB).

## Install to cache

`make mac` looks for the dylib + headers under
`~/.mob/cache/tflite-2.16.1-mac_arm64/` by default (override with
`MAC_TFLITE_DIR=…`). Install layout:

```bash
mkdir -p ~/.mob/cache/tflite-2.16.1-mac_arm64/{lib,include/tensorflow/lite/{c,core/{c,async/c}}}

cp /tmp/tflite_build_mac/libtensorflowlite_c.dylib \
   ~/.mob/cache/tflite-2.16.1-mac_arm64/lib/

cp ~/code/tensorflow_src/tensorflow/lite/*.h \
   ~/.mob/cache/tflite-2.16.1-mac_arm64/include/tensorflow/lite/

cp ~/code/tensorflow_src/tensorflow/lite/c/*.h \
   ~/.mob/cache/tflite-2.16.1-mac_arm64/include/tensorflow/lite/c/

cp ~/code/tensorflow_src/tensorflow/lite/core/*.h \
   ~/.mob/cache/tflite-2.16.1-mac_arm64/include/tensorflow/lite/core/

cp ~/code/tensorflow_src/tensorflow/lite/core/c/*.h \
   ~/.mob/cache/tflite-2.16.1-mac_arm64/include/tensorflow/lite/core/c/

cp ~/code/tensorflow_src/tensorflow/lite/core/async/c/*.h \
   ~/.mob/cache/tflite-2.16.1-mac_arm64/include/tensorflow/lite/core/async/c/
```

## Verify

```bash
cd ~/code/nx_tflite_mob
make mac          # produces priv/mac/libtflite_nif.so (~50 KB)
cp priv/mac/libtflite_nif.so priv/native/libtflite_nif.so
mix test          # 16 tests, 0 failures
```

## Why this is host-test-only

The dylib produced here links against `@rpath/libtensorflowlite_c.dylib`
with the rpath pointing into `~/.mob/cache/`. It's not packaged into the
Hex release — production phone builds use the prebuilt Android AAR /
iOS xcframework. Mac-host builds exist solely so `mix test` can exercise
the NIF locally before shipping.

## Future work

When LiteRT (Google's TFLite successor) lands on Maven Central +
CocoaPods, the per-platform `make {android, ios_device, ios_sim}`
targets become candidates for migration. As of v2.1.4 LiteRT only ships
via PyPI (Mac/Linux) and GitHub-Releases-headers + Bazel — no
Android/iOS-consumable artifacts, so the cross-platform story isn't
ready for migration. See the `nx_tflite_mob` repo's `CHANGELOG.md`
"future work" notes.
