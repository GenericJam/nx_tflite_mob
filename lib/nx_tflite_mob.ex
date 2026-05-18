defmodule NxTfliteMob do
  @moduledoc """
  Call TensorFlow Lite models from Elixir / BEAM, with full vendor
  accelerator access on phones — Apple Neural Engine on iOS, MediaTek
  / Qualcomm GPU+NPU HALs on Android.

  ## This is NOT an `Nx.Backend`

  `NxTfliteMob` does not replace `Nx.BinaryBackend`, `EMLX.Backend`,
  `NxVulkan.Backend`, etc. There is no `NxTfliteMob.Backend` module
  to set via `Nx.global_default_backend/1`.

  TFLite executes pre-compiled model graphs (`.tflite` files)
  end-to-end through vendor-optimised delegates. The whole graph
  stays opaque so the delegate can fuse + schedule it for ANE / GPU
  / NPU. You don't compose your own ops here — you call a pre-trained
  model.

  Use `NxTfliteMob` when you have a pre-trained model to run. Use
  Nx backends when you're writing arbitrary tensor math in Elixir.

  Both can coexist in the same app.

  ## API surface

  Three functions: `load_module/2`, `call/2`, `release_module/1`.

      iex> tflite = File.read!("priv/yolov8n_float16.tflite")
      iex> {:ok, handle} = NxTfliteMob.load_module(tflite,
      ...>   delegate: "coreml", coreml_ane_only: false)
      iex> {:ok, [output_bytes]} = NxTfliteMob.call(handle, [input_bytes])
      iex> NxTfliteMob.release_module(handle)
      :ok

  See [the YOLO walkthrough](yolo_walkthrough.html) for a complete
  end-to-end example with input prep, inference, and output decode.

  ## Delegate options

  The `delegate` opt selects how the model graph runs. Per-platform
  recommendations (see also [the delegates guide](delegates.html)):

  ### Android — `delegate: "nnapi"`

  NNAPI is Android's neural-net dispatch API. It picks a vendor HAL
  driver based on the `accelerator` name:

  | `accelerator:` value | What it routes to |
  |---|---|
  | `"mtk-gpu_shim"` | MediaTek's GPU HAL — fastest for YOLO on Dimensity chips |
  | `"mtk-neuron_shim"` | MediaTek's APU/NPU — only worthwhile if your graph is pure conv (no concat/reshape post-processing — TFLite falls back to CPU for those, transfer overhead dominates) |
  | `"qti-gpu"` | Qualcomm Snapdragon GPU |
  | `"google-edgetpu"` | Pixel TPU |
  | `nil` (no key) | NNAPI auto-picks — often the WRONG choice for YOLO (defaults to NPU on MediaTek, which is 5× slower) |

  Discover available accelerators on a connected device with
  `adb shell` + the standalone `bench` CLI's `list-nnapi` mode (see
  the package's `scripts/bench_android/`).

  Other Android opts:

    * `num_threads:` — XNNPACK CPU thread count (default 6)
    * `allow_fp16:` — let NNAPI run FP32 ops in FP16 (default `true`)

  ### iOS — `delegate: "coreml"`

  Core ML routes the delegated portion through Apple's Core ML
  framework, which internally schedules to the Apple Neural Engine
  when ops are supported. For YOLOv8n FP16, ~56% of nodes delegate
  to the ANE on an iPhone SE 3rd gen A15 (the rest fall to CPU via
  XNNPACK), hitting **24 ms** per inference.

  Caveats:

    * **INT8 + Core ML doesn't work.** Core ML's tooling doesn't
      understand the Ultralytics INT8 quant flavour — 0/256 nodes
      delegate. Use the FP16 model variant for Core ML.
    * `coreml_ane_only:` (default `false`) — when `true`, the
      delegate returns `nil` instead of falling back to CPU on
      devices without an ANE. Useful for "ANE-only or skip" logic;
      irrelevant on A11+ devices where the ANE is always present.

  ### iOS — `delegate: "metal"` (planned)

  TFLite ships `TensorFlowLiteCMetal.xcframework` for Metal GPU
  inference but the current NIF doesn't expose it as a `delegate:`
  option yet. PR welcome. Core ML is usually faster anyway on Apple
  Silicon devices (Core ML can pick GPU when ANE ops are unsupported).

  ### XNNPACK CPU — `delegate: "xnnpack"` (default)

  Bundled into TFLite. Highly-optimised CPU+SIMD path. Default when
  no other delegate is set. Surprisingly competitive on modern phones
  — ~77 ms on the Moto G Power 5G (tied with the GPU path) and 27-37
  ms on iPhone SE 3rd gen A15. Use this when:

    * You're on a device without GPU/NPU acceleration
    * The vendor delegate fails to delegate (e.g. INT8 + Core ML)
    * You want deterministic, reproducible numbers (CPU paths don't
      thermal-throttle as aggressively as GPUs)

  ## Input + output byte layout

  `call/2` is raw-bytes-in, raw-bytes-out. **The byte layout is
  model-specific** — you have to match what the `.tflite` model
  expects.

  Inspect a model's expected shape/dtype via `mix` Python helpers or
  TFLite's `flatc` tool. Or `:erlang.load_nif/2` an inspector NIF
  built against TFLite's `TfLiteInterpreterGetInputTensor` —
  exposing this in the Elixir API is on the roadmap.

  Common shapes:

  | Model | Input | Output |
  |---|---|---|
  | YOLOv8n INT8 (Ultralytics full_integer_quant) | 1×640×640×3 INT8 NHWC (`1228800` bytes) | 1×84×8400 INT8 (`705600` bytes) |
  | YOLOv8n FP16 (Ultralytics float16) | 1×640×640×3 FP32 NHWC (`4915200` bytes — the FP16 model accepts FP32 input that's cast internally) | 1×84×8400 FP32 normalised (`2822400` bytes) |
  | YOLOv8n FP32 | 1×640×640×3 FP32 NHWC | 1×84×8400 FP32 |
  | MobileNetV2 (ImageNet) | 1×224×224×3 FP32 NHWC | 1×1001 FP32 (class logits) |

  See the YOLO walkthrough for the layout-aware decoder we use in
  production (pure-BEAM, 13 ms for the full INT8 NMS pass).

  ## Where Nx fits in (optionally)

  You CAN use Nx tensors on either side of `call/2`. It's optional —
  bytes-in/bytes-out is the canonical interface.

  Input prep with Nx:

      input_bytes =
        camera_frame_f32_binary
        |> Nx.from_binary(:f32)
        |> Nx.reshape({1, 640, 640, 3})
        |> Nx.as_type(:s8)      # quantize for INT8 model
        |> Nx.to_binary()

      {:ok, [out]} = NxTfliteMob.call(handle, [input_bytes])

  Output decode with Nx:

      detections =
        out
        |> Nx.from_binary(:s8)
        |> Nx.reshape({1, 84, 8400})
        |> Nx.as_type(:f32)
        |> Nx.multiply(scale)
        |> Nx.subtract(zero_point)
        |> extract_detections()

  In practice we bypass Nx for performance-critical decoding —
  `Nx.BinaryBackend` for an argmax across `{80, 8400}` is 1700 ms;
  a pure-BEAM `:binary.at/2` loop is 13 ms (130× faster). See
  `NxeigenProbe.LiveYoloScreen` for the pure-BEAM decoder pattern.

  ## Using with Mob

  If you're building a [Mob](https://github.com/GenericJam/mob) app,
  the easiest path is mob_dev's Igniter task:

      mix mob.enable tflite

  This adds the dep + generates a per-platform default-opts helper
  and registers the NIF in mob_dev's static-NIF table. Requires
  `mob_dev >= 0.5.9`. See mob_dev's
  [`mob.enable` docs](https://hexdocs.pm/mob_dev/Mix.Tasks.Mob.Enable.html)
  for details.

  After `mix mob.enable tflite`, the auto-generated helper picks
  delegate opts per platform:

      {:ok, h} = NxTfliteMob.load_module(model_bytes,
                   MyApp.TfliteInit.default_opts())

  ## Building from source (non-Mob)

  See the package's `Makefile` — targets `android`, `ios_device`,
  `ios_sim`, `mac`. Each requires platform-appropriate TFLite
  distribution (cached at `~/.mob/cache/` by mob_dev's downloader, or
  per-target overrides for standalone builds).

  Mac builds require building `libtensorflowlite_c.dylib` from TF
  source first — TFLite has no Mac arm64 prebuilt. See
  `docs/build_mac_tflite.md` in the repo.
  """

  alias NxTfliteMob.NIF

  @typedoc """
  Opaque handle to a loaded TFLite model. Pass to `call/2` and free
  with `release_module/1`. Closed handles also get freed when garbage
  collected, but explicit release is recommended for short-lived
  inferences.
  """
  @type module_handle :: reference()

  @doc """
  Load a TFLite model from raw `.tflite` FlatBuffer bytes.

  Returns `{:ok, handle}` on success or `{:error, message}` if the
  bytes aren't a valid TFLite model or delegate creation fails.

  ## Options

  All options are documented in detail in the moduledoc:

    * `:delegate` (string) — `"xnnpack"` (default), `"nnapi"` (Android),
      `"coreml"` (iOS)
    * `:accelerator` (string) — vendor accelerator name for NNAPI
      (e.g. `"mtk-gpu_shim"`)
    * `:num_threads` (integer) — XNNPACK CPU thread count (default 6)
    * `:allow_fp16` (boolean) — NNAPI FP32→FP16 promotion (default
      `true`)
    * `:coreml_ane_only` (boolean) — Core ML requires ANE (default
      `false` — falls back to CPU/GPU)

  ## Examples

      # XNNPACK CPU (cross-platform default)
      {:ok, h} = NxTfliteMob.load_module(tflite_bytes, [])

      # Android NNAPI → MediaTek GPU HAL
      {:ok, h} = NxTfliteMob.load_module(tflite_bytes,
                   delegate: "nnapi",
                   accelerator: "mtk-gpu_shim",
                   allow_fp16: true)

      # iOS Core ML → ANE
      {:ok, h} = NxTfliteMob.load_module(tflite_bytes,
                   delegate: "coreml",
                   coreml_ane_only: false)
  """
  @spec load_module(binary(), keyword()) ::
          {:ok, module_handle()} | {:error, String.t() | charlist()}
  def load_module(model_bytes, opts \\ []) when is_binary(model_bytes) do
    NIF.load_module(model_bytes, normalize(opts))
  end

  @doc """
  Run inference on a loaded model.

  `inputs` is a list of binaries — one per input tensor in the model's
  declared input order. Each binary must match the model's expected
  shape × dtype byte layout exactly (1×640×640×3 INT8 = 1228800 bytes
  for YOLOv8n full_integer_quant, for example).

  Returns `{:ok, outputs}` where `outputs` is a list of binaries — one
  per output tensor, also in declared order. Decode each according to
  the model's documented output layout.

  ## Examples

      # YOLOv8n INT8 — 1×640×640×3 INT8 input, 1×84×8400 INT8 output
      input = <<…1228800 INT8 bytes…>>
      {:ok, [output]} = NxTfliteMob.call(handle, [input])
      true = byte_size(output) == 705600

  ## Errors

  Returns `{:error, message}` for:

    * Input list length doesn't match the model's input-tensor count
    * Any input binary's size doesn't match the model's expected size
    * The model's `TfLiteInterpreterInvoke` returns non-OK status
  """
  @spec call(module_handle(), [binary()]) ::
          {:ok, [binary()]} | {:error, String.t() | charlist()}
  def call(handle, inputs) when is_reference(handle) and is_list(inputs),
    do: NIF.call(handle, inputs)

  @doc """
  Free the model + delegate + interpreter held by `handle`.

  Idempotent — calling on an already-released handle returns `:ok`
  (the underlying resource is zero'd and re-releasing is a no-op).

  Resources are also freed on GC if `release_module/1` isn't called,
  but explicit release is recommended for tight loops or short-lived
  inferences to keep memory predictable.
  """
  @spec release_module(module_handle()) :: :ok
  def release_module(handle) when is_reference(handle), do: NIF.release_module(handle)

  # Coerce opt values to types the NIF's proplist parser understands
  # (strings, ints, atoms). Bools and atoms become strings.
  defp normalize(opts) do
    Enum.map(opts, fn
      {k, v} when is_boolean(v) -> {k, to_string(v)}
      {k, v} when is_atom(v) -> {k, to_string(v)}
      {k, v} -> {k, v}
    end)
  end
end

defmodule NxTfliteMob.NIF do
  @moduledoc false

  @on_load :load_nifs

  def load_nifs do
    path =
      try do
        case :code.priv_dir(:nx_tflite_mob) do
          {:error, _} -> ~c"libtflite_nif"
          dir when is_list(dir) -> :filename.join(dir, ~c"native/libtflite_nif")
        end
      rescue
        _ -> ~c"libtflite_nif"
      end

    :erlang.load_nif(path, 0)
  end

  def load_module(_bytes, _opts), do: :erlang.nif_error(:nif_not_loaded)
  def call(_h, _inputs), do: :erlang.nif_error(:nif_not_loaded)
  def release_module(_h), do: :erlang.nif_error(:nif_not_loaded)
end
