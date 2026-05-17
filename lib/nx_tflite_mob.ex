defmodule NxTfliteMob do
  @moduledoc """
  TensorFlow Lite NIF for Mob apps.

  Loads `.tflite` model bytes, runs inference via TFLite's bundled
  XNNPACK CPU path or via the NNAPI delegate (which on MediaTek
  devices like the Moto G Power 5G 2024 routes to the
  `mtk-gpu_shim` GPU accelerator and gets us ~150 ms YOLOv8n).

  ## Example

      tflite = File.read!("priv/yolov8n_full_integer_quant.tflite")
      {:ok, m} = NxTfliteMob.load_module(tflite,
                   delegate: "nnapi",
                   accelerator: "mtk-gpu_shim",
                   allow_fp16: true)

      input_int8 = File.read!("priv/input_int8.bin")  # 1x640x640x3 INT8
      {:ok, [out_bin]} = NxTfliteMob.call(m, [input_int8])

      :ok = NxTfliteMob.release_module(m)

  ## Delegates

  * `:delegate` — `"xnnpack"` (default, CPU INT8/FP32) or `"nnapi"`
    (vendor NN HAL, hits the GPU / NPU when present)
  * `:accelerator` — only meaningful with `delegate: "nnapi"`; the
    accelerator name to request. Discover with
    `NxTfliteMob.NIF.list_nnapi_devices/0` (planned). Known values
    on the Moto BXM-8-256:
    * `"mtk-gpu_shim"` — PowerVR GPU through MediaTek NNAPI HAL
      (best result for YOLOv8n)
    * `"mtk-neuron_shim"` — APU NPU; only partial op coverage for
      YOLO, falls back partial
    * `"nnapi-reference"` — NNAPI's CPU emulation (slow)
  * `:num_threads` — XNNPACK CPU thread count (default 6)
  * `:allow_fp16` — NNAPI may run FP32 ops in FP16 (default true)
  """

  alias NxTfliteMob.NIF

  @type module_handle :: reference()

  @spec load_module(binary(), keyword()) ::
          {:ok, module_handle()} | {:error, String.t()}
  def load_module(model_bytes, opts \\ []) when is_binary(model_bytes) do
    NIF.load_module(model_bytes, normalize(opts))
  end

  @spec call(module_handle(), [binary()]) ::
          {:ok, [binary()]} | {:error, String.t()}
  def call(handle, inputs) when is_reference(handle) and is_list(inputs),
    do: NIF.call(handle, inputs)

  @spec release_module(module_handle()) :: :ok
  def release_module(handle) when is_reference(handle), do: NIF.release_module(handle)

  defp normalize(opts) do
    # NIF expects a proplist with atom keys and binary/atom/int/string values.
    # Coerce strings, bools, etc.
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
