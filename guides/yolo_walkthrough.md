# YOLOv8n end-to-end: from camera frame to bounding boxes

A complete worked example of using `nx_tflite_mob` to run live YOLOv8n
object detection on a Mob app, with the full timing breakdown of
where the milliseconds go.

The numbers in this guide are measured on:

* **iPhone SE 3rd gen** (Apple A15 + Neural Engine)
* **Moto G Power 5G (2024)** (MediaTek Dimensity 7020 + IMG PowerVR
  BXM-8-256)

## Why YOLO as the example

YOLO is hard. Real-time object detection chews through GPU/NPU
budget, the post-processing has tricky byte layouts, and the
`.tflite` model exists in several variants (INT8 / FP16 / FP32) with
different perf+accuracy tradeoffs. If you can do YOLO end-to-end, you
can do most things.

## 1. Get the model

We're using the **Ultralytics YOLOv8n** export — 3.4 MB INT8 (or 6 MB
FP16) trained on COCO.

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=tflite int8=True imgsz=640
# → yolov8n_saved_model/yolov8n_full_integer_quant.tflite
# → yolov8n_saved_model/yolov8n_float16.tflite
# → yolov8n_saved_model/yolov8n_float32.tflite
```

The script also produces `yolov8n_int8.tflite` (different quant
flavour — symmetric int8 weights, asymmetric int8 activations) and
`yolov8n_integer_quant.tflite` (post-training quant without
calibration). For our purposes the `full_integer_quant` (uniform INT8
with calibration) is the right choice — best size + best NNAPI
delegation rate on Android.

Drop the model into your Mob app:

    mkdir -p priv/yolo
    cp yolov8n_saved_model/yolov8n_full_integer_quant.tflite priv/yolo/
    cp yolov8n_saved_model/yolov8n_float16.tflite priv/yolo/

The COCO class labels you'll need for display:

```elixir
@coco ~w(
  person bicycle car motorcycle airplane bus train truck boat
  traffic-light fire-hydrant stop-sign parking-meter bench bird cat
  dog horse sheep cow elephant bear zebra giraffe backpack umbrella
  handbag tie suitcase frisbee skis snowboard sports-ball kite
  baseball-bat baseball-glove skateboard surfboard tennis-racket
  bottle wine-glass cup fork knife spoon bowl banana apple sandwich
  orange broccoli carrot hot-dog pizza donut cake chair couch
  potted-plant bed dining-table toilet tv laptop mouse remote
  keyboard cell-phone microwave oven toaster sink refrigerator book
  clock vase scissors teddy-bear hair-drier toothbrush
)
```

## 2. Pick a model variant per platform

The platform-aware default-opts helper:

```elixir
defmodule MyApp.YoloOpts do
  @doc "Returns `{model_file, opts}` for the best TFLite path on this device."
  def best do
    case :mob_nif.platform() do
      :ios ->
        # Core ML delegate hits the A15 Neural Engine. FP16 model gets
        # 56% of nodes delegated (24-25 ms median). INT8 + Core ML
        # delegates 0 nodes — don't use it on iOS.
        {"yolov8n_float16.tflite", [delegate: "coreml", coreml_ane_only: false]}

      :android ->
        # NNAPI's mtk-gpu_shim is the MediaTek GPU HAL on Dimensity
        # chips (~75-117 ms). Without an explicit accelerator name,
        # NNAPI defaults to mtk-neuron_shim NPU which is 5× SLOWER
        # for YOLO (post-processing falls back to CPU with cross-device
        # transfers). Always pass an explicit accelerator string.
        {"yolov8n_full_integer_quant.tflite",
         [delegate: "nnapi", accelerator: "mtk-gpu_shim", allow_fp16: true]}

      _ ->
        # Mac dev / Linux dev — XNNPACK CPU+SIMD path. Works
        # everywhere; surprisingly competitive (~75 ms on the same
        # Android chip).
        {"yolov8n_full_integer_quant.tflite", [delegate: "xnnpack"]}
    end
  end
end
```

The Android `mtk-gpu_shim` string is MediaTek-specific. For other OEMs
substitute:

| OEM | Accelerator string |
|---|---|
| MediaTek (Dimensity) | `mtk-gpu_shim`, `mtk-neuron_shim` |
| Qualcomm (Snapdragon) | `qti-gpu`, `qti-dsp` |
| Samsung (Exynos) | `samsung-gpu` |
| Google (Pixel) | `google-edgetpu` |

See [the delegates guide](delegates.html) for how to discover
accelerators on a connected device.

## 3. Load the model

Load once at app start (or screen mount). Don't reload per-inference
— the delegate-init cost is ~few hundred ms on first load.

```elixir
defmodule MyApp.YoloScreen do
  use Mob.Screen
  require Logger

  def mount(_params, _session, socket) do
    {model_file, opts} = MyApp.YoloOpts.best()
    priv = :code.priv_dir(:my_app) |> to_string()
    model_path = Path.join([priv, "yolo", model_file])

    handle =
      case NxTfliteMob.load_module(File.read!(model_path), opts) do
        {:ok, h} -> h
        {:error, reason} ->
          Logger.error("YOLO model load failed: \#{inspect(reason)}")
          nil
      end

    socket =
      socket
      |> Mob.Socket.assign(:tflite, handle)
      |> Mob.Socket.assign(:detections, [])
      |> Mob.Permissions.request(:camera)

    {:ok, socket}
  end
end
```

## 4. Hook up the camera

YOLOv8n's INT8 export expects `1×640×640×3 INT8` NHWC input. The FP16
export expects `1×640×640×3 FP32` NHWC input (FP16 is the weight
precision, not the input — the model casts internally).

The pre-frame conversion is the dominant non-inference cost — pick a
camera format that minimises it:

```elixir
def handle_info({:permission, :camera, :granted}, socket) do
  format =
    case socket.assigns.quant do
      :int8 -> :bgra_u8   # 4 bytes/px; reorder + subtract 128 → INT8
      :fp16 -> :rgb_f32   # 12 bytes/px; pass straight through
    end

  socket =
    socket
    |> Mob.Camera.start_preview(facing: :back)
    |> Mob.Camera.start_frame_stream(
         width: 640, height: 640,
         format: format,
         facing: :back,
         throttle_ms: 40)

  {:noreply, socket}
end
```

`:rgb_f32` is heavier (4.9 MB per frame) but converts to FP16 model
input with zero work — the model accepts FP32 directly. The 0 ms
conversion makes it the winning choice on iOS.

`:bgra_u8` is lighter (1.6 MB per frame) and converts to INT8 via
pure byte arithmetic (drop alpha, swap BGR→RGB, subtract 128). About
15 ms in BEAM on the Moto BXM.

## 5. The inference call

```elixir
def handle_info({:camera, :frame, %{bytes: bytes}}, socket) do
  t0 = System.monotonic_time(:millisecond)

  input = prepare_input(socket.assigns.quant, bytes)
  t_call = System.monotonic_time(:millisecond)

  case NxTfliteMob.call(socket.assigns.tflite, [input]) do
    {:ok, [output_bytes]} ->
      t_decode = System.monotonic_time(:millisecond)
      detections = decode_output(socket.assigns.quant, output_bytes)
      t_end = System.monotonic_time(:millisecond)

      Logger.debug(
        "yolo: conv=\#{t_call - t0}ms call=\#{t_decode - t_call}ms " <>
        "dec=\#{t_end - t_decode}ms detections=\#{length(detections)}"
      )

      {:noreply, Mob.Socket.assign(socket, :detections, detections)}

    {:error, reason} ->
      Logger.error("yolo: call failed: \#{inspect(reason)}")
      {:noreply, socket}
  end
end

# :int8 → camera bgra_u8 → model int8 NHWC (drop alpha, BGR→RGB, -128)
defp prepare_input(:int8, bgra_bytes) do
  for <<b, g, r, _a <- bgra_bytes>>, into: <<>> do
    <<r - 128::signed-8, g - 128::signed-8, b - 128::signed-8>>
  end
end

# :fp16 → camera rgb_f32 passes through (FP16 model takes FP32 input)
defp prepare_input(:fp16, rgb_f32_bytes), do: rgb_f32_bytes
```

## 6. Decode the output

YOLOv8n's output is `1×84×8400` (4 box coords + 80 class probs ×
8400 anchors). On INT8 models it's INT8 bytes with quantization
scale + zero_point. On FP16/FP32 models it's FP32 with box coords
normalised to [0, 1] (Ultralytics convention).

Skip Nx for this — the BinaryBackend argmax over `{80, 8400}` was
**1700 ms** in our profiling; a pure-BEAM byte loop is **13 ms**:

```elixir
@n_anchors 8400
@n_classes 80
@scale 0.00659          # INT8 quant scale (from the .tflite metadata)
@zero -128              # INT8 zero_point

defp decode_output(:int8, bytes) do
  conf_int8_threshold = trunc(0.25 / @scale) + @zero

  Enum.reduce(0..(@n_anchors - 1), [], fn anchor, acc ->
    {class_id, max_v} = max_class_int8(bytes, anchor, 0, 0, -129)

    if max_v > conf_int8_threshold do
      [build_candidate_int8(bytes, anchor, class_id, max_v) | acc]
    else
      acc
    end
  end)
  |> Enum.sort_by(& &1.confidence, :desc)
  |> Enum.take(20)
  |> nms(0.45)
end

# Scan 80 class bytes for `anchor`. Class probs occupy
# bytes[4*8400 + cls*8400 + anchor] for cls in 0..79.
defp max_class_int8(_bytes, _anchor, @n_classes, max_cls, max_v),
  do: {max_cls, max_v}

defp max_class_int8(bytes, anchor, cls, max_cls, max_v) do
  raw = :binary.at(bytes, 4 * @n_anchors + cls * @n_anchors + anchor)
  v = if raw > 127, do: raw - 256, else: raw  # uint8 -> int8

  if v > max_v do
    max_class_int8(bytes, anchor, cls + 1, cls, v)
  else
    max_class_int8(bytes, anchor, cls + 1, max_cls, max_v)
  end
end

defp build_candidate_int8(bytes, anchor, class_id, max_v) do
  cx = int8_to_f32(:binary.at(bytes, anchor))
  cy = int8_to_f32(:binary.at(bytes, @n_anchors + anchor))
  w  = int8_to_f32(:binary.at(bytes, 2 * @n_anchors + anchor))
  h  = int8_to_f32(:binary.at(bytes, 3 * @n_anchors + anchor))

  %{
    class_id: class_id,
    confidence: int8_to_f32(max_v),
    x1: cx - w / 2, y1: cy - h / 2,
    x2: cx + w / 2, y2: cy + h / 2
  }
end

defp int8_to_f32(b) do
  signed = if b > 127, do: b - 256, else: b
  (signed - @zero) * @scale
end

# Non-max suppression. Already top-20, so the O(N²) loop is cheap.
defp nms([], _), do: []
defp nms([best | rest], iou_th) do
  rest = Enum.reject(rest, &(iou(best, &1) > iou_th))
  [best | nms(rest, iou_th)]
end

defp iou(a, b) do
  ix1 = max(a.x1, b.x1); iy1 = max(a.y1, b.y1)
  ix2 = min(a.x2, b.x2); iy2 = min(a.y2, b.y2)
  inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
  union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
  if union <= 0, do: 0.0, else: inter / union
end
```

The FP32 decoder is the same shape but reads 4-byte floats and
skips dequantization. Box coords need a `× 640` scale because
Ultralytics FP32 normalizes them to [0, 1].

## 7. Where the milliseconds go

End-to-end per-frame breakdown on a **Moto G Power 5G** (best case
after warmup):

| Stage | Time | What it does |
|---|---|---|
| `prepare_input/2` (bgra→int8) | 15 ms | BEAM bitstring comprehension over 410k pixels |
| `NxTfliteMob.call/2` | 75-117 ms | TFLite + NNAPI → MediaTek GPU HAL |
| `decode_output/2` + NMS | 13 ms | Pure-BEAM byte scan over 8400×84 |
| **Total** | **~110-150 ms** | **6-9 FPS sustained** |

On **iPhone SE 3rd gen**:

| Stage | Time | What it does |
|---|---|---|
| `prepare_input/2` (rgb_f32 passthrough) | 0 ms | Pass camera bytes straight to the FP16 model |
| `NxTfliteMob.call/2` | 25-40 ms | TFLite + Core ML → Apple Neural Engine |
| `decode_output/2` + NMS | 60-85 ms | Pure-BEAM FP32 decode (slower than INT8 — 4-byte float reads vs 1-byte) |
| **Total** | **~90-125 ms** | **8-11 FPS sustained** |

Notable: the FP32 decoder is the iOS bottleneck. The Apple Neural
Engine call itself is 25 ms — the same as the standalone bench. The
remaining 60-85 ms is BEAM byte processing of the f32 output tensor.
Future optimization: a tiny NIF helper for the float-scan
(`:binary.at` + 4-byte pattern matches are the slow part).

## 8. Optional Nx interop

If you prefer `Nx` for the prep/decode (slower but more declarative):

```elixir
# Input prep via Nx (FP16 path on iOS — no quant math needed)
input =
  camera_rgb_f32_bytes
  |> Nx.from_binary(:f32, backend: Nx.BinaryBackend)
  |> Nx.reshape({1, 640, 640, 3})
  |> Nx.to_binary()

{:ok, [out]} = NxTfliteMob.call(handle, [input])

# Output decode via Nx
out
|> Nx.from_binary(:f32, backend: Nx.BinaryBackend)
|> Nx.reshape({1, 84, 8400})
|> ...  # box decode + NMS
```

The `backend:` option is critical — `Nx.BinaryBackend` is pure Elixir
and very slow for million-element ops. Using `EMLX.Backend` for the
Nx side on iOS speeds this up substantially:

```elixir
input =
  camera_rgb_f32_bytes
  |> Nx.from_binary(:f32, backend: {EMLX.Backend, device: :gpu})
  |> ...
```

This composes — `EMLX.Backend` for the Nx-side math, `NxTfliteMob`
for the TFLite-side model call. Two distinct compute paths in the
same screen.

## See also

* [delegates guide](delegates.html) — picking the right delegate per
  platform, accelerator discovery
* [docs/build_mac_tflite.md](build_mac_tflite.html) — building
  `libtensorflowlite_c.dylib` from source for Mac host tests
* The
  [LiveYoloScreen](https://github.com/GenericJam/nxeigen_probe/blob/main/lib/nxeigen_probe/live_yolo_screen.ex)
  in the `nxeigen_probe` reference app for a working version of all
  of the above
