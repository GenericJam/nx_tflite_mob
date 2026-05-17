defmodule NxTfliteMob.MixProject do
  use Mix.Project

  @version "0.0.1"
  @source_url "https://github.com/GenericJam/nx_tflite_mob"

  def project do
    [
      app: :nx_tflite_mob,
      version: @version,
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description:
        "TensorFlow Lite NIF for Mob apps — INT8 YOLO on Android NPU/GPU at real-time",
      package: package(),
      source_url: @source_url
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:nx, "~> 0.10"}
    ]
  end

  defp package do
    [
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url}
    ]
  end
end
