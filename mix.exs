defmodule NxTfliteMob.MixProject do
  use Mix.Project

  @version "0.0.3"
  @source_url "https://github.com/GenericJam/nx_tflite_mob"

  def project do
    [
      app: :nx_tflite_mob,
      version: @version,
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: "TensorFlow Lite NIF for Mob apps — INT8 YOLO on Android NPU/GPU at real-time",
      package: package(),
      source_url: @source_url,
      docs: docs()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:nx, "~> 0.10"},
      # Hex publishing builds + uploads hexdocs from this dep. Dev-only,
      # not loaded at runtime — zero impact on downstream Mob apps.
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
      licenses: ["Apache-2.0"],
      links: %{
        "GitHub" => @source_url,
        "Changelog" => "#{@source_url}/blob/main/CHANGELOG.md"
      },
      files: ~w(lib c_src Makefile mix.exs README.md CHANGELOG.md LICENSE)
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: [
        "README.md",
        "CHANGELOG.md",
        "docs/build_mac_tflite.md"
      ],
      source_url: @source_url
    ]
  end
end
