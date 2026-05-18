defmodule NxTfliteMobTest do
  use ExUnit.Case, async: true

  # ── Tests that don't need a built NIF ──────────────────────────────────────
  # (pure-Elixir surface — module shape, public API existence)

  describe "module surface" do
    test "NxTfliteMob exports the three documented functions" do
      assert function_exported?(NxTfliteMob, :load_module, 1)
      assert function_exported?(NxTfliteMob, :load_module, 2)
      assert function_exported?(NxTfliteMob, :call, 2)
      assert function_exported?(NxTfliteMob, :release_module, 1)
    end

    test "NxTfliteMob.NIF declares the three NIF stubs" do
      # These exist as `:erlang.nif_error/1`-raising stubs when the NIF
      # isn't loaded — proves the module structure matches what
      # ERL_NIF_INIT registered against.
      assert function_exported?(NxTfliteMob.NIF, :load_module, 2)
      assert function_exported?(NxTfliteMob.NIF, :call, 2)
      assert function_exported?(NxTfliteMob.NIF, :release_module, 1)
    end
  end

  describe "package metadata" do
    test "mix.exs declares an Apache-2.0 licence" do
      project = NxTfliteMob.MixProject.project()
      assert project[:package][:licenses] == ["Apache-2.0"]
    end

    test "version is a valid SemVer 0.x.y string" do
      project = NxTfliteMob.MixProject.project()
      assert Regex.match?(~r/^0\.\d+\.\d+$/, project[:version])
    end

    test "source_url points at the GenericJam/nx_tflite_mob GitHub repo" do
      project = NxTfliteMob.MixProject.project()
      assert project[:source_url] == "https://github.com/GenericJam/nx_tflite_mob"
    end

    test "depends on :nx" do
      deps = NxTfliteMob.MixProject.project()[:deps]
      assert Enum.any?(deps, fn {dep, _} -> dep == :nx end)
    end
  end

  # ── Integration tests (need the built NIF) ─────────────────────────────────
  # All require priv/native/libtflite_nif.so + libtensorflowlite_c.dylib to be
  # available on the host. test_helper.exs excludes :integration when the NIF
  # isn't built so `mix test` still passes the smoke tier.

  @add_fixture Path.expand("fixtures/add.bin", __DIR__)

  describe "load_module/2 (integration)" do
    @describetag :integration

    test "loads a valid .tflite model and returns a reference handle" do
      bytes = File.read!(@add_fixture)
      assert {:ok, handle} = NxTfliteMob.load_module(bytes, [])
      assert is_reference(handle)
      :ok = NxTfliteMob.release_module(handle)
    end

    test "returns an error for an invalid model body" do
      assert {:error, msg} = NxTfliteMob.load_module(<<0, 0, 0, 0>>, [])
      assert is_list(msg)
    end

    test "accepts xnnpack as an explicit delegate (default behaviour)" do
      bytes = File.read!(@add_fixture)
      assert {:ok, handle} = NxTfliteMob.load_module(bytes, delegate: "xnnpack")
      :ok = NxTfliteMob.release_module(handle)
    end
  end

  describe "call/2 (integration)" do
    @describetag :integration

    setup do
      bytes = File.read!(@add_fixture)
      {:ok, handle} = NxTfliteMob.load_module(bytes, [])
      on_exit(fn -> NxTfliteMob.release_module(handle) end)
      {:ok, handle: handle}
    end

    test "add.bin computes output = 3*input on a 1x8x8x3 float32 tensor", %{handle: handle} do
      # Input: 192 float32s all = 1.0 (1*8*8*3 NHWC).
      ones_f32 = for _ <- 1..192, into: <<>>, do: <<1.0::float-32-native>>
      assert byte_size(ones_f32) == 768

      assert {:ok, [out]} = NxTfliteMob.call(handle, [ones_f32])
      assert byte_size(out) == 768

      # First three output values should be 3.0 (i.e. input + input + input).
      <<v1::float-32-native, v2::float-32-native, v3::float-32-native, _::binary>> = out
      assert_in_delta v1, 3.0, 0.0001
      assert_in_delta v2, 3.0, 0.0001
      assert_in_delta v3, 3.0, 0.0001
    end

    test "call returns an error when the input byte-size is wrong", %{handle: handle} do
      # Model wants 768 bytes; we give 4.
      assert {:error, msg} = NxTfliteMob.call(handle, [<<0, 0, 0, 0>>])
      assert msg |> to_string() =~ "input"
    end

    test "call returns an error when the input list length doesn't match input-tensor count",
         %{handle: handle} do
      # Model has 1 input tensor; we pass 2 binaries.
      assert {:error, msg} = NxTfliteMob.call(handle, [<<>>, <<>>])
      assert msg |> to_string() =~ "input count"
    end
  end

  describe "release_module/1 (integration)" do
    @describetag :integration

    test "release on a fresh handle returns :ok" do
      {:ok, handle} = NxTfliteMob.load_module(File.read!(@add_fixture), [])
      assert :ok = NxTfliteMob.release_module(handle)
    end

    test "double-release is idempotent (returns :ok both times)" do
      {:ok, handle} = NxTfliteMob.load_module(File.read!(@add_fixture), [])
      assert :ok = NxTfliteMob.release_module(handle)
      # Releasing again is a no-op since the second call sees a zeroed
      # resource (the dtor + explicit release_module/1 both NULL the
      # interp/opts/model pointers).
      assert :ok = NxTfliteMob.release_module(handle)
    end
  end

  describe "opt normalisation (integration — exercises proplist conversion)" do
    @describetag :integration

    test "boolean opts get stringified before reaching the NIF" do
      # allow_fp16 is a bool option (consumed by NNAPI on Android). On Mac
      # XNNPACK ignores it but the proplist still has to be encodable —
      # if normalize/1 broke the conversion this would crash badarg.
      bytes = File.read!(@add_fixture)
      assert {:ok, handle} = NxTfliteMob.load_module(bytes, allow_fp16: true)
      :ok = NxTfliteMob.release_module(handle)
    end

    test "atom opts get stringified (e.g. delegate: :xnnpack instead of \"xnnpack\")" do
      bytes = File.read!(@add_fixture)
      assert {:ok, handle} = NxTfliteMob.load_module(bytes, delegate: :xnnpack)
      :ok = NxTfliteMob.release_module(handle)
    end
  end
end
