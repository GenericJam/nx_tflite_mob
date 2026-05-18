# Most tests in this package require the host NIF to be built. We don't
# auto-build it here (the Makefile is the SoT for compile rules) — we
# just exclude integration tests when the .so isn't present.

priv_dir = :code.priv_dir(:nx_tflite_mob) |> to_string()
nif_path = Path.join(priv_dir, "native/libtflite_nif.so")

excludes =
  if File.regular?(nif_path) do
    []
  else
    IO.puts("""

    [nx_tflite_mob] No NIF at #{nif_path} — skipping :integration tests.
    Build the Mac NIF (and its TFLite dylib dep) first:

        make mac

    See docs/build_mac_tflite.md for the libtensorflowlite_c.dylib step
    (it has to be built from source — TFLite has no Mac arm64 prebuilt
    distribution).

    """)

    [:integration]
  end

ExUnit.start(exclude: excludes)
