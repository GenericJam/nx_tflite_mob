# nx_tflite_mob — cross-compile tflite_nif.c for desktop & mobile targets.
#
# Two output flavors per target:
#   priv/<target>/libtflite_nif.so   — dynamic .so for standalone iex / bench
#                                       (android only; iOS forbids dlopen)
#   priv/<target>/libtflite_nif.a    — static archive for Mob's static-NIF
#                                       table (linked into launcher binary)
#
# Targets:
#   android         — android_arm64 .so + .a (NNAPI + XNNPACK)
#   ios_device      — ios_device .a (CoreML + Metal + XNNPACK)
#   ios_sim         — ios_sim .a (XNNPACK only — simulator has no ANE)
#   all_mobile      — android + ios_device + ios_sim
#   mac             — mac_arm64 .so for HOST-side tests (XNNPACK only,
#                     against a Bazel/CMake-built libtensorflowlite_c.dylib —
#                     TFLite has no Mac arm64 prebuilt distribution). Build
#                     the dylib via docs/build_mac_tflite.md first.

ERLANG_PATH := $(shell erl -noshell -eval 'io:format("~s/erts-16.4/include", [code:root_dir()])' -s init stop)

# ── Android ────────────────────────────────────────────────────────────────
ANDROID_NDK    ?= /Users/kevin/Library/Android/sdk/ndk/27.2.12479018
NDK_HOST       := darwin-x86_64
NDK_CC         := $(ANDROID_NDK)/toolchains/llvm/prebuilt/$(NDK_HOST)/bin/aarch64-linux-android29-clang
NDK_AR         := $(ANDROID_NDK)/toolchains/llvm/prebuilt/$(NDK_HOST)/bin/llvm-ar
NDK_RANLIB     := $(ANDROID_NDK)/toolchains/llvm/prebuilt/$(NDK_HOST)/bin/llvm-ranlib
TFLITE_AAR     := /tmp/tflite_android/aar
TFLITE_HDRS    := $(TFLITE_AAR)/headers
TFLITE_SO      := $(TFLITE_AAR)/jni/arm64-v8a/libtensorflowlite_jni.so

# ── iOS ────────────────────────────────────────────────────────────────────
IOS_MIN_VER    := 15.0
TFLITE_IOS_DIR := /tmp/tflite_ios/extracted/TensorFlowLiteC-2.17.0/Frameworks
# xcframework slices — one path per slice (device / simulator-arm64).
IOS_DEVICE_FW  := $(TFLITE_IOS_DIR)/TensorFlowLiteC.xcframework/ios-arm64
IOS_SIM_FW     := $(TFLITE_IOS_DIR)/TensorFlowLiteC.xcframework/ios-arm64_x86_64-simulator
IOS_DEVICE_CML := $(TFLITE_IOS_DIR)/TensorFlowLiteCCoreML.xcframework/ios-arm64
IOS_SIM_CML    := $(TFLITE_IOS_DIR)/TensorFlowLiteCCoreML.xcframework/ios-arm64_x86_64-simulator

IOS_DEV_SDK    := $(shell xcrun --sdk iphoneos --show-sdk-path)
IOS_SIM_SDK    := $(shell xcrun --sdk iphonesimulator --show-sdk-path)
IOS_CC         := $(shell xcrun --find clang)
IOS_AR         := $(shell xcrun --find ar)
IOS_RANLIB     := $(shell xcrun --find ranlib)

# ── Mac arm64 (host-side tests) ────────────────────────────────────────────
# Set MAC_TFLITE_DIR to the cache where you installed the dylib + headers.
# Default matches docs/build_mac_tflite.md's recipe.
MAC_TFLITE_DIR ?= $(HOME)/.mob/cache/tflite-2.16.1-mac_arm64
MAC_TFLITE_HDRS := $(MAC_TFLITE_DIR)/include
MAC_TFLITE_LIB  := $(MAC_TFLITE_DIR)/lib/libtensorflowlite_c.dylib
MAC_CC          := $(shell xcrun --find clang)

# ── Common ─────────────────────────────────────────────────────────────────
SRC            := c_src/tflite_nif.c

# Symbol name for the static NIF table. Mob's driver_tab references
# `tflite_nif_init` (see MobDev.StaticNifs default entry).
STATIC_NIF_DEF := -DSTATIC_ERLANG_NIF_LIBNAME=tflite_nif

# ============================================================================
# Targets
# ============================================================================

.PHONY: android ios_device ios_sim mac all_mobile clean

all_mobile: android ios_device ios_sim

# ── Android ────────────────────────────────────────────────────────────────

android: priv/android_arm64/libtflite_nif.so priv/android_arm64/libtflite_nif.a

priv/android_arm64/libtflite_nif.so: $(SRC)
	@mkdir -p $(@D)
	$(NDK_CC) -O2 -Wall -fPIC -shared \
	    -I$(ERLANG_PATH) \
	    -I$(TFLITE_HDRS) \
	    -Wl,--undefined-version \
	    -Wl,-rpath,'$$ORIGIN' \
	    -Wl,--enable-new-dtags \
	    $< $(TFLITE_SO) \
	    -ldl -llog -lm \
	    -o $@
	@echo "built $@"

priv/android_arm64/libtflite_nif.a: $(SRC)
	@mkdir -p $(@D)
	$(NDK_CC) -O2 -Wall -fPIC -c \
	    $(STATIC_NIF_DEF) \
	    -I$(ERLANG_PATH) \
	    -I$(TFLITE_HDRS) \
	    $< -o $(@D)/tflite_nif.o
	$(NDK_AR) rcs $@ $(@D)/tflite_nif.o
	$(NDK_RANLIB) $@
	@echo "built $@ ($$($(NDK_AR) -t $@ | wc -l) members)"
	@$(ANDROID_NDK)/toolchains/llvm/prebuilt/$(NDK_HOST)/bin/llvm-nm $@ | grep tflite_nif_nif_init || (echo "ERROR: tflite_nif_nif_init symbol missing!" && exit 1)

# Backwards-compat: old layout had priv/android/libtflite_nif.so
priv/android/libtflite_nif.so: priv/android_arm64/libtflite_nif.so
	@mkdir -p $(@D) && cp $< $@

# ── iOS device (arm64) ─────────────────────────────────────────────────────

ios_device: priv/ios_device/libtflite_nif.a

priv/ios_device/libtflite_nif.a: $(SRC)
	@mkdir -p $(@D)
	$(IOS_CC) -O2 -Wall -fPIC -c \
	    -arch arm64 \
	    -target arm64-apple-ios$(IOS_MIN_VER) \
	    -isysroot $(IOS_DEV_SDK) \
	    -fembed-bitcode-marker \
	    $(STATIC_NIF_DEF) \
	    -I$(ERLANG_PATH) \
	    -F$(IOS_DEVICE_FW) -F$(IOS_DEVICE_CML) \
	    $< -o $(@D)/tflite_nif.o
	$(IOS_AR) rcs $@ $(@D)/tflite_nif.o
	$(IOS_RANLIB) $@
	@echo "built $@"
	@nm $@ | grep tflite_nif_nif_init || (echo "ERROR: tflite_nif_nif_init symbol missing!" && exit 1)

# ── iOS simulator (arm64 — apple silicon only; no x86_64) ──────────────────

ios_sim: priv/ios_sim/libtflite_nif.a

priv/ios_sim/libtflite_nif.a: $(SRC)
	@mkdir -p $(@D)
	$(IOS_CC) -O2 -Wall -fPIC -c \
	    -arch arm64 \
	    -target arm64-apple-ios$(IOS_MIN_VER)-simulator \
	    -isysroot $(IOS_SIM_SDK) \
	    $(STATIC_NIF_DEF) \
	    -I$(ERLANG_PATH) \
	    -F$(IOS_SIM_FW) -F$(IOS_SIM_CML) \
	    $< -o $(@D)/tflite_nif.o
	$(IOS_AR) rcs $@ $(@D)/tflite_nif.o
	$(IOS_RANLIB) $@
	@echo "built $@"
	@nm $@ | grep tflite_nif_nif_init || (echo "ERROR: tflite_nif_nif_init symbol missing!" && exit 1)

# ── Mac arm64 (host) ───────────────────────────────────────────────────────
# Dynamic .so for `iex -S mix` + `mix test`. Links against the CMake-built
# libtensorflowlite_c.dylib at MAC_TFLITE_DIR (default ~/.mob/cache/...).
# Embedded rpath of $ORIGIN means the .so finds its TFLite dylib if you
# copy it alongside; we also use -Wl,-rpath,<absolute-cache-path> so it
# resolves out of the cache without copying.

mac: priv/mac/libtflite_nif.so

MAC_SDK := $(shell xcrun --sdk macosx --show-sdk-path)

priv/mac/libtflite_nif.so: $(SRC)
	@if [ ! -f "$(MAC_TFLITE_LIB)" ]; then \
		echo "ERROR: $(MAC_TFLITE_LIB) not found."; \
		echo "Build it first — see docs/build_mac_tflite.md"; \
		exit 1; \
	fi
	@mkdir -p $(@D)
	$(MAC_CC) -O2 -Wall -fPIC -dynamiclib \
	    -isysroot $(MAC_SDK) \
	    -I$(ERLANG_PATH) \
	    -I$(MAC_TFLITE_HDRS) \
	    -Wl,-rpath,$(MAC_TFLITE_DIR)/lib \
	    -undefined dynamic_lookup \
	    $< $(MAC_TFLITE_LIB) \
	    -o $@
	@echo "built $@"

# ── Clean ──────────────────────────────────────────────────────────────────

clean:
	rm -rf priv/android priv/android_arm64 priv/ios_device priv/ios_sim priv/mac priv/native
