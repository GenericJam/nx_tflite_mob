# Build the tflite_nif as a NIF .so for Mac (host) or Android arm64.
#
# Mac uses an x86_64/arm64-universal libtensorflowlite_jni.so — not
# normally shipped. For the host smoke test, the easier path is to
# build against the pip-installed runtime (TODO). For now this Makefile
# focuses on the Android cross-compile path, which is what we actually
# ship inside a Mob app.

ERLANG_PATH := $(shell erl -noshell -eval 'io:format("~s/erts-16.4/include", [code:root_dir()])' -s init stop)
TFLITE_AAR  := /tmp/tflite_android/aar
TFLITE_HDRS := $(TFLITE_AAR)/headers

ANDROID_NDK ?= /Users/kevin/Library/Android/sdk/ndk/27.2.12479018
NDK_HOST     := darwin-x86_64
NDK_CC       := $(ANDROID_NDK)/toolchains/llvm/prebuilt/$(NDK_HOST)/bin/aarch64-linux-android29-clang
TFLITE_SO    := $(TFLITE_AAR)/jni/arm64-v8a/libtensorflowlite_jni.so

priv/android/libtflite_nif.so: c_src/tflite_nif.c
	@mkdir -p priv/android
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

android: priv/android/libtflite_nif.so

clean:
	rm -rf priv/android priv/native
