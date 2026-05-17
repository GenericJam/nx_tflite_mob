// tflite_nif.c — Erlang NIF wrapping TensorFlow Lite for Mob apps.
//
// Exposes:
//   load_module(model_bytes, opts)     -> {:ok, handle} | {:error, reason}
//   call(handle, list_of_input_binaries) -> {:ok, list_of_output_binaries} | {:error, reason}
//   release_module(handle)              -> :ok
//
// opts is a map (passed as keyword list converted to proplist):
//   delegate:        "xnnpack" | "nnapi"               (default: "xnnpack")
//   accelerator:     "mtk-neuron_shim" | "mtk-gpu_shim" | nil (only for nnapi)
//   num_threads:     integer (XNNPACK)                  (default: 6)
//   allow_fp16:      boolean (NNAPI)                    (default: true)
//
// Designed for the YOLO live-detect flow: load once, call many times.
// Output binaries are returned in the model's tensor order, raw bytes.

#include <erl_nif.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_experimental.h"
#include "tensorflow/lite/c/common.h"

// NNAPI delegate struct (TFLite v2.16.1 layout — see
// tensorflow/lite/delegates/nnapi/nnapi_delegate_c_api.h).
typedef struct {
    int  execution_preference;
    const char* accelerator_name;
    const char* cache_dir;
    const char* model_token;
    int  disallow_nnapi_cpu;
    int  allow_fp16;
    int  max_number_delegated_partitions;
    void* nnapi_support_library_handle;
} TfLiteNnapiDelegateOptions;

extern TfLiteNnapiDelegateOptions TfLiteNnapiDelegateOptionsDefault(void);
extern TfLiteDelegate* TfLiteNnapiDelegateCreate(const TfLiteNnapiDelegateOptions* options);
extern void TfLiteNnapiDelegateDelete(TfLiteDelegate* delegate);

// ── Resource ──────────────────────────────────────────────────────────────

typedef struct {
    TfLiteModel*              model;
    TfLiteInterpreter*        interp;
    TfLiteInterpreterOptions* opts;
    TfLiteDelegate*           delegate;
    int                       delegate_is_nnapi;
} tflite_module_t;

static ErlNifResourceType* RES_TYPE = NULL;

static void module_dtor(ErlNifEnv* env, void* obj) {
    (void)env;
    tflite_module_t* m = (tflite_module_t*)obj;
    if (m->interp)  TfLiteInterpreterDelete(m->interp);
    if (m->opts)    TfLiteInterpreterOptionsDelete(m->opts);
    // The only delegate we create explicitly is NNAPI. XNNPACK is bundled
    // into TFLite and attached implicitly when no other delegate is set,
    // so we never own an XNNPACK delegate handle to free.
    if (m->delegate && m->delegate_is_nnapi) {
        TfLiteNnapiDelegateDelete(m->delegate);
    }
    if (m->model)   TfLiteModelDelete(m->model);
    memset(m, 0, sizeof(*m));
}

// ── Helpers ───────────────────────────────────────────────────────────────

static ERL_NIF_TERM mk_error(ErlNifEnv* env, const char* msg) {
    return enif_make_tuple2(
        env,
        enif_make_atom(env, "error"),
        enif_make_string(env, msg, ERL_NIF_LATIN1));
}

// Pull a string value from a proplist for the given key atom.
// Returns 1 on success, 0 if missing (out_buf left as default).
static int proplist_get_string(ErlNifEnv* env, ERL_NIF_TERM list,
                                const char* key,
                                char* out_buf, size_t out_sz) {
    ERL_NIF_TERM head, tail = list;
    while (enif_get_list_cell(env, tail, &head, &tail)) {
        const ERL_NIF_TERM* tup;
        int arity;
        if (!enif_get_tuple(env, head, &arity, &tup) || arity != 2) continue;
        char k[64];
        if (enif_get_atom(env, tup[0], k, sizeof(k), ERL_NIF_LATIN1) <= 0) continue;
        if (strcmp(k, key) != 0) continue;
        ErlNifBinary bin;
        if (enif_inspect_iolist_as_binary(env, tup[1], &bin) ||
            enif_inspect_binary(env, tup[1], &bin)) {
            size_t n = bin.size < out_sz - 1 ? bin.size : out_sz - 1;
            memcpy(out_buf, bin.data, n);
            out_buf[n] = '\0';
            return 1;
        }
        if (enif_get_atom(env, tup[1], out_buf, (unsigned)out_sz, ERL_NIF_LATIN1) > 0) {
            return 1;
        }
        return 0;
    }
    return 0;
}

static int proplist_get_int(ErlNifEnv* env, ERL_NIF_TERM list,
                             const char* key, int* out) {
    ERL_NIF_TERM head, tail = list;
    while (enif_get_list_cell(env, tail, &head, &tail)) {
        const ERL_NIF_TERM* tup;
        int arity;
        if (!enif_get_tuple(env, head, &arity, &tup) || arity != 2) continue;
        char k[64];
        if (enif_get_atom(env, tup[0], k, sizeof(k), ERL_NIF_LATIN1) <= 0) continue;
        if (strcmp(k, key) != 0) continue;
        return enif_get_int(env, tup[1], out);
    }
    return 0;
}

static int proplist_get_bool(ErlNifEnv* env, ERL_NIF_TERM list,
                              const char* key, int* out) {
    char val[16];
    if (!proplist_get_string(env, list, key, val, sizeof(val))) return 0;
    if (strcmp(val, "true") == 0)  { *out = 1; return 1; }
    if (strcmp(val, "false") == 0) { *out = 0; return 1; }
    return 0;
}

// ── load_module(model_bytes, opts) ────────────────────────────────────────

static ERL_NIF_TERM nif_load_module(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    ErlNifBinary model_bin;
    if (!enif_inspect_binary(env, argv[0], &model_bin)) {
        return enif_make_badarg(env);
    }

    char delegate_name[32] = "xnnpack";
    char accelerator[64] = "";
    int  num_threads = 6;
    int  allow_fp16 = 1;
    proplist_get_string(env, argv[1], "delegate", delegate_name, sizeof(delegate_name));
    proplist_get_string(env, argv[1], "accelerator", accelerator, sizeof(accelerator));
    proplist_get_int(env, argv[1], "num_threads", &num_threads);
    proplist_get_bool(env, argv[1], "allow_fp16", &allow_fp16);

    tflite_module_t* m = enif_alloc_resource(RES_TYPE, sizeof(tflite_module_t));
    memset(m, 0, sizeof(*m));

    m->model = TfLiteModelCreate(model_bin.data, model_bin.size);
    if (!m->model) {
        enif_release_resource(m);
        return mk_error(env, "TfLiteModelCreate failed");
    }

    m->opts = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(m->opts, num_threads);

    if (strcmp(delegate_name, "nnapi") == 0) {
        TfLiteNnapiDelegateOptions nnopts = TfLiteNnapiDelegateOptionsDefault();
        nnopts.execution_preference = 1;  // fast_single_answer
        nnopts.allow_fp16 = allow_fp16;
        nnopts.disallow_nnapi_cpu = 1;
        nnopts.accelerator_name = accelerator[0] ? accelerator : NULL;
        m->delegate = TfLiteNnapiDelegateCreate(&nnopts);
        if (!m->delegate) {
            enif_release_resource(m);
            return mk_error(env, "TfLiteNnapiDelegateCreate failed");
        }
        m->delegate_is_nnapi = 1;
        TfLiteInterpreterOptionsAddDelegate(m->opts, m->delegate);
    }
    // xnnpack is default — no explicit delegate attach needed; TFLite uses it
    // automatically when no other delegate is bound.

    m->interp = TfLiteInterpreterCreate(m->model, m->opts);
    if (!m->interp) {
        enif_release_resource(m);
        return mk_error(env, "TfLiteInterpreterCreate failed");
    }
    if (TfLiteInterpreterAllocateTensors(m->interp) != kTfLiteOk) {
        enif_release_resource(m);
        return mk_error(env, "AllocateTensors failed");
    }

    ERL_NIF_TERM handle = enif_make_resource(env, m);
    enif_release_resource(m);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), handle);
}

// ── call(handle, [input_bin, ...]) -> {:ok, [output_bin, ...]} ────────────

static ERL_NIF_TERM nif_call(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    tflite_module_t* m;
    if (!enif_get_resource(env, argv[0], RES_TYPE, (void**)&m)) {
        return enif_make_badarg(env);
    }

    unsigned in_count = 0;
    if (!enif_get_list_length(env, argv[1], &in_count)) {
        return enif_make_badarg(env);
    }
    int model_in_count = TfLiteInterpreterGetInputTensorCount(m->interp);
    if ((int)in_count != model_in_count) {
        char buf[128];
        snprintf(buf, sizeof(buf),
                 "input count mismatch: list has %u, model wants %d",
                 in_count, model_in_count);
        return mk_error(env, buf);
    }

    ERL_NIF_TERM head, tail = argv[1];
    for (int i = 0; i < model_in_count; i++) {
        if (!enif_get_list_cell(env, tail, &head, &tail)) {
            return mk_error(env, "input list exhausted prematurely");
        }
        ErlNifBinary bin;
        if (!enif_inspect_binary(env, head, &bin)) {
            return mk_error(env, "input entry not a binary");
        }
        TfLiteTensor* t = TfLiteInterpreterGetInputTensor(m->interp, i);
        size_t want = TfLiteTensorByteSize(t);
        if (bin.size != want) {
            char buf[160];
            snprintf(buf, sizeof(buf),
                     "input %d byte-size mismatch: got %zu, model wants %zu",
                     i, bin.size, want);
            return mk_error(env, buf);
        }
        if (TfLiteTensorCopyFromBuffer(t, bin.data, bin.size) != kTfLiteOk) {
            return mk_error(env, "TfLiteTensorCopyFromBuffer failed");
        }
    }

    if (TfLiteInterpreterInvoke(m->interp) != kTfLiteOk) {
        return mk_error(env, "TfLiteInterpreterInvoke failed");
    }

    int out_count = TfLiteInterpreterGetOutputTensorCount(m->interp);
    ERL_NIF_TERM out_list = enif_make_list(env, 0);
    for (int i = out_count - 1; i >= 0; i--) {
        const TfLiteTensor* t = TfLiteInterpreterGetOutputTensor(m->interp, i);
        size_t sz = TfLiteTensorByteSize(t);
        ErlNifBinary out_bin;
        enif_alloc_binary(sz, &out_bin);
        if (TfLiteTensorCopyToBuffer(t, out_bin.data, sz) != kTfLiteOk) {
            enif_release_binary(&out_bin);
            return mk_error(env, "TfLiteTensorCopyToBuffer failed");
        }
        out_list = enif_make_list_cell(env, enif_make_binary(env, &out_bin), out_list);
    }

    return enif_make_tuple2(env, enif_make_atom(env, "ok"), out_list);
}

// ── release_module(handle) ────────────────────────────────────────────────

static ERL_NIF_TERM nif_release_module(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    tflite_module_t* m;
    if (!enif_get_resource(env, argv[0], RES_TYPE, (void**)&m)) {
        return enif_make_badarg(env);
    }
    if (m->interp)  { TfLiteInterpreterDelete(m->interp);        m->interp = NULL; }
    if (m->opts)    { TfLiteInterpreterOptionsDelete(m->opts);    m->opts = NULL; }
    // Only NNAPI is an owned delegate (XNNPACK is implicit/free).
    if (m->delegate && m->delegate_is_nnapi) {
        TfLiteNnapiDelegateDelete(m->delegate);
    }
    m->delegate = NULL;
    if (m->model)   { TfLiteModelDelete(m->model);                m->model = NULL; }
    return enif_make_atom(env, "ok");
}

// ── NIF init ──────────────────────────────────────────────────────────────

static int load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info) {
    (void)priv_data; (void)load_info;
    ErlNifResourceFlags flags = ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER;
    RES_TYPE = enif_open_resource_type(env, NULL, "tflite_module",
                                        module_dtor, flags, NULL);
    return RES_TYPE ? 0 : -1;
}

static ErlNifFunc nif_funcs[] = {
    {"load_module",    2, nif_load_module,    ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"call",           2, nif_call,           ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"release_module", 1, nif_release_module, 0}
};

ERL_NIF_INIT(Elixir.NxTfliteMob.NIF, nif_funcs, load, NULL, NULL, NULL)
