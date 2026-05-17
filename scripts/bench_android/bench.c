// bench.c — minimal TFLite C-API benchmark on Android.
// Loads a .tflite, optionally attaches an external delegate (e.g.
// MediaTek's libneuron_graph_delegate.mtk.so), invokes the model in
// a warm-loop, prints us / inference.
//
// Usage:
//   bench <model.tflite> <input.bin> [<delegate.so>|nnapi[:accel] [<delegate_opt=val> ...]]
//
// If delegate.so is omitted, runs on the default TFLite XNNPACK CPU path.
// If "nnapi" is passed, attaches TFLite's bundled NNAPI delegate (which routes
// to whatever vendor NN HAL is on the device — MediaTek's APU on this phone).
// Optional accelerator name after a colon (e.g. nnapi:mtk-neuron) restricts
// NNAPI to a specific accelerator.

#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_experimental.h"
#include "tensorflow/lite/c/common.h"

// NNAPI delegate (built into libtensorflowlite_jni.so but header not in AAR).
// Struct layout is the one TFLite v2.16.1 ships in
// tensorflow/lite/delegates/nnapi/nnapi_delegate_c_api.h — getting this
// wrong = segfault.
typedef struct {
    int  execution_preference;          // -1 undef, 0 low_power, 1 fast_single, 2 sustained
    const char* accelerator_name;       // nullptr = any accelerator
    const char* cache_dir;
    const char* model_token;
    int  disallow_nnapi_cpu;            // default 1
    int  allow_fp16;
    int  max_number_delegated_partitions;
    void* nnapi_support_library_handle;
} TfLiteNnapiDelegateOptions;

extern TfLiteNnapiDelegateOptions TfLiteNnapiDelegateOptionsDefault(void);
extern TfLiteDelegate* TfLiteNnapiDelegateCreate(const TfLiteNnapiDelegateOptions* options);
extern void TfLiteNnapiDelegateDelete(TfLiteDelegate* delegate);

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static uint8_t* read_file(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    struct stat st; fstat(fileno(f), &st);
    uint8_t* buf = (uint8_t*)malloc(st.st_size);
    if (fread(buf, 1, st.st_size, f) != (size_t)st.st_size) {
        fprintf(stderr, "short read on %s\n", path); free(buf); fclose(f); return NULL;
    }
    fclose(f); *out_size = st.st_size; return buf;
}

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// External delegate interface — what every TFLite "delegate plugin"
// .so is required to export. Same signature MediaTek's Neuron delegate
// uses, same as Hexagon's, same as NNAPI's.
typedef TfLiteDelegate* (*tflite_plugin_create_delegate_t)(
    char** options_keys, char** options_values, size_t num_options,
    void (*report_error)(const char*));
typedef void (*tflite_plugin_destroy_delegate_t)(TfLiteDelegate*);

static void delegate_err(const char* msg) {
    fprintf(stderr, "[delegate] %s\n", msg);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <model.tflite> <input.bin> "
                "[<delegate.so>|nnapi[:accel] [<key=val> ...]]\n"
                "       %s list-nnapi\n", argv[0], argv[0]);
        return 1;
    }
    const char* model_path    = argv[1];

    // Special case: "list-nnapi" prints available NNAPI device names then exits.
    if (strcmp(model_path, "list-nnapi") == 0) {
        void* nnapi_lib = dlopen("libneuralnetworks.so", RTLD_NOW);
        if (!nnapi_lib) {
            fprintf(stderr, "dlopen libneuralnetworks.so: %s\n", dlerror());
            return 1;
        }
        int (*get_count)(uint32_t*) = dlsym(nnapi_lib, "ANeuralNetworks_getDeviceCount");
        int (*get_dev)(uint32_t, void**) = dlsym(nnapi_lib, "ANeuralNetworks_getDevice");
        int (*get_name)(void*, const char**) = dlsym(nnapi_lib, "ANeuralNetworksDevice_getName");
        int (*get_type)(void*, int32_t*) = dlsym(nnapi_lib, "ANeuralNetworksDevice_getType");
        if (!get_count || !get_dev || !get_name) {
            fprintf(stderr, "missing ANeuralNetworks_* symbols\n");
            return 1;
        }
        uint32_t n = 0; get_count(&n);
        printf("NNAPI devices: %u\n", n);
        for (uint32_t i = 0; i < n; i++) {
            void* d = NULL; const char* name = NULL; int32_t typ = -1;
            get_dev(i, &d);
            get_name(d, &name);
            if (get_type) get_type(d, &typ);
            const char* tname = "?";
            switch (typ) {
                case 0: tname = "other"; break;
                case 1: tname = "cpu"; break;
                case 2: tname = "gpu"; break;
                case 3: tname = "accelerator"; break;
            }
            printf("  [%u] %s (type=%s)\n", i, name ? name : "<null>", tname);
        }
        return 0;
    }

    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.tflite> <input.bin> [...]\n", argv[0]);
        return 1;
    }
    const char* input_path    = argv[2];
    const char* delegate_path = argc > 3 ? argv[3] : NULL;

    size_t model_sz = 0;
    uint8_t* model_buf = read_file(model_path, &model_sz);
    if (!model_buf) return 1;
    TfLiteModel* model = TfLiteModelCreate(model_buf, model_sz);
    if (!model) { fprintf(stderr, "TfLiteModelCreate failed\n"); return 1; }

    TfLiteInterpreterOptions* opts = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(opts, 6);

    // Load + attach delegate if requested.
    void* dl = NULL;
    TfLiteDelegate* dlg = NULL;
    tflite_plugin_destroy_delegate_t dlg_destroy = NULL;
    int is_nnapi = (delegate_path && strncmp(delegate_path, "nnapi", 5) == 0);

    if (is_nnapi) {
        TfLiteNnapiDelegateOptions nnopts = TfLiteNnapiDelegateOptionsDefault();
        nnopts.execution_preference = 1;   // fast_single_answer
        nnopts.allow_fp16 = 1;
        nnopts.disallow_nnapi_cpu = 1;     // require real accelerator
        // After "nnapi:" comes an accelerator name like "mtk-neuron".
        if (strlen(delegate_path) > 6 && delegate_path[5] == ':') {
            nnopts.accelerator_name = delegate_path + 6;
        }
        dlg = TfLiteNnapiDelegateCreate(&nnopts);
        if (!dlg) {
            fprintf(stderr, "TfLiteNnapiDelegateCreate returned NULL "
                            "(NNAPI not available or accel '%s' not found)\n",
                    nnopts.accelerator_name ? nnopts.accelerator_name : "<any>");
            return 1;
        }
        TfLiteInterpreterOptionsAddDelegate(opts, dlg);
        fprintf(stderr, "[bench] attached NNAPI delegate (accel=%s, allow_fp16=%d)\n",
                nnopts.accelerator_name ? nnopts.accelerator_name : "<any>",
                nnopts.allow_fp16);
    } else if (delegate_path) {
        dl = dlopen(delegate_path, RTLD_NOW | RTLD_LOCAL);
        if (!dl) {
            fprintf(stderr, "dlopen %s: %s\n", delegate_path, dlerror());
            return 1;
        }
        tflite_plugin_create_delegate_t create =
            (tflite_plugin_create_delegate_t)dlsym(dl, "tflite_plugin_create_delegate");
        dlg_destroy = (tflite_plugin_destroy_delegate_t)dlsym(dl, "tflite_plugin_destroy_delegate");
        if (!create || !dlg_destroy) {
            fprintf(stderr,
                    "%s: missing tflite_plugin_create_delegate / "
                    "tflite_plugin_destroy_delegate symbols\n",
                    delegate_path);
            return 1;
        }
        // Pass delegate options as key=val pairs from argv[4..].
        int num_opts = argc - 4;
        char** keys = num_opts > 0 ? calloc(num_opts, sizeof(char*)) : NULL;
        char** vals = num_opts > 0 ? calloc(num_opts, sizeof(char*)) : NULL;
        for (int i = 0; i < num_opts; i++) {
            char* dup = strdup(argv[4 + i]);
            char* eq = strchr(dup, '=');
            if (eq) { *eq = '\0'; keys[i] = dup; vals[i] = eq + 1; }
            else    { keys[i] = dup; vals[i] = (char*)""; }
        }
        dlg = create(keys, vals, num_opts, delegate_err);
        if (!dlg) {
            fprintf(stderr, "delegate create returned NULL\n");
            return 1;
        }
        TfLiteInterpreterOptionsAddDelegate(opts, dlg);
        fprintf(stderr, "[bench] attached delegate from %s\n", delegate_path);
    }

    TfLiteInterpreter* interp = TfLiteInterpreterCreate(model, opts);
    if (!interp) { fprintf(stderr, "TfLiteInterpreterCreate failed\n"); return 1; }

    if (TfLiteInterpreterAllocateTensors(interp) != kTfLiteOk) {
        fprintf(stderr, "AllocateTensors failed\n"); return 1;
    }

    int in_count = TfLiteInterpreterGetInputTensorCount(interp);
    int out_count = TfLiteInterpreterGetOutputTensorCount(interp);
    fprintf(stderr, "[bench] inputs=%d outputs=%d\n", in_count, out_count);
    TfLiteTensor* in_t = TfLiteInterpreterGetInputTensor(interp, 0);
    fprintf(stderr, "[bench] input[0] type=%d bytes=%zu\n",
            (int)TfLiteTensorType(in_t), TfLiteTensorByteSize(in_t));

    size_t in_sz = 0;
    uint8_t* in_buf = read_file(input_path, &in_sz);
    if (!in_buf) return 1;
    if (in_sz != TfLiteTensorByteSize(in_t)) {
        fprintf(stderr,
                "[bench] input size mismatch: file=%zu, model=%zu — "
                "may be a quant-type mismatch. Continuing with min(file, model) bytes.\n",
                in_sz, TfLiteTensorByteSize(in_t));
    }
    size_t copy = in_sz < TfLiteTensorByteSize(in_t) ? in_sz : TfLiteTensorByteSize(in_t);
    TfLiteTensorCopyFromBuffer(in_t, in_buf, copy);

    // Warm + bench
    for (int i = 0; i < 3; i++) TfLiteInterpreterInvoke(interp);

    int reps = 5;
    double sum = 0, mn = 1e9, mx = 0;
    for (int i = 0; i < reps; i++) {
        double t0 = now_ms();
        TfLiteStatus st = TfLiteInterpreterInvoke(interp);
        double dt = now_ms() - t0;
        if (st != kTfLiteOk) {
            fprintf(stderr, "Invoke failed at iter %d\n", i); return 1;
        }
        fprintf(stderr, "[bench] iter %d: %.1f ms\n", i, dt);
        sum += dt;
        if (dt < mn) mn = dt;
        if (dt > mx) mx = dt;
    }
    fprintf(stderr, "[bench] mean=%.1f min=%.1f max=%.1f ms (%d reps)\n",
            sum / reps, mn, mx, reps);

    TfLiteInterpreterDelete(interp);
    TfLiteInterpreterOptionsDelete(opts);
    if (dlg) {
        if (is_nnapi) TfLiteNnapiDelegateDelete(dlg);
        else if (dlg_destroy) dlg_destroy(dlg);
    }
    if (dl) dlclose(dl);
    TfLiteModelDelete(model);
    free(model_buf); free(in_buf);
    return 0;
}
