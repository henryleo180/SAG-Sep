#include "forward.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arena.h"
#include "features.h"

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_c_api.h>

static const char *const kInputNames[3]  = { "edge_index", "node_attr", "edge_attr" };
static const char *const kOutputNames[1] = { "probs" };

#define SAG_ONNX_LATENT_DIM 256

#define SAG_KCHUNK_DEFAULT_BYTES ((int64_t)4 * 1024 * 1024 * 1024)

#define SAG_KCHUNK_AUTO (-1)
#define SAG_KCHUNK_OFF  (0)
#endif

struct sag_forward {
    sag_forward_backend kind;
    char                describe[600];

    /* ---- onnx ---- */
#ifdef HAVE_ONNXRUNTIME
    const OrtApi      *api;
    OrtEnv            *env;
    OrtSessionOptions *opts;
    OrtSession        *session;
    OrtMemoryInfo     *mem;
    bool               cuda;

    /* ---- K chunking (per-replica blocks, bit-for-bit identical; see kchunk_decide) ---- */
    int64_t kchunk_cfg;
    int64_t kchunk_bytes;
    int64_t kchunk_cap;
    int64_t kchunk_logged;
    bool    kchunk_debug;

#endif
    float  *out_buf;
    int64_t out_cap;
};

/* ===================================================================== */
/* onnx backend                                                          */
/* ===================================================================== */

#ifdef HAVE_ONNXRUNTIME

static void ort_check(const sag_forward *f, OrtStatus *st, const char *what)
{
    if (!st) return;
    const char *msg  = f->api->GetErrorMessage(st);
    OrtErrorCode code = f->api->GetErrorCode(st);
    char buf[512];
    snprintf(buf, sizeof buf, "%s", msg ? msg : "(no message)");
    f->api->ReleaseStatus(st);
    sag_die("forward/onnx: %s failed (code %d): %s", what, (int)code, buf);
}

static void ort_verify_signature(sag_forward *f, const char *model_path)
{
    OrtAllocator *alloc = NULL;
    ort_check(f, f->api->GetAllocatorWithDefaultOptions(&alloc), "GetAllocatorWithDefaultOptions");

    size_t n_in = 0, n_out = 0;
    ort_check(f, f->api->SessionGetInputCount(f->session, &n_in), "SessionGetInputCount");
    ort_check(f, f->api->SessionGetOutputCount(f->session, &n_out), "SessionGetOutputCount");
    if (n_in != 3 || n_out != 1)
        sag_die("forward/onnx: %s has %zu inputs / %zu outputs, expected 3 / 1 "
                "(re-export the model with the ONNX export script)", model_path, n_in, n_out);

    for (size_t i = 0; i < 3; ++i) {
        char *name = NULL;
        ort_check(f, f->api->SessionGetInputName(f->session, i, alloc, &name),
                  "SessionGetInputName");
        int bad = (name == NULL) || strcmp(name, kInputNames[i]) != 0;
        char got[64];
        snprintf(got, sizeof got, "%s", name ? name : "(null)");
        if (name) ort_check(f, f->api->AllocatorFree(alloc, name), "AllocatorFree");
        if (bad)
            sag_die("forward/onnx: %s input %zu is '%s', expected '%s'",
                    model_path, i, got, kInputNames[i]);
    }
    char *oname = NULL;
    ort_check(f, f->api->SessionGetOutputName(f->session, 0, alloc, &oname),
              "SessionGetOutputName");
    int bad = (oname == NULL) || strcmp(oname, kOutputNames[0]) != 0;
    char got[64];
    snprintf(got, sizeof got, "%s", oname ? oname : "(null)");
    if (oname) ort_check(f, f->api->AllocatorFree(alloc, oname), "AllocatorFree");
    if (bad)
        sag_die("forward/onnx: %s output 0 is '%s', expected '%s'",
                model_path, got, kOutputNames[0]);
}

static int parse_provider(const char *provider, int *device_id)
{
    *device_id = 0;
    if (!provider || !*provider) return 0;
    if (strcmp(provider, "cpu") == 0) return 0;
    if (strcmp(provider, "cuda") == 0) return 1;
    if (strncmp(provider, "cuda:", 5) == 0) {
        const char *p = provider + 5;
        if (!*p) sag_die("forward/onnx: provider '%s' has no device id (use 'cuda' or 'cuda:0')",
                         provider);
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (!end || *end || v < 0 || v > 64)
            sag_die("forward/onnx: provider '%s' has an invalid device id", provider);
        *device_id = (int)v;
        return 1;
    }
    sag_die("forward/onnx: provider '%s' is not implemented "
            "(only 'cpu' / 'cuda' / 'cuda:<id>')", provider);
    return 0;
}

static int64_t kchunk_decide(const sag_forward *f, int64_t K, int64_t E)
{
    if (f->kchunk_cfg == SAG_KCHUNK_OFF) return K;

    int64_t kc;
    if (f->kchunk_cfg > 0) {
        kc = f->kchunk_cfg;
    } else {
        const int64_t per_replica = E * (int64_t)SAG_ONNX_LATENT_DIM * (int64_t)sizeof(float);
        if (per_replica <= 0) return K;
        if (per_replica >= f->kchunk_bytes) {
            kc = 1;
        } else {
            const int64_t cap = f->kchunk_bytes / per_replica;
            const int64_t nblk = (K + cap - 1) / cap;
            kc = (K + nblk - 1) / nblk;
        }
    }
    if (f->kchunk_cap > 0 && kc > f->kchunk_cap) kc = f->kchunk_cap;
    return kc < K ? kc : K;
}

sag_forward *sag_forward_open_onnx(const char *model_path, const char *provider)
{
    if (!model_path || !*model_path)
        sag_die("forward/onnx: model path is empty");

    int device_id = 0;
    const int want_cuda = parse_provider(provider, &device_id);

    sag_forward *f = (sag_forward *)calloc(1, sizeof *f);
    if (!f) sag_die("forward: out of memory");
    f->kind = SAG_FORWARD_ONNX;
    f->cuda = (want_cuda != 0);

    f->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!f->api)
        sag_die("forward/onnx: onnxruntime is too old for ORT_API_VERSION %d "
                "(built against %s)", (int)ORT_API_VERSION,
                OrtGetApiBase()->GetVersionString());

    ort_check(f, f->api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "sagsep", &f->env), "CreateEnv");
    ort_check(f, f->api->CreateSessionOptions(&f->opts), "CreateSessionOptions");
    {
        const char *env_thr = getenv("SAGSEP_ORT_THREADS");
        int thr = 1;
        if (env_thr && *env_thr) {
            long v = strtol(env_thr, NULL, 10);
            if (v >= 1 && v <= 256) thr = (int)v;
        }
        if (thr > 1 && !f->cuda)
            sag_die("forward/onnx: SAGSEP_ORT_THREADS=%d is unsafe on the CPU provider -- "
                    "ORT's ScatterND(reduction=add) CPU kernel accumulates duplicate "
                    "indices non-atomically and silently returns wrong probs "
                    "(measured max|err| up to 1.3e-01; ORT 1.23.2/1.26.0/1.28.0 all affected). "
                    "Use SAGSEP_ORT_THREADS=1, or --device cuda:<id> for speed. "
                    "Correct probs on the CPU provider require exactly one intra-op thread.", thr);
        ort_check(f, f->api->SetIntraOpNumThreads(f->opts, thr), "SetIntraOpNumThreads");
        ort_check(f, f->api->SetInterOpNumThreads(f->opts, 1), "SetInterOpNumThreads");
    }

    {
        f->kchunk_cfg   = SAG_KCHUNK_AUTO;
        f->kchunk_bytes = SAG_KCHUNK_DEFAULT_BYTES;
        const char *env_kc = getenv("SAGSEP_ORT_KCHUNK");
        if (env_kc && *env_kc && strcmp(env_kc, "auto") != 0) {
            char *end = NULL;
            long long v = strtoll(env_kc, &end, 10);
            if (!end || *end || v < 0)
                sag_die("forward/onnx: SAGSEP_ORT_KCHUNK='%s' is not valid "
                        "('auto' / 0=off / positive integer=forced block size)", env_kc);
            f->kchunk_cfg = (int64_t)v;
        }
        const char *env_kb = getenv("SAGSEP_ORT_KBYTES");
        if (env_kb && *env_kb) {
            char *end = NULL;
            long long v = strtoll(env_kb, &end, 10);
            if (!end || *end || v < (long long)(1 << 20))
                sag_die("forward/onnx: SAGSEP_ORT_KBYTES='%s' is not valid "
                        "(per-tensor byte budget, at least 1 MiB)", env_kb);
            f->kchunk_bytes = (int64_t)v;
        }
        const char *env_kd = getenv("SAGSEP_ORT_KDEBUG");
        f->kchunk_debug = (env_kd && *env_kd && strcmp(env_kd, "0") != 0);
    }

    ort_check(f, f->api->SetSessionExecutionMode(f->opts, ORT_SEQUENTIAL),
              "SetSessionExecutionMode");
    ort_check(f, f->api->SetSessionGraphOptimizationLevel(f->opts, ORT_ENABLE_ALL),
              "SetSessionGraphOptimizationLevel");

    if (f->cuda) {
        OrtCUDAProviderOptionsV2 *cuda_opts = NULL;
        char dev[16];
        snprintf(dev, sizeof dev, "%d", device_id);
        const char *keys[]   = { "device_id" };
        const char *values[] = { dev };
        const char *stage = "CreateCUDAProviderOptions";
        OrtStatus *st = f->api->CreateCUDAProviderOptions(&cuda_opts);
        if (!st) {
            stage = "UpdateCUDAProviderOptions";
            st = f->api->UpdateCUDAProviderOptions(cuda_opts, keys, values, 1);
        }
        if (!st) {
            stage = "SessionOptionsAppendExecutionProvider_CUDA_V2";
            st = f->api->SessionOptionsAppendExecutionProvider_CUDA_V2(f->opts, cuda_opts);
        }
        if (cuda_opts) f->api->ReleaseCUDAProviderOptions(cuda_opts);
        if (st) {
            const char *msg = f->api->GetErrorMessage(st);
            char buf[512];
            snprintf(buf, sizeof buf, "%s", msg ? msg : "(no message)");
            f->api->ReleaseStatus(st);
            sag_die("forward/onnx: could not enable the CUDA EP (device %d, %s): %s\n"
                    "  * this requires a GPU build of onnxruntime (point ONNXRUNTIME_HOME\n"
                    "    at a GPU release; its lib/ must contain "
                    "libonnxruntime_providers_cuda.so, which a CPU-only package lacks -- "
                    "that fails at the very first step with 'not enabled in this build')\n"
                    "  * and the CUDA / cuDNN runtimes must be present at run time\n"
                    "  * otherwise use --device cpu", device_id, stage, buf);
        }
    }

    ort_check(f, f->api->CreateSession(f->env, model_path, f->opts, &f->session),
              "CreateSession");
    ort_check(f, f->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &f->mem),
              "CreateCpuMemoryInfo");
    ort_verify_signature(f, model_path);

    if (f->cuda)
        snprintf(f->describe, sizeof f->describe,
                 "onnx(%s, cuda:%d, ort %s, non-bitwise)",
                 model_path, device_id, OrtGetApiBase()->GetVersionString());
    else
        snprintf(f->describe, sizeof f->describe, "onnx(%s, cpu, ort %s)",
                 model_path, OrtGetApiBase()->GetVersionString());
    return f;
}

static void kchunk_log(sag_forward *f, int64_t N, int64_t K, int64_t E, int64_t kc)
{
    if (f->kchunk_debug || (kc < K && kc != f->kchunk_logged)) {
        fprintf(stderr, "[forward/onnx] K-chunk: N=%lld K=%lld E=%lld -> block=%lld "
                "(%lld blocks, ~%.2f GiB/act)\n",
                (long long)N, (long long)K, (long long)E, (long long)kc,
                (long long)((K + kc - 1) / kc),
                (double)(E * kc * SAG_ONNX_LATENT_DIM * (int64_t)sizeof(float))
                    / (1024.0 * 1024.0 * 1024.0));
        f->kchunk_logged = kc;
    }
}

const float *sag_forward_probs_graph(sag_forward *f, const sag_graph_features *g)
{
    if (!f) sag_die("forward: NULL handle");
    if (!g) sag_die("forward/onnx: NULL graph features");
    if (g->n_nodes < 2 || g->vehicle_count < 1 || g->n_edges < 1)
        sag_die("forward/onnx: bad features N=%lld K=%lld E=%lld "
                "(the graph builder guarantees at least one edge)", (long long)g->n_nodes,
                (long long)g->vehicle_count, (long long)g->n_edges);
    if (!g->edge_index || !g->edge_attr || !g->node_attr)
        sag_die("forward/onnx: features arrays must be non-NULL");

    const int64_t N = g->n_nodes, K = g->vehicle_count, E = g->n_edges;
    const int64_t n_out = K * N;
    if (f->out_cap < n_out) {
        free(f->out_buf);
        f->out_buf = (float *)malloc((size_t)n_out * sizeof(float));
        if (!f->out_buf) sag_die("forward/onnx: out of memory for %lld probs", (long long)n_out);
        f->out_cap = n_out;
    }

    const int64_t ei_shape[2] = { 2, E };
    const int64_t ea_shape[2] = { E, (int64_t)SAG_EDGE_DIM };

    int64_t kc = kchunk_decide(f, K, E);
    if (kc < 1 || kc > K)
        sag_die("forward/onnx: kchunk_decide returned %lld, out of range [1,%lld]",
                (long long)kc, (long long)K);
    kchunk_log(f, N, K, E, kc);

    OrtValue *in[3] = { NULL, NULL, NULL };
    ort_check(f, f->api->CreateTensorWithDataAsOrtValue(
                     f->mem, (void *)(uintptr_t)g->edge_index,
                     (size_t)(2 * E) * sizeof(int64_t), ei_shape, 2,
                     ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &in[0]),
              "CreateTensorWithDataAsOrtValue(edge_index)");
    ort_check(f, f->api->CreateTensorWithDataAsOrtValue(
                     f->mem, (void *)(uintptr_t)g->edge_attr,
                     (size_t)(E * SAG_EDGE_DIM) * sizeof(float), ea_shape, 2,
                     ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[2]),
              "CreateTensorWithDataAsOrtValue(edge_attr)");

    int64_t k0 = 0;
    while (k0 < K) {
        const int64_t kb = (K - k0 < kc) ? (K - k0) : kc;
        const int64_t na_shape[3] = { kb, N, (int64_t)SAG_NODE_DIM };
        ort_check(f, f->api->CreateTensorWithDataAsOrtValue(
                         f->mem,
                         (void *)(uintptr_t)(g->node_attr + k0 * N * SAG_NODE_DIM),
                         (size_t)(kb * N * SAG_NODE_DIM) * sizeof(float), na_shape, 3,
                         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[1]),
                  "CreateTensorWithDataAsOrtValue(node_attr)");

        OrtValue *out = NULL;
        {
            OrtStatus *st = f->api->Run(f->session, NULL, kInputNames,
                                        (const OrtValue *const *)in, 3,
                                        kOutputNames, 1, &out);
            if (st) {
                char buf[512];
                const char *msg = f->api->GetErrorMessage(st);
                const OrtErrorCode code = f->api->GetErrorCode(st);
                snprintf(buf, sizeof buf, "%s", msg ? msg : "(no message)");
                f->api->ReleaseStatus(st);
                const bool alloc_fail =
                    strstr(buf, "Failed to allocate") != NULL ||
                    strstr(buf, "out of memory")      != NULL ||
                    strstr(buf, "OUT_OF_MEMORY")      != NULL;
                if (alloc_fail && f->kchunk_cfg != SAG_KCHUNK_OFF && kb > 1) {
                    const int64_t next = kb / 2;
                    fprintf(stderr, "[forward/onnx] K-chunk: allocation failed (N=%lld K=%lld "
                            "E=%lld block=%lld), backing off to block=%lld, retrying this block\n",
                            (long long)N, (long long)K, (long long)E,
                            (long long)kb, (long long)next);
                    kc = next;
                    f->kchunk_cap = next;
                    f->kchunk_logged = next;
                    f->api->ReleaseValue(in[1]);
                    in[1] = NULL;
                    continue;
                }
                sag_die("forward/onnx: Run failed (code %d) at K-block [%lld,%lld) of "
                        "K=%lld (N=%lld E=%lld, block=%lld): %s",
                        (int)code, (long long)k0, (long long)(k0 + kb),
                        (long long)K, (long long)N, (long long)E, (long long)kb, buf);
            }
        }

        OrtTensorTypeAndShapeInfo *info = NULL;
        ort_check(f, f->api->GetTensorTypeAndShape(out, &info), "GetTensorTypeAndShape");
        size_t n_dims = 0;
        ort_check(f, f->api->GetDimensionsCount(info, &n_dims), "GetDimensionsCount");
        int64_t dims[4] = { 0, 0, 0, 0 };
        if (n_dims != 2) {
            f->api->ReleaseTensorTypeAndShapeInfo(info);
            sag_die("forward/onnx: probs has %zu dims, expected 2", n_dims);
        }
        ort_check(f, f->api->GetDimensions(info, dims, 2), "GetDimensions");
        f->api->ReleaseTensorTypeAndShapeInfo(info);
        if (dims[0] != kb || dims[1] != N) {
            f->api->ReleaseValue(out);
            for (int i = 0; i < 3; ++i) f->api->ReleaseValue(in[i]);
            sag_die("forward/onnx: probs shape (%lld,%lld) != (K_block=%lld, N=%lld)",
                    (long long)dims[0], (long long)dims[1], (long long)kb, (long long)N);
        }

        float *src = NULL;
        ort_check(f, f->api->GetTensorMutableData(out, (void **)&src), "GetTensorMutableData");
        memcpy(f->out_buf + k0 * N, src, (size_t)(kb * N) * sizeof(float));

        f->api->ReleaseValue(out);
        f->api->ReleaseValue(in[1]);
        in[1] = NULL;
        k0 += kb;
    }

    f->api->ReleaseValue(in[0]);
    f->api->ReleaseValue(in[2]);
    return f->out_buf;
}

bool sag_forward_onnx_available(void) { return true; }

static void ort_close(sag_forward *f)
{
    if (f->mem)     f->api->ReleaseMemoryInfo(f->mem);
    if (!f->cuda) {
        if (f->session) f->api->ReleaseSession(f->session);
        if (f->opts)    f->api->ReleaseSessionOptions(f->opts);
        if (f->env)     f->api->ReleaseEnv(f->env);
    }
    f->mem = NULL; f->session = NULL; f->opts = NULL; f->env = NULL;
}

#else

sag_forward *sag_forward_open_onnx(const char *model_path, const char *provider)
{
    (void)provider;
    sag_die("forward/onnx: this build has no onnxruntime (model %s). "
            "Rebuild with ONNXRUNTIME_HOME=<onnxruntime-linux-x64-*> "
            "(needs include/onnxruntime_c_api.h + lib/libonnxruntime.so).",
            model_path ? model_path : "(null)");
}

const float *sag_forward_probs_graph(sag_forward *f, const sag_graph_features *g)
{
    (void)g;
    sag_die("forward: sag_forward_probs_graph needs an onnxruntime build; "
            "backend is %s", f ? f->describe : "(null)");
}

bool sag_forward_onnx_available(void) { return false; }

static void ort_close(sag_forward *f) { (void)f; }

#endif

/* ===================================================================== */
/* Common                                                                */
/* ===================================================================== */

sag_forward_backend sag_forward_kind(const sag_forward *f)
{
    if (!f) sag_die("forward: NULL handle");
    return f->kind;
}

const char *sag_forward_describe(const sag_forward *f)
{
    if (!f) sag_die("forward: NULL handle");
    return f->describe;
}

void sag_forward_close(sag_forward *f)
{
    if (!f) return;
    ort_close(f);
    free(f->out_buf);
    f->out_buf = NULL;
    f->out_cap = 0;
    free(f);
}
