#define _POSIX_C_SOURCE 200809L

#include "separator_sagsep.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sep_api.h"
#include "util.h"

typedef struct {
    bc_separator base;
    sag_separator *engine;
    const char *preset;
    const char *device;
    int64_t replicas;
    int     threads;
    const char *tree_commit;
    int64_t *accepted;
    int64_t  accepted_capacity;
    bool     awaiting_commit;
} bc_sagsep;

int64_t bc_vehicle_replica_count(const bc_instance *inst)
{
    float total = 0.0f;
    for (int64_t i = 0; i < inst->n_nodes; ++i) total += (float)inst->demand[i];
    float capacity = (float)inst->capacity;
    if (capacity < 1e-6f) capacity = 1e-6f;
    const int64_t k = (int64_t)ceilf(total / capacity);
    return k > 1 ? k : 1;
}

const char *bc_preset_for_size(int64_t n_customers)
{
    static const struct { const char *tag; int64_t size; } presets[] = {
        { "n50", 50 }, { "n75", 75 }, { "n100", 100 }, { "n200", 200 }, { "n300", 300 },
        { "n400", 400 }, { "n500", 500 }, { "n750", 750 }, { "n1000", 1000 },
    };
    const size_t count = sizeof presets / sizeof presets[0];
    for (size_t i = 0; i < count; ++i) {
        if (n_customers <= presets[i].size) return presets[i].tag;
    }
    return presets[count - 1].tag;
}

static void sagsep_propose(bc_separator *base, int64_t m, const int64_t *u, const int64_t *v,
                           const double *x, bc_proposals *out)
{
    bc_sagsep *sep = (bc_sagsep *)base;
    sag_separator_cuts cuts;
    const int64_t rows = sag_separator_separate(sep->engine, m, u, v, x, &cuts, NULL);
    sep->awaiting_commit = true;

    if (rows > sep->accepted_capacity) {
        sep->accepted_capacity = rows;
        sep->accepted = (int64_t *)bc_realloc(sep->accepted,
                                              (size_t)rows * sizeof *sep->accepted);
    }
    out->rows = rows;
    out->indptr = cuts.indptr;
    out->nodes = cuts.customers;
    out->rhs = cuts.rhs;
}

static void sagsep_commit(bc_separator *base, const int64_t *accepted, int64_t count,
                          bool retain)
{
    bc_sagsep *sep = (bc_sagsep *)base;
    if (!sep->awaiting_commit)
        bc_die("sagseptts commit: no separation is awaiting a commit");
    sep->awaiting_commit = false;
    if (!retain) {
        sag_separator_commit(sep->engine, NULL, 0);
        return;
    }
    for (int64_t i = 0; i < count; ++i) sep->accepted[i] = accepted[i];
    for (int64_t i = 1; i < count; ++i) {
        const int64_t key = sep->accepted[i];
        int64_t j = i;
        while (j > 0 && sep->accepted[j - 1] > key) { sep->accepted[j] = sep->accepted[j - 1]; --j; }
        sep->accepted[j] = key;
    }
    sag_separator_commit(sep->engine, count > 0 ? sep->accepted : NULL, count);
}

static void sagsep_describe(bc_separator *base, char *buf, size_t cap)
{
    const bc_sagsep *sep = (const bc_sagsep *)base;
    snprintf(buf, cap,
             "sagseptts(preset=%s, device=%s, vehicle_replicas=%lld, threads=%d, "
             "tree_commit=%s)",
             sep->preset, sep->device, (long long)sep->replicas, sep->threads,
             sep->tree_commit);
}

static void sagsep_destroy(bc_separator *base)
{
    bc_sagsep *sep = (bc_sagsep *)base;
    sag_separator_destroy(sep->engine);
    free(sep->accepted);
    free(sep);
}

bc_separator *bc_separator_sagsep_create(const bc_instance *inst, const char *model_path,
                                         const char *device, const char *preset,
                                         int threads, const char *tree_commit)
{
    if (!model_path || !*model_path)
        bc_die("--separator sagseptts needs a policy graph: pass --onnx <model.onnx> "
               "or set SAGSEP_ONNX");
    if (threads < 1) bc_die("--threads must be at least 1, got %d", threads);

    char thread_text[16];
    snprintf(thread_text, sizeof thread_text, "%d", threads);
    if (setenv("SAGSEP_ORT_THREADS", thread_text, 1) != 0)
        bc_die("could not set SAGSEP_ORT_THREADS=%s", thread_text);

    bc_sagsep *sep = (bc_sagsep *)bc_calloc(1, sizeof *sep);
    sep->base.name = "sagseptts";
    sep->base.propose = sagsep_propose;
    sep->base.commit = sagsep_commit;
    sep->base.describe = sagsep_describe;
    sep->base.destroy = sagsep_destroy;
    sep->device = device;
    sep->preset = preset ? preset : bc_preset_for_size(inst->n_nodes - 1);
    sep->replicas = bc_vehicle_replica_count(inst);
    sep->threads = threads;
    sep->tree_commit = tree_commit ? tree_commit : "offered";

    sag_separator_options options;
    memset(&options, 0, sizeof options);
    options.model_path = model_path;
    options.provider = device;
    options.instance_name = inst->name;
    options.preset = sep->preset;
    options.n_nodes = inst->n_nodes;
    options.capacity = inst->capacity;
    options.demand = inst->demand;
    sep->engine = sag_separator_create(&options);
    return &sep->base;
}
