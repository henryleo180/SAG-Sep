#define _POSIX_C_SOURCE 200809L

#include "sep_api.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arena.h"
#include "bitset.h"
#include "features.h"
#include "forward.h"
#include "round.h"
#include "sagsep.h"
#include "spine.h"

#define SEP_DEFAULT_ITER_BYTES (1024u << 20)
#define SEP_PERSISTENT_SLACK   (16u << 20)

struct sag_separator {
    sag_config    cfg;
    sag_instance  inst;
    sag_arena     persistent;
    sag_arena     iteration;
    sag_forward  *forward;
    sag_old_bank  bank;
    sag_key_set   blocked;
    sag_round_env env;
    int64_t       words;
    int64_t       replicas;
    int64_t       iteration_count;
    int64_t       committed_last;
    int64_t         pending_rows;
    const uint64_t *pending_masks;
    bool            pending_commit;
    double        load_seconds;
    char          description[768];
};

/* ===================================================================== */
/* Option checking                                                       */
/* ===================================================================== */

static bool ends_with_toml(const char *s)
{
    const size_t n = strlen(s);
    return n >= 5 && strcmp(s + n - 5, ".toml") == 0;
}

static void check_options(const sag_separator_options *options)
{
    if (!options) sag_die("sag_separator_create: NULL options");

    if (!(options->model_path && options->model_path[0]))
        sag_die("sag_separator_create: model_path is required -- the path of the "
                "exported ONNX policy graph");
    if (!(options->provider && options->provider[0]))
        sag_die("sag_separator_create: provider is required with model_path "
                "(\"cpu\" for bit-for-bit reproducible runs, or \"cuda\"/\"cuda:<id>\")");

    if (!options->instance_name || !options->instance_name[0])
        sag_die("sag_separator_create: instance_name is required -- it is hashed into "
                "the RNG payload character by character, so it must be the instance's "
                "exact name, not a stand-in");
    if (strlen(options->instance_name) >= SAG_NAME_MAX)
        sag_die("sag_separator_create: instance_name is longer than %d bytes", SAG_NAME_MAX - 1);

    if (!options->preset || !options->preset[0])
        sag_die("sag_separator_create: preset is required (a frozen size preset name, "
                "or a path to a *.toml config)");

    if (options->n_nodes < 2)
        sag_die("sag_separator_create: n_nodes = %" PRId64 " (need the depot plus at "
                "least one customer)", options->n_nodes);
    if (options->capacity <= 0)
        sag_die("sag_separator_create: capacity = %" PRId64 " (must be positive)",
                options->capacity);
    if (!options->demand) sag_die("sag_separator_create: demand is required");
}

/* ===================================================================== */
/* Lifecycle                                                             */
/* ===================================================================== */

sag_separator *sag_separator_create(const sag_separator_options *options)
{
    check_options(options);

    sag_separator *sep = (sag_separator *)calloc(1, sizeof *sep);
    if (!sep) sag_die("sag_separator_create: out of memory");

    /* ---- recipe ---- */
    if (ends_with_toml(options->preset)) {
        sag_toml_config tc;
        sag_config_from_toml(options->preset, &tc);
        sep->cfg = tc.cfg;
    } else {
        sep->cfg = sag_config_preset(options->preset);
    }
    switch (options->cross_depot) {
    case SAG_CROSS_DEPOT_DEFAULT:  break;
    case SAG_CROSS_DEPOT_ENABLED:  sep->cfg.features.partial_connect_cross_depot = true;  break;
    case SAG_CROSS_DEPOT_DISABLED: sep->cfg.features.partial_connect_cross_depot = false; break;
    default:
        sag_die("sag_separator_create: cross_depot = %d is not one of "
                "SAG_CROSS_DEPOT_DEFAULT/ENABLED/DISABLED", (int)options->cross_depot);
    }
    sag_config_check_frozen(&sep->cfg);

    /* ---- instance ---- */
    const int64_t n = options->n_nodes;
    size_t persistent_bytes = options->persistent_arena_bytes;
    if (persistent_bytes == 0) {
        const int64_t words = sag_mask_words(n);
        const int64_t cap = sep->cfg.tts.old_repair_bank_size + 1;
        persistent_bytes = (size_t)cap * (size_t)words * sizeof(uint64_t)
                         + (size_t)cap * 16u
                         + (size_t)cap * 4u * sizeof(int32_t)
                         + (size_t)n * sizeof(int64_t)
                         + SEP_PERSISTENT_SLACK;
    }
    sag_arena_init(&sep->persistent, persistent_bytes, "sep_api.persistent");
    sag_arena_init(&sep->iteration,
                   options->iteration_arena_bytes ? options->iteration_arena_bytes
                                                  : SEP_DEFAULT_ITER_BYTES,
                   "sep_api.iteration");

    int64_t *demand = SAG_NEW(&sep->persistent, int64_t, n);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t d = options->demand[i];
        if (d < 0)
            sag_die("sag_separator_create: demand[%" PRId64 "] = %" PRId64 " is negative", i, d);
        if (i == 0 && d != 0)
            sag_die("sag_separator_create: demand[0] = %" PRId64 " -- node 0 is the depot "
                    "and must have zero demand", d);
        demand[i] = d;
    }

    memset(&sep->inst, 0, sizeof sep->inst);
    snprintf(sep->inst.name, sizeof sep->inst.name, "%s", options->instance_name);
    sep->inst.n_nodes = n;
    sep->inst.depot = SAG_DEPOT;
    sep->inst.capacity = options->capacity;
    sep->inst.demand = demand;
    sep->replicas = sag_vehicle_replica_count(&sep->inst);
    if (sep->replicas <= 0)
        sag_die("sag_separator_create: K_replica = %" PRId64 " (ceil(d(V)/Q) must be positive; "
                "check demand and capacity)", sep->replicas);
    sep->inst.vehicles = sep->replicas;
    sep->inst.vehicle_count = sep->replicas;

    /* ---- cross-round state ---- */
    sep->words = sag_mask_words(n);
    sag_old_bank_init(&sep->bank, &sep->persistent, n, sep->cfg.tts.old_repair_bank_size);
    sag_key_set_init(&sep->blocked, sep->words);

    /* ---- forward backend ---- */
    const double load_start = sag_round_now_seconds();
    sep->forward = sag_forward_open_onnx(options->model_path, options->provider);
    sep->load_seconds = sag_round_now_seconds() - load_start;

    sep->env.inst = &sep->inst;
    sep->env.cfg = &sep->cfg;
    sep->env.forward = sep->forward;
    sep->env.replica_count = sep->replicas;
    sep->env.bank = &sep->bank;

    snprintf(sep->description, sizeof sep->description,
             "sagsep separate-once: forward=%s preset=%s cross_depot=%s instance=%s "
             "N=%" PRId64 " Q=%" PRId64 " K_replica=%" PRId64,
             sag_forward_describe(sep->forward), options->preset,
             sep->cfg.features.partial_connect_cross_depot ? "enabled" : "disabled",
             sep->inst.name, n, sep->inst.capacity, sep->replicas);
    return sep;
}

void sag_separator_destroy(sag_separator *sep)
{
    if (!sep) return;
    if (sep->forward) sag_forward_close(sep->forward);
    sag_key_set_free(&sep->blocked);
    sag_arena_destroy(&sep->iteration);
    sag_arena_destroy(&sep->persistent);
    free(sep);
}

const char *sag_separator_description(const sag_separator *sep)
{
    if (!sep) sag_die("sag_separator_description: NULL separator");
    return sep->description;
}

/* ===================================================================== */
/* Separate                                                              */
/* ===================================================================== */

static void validate_support(const sag_separator *sep, int64_t m,
                             const int64_t *u, const int64_t *v, const double *x)
{
    if (m <= 0 || !u || !v || !x)
        sag_die("sag_separator_separate: the support must be non-empty (got m=%" PRId64 ") -- "
                "no policy input graph can be built from an empty x*", m);
    int64_t prev_u = -1, prev_v = -1;
    for (int64_t e = 0; e < m; ++e) {
        if (u[e] < 0 || u[e] >= v[e] || v[e] >= sep->inst.n_nodes)
            sag_die("sag_separator_separate: support arc %" PRId64 " = (%" PRId64 ",%" PRId64
                    ") is not 0 <= u < v < n_nodes (%" PRId64 ")",
                    e, u[e], v[e], sep->inst.n_nodes);
        if (!(isfinite(x[e]) && x[e] > 1e-8))
            sag_die("sag_separator_separate: support value %" PRId64 " = %.17g -- values must be "
                    "finite and strictly greater than 1e-8 (filter the arcs, do not clamp them)",
                    e, x[e]);
        if (e > 0 && (u[e] < prev_u || (u[e] == prev_u && v[e] <= prev_v)))
            sag_die("sag_separator_separate: support arc %" PRId64 " = (%" PRId64 ",%" PRId64
                    ") breaks the strict (u,v) lexicographic order -- the arc order fixes the "
                    "graph edge order and therefore the cuts", e, u[e], v[e]);
        prev_u = u[e];
        prev_v = v[e];
    }
}

int64_t sag_separator_separate(sag_separator *sep,
                               int64_t m, const int64_t *u, const int64_t *v,
                               const double *x,
                               sag_separator_cuts *cuts,
                               sag_separator_stats *stats)
{
    if (!sep || !cuts) sag_die("sag_separator_separate: NULL separator or cuts");
    if (sep->pending_commit)
        sag_die("sag_separator_separate: the previous result was never committed -- call "
                "sag_separator_commit after every separate, with count=0 if you accepted none");
    validate_support(sep, m, u, v, x);

    memset(cuts, 0, sizeof *cuts);
    if (stats) memset(stats, 0, sizeof *stats);

    sag_arena *a = &sep->iteration;
    sag_arena_reset(a);
    sep->iteration_count += 1;

    const int64_t blocked_before = sep->blocked.count;

    sag_round_result round;
    sag_round_execute(&sep->env, a, sep->iteration_count, u, v, x, m,
                      sep->blocked.masks, sep->blocked.count, NULL, &round);

    const int64_t rows = round.spine.cuts.rows;
    const int64_t nnz = round.spine.cuts.nnz;

    int32_t  *customers = SAG_NEW(a, int32_t, nnz > 0 ? nnz : 1);
    double   *boundary = SAG_NEW(a, double, rows > 0 ? rows : 1);
    double   *violation = SAG_NEW(a, double, rows > 0 ? rows : 1);
    uint64_t *masks = SAG_NEW0(a, uint64_t, (rows > 0 ? rows : 1) * sep->words);

    for (int64_t r = 0; r < rows; ++r) {
        (void)sag_round_cut_key(&round.spine.cuts, r, sep->inst.n_nodes, sep->words,
                                masks + r * sep->words);
        const int64_t pool_row = round.spine.cuts.pool_rows[r];
        if (pool_row < 0 || pool_row >= round.spine.pool.unique.rows)
            sag_die("sag_separator_separate: materialized row %" PRId64 " points at pool row "
                    "%" PRId64 " of %" PRId64, r, pool_row, round.spine.pool.unique.rows);
        boundary[r] = round.spine.scored.boundary[pool_row];
        violation[r] = round.spine.scored.violation[pool_row];
    }
    for (int64_t t = 0; t < nnz; ++t) {
        customers[t] = round.spine.cuts.indices[t] - 1;
    }

    cuts->rows = rows;
    cuts->nnz = nnz;
    cuts->indptr = round.spine.cuts.indptr;
    cuts->customers = customers;
    cuts->rhs = round.spine.cuts.rhs;
    cuts->boundary = boundary;
    cuts->violation = violation;

    sep->pending_rows = rows;
    sep->pending_masks = masks;
    sep->pending_commit = true;

    if (stats) {
        stats->iteration = sep->iteration_count;
        stats->replicas = sep->replicas;
        stats->support_edges = round.support_edges;
        stats->model_edges = round.model_edges;
        stats->sweep_candidates = round.sweep_rows;
        stats->onion_candidates = round.onion_rows;
        stats->tts_candidates = round.tts_rows;
        stats->raw_candidates = round.spine.raw.rows;
        stats->unique_candidates = round.spine.dedup.unique.rows;
        stats->blocked_candidates = round.spine.diag.blocked;
        stats->nonviolated_candidates = round.spine.diag.nonviolated;
        stats->materialized_cuts = rows;
        stats->committed_cuts = sep->committed_last;
        stats->blocked_keys = blocked_before;
        stats->old_bank_rows = sag_old_bank_count(&sep->bank);
        stats->arena_used = sag_arena_used(a);
        stats->arena_high_water = sag_arena_high_water(a);
        stats->min_violation = sep->cfg.loop.min_violation;
        stats->load_seconds = sep->load_seconds;
        stats->support_seconds = round.prepare_lp_seconds;
        stats->graph_seconds = round.graph_seconds;
        stats->forward_seconds = round.forward_seconds;
        stats->family_seconds = round.sweep_seconds + round.onion_seconds + round.tts_seconds;
        stats->spine_seconds = round.spine_seconds;
        stats->total_seconds = round.total_seconds;
    }
    sep->load_seconds = 0.0;

    return rows;
}

/* ===================================================================== */
/* Commit / reset                                                        */
/* ===================================================================== */

void sag_separator_commit(sag_separator *sep, const int64_t *accepted_rows, int64_t count)
{
    if (!sep) sag_die("sag_separator_commit: NULL separator");
    if (!sep->pending_commit)
        sag_die("sag_separator_commit: no separation is awaiting a commit");
    if (count < 0 || count > sep->pending_rows)
        sag_die("sag_separator_commit: count = %" PRId64 " is outside [0, %" PRId64 "]",
                count, sep->pending_rows);
    if (count > 0 && !accepted_rows)
        sag_die("sag_separator_commit: accepted_rows is NULL but count = %" PRId64, count);

    int64_t previous = -1;
    for (int64_t i = 0; i < count; ++i) {
        const int64_t row = accepted_rows[i];
        if (row < 0 || row >= sep->pending_rows || row <= previous)
            sag_die("sag_separator_commit: accepted_rows[%" PRId64 "] = %" PRId64 " -- the rows "
                    "must be strictly ascending indices into the last result (%" PRId64 " rows)",
                    i, row, sep->pending_rows);
        sag_key_set_add(&sep->blocked, sep->pending_masks + row * sep->words);
        previous = row;
    }

    sep->committed_last = count;
    sep->pending_rows = 0;
    sep->pending_masks = NULL;
    sep->pending_commit = false;
}

void sag_separator_reset_blocked(sag_separator *sep)
{
    if (!sep) sag_die("sag_separator_reset_blocked: NULL separator");
    if (sep->pending_commit)
        sag_die("sag_separator_reset_blocked: a separation is awaiting its commit -- "
                "resetting now would drop the record of what that round produced");
    sag_key_set_clear(&sep->blocked);
}

void sag_separator_set_blocked(sag_separator *sep, int64_t rows,
                               const int64_t *indptr, const int32_t *customers)
{
    if (!sep) sag_die("sag_separator_set_blocked: NULL separator");
    if (sep->pending_commit)
        sag_die("sag_separator_set_blocked: a separation is awaiting its commit -- "
                "replacing the blocked set now would drop the record of what that "
                "round produced");
    if (rows < 0)
        sag_die("sag_separator_set_blocked: rows = %" PRId64 " is negative", rows);
    if (rows > 0 && (!indptr || !customers))
        sag_die("sag_separator_set_blocked: indptr/customers are NULL but rows = %" PRId64, rows);
    if (rows == 0) { sag_key_set_clear(&sep->blocked); return; }

    const int64_t n = sep->inst.n_nodes;

    if (indptr[0] != 0)
        sag_die("sag_separator_set_blocked: indptr[0] = %" PRId64 ", must be 0", indptr[0]);
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t lo = indptr[r], hi = indptr[r + 1];
        if (hi < lo)
            sag_die("sag_separator_set_blocked: indptr is not non-decreasing at row %" PRId64
                    " (%" PRId64 " then %" PRId64 ")", r, lo, hi);
        if (hi == lo)
            sag_die("sag_separator_set_blocked: row %" PRId64 " is empty -- an empty subset "
                    "is not a cut", r);
        int32_t prev = 0;
        for (int64_t t = lo; t < hi; ++t) {
            const int32_t c = customers[t];
            if (c <= 0 || (int64_t)c >= n)
                sag_die("sag_separator_set_blocked: row %" PRId64 " holds customer %d, outside "
                        "1..%" PRId64 " (node 0 is the depot and is never part of a cut subset)",
                        r, c, n - 1);
            if (t > lo && c <= prev)
                sag_die("sag_separator_set_blocked: row %" PRId64 " is not strictly ascending "
                        "(%d after %d)", r, c, prev);
            prev = c;
        }
    }

    const size_t mark = sag_arena_mark(&sep->iteration);
    const int64_t nnz = indptr[rows];
    int64_t *ptr_copy = SAG_NEW(&sep->iteration, int64_t, rows + 1);
    int32_t *external = SAG_NEW(&sep->iteration, int32_t, nnz > 0 ? nnz : 1);
    uint64_t *key = SAG_NEW0(&sep->iteration, uint64_t, sep->words);
    memcpy(ptr_copy, indptr, (size_t)(rows + 1) * sizeof(int64_t));
    for (int64_t t = 0; t < nnz; ++t) external[t] = customers[t] + 1;

    sag_materialized_cuts view;
    memset(&view, 0, sizeof view);
    view.rows = rows;
    view.nnz = nnz;
    view.indptr = ptr_copy;
    view.indices = external;

    sag_key_set_clear(&sep->blocked);
    for (int64_t r = 0; r < rows; ++r) {
        (void)sag_round_cut_key(&view, r, n, sep->words, key);
        sag_key_set_add(&sep->blocked, key);
    }
    sag_arena_release(&sep->iteration, mark);
}
