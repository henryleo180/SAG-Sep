#define _POSIX_C_SOURCE 200809L

#include "round.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bitset.h"
#include "csr.h"
#include "families/onion.h"
#include "families/sweep.h"
#include "features.h"

/* ===================================================================== */
/* 1. Timing                                                             */
/* ===================================================================== */

double sag_round_now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ===================================================================== */
/* 2. Blocked-key set                                                    */
/* ===================================================================== */

static void keyset_rehash(sag_key_set *s, int64_t new_cap)
{
    free(s->hash);
    s->hash_cap = new_cap;
    s->hash = (int32_t *)malloc((size_t)new_cap * sizeof(int32_t));
    if (!s->hash) sag_die("round: failed to grow the blocked-key hash (%lld slots)", (long long)new_cap);
    for (int64_t i = 0; i < new_cap; ++i) s->hash[i] = -1;
    for (int64_t r = 0; r < s->count; ++r) {
        const uint64_t *m = s->masks + r * s->words;
        uint64_t h = sag_bitset_hash(m, s->words) & (uint64_t)(new_cap - 1);
        while (s->hash[h] >= 0) h = (h + 1) & (uint64_t)(new_cap - 1);
        s->hash[h] = (int32_t)r;
    }
}

void sag_key_set_init(sag_key_set *s, int64_t words)
{
    memset(s, 0, sizeof *s);
    s->words = words;
    s->cap = 1024;
    s->masks = (uint64_t *)malloc((size_t)s->cap * (size_t)words * sizeof(uint64_t));
    if (!s->masks) sag_die("round: initial allocation of the blocked-key set failed");
    keyset_rehash(s, 4096);
}

void sag_key_set_free(sag_key_set *s)
{
    free(s->masks);
    free(s->hash);
    memset(s, 0, sizeof *s);
}

void sag_key_set_clear(sag_key_set *s)
{
    s->count = 0;
    for (int64_t i = 0; i < s->hash_cap; ++i) s->hash[i] = -1;
}

bool sag_key_set_contains(const sag_key_set *s, const uint64_t *mask)
{
    uint64_t h = sag_bitset_hash(mask, s->words) & (uint64_t)(s->hash_cap - 1);
    while (s->hash[h] >= 0) {
        const int32_t row = s->hash[h];
        if (sag_bitset_equal(s->masks + (int64_t)row * s->words, mask, s->words)) return true;
        h = (h + 1) & (uint64_t)(s->hash_cap - 1);
    }
    return false;
}

bool sag_key_set_add(sag_key_set *s, const uint64_t *mask)
{
    if (sag_key_set_contains(s, mask)) return false;
    if (s->count == s->cap) {
        const int64_t new_cap = s->cap * 2;
        uint64_t *grown = (uint64_t *)realloc(s->masks,
                            (size_t)new_cap * (size_t)s->words * sizeof(uint64_t));
        if (!grown) sag_die("round: failed to grow the blocked-key set (%lld rows)", (long long)new_cap);
        s->masks = grown;
        s->cap = new_cap;
    }
    sag_bitset_copy(s->masks + s->count * s->words, mask, s->words);
    s->count += 1;
    if ((s->count + 1) * 2 > s->hash_cap) keyset_rehash(s, s->hash_cap * 2);
    else {
        uint64_t h = sag_bitset_hash(mask, s->words) & (uint64_t)(s->hash_cap - 1);
        while (s->hash[h] >= 0) h = (h + 1) & (uint64_t)(s->hash_cap - 1);
        s->hash[h] = (int32_t)(s->count - 1);
    }
    return true;
}

/* ===================================================================== */
/* 3. Old-cut bank                                                       */
/* ===================================================================== */

static void bank_hash_rebuild(sag_old_bank *b)
{
    for (int64_t i = 0; i < b->hash_cap; ++i) b->hash[i] = -1;
    b->hash_live = 0;
    b->hash_tomb = 0;
    for (int32_t r = b->head; r >= 0; r = b->next[r]) {
        uint64_t h = sag_bitset_hash(b->masks + (int64_t)r * b->words, b->words)
                   & (uint64_t)(b->hash_cap - 1);
        while (b->hash[h] >= 0) h = (h + 1) & (uint64_t)(b->hash_cap - 1);
        b->hash[h] = r;
        b->hash_live += 1;
    }
}

void sag_old_bank_init(sag_old_bank *b, sag_arena *a, int64_t n_nodes, int64_t limit)
{
    if (limit <= 0) sag_die("round: old_repair_bank_size must be > 0 (got %lld)", (long long)limit);
    memset(b, 0, sizeof *b);
    b->n_nodes = n_nodes;
    b->words = sag_mask_words(n_nodes);
    b->limit = limit;
    b->cap = limit + 1;
    b->masks = SAG_NEW0(a, uint64_t, b->cap * b->words);
    b->prev = SAG_NEW(a, int32_t, b->cap);
    b->next = SAG_NEW(a, int32_t, b->cap);
    int64_t hcap = 1;
    while (hcap < 4 * b->cap) hcap <<= 1;
    b->hash_cap = hcap;
    b->hash = SAG_NEW(a, int32_t, hcap);
    for (int64_t i = 0; i < hcap; ++i) b->hash[i] = -1;
    for (int64_t i = 0; i < b->cap; ++i) {
        b->next[i] = (int32_t)(i + 1 < b->cap ? i + 1 : -1);
        b->prev[i] = -1;
    }
    b->free_head = 0;
    b->head = b->tail = -1;
    b->count = 0;
    b->scratch = SAG_NEW0(a, uint64_t, b->words);
}

static int32_t bank_find(const sag_old_bank *b, const uint64_t *mask)
{
    uint64_t h = sag_bitset_hash(mask, b->words) & (uint64_t)(b->hash_cap - 1);
    while (b->hash[h] != -1) {
        const int32_t r = b->hash[h];
        if (r >= 0 && sag_bitset_equal(b->masks + (int64_t)r * b->words, mask, b->words))
            return r;
        h = (h + 1) & (uint64_t)(b->hash_cap - 1);
    }
    return -1;
}

static void bank_list_unlink(sag_old_bank *b, int32_t r)
{
    if (b->prev[r] >= 0) b->next[b->prev[r]] = b->next[r]; else b->head = b->next[r];
    if (b->next[r] >= 0) b->prev[b->next[r]] = b->prev[r]; else b->tail = b->prev[r];
    b->prev[r] = b->next[r] = -1;
}

static void bank_list_push_back(sag_old_bank *b, int32_t r)
{
    b->prev[r] = b->tail;
    b->next[r] = -1;
    if (b->tail >= 0) b->next[b->tail] = r; else b->head = r;
    b->tail = r;
}

static void bank_hash_insert(sag_old_bank *b, int32_t r)
{
    if ((b->hash_live + b->hash_tomb + 1) * 2 > b->hash_cap) bank_hash_rebuild(b);
    uint64_t h = sag_bitset_hash(b->masks + (int64_t)r * b->words, b->words)
               & (uint64_t)(b->hash_cap - 1);
    while (b->hash[h] >= 0) h = (h + 1) & (uint64_t)(b->hash_cap - 1);
    if (b->hash[h] == -2) b->hash_tomb -= 1;
    b->hash[h] = r;
    b->hash_live += 1;
}

static void bank_hash_erase(sag_old_bank *b, int32_t r)
{
    uint64_t h = sag_bitset_hash(b->masks + (int64_t)r * b->words, b->words)
               & (uint64_t)(b->hash_cap - 1);
    while (b->hash[h] != -1) {
        if (b->hash[h] == r) { b->hash[h] = -2; b->hash_live -= 1; b->hash_tomb += 1; return; }
        h = (h + 1) & (uint64_t)(b->hash_cap - 1);
    }
    sag_die("round: record %d to be deleted is not in the bank hash", r);
}

static void bank_pop_oldest(sag_old_bank *b)
{
    const int32_t r = b->head;
    if (r < 0) sag_die("round: asked to evict from an empty bank");
    bank_hash_erase(b, r);
    bank_list_unlink(b, r);
    b->next[r] = b->free_head;
    b->free_head = r;
    b->count -= 1;
}

void sag_old_bank_remember(sag_old_bank *b, const sag_materialized_cuts *cuts)
{
    if (!cuts || cuts->rows <= 0) return;
    uint64_t *scratch_mask = b->scratch;
    for (int64_t r = 0; r < cuts->rows; ++r) {
        sag_bitset_zero(scratch_mask, b->words);
        const int64_t lo = cuts->indptr[r], hi = cuts->indptr[r + 1];
        for (int64_t t = lo; t < hi; ++t) {
            const int64_t local = (int64_t)cuts->indices[t] - 1;
            if (local > 0 && local < b->n_nodes) sag_bit_set(scratch_mask, local);
        }
        const int32_t found = bank_find(b, scratch_mask);
        if (found >= 0) {
            bank_list_unlink(b, found);
            bank_list_push_back(b, found);
        } else {
            if (b->free_head < 0) sag_die("round: bank record pool exhausted (cap=%lld)", (long long)b->cap);
            const int32_t slot = b->free_head;
            b->free_head = b->next[slot];
            b->prev[slot] = b->next[slot] = -1;
            sag_bitset_copy(b->masks + (int64_t)slot * b->words, scratch_mask, b->words);
            bank_list_push_back(b, slot);
            bank_hash_insert(b, slot);
            b->count += 1;
        }
        while (b->count > b->limit) bank_pop_oldest(b);
    }
}

void sag_old_bank_snapshot(const sag_old_bank *b, sag_arena *a, sag_bank *out)
{
    memset(out, 0, sizeof *out);
    out->n_nodes = b->n_nodes;
    out->words = b->words;
    out->rows = b->count;
    if (b->count == 0) {
        int64_t *indptr = SAG_NEW0(a, int64_t, 1);
        out->indptr = indptr;
        out->mask = NULL;
        out->indices = NULL;
        return;
    }
    uint64_t *mask = SAG_NEW(a, uint64_t, b->count * b->words);
    int64_t  *indptr = SAG_NEW(a, int64_t, b->count + 1);
    int64_t nnz = 0;
    int64_t row = 0;
    for (int32_t r = b->head; r >= 0; r = b->next[r], ++row) {
        const uint64_t *src = b->masks + (int64_t)r * b->words;
        sag_bitset_copy(mask + row * b->words, src, b->words);
        nnz += sag_bitset_popcount(src, b->words);
    }
    int32_t *indices = SAG_NEW(a, int32_t, nnz > 0 ? nnz : 1);
    indptr[0] = 0;
    row = 0;
    int64_t pos = 0;
    for (int32_t r = b->head; r >= 0; r = b->next[r], ++row) {
        const uint64_t *src = b->masks + (int64_t)r * b->words;
        pos += sag_bitset_to_indices(src, b->words, indices + pos, nnz - pos);
        indptr[row + 1] = pos;
    }
    if (row != b->count || pos != nnz)
        sag_die("round: bank snapshot self-check failed (rows %lld/%lld nnz %lld/%lld)",
                (long long)row, (long long)b->count, (long long)pos, (long long)nnz);
    out->mask = mask;
    out->indptr = indptr;
    out->indices = indices;
}

/* ===================================================================== */
/* 4. Cut key of a materialized row                                      */
/* ===================================================================== */

int64_t sag_round_cut_key(const sag_materialized_cuts *cuts, int64_t row,
                          int64_t n_nodes, int64_t words, uint64_t *key)
{
    sag_bitset_zero(key, words);
    const int64_t lo = cuts->indptr[row], hi = cuts->indptr[row + 1];
    int64_t size = 0;
    int32_t prev_ext = 0;
    for (int64_t t = lo; t < hi; ++t) {
        const int32_t ext = cuts->indices[t];
        if (ext < 2 || (int64_t)ext > n_nodes)
            sag_die("round: materialized row %lld has external id %d out of range [2,%lld]"
                    " (a candidate batch never contains the depot)", (long long)row, ext,
                    (long long)n_nodes);
        if (ext <= prev_ext)
            sag_die("round: materialized row %lld has ids that are not strictly ascending (%d after %d)",
                    (long long)row, ext, prev_ext);
        prev_ext = ext;
        sag_bit_set(key, ext - 1);
        size += 1;
    }
    return size;
}

/* ===================================================================== */
/* 5. The per-round engine                                               */
/* ===================================================================== */

void sag_round_execute(const sag_round_env *env, sag_arena *a, int64_t iteration,
                       const int64_t *sup_u, const int64_t *sup_v, const double *sup_x,
                       int64_t support_count,
                       const uint64_t *blocked, int64_t n_blocked,
                       const sag_round_probes *probes,
                       sag_round_result *out)
{
    if (!env || !a || !out) sag_die("sag_round_execute: NULL argument");
    if (!env->inst || !env->cfg || !env->forward || !env->bank)
        sag_die("sag_round_execute: incomplete environment");

    memset(out, 0, sizeof *out);

    const sag_instance *inst = env->inst;
    const sag_config   *cfg = env->cfg;
    const int64_t n = inst->n_nodes;
    const double round_start = sag_round_now_seconds();

    /* --- unified_prepare part 1: the LP support snapshot --- */
    const double prep_start = sag_round_now_seconds();
    sag_lp_support lpsup;
    sag_build_lp_support(a, sup_u, sup_v, sup_x, support_count, n,
                         cfg->lp.lp_support_edge_threshold, &lpsup);
    const double prepare_lp_time = sag_round_now_seconds() - prep_start;

    /* --- model graph features --- */
    const double graph_start = sag_round_now_seconds();
    sag_graph_features gf;
    memset(&gf, 0, sizeof gf);
    sag_build_graph_features(a, inst, &lpsup, env->replica_count,
                             sup_u, sup_v, sup_x, support_count,
                             cfg->features.partial_connect_cross_depot, &gf);
    const double graph_time = sag_round_now_seconds() - graph_start;

    /* --- forward --- */
    const double forward_start = sag_round_now_seconds();
    const float *probs = sag_forward_probs_graph(env->forward, &gf);
    const double forward_time = sag_round_now_seconds() - forward_start;

    /* --- unified_prepare part 2: bank snapshot + candidate context --- */
    const double ctx_start = sag_round_now_seconds();
    sag_bank snap;
    sag_old_bank_snapshot(env->bank, a, &snap);
    if (probes && probes->bank) probes->bank(probes->user, iteration, &snap);
    sag_candidate_ctx ctx;
    memset(&ctx, 0, sizeof ctx);
    ctx.probs = probs;
    ctx.lp = &lpsup;
    ctx.demand = inst->demand;
    ctx.capacity = inst->capacity;
    ctx.n_nodes = n;
    ctx.vehicle_count = env->replica_count;
    ctx.probs_rows    = env->replica_count;
    ctx.iteration = iteration;
    ctx.seed = cfg->mbp_seed;
    ctx.tts_seed = cfg->tts.seed;
    ctx.instance = inst->name;
    ctx.bank = &snap;
    const double prepare_time = prepare_lp_time + (sag_round_now_seconds() - ctx_start);

    const double decode_start = sag_round_now_seconds();
    sag_candidate_batch sweep_batch, onion_batch, tts_batch;
    const double sweep_start = sag_round_now_seconds();
    sag_generate_sweep(a, &ctx, &cfg->sweep, &cfg->growth, &sweep_batch);
    const double sweep_time = sag_round_now_seconds() - sweep_start;
    const double onion_start = sag_round_now_seconds();
    sag_generate_onion(a, &ctx, &cfg->onion, &onion_batch);
    const double onion_time = sag_round_now_seconds() - onion_start;
    const double decode_wall = sag_round_now_seconds() - decode_start;
    const double tts_start = sag_round_now_seconds();
    if (cfg->tts.enabled) {
        sag_generate_tts(a, &ctx, &cfg->tts, &tts_batch, &out->tts_diag);
    } else {
        tts_batch = sag_batch_alloc(a, n, 0, 0);
        out->tts_diag.vehicles = NULL;
        out->tts_diag.count = 0;
        out->tts_diag.cap = 0;
    }
    const double tts_time = sag_round_now_seconds() - tts_start;

    const double family_elapsed[SAG_FAMILY_COUNT] = { sweep_time, onion_time, 0.0 };
    const double spine_start = sag_round_now_seconds();
    sag_run_spine(a, &ctx, &sweep_batch, &onion_batch, &tts_batch,
                  blocked, n_blocked, &cfg->budget, family_elapsed,
                  cfg->compute_cut_values, &out->spine);
    const double spine_time = sag_round_now_seconds() - spine_start;
    if (probes && probes->spine)
        probes->spine(probes->user, iteration, &sweep_batch, &onion_batch,
                      &tts_batch, blocked, n_blocked, &out->spine);
    sag_old_bank_remember(env->bank, &out->spine.cuts);

    const double total_time = sag_round_now_seconds() - round_start;

    double mean_set = 0.0;
    if (out->spine.cuts.rows > 0) {
        const size_t mk = sag_arena_mark(a);
        double *sz = SAG_NEW(a, double, out->spine.cuts.rows);
        for (int64_t r = 0; r < out->spine.cuts.rows; ++r)
            sz[r] = (double)out->spine.cuts.sizes[r];
        mean_set = sag_f64_sum_numpy(sz, out->spine.cuts.rows) / (double)out->spine.cuts.rows;
        sag_arena_release(a, mk);
    }

    out->mean_set_size = mean_set;
    out->support_edges = lpsup.n_edges;
    out->model_edges = gf.n_edges;
    out->sweep_rows = sweep_batch.rows;
    out->onion_rows = onion_batch.rows;
    out->tts_rows = tts_batch.rows;
    out->prepare_lp_seconds = prepare_lp_time;
    out->graph_seconds = graph_time;
    out->forward_seconds = forward_time;
    out->prepare_seconds = prepare_time;
    out->sweep_seconds = sweep_time;
    out->onion_seconds = onion_time;
    out->tts_seconds = tts_time;
    out->decode_wall_seconds = decode_wall;
    out->spine_seconds = spine_time;
    out->total_seconds = total_time;
}
