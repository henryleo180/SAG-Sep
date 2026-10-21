#include "families/tts/adddrop.h"

#include <math.h>
#include <string.h>

#include "bitset.h"

/* ===================================================================== */
/* 0. Replica of numpy DOUBLE_pairwise_sum                                */
/* ===================================================================== */

#define SAG_PW_BLOCKSIZE 128

static double pairwise_sum_f64(const double *a, int64_t n)
{
    if (n < 8) {
        double res = 0.0;
        for (int64_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= SAG_PW_BLOCKSIZE) {
        double r[8];
        for (int k = 0; k < 8; ++k) r[k] = a[k];
        int64_t i = 8;
        const int64_t end = n - (n % 8);
        for (; i < end; i += 8) {
            r[0] += a[i + 0]; r[1] += a[i + 1]; r[2] += a[i + 2]; r[3] += a[i + 3];
            r[4] += a[i + 4]; r[5] += a[i + 5]; r[6] += a[i + 6]; r[7] += a[i + 7];
        }
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    int64_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum_f64(a, n2) + pairwise_sum_f64(a + n2, n - n2);
}

#define SAG_ADDDROP_NUMPY_BUFSIZE 8192

static double sum_numpy_f64(const double *a, int64_t n)
{
    double res = 0.0;
    int64_t off = 0;
    while (off < n) {
        int64_t chunk = n - off;
        if (chunk > SAG_ADDDROP_NUMPY_BUFSIZE) chunk = SAG_ADDDROP_NUMPY_BUFSIZE;
        res += pairwise_sum_f64(a + off, chunk);
        off += chunk;
    }
    return res;
}

void sag_tts_adddrop_diag_init(sag_tts_adddrop_diag *d)
{
    memset(d, 0, sizeof(*d));
    d->compressed_supernodes = -1;
}

static void update_best(bool *has, double *slot, double value)
{
    if (!*has || value > *slot) {
        *has  = true;
        *slot = value;
    }
}

static int32_t *build_node2super(sag_arena *a, const sag_tts_compress_view *comp)
{
    const int64_t n = comp->n_nodes;
    int32_t *node2super = SAG_NEW(a, int32_t, n);
    for (int64_t j = 0; j < n; ++j) node2super[j] = -1;
    for (int64_t i = 0; i < comp->n_super; ++i) {
        const float *row = comp->member + i * n;
        for (int64_t j = 0; j < n; ++j) {
            if (row[j] > 0.5f) {
                if (node2super[j] >= 0)
                    sag_die("tts/adddrop: compression.member is not a partition (node %lld)",
                            (long long)j);
                node2super[j] = (int32_t)i;
            }
        }
    }
    for (int64_t j = 0; j < n; ++j)
        if (node2super[j] < 0)
            sag_die("tts/adddrop: compression.member is not a partition (node %lld unassigned)",
                    (long long)j);
    if (node2super[0] != 0)
        sag_die("tts/adddrop: the depot must belong to supernode 0 (got %d)", (int)node2super[0]);
    return node2super;
}

/* ===================================================================== */
/* 1. Seed selection (`select_adddrop_seeds_from_bank`)                     */
/* ===================================================================== */

static int64_t bank_unique_masks(sag_arena *scratch, const float *bank_dense,
                                 int64_t bank_rows, int64_t n, int64_t words,
                                 uint64_t **out_masks, int64_t **out_keep)
{
    uint64_t *masks_ne = SAG_NEW(scratch, uint64_t, bank_rows * words);
    uint64_t *tmp_mask = SAG_NEW(scratch, uint64_t, words);
    int64_t   n_ne = 0;
    for (int64_t r = 0; r < bank_rows; ++r) {
        const float *src = bank_dense + r * n;
        sag_bitset_zero(tmp_mask, words);
        for (int64_t j = 1; j < n; ++j)
            if (src[j] > 0.5f) sag_bit_set(tmp_mask, j);
        if (sag_bitset_popcount(tmp_mask, words) > 0) {
            sag_bitset_copy(masks_ne + n_ne * words, tmp_mask, words);
            ++n_ne;
        }
    }
    *out_masks = masks_ne;
    *out_keep  = NULL;
    if (n_ne == 0) return 0;

    int64_t *keep = SAG_NEW(scratch, int64_t, n_ne);
    *out_keep = keep;
    return sag_tts_first_seen_unique(scratch, masks_ne, n_ne, words, keep);
}

int64_t sag_tts_adddrop_bank_unique_count(sag_arena *scratch, const float *bank_dense,
                                          int64_t bank_rows, int64_t n)
{
    if (bank_rows <= 0) return 0;
    const int64_t words = sag_mask_words(n);
    const size_t  mark  = sag_arena_mark(scratch);
    uint64_t *masks_ne = NULL;
    int64_t  *keep     = NULL;
    const int64_t n_pool = bank_unique_masks(scratch, bank_dense, bank_rows, n, words,
                                             &masks_ne, &keep);
    sag_arena_release(scratch, mark);
    return n_pool;
}

int64_t sag_tts_adddrop_seeds_from_bank(sag_arena *out_arena, sag_arena *scratch,
                                        const float *bank_dense, int64_t bank_rows,
                                        int64_t n, const float *adj, const float *deg,
                                        const float *demand, double capacity,
                                        const sag_tts_score_ctx *sctx,
                                        float **out_seeds)
{
    *out_seeds = NULL;
    if (bank_rows <= 0) return 0;
    if (out_arena == scratch)
        sag_die("tts/adddrop: out_arena and scratch must be two distinct arenas");

    const int64_t words = sag_mask_words(n);
    const size_t  mark  = sag_arena_mark(scratch);

    uint64_t *masks_ne = NULL;
    int64_t  *keep     = NULL;
    const int64_t n_pool = bank_unique_masks(scratch, bank_dense, bank_rows, n, words,
                                             &masks_ne, &keep);
    if (n_pool == 0) {
        sag_arena_release(scratch, mark);
        return 0;
    }

    float *pool = SAG_NEW0(scratch, float, n_pool * n);
    for (int64_t i = 0; i < n_pool; ++i) {
        const uint64_t *m = masks_ne + keep[i] * words;
        float *dst = pool + i * n;
        for (int64_t j = 1; j < n; ++j)
            if (sag_bit_test(m, j)) dst[j] = 1.0f;
    }

    float *viol = SAG_NEW(scratch, float, n_pool);
    float *sscr = SAG_NEW(scratch, float, 3 * n);
    sag_tts_score_subsets(pool, n_pool, n, adj, deg, demand, capacity, sctx,
                          viol, NULL, NULL, NULL, sscr);

    float   *neg   = SAG_NEW(scratch, float, n_pool);
    int32_t *order = SAG_NEW(scratch, int32_t, n_pool);
    int32_t *osort = SAG_NEW(scratch, int32_t, n_pool);
    for (int64_t i = 0; i < n_pool; ++i) neg[i] = -viol[i];
    sag_tts_stable_argsort_f32(neg, n_pool, order, osort);

    float *seeds = SAG_NEW(out_arena, float, n_pool * n);
    for (int64_t i = 0; i < n_pool; ++i)
        memcpy(seeds + i * n, pool + (int64_t)order[i] * n, (size_t)n * sizeof(float));

    sag_arena_release(scratch, mark);
    *out_seeds = seeds;
    return n_pool;
}

/* ===================================================================== */
/* 2. native AddDrop (`native_adddrop_subsets`)                            */
/* ===================================================================== */

static int64_t argmin_f64(const double *v, int64_t n)
{
    double  best = v[0];
    int64_t idx  = 0;
    if (isnan(best)) return 0;
    for (int64_t i = 1; i < n; ++i) {
        if (!(v[i] >= best)) {
            best = v[i];
            idx  = i;
            if (isnan(v[i])) break;
        }
    }
    return idx;
}

static int64_t argmax_f64(const double *v, int64_t n)
{
    double  best = v[0];
    int64_t idx  = 0;
    if (isnan(best)) return 0;
    for (int64_t i = 1; i < n; ++i) {
        if (!(v[i] <= best)) {
            best = v[i];
            idx  = i;
            if (isnan(v[i])) break;
        }
    }
    return idx;
}

bool sag_tts_adddrop_has_oversized(const sag_tts_compress_view *comp, double capacity)
{
    for (int64_t i = 1; i < comp->n_super; ++i)
        if ((double)comp->demand[i] > capacity) return true;
    return false;
}

int64_t sag_tts_native_adddrop_subsets(sag_arena *out_arena, sag_arena *scratch,
                                       const sag_tts_compress_view *comp, double capacity,
                                       const float *seeds, int64_t n_seeds,
                                       double min_violation,
                                       sag_tts_adddrop_diag *diag, float **out_super)
{
    *out_super = NULL;
    if (out_arena == scratch)
        sag_die("tts/adddrop: out_arena and scratch must be two distinct arenas");

    const int64_t ns = comp->n_super;
    const int64_t n  = comp->n_nodes;
    if (diag) {
        diag->bank_size  = n_seeds;
        diag->seed_count = n_seeds;
    }
    if (ns <= 1 || capacity <= 0.0 || n_seeds == 0) return 0;

    const size_t mark = sag_arena_mark(scratch);

    double *dem64 = SAG_NEW(scratch, double, ns);
    for (int64_t i = 0; i < ns; ++i) dem64[i] = (double)comp->demand[i];

    if (sag_tts_adddrop_has_oversized(comp, capacity)) {
        if (diag) diag->oversized_supernode = true;
        sag_arena_release(scratch, mark);
        return 0;
    }

    double *adj = SAG_NEW(scratch, double, ns * ns);
    for (int64_t i = 0; i < ns; ++i)
        for (int64_t j = 0; j < ns; ++j)
            adj[i * ns + j] = (i == j) ? 0.0 : (double)comp->aggregate[i * ns + j];

    const int32_t *node2super = build_node2super(scratch, comp);
    const int64_t  ws = sag_mask_words(ns);
    uint64_t *pmasks = SAG_NEW(scratch, uint64_t, n_seeds * ws);
    uint64_t *tmp    = SAG_NEW(scratch, uint64_t, ws);
    int64_t   n_ne   = 0;
    for (int64_t r = 0; r < n_seeds; ++r) {
        const float *src = seeds + r * n;
        sag_bitset_zero(tmp, ws);
        for (int64_t j = 0; j < n; ++j)
            if (src[j] > 0.5f) sag_bit_set(tmp, node2super[j]);
        sag_bit_clear(tmp, 0);
        if (sag_bitset_popcount(tmp, ws) > 0) {
            sag_bitset_copy(pmasks + n_ne * ws, tmp, ws);
            ++n_ne;
        }
    }
    if (n_ne == 0) {
        if (diag) diag->mapped_count = 0;
        sag_arena_release(scratch, mark);
        return 0;
    }
    int64_t *keep = SAG_NEW(scratch, int64_t, n_ne);
    const int64_t n_proj = sag_tts_first_seen_unique(scratch, pmasks, n_ne, ws, keep);
    if (diag) diag->mapped_count = n_proj;

    const double Q = capacity;
    bool    *in_set   = SAG_NEW(scratch, bool, ns);
    double  *x_to_set = SAG_NEW(scratch, double, ns);
    double  *gather   = SAG_NEW(scratch, double, ns);
    double  *scores   = SAG_NEW(scratch, double, ns);
    int32_t *idxbuf   = SAG_NEW(scratch, int32_t, ns);
    bool    *crosses  = SAG_NEW(scratch, bool, ns);
    double  *cross_buf = SAG_NEW(scratch, double, ns * ns);
    float   *outputs   = SAG_NEW0(out_arena, float, n_proj * ns);
    sag_tts_keyset out_keys;
    sag_tts_keyset_init(scratch, &out_keys, ws, n_proj);

    int64_t n_out = 0;
    for (int64_t s = 0; s < n_proj; ++s) {
        const uint64_t *seed_mask = pmasks + keep[s] * ws;
        for (int64_t i = 0; i < ns; ++i) in_set[i] = sag_bit_test(seed_mask, i) != 0;

        int64_t k = 0;
        for (int64_t i = 0; i < ns; ++i) if (in_set[i]) gather[k++] = dem64[i];
        double demand_sum = sum_numpy_f64(gather, k);
        int64_t vehicles = (int64_t)ceil(demand_sum / Q);
        if (vehicles < 1) vehicles = 1;
        double capacity_sum = (double)vehicles * Q;
        double remaining_slack = demand_sum - (double)(vehicles - 1) * Q - 1.0;

        for (int64_t i = 0; i < ns; ++i) {
            int64_t m = 0;
            const double *row = adj + i * ns;
            for (int64_t j = 0; j < ns; ++j) if (in_set[j]) gather[m++] = row[j];
            x_to_set[i] = sum_numpy_f64(gather, m);
        }

        for (;;) {
            /* --- remove --- */
            int64_t n_rem = 0;
            for (int64_t j = 1; j < ns; ++j) {
                if (in_set[j] && dem64[j] <= remaining_slack && x_to_set[j] <= 0.99) {
                    idxbuf[n_rem] = (int32_t)j;
                    scores[n_rem] = x_to_set[j];
                    ++n_rem;
                }
            }
            if (n_rem > 0) {
                const int64_t node = idxbuf[argmin_f64(scores, n_rem)];
                in_set[node] = false;
                demand_sum      -= dem64[node];
                remaining_slack -= dem64[node];
                for (int64_t i = 0; i < ns; ++i) x_to_set[i] -= adj[i * ns + node];
                if (diag) diag->remove_moves += 1;
                continue;
            }

            /* --- candidates for add / exchange --- */
            int64_t n_out_nodes = 0;
            for (int64_t j = 1; j < ns; ++j) {
                if (!in_set[j]) {
                    idxbuf[n_out_nodes]  = (int32_t)j;
                    crosses[n_out_nodes] = (demand_sum + dem64[j] > capacity_sum);
                    scores[n_out_nodes]  = x_to_set[j] + (crosses[n_out_nodes] ? 1.0 : 0.0);
                    ++n_out_nodes;
                }
            }
            if (n_out_nodes == 0) break;
            const int64_t best_offset = argmax_f64(scores, n_out_nodes);
            const int64_t new_node    = idxbuf[best_offset];
            const double  best_score  = scores[best_offset];
            const bool    best_cross  = crosses[best_offset];

            if (best_score >= 1.01) {
                in_set[new_node] = true;
                demand_sum += dem64[new_node];
                vehicles = (int64_t)ceil(demand_sum / Q);
                if (vehicles < 1) vehicles = 1;
                capacity_sum = (double)vehicles * Q;
                remaining_slack = demand_sum - (double)(vehicles - 1) * Q - 1.0;
                for (int64_t i = 0; i < ns; ++i) x_to_set[i] += adj[i * ns + new_node];
                if (diag) diag->add_moves += 1;
                continue;
            }
            if (best_score < 0.01 || best_cross) break;

            /* --- exchange --- */
            int64_t n_in_nodes = 0;
            const double *row_new = adj + new_node * ns;
            for (int64_t j = 1; j < ns; ++j) {
                if (in_set[j] && (dem64[j] - dem64[new_node]) <= remaining_slack) {
                    idxbuf[n_in_nodes] = (int32_t)j;
                    scores[n_in_nodes] = x_to_set[j] + row_new[j];
                    ++n_in_nodes;
                }
            }
            if (n_in_nodes == 0) break;
            const int64_t remove_offset = argmin_f64(scores, n_in_nodes);
            const int64_t old_node      = idxbuf[remove_offset];
            if (best_score < scores[remove_offset] + 0.01) break;

            in_set[new_node] = true;
            for (int64_t i = 0; i < ns; ++i) x_to_set[i] += adj[i * ns + new_node];
            demand_sum += dem64[new_node];
            in_set[old_node] = false;
            for (int64_t i = 0; i < ns; ++i) x_to_set[i] -= adj[i * ns + old_node];
            demand_sum -= dem64[old_node];
            vehicles = (int64_t)ceil(demand_sum / Q);
            if (vehicles < 1) vehicles = 1;
            capacity_sum = (double)vehicles * Q;
            remaining_slack = demand_sum - (double)(vehicles - 1) * Q - 1.0;
            if (diag) diag->exchange_moves += 1;
        }

        int64_t m = 0;
        for (int64_t i = 0; i < ns; ++i) {
            if (!in_set[i]) continue;
            const double *row = adj + i * ns;
            for (int64_t j = 0; j < ns; ++j)
                if (!in_set[j]) cross_buf[m++] = row[j];
        }
        const double boundary  = sum_numpy_f64(cross_buf, m);
        const double violation = 2.0 * ceil(demand_sum / Q) - boundary;
        if (diag) update_best(&diag->has_best_violation, &diag->best_violation, violation);
        if (violation <= min_violation) continue;

        sag_bitset_zero(tmp, ws);
        for (int64_t i = 0; i < ns; ++i) if (in_set[i]) sag_bit_set(tmp, i);
        if (!sag_tts_keyset_add(&out_keys, tmp)) continue;
        float *dst = outputs + n_out * ns;
        for (int64_t i = 0; i < ns; ++i) dst[i] = in_set[i] ? 1.0f : 0.0f;
        ++n_out;
    }

    if (diag) diag->candidate_count = n_out;
    sag_arena_release(scratch, mark);
    if (n_out == 0) return 0;
    *out_super = outputs;
    return n_out;
}

/* ===================================================================== */
/* 3. adddrop_proposals (the same-named reference function)                */
/* ===================================================================== */

int64_t sag_tts_adddrop_proposals(sag_arena *out_arena, sag_arena *scratch,
                                  const float *seeds, int64_t n_seeds,
                                  int64_t n, const float *adj, const float *deg,
                                  const float *demand, double capacity,
                                  const sag_tts_score_ctx *sctx,
                                  const sag_tts_config *cfg,
                                  const sag_tts_keyset *seen_keys,
                                  const sag_tts_compress_view *comp,
                                  sag_tts_adddrop_diag *diag,
                                  sag_tts_proposal *out, int64_t cap_out)
{
    if (comp->n_nodes != n)
        sag_die("tts/adddrop: compression.n_nodes=%lld != n=%lld",
                (long long)comp->n_nodes, (long long)n);
    if (out_arena == scratch)
        sag_die("tts/adddrop: out_arena and scratch must be two distinct arenas");
    if (cfg->adddrop_seed_limit != 0)
        sag_die("tts/adddrop: only adddrop_seed_limit=0 (whole pool) is implemented; got %lld",
                (long long)cfg->adddrop_seed_limit);
    if (cfg->adddrop_seed_source_count != 0)
        sag_die("tts/adddrop: cascade-source seeding is not implemented (adddrop_seed_sources must be empty)");

    const int64_t ns    = comp->n_super;
    const int64_t words = sag_mask_words(n);
    const size_t  mark  = sag_arena_mark(scratch);

    float *super = NULL;
    const int64_t n_super_sets = sag_tts_native_adddrop_subsets(
        out_arena, scratch, comp, capacity, seeds, n_seeds, cfg->min_violation,
        diag, &super);

    if (diag) {
        diag->seed_count            = n_seeds;
        diag->seed_from_old_bank    = n_seeds;
        diag->compressed_supernodes = ns - 1;
        diag->exact_candidate_count = 0;
    }
    if (n_super_sets == 0) {
        sag_arena_release(scratch, mark);
        return 0;
    }

    const int32_t *node2super = build_node2super(scratch, comp);
    float *full = SAG_NEW0(scratch, float, n_super_sets * n);
    for (int64_t r = 0; r < n_super_sets; ++r) {
        const float *srow = super + r * ns;
        float *drow = full + r * n;
        for (int64_t j = 1; j < n; ++j)
            if (srow[node2super[j]] > 0.5f) drow[j] = 1.0f;
        drow[0] = 0.0f;
    }

    float *viol = SAG_NEW(scratch, float, n_super_sets);
    float *sscr = SAG_NEW(scratch, float, 3 * n);
    sag_tts_score_subsets(full, n_super_sets, n, adj, deg, demand, capacity, sctx,
                          viol, NULL, NULL, NULL, sscr);

    const double max_size = cfg->max_size_frac * (double)(n - 1);
    uint64_t *mask_tmp = SAG_NEW(scratch, uint64_t, words);
    sag_tts_keyset out_keys;
    sag_tts_keyset_init(scratch, &out_keys, words, n_super_sets);

    int64_t n_out = 0;
    bool   has_best = false;
    double best_violation = 0.0;
    for (int64_t i = 0; i < n_super_sets; ++i) {
        const float *row = full + i * n;
        const double violation = (double)viol[i];
        sag_tts_pack_selection(mask_tmp, row, n, words);
        const double size = (double)sag_bitset_popcount(mask_tmp, words);
        update_best(&has_best, &best_violation, violation);
        if (violation <= cfg->min_violation || size < (double)cfg->min_size || size > max_size)
            continue;
        if (seen_keys && sag_tts_keyset_has(seen_keys, mask_tmp)) continue;
        if (!sag_tts_keyset_add(&out_keys, mask_tmp)) continue;

        if (n_out >= cap_out)
            sag_die("tts/adddrop: proposal overflow (cap_out=%lld) -- the caller sized it wrong",
                    (long long)cap_out);
        const double denom = (size > 1.0) ? size : 1.0;
        float    *sel_out  = SAG_NEW(out_arena, float, n);
        uint64_t *mask_out = SAG_NEW(out_arena, uint64_t, words);
        memcpy(sel_out, row, (size_t)n * sizeof(float));
        sag_bitset_copy(mask_out, mask_tmp, words);
        out[n_out].selection = sel_out;
        out[n_out].mask      = mask_out;
        out[n_out].source    = (int8_t)SAG_VAR_ADDDROP;
        out[n_out].violation = violation;
        out[n_out].score     = violation / sqrt(denom);
        ++n_out;
    }

    if (diag) {
        diag->exact_candidate_count = n_out;
        diag->raw_pool_count += n_super_sets;
        diag->dedup_count    += n_out;
        if (has_best)
            update_best(&diag->has_raw_best_violation, &diag->raw_best_violation,
                        best_violation);
    }
    sag_arena_release(scratch, mark);
    return n_out;
}
