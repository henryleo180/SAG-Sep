#include "families/tts/old_repair.h"

#include <math.h>
#include <string.h>

#include "bitset.h"
#include "f32_sparse.h"
#include "features.h"

/* ===================================================================== */
/* 0. Small helpers                                                       */
/* ===================================================================== */

void sag_tts_old_repair_diag_init(sag_tts_old_repair_diag *d)
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

/* ===================================================================== */
/* 1. Flat-graph main body (`_flat_old_repair`)                            */
/* ===================================================================== */

typedef struct {
    int64_t           min_size;
    double            max_size_frac;
    double            min_violation;
    int64_t           top_b;
    sag_tts_ls_params ls;
} flat_params;

static int64_t flat_old_repair(sag_arena *out_arena, sag_arena *scratch,
                               const float *rows_src, int64_t rows, int64_t n,
                               const float *adj, const float *deg, const float *demand,
                               double capacity, const sag_tts_score_ctx *sctx,
                               const flat_params *p,
                               sag_tts_old_repair_diag *diag,
                               sag_tts_proposal *out, int64_t cap_out)
{
    const int64_t words   = sag_mask_words(n);
    const int64_t n_cust  = n - 1;
    const double  max_size = p->max_size_frac * (double)n_cust;

    const size_t mark = sag_arena_mark(scratch);

    uint64_t *masks_ok = NULL;
    int64_t   n_ok     = 0;
    if (rows > 0) {
        masks_ok = SAG_NEW(scratch, uint64_t, rows * words);
        uint64_t *tmp_mask = SAG_NEW(scratch, uint64_t, words);
        for (int64_t r = 0; r < rows; ++r) {
            const float *src = rows_src + r * n;
            sag_bitset_zero(tmp_mask, words);
            for (int64_t i = 1; i < n; ++i)
                if (src[i] > 0.5f) sag_bit_set(tmp_mask, i);
            const double size = (double)sag_bitset_popcount(tmp_mask, words);
            if (size >= (double)p->min_size && size <= max_size) {
                sag_bitset_copy(masks_ok + n_ok * words, tmp_mask, words);
                ++n_ok;
            }
        }
    }

    int64_t  n_pool = 0;
    float    *pool  = NULL;
    if (n_ok > 0) {
        int64_t *keep = SAG_NEW(scratch, int64_t, n_ok);
        n_pool = sag_tts_first_seen_unique(scratch, masks_ok, n_ok, words, keep);
        pool   = SAG_NEW0(scratch, float, n_pool * n);
        for (int64_t i = 0; i < n_pool; ++i) {
            const uint64_t *m = masks_ok + keep[i] * words;
            float *dst = pool + i * n;
            for (int64_t j = 1; j < n; ++j)
                if (sag_bit_test(m, j)) dst[j] = 1.0f;
        }
    }

    if (diag) {
        diag->bank_size      = rows;
        diag->eligible_count = n_pool;
        diag->raw_pool_count += n_pool;
    }
    if (n_pool == 0 || p->top_b <= 0) {
        sag_arena_release(scratch, mark);
        return 0;
    }

    float *viol    = SAG_NEW(scratch, float, n_pool);
    float *sscr    = SAG_NEW(scratch, float, 3 * n);
    sag_tts_score_subsets(pool, n_pool, n, adj, deg, demand, capacity, sctx,
                          viol, NULL, NULL, NULL, sscr);

    const int64_t top_count = (p->top_b < n_pool) ? p->top_b : n_pool;
    float   *neg   = SAG_NEW(scratch, float, n_pool);
    int32_t *order = SAG_NEW(scratch, int32_t, n_pool);
    int32_t *osort = SAG_NEW(scratch, int32_t, n_pool);
    for (int64_t i = 0; i < n_pool; ++i) neg[i] = -viol[i];
    sag_tts_stable_argsort_f32(neg, n_pool, order, osort);

    if (diag) {
        diag->top_b_count = top_count;
        float mx = viol[0];
        for (int64_t i = 1; i < n_pool; ++i) {
            if (isnan(mx)) break;
            if (!(viol[i] <= mx)) mx = viol[i];
        }
        update_best(&diag->has_raw_best_violation, &diag->raw_best_violation, (double)mx);
    }

    sag_tts_keyset out_keys;
    sag_tts_keyset_init(scratch, &out_keys, words, top_count);
    sag_tts_ls_work ls_work;
    sag_tts_ls_work_init(scratch, &ls_work, n);
    float    *cand = SAG_NEW(scratch, float, n);
    uint64_t *cmask_tmp = SAG_NEW(scratch, uint64_t, words);

    int64_t n_out = 0;
    for (int64_t t = 0; t < top_count; ++t) {
        const int64_t index = order[t];
        const float  *seed  = pool + index * n;
        const double  seed_violation = (double)viol[index];

        double candidate_violation;
        memcpy(cand, seed, (size_t)n * sizeof(float));
        if (seed_violation > p->min_violation) {
            if (diag) diag->still_violated_count += 1;
            candidate_violation = seed_violation;
        } else {
            if (diag) diag->repair_attempt_count += 1;
            sag_tts_ls_diag ls_diag = {0.0, 0.0, 0.0, 0, false};
            candidate_violation = sag_tts_local_search_1flip(
                cand, n, adj, deg, demand, capacity, sctx, &p->ls, &ls_diag, &ls_work);
            if (diag) {
                update_best(&diag->has_ls_best_violation, &diag->ls_best_violation,
                            candidate_violation);
                update_best(&diag->has_ls_best_gain, &diag->ls_best_gain, ls_diag.gain);
                if (ls_diag.steps > diag->ls_max_steps) diag->ls_max_steps = ls_diag.steps;
                diag->ls_hit_max_flips = diag->ls_hit_max_flips || ls_diag.hit_max_flips;
            }
            if (candidate_violation <= p->min_violation) continue;
            if (diag) diag->repair_success_count += 1;
        }

        sag_tts_pack_selection(cmask_tmp, cand, n, words);
        if (!sag_tts_keyset_add(&out_keys, cmask_tmp)) continue;

        if (n_out >= cap_out)
            sag_die("tts/old_repair: proposal overflow (cap_out=%lld) -- the caller sized it wrong",
                    (long long)cap_out);
        const double size  = (double)sag_bitset_popcount(cmask_tmp, words);
        const double denom = (size > 1.0) ? size : 1.0;

        float    *sel_out  = SAG_NEW(out_arena, float, n);
        uint64_t *mask_out = SAG_NEW(out_arena, uint64_t, words);
        memcpy(sel_out, cand, (size_t)n * sizeof(float));
        sag_bitset_copy(mask_out, cmask_tmp, words);
        out[n_out].selection = sel_out;
        out[n_out].mask      = mask_out;
        out[n_out].source    = (int8_t)SAG_VAR_OLD_REPAIR;
        out[n_out].violation = candidate_violation;
        out[n_out].score     = candidate_violation / sqrt(denom);
        ++n_out;
    }

    if (diag) {
        diag->dedup_count    += n_out;
        diag->candidate_count = n_out;
    }
    sag_arena_release(scratch, mark);
    return n_out;
}

/* ===================================================================== */
/* 2. Compressed-graph dispatch entry (`old_repair_proposals`)              */
/* ===================================================================== */

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
                    sag_die("tts/old_repair: compression.member is not a partition (node %lld "
                            "belongs to supernodes %d and %lld at once)",
                            (long long)j, (int)node2super[j], (long long)i);
                node2super[j] = (int32_t)i;
            }
        }
    }
    for (int64_t j = 0; j < n; ++j)
        if (node2super[j] < 0)
            sag_die("tts/old_repair: compression.member is not a partition (node %lld unassigned)",
                    (long long)j);
    if (node2super[0] != 0)
        sag_die("tts/old_repair: the depot must belong to supernode 0 (got %d)", (int)node2super[0]);
    return node2super;
}

int64_t sag_tts_old_repair_proposals(sag_arena *out_arena, sag_arena *scratch,
                                     const float *bank_dense, int64_t bank_rows,
                                     int64_t n, const float *adj, const float *deg,
                                     const float *demand, double capacity,
                                     const sag_tts_score_ctx *sctx,
                                     const sag_tts_config *cfg,
                                     const sag_tts_compress_view *comp,
                                     sag_tts_old_repair_diag *diag,
                                     sag_tts_proposal *out, int64_t cap_out)
{
    if (comp->n_nodes != n)
        sag_die("tts/old_repair: compression.n_nodes=%lld != n=%lld",
                (long long)comp->n_nodes, (long long)n);
    if (cfg->compress_mode != SAG_COMPRESS_FULL)
        sag_die("tts/old_repair: only compress_mode=full is implemented (minimal surface)");
    if (out_arena == scratch)
        sag_die("tts/old_repair: out_arena and scratch must be two distinct arenas"
                " (releasing scratch would roll back the proposal payloads too)");

    flat_params p = {
        .min_size      = cfg->min_size,
        .max_size_frac = cfg->max_size_frac,
        .min_violation = cfg->min_violation,
        .top_b         = cfg->old_repair_top_b,
        .ls = {
            .max_flips      = cfg->max_flips,
            .min_gain       = cfg->min_gain,
            .max_size_frac  = cfg->max_size_frac,
            .swap_max_pairs = cfg->ls_swap_max_pairs,
            .min_keep       = cfg->min_size,
        },
    };

    if (comp->n_super >= n)
        return flat_old_repair(out_arena, scratch, bank_dense, bank_rows, n,
                               adj, deg, demand, capacity, sctx, &p, diag, out, cap_out);

    const int64_t n_super = comp->n_super;
    const int64_t words   = sag_mask_words(n);
    const size_t  mark    = sag_arena_mark(scratch);

    const int32_t *node2super = build_node2super(scratch, comp);
    float *projected = NULL;
    if (bank_rows > 0) {
        projected = SAG_NEW0(scratch, float, bank_rows * n_super);
        for (int64_t r = 0; r < bank_rows; ++r) {
            const float *src = bank_dense + r * n;
            float *dst = projected + r * n_super;
            for (int64_t j = 0; j < n; ++j)
                if (src[j] > 0.5f) dst[node2super[j]] = 1.0f;
            dst[0] = 0.0f;
        }
    }

    float *adj_super = SAG_NEW(scratch, float, n_super * n_super);
    memcpy(adj_super, comp->aggregate, (size_t)(n_super * n_super) * sizeof(float));
    for (int64_t i = 0; i < n_super; ++i) adj_super[i * n_super + i] = 0.0f;

    const int64_t *sup_indptr = NULL;
    const int32_t *sup_indices = NULL;
    bool           sup_clean   = false;
    sag_tts_build_adj_csr(scratch, adj_super, n_super, &sup_indptr, &sup_indices, &sup_clean);
    const sag_tts_score_ctx sctx_super = {
        .adj_indptr  = sup_clean ? sup_indptr : NULL,
        .adj_indices = sup_indices,
        .rhs         = (sctx != NULL) ? sctx->rhs : NULL,
    };

    float *deg_super = SAG_NEW(scratch, float, n_super);
    if (sup_clean)
        for (int64_t i = 0; i < n_super; ++i)
            deg_super[i] = sag_f32_sum_sparse(adj_super + i * n_super,
                                              sup_indices + sup_indptr[i],
                                              sup_indptr[i + 1] - sup_indptr[i], n_super);
    else
        for (int64_t i = 0; i < n_super; ++i)
            deg_super[i] = sag_f32_sum_numpy(adj_super + i * n_super, n_super);

    flat_params sp = p;
    sp.min_size          = 1;
    sp.max_size_frac     = 1.0;
    sp.ls.max_size_frac  = 1.0;
    sp.ls.min_keep       = 1;

    const int64_t cap_super = p.top_b > 0 ? p.top_b : 1;
    sag_tts_proposal *super_props = SAG_NEW(scratch, sag_tts_proposal, cap_super);
    const int64_t n_super_props = flat_old_repair(out_arena, scratch, projected, bank_rows,
                                                  n_super, adj_super, deg_super,
                                                  comp->demand, capacity, &sctx_super, &sp, diag,
                                                  super_props, cap_super);

    const double max_size = p.max_size_frac * (double)(n - 1);
    sag_tts_keyset out_keys;
    sag_tts_keyset_init(scratch, &out_keys, words, (n_super_props > 0) ? n_super_props : 1);
    float    *full  = SAG_NEW(scratch, float, n);
    uint64_t *fmask = SAG_NEW(scratch, uint64_t, words);
    float    *sscr  = SAG_NEW(scratch, float, 3 * n);

    int64_t n_out = 0;
    for (int64_t i = 0; i < n_super_props; ++i) {
        const float *ssel = super_props[i].selection;
        memset(full, 0, (size_t)n * sizeof(float));
        for (int64_t j = 1; j < n; ++j)
            if (ssel[node2super[j]] > 0.5f) full[j] = 1.0f;
        full[0] = 0.0f;

        sag_tts_pack_selection(fmask, full, n, words);
        const double size = (double)sag_bitset_popcount(fmask, words);
        if (size < (double)p.min_size || size > max_size) continue;
        if (sag_tts_keyset_has(&out_keys, fmask)) continue;

        const double violation =
            (double)sag_tts_score_one(full, n, adj, deg, demand, capacity, sctx, sscr);
        if (violation <= p.min_violation) continue;
        sag_tts_keyset_add(&out_keys, fmask);

        if (n_out >= cap_out)
            sag_die("tts/old_repair: proposal overflow (cap_out=%lld)", (long long)cap_out);
        const double denom = (size > 1.0) ? size : 1.0;
        float    *sel_out  = SAG_NEW(out_arena, float, n);
        uint64_t *mask_out = SAG_NEW(out_arena, uint64_t, words);
        memcpy(sel_out, full, (size_t)n * sizeof(float));
        sag_bitset_copy(mask_out, fmask, words);
        out[n_out].selection = sel_out;
        out[n_out].mask      = mask_out;
        out[n_out].source    = (int8_t)SAG_VAR_OLD_REPAIR;
        out[n_out].violation = violation;
        out[n_out].score     = violation / sqrt(denom);
        ++n_out;
    }

    if (diag) {
        diag->compressed_supernodes = n_super - 1;
        diag->candidate_count       = n_out;
    }
    sag_arena_release(scratch, mark);
    return n_out;
}
