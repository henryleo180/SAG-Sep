#include <stdlib.h>
#include "families/tts/core.h"

#include <inttypes.h>
#include <math.h>
#include <string.h>

#include "bitset.h"
#include "csr.h"
#include "families/tts/adddrop.h"
#include "families/tts/compress.h"
#include "families/tts/fcapfix.h"
#include "families/tts/grow.h"
#include "families/tts/old_repair.h"
#include "families/tts/proposal.h"
#include "families/tts/score.h"
#include "features.h"
#include "rng.h"

/* ===================================================================== */
/* 1. Diagnostics helpers. Keys are emitted in first-insertion order.     */
/* ===================================================================== */

static void diag_push_key(sag_tts_diag_vehicle *d, sag_tts_diag_key key)
{
    if (d->n_keys >= SAG_TTS_DK_COUNT)
        sag_die("tts/core: diagnostics key table overflow (%lld)", (long long)d->n_keys);
    d->key_order[d->n_keys++] = (int8_t)key;
}

static void diag_touch_source(sag_tts_diag_vehicle *d, int slot)
{
    if (d->n_stats == 0) diag_push_key(d, SAG_TTS_DK_SOURCES);
    if (!d->sources_present[slot]) {
        d->sources_present[slot] = true;
        d->stats_order[d->n_stats++] = (int8_t)slot;
    }
}

/* ===================================================================== */
/* 2. Per-vehicle candidate generation                                    */
/* ===================================================================== */

typedef struct {
    sag_tts_proposal *items;
    int64_t           count;
    int64_t           cap;
    uint64_t         *mask_pool;
    int64_t           words;
} proposal_pool;

static sag_tts_proposal *pool_next(proposal_pool *p)
{
    if (p->count >= p->cap)
        sag_die("tts/core: proposal pool overflow (cap=%lld) -- the reserved size was miscomputed", (long long)p->cap);
    sag_tts_proposal *it = &p->items[p->count];
    it->selection = NULL;
    it->mask      = p->mask_pool + p->count * p->words;
    it->origin_stage = SAG_TTS_STAGE_RAW;
    it->origin_seed  = -1;
    p->count += 1;
    return it;
}

static void pool_copy_in(proposal_pool *p, const sag_tts_proposal *src)
{
    sag_tts_proposal *dst = pool_next(p);
    sag_bitset_copy(dst->mask, src->mask, p->words);
    dst->source    = src->source;
    dst->violation = src->violation;
    dst->score     = src->score;
    dst->origin_stage = src->origin_stage;
    dst->origin_seed  = src->origin_seed;
}

typedef struct {
    sag_arena                 *scratch;
    const sag_tts_compression *comp;
    sag_tts_compress_view      view;
    const float               *bank_dense;
    int64_t                    n_bank;
    sag_tts_ls_work           *ls_work;
    const sag_tts_score_ctx   *sctx;
} vehicle_env;

static void vehicle_proposals(sag_arena *a, int64_t vehicle, const float *probs_v,
                              const float *adj, const float *deg, const float *dem_f32,
                              double capacity, const sag_tts_config *cfg,
                              const sag_candidate_ctx *ctx, const vehicle_env *env,
                              sag_tts_diag_vehicle *diag, proposal_pool *props)
{
    const int64_t n      = ctx->n_nodes;
    const int64_t max_prefix = cfg->max_prefix > 0 ? cfg->max_prefix
                                                   : sag_preset_max_prefix_for_customers(n - 1);
    const int64_t n_cust = n - 1;
    const int64_t words  = props->words;
    const sag_tts_compression *comp = env->comp;
    const size_t  mark   = sag_arena_mark(a);

    /* ---- The two pool-shaped sources, in the order they are appended ---- */
    const float *part_rows[2]   = { NULL, NULL };
    int64_t      part_count[2]  = { 0, 0 };
    int8_t       part_source[2] = { SAG_VAR_LP_GROW, SAG_VAR_FCAPFIX };
    const int32_t *part_seed[2] = { NULL, NULL };
    int64_t      n_parts = 0;

    /* --- lp_grow: native rules, growing on the compressed graph, all vehicles --- */
    if (cfg->lp_grow_seeds > 0) {
        sag_pcg64 rng = sag_derive_source_rng(cfg->seed, ctx->instance, ctx->iteration,
                                              vehicle, "lp_grow");
        const int64_t ns = comp->n_super;
        const int64_t m  = ns - 1;
        double *adj_grow = SAG_NEW(a, double, (m > 0 ? m * m : 1));
        sag_tts_grow_adj_from_aggregate(comp->aggregate, ns, adj_grow);
        double *dem_super = SAG_NEW(a, double, (m > 0 ? m : 1));
        for (int64_t i = 0; i < m; ++i) dem_super[i] = (double)comp->demand[i + 1];

        sag_tts_grow_out g;
        sag_tts_lp_grow(a, adj_grow, dem_super, comp->sizes + 1, comp->internal_x + 1, m,
                        capacity, cfg->lp_grow_seeds, &rng, cfg->min_violation,
                        max_prefix, &g);
        float *grow_full = SAG_NEW(a, float, (g.rows > 0 ? g.rows : 1) * n);
        for (int64_t r = 0; r < g.rows; ++r) {
            const uint64_t *row = g.masks + r * g.words;
            float          *dst = grow_full + r * n;
            for (int64_t i = 0; i < n; ++i) {
                const int64_t s = comp->labels[i];
                dst[i] = (s >= 1 && sag_bit_test(row, s - 1)) ? 1.0f : 0.0f;
            }
        }
        part_rows[n_parts]   = grow_full;
        part_count[n_parts]  = g.rows;
        part_source[n_parts] = SAG_VAR_LP_GROW;
        part_seed[n_parts]   = g.row_seed;
        n_parts += 1;
    }

    /* --- fcapfix: vehicle 0 only --- */
    if (vehicle == 0) {
        sag_tts_fcapfix_out fx;
        sag_tts_fcapfix(a, n, capacity, cfg->fcapfix_rounds, cfg->fcapfix_seed_limit,
                        probs_v, comp, &fx);
        float *fcap_full = SAG_NEW(a, float, (fx.rows > 0 ? fx.rows : 1) * n);
        for (int64_t r = 0; r < fx.rows; ++r) {
            const uint64_t *row = fx.masks + r * fx.words;
            float          *dst = fcap_full + r * n;
            for (int64_t i = 0; i < n; ++i) dst[i] = sag_bit_test(row, i) ? 1.0f : 0.0f;
        }
        part_rows[n_parts]   = fcap_full;
        part_count[n_parts]  = fx.rows;
        part_source[n_parts] = SAG_VAR_FCAPFIX;
        part_seed[n_parts]   = NULL;
        n_parts += 1;
    }

    /* --- old_repair: vehicle 0 only, and only when the bank is non-empty --- */
    int64_t           n_old     = 0;
    const int64_t     cap_old   = cfg->old_repair_top_b + 1;
    sag_tts_proposal *old_props = SAG_NEW(a, sag_tts_proposal, cap_old);
    if (vehicle == 0 && env->n_bank > 0) {
        sag_tts_old_repair_diag ordiag;
        sag_tts_old_repair_diag_init(&ordiag);
        n_old = sag_tts_old_repair_proposals(a, env->scratch, env->bank_dense, env->n_bank,
                                             n, adj, deg, dem_f32, capacity, env->sctx, cfg,
                                             &env->view, diag ? &ordiag : NULL, old_props, cap_old);
        if (diag) {
            diag->has_old_repair = true;
            diag->old_repair     = ordiag;
            diag_push_key(diag, SAG_TTS_DK_OLD_REPAIR);
            diag_touch_source(diag, 0);
        }
    }

    /* --- adddrop: vehicle 0 only, seeded from the whole old-cut bank --- */
    int64_t           n_add     = 0;
    const int64_t     cap_add   = (env->n_bank > 0 ? env->n_bank : 1);
    sag_tts_proposal *add_props = SAG_NEW(a, sag_tts_proposal, cap_add);
    if (vehicle == 0) {
        const bool early = comp->n_super > 1 && capacity > 0.0 &&
                           sag_tts_adddrop_has_oversized(&env->view, capacity);

        float  *seeds   = NULL;
        int64_t n_seeds = 0;
        if (cfg->adddrop_seed_old_bank && env->n_bank > 0) {
            if (early)
                n_seeds = sag_tts_adddrop_bank_unique_count(env->scratch, env->bank_dense,
                                                            env->n_bank, n);
            else
                n_seeds = sag_tts_adddrop_seeds_from_bank(a, env->scratch, env->bank_dense,
                                                          env->n_bank, n, adj, deg, dem_f32,
                                                          capacity, env->sctx, &seeds);
        }
        sag_tts_keyset seen;
        sag_tts_keyset_init(a, &seen, words, n_old > 0 ? n_old : 1);
        for (int64_t i = 0; i < n_old; ++i) sag_tts_keyset_add(&seen, old_props[i].mask);

        sag_tts_adddrop_diag addiag;
        sag_tts_adddrop_diag_init(&addiag);
        if (early && n_seeds > 0) {
            n_add = 0;
            addiag.bank_size             = n_seeds;
            addiag.seed_count            = n_seeds;
            addiag.oversized_supernode   = true;
            addiag.seed_from_old_bank    = n_seeds;
            addiag.compressed_supernodes = comp->n_super - 1;
            addiag.exact_candidate_count = 0;
        } else {
            n_add = sag_tts_adddrop_proposals(a, env->scratch, seeds, n_seeds, n, adj, deg,
                                              dem_f32, capacity, env->sctx, cfg, &seen,
                                              &env->view, diag ? &addiag : NULL,
                                              add_props, cap_add);
        }
        if (diag) {
            diag->has_adddrop = true;
            diag->adddrop     = addiag;
            diag_push_key(diag, SAG_TTS_DK_ADDDROP);
            if (addiag.candidate_count > 0) diag_touch_source(diag, 1);
        }
    }

    /* --- supplemental = adddrop + old_repair --- */
    for (int64_t i = 0; i < n_add; ++i) pool_copy_in(props, &add_props[i]);
    for (int64_t i = 0; i < n_old; ++i) pool_copy_in(props, &old_props[i]);
    const int64_t n_supplemental = props->count;

    /* ---- Pool-shaped sources: merge, gate, dedup, score, top-B, local search ---- */
    int64_t n_pool_rows = 0;
    for (int64_t p = 0; p < n_parts; ++p) if (part_count[p] > 0) n_pool_rows += part_count[p];
    if (n_pool_rows == 0) { sag_arena_release(a, mark); return; }

    float   *pool      = SAG_NEW(a, float, n_pool_rows * n);
    int8_t  *pool_src  = SAG_NEW(a, int8_t, n_pool_rows);
    int32_t *pool_seed = SAG_NEW(a, int32_t, n_pool_rows);
    {
        int64_t w = 0;
        for (int64_t p = 0; p < n_parts; ++p) {
            if (part_count[p] <= 0) continue;
            for (int64_t r = 0; r < part_count[p]; ++r, ++w) {
                memcpy(pool + w * n, part_rows[p] + r * n, (size_t)n * sizeof(float));
                pool_src[w]  = part_source[p];
                pool_seed[w] = part_seed[p] ? part_seed[p][r] : -1;
            }
        }
    }
    const float min_size32 = (float)(double)cfg->min_size;
    const float max_size32 = (float)(cfg->max_size_frac * (double)n_cust);
    int64_t n_keep = 0;
    for (int64_t r = 0; r < n_pool_rows; ++r) {
        float *row = pool + r * n;
        row[0] = 0.0f;
        const float sz = sag_f32_sum_numpy(row, n);
        if (!(sz >= min_size32 && sz <= max_size32)) continue;
        if (n_keep != r) {
            memmove(pool + n_keep * n, row, (size_t)n * sizeof(float));
            pool_src[n_keep]  = pool_src[r];
            pool_seed[n_keep] = pool_seed[r];
        }
        n_keep += 1;
    }
    if (n_keep == 0) { sag_arena_release(a, mark); return; }

    uint64_t *pool_masks = SAG_NEW(a, uint64_t, n_keep * words);
    for (int64_t r = 0; r < n_keep; ++r)
        sag_tts_pack_selection(pool_masks + r * words, pool + r * n, n, words);
    int64_t *keep = SAG_NEW(a, int64_t, n_keep);
    const int64_t n_uniq = sag_tts_first_seen_unique(a, pool_masks, n_keep, words, keep);
    if (n_uniq == 0) { sag_arena_release(a, mark); return; }
    for (int64_t r = 0; r < n_uniq; ++r) {
        const int64_t src = keep[r];
        if (src != r) {
            memmove(pool + r * n, pool + src * n, (size_t)n * sizeof(float));
            sag_bitset_copy(pool_masks + r * words, pool_masks + src * words, words);
            pool_src[r]  = pool_src[src];
            pool_seed[r] = pool_seed[src];
        }
    }

    float *viol  = SAG_NEW(a, float, n_uniq);
    float *score = SAG_NEW(a, float, n_uniq);
    float *scr2  = SAG_NEW(a, float, 3 * n);
    sag_tts_score_subsets(pool, n_uniq, n, adj, deg, dem_f32, capacity, env->sctx,
                          viol, NULL, NULL, NULL, scr2);
    for (int64_t r = 0; r < n_uniq; ++r) {
        float nnz = sag_f32_sum_numpy(pool + r * n, n);
        if (nnz < 1.0f) nnz = 1.0f;
        score[r] = viol[r] / sqrtf(nnz);
    }
    float *negscore = SAG_NEW(a, float, n_uniq);
    for (int64_t r = 0; r < n_uniq; ++r) negscore[r] = -score[r];
    int32_t *order = SAG_NEW(a, int32_t, n_uniq);
    int32_t *osrc  = SAG_NEW(a, int32_t, n_uniq);
    sag_tts_stable_argsort_f32(negscore, n_uniq, order, osrc);

    int64_t top_count = (cfg->top_b > 1) ? cfg->top_b : 1;
    if (top_count > n_uniq) top_count = n_uniq;

    sag_tts_keyset out_keys;
    sag_tts_keyset_init(a, &out_keys, words, props->cap);
    for (int64_t i = 0; i < n_supplemental; ++i)
        sag_tts_keyset_add(&out_keys, props->items[i].mask);

    const float min_viol32 = (float)cfg->min_violation;
    for (int64_t t = 0; t < top_count; ++t) {
        const int64_t r = order[t];
        if (!(viol[r] > min_viol32)) continue;
        if (!sag_tts_keyset_add(&out_keys, pool_masks + r * words)) continue;
        sag_tts_proposal *it = pool_next(props);
        sag_bitset_copy(it->mask, pool_masks + r * words, words);
        it->source    = pool_src[r];
        it->violation = (double)viol[r];
        it->score     = (double)score[r];
        it->origin_stage = SAG_TTS_STAGE_RAW;
        it->origin_seed  = pool_seed[r];
    }

    const sag_tts_ls_params ls = {
        .max_flips      = cfg->max_flips,
        .min_gain       = cfg->min_gain,
        .max_size_frac  = cfg->max_size_frac,
        .swap_max_pairs = cfg->ls_swap_max_pairs,
        .min_keep       = cfg->min_size,
    };
    const int64_t ls_count = (cfg->top_l < top_count) ? cfg->top_l : top_count;
    float    *new_sel  = SAG_NEW(a, float, n);
    uint64_t *new_mask = SAG_NEW(a, uint64_t, words);
    for (int64_t t = 0; t < ls_count; ++t) {
        const int64_t r = order[t];
        memcpy(new_sel, pool + r * n, (size_t)n * sizeof(float));
        const double new_viol = sag_tts_local_search_1flip(new_sel, n, adj, deg, dem_f32,
                                                           capacity, env->sctx, &ls, NULL,
                                                           env->ls_work);
        if (new_viol <= cfg->min_violation) continue;
        const double new_size = (double)sag_f32_sum_numpy(new_sel, n);
        if (new_size < (double)cfg->min_size) continue;
        sag_tts_pack_selection(new_mask, new_sel, n, words);
        if (!sag_tts_keyset_add(&out_keys, new_mask)) continue;
        sag_tts_proposal *it = pool_next(props);
        sag_bitset_copy(it->mask, new_mask, words);
        it->source    = pool_src[r];
        it->violation = new_viol;
        it->origin_stage = SAG_TTS_STAGE_LS;
        it->origin_seed  = pool_seed[r];
        it->score     = new_viol / sqrt(new_size > 1.0 ? new_size : 1.0);
    }

    sag_arena_release(a, mark);
}

/* ===================================================================== */
/* 3. _finalize_proposals                                                 */
/* ===================================================================== */

static int64_t finalize_proposals(sag_arena *a, proposal_pool *props, const sag_tts_config *cfg,
                                  sag_tts_flow_state *fs, sag_tts_diag_vehicle *diag,
                                  int64_t *out_idx)
{
    sag_tts_flow_diag  flow_diag;
    sag_tts_flow_diag *fdp = diag ? &flow_diag : NULL;
    const int64_t n_sel = sag_tts_apply_cut_flow(a, props->items, props->count, cfg, fs,
                                                 out_idx, fdp);
    if (!diag) return n_sel;

    diag_push_key(diag, SAG_TTS_DK_FLOW);
    diag->flow_input_count    = props->count;
    diag->flow_output_count   = n_sel;
    diag->flow_global_emitted = fs->emitted;
    diag->flow_vehicle_scale  = sag_tts_flow_scale(cfg, fs->vehicle_count);
    diag->flow                = flow_diag;

    static const int8_t src_of_slot[2] = { SAG_VAR_OLD_REPAIR, SAG_VAR_ADDDROP };
    int64_t emitted[2] = { 0, 0 };
    for (int k = 0; k < 2; ++k)
        for (int64_t i = 0; i < n_sel; ++i)
            if (props->items[out_idx[i]].source == src_of_slot[k]) emitted[k] += 1;
    for (int k = 0; k < 2; ++k) {
        if (emitted[k] <= 0) continue;
        diag_touch_source(diag, k);
        diag->stats_emitted_count[k] += emitted[k];
    }
    if (diag->has_old_repair) diag->old_repair_emitted_count = emitted[0];
    if (diag->has_adddrop)    diag->adddrop_emitted_count    = emitted[1];

    return n_sel;
}

/* ===================================================================== */
/* 4. Family entry point                                                 */
/* ===================================================================== */

void sag_generate_tts(sag_arena *a, const sag_candidate_ctx *ctx, const sag_tts_config *cfg,
                      sag_candidate_batch *out, sag_tts_diag *diag)
{
    const int64_t n     = ctx->n_nodes;
    const int64_t K     = ctx->vehicle_count;
    const double  cap_q = (double)ctx->capacity;
    if (diag) { diag->vehicles = NULL; diag->count = 0; diag->cap = 0; }
    if (n <= 1 || cap_q <= 0.0 || K <= 0) {
        *out = sag_batch_alloc(a, n >= 1 ? n : 1, 0, 0);
        return;
    }
    if (cfg->per_vehicle_quota <= 0 || cfg->global_quota <= 0)
        sag_die("tts/core: per_vehicle_quota=%lld / global_quota=%lld must both be positive"
                " (the unlimited branch is not implemented)",
                (long long)cfg->per_vehicle_quota, (long long)cfg->global_quota);
    if (!cfg->flow_enabled)
        sag_die("tts/core: flow_enabled=false is not implemented (it is frozen to true)");
    if (cfg->compress_mode != SAG_COMPRESS_FULL)
        sag_die("tts/core: only compress_mode=full is implemented");
    if (cfg->score_mode != SAG_SCORE_EFFICACY)
        sag_die("tts/core: only score_mode=efficacy is implemented");
    if (cfg->rng_mode != SAG_RNG_DERIVED)
        sag_die("tts/core: only rng_mode=derived is implemented");
    if (ctx->tts_seed != cfg->seed)
        sag_die("tts/core: ctx.tts_seed=%lld != cfg.seed=%lld -- the RNG derives from cfg.seed, "
                "so a mismatch means the caller built the context from the outdated "
                "docstring convention instead",
                (long long)ctx->tts_seed, (long long)cfg->seed);
    if (!cfg->lp_grow_native_rules)
        sag_die("tts/core: only lp_grow_native_rules=true is implemented");
    if (cfg->adddrop_seed_source_count != 0 || cfg->adddrop_seed_limit != 0)
        sag_die("tts/core: adddrop only implements whole-pool seeding from the old bank (sources=(), limit=0)");

    const int64_t words = sag_mask_words(n);
    int64_t cap_rows = K * cfg->per_vehicle_quota;
    if (cap_rows > cfg->global_quota) cap_rows = cfg->global_quota;
    *out = sag_batch_alloc(a, n, cap_rows, cap_rows * (n - 1));

    const size_t mark = sag_arena_mark(a);

    sag_arena scratch;
    {
        size_t bytes = sag_arena_remaining(a) / 2;
        bytes -= bytes % 64;
        if (bytes < (size_t)1 << 20)
            sag_die("tts/core: only %zu bytes left in the arena, not enough for the tts scratch (2 MiB minimum)",
                    sag_arena_remaining(a));
        scratch.base       = (unsigned char *)sag_arena_alloc(a, bytes, 64);
        scratch.cap        = bytes;
        scratch.used       = 0;
        scratch.high_water = 0;
        scratch.name       = "tts.scratch";
    }

    const float *adj = ctx->lp->dense;
    const float *deg = ctx->lp->degree;
    float *dem_f32 = SAG_NEW(a, float, n);
    for (int64_t i = 0; i < n; ++i) dem_f32[i] = (float)ctx->demand[i];

    const int64_t *adj_indptr = NULL;
    const int32_t *adj_indices = NULL;
    bool           adj_clean   = false;
    sag_tts_build_adj_csr(a, adj, n, &adj_indptr, &adj_indices, &adj_clean);
    const sag_tts_score_ctx sctx = {
        .adj_indptr  = adj_clean ? adj_indptr : NULL,
        .adj_indices = adj_indices,
        .rhs         = NULL,
    };

    const int64_t n_bank = (ctx->bank != NULL) ? ctx->bank->rows : 0;
    float *bank_dense = SAG_NEW0(a, float, (n_bank > 0 ? n_bank : 1) * n);
    for (int64_t b = 0; b < n_bank; ++b) {
        const int64_t lo = ctx->bank->indptr[b], hi = ctx->bank->indptr[b + 1];
        for (int64_t t = lo; t < hi; ++t) bank_dense[b * n + ctx->bank->indices[t]] = 1.0f;
    }

    sag_tts_compression comp;
    sag_tts_build_compression(a, adj, dem_f32, cap_q, bank_dense, n_bank, n,
                              SAG_TTS_SHRINK_EPS, &comp);

    sag_tts_flow_state fs;
    sag_tts_flow_state_init(a, &fs, K, n, words, cap_rows);

    sag_tts_ls_work ls_work;
    sag_tts_ls_work_init(a, &ls_work, n);

    vehicle_env env;
    env.scratch         = &scratch;
    env.comp            = &comp;
    env.view.n_super    = comp.n_super;
    env.view.n_nodes    = comp.n_nodes;
    env.view.member     = comp.member;
    env.view.aggregate  = comp.aggregate;
    env.view.demand     = comp.demand;
    env.bank_dense      = (n_bank > 0) ? bank_dense : NULL;
    env.n_bank          = n_bank;
    env.ls_work         = &ls_work;
    env.sctx            = &sctx;

    const int64_t cap_props = n_bank + cfg->old_repair_top_b + cfg->top_b + cfg->top_l + 8;
    proposal_pool props;
    props.items     = SAG_NEW(a, sag_tts_proposal, cap_props);
    props.mask_pool = SAG_NEW(a, uint64_t, cap_props * words);
    props.cap       = cap_props;
    props.count     = 0;
    props.words     = words;
    int64_t *sel_idx = SAG_NEW(a, int64_t, cap_props);

    sag_tts_diag_vehicle *dv_all = NULL;
    if (diag) {
        dv_all = SAG_NEW(a, sag_tts_diag_vehicle, K);
        diag->vehicles = dv_all;
        diag->cap      = K;
        diag->count    = 0;
    }

    const int64_t gq_limit = sag_tts_scaled_limit(cfg->global_quota,
                                                  sag_tts_flow_scale(cfg, ctx->vehicle_count));

    const int64_t max_rep = (cfg->max_replicas > 0 && cfg->max_replicas < K)
                                ? cfg->max_replicas : K;

    for (int64_t v = 0; v < max_rep; ++v) {
        if (gq_limit > 0 && fs.emitted >= gq_limit) break;

        sag_tts_diag_vehicle *dv = NULL;
        if (diag) {
            dv = &dv_all[v];
            memset(dv, 0, sizeof *dv);
            diag_push_key(dv, SAG_TTS_DK_COMPRESSION);
            dv->one_edge_customer_supernodes = comp.one_edge_customer_supernodes;
            dv->full_customer_supernodes     = comp.n_super - 1;
            int64_t frozen_cust = 0;
            for (int64_t s = 1; s < comp.n_super; ++s) if (comp.frozen[s]) frozen_cust += 1;
            dv->frozen_customer_supernodes = frozen_cust;
        }

        props.count = 0;
        sag_arena_reset(&scratch);
        vehicle_proposals(a, v, ctx->probs + v * n, adj, deg, dem_f32, cap_q, cfg, ctx,
                          &env, dv, &props);
        const int64_t n_sel = finalize_proposals(a, &props, cfg, &fs, dv, sel_idx);

        for (int64_t i = 0; i < n_sel; ++i) {
            const sag_tts_proposal *it = &props.items[sel_idx[i]];
            if (sag_bitset_popcount(it->mask, words) == 0) continue;
            sag_batch_push_mask(out, it->mask, (int8_t)SAG_FAMILY_TTS, (int32_t)v,
                                it->source, sag_quiet_nan());
        }
        if (diag) diag->count += 1;
    }

    if (diag && diag->count > 0) {
        const int64_t cnt = diag->count;
        sag_arena_release(a, mark);
        sag_tts_diag_vehicle *dst = SAG_NEW(a, sag_tts_diag_vehicle, cnt);
        memmove(dst, dv_all, (size_t)cnt * sizeof(sag_tts_diag_vehicle));
        diag->vehicles = dst;
        diag->cap      = cnt;
    } else {
        sag_arena_release(a, mark);
        if (diag) diag->vehicles = NULL;
    }
}

/* ===================================================================== */
/* 5. Rendering the diagnostics JSON (the tts_json slot in json_out.h)   */
/* ===================================================================== */

static void jindent(FILE *f, int depth) { for (int i = 0; i < depth * 2; ++i) fputc(' ', f); }

static void jf64(FILE *f, double v)
{
    if (isnan(v)) { fputs("NaN", f); return; }
    if (isinf(v)) { fputs(v > 0 ? "Infinity" : "-Infinity", f); return; }
    char buf[64];
    snprintf(buf, sizeof buf, "%.17g", v);
    bool plain = true;
    for (const char *p = buf; *p; ++p)
        if (*p == '.' || *p == 'e' || *p == 'E') { plain = false; break; }
    fputs(buf, f);
    if (plain) fputs(".0", f);
}

static void jkv_i64(FILE *f, int depth, const char *k, int64_t v, bool comma)
{
    jindent(f, depth); fprintf(f, "\"%s\": %lld%s\n", k, (long long)v, comma ? "," : "");
}

static void jkv_f64_or_null(FILE *f, int depth, const char *k, bool has, double v, bool comma)
{
    jindent(f, depth); fprintf(f, "\"%s\": ", k);
    if (!has) fputs("null", f); else jf64(f, v);
    fputs(comma ? ",\n" : "\n", f);
}

static void jkv_bool(FILE *f, int depth, const char *k, bool v, bool comma)
{
    jindent(f, depth);
    fprintf(f, "\"%s\": %s%s\n", k, v ? "true" : "false", comma ? "," : "");
}

static void write_source_stats(FILE *f, int depth, int64_t raw_pool_count, int64_t dedup_count,
                               bool has_raw, double raw_best, bool has_lsv, double ls_best,
                               bool has_lsg, double ls_gain, int64_t ls_max_steps,
                               bool ls_hit, int64_t emitted)
{
    fputs("{\n", f);
    jkv_i64(f, depth + 1, "raw_pool_count", raw_pool_count, true);
    jkv_i64(f, depth + 1, "dedup_count", dedup_count, true);
    jkv_f64_or_null(f, depth + 1, "raw_best_violation", has_raw, raw_best, true);
    jkv_f64_or_null(f, depth + 1, "ls_best_violation", has_lsv, ls_best, true);
    jkv_f64_or_null(f, depth + 1, "ls_best_gain", has_lsg, ls_gain, true);
    jkv_i64(f, depth + 1, "ls_max_steps", ls_max_steps, true);
    jkv_bool(f, depth + 1, "ls_hit_max_flips", ls_hit, true);
    jkv_i64(f, depth + 1, "emitted_count", emitted, false);
    jindent(f, depth); fputc('}', f);
}

static void write_flow(FILE *f, int depth, const sag_tts_diag_vehicle *d)
{
    fputs("{\n", f);
    jkv_i64(f, depth + 1, "input_count", d->flow_input_count, true);
    jkv_i64(f, depth + 1, "output_count", d->flow_output_count, true);
    jkv_i64(f, depth + 1, "global_emitted", d->flow_global_emitted, true);
    jindent(f, depth + 1); fputs("\"vehicle_scale\": ", f);
    jf64(f, d->flow_vehicle_scale); fputs(",\n", f);
    jkv_i64(f, depth + 1, "near_duplicate_hit_count", d->flow.near_duplicate_hit_count, true);
    char    keys[SAG_TTS_PAIR_SLOTS * SAG_TTS_PAIR_SLOTS][48];
    int64_t vals[SAG_TTS_PAIR_SLOTS * SAG_TTS_PAIR_SLOTS];
    int64_t n_pairs = 0;
    for (int i = 0; i < SAG_TTS_PAIR_SLOTS; ++i) {
        for (int j = 0; j < SAG_TTS_PAIR_SLOTS; ++j) {
            if (d->flow.pair_counts[i][j] == 0) continue;
            const char *from = (i < SAG_VARIANT_COUNT) ? sag_variant_names[i] : "unknown";
            const char *to   = (j < SAG_VARIANT_COUNT) ? sag_variant_names[j] : "unknown";
            snprintf(keys[n_pairs], sizeof keys[0], "%s->%s", from, to);
            vals[n_pairs] = d->flow.pair_counts[i][j];
            n_pairs += 1;
        }
    }
    for (int64_t i = 1; i < n_pairs; ++i) {
        char kb[48]; int64_t vb = vals[i];
        memcpy(kb, keys[i], sizeof kb);
        int64_t j = i - 1;
        while (j >= 0 && strcmp(keys[j], kb) > 0) {
            memcpy(keys[j + 1], keys[j], sizeof kb); vals[j + 1] = vals[j]; j -= 1;
        }
        memcpy(keys[j + 1], kb, sizeof kb); vals[j + 1] = vb;
    }
    jindent(f, depth + 1); fputs("\"near_duplicate_source_pairs\": ", f);
    if (n_pairs == 0) {
        fputs("{}\n", f);
    } else {
        fputs("{\n", f);
        for (int64_t i = 0; i < n_pairs; ++i)
            jkv_i64(f, depth + 2, keys[i], vals[i], i + 1 < n_pairs);
        jindent(f, depth + 1); fputs("}\n", f);
    }
    jindent(f, depth); fputc('}', f);
}

static void write_vehicle(FILE *f, int depth, const sag_tts_diag_vehicle *d)
{
    fputs("{\n", f);
    for (int64_t k = 0; k < d->n_keys; ++k) {
        const bool last = (k + 1 == d->n_keys);
        switch ((sag_tts_diag_key)d->key_order[k]) {
        case SAG_TTS_DK_COMPRESSION:
            jindent(f, depth + 1); fputs("\"compression\": {\n", f);
            jindent(f, depth + 2); fputs("\"mode\": \"full\",\n", f);
            jkv_i64(f, depth + 2, "one_edge_customer_supernodes",
                    d->one_edge_customer_supernodes, true);
            jkv_i64(f, depth + 2, "full_customer_supernodes", d->full_customer_supernodes, true);
            jkv_i64(f, depth + 2, "frozen_customer_supernodes",
                    d->frozen_customer_supernodes, false);
            jindent(f, depth + 1); fputs(last ? "}\n" : "},\n", f);
            break;
        case SAG_TTS_DK_OLD_REPAIR: {
            const sag_tts_old_repair_diag *o = &d->old_repair;
            const bool has_cs = (o->compressed_supernodes >= 0);
            jindent(f, depth + 1); fputs("\"old_repair\": {\n", f);
            jkv_i64(f, depth + 2, "bank_size", o->bank_size, true);
            jkv_i64(f, depth + 2, "eligible_count", o->eligible_count, true);
            jkv_i64(f, depth + 2, "top_b_count", o->top_b_count, true);
            jkv_i64(f, depth + 2, "still_violated_count", o->still_violated_count, true);
            jkv_i64(f, depth + 2, "repair_attempt_count", o->repair_attempt_count, true);
            jkv_i64(f, depth + 2, "repair_success_count", o->repair_success_count, true);
            jkv_i64(f, depth + 2, "candidate_count", o->candidate_count, true);
            jkv_i64(f, depth + 2, "emitted_count", d->old_repair_emitted_count, has_cs);
            if (has_cs)
                jkv_i64(f, depth + 2, "compressed_supernodes", o->compressed_supernodes, false);
            jindent(f, depth + 1); fputs(last ? "}\n" : "},\n", f);
            break;
        }
        case SAG_TTS_DK_SOURCES:
            jindent(f, depth + 1); fputs("\"sources\": {\n", f);
            for (int64_t s = 0; s < d->n_stats; ++s) {
                const int slot = d->stats_order[s];
                jindent(f, depth + 2);
                fprintf(f, "\"%s\": ", slot == 0 ? "old_repair" : "adddrop");
                if (slot == 0) {
                    const sag_tts_old_repair_diag *o = &d->old_repair;
                    write_source_stats(f, depth + 2, o->raw_pool_count, o->dedup_count,
                                       o->has_raw_best_violation, o->raw_best_violation,
                                       o->has_ls_best_violation, o->ls_best_violation,
                                       o->has_ls_best_gain, o->ls_best_gain,
                                       o->ls_max_steps, o->ls_hit_max_flips,
                                       d->stats_emitted_count[0]);
                } else {
                    const sag_tts_adddrop_diag *ad = &d->adddrop;
                    write_source_stats(f, depth + 2, ad->raw_pool_count, ad->dedup_count,
                                       ad->has_raw_best_violation, ad->raw_best_violation,
                                       false, 0.0, false, 0.0, 0, false,
                                       d->stats_emitted_count[1]);
                }
                fputs(s + 1 < d->n_stats ? ",\n" : "\n", f);
            }
            jindent(f, depth + 1); fputs(last ? "}\n" : "},\n", f);
            break;
        case SAG_TTS_DK_ADDDROP: {
            const sag_tts_adddrop_diag *ad = &d->adddrop;
            jindent(f, depth + 1); fputs("\"adddrop\": {\n", f);
            jkv_i64(f, depth + 2, "bank_size", ad->bank_size, true);
            jkv_i64(f, depth + 2, "seed_count", ad->seed_count, true);
            jkv_i64(f, depth + 2, "mapped_count", ad->mapped_count, true);
            jkv_i64(f, depth + 2, "remove_moves", ad->remove_moves, true);
            jkv_i64(f, depth + 2, "add_moves", ad->add_moves, true);
            jkv_i64(f, depth + 2, "exchange_moves", ad->exchange_moves, true);
            jkv_i64(f, depth + 2, "candidate_count", ad->candidate_count, true);
            jkv_f64_or_null(f, depth + 2, "best_violation", ad->has_best_violation,
                            ad->best_violation, true);
            jkv_bool(f, depth + 2, "oversized_supernode", ad->oversized_supernode, true);
            jindent(f, depth + 2); fputs("\"seed_source_counts\": {},\n", f);
            jkv_i64(f, depth + 2, "seed_from_old_bank", ad->seed_from_old_bank, true);
            jkv_i64(f, depth + 2, "compressed_supernodes", ad->compressed_supernodes, true);
            jkv_i64(f, depth + 2, "exact_candidate_count", ad->exact_candidate_count, true);
            jkv_i64(f, depth + 2, "emitted_count", d->adddrop_emitted_count, false);
            jindent(f, depth + 1); fputs(last ? "}\n" : "},\n", f);
            break;
        }
        case SAG_TTS_DK_FLOW:
            jindent(f, depth + 1); fputs("\"flow\": ", f);
            write_flow(f, depth + 1, d);
            fputs(last ? "\n" : ",\n", f);
            break;
        default:
            sag_die("tts/core: unknown diagnostics key %d", (int)d->key_order[k]);
        }
    }
    jindent(f, depth); fputc('}', f);
}

void sag_tts_diag_write_json(FILE *out, const sag_tts_diag *diag)
{
    fputs("{\n", out);
    jindent(out, 1); fputs("\"vehicles\": ", out);
    if (diag->count == 0) {
        fputs("[]\n", out);
    } else {
        fputs("[\n", out);
        for (int64_t i = 0; i < diag->count; ++i) {
            jindent(out, 2);
            write_vehicle(out, 2, &diag->vehicles[i]);
            fputs(i + 1 < diag->count ? ",\n" : "\n", out);
        }
        jindent(out, 1); fputs("]\n", out);
    }
    fputc('}', out);
}
