#include "families/onion.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bitset.h"
#include "csr.h"
#include "features.h"

/* ------------------------------------------------------------------ */
/* 0. Primitives reproducing the reference numeric semantics           */
/* ------------------------------------------------------------------ */

static inline int npy_lt_f32(float a, float b)
{
    return a < b || (b != b && a == a);
}

static void stable_argsort_f32(const float *key, int64_t n, int32_t *idx, int32_t *scratch)
{
    for (int64_t i = 0; i < n; ++i) idx[i] = (int32_t)i;

    for (int64_t width = 1; width < n; width += width) {
        for (int64_t lo = 0; lo < n; lo += 2 * width) {
            const int64_t mid = (lo + width     < n) ? lo + width     : n;
            const int64_t hi  = (lo + 2 * width < n) ? lo + 2 * width : n;
            int64_t i = lo, j = mid, k = lo;
            while (i < mid && j < hi)
                scratch[k++] = npy_lt_f32(key[idx[j]], key[idx[i]]) ? idx[j++] : idx[i++];
            while (i < mid) scratch[k++] = idx[i++];
            while (j < hi)  scratch[k++] = idx[j++];
        }
        memcpy(idx, scratch, (size_t)n * sizeof(int32_t));
    }
}

static void check_int24_precondition(const sag_candidate_ctx *ctx)
{
    const int64_t limit = (int64_t)1 << 24;
    if (ctx->capacity >= limit)
        sag_die("onion: capacity %" PRId64 " >= 2**24 breaks the float32 exactness "
                "precondition", ctx->capacity);
    if (ctx->n_nodes <= 0) return;
    int64_t dmax = ctx->demand[0], total = 0;
    for (int64_t i = 0; i < ctx->n_nodes; ++i) {
        if (ctx->demand[i] > dmax) dmax = ctx->demand[i];
        total += ctx->demand[i];
    }
    if (dmax >= limit || total >= limit)
        sag_die("onion: demand max=%" PRId64 " sum=%" PRId64 " >= 2**24 breaks the "
                "float32 exactness precondition",
                dmax, total);
}

/* ------------------------------------------------------------------ */
/* 1. generate_onion                                                    */
/* ------------------------------------------------------------------ */

void sag_generate_onion(sag_arena *a, const sag_candidate_ctx *ctx,
                        const sag_onion_config *cfg, sag_candidate_batch *out)
{
    if (!a || !ctx || !cfg || !out) sag_die("onion: NULL argument");

    if (cfg->mode != SAG_ONION_MODE_MEAN)
        sag_die("onion: mode is frozen to 'mean' (max/vote/off are not implemented)");
    if (cfg->cutpoint != SAG_ONION_CUT_BAND_FIRST)
        sag_die("onion: cutpoint is frozen to 'band_first' (band_argmax is not implemented)");
    if (cfg->m_max_cap != 0)
        sag_die("onion: m_max_cap=%" PRId64 " is not implemented (frozen recipe: always 0 = no cap)",
                cfg->m_max_cap);
    if (cfg->skip_below_M != 0)
        sag_die("onion: skip_below_M=%" PRId64 " is not implemented (frozen recipe: always 0)",
                cfg->skip_below_M);

    const int64_t n = ctx->n_nodes;
    if (ctx->probs_rows != ctx->vehicle_count)
        sag_die("onion: ctx.probs_rows=%" PRId64 " != ctx.vehicle_count=%" PRId64
                " (the row count must come from probs itself)", ctx->probs_rows, ctx->vehicle_count);
    const int64_t k_count = ctx->vehicle_count;
    if (n < 0 || k_count < 0)
        sag_die("onion: n_nodes=%" PRId64 " vehicle_count=%" PRId64 " must be non-negative", n, k_count);

    if (!ctx->demand) sag_die("onion: ctx is missing demand");
    if (ctx->capacity > 0) check_int24_precondition(ctx);

    if (k_count == 0 || (double)ctx->capacity <= 0.0 || n <= 1) {
        *out = sag_batch_alloc(a, n >= 1 ? n : 1, 0, 0);
        return;
    }
    if (!ctx->probs || !ctx->lp)
        sag_die("onion: ctx is missing probs/lp");
    if (ctx->lp->n_nodes != n)
        sag_die("onion: lp->n_nodes %" PRId64 " != ctx->n_nodes %" PRId64,
                ctx->lp->n_nodes, n);
    if (!ctx->lp->degree || !ctx->lp->indptr || !ctx->lp->indices || !ctx->lp->data_f32)
        sag_die("onion: the LP snapshot is missing degree/CSR, which the onion violation chain needs");

    const int64_t L = n - 1;
    const int64_t words = sag_mask_words(n);

    int64_t demand_total = 0;
    for (int64_t i = 1; i < n; ++i) demand_total += ctx->demand[i];
    const int64_t m_cap_bound = (demand_total + ctx->capacity - 1) / ctx->capacity;
    if (m_cap_bound <= 0) {
        *out = sag_batch_alloc(a, n, 0, 0);
        return;
    }

    /* --- Small buffers that must survive into the second pass --- */
    int32_t  *ranked   = SAG_NEW(a, int32_t,  L);
    int64_t  *cand     = SAG_NEW(a, int64_t,  m_cap_bound);
    int32_t  *band     = SAG_NEW(a, int32_t,  m_cap_bound);
    uint8_t  *keep     = SAG_NEW0(a, uint8_t, m_cap_bound);
    uint64_t *rowmask  = SAG_NEW0(a, uint64_t, words);

    int64_t n_cand = 0, n_keep = 0, nnz_keep = 0;

    {
        const size_t mark = sag_arena_mark(a);
        float    *prio    = SAG_NEW(a, float,   L);
        float    *negkey  = SAG_NEW(a, float,   L);
        int32_t  *ordi    = SAG_NEW(a, int32_t, L);
        int32_t  *scratch = SAG_NEW(a, int32_t, L);
        int64_t  *ceils   = SAG_NEW(a, int64_t, L);
        float    *col     = SAG_NEW(a, float,   n);
        float    *tmp     = SAG_NEW(a, float,   n);
        uint64_t *member  = SAG_NEW0(a, uint64_t, words);

        for (int64_t j = 0; j < L; ++j) prio[j] = ctx->probs[1 + j];
        for (int64_t v = 1; v < k_count; ++v) {
            const float *p = ctx->probs + v * n;
            for (int64_t j = 0; j < L; ++j) prio[j] += p[1 + j];
        }
        const float kf = (float)k_count;
        for (int64_t j = 0; j < L; ++j) prio[j] /= kf;

        for (int64_t j = 0; j < L; ++j) negkey[j] = -prio[j];
        stable_argsort_f32(negkey, L, ordi, scratch);
        for (int64_t r = 0; r < L; ++r) ranked[r] = ordi[r] + 1;

        const double qdiv = ((double)ctx->capacity > 1e-9) ? (double)ctx->capacity : 1e-9;
        double cum = 0.0;
        for (int64_t r = 0; r < L; ++r) {
            const float dj = (float)ctx->demand[ranked[r]];
            cum += (double)dj;
            ceils[r] = (int64_t)ceil(cum / qdiv);
        }
        const int64_t m_cap = ceils[L - 1];
        if (m_cap != m_cap_bound)
            sag_die("onion: m_cap self-check failed (float64 chain %" PRId64 " vs integer ceildiv %"
                    PRId64 ")", m_cap, m_cap_bound);

        const float qf = (float)qdiv;
        const float min_viol = (float)cfg->min_violation;
        int64_t cursor = 0;

        for (int64_t r = 0; r < L; ++r) {
            if (r > 0 && ceils[r] == ceils[r - 1]) continue;
            const int64_t m_int = ceils[r];
            if (m_int <= 0 || m_int > m_cap) continue;

            if (n_cand >= m_cap_bound)
                sag_die("onion: candidate count exceeds the m_cap bound %" PRId64 " (bands must not repeat)",
                        m_cap_bound);

            for (; cursor <= r; ++cursor) sag_bit_set(member, ranked[cursor]);

            float sum_deg = 0.0f;
            for (int64_t j = 0; j < n; ++j)
                if (sag_bit_test(member, j)) sum_deg += ctx->lp->degree[j];

            memset(col, 0, (size_t)n * sizeof(float));
            for (int64_t i = 0; i < n; ++i) {
                if (!sag_bit_test(member, i)) continue;
                const int64_t lo = ctx->lp->indptr[i], hi = ctx->lp->indptr[i + 1];
                for (int64_t t = lo; t < hi; ++t)
                    col[ctx->lp->indices[t]] += ctx->lp->data_f32[t];
            }
            for (int64_t j = 0; j < n; ++j)
                tmp[j] = sag_bit_test(member, j) ? col[j] : 0.0f;
            const float intra = sag_f32_sum_numpy(tmp, n);

            float d_S = 0.0f;
            for (int64_t j = 0; j < n; ++j)
                if (sag_bit_test(member, j)) d_S += (float)ctx->demand[j];

            const float lhs  = sum_deg - intra;
            const float rhs  = 2.0f * ceilf(d_S / qf);
            const float viol = rhs - lhs;

            cand[n_cand] = r;
            band[n_cand] = (int32_t)m_int;
            if (viol > min_viol) {
                keep[n_cand] = 1;
                n_keep += 1;
                nnz_keep += r + 1;
            }
            n_cand += 1;
        }

        sag_arena_release(a, mark);
    }

    /* ------------- Second pass: build the output batch from `keep` ------------- */
    *out = sag_batch_alloc(a, n, n_keep, nnz_keep);
    if (n_keep == 0) return;

    int64_t cursor = 0;
    for (int64_t c = 0; c < n_cand; ++c) {
        for (; cursor <= cand[c]; ++cursor) sag_bit_set(rowmask, ranked[cursor]);
        if (!keep[c]) continue;
        sag_batch_push_mask(out, rowmask, (int8_t)SAG_FAMILY_ONION, band[c],
                            (int8_t)SAG_VAR_BAND_FIRST, sag_quiet_nan());
    }
}
