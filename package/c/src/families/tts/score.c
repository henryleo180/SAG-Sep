#include "families/tts/score.h"

#include <math.h>
#include <string.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "f32_sparse.h"
#include "features.h"

/* ===================================================================== */
/* 0. Sparse-acceleration helpers                                        */
/* ===================================================================== */

static inline bool f32_bits_is_pos_zero(float x)
{
    uint32_t bits;
    memcpy(&bits, &x, sizeof bits);
    return bits == 0u;
}

void sag_tts_build_adj_csr(sag_arena *a, const float *adj, int64_t n,
                           const int64_t **out_indptr, const int32_t **out_indices,
                           bool *out_clean)
{
    int64_t *indptr = SAG_NEW(a, int64_t, n + 1);
    bool     clean  = true;
    int64_t  nnz    = 0;
    for (int64_t i = 0; i < n; ++i) {
        indptr[i] = nnz;
        const float *row = adj + i * n;
        for (int64_t j = 0; j < n; ++j) {
            const float x = row[j];
            if (!f32_bits_is_pos_zero(x)) {
                ++nnz;
                if (!(x > 0.0f) || !isfinite(x)) clean = false;
            }
        }
    }
    indptr[n] = nnz;

    int32_t *indices = SAG_NEW(a, int32_t, nnz > 0 ? nnz : 1);
    int64_t  pos     = 0;
    for (int64_t i = 0; i < n; ++i) {
        const float *row = adj + i * n;
        for (int64_t j = 0; j < n; ++j)
            if (!f32_bits_is_pos_zero(row[j])) indices[pos++] = (int32_t)j;
    }

    *out_indptr  = indptr;
    *out_indices = indices;
    *out_clean   = clean;
}

static inline float tts_rhs(const sag_tts_score_ctx *sctx, float new_d, float Q32)
{
    if (sctx != NULL && sctx->rhs != NULL)
        return sag_rhs_table_lookup(sctx->rhs, new_d);
    return 2.0f * ceilf(new_d / Q32);
}

/* ===================================================================== */
/* 1. numpy primitives                                                   */
/* ===================================================================== */

int64_t sag_tts_argmax_f32(const float *v, int64_t n)
{
    if (n <= 0) sag_die("tts/score: argmax received n=%lld (caller error)", (long long)n);
    float   mp  = v[0];
    int64_t idx = 0;
    if (isnan(mp)) return 0;
    for (int64_t i = 1; i < n; ++i) {
        const float x = v[i];
        if (!(x <= mp)) {
            mp  = x;
            idx = i;
            if (isnan(x)) break;
        }
    }
    return idx;
}

static inline bool f32_lt(float a, float b)
{
    return (a < b) || ((b != b) && (a == a));
}

void sag_tts_stable_argsort_f32(const float *key, int64_t n, int32_t *idx, int32_t *scratch)
{
    for (int64_t i = 0; i < n; ++i) idx[i] = (int32_t)i;
    if (n < 2) return;
    int32_t *src = idx;
    int32_t *dst = scratch;
    for (int64_t width = 1; width < n; width *= 2) {
        for (int64_t lo = 0; lo < n; lo += 2 * width) {
            const int64_t mid = (lo + width < n) ? lo + width : n;
            const int64_t hi  = (lo + 2 * width < n) ? lo + 2 * width : n;
            int64_t i = lo, j = mid, k = lo;
            while (i < mid && j < hi) {
                if (f32_lt(key[src[j]], key[src[i]])) dst[k++] = src[j++];
                else                                  dst[k++] = src[i++];
            }
            while (i < mid) dst[k++] = src[i++];
            while (j < hi)  dst[k++] = src[j++];
        }
        int32_t *tmp = src; src = dst; dst = tmp;
    }
    if (src != idx) memcpy(idx, src, (size_t)n * sizeof(int32_t));
}

/* ===================================================================== */
/* 2. score_subsets                                                       */
/* ===================================================================== */

void sag_tts_score_subsets(const float *subsets, int64_t rows, int64_t n,
                           const float *adj, const float *deg, const float *demand,
                           double capacity, const sag_tts_score_ctx *sctx,
                           float *viol, float *lhs, float *rhs, float *d_S,
                           float *scratch)
{
    const double cap_d = (capacity > 1e-9) ? capacity : 1e-9;
    const float  cap32 = (float)cap_d;

    const bool     sparse  = (sctx != NULL && sctx->adj_indptr != NULL);
    const int64_t *indptr  = sparse ? sctx->adj_indptr  : NULL;
    const int32_t *indices = sparse ? sctx->adj_indices : NULL;

    float   *prod  = scratch;
    float   *tmp   = scratch + n;
    int32_t *S_idx = (int32_t *)(scratch + 2 * n);

    for (int64_t r = 0; r < rows; ++r) {
        const float *sel = subsets + r * n;
        float sum_deg_S, intra, d_r;

        if (sparse) {
            int64_t nS = 0;
            for (int64_t j = 0; j < n; ++j) if (sel[j] > 0.5f) S_idx[nS++] = (int32_t)j;

            sum_deg_S = sag_f32_sum_sparse_scaled(sel, deg, S_idx, nS, n);

            for (int64_t k = 0; k < nS; ++k) {
                const int64_t j  = S_idx[k];
                const float  *row = adj + j * n;
                const int64_t lo = indptr[j], hi = indptr[j + 1];
                tmp[j] = sel[j] * sag_f32_sum_sparse_scaled(row, sel, indices + lo, hi - lo, n);
            }
            intra = sag_f32_sum_sparse(tmp, S_idx, nS, n);

            d_r = sag_f32_sum_sparse_scaled(sel, demand, S_idx, nS, n);
        } else {
            /* ---- Dense fallback ---- */
            for (int64_t j = 0; j < n; ++j) prod[j] = sel[j] * deg[j];
            sum_deg_S = sag_f32_sum_numpy(prod, n);

            for (int64_t j = 0; j < n; ++j) {
                if (sel[j] > 0.5f) {
                    const float *row = adj + j * n;
                    for (int64_t u = 0; u < n; ++u) prod[u] = row[u] * sel[u];
                    tmp[j] = sel[j] * sag_f32_sum_numpy(prod, n);
                } else {
                    tmp[j] = 0.0f;
                }
            }
            intra = sag_f32_sum_numpy(tmp, n);

            for (int64_t j = 0; j < n; ++j) prod[j] = sel[j] * demand[j];
            d_r = sag_f32_sum_numpy(prod, n);
        }

        const float lhs_r = sum_deg_S - intra;
        const float rhs_r = tts_rhs(sctx, d_r, cap32);

        if (viol) viol[r] = rhs_r - lhs_r;
        if (lhs)  lhs[r]  = lhs_r;
        if (rhs)  rhs[r]  = rhs_r;
        if (d_S)  d_S[r]  = d_r;
    }
}

float sag_tts_score_one(const float *sel, int64_t n, const float *adj, const float *deg,
                        const float *demand, double capacity, const sag_tts_score_ctx *sctx,
                        float *scratch)
{
    float v = 0.0f;
    sag_tts_score_subsets(sel, 1, n, adj, deg, demand, capacity, sctx,
                          &v, NULL, NULL, NULL, scratch);
    return v;
}

/* ===================================================================== */
/* 3. local_search_1flip                                                  */
/* ===================================================================== */

void sag_tts_ls_work_init(sag_arena *a, sag_tts_ls_work *w, int64_t n_nodes)
{
    w->n_nodes    = n_nodes;
    w->deg_in_S   = SAG_NEW(a, float, n_nodes);
    w->prod       = SAG_NEW(a, float, n_nodes);
    w->in_idx     = SAG_NEW(a, int32_t, n_nodes);
    w->out_idx    = SAG_NEW(a, int32_t, n_nodes);
    w->demand_in  = SAG_NEW(a, float, n_nodes);
    w->rem_in     = SAG_NEW(a, float, n_nodes);
    w->demand_out = SAG_NEW(a, float, n_nodes);
    w->add_out    = SAG_NEW(a, float, n_nodes);
}

static inline void ls_deg_in_S_update(bool sparse, const int64_t *indptr,
                                      const int32_t *indices, const float *adj, int64_t n,
                                      float *deg_in_S, int64_t w, bool add)
{
    const float *row = adj + w * n;
    if (sparse) {
        const int64_t lo = indptr[w], hi = indptr[w + 1];
        if (add)
            for (int64_t k = lo; k < hi; ++k) { const int32_t j = indices[k]; deg_in_S[j] = deg_in_S[j] + row[j]; }
        else
            for (int64_t k = lo; k < hi; ++k) { const int32_t j = indices[k]; deg_in_S[j] = deg_in_S[j] - row[j]; }
    } else {
        if (add)
            for (int64_t j = 0; j < n; ++j) deg_in_S[j] = deg_in_S[j] + row[j];
        else
            for (int64_t j = 0; j < n; ++j) deg_in_S[j] = deg_in_S[j] - row[j];
    }
}

/* ===================================================================== */
/* 3b. 8-lane SIMD filters for the LS scans (remove/add/swap; bit-exact)  */
/* ===================================================================== */

typedef struct {
    float                    dS_f;
    float                    lhs_f;
    float                    viol_f;
    float                    Q32;
    const float             *demand;
    const float             *deg;
    const float             *deg_in_S;
    const float             *adj;
    int64_t                  n;
    const sag_tts_score_ctx *sctx;
} sag_tts_ls_scan;

static inline bool ls_scan_remove_block(const sag_tts_ls_scan *c, const int32_t *in_idx,
                                        int64_t t0, int64_t t1, float *best_val, int64_t *best_pos)
{
    for (int64_t t = t0; t < t1; ++t) {
        const int64_t v       = in_idx[t];
        const float   new_d   = c->dS_f - c->demand[v];
        const float   new_rhs = tts_rhs(c->sctx, new_d, c->Q32);
        const float   delta   = -c->deg[v] + 2.0f * c->deg_in_S[v];
        const float   new_lhs = c->lhs_f + delta;
        const float   gain    = (new_rhs - new_lhs) - c->viol_f;
        if (t == 0) {
            *best_val = gain; *best_pos = 0;
            if (isnan(gain)) return true;
        } else if (!(gain <= *best_val)) {
            *best_val = gain; *best_pos = t;
            if (isnan(gain)) return true;
        }
    }
    return false;
}

static inline bool ls_scan_add_block(const sag_tts_ls_scan *c, const int32_t *out_idx,
                                     int64_t t0, int64_t t1, float *best_val, int64_t *best_pos)
{
    for (int64_t t = t0; t < t1; ++t) {
        const int64_t v       = out_idx[t];
        const float   new_d   = c->dS_f + c->demand[v];
        const float   new_rhs = tts_rhs(c->sctx, new_d, c->Q32);
        const float   delta   = c->deg[v] - 2.0f * c->deg_in_S[v];
        const float   new_lhs = c->lhs_f + delta;
        const float   gain    = (new_rhs - new_lhs) - c->viol_f;
        if (t == 0) {
            *best_val = gain; *best_pos = 0;
            if (isnan(gain)) return true;
        } else if (!(gain <= *best_val)) {
            *best_val = gain; *best_pos = t;
            if (isnan(gain)) return true;
        }
    }
    return false;
}

static inline bool ls_scan_swap_block(const sag_tts_ls_scan *c, const int32_t *out_idx,
                                      int64_t ti, int64_t u, float d_minus, float lhs_u,
                                      int64_t tj0, int64_t tj1,
                                      float *best_val, int64_t *best_i, int64_t *best_j)
{
    for (int64_t tj = tj0; tj < tj1; ++tj) {
        const int64_t v        = out_idx[tj];
        const float   new_d    = d_minus + c->demand[v];
        const float   new_rhs  = tts_rhs(c->sctx, new_d, c->Q32);
        const float   add_term = c->deg[v] - 2.0f * c->deg_in_S[v];
        const float   cross    = 2.0f * c->adj[u * c->n + v];
        const float   new_lhs  = (lhs_u + add_term) + cross;
        const float   gain     = (new_rhs - new_lhs) - c->viol_f;
        if (ti == 0 && tj == 0) {
            *best_val = gain; *best_i = 0; *best_j = 0;
            if (isnan(gain)) return true;
        } else if (!(gain <= *best_val)) {
            *best_val = gain; *best_i = ti; *best_j = tj;
            if (isnan(gain)) return true;
        }
    }
    return false;
}

#ifdef __AVX2__
static inline float ls_hmax8(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m  = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 0x1));
    return _mm_cvtss_f32(m);
}
#endif

static double ls_1flip_core(float *sel, int64_t n, const float *adj, const float *deg,
                            const float *demand, double capacity,
                            const sag_tts_score_ctx *sctx,
                            const sag_tts_ls_params *p, sag_tts_ls_diag *diag,
                            sag_tts_ls_work *work, bool allow_simd)
{
    if (work->n_nodes < n)
        sag_die("tts/score: ls_work capacity %lld < n=%lld", (long long)work->n_nodes, (long long)n);

    float *deg_in_S = work->deg_in_S;
    float *prod     = work->prod;
    int32_t *in_idx = work->in_idx;
    int32_t *out_idx = work->out_idx;

    const bool     sparse  = (sctx != NULL && sctx->adj_indptr != NULL);
    const int64_t *indptr  = sparse ? sctx->adj_indptr  : NULL;
    const int32_t *indices = sparse ? sctx->adj_indices : NULL;

#ifdef __AVX2__
    const bool use_simd = allow_simd && sparse && (sctx->rhs == NULL);
    float *demand_in  = work->demand_in;
    float *rem_in     = work->rem_in;
    float *demand_out = work->demand_out;
    float *add_out    = work->add_out;
#else
    (void)allow_simd;
#endif

    sel[0] = 0.0f;

    double d_S, sum_deg, intra_d;
    if (sparse) {
        for (int64_t u = 0; u < n; ++u) {
            const int64_t lo = indptr[u], hi = indptr[u + 1];
            deg_in_S[u] = (lo < hi)
                ? sag_f32_sum_sparse_scaled(adj + u * n, sel, indices + lo, hi - lo, n)
                : 0.0f;
        }
        int64_t nS = 0;
        for (int64_t j = 0; j < n; ++j) if (sel[j] > 0.5f) in_idx[nS++] = (int32_t)j;
        d_S     = (double)sag_f32_sum_sparse_scaled(demand,   sel, in_idx, nS, n);
        sum_deg = (double)sag_f32_sum_sparse_scaled(deg,      sel, in_idx, nS, n);
        intra_d = (double)sag_f32_sum_sparse_scaled(deg_in_S, sel, in_idx, nS, n);
    } else {
        for (int64_t u = 0; u < n; ++u) {
            const float *row = adj + u * n;
            for (int64_t j = 0; j < n; ++j) prod[j] = row[j] * sel[j];
            deg_in_S[u] = sag_f32_sum_numpy(prod, n);
        }
        for (int64_t j = 0; j < n; ++j) prod[j] = demand[j] * sel[j];
        d_S = (double)sag_f32_sum_numpy(prod, n);
        for (int64_t j = 0; j < n; ++j) prod[j] = deg[j] * sel[j];
        sum_deg = (double)sag_f32_sum_numpy(prod, n);
        for (int64_t j = 0; j < n; ++j) prod[j] = deg_in_S[j] * sel[j];
        intra_d = (double)sag_f32_sum_numpy(prod, n);
    }

    double       lhs  = sum_deg - intra_d;
    const double Q    = (capacity > 1e-9) ? capacity : 1e-9;
    const float  Q32  = (float)Q;
    double       rhs  = 2.0 * ceil(d_S / Q);
    double       viol = rhs - lhs;

    const double initial_viol = viol;
    int64_t      steps        = 0;
    const int64_t max_size = (int64_t)floor(p->max_size_frac * (double)(n - 1));

    for (int64_t flip = 0; flip < p->max_flips; ++flip) {
        double  best_gain   = 0.0;
        int64_t best_v      = -1;
        bool    best_is_add = false;
        int64_t best_swap_u = -1;

        int64_t n_in = 0, n_out = 0;
        for (int64_t v = 1; v < n; ++v) {
            if (sel[v] > 0.5f) in_idx[n_in++] = (int32_t)v;
            else               out_idx[n_out++] = (int32_t)v;
        }

        const sag_tts_ls_scan c = {
            .dS_f = (float)d_S, .lhs_f = (float)lhs, .viol_f = (float)viol, .Q32 = Q32,
            .demand = demand, .deg = deg, .deg_in_S = deg_in_S, .adj = adj, .n = n, .sctx = sctx,
        };

#ifdef __AVX2__
        if (use_simd) {
            for (int64_t t = 0; t < n_in; ++t) {
                const int64_t v = in_idx[t];
                demand_in[t] = demand[v];
                rem_in[t]    = -deg[v] + 2.0f * deg_in_S[v];
            }
            for (int64_t t = 0; t < n_out; ++t) {
                const int64_t v = out_idx[t];
                demand_out[t] = demand[v];
                add_out[t]    = deg[v] - 2.0f * deg_in_S[v];
            }
        }
#endif

        /* --- remove (v in S): keep at least min_keep customers --- */
        if (n_in > p->min_keep && n_in > 0) {
            float   best_val = 0.0f;
            int64_t best_pos = -1;
#ifdef __AVX2__
            if (use_simd) {
                const __m256 dS_v   = _mm256_set1_ps(c.dS_f);
                const __m256 lhs_v  = _mm256_set1_ps(c.lhs_f);
                const __m256 viol_v = _mm256_set1_ps(c.viol_f);
                const __m256 Q_v    = _mm256_set1_ps(c.Q32);
                const __m256 two_v  = _mm256_set1_ps(2.0f);
                int64_t t     = 0;
                bool    broke = false;
                for (; t + 8 <= n_in; t += 8) {
                    const __m256 dem   = _mm256_loadu_ps(demand_in + t);
                    const __m256 rem   = _mm256_loadu_ps(rem_in + t);
                    const __m256 new_d = _mm256_sub_ps(dS_v, dem);
                    const __m256 nrhs  = _mm256_mul_ps(two_v, _mm256_ceil_ps(_mm256_div_ps(new_d, Q_v)));
                    const __m256 nlhs  = _mm256_add_ps(lhs_v, rem);
                    const __m256 gain  = _mm256_sub_ps(_mm256_sub_ps(nrhs, nlhs), viol_v);
                    const int    nanm  = _mm256_movemask_ps(_mm256_cmp_ps(gain, gain, _CMP_UNORD_Q));
                    bool rescan = (best_pos < 0) || (nanm != 0);
                    if (!rescan) rescan = !(ls_hmax8(gain) <= best_val);
                    if (rescan && ls_scan_remove_block(&c, in_idx, t, t + 8, &best_val, &best_pos)) { broke = true; break; }
                }
                if (!broke && t < n_in)
                    (void)ls_scan_remove_block(&c, in_idx, t, n_in, &best_val, &best_pos);
            } else
#endif
            {
                (void)ls_scan_remove_block(&c, in_idx, 0, n_in, &best_val, &best_pos);
            }
            if (best_pos >= 0 && (double)best_val > best_gain) {
                best_gain   = (double)best_val;
                best_v      = in_idx[best_pos];
                best_is_add = false;
            }
        }

        /* --- add (v not in S, limited by the size cap) --- */
        if (n_in < max_size && n_out > 0) {
            float   best_val = 0.0f;
            int64_t best_pos = -1;
#ifdef __AVX2__
            if (use_simd) {
                const __m256 dS_v   = _mm256_set1_ps(c.dS_f);
                const __m256 lhs_v  = _mm256_set1_ps(c.lhs_f);
                const __m256 viol_v = _mm256_set1_ps(c.viol_f);
                const __m256 Q_v    = _mm256_set1_ps(c.Q32);
                const __m256 two_v  = _mm256_set1_ps(2.0f);
                int64_t t     = 0;
                bool    broke = false;
                for (; t + 8 <= n_out; t += 8) {
                    const __m256 dem   = _mm256_loadu_ps(demand_out + t);
                    const __m256 add   = _mm256_loadu_ps(add_out + t);
                    const __m256 new_d = _mm256_add_ps(dS_v, dem);
                    const __m256 nrhs  = _mm256_mul_ps(two_v, _mm256_ceil_ps(_mm256_div_ps(new_d, Q_v)));
                    const __m256 nlhs  = _mm256_add_ps(lhs_v, add);
                    const __m256 gain  = _mm256_sub_ps(_mm256_sub_ps(nrhs, nlhs), viol_v);
                    const int    nanm  = _mm256_movemask_ps(_mm256_cmp_ps(gain, gain, _CMP_UNORD_Q));
                    bool rescan = (best_pos < 0) || (nanm != 0);
                    if (!rescan) rescan = !(ls_hmax8(gain) <= best_val);
                    if (rescan && ls_scan_add_block(&c, out_idx, t, t + 8, &best_val, &best_pos)) { broke = true; break; }
                }
                if (!broke && t < n_out)
                    (void)ls_scan_add_block(&c, out_idx, t, n_out, &best_val, &best_pos);
            } else
#endif
            {
                (void)ls_scan_add_block(&c, out_idx, 0, n_out, &best_val, &best_pos);
            }
            if (best_pos >= 0 && (double)best_val > best_gain) {
                best_gain   = (double)best_val;
                best_v      = out_idx[best_pos];
                best_is_add = true;
            }
        }

        /* --- swap (drop u in S, add v not in S; the size is unchanged) --- */
        if (n_in > 0 && n_out > 0 && n_in * n_out <= p->swap_max_pairs) {
            float   best_val = 0.0f;
            int64_t best_i = -1, best_j = -1;
            bool    stop = false;
            for (int64_t ti = 0; ti < n_in && !stop; ++ti) {
                const int64_t u = in_idx[ti];
                const float rem_term = -deg[u] + 2.0f * deg_in_S[u];
                const float d_minus  = (float)d_S - demand[u];
                const float lhs_u    = (float)lhs + rem_term;
#ifdef __AVX2__
                if (use_simd) {
                    const __m256 dm_v   = _mm256_set1_ps(d_minus);
                    const __m256 lu_v   = _mm256_set1_ps(lhs_u);
                    const __m256 viol_v = _mm256_set1_ps(c.viol_f);
                    const __m256 Q_v    = _mm256_set1_ps(c.Q32);
                    const __m256 two_v  = _mm256_set1_ps(2.0f);
                    int64_t tj = 0;
                    if (ti == 0) {
                        const int64_t e = (n_out < 8) ? n_out : 8;
                        if (ls_scan_swap_block(&c, out_idx, ti, u, d_minus, lhs_u, 0, e, &best_val, &best_i, &best_j)) { stop = true; break; }
                        tj = e;
                    }
                    for (; tj + 8 <= n_out; tj += 8) {
                        const __m256 dem   = _mm256_loadu_ps(demand_out + tj);
                        const __m256 add   = _mm256_loadu_ps(add_out + tj);
                        const __m256 new_d = _mm256_add_ps(dm_v, dem);
                        const __m256 nrhs  = _mm256_mul_ps(two_v, _mm256_ceil_ps(_mm256_div_ps(new_d, Q_v)));
                        const __m256 base  = _mm256_add_ps(lu_v, add);
                        const __m256 ub    = _mm256_sub_ps(_mm256_sub_ps(nrhs, base), viol_v);
                        const int    nanm  = _mm256_movemask_ps(_mm256_cmp_ps(ub, ub, _CMP_UNORD_Q));
                        bool rescan = (nanm != 0);
                        if (!rescan) rescan = !(ls_hmax8(ub) <= best_val);
                        if (rescan && ls_scan_swap_block(&c, out_idx, ti, u, d_minus, lhs_u, tj, tj + 8, &best_val, &best_i, &best_j)) { stop = true; break; }
                    }
                    if (stop) break;
                    if (tj < n_out && ls_scan_swap_block(&c, out_idx, ti, u, d_minus, lhs_u, tj, n_out, &best_val, &best_i, &best_j)) { stop = true; break; }
                } else
#endif
                {
                    if (ls_scan_swap_block(&c, out_idx, ti, u, d_minus, lhs_u, 0, n_out, &best_val, &best_i, &best_j)) { stop = true; break; }
                }
            }
            if (best_i >= 0 && (double)best_val > best_gain) {
                best_gain   = (double)best_val;
                best_swap_u = in_idx[best_i];
                best_v      = out_idx[best_j];
                best_is_add = true;
            }
        }

        if (best_v < 0 || best_gain <= p->min_gain) break;
        steps += 1;

        if (best_swap_u >= 0) {
            sel[best_swap_u] = 0.0f;
            d_S     -= (double)demand[best_swap_u];
            sum_deg -= (double)deg[best_swap_u];
            intra_d -= 2.0 * (double)deg_in_S[best_swap_u];
            ls_deg_in_S_update(sparse, indptr, indices, adj, n, deg_in_S, best_swap_u, false);
        }
        const double old_dis_v = (double)deg_in_S[best_v];
        if (best_is_add) {
            sel[best_v] = 1.0f;
            d_S     += (double)demand[best_v];
            sum_deg += (double)deg[best_v];
            intra_d += 2.0 * old_dis_v;
            ls_deg_in_S_update(sparse, indptr, indices, adj, n, deg_in_S, best_v, true);
        } else {
            sel[best_v] = 0.0f;
            d_S     -= (double)demand[best_v];
            sum_deg -= (double)deg[best_v];
            intra_d -= 2.0 * old_dis_v;
            ls_deg_in_S_update(sparse, indptr, indices, adj, n, deg_in_S, best_v, false);
        }
        if (d_S < 0.0) d_S = 0.0;
        rhs  = 2.0 * ceil(d_S / Q);
        lhs  = sum_deg - intra_d;
        viol = rhs - lhs;
    }

    sel[0] = 0.0f;
    if (diag) {
        diag->initial_violation = initial_viol;
        diag->best_violation    = viol;
        diag->gain              = viol - initial_viol;
        diag->steps             = steps;
        diag->hit_max_flips     = (p->max_flips > 0 && steps >= p->max_flips);
    }
    return viol;
}

double sag_tts_local_search_1flip(float *sel, int64_t n, const float *adj, const float *deg,
                                  const float *demand, double capacity,
                                  const sag_tts_score_ctx *sctx,
                                  const sag_tts_ls_params *p, sag_tts_ls_diag *diag,
                                  sag_tts_ls_work *work)
{
    return ls_1flip_core(sel, n, adj, deg, demand, capacity, sctx, p, diag, work, true);
}

double sag_tts_local_search_1flip_scalar_ref(float *sel, int64_t n, const float *adj,
                                             const float *deg, const float *demand,
                                             double capacity, const sag_tts_score_ctx *sctx,
                                             const sag_tts_ls_params *p, sag_tts_ls_diag *diag,
                                             sag_tts_ls_work *work)
{
    return ls_1flip_core(sel, n, adj, deg, demand, capacity, sctx, p, diag, work, false);
}
