#include "families/growth.h"
#include "families/sweep.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "bitset.h"
#include "csr.h"
#include "features.h"

/* ------------------------------------------------------------------ */
/* 0. numpy semantic primitives                                         */
/* ------------------------------------------------------------------ */

static inline int npy_lt_f64(double a, double b)
{
    return a < b || (b != b && a == a);
}

void sag_sweep_stable_argsort(const double *key, int64_t n, int32_t *idx, int32_t *scratch)
{
    if (n < 0) sag_die("sweep: argsort length is negative (%" PRId64 ")", n);
    for (int64_t i = 0; i < n; ++i) idx[i] = (int32_t)i;

    for (int64_t width = 1; width < n; width += width) {
        for (int64_t lo = 0; lo < n; lo += 2 * width) {
            const int64_t mid = (lo + width      < n) ? lo + width      : n;
            const int64_t hi  = (lo + 2 * width  < n) ? lo + 2 * width  : n;
            int64_t i = lo, j = mid, k = lo;
            while (i < mid && j < hi)
                scratch[k++] = npy_lt_f64(key[idx[j]], key[idx[i]]) ? idx[j++] : idx[i++];
            while (i < mid) scratch[k++] = idx[i++];
            while (j < hi)  scratch[k++] = idx[j++];
        }
        memcpy(idx, scratch, (size_t)n * sizeof(int32_t));
    }
}

int64_t sag_sweep_argmax_f64(const double *v, int64_t n)
{
    if (n <= 0) sag_die("sweep: argmax_f64 received an empty array");
    double mp = v[0];
    int64_t best = 0;
    if (mp != mp) return 0;
    for (int64_t i = 1; i < n; ++i) {
        if (!(v[i] <= mp)) {
            mp = v[i];
            best = i;
            if (mp != mp) break;
        }
    }
    return best;
}

int64_t sag_sweep_argmax_f32(const float *v, int64_t n)
{
    if (n <= 0) sag_die("sweep: argmax_f32 received an empty array");
    float mp = v[0];
    int64_t best = 0;
    if (mp != mp) return 0;
    for (int64_t i = 1; i < n; ++i) {
        if (!(v[i] <= mp)) {
            mp = v[i];
            best = i;
            if (mp != mp) break;
        }
    }
    return best;
}

/* ------------------------------------------------------------------ */
/* 1. generate_sweep                                                    */
/* ------------------------------------------------------------------ */

void sag_generate_sweep(sag_arena *a, const sag_candidate_ctx *ctx,
                        const sag_sweep_config *cfg, const sag_growth_config *growth,
                        sag_candidate_batch *out)
{
    if (!a || !ctx || !cfg || !out) sag_die("sweep: NULL argument");

    if (cfg->min_size < 1)
        sag_die("sweep: min_size=%" PRId64 " < 1 is unimplemented (on the python side "
                "this is negative-slice semantics; frozen recipe min_size=2)", cfg->min_size);

    const int64_t n = ctx->n_nodes;
    if (ctx->probs_rows != ctx->vehicle_count)
        sag_die("sweep: ctx.probs_rows=%" PRId64 " != ctx.vehicle_count=%" PRId64
                " (the reference implementation only looks at probs.shape[0])", ctx->probs_rows, ctx->vehicle_count);
    const int64_t k_count = ctx->vehicle_count;
    if (n < 0 || k_count < 0)
        sag_die("sweep: n_nodes=%" PRId64 " vehicle_count=%" PRId64 " must be non-negative",
                n, k_count);

    if (k_count == 0 || n <= 1) {
        *out = sag_batch_alloc(a, n >= 1 ? n : 1, 0, 0);
        return;
    }

    if (!ctx->probs || !ctx->demand || !ctx->lp)
        sag_die("sweep: ctx is missing probs/demand/lp");
    if (ctx->lp->n_nodes != n)
        sag_die("sweep: lp->n_nodes %" PRId64 " != ctx->n_nodes %" PRId64,
                ctx->lp->n_nodes, n);
    if (!ctx->lp->dense || !ctx->lp->degree)
        sag_die("sweep: the lp snapshot is missing dense/degree (sweep's boundary chain needs them)");

    const int64_t L = n - 1;
    const int64_t g_rows = sag_growth_row_reserve(growth, L);
    const int64_t cap_rows = k_count + g_rows;
    *out = sag_batch_alloc(a, n, cap_rows, cap_rows * L);
    const int64_t words = out->words;

    const size_t mark = sag_arena_mark(a);
    double   *key    = SAG_NEW(a, double,   L);
    int32_t  *ordi   = SAG_NEW(a, int32_t,  L);
    int32_t  *scratch= SAG_NEW(a, int32_t,  L);
    int32_t  *onode  = SAG_NEW(a, int32_t,  L);
    float    *rowbuf = SAG_NEW(a, float,    L);
    float    *intra  = SAG_NEW(a, float,    L);
    double   *viol   = SAG_NEW(a, double,   L);
    uint64_t *mask   = SAG_NEW(a, uint64_t, words);

    const int sweep_ok = (n > cfg->min_size) && ((double)ctx->capacity > 0.0);
    uint64_t *taken = g_rows > 0 ? SAG_NEW0(a, uint64_t, cap_rows * words) : NULL;
    int64_t n_taken = 0;
    int64_t n_masked = cfg->min_size - 1;
    if (n_masked > L) n_masked = L;

    for (int64_t v = 0; v < k_count; ++v) {
        const float *p = ctx->probs + v * n;
        int     winner = 0;
        int64_t best = 0;

        if (sweep_ok) {
            for (int64_t j = 0; j < L; ++j) key[j] = -(double)p[1 + j];
            sag_sweep_stable_argsort(key, L, ordi, scratch);
            for (int64_t r = 0; r < L; ++r) onode[r] = ordi[r] + 1;

            memset(rowbuf, 0, (size_t)L * sizeof(float));
            for (int64_t r = 0; r < L; ++r) {
                const float *arow = ctx->lp->dense + (int64_t)onode[r] * n;
                for (int64_t c = 0; c < r; ++c) rowbuf[c] = arow[onode[c]];
                intra[r] = sag_f32_sum_numpy(rowbuf, L);
            }

            const double capacity = (double)ctx->capacity;
            double acc_deg = 0.0, acc_intra = 0.0, acc_dem = 0.0;
            for (int64_t r = 0; r < L; ++r) {
                const double dgv = (double)ctx->lp->degree[onode[r]];
                const double inv = (double)intra[r];
                const double dmv = (double)ctx->demand[onode[r]];
                acc_deg   = (r == 0) ? dgv : acc_deg   + dgv;
                acc_intra = (r == 0) ? inv : acc_intra + inv;
                acc_dem   = (r == 0) ? dmv : acc_dem   + dmv;
                viol[r] = 2.0 * ceil(acc_dem / capacity) - (acc_deg - 2.0 * acc_intra);
            }
            for (int64_t r = 0; r < n_masked; ++r) viol[r] = -INFINITY;

            best = sag_sweep_argmax_f64(viol, L);
            winner = !(viol[best] <= cfg->min_violation);
        }

        sag_bitset_zero(mask, words);
        int64_t n_set = 0;
        if (winner) {
            for (int64_t r = 0; r <= best; ++r) sag_bit_set(mask, onode[r]);
            n_set = best + 1;
        } else {
            const float thr = (float)cfg->threshold;
            for (int64_t j = 1; j < n; ++j) {
                if (p[j] > thr) { sag_bit_set(mask, j); ++n_set; }
            }
            if (n_set == 0) {
                const int64_t am = sag_sweep_argmax_f32(p + 1, L);
                sag_bit_set(mask, am + 1);
                n_set = 1;
            }
        }

        if (n_set == 0) continue;
        sag_batch_push_mask(out, mask, (int8_t)SAG_FAMILY_SWEEP, (int32_t)v,
                            (int8_t)(winner ? SAG_VAR_WINNER : SAG_VAR_THRESHOLD_FALLBACK),
                            sag_quiet_nan());
    }

    if (g_rows > 0) {
        for (int64_t r = 0; r < out->rows; ++r)
            sag_bitset_copy(taken + r * words, out->mask + r * words, words);
        n_taken = out->rows;
        sag_growth_stats gs;
        sag_generate_growth(a, ctx, cfg, growth, out, taken, &n_taken, cap_rows, words, &gs);
        if (growth->diag) {
            fprintf(stderr,
                    "[GROWTH] rollouts=%lld seeds=%lld steps=%lld frontier_empty=%lld"
                    " certified=%lld emitted=%lld rej_dup=%lld rej_gate=%lld"
                    " mean_gate_margin=%.3e seeds_emitting=%lld emit_per_seed_mean=%.2f"
                    " emit_per_seed_max=%lld dup_vs_sweep=%lld dup_vs_seed=%lld"
                    " final_distinct=%lld final_collisions=%lld attempts=%lld restarts=%lld"
                    " pool_exhausted=%lld ptabu_swaps=%lld ptabu_dead=%lld seconds=%.6f\n",
                    (long long)gs.rollouts, (long long)gs.seeds, (long long)gs.steps,
                    (long long)gs.frontier_empty, (long long)gs.certified,
                    (long long)gs.emitted, (long long)gs.rej_dup, (long long)gs.rej_gate,
                    gs.rej_gate ? gs.margin_sum / (double)gs.rej_gate : 0.0,
                    (long long)gs.seeds_emitting,
                    gs.seeds_emitting ? (double)gs.emitted / (double)gs.seeds_emitting : 0.0,
                    (long long)gs.emit_per_seed_max, (long long)gs.dup_vs_sweep,
                    (long long)gs.dup_vs_seed, (long long)gs.final_distinct,
                    (long long)gs.final_collisions, (long long)gs.attempts,
                    (long long)gs.restarts, (long long)gs.pool_exhausted,
                    (long long)gs.ptabu_swaps, (long long)gs.ptabu_dead, gs.seconds);
        }
    }

    sag_arena_release(a, mark);
}
