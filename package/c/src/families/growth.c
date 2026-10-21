#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "families/growth.h"

#include "bitset.h"
#include "csr.h"
#include "families/sweep.h"

#define GROWTH_TOL 1e-6
#define GROWTH_POOL_FACTOR 4

int64_t sag_growth_row_reserve(const sag_growth_config *g, int64_t n_customers)
{
    if (!g->enabled || g->rollouts <= 0) return 0;
    const int64_t horizon = g->horizon < n_customers ? g->horizon : n_customers;
    return GROWTH_POOL_FACTOR * g->rollouts * horizon;
}

static const double *g_sort_viol;
static int growth_cmp_viol(const void *pa, const void *pb)
{
    const int64_t ia = *(const int64_t *)pa, ib = *(const int64_t *)pb;
    if (g_sort_viol[ia] > g_sort_viol[ib]) return -1;
    if (g_sort_viol[ia] < g_sort_viol[ib]) return 1;
    return ia < ib ? -1 : (ia > ib);
}

static uint64_t growth_mask_hash(const uint64_t *mask, int64_t words)
{
    uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (int64_t w = 0; w < words; ++w) {
        uint64_t x = mask[w] + 0x9E3779B97F4A7C15ULL * (uint64_t)(w + 1);
        x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL; x ^= x >> 27; x *= 0x94D049BB133111EBULL; x ^= x >> 31;
        h ^= x + 0x632BE59BD9B4E019ULL + (h << 6) + (h >> 2);
    }
    return h | 1ULL;
}

static int growth_ptabu_probe(uint64_t *tab, uint64_t tmask, uint64_t h, int add)
{
    uint64_t i = h & tmask;
    for (;;) {
        if (tab[i] == h) return 1;
        if (tab[i] == 0) { if (add) tab[i] = h; return 0; }
        i = (i + 1) & tmask;
    }
}

static int64_t growth_seeds(const sag_candidate_ctx *ctx, const int32_t *ordz, int64_t L,
                            int64_t n_seed, int32_t *out)
{
    const int64_t n = ctx->n_nodes;
    int64_t got = 0;
    for (int64_t r = 0; r < L && got < n_seed; ++r) {
        const int32_t j = ordz[r];
        int adjacent = 0;
        for (int64_t s = 0; s < got && !adjacent; ++s)
            adjacent = (double)ctx->lp->dense[(int64_t)j * n + out[s]] > GROWTH_TOL;
        if (!adjacent) out[got++] = j;
    }
    for (int64_t r = 0; r < L && got < n_seed; ++r) {
        const int32_t j = ordz[r];
        int seen = 0;
        for (int64_t s = 0; s < got && !seen; ++s) seen = (out[s] == j);
        if (!seen) out[got++] = j;
    }
    return got;
}

void sag_generate_growth(sag_arena *a, const sag_candidate_ctx *ctx,
                         const sag_sweep_config *sweep, const sag_growth_config *g,
                         sag_candidate_batch *out,
                         uint64_t *taken, int64_t *n_taken, int64_t cap_taken,
                         int64_t words, sag_growth_stats *stats)
{
    memset(stats, 0, sizeof *stats);
    const int64_t budget = g->enabled ? g->rollouts : 0;
    if (budget <= 0) return;

    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    const int64_t n = ctx->n_nodes, L = n - 1;
    if (L <= 0) return;
    const double Q = (double)ctx->capacity > 0.0 ? (double)ctx->capacity : 1.0;
    const double alpha = 0.5;
    const int tabu = g->restart_tabu ? 1 : 0;
    int64_t horizon = g->horizon;
    if (horizon > L) horizon = L;
    const int64_t ptabu = g->prefix_tabu;
    const int64_t n_bands = g->n_bands;
    const int64_t per_band = n_bands > 0 ? budget / n_bands : 0;
    if (per_band <= 0) return;

    const size_t mark = sag_arena_mark(a);
    double  *key   = SAG_NEW(a, double,  L);
    int32_t *ordi  = SAG_NEW(a, int32_t, L);
    int32_t *scr   = SAG_NEW(a, int32_t, L);
    int32_t *ordz  = SAG_NEW(a, int32_t, L);
    const int64_t pool = tabu ? per_band * GROWTH_POOL_FACTOR : per_band;
    int32_t *seeds = SAG_NEW(a, int32_t, pool);
    double  *w     = SAG_NEW(a, double,  n);
    uint8_t *inS   = SAG_NEW(a, uint8_t, n);
    uint64_t *mask = SAG_NEW(a, uint64_t, words);

    const int64_t gmax = g->max_rows > 0 ? g->max_rows : 0;
    const int64_t gcap = gmax > 0 ? (pool * n_bands) * (horizon + 1) : 0;
    uint64_t *gmasks = gmax > 0 ? SAG_NEW(a, uint64_t, (size_t)gcap * (size_t)words) : NULL;
    double   *gviol  = gmax > 0 ? SAG_NEW(a, double,  gcap) : NULL;
    int64_t  *gsize  = gmax > 0 ? SAG_NEW(a, int64_t, gcap) : NULL;
    int32_t  *gm     = gmax > 0 ? SAG_NEW(a, int32_t, gcap) : NULL;
    int32_t  *growner = gmax > 0 ? SAG_NEW(a, int32_t, gcap) : NULL;
    int64_t   gcount = 0;

    uint64_t *ptab = NULL; uint64_t ptmask = 0;
    if (ptabu > 0) {
        uint64_t need = (uint64_t)(pool * n_bands) * (uint64_t)(horizon + 2) * 4ULL + 64ULL;
        uint64_t pcap = 1024; while (pcap < need) pcap <<= 1;
        ptab = SAG_NEW0(a, uint64_t, pcap); ptmask = pcap - 1;
    }

    const int64_t n_before = *n_taken;
    const int64_t n_roll = pool * n_bands;
    int64_t  *per_seed = SAG_NEW0(a, int64_t, n_roll > 0 ? n_roll : 1);
    uint64_t *finals  = SAG_NEW0(a, uint64_t, (size_t)(n_roll > 0 ? n_roll : 1) * (size_t)words);
    int64_t roll_id = 0;

    for (int64_t b = 0; b < n_bands; ++b) {
        const int64_t k = g->bands[b];
        const int64_t m0 = k - 1;
        const int64_t m = m0 >= ctx->probs_rows ? ctx->probs_rows - 1 : m0;
        if (m < 0) continue;
        const float *p = ctx->probs + m * n;

        for (int64_t j = 0; j < L; ++j) key[j] = -(double)p[1 + j];
        sag_sweep_stable_argsort(key, L, ordi, scr);
        for (int64_t r = 0; r < L; ++r) ordz[r] = ordi[r] + 1;

        const int64_t n_seed = growth_seeds(ctx, ordz, L, pool, seeds);
        const double hi = (double)k * Q;

        int64_t productive = 0;
        for (int64_t si = 0; si < n_seed; ++si) {
            if (tabu ? productive >= per_band : si >= per_band) break;
            const int32_t seed = seeds[si];
            stats->rollouts += 1;
            stats->attempts += 1;
            stats->seeds += 1;
            const int64_t emitted_before = stats->emitted;
            const int64_t gcount_before = gcount;

            memset(inS, 0, (size_t)n);
            inS[0] = 1;
            inS[seed] = 1;
            sag_bitset_zero(mask, words);
            sag_bit_set(mask, seed);
            if (ptab) growth_ptabu_probe(ptab, ptmask, growth_mask_hash(mask, words), 1);

            const float *srow = ctx->lp->dense + (int64_t)seed * n;
            for (int64_t j = 0; j < n; ++j) w[j] = (double)srow[j];
            double boundary = (double)ctx->lp->degree[seed];
            int64_t dS = ctx->demand[seed];
            int64_t size = 1;

            for (int64_t step = 0; step < horizon; ++step) {
                if (size >= sweep->min_size) {
                    const double band = ceil((double)dS / Q);
                    const double viol = 2.0 * band - boundary;
                    if (viol > sweep->min_violation) {
                        stats->certified += 1;
                        if (gmax > 0) {
                            int pre = 0;
                            for (int64_t q = 0; q < n_before && !pre; ++q)
                                if (sag_bitset_equal(taken + q * words, mask, words)) pre = 1;
                            if (pre) { stats->rej_dup += 1; stats->dup_vs_sweep += 1; }
                            else if (gcount < gcap) {
                                sag_bitset_copy(gmasks + (size_t)gcount * (size_t)words, mask, words);
                                gviol[gcount] = viol; gsize[gcount] = size; gm[gcount] = (int32_t)m;
                                growner[gcount] = (int32_t)roll_id; gcount += 1;
                            }
                        } else {
                            int seen = 0;
                            int64_t hit_at = -1;
                            for (int64_t q = 0; q < *n_taken && !seen; ++q)
                                if (sag_bitset_equal(taken + q * words, mask, words)) {
                                    seen = 1; hit_at = q;
                                }
                            if (seen) {
                                stats->rej_dup += 1;
                                if (hit_at < n_before) stats->dup_vs_sweep += 1;
                                else stats->dup_vs_seed += 1;
                            } else if (*n_taken < cap_taken && out->rows < out->cap_rows
                                       && out->indptr[out->rows] + size <= out->cap_nnz) {
                                per_seed[roll_id] += 1;
                                sag_bitset_copy(taken + *n_taken * words, mask, words);
                                *n_taken += 1;
                                sag_batch_push_mask(out, mask, (int8_t)SAG_FAMILY_SWEEP,
                                                    (int32_t)(m), (int8_t)SAG_VAR_GROWTH,
                                                    sag_quiet_nan());
                                stats->emitted += 1;
                            }
                        }
                    } else {
                        stats->rej_gate += 1;
                        stats->margin_sum += sweep->min_violation - viol;
                    }
                }
                if (step + 1 >= horizon) break;

                double wmax = 0.0;
                for (int64_t j = 1; j < n; ++j)
                    if (!inS[j] && w[j] > GROWTH_TOL && w[j] > wmax) wmax = w[j];
                if (wmax <= 0.0) { stats->frontier_empty += 1; break; }
                const double inv_wmax = 1.0 / wmax;

                const double gap = hi + 1.0 - (double)dS;
                int32_t best = -1;
                double best_sc = 0.0;
                enum { PT_MAX = 9 };
                int32_t cand[PT_MAX]; double cand_sc[PT_MAX]; int ncand = 0;
                const int kmax = ptab ? (int)(ptabu < PT_MAX - 1 ? ptabu + 1 : PT_MAX) : 0;
                for (int64_t j = 1; j < n; ++j) {
                    if (inS[j] || !(w[j] > GROWTH_TOL)) continue;
                    double dev = fabs(gap - (double)ctx->demand[j]);
                    if (dev > Q) dev = Q;
                    const double sc = w[j] * inv_wmax + alpha * (1.0 - dev / Q);
                    if (best < 0 || sc > best_sc) { best = (int32_t)j; best_sc = sc; }
                    if (kmax > 0) {
                        int pos = ncand;
                        while (pos > 0 && cand_sc[pos - 1] < sc) --pos;
                        if (pos < kmax) {
                            int last = ncand < kmax ? ncand : kmax - 1;
                            for (int q = last; q > pos; --q) { cand[q] = cand[q - 1]; cand_sc[q] = cand_sc[q - 1]; }
                            cand[pos] = (int32_t)j; cand_sc[pos] = sc;
                            if (ncand < kmax) ++ncand;
                        }
                    }
                }
                if (best < 0) { stats->frontier_empty += 1; break; }
                if (ptab) {
                    int32_t choice = best; int found = 0;
                    for (int q = 0; q < ncand; ++q) {
                        sag_bit_set(mask, cand[q]);
                        const uint64_t h = growth_mask_hash(mask, words);
                        sag_bit_clear(mask, cand[q]);
                        if (!growth_ptabu_probe(ptab, ptmask, h, 0)) { choice = cand[q]; found = 1; break; }
                    }
                    if (!found) stats->ptabu_dead += 1;
                    else if (choice != best) stats->ptabu_swaps += 1;
                    best = choice;
                }

                boundary += (double)ctx->lp->degree[best] - 2.0 * w[best];
                inS[best] = 1;
                sag_bit_set(mask, best);
                dS += ctx->demand[best];
                size += 1;
                stats->steps += 1;
                const float *arow = ctx->lp->dense + (int64_t)best * n;
                for (int64_t j = 0; j < n; ++j) w[j] += (double)arow[j];
                if (ptab) growth_ptabu_probe(ptab, ptmask, growth_mask_hash(mask, words), 1);
            }

            sag_bitset_copy(finals + roll_id * words, mask, words);
            roll_id += 1;
            if (stats->emitted > emitted_before || gcount > gcount_before) productive += 1;
            else if (tabu) stats->restarts += 1;
        }
        if (tabu && productive < per_band) stats->pool_exhausted += 1;
    }

    if (gmax > 0 && gcount > 0) {
        int64_t *ord = SAG_NEW(a, int64_t, gcount);
        for (int64_t i = 0; i < gcount; ++i) ord[i] = i;
        g_sort_viol = gviol;
        qsort(ord, (size_t)gcount, sizeof *ord, growth_cmp_viol);
        int64_t pushed = 0;
        for (int64_t oi = 0; oi < gcount && pushed < gmax; ++oi) {
            const int64_t i = ord[oi];
            const uint64_t *gmk = gmasks + (size_t)i * (size_t)words;
            int seen = 0;
            for (int64_t q = 0; q < *n_taken && !seen; ++q)
                if (sag_bitset_equal(taken + q * words, gmk, words)) seen = 1;
            if (seen) { stats->rej_dup += 1; stats->dup_vs_seed += 1; continue; }
            if (!(*n_taken < cap_taken && out->rows < out->cap_rows
                  && out->indptr[out->rows] + gsize[i] <= out->cap_nnz)) break;
            per_seed[growner[i]] += 1;
            sag_bitset_copy(taken + *n_taken * words, gmk, words);
            *n_taken += 1;
            sag_batch_push_mask(out, gmk, (int8_t)SAG_FAMILY_SWEEP, gm[i], (int8_t)SAG_VAR_GROWTH,
                                sag_quiet_nan());
            stats->emitted += 1; pushed += 1;
        }
    }

    for (int64_t r = 0; r < roll_id; ++r) {
        if (per_seed[r] > 0) stats->seeds_emitting += 1;
        if (per_seed[r] > stats->emit_per_seed_max) stats->emit_per_seed_max = per_seed[r];
        int repeat = 0;
        for (int64_t q = 0; q < r && !repeat; ++q)
            repeat = sag_bitset_equal(finals + q * words, finals + r * words, words);
        if (repeat) stats->final_collisions += 1; else stats->final_distinct += 1;
    }

    sag_arena_release(a, mark);
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    stats->seconds = (double)(ts1.tv_sec - ts0.tv_sec) + 1e-9 * (double)(ts1.tv_nsec - ts0.tv_nsec);
}
