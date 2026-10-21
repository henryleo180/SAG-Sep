#include "families/tts/fcapfix.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "bitset.h"
#include "families/tts/score.h"
#include "features.h"

#if !defined(SAG_FP_CONTRACT_OFF)
#error "families/tts/fcapfix.c must be compiled with -ffp-contract=off (the \
Makefile's FPFLAGS pass -DSAG_FP_CONTRACT_OFF=1 at the same time): once x*V+1.0 \
is contracted into an FMA the floor result diverges from numpy, the minimum cut \
changes, and bit-for-bit parity is lost"
#endif

#define SAG_FCAPFIX_CAP_MAX 2147483647LL

/* ------------------------------------------------------------------ *
 * Dinic
 * ------------------------------------------------------------------ */

typedef struct {
    int32_t *head;
    int32_t *next;
    int32_t *to;
    int64_t *cap;
    int64_t *base;
    int32_t *level;
    int32_t *iter;
    int32_t *queue;
    int64_t  n_vert;
    int64_t  n_arc;
} sag_fc_flow;

static void sag_fc_add_pair(sag_fc_flow *g, int64_t u, int64_t v,
                            int64_t cap_uv, int64_t cap_vu, int64_t *cursor)
{
    const int64_t e = *cursor;
    g->to[e] = (int32_t)v;       g->base[e] = cap_uv;
    g->next[e] = g->head[u];     g->head[u] = (int32_t)e;
    g->to[e + 1] = (int32_t)u;   g->base[e + 1] = cap_vu;
    g->next[e + 1] = g->head[v]; g->head[v] = (int32_t)(e + 1);
    *cursor = e + 2;
}

static int sag_fc_bfs(sag_fc_flow *g, int64_t source, int64_t sink)
{
    for (int64_t i = 0; i < g->n_vert; i++) g->level[i] = -1;
    int64_t qh = 0, qt = 0;
    g->level[source] = 0;
    g->queue[qt++] = (int32_t)source;
    while (qh < qt) {
        const int64_t u = g->queue[qh++];
        for (int32_t e = g->head[u]; e >= 0; e = g->next[e]) {
            if (g->cap[e] <= 0) continue;
            const int32_t v = g->to[e];
            if (g->level[v] >= 0) continue;
            g->level[v] = g->level[u] + 1;
            g->queue[qt++] = v;
        }
    }
    return g->level[sink] >= 0;
}

static int64_t sag_fc_dfs(sag_fc_flow *g, int64_t u, int64_t sink, int64_t limit)
{
    if (u == sink) return limit;
    for (int32_t *e = &g->iter[u]; *e >= 0; *e = g->next[*e]) {
        const int32_t arc = *e;
        const int32_t v = g->to[arc];
        if (g->cap[arc] <= 0 || g->level[v] != g->level[u] + 1) continue;
        const int64_t room = (limit < g->cap[arc]) ? limit : g->cap[arc];
        const int64_t got = sag_fc_dfs(g, v, sink, room);
        if (got > 0) {
            g->cap[arc] -= got;
            g->cap[arc ^ 1] += got;
            return got;
        }
    }
    return 0;
}

static void sag_fc_max_flow(sag_fc_flow *g, int64_t source, int64_t sink)
{
    while (sag_fc_bfs(g, source, sink)) {
        for (int64_t i = 0; i < g->n_vert; i++) g->iter[i] = g->head[i];
        while (sag_fc_dfs(g, source, sink, INT64_MAX) > 0) { }
    }
}

static void sag_fc_sink_side(sag_fc_flow *g, int64_t source, int64_t sink,
                             const int32_t *src_arc, const int32_t *snk_arc,
                             const int64_t *force_source, int64_t n_force_source,
                             const int64_t *force_sink, int64_t n_force_sink,
                             int64_t inf_cap, int64_t m,
                             uint8_t *reach_scratch, uint8_t *sink_side)
{
    memcpy(g->cap, g->base, (size_t)g->n_arc * sizeof(int64_t));
    for (int64_t i = 0; i < n_force_source; i++) g->cap[src_arc[force_source[i]]] = inf_cap;
    for (int64_t i = 0; i < n_force_sink; i++) g->cap[snk_arc[force_sink[i]]] = inf_cap;

    sag_fc_max_flow(g, source, sink);

    memset(reach_scratch, 0, (size_t)g->n_vert);
    int64_t qh = 0, qt = 0;
    reach_scratch[source] = 1;
    g->queue[qt++] = (int32_t)source;
    while (qh < qt) {
        const int64_t u = g->queue[qh++];
        for (int32_t e = g->head[u]; e >= 0; e = g->next[e]) {
            if (g->cap[e] <= 0) continue;
            const int32_t v = g->to[e];
            if (reach_scratch[v]) continue;
            reach_scratch[v] = 1;
            g->queue[qt++] = v;
        }
    }
    for (int64_t j = 0; j < m; j++) sink_side[j] = (uint8_t)(!reach_scratch[j]);
}

/* ------------------------------------------------------------------ *
 * numpy sort semantics (float64 version; the float32 version is in score.h)
 * ------------------------------------------------------------------ */

static int sag_fc_lt(double a, double b)
{
    return (a < b) || (isnan(b) && !isnan(a));
}

static void sag_fc_stable_argsort_f64(const double *key, int64_t n, int64_t *idx, int64_t *tmp)
{
    for (int64_t i = 0; i < n; i++) idx[i] = i;
    for (int64_t width = 1; width < n; width *= 2) {
        for (int64_t lo = 0; lo < n; lo += 2 * width) {
            const int64_t mid = (lo + width < n) ? lo + width : n;
            const int64_t hi = (lo + 2 * width < n) ? lo + 2 * width : n;
            int64_t i = lo, j = mid, k = lo;
            while (i < mid && j < hi)
                tmp[k++] = sag_fc_lt(key[idx[j]], key[idx[i]]) ? idx[j++] : idx[i++];
            while (i < mid) tmp[k++] = idx[i++];
            while (j < hi) tmp[k++] = idx[j++];
            for (int64_t t = lo; t < hi; t++) idx[t] = tmp[t];
        }
    }
}

/* ------------------------------------------------------------------ *
 * Node scores -> supernode scores
 * ------------------------------------------------------------------ */

void sag_tts_super_scores(const sag_tts_compression *compression,
                          const float *node_scores, int64_t n_nodes, double *out)
{
    if (!compression || compression->n_nodes != n_nodes)
        sag_die("super_scores: compression.n_nodes(%lld) != n_nodes(%lld)",
                compression ? (long long)compression->n_nodes : -1LL, (long long)n_nodes);
    const float *member = compression->member;
    for (int64_t i = 0; i < compression->n_super; i++) {
        const float *mrow = member + i * n_nodes;
        double acc = 0.0;
        for (int64_t j = 0; j < n_nodes; j++) acc += (double)mrow[j] * (double)node_scores[j];
        float cnt = sag_f32_sum_numpy(mrow, n_nodes);
        if (cnt < 1.0f) cnt = 1.0f;
        out[i] = acc / (double)cnt;
    }
}

/* ------------------------------------------------------------------ *
 * Main body
 * ------------------------------------------------------------------ */

void sag_tts_fcapfix(sag_arena *a,
                     int64_t                     n_nodes,
                     double                      capacity,
                     int64_t                     rounds,
                     int64_t                     seed_limit,
                     const float                *seed_scores,
                     const sag_tts_compression  *compression,
                     sag_tts_fcapfix_out        *out)
{
    const int64_t words = (n_nodes > 0) ? sag_mask_words(n_nodes) : 0;
    out->masks = (uint64_t *)sag_arena_alloc(a, 0, _Alignof(uint64_t));
    out->rows = 0;
    out->words = words;
    out->n_nodes = n_nodes;

    if (n_nodes <= 2 || rounds <= 0 || capacity <= 0.0) return;
    if (!compression || compression->n_nodes != n_nodes)
        sag_die("fcapfix: compression.n_nodes(%lld) != n_nodes(%lld)",
                compression ? (long long)compression->n_nodes : -1LL, (long long)n_nodes);

    const int64_t n_super = compression->n_super;
    const int64_t m = n_super - 1;
    if (m <= 1) return;

    const float  *aggregate  = compression->aggregate;
    const float  *dem_super  = compression->demand;
    const float  *member     = compression->member;

    const int64_t n_order_max = (seed_limit > 0 && seed_limit < m) ? seed_limit : m;
    const int64_t cap_rows = rounds * n_order_max;
    out->masks = SAG_NEW(a, uint64_t, sag_size_mul((size_t)cap_rows, (size_t)words));

    /* --- integer network -------------------------------------------- */
    double  virtual_capacity = capacity;
    int64_t flow_scale = 1;
    for (int64_t guard = 0; virtual_capacity < 1000.0; guard++) {
        virtual_capacity *= 10.0;
        flow_scale *= 10;
        if (guard > 64)
            sag_die("fcapfix: virtual_capacity did not converge (capacity=%.17g)", capacity);
    }
    const int64_t source = m, sink = m + 1, n_vert = m + 2;

    const size_t mark = sag_arena_mark(a);

    int64_t n_edge = 0;
    for (int64_t u = 0; u < m; u++) {
        const float *row = aggregate + (u + 1) * n_super;
        for (int64_t v = u + 1; v < m; v++)
            if (row[v + 1] > 0.0f) n_edge++;
    }
    const int64_t n_arc = 2 * (n_edge + 2 * m);
    if (n_arc > INT32_MAX)
        sag_die("fcapfix: arc count %lld exceeds the int32 index range", (long long)n_arc);

    sag_fc_flow g;
    g.n_vert = n_vert;
    g.n_arc = n_arc;
    g.head = SAG_NEW(a, int32_t, n_vert);
    g.next = SAG_NEW(a, int32_t, n_arc);
    g.to = SAG_NEW(a, int32_t, n_arc);
    g.cap = SAG_NEW(a, int64_t, n_arc);
    g.base = SAG_NEW(a, int64_t, n_arc);
    g.level = SAG_NEW(a, int32_t, n_vert);
    g.iter = SAG_NEW(a, int32_t, n_vert);
    g.queue = SAG_NEW(a, int32_t, n_vert);
    for (int64_t i = 0; i < n_vert; i++) g.head[i] = -1;

    int32_t *src_arc = SAG_NEW(a, int32_t, m);
    int32_t *snk_arc = SAG_NEW(a, int32_t, m);
    int64_t cursor = 0;

    for (int64_t u = 0; u < m; u++) {
        const float *row = aggregate + (u + 1) * n_super;
        for (int64_t v = u + 1; v < m; v++) {
            const float x = row[v + 1];
            if (!(x > 0.0f)) continue;
            const int64_t c = (int64_t)floor((double)x * virtual_capacity + 1.0);
            if (c > SAG_FCAPFIX_CAP_MAX || c < 0)
                sag_die("fcapfix: edge capacity %lld out of scipy's int32 domain", (long long)c);
            sag_fc_add_pair(&g, u, v, c, c, &cursor);
        }
    }

    int64_t source_cap_sum = 0;
    const float *depot_row = aggregate;
    for (int64_t j = 0; j < m; j++) {
        const double unary = ((double)depot_row[j + 1] * virtual_capacity + 1.0)
                             - (2.0 * (double)flow_scale) * (double)dem_super[j + 1];
        int64_t cs = 0, ck = 0;
        if (unary > 0.0) {
            cs = (int64_t)floor(unary);
            source_cap_sum += cs;
        } else {
            ck = (int64_t)(-floor(unary));
        }
        if (cs > SAG_FCAPFIX_CAP_MAX || ck > SAG_FCAPFIX_CAP_MAX || cs < 0 || ck < 0)
            sag_die("fcapfix: unary-term capacity out of scipy's int32 domain (%lld / %lld)",
                    (long long)cs, (long long)ck);
        src_arc[j] = (int32_t)cursor;
        sag_fc_add_pair(&g, source, j, cs, 0, &cursor);
        snk_arc[j] = (int32_t)cursor;
        sag_fc_add_pair(&g, j, sink, ck, 0, &cursor);
    }
    if (cursor != n_arc)
        sag_die("fcapfix: inconsistent arc count (%lld != %lld)", (long long)cursor, (long long)n_arc);

    const int64_t inf_cap = (int64_t)((double)source_cap_sum + 2.0 * virtual_capacity);
    if (inf_cap > SAG_FCAPFIX_CAP_MAX || inf_cap <= 0)
        sag_die("fcapfix: inf_cap=%lld out of scipy's int32 domain", (long long)inf_cap);

    /* --- seed ordering ----------------------------------------------- */
    double *score_super = SAG_NEW(a, double, n_super);
    sag_tts_super_scores(compression, seed_scores, n_nodes, score_super);
    double  *neg = SAG_NEW(a, double, m);
    int64_t *seed_order = SAG_NEW(a, int64_t, m);
    int64_t *sort_tmp = SAG_NEW(a, int64_t, m);
    for (int64_t i = 0; i < m; i++) neg[i] = -score_super[i + 1];
    sag_fc_stable_argsort_f64(neg, m, seed_order, sort_tmp);
    const int64_t n_order = (seed_limit > 0 && seed_limit < m) ? seed_limit : m;

    /* --- state ------------------------------------------------------- */
    const int64_t hist_cap = rounds * n_order;
    const int64_t words_m = sag_mask_words(m);

    uint8_t  *use_seed     = SAG_NEW(a, uint8_t, m);
    uint8_t  *singleton    = SAG_NEW0(a, uint8_t, m);
    uint8_t  *node_set     = SAG_NEW(a, uint8_t, m);
    uint8_t  *sink_side    = SAG_NEW(a, uint8_t, m);
    uint8_t  *reach        = SAG_NEW(a, uint8_t, n_vert);
    uint8_t  *in_fixed     = SAG_NEW(a, uint8_t, m);
    int64_t  *nodes        = SAG_NEW(a, int64_t, m);
    int64_t  *fixed_src    = SAG_NEW(a, int64_t, hist_cap + 1);
    float    *gather       = SAG_NEW(a, float, m);
    float    *connectivity = SAG_NEW(a, float, m);
    float    *cand_val     = SAG_NEW(a, float, m);
    int64_t  *cand         = SAG_NEW(a, int64_t, m);
    uint64_t *hist_mask    = SAG_NEW(a, uint64_t, sag_size_mul((size_t)hist_cap, (size_t)words_m));
    int64_t  *hist_size    = SAG_NEW(a, int64_t, hist_cap);
    int64_t  *hist_seed    = SAG_NEW(a, int64_t, hist_cap);
    uint64_t *fixed_mask   = SAG_NEW(a, uint64_t, words_m);
    uint64_t *row_mask     = SAG_NEW(a, uint64_t, words);
    int64_t   fixed_sink[2];
    for (int64_t i = 0; i < m; i++) use_seed[i] = 1;
    int64_t hist_n = 0;

    uint64_t *out_masks = out->masks;
    int64_t   out_rows = 0;

    for (int64_t round = 0; round < rounds; round++) {
        for (int64_t oi = 0; oi < n_order; oi++) {
            const int64_t seed = seed_order[oi];
            if (!use_seed[seed]) continue;

            int64_t n_fixed = 0;
            sag_bitset_zero(fixed_mask, words_m);
            memset(in_fixed, 0, (size_t)m);
            for (int64_t h = hist_n - 1; h >= 0; h--) {
                const uint64_t *pm = hist_mask + h * words_m;
                if (hist_size[h] == 1 || !sag_bit_test(pm, seed)) continue;
                if (sag_bitset_intersection(fixed_mask, pm, words_m) > 0) continue;
                int64_t cover = hist_seed[h];
                if (cover == seed) {
                    cover = -1;
                    for (int64_t j = 0; j < m; j++)
                        if (j != seed && sag_bit_test(pm, j)) { cover = j; break; }
                    if (cover < 0)
                        sag_die("fcapfix: history record holds only the seed itself (size=%lld)",
                                (long long)hist_size[h]);
                }
                if (!in_fixed[cover]) {
                    in_fixed[cover] = 1;
                    sag_bit_set(fixed_mask, cover);
                    fixed_src[n_fixed++] = cover;
                }
            }

            int64_t n_sink = 0;
            fixed_sink[n_sink++] = seed;
            if (singleton[seed]) {
                const float *srow = aggregate + (seed + 1) * n_super;
                int64_t nc = 0;
                for (int64_t j = 0; j < m; j++)
                    if (srow[j + 1] > 0.0f && !in_fixed[j]) {
                        cand[nc] = j;
                        cand_val[nc] = srow[j + 1];
                        nc++;
                    }
                if (nc == 0) {
                    use_seed[seed] = 0;
                    continue;
                }
                fixed_sink[n_sink++] = cand[sag_tts_argmax_f32(cand_val, nc)];
            }

            sag_fc_sink_side(&g, source, sink, src_arc, snk_arc,
                             fixed_src, n_fixed, fixed_sink, n_sink,
                             inf_cap, m, reach, sink_side);

            int64_t n_sel = 0;
            memset(node_set, 0, (size_t)m);
            for (int64_t j = 0; j < m; j++)
                if (sink_side[j]) { nodes[n_sel++] = j; node_set[j] = 1; }
            if (n_sel == 0) continue;

            for (int64_t k = 0; k < n_sel; k++) gather[k] = dem_super[nodes[k] + 1];
            double demand_sum = (double)sag_f32_sum_numpy(gather, n_sel);
            int64_t veh = (int64_t)ceil(demand_sum / capacity);
            double capacity_sum = capacity * (double)(veh < 1 ? 1 : veh);

            double max_outside = 0.0;
            int    have_outside = 0;
            for (int64_t j = 0; j < m; j++) {
                if (node_set[j]) continue;
                const double d = (double)dem_super[j + 1];
                if (!have_outside || d > max_outside) { max_outside = d; have_outside = 1; }
            }

            if (demand_sum + max_outside > capacity_sum) {
                for (;;) {
                    for (int64_t i = 0; i < m; i++) {
                        const float *row = aggregate + (i + 1) * n_super;
                        for (int64_t k = 0; k < n_sel; k++) gather[k] = row[nodes[k] + 1];
                        connectivity[i] = sag_f32_sum_numpy(gather, n_sel);
                    }
                    int64_t nc = 0;
                    for (int64_t i = 0; i < m; i++) {
                        if (node_set[i] || in_fixed[i]) continue;
                        if (!(connectivity[i] > 0.0f)) continue;
                        if (!(demand_sum + (double)dem_super[i + 1] > capacity_sum)) continue;
                        cand[nc] = i;
                        cand_val[nc] = connectivity[i];
                        nc++;
                    }
                    if (nc == 0) break;
                    const int64_t add_node = cand[sag_tts_argmax_f32(cand_val, nc)];
                    nodes[n_sel++] = add_node;
                    node_set[add_node] = 1;
                    demand_sum += (double)dem_super[add_node + 1];
                    veh = (int64_t)ceil(demand_sum / capacity);
                    capacity_sum = capacity * (double)(veh < 1 ? 1 : veh);
                }
            }

            if (n_sel == 1) singleton[seed] = 1;

            if (hist_n >= hist_cap)
                sag_die("fcapfix: history exceeded its upper bound (%lld)", (long long)hist_cap);
            uint64_t *hm = hist_mask + hist_n * words_m;
            sag_bitset_zero(hm, words_m);
            for (int64_t k = 0; k < n_sel; k++) sag_bit_set(hm, nodes[k]);
            hist_size[hist_n] = n_sel;
            hist_seed[hist_n] = seed;
            hist_n++;

            sag_bitset_zero(row_mask, words);
            for (int64_t k = 0; k < n_sel; k++) {
                const float *mrow = member + (nodes[k] + 1) * n_nodes;
                for (int64_t j = 0; j < n_nodes; j++)
                    if (mrow[j] != 0.0f) sag_bit_set(row_mask, j);
            }
            sag_bit_clear(row_mask, 0);

            int dup = 0;
            for (int64_t r = 0; r < out_rows; r++)
                if (sag_bitset_equal(out_masks + r * words, row_mask, words)) { dup = 1; break; }
            if (!dup) {
                if (out_rows >= cap_rows)
                    sag_die("fcapfix: output rows exceed the upper bound (%lld)", (long long)cap_rows);
                sag_bitset_copy(out_masks + out_rows * words, row_mask, words);
                out_rows++;
            }
        }
    }

    sag_arena_release(a, mark);
    out->rows = out_rows;
}
