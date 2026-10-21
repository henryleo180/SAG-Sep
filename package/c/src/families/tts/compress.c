#include "families/tts/compress.h"

#include <math.h>
#include <string.h>

/* ===================================================================== */
/* 0. numpy DOUBLE_pairwise_sum (structure = features.c float32 version) */
/* ===================================================================== */

#define SAG_PW_BLOCKSIZE 128

double sag_tts_f64_pairwise_sum(const double *a, int64_t n)
{
    if (n < 8) {
        double res = 0.0;
        for (int64_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= SAG_PW_BLOCKSIZE) {
        double r[8];
        for (int k = 0; k < 8; ++k) r[k] = a[k];
        int64_t       i   = 8;
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
    return sag_tts_f64_pairwise_sum(a, n2) + sag_tts_f64_pairwise_sum(a + n2, n - n2);
}

#define SAG_TTS_NUMPY_BUFSIZE 8192

double sag_tts_f64_sum_numpy(const double *a, int64_t n)
{
    double res = 0.0;
    int64_t off = 0;
    while (off < n) {
        int64_t chunk = n - off;
        if (chunk > SAG_TTS_NUMPY_BUFSIZE) chunk = SAG_TTS_NUMPY_BUFSIZE;
        res += sag_tts_f64_pairwise_sum(a + off, chunk);
        off += chunk;
    }
    return res;
}

/* ===================================================================== */
/* 1. union-find                                                          */
/* ===================================================================== */

static int64_t uf_find(int64_t *parent, int64_t node)
{
    while (parent[node] != node) {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    return node;
}

static void uf_merge(int64_t *parent, int64_t left, int64_t right)
{
    int64_t rl = uf_find(parent, left);
    int64_t rr = uf_find(parent, right);
    if (rl == rr) return;
    if (rl > rr) { const int64_t t = rl; rl = rr; rr = t; }
    parent[rr] = rl;
}

static int64_t normalise_labels(int64_t *parent, int64_t n, int64_t *labels, int64_t *rank)
{
    for (int64_t i = 0; i < n; ++i) rank[i] = -1;
    for (int64_t i = 0; i < n; ++i) labels[i] = uf_find(parent, i);
    for (int64_t i = 0; i < n; ++i) rank[labels[i]] = -2;
    int64_t n_super = 0;
    for (int64_t root = 0; root < n; ++root)
        if (rank[root] == -2) rank[root] = n_super++;
    for (int64_t i = 0; i < n; ++i) labels[i] = rank[labels[i]];
    return n_super;
}

/* ===================================================================== */
/* 2. _one_edge_partition                                                 */
/* ===================================================================== */

static int64_t one_edge_partition_f32(const float *adj, int64_t n, double eps,
                                      int64_t *parent, int64_t *labels, int64_t *rank)
{
    for (int64_t i = 0; i < n; ++i) parent[i] = i;
    const float thr = (float)(1.0 - eps);
    for (int64_t tail = 1; tail < n; ++tail) {
        const float *row = adj + tail * n;
        for (int64_t head = tail + 1; head < n; ++head) {
            if (!(row[head] >= thr)) continue;
            const int64_t rt = uf_find(parent, tail);
            const int64_t rh = uf_find(parent, head);
            if (rt == rh) continue;
            parent[rh] = rt;
        }
    }
    return normalise_labels(parent, n, labels, rank);
}

static void one_edge_partition_f64(const double *adj, int64_t n, double eps,
                                   int64_t *parent, int64_t *labels, int64_t *rank)
{
    for (int64_t i = 0; i < n; ++i) parent[i] = i;
    const double thr = 1.0 - eps;
    for (int64_t tail = 1; tail < n; ++tail) {
        const double *row = adj + tail * n;
        for (int64_t head = tail + 1; head < n; ++head) {
            if (!(row[head] >= thr)) continue;
            const int64_t rt = uf_find(parent, tail);
            const int64_t rh = uf_find(parent, head);
            if (rt == rh) continue;
            parent[rh] = rt;
        }
    }
    (void)normalise_labels(parent, n, labels, rank);
}

/* ===================================================================== */
/* 3. _partition_data                                                     */
/* ===================================================================== */

static void partition_data(const double *adj64, int64_t n, const int64_t *labels, int64_t n_super,
                           double *member64, double *aggregate64, int64_t *sizes, double *internal)
{
    if (member64) {
        memset(member64, 0, (size_t)n_super * (size_t)n * sizeof(double));
        for (int64_t i = 0; i < n; ++i) member64[labels[i] * n + i] = 1.0;
    }
    memset(aggregate64, 0, (size_t)n_super * (size_t)n_super * sizeof(double));
    for (int64_t i = 0; i < n; ++i) {
        const int64_t li  = labels[i];
        const double *row = adj64 + i * n;
        double       *dst = aggregate64 + li * n_super;
        for (int64_t j = 0; j < n; ++j) dst[labels[j]] += row[j];
    }
    for (int64_t s = 0; s < n_super; ++s) sizes[s] = 0;
    for (int64_t i = 0; i < n; ++i) sizes[labels[i]] += 1;
    for (int64_t s = 0; s < n_super; ++s) internal[s] = 0.5 * aggregate64[s * n_super + s];
}

/* ===================================================================== */
/* 4. _compress_partition                                                 */
/* ===================================================================== */

typedef struct {
    int64_t *parent;
    int64_t *labels;
    int64_t *rank;
    double  *aggregate64;
    int64_t *sizes;
    double  *internal;
    bool    *frozen;
    bool    *shrinkable;
    int64_t *reps;
    double  *buf;
    int64_t *tight_rows;
    int64_t  n_tight;
    int64_t *comp_buf;
    bool    *comp_seen;
} compress_work;

static inline bool agg_positive(const double *agg, int64_t ns, const bool *shrinkable,
                                int64_t a, int64_t b)
{
    return (agg[a * ns + b] > 0.01) && shrinkable[a] && shrinkable[b];
}

static int64_t compress_partition(const double *adj64, int64_t n, const float *demand_f32,
                                  double capacity, const float *old_sets, int64_t n_old,
                                  double eps, compress_work *w, int64_t *out_labels,
                                  bool *out_frozen)
{
    int64_t *parent = w->parent;

    /* --- initial one-edge partition (float64 copy + double threshold chain) --- */
    one_edge_partition_f64(adj64, n, eps, parent, w->labels, w->rank);
    {
        int64_t *rep_of_label = w->rank;
        for (int64_t i = 0; i < n; ++i) rep_of_label[i] = -1;
        for (int64_t i = 0; i < n; ++i) parent[i] = i;
        for (int64_t node = 1; node < n; ++node) {
            const int64_t label = w->labels[node];
            if (rep_of_label[label] < 0) rep_of_label[label] = node;
            uf_merge(parent, rep_of_label[label], node);
        }
    }

    /* --- tight one-vehicle old sets (remember only the bank row index, expand on use) --- */
    w->n_tight = 0;
    if (capacity > 0.0) {
        for (int64_t b = 0; b < n_old; ++b) {
            const float *selection = old_sets + b * n;
            int64_t m = 0;
            for (int64_t i = 1; i < n; ++i)
                if (selection[i] > 0.5f) w->comp_buf[m++] = i;
            if (m == 0) continue;
            for (int64_t t = 0; t < m; ++t) w->buf[t] = (double)demand_f32[w->comp_buf[t]];
            const double demand_sum = sag_tts_f64_sum_numpy(w->buf, m);
            if ((int64_t)ceil(demand_sum / capacity) != 1) continue;
            for (int64_t r = 0; r < m; ++r) {
                const double *row = adj64 + w->comp_buf[r] * n;
                double       *dst = w->buf + r * m;
                for (int64_t c = 0; c < m; ++c) dst[c] = (c > r) ? row[w->comp_buf[c]] : 0.0;
            }
            const double internal_sum = sag_tts_f64_sum_numpy(w->buf, m * m);
            if ((double)m - internal_sum - 1.0 <= 0.01) w->tight_rows[w->n_tight++] = b;
        }
    }

    /* --- main loop --- */
    int64_t n_super = 1;
    for (;;) {
        n_super = normalise_labels(parent, n, w->labels, w->rank);
        partition_data(adj64, n, w->labels, n_super, NULL, w->aggregate64, w->sizes, w->internal);

        for (int64_t s = 0; s < n_super; ++s)
            w->frozen[s] = (w->internal[s] >= ((double)w->sizes[s] - 1.0 + 0.01));
        w->frozen[0] = true;

        for (int64_t s = 0; s < n_super; ++s) w->reps[s] = -1;
        for (int64_t i = 0; i < n; ++i)
            if (w->reps[w->labels[i]] < 0) w->reps[w->labels[i]] = i;

        for (int64_t s = 0; s < n_super; ++s) w->shrinkable[s] = !w->frozen[s];
        w->shrinkable[0] = false;

        bool merged_pair = false;
        for (int64_t left = 0; left < n_super; ++left) {
            if (!w->shrinkable[left]) continue;
            const double *arow = w->aggregate64 + left * n_super;
            for (int64_t right = left + 1; right < n_super; ++right) {
                if (!w->shrinkable[right]) continue;
                if (!(arow[right] >= 0.999)) continue;
                uf_merge(parent, w->reps[left], w->reps[right]);
                merged_pair = true;
            }
        }
        if (merged_pair) continue;

        bool    have_triple = false;
        int64_t t_first = 0, t_second = 0, t_third = 0;
        {
            int64_t n_upper = 0;
            for (int64_t a = 0; a < n_super; ++a)
                for (int64_t b = a + 1; b < n_super; ++b)
                    w->buf[n_upper++] = w->aggregate64[a * n_super + b];
            double bound = 0.0;
            if (n_upper >= 3) {
                double m0 = -INFINITY, m1 = -INFINITY, m2 = -INFINITY;
                for (int64_t i = 0; i < n_upper; ++i) {
                    const double v = w->buf[i];
                    if (v > m0)      { m2 = m1; m1 = m0; m0 = v; }
                    else if (v > m1) { m2 = m1; m1 = v; }
                    else if (v > m2) { m2 = v; }
                }
                bound = (m0 + m1) + m2;
                const double b1 = (m0 + m2) + m1;
                const double b2 = (m1 + m2) + m0;
                double lo = bound, hi = bound;
                if (b1 < lo) lo = b1;
                if (b1 > hi) hi = b1;
                if (b2 < lo) lo = b2;
                if (b2 > hi) hi = b2;
                if ((lo >= 1.999) != (hi >= 1.999))
                    sag_die("tts/compress: the triple pruning upper bound falls in the association-"
                            "order-sensitive interval (%.17g .. %.17g straddles 1.999) -- the landing "
                            "order of np.partition is not replicable, so C and python give different "
                            "supernode partitions here. Top three values = %.17g / %.17g / %.17g", lo, hi, m0, m1, m2);
            }
            if (bound >= 1.999) {
                for (int64_t first = 1; first < n_super - 2 && !have_triple; ++first) {
                    if (w->frozen[first]) continue;
                    for (int64_t second = first + 1; second < n_super - 1; ++second) {
                        if (!agg_positive(w->aggregate64, n_super, w->shrinkable, first, second))
                            continue;
                        const double base = w->aggregate64[first * n_super + second];
                        for (int64_t third = second + 1; third < n_super; ++third) {
                            if (!agg_positive(w->aggregate64, n_super, w->shrinkable, first, third))
                                continue;
                            if (!agg_positive(w->aggregate64, n_super, w->shrinkable, second, third))
                                continue;
                            const double total = base
                                               + w->aggregate64[first * n_super + third]
                                               + w->aggregate64[second * n_super + third];
                            if (total >= 1.999) {
                                have_triple = true;
                                t_first = first; t_second = second; t_third = third;
                                break;
                            }
                        }
                        if (have_triple) break;
                    }
                }
            }
        }
        if (have_triple) {
            uf_merge(parent, w->reps[t_first], w->reps[t_second]);
            uf_merge(parent, w->reps[t_first], w->reps[t_third]);
            continue;
        }

        bool merged_old = false;
        for (int64_t t = 0; t < w->n_tight && !merged_old; ++t) {
            const float *selection = old_sets + w->tight_rows[t] * n;
            for (int64_t s = 0; s < n_super; ++s) w->comp_seen[s] = false;
            for (int64_t i = 1; i < n; ++i)
                if (selection[i] > 0.5f) w->comp_seen[w->labels[i]] = true;
            int64_t n_comp = 0;
            for (int64_t s = 1; s < n_super; ++s)
                if (w->comp_seen[s]) w->comp_buf[n_comp++] = s;
            if (n_comp <= 1) continue;
            bool any_frozen = false;
            for (int64_t i = 0; i < n_comp; ++i)
                if (w->frozen[w->comp_buf[i]]) { any_frozen = true; break; }
            if (any_frozen) continue;
            for (int64_t i = 1; i < n_comp; ++i)
                uf_merge(parent, w->reps[w->comp_buf[0]], w->reps[w->comp_buf[i]]);
            merged_old = true;
        }
        if (merged_old) continue;
        break;
    }

    memcpy(out_labels, w->labels, (size_t)n * sizeof(int64_t));
    memcpy(out_frozen, w->frozen, (size_t)n_super * sizeof(bool));
    return n_super;
}

/* ===================================================================== */
/* 5. build_compression_state                                             */
/* ===================================================================== */

void sag_tts_build_compression(sag_arena *a, const float *adj_view, const float *demand_f32,
                               double capacity, const float *old_sets, int64_t n_old,
                               int64_t n, double eps, sag_tts_compression *out)
{
    if (n <= 0) sag_die("tts/compress: n_nodes=%lld", (long long)n);

    int64_t *labels = SAG_NEW(a, int64_t, n);
    bool    *frozen = SAG_NEW(a, bool, n);

    int64_t n_super = 0;
    int64_t one_edge_cust = 0;
    {
        const size_t mark = sag_arena_mark(a);
        int64_t *parent = SAG_NEW(a, int64_t, n);
        int64_t *rank   = SAG_NEW(a, int64_t, n);
        int64_t *tmp_lb = SAG_NEW(a, int64_t, n);

        (void)one_edge_partition_f32(adj_view, n, eps, parent, tmp_lb, rank);
        {
            bool *seen = SAG_NEW0(a, bool, n);
            for (int64_t i = 1; i < n; ++i)
                if (!seen[tmp_lb[i]]) { seen[tmp_lb[i]] = true; one_edge_cust += 1; }
        }

        double *adj64 = SAG_NEW(a, double, n * n);
        for (int64_t i = 0; i < n; ++i) {
            const float  *src = adj_view + i * n;
            double       *dst = adj64 + i * n;
            for (int64_t j = 0; j < n; ++j) dst[j] = (i == j) ? 0.0 : (double)src[j];
        }

        compress_work w;
        w.parent      = parent;
        w.labels      = tmp_lb;
        w.rank        = rank;
        w.aggregate64 = SAG_NEW(a, double, n * n);
        w.sizes       = SAG_NEW(a, int64_t, n);
        w.internal    = SAG_NEW(a, double, n);
        w.frozen      = SAG_NEW(a, bool, n);
        w.shrinkable  = SAG_NEW(a, bool, n);
        w.reps        = SAG_NEW(a, int64_t, n);
        w.buf         = SAG_NEW(a, double, n * n);
        w.tight_rows  = SAG_NEW(a, int64_t, n_old > 0 ? n_old : 1);
        w.n_tight     = 0;
        w.comp_buf    = SAG_NEW(a, int64_t, n);
        w.comp_seen   = SAG_NEW(a, bool, n);

        n_super = compress_partition(adj64, n, demand_f32, capacity, old_sets, n_old,
                                     eps, &w, labels, frozen);
        sag_arena_release(a, mark);
    }

    float   *member    = SAG_NEW(a, float, n_super * n);
    float   *aggregate = SAG_NEW(a, float, n_super * n_super);
    float   *dem_super = SAG_NEW(a, float, n_super);
    int64_t *sizes     = SAG_NEW(a, int64_t, n_super);
    double  *internal  = SAG_NEW(a, double, n_super);
    {
        const size_t mark = sag_arena_mark(a);
        double *raw64 = SAG_NEW(a, double, n * n);
        for (int64_t i = 0; i < n * n; ++i) raw64[i] = (double)adj_view[i];
        double *member64 = SAG_NEW(a, double, n_super * n);
        double *agg64    = SAG_NEW(a, double, n_super * n_super);
        partition_data(raw64, n, labels, n_super, member64, agg64, sizes, internal);
        for (int64_t i = 0; i < n_super * n; ++i)        member[i]    = (float)member64[i];
        for (int64_t i = 0; i < n_super * n_super; ++i)  aggregate[i] = (float)agg64[i];
        sag_arena_release(a, mark);
    }
    for (int64_t s = 0; s < n_super; ++s) {
        int64_t acc = 0;
        const float *row = member + s * n;
        for (int64_t j = 0; j < n; ++j)
            if (row[j] > 0.5f) acc += (int64_t)demand_f32[j];
        dem_super[s] = (float)acc;
    }

    out->n_nodes    = n;
    out->n_super    = n_super;
    out->labels     = labels;
    out->frozen     = frozen;
    out->member     = member;
    out->aggregate  = aggregate;
    out->demand     = dem_super;
    out->sizes      = sizes;
    out->internal_x = internal;
    out->one_edge_customer_supernodes = one_edge_cust;
}
