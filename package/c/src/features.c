#include "features.h"

#include <math.h>
#include <stdbool.h>
#include <string.h>

/* ===================================================================== */
/* 0. numpy float32 pairwise summation                                    */
/* ===================================================================== */

#define SAG_PW_BLOCKSIZE 128

float sag_f32_pairwise_sum(const float *a, int64_t n)
{
    if (n < 8) {
        float res = 0.0f;
        for (int64_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= SAG_PW_BLOCKSIZE) {
        float r[8];
        for (int k = 0; k < 8; ++k) r[k] = a[k];
        int64_t i = 8;
        const int64_t end = n - (n % 8);
        for (; i < end; i += 8) {
            r[0] += a[i + 0]; r[1] += a[i + 1]; r[2] += a[i + 2]; r[3] += a[i + 3];
            r[4] += a[i + 4]; r[5] += a[i + 5]; r[6] += a[i + 6]; r[7] += a[i + 7];
        }
        float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    int64_t n2 = n / 2;
    n2 -= n2 % 8;
    return sag_f32_pairwise_sum(a, n2) + sag_f32_pairwise_sum(a + n2, n - n2);
}

#define SAG_F32_NUMPY_BUFSIZE 8192

float sag_f32_sum_numpy(const float *a, int64_t n)
{
    float res = 0.0f;
    int64_t off = 0;
    while (off < n) {
        int64_t chunk = n - off;
        if (chunk > SAG_F32_NUMPY_BUFSIZE) chunk = SAG_F32_NUMPY_BUFSIZE;
        res += sag_f32_pairwise_sum(a + off, chunk);
        off += chunk;
    }
    return res;
}

/* ===================================================================== */
/* 1. vehicle replica count                                               */
/* ===================================================================== */

int64_t sag_vehicle_replica_count(const sag_instance *inst)
{
    const int64_t limit = (int64_t)1 << 24;
    if (inst->capacity >= limit)
        sag_die("features: capacity %lld >= 2**24 breaks the float32 exactness precondition",
                (long long)inst->capacity);
    int64_t total = 0;
    for (int64_t i = 0; i < inst->n_nodes; ++i) {
        if (inst->demand[i] >= limit)
            sag_die("features: demand[%lld]=%lld >= 2**24 breaks the float32 exactness precondition",
                    (long long)i, (long long)inst->demand[i]);
        total += inst->demand[i];
    }
    if (total >= limit)
        sag_die("features: sum(demand)=%lld >= 2**24 breaks the float32 exactness precondition", (long long)total);

    const float total32 = (float)total;
    float cap32 = (float)inst->capacity;
    if (cap32 < 1e-6f) cap32 = 1e-6f;
    const int64_t k = (int64_t)ceilf(total32 / cap32);
    return k > 1 ? k : 1;
}

/* ===================================================================== */
/* 2. LP support (= _build_mbp_lp_adjacency + LPSupport.from_dense)      */
/* ===================================================================== */

static void csr_from_dense(sag_arena *a, const float *dense, int64_t n, int64_t cap_nnz,
                           int64_t **out_indptr, int32_t **out_indices,
                           float **out_f32, double **out_f64, int64_t *out_nnz)
{
    int64_t *indptr  = SAG_NEW(a, int64_t, n + 1);
    int32_t *indices = SAG_NEW(a, int32_t, cap_nnz);
    float   *data32  = SAG_NEW(a, float,   cap_nnz);
    double  *data64  = SAG_NEW(a, double,  cap_nnz);

    int64_t nnz = 0;
    for (int64_t i = 0; i < n; ++i) {
        indptr[i] = nnz;
        const float *row = dense + i * n;
        for (int64_t j = 0; j < n; ++j) {
            if (row[j] != 0.0f) {
                if (nnz >= cap_nnz)
                    sag_die("features: CSR nnz exceeds the reserved %lld (do the support arcs contain duplicates?)",
                            (long long)cap_nnz);
                indices[nnz] = (int32_t)j;
                data32[nnz]  = row[j];
                data64[nnz]  = (double)row[j];
                ++nnz;
            }
        }
    }
    indptr[n] = nnz;

    *out_indptr  = indptr;
    *out_indices = indices;
    *out_f32     = data32;
    *out_f64     = data64;
    *out_nnz     = nnz;
}

void sag_build_lp_support(sag_arena *a,
                          const int64_t *edge_u, const int64_t *edge_v,
                          const double *edge_x, int64_t m,
                          int64_t n_nodes, double threshold,
                          sag_lp_support *out)
{
    if (n_nodes <= 0) sag_die("features: n_nodes=%lld is invalid", (long long)n_nodes);
    if (m < 0) sag_die("features: m=%lld is invalid", (long long)m);
    if (m > 0 && (!edge_u || !edge_v || !edge_x))
        sag_die("features: build_lp_support is missing the support-arc arrays (m=%lld)", (long long)m);

    const int64_t n = n_nodes;
    float *dense = SAG_NEW0(a, float, n * n);

    const float thr32 = (float)threshold;

    for (int pass = 0; pass < 2; ++pass) {
        for (int64_t i = 0; i < m; ++i) {
            const float val = (float)edge_x[i];
            if (!(val >= thr32)) continue;
            const int64_t u = edge_u[i], v = edge_v[i];
            if (u == v || u < 0 || v < 0 || u >= n || v >= n) continue;
            if (pass == 0) dense[u * n + v] = val;
            else           dense[v * n + u] = val;
        }
    }

    for (int64_t i = 0; i < m; ++i) {
        const float val = (float)edge_x[i];
        if (!(val >= thr32)) continue;
        const int64_t u = edge_u[i], v = edge_v[i];
        if (u == v || u < 0 || v < 0 || u >= n || v >= n) continue;
        if (memcmp(&dense[u * n + v], &dense[v * n + u], sizeof(float)) != 0)
            sag_die("features: the support arcs have a reverse duplicate on the unordered pair (%lld,%lld) -> dense is not symmetric; "
                    "the contract requires u<v and (u,v) strictly increasing in lexicographic order with no duplicates",
                    (long long)u, (long long)v);
    }

    float *degree = SAG_NEW(a, float, n);
    for (int64_t i = 0; i < n; ++i)
        degree[i] = sag_f32_sum_numpy(dense + i * n, n);

    int64_t *indptr = NULL; int32_t *indices = NULL;
    float *data32 = NULL; double *data64 = NULL; int64_t nnz = 0;
    csr_from_dense(a, dense, n, 2 * m, &indptr, &indices, &data32, &data64, &nnz);

    int64_t *eu = SAG_NEW(a, int64_t, m);
    int64_t *ev = SAG_NEW(a, int64_t, m);
    double  *ex = SAG_NEW(a, double,  m);
    int64_t ne = 0;
    for (int64_t i = 0; i < n; ++i) {
        const float *row = dense + i * n;
        for (int64_t j = i + 1; j < n; ++j) {
            if (row[j] != 0.0f) {
                if (ne >= m)
                    sag_die("features: upper-triangular COO exceeds the reserved %lld", (long long)m);
                eu[ne] = i; ev[ne] = j; ex[ne] = (double)row[j];
                ++ne;
            }
        }
    }

    out->n_nodes   = n;
    out->threshold = threshold;
    out->indptr    = indptr;
    out->indices   = indices;
    out->data_f32  = data32;
    out->data_f64  = data64;
    out->nnz       = nnz;
    out->dense     = dense;
    out->degree    = degree;
    out->edge_u    = eu;
    out->edge_v    = ev;
    out->edge_x    = ex;
    out->n_edges   = ne;
}

/* ===================================================================== */
/* 3. 2-hop shortcut (= build_partial_connected_graph, neighbor_dist=2)  */
/* ===================================================================== */

static int64_t two_hop_shortcuts(sag_arena *a, const float *dense, int64_t n,
                                 const int64_t *indptr, const int32_t *indices,
                                 const float *data, bool cross_depot,
                                 int64_t **out_src, int64_t **out_dst, float **out_val)
{
    int64_t cap = 0;
    const int64_t cap_pairs = n * (n - 1);
    for (int64_t b = 0; b < n; ++b) {
        if (!cross_depot && b == 0) continue;
        const int64_t d = indptr[b + 1] - indptr[b];
        cap += d * d;
        if (cap >= cap_pairs) { cap = cap_pairs; break; }
    }

    int64_t *src = SAG_NEW(a, int64_t, cap);
    int64_t *dst = SAG_NEW(a, int64_t, cap);
    float   *val = SAG_NEW(a, float,   cap);
    int64_t *stamp = SAG_NEW0(a, int64_t, n);
    int64_t *slot  = SAG_NEW(a, int64_t, n);

    int64_t count = 0;
    for (int64_t s = 0; s < n; ++s) {
        for (int64_t kb = indptr[s]; kb < indptr[s + 1]; ++kb) {
            const int64_t b = indices[kb];
            if (!cross_depot && s != 0 && b == 0) continue;
            const float w_sb = dense[b * n + s];
            for (int64_t kc = indptr[b]; kc < indptr[b + 1]; ++kc) {
                const int64_t c = indices[kc];
                if (!cross_depot && s != 0 && c == 0) continue;
                if (c == s) continue;
                if (dense[s * n + c] != 0.0f) continue;
                const float w_bc = data[kc];
                const float v = (w_sb < w_bc) ? w_sb : w_bc;
                if (stamp[c] != s + 1) {
                    stamp[c] = s + 1;
                    slot[c] = count;
                    if (count >= cap)
                        sag_die("features: shortcuts exceed the upper bound %lld", (long long)cap);
                    src[count] = s; dst[count] = c; val[count] = v;
                    ++count;
                } else if (v > val[slot[c]]) {
                    val[slot[c]] = v;
                }
            }
        }
    }

    const double scale = pow(SAG_PARTIAL_EDGE_COEF, (double)(SAG_PARTIAL_CONNECT - 1));
    for (int64_t e = 0; e < count; ++e)
        val[e] = (float)((double)val[e] * scale);

    *out_src = src; *out_dst = dst; *out_val = val;
    return count;
}

/* ===================================================================== */
/* 4. graph features                                                      */
/* ===================================================================== */

void sag_build_graph_features(sag_arena *a, const sag_instance *inst,
                              const sag_lp_support *lp, int64_t vehicle_count,
                              const int64_t *sup_u, const int64_t *sup_v,
                              const double *sup_x, int64_t sup_m,
                              bool cross_depot,
                              sag_graph_features *out)
{
    const int64_t n = inst->n_nodes;
    if (lp->n_nodes != n)
        sag_die("features: lp.n_nodes=%lld != instance.n_nodes=%lld",
                (long long)lp->n_nodes, (long long)n);
    if (vehicle_count < 1)
        sag_die("features: vehicle_count=%lld must be >= 1", (long long)vehicle_count);
    const int64_t k = vehicle_count;

    int64_t n_keep = 0;
    for (int64_t i = 0; i < lp->n_edges; ++i)
        if (lp->edge_x[i] >= lp->threshold) ++n_keep;

    const int64_t *pair_u = NULL, *pair_v = NULL;
    const float   *pair_w = NULL;
    const float   *dense  = NULL;
    const int64_t *indptr = NULL;
    const int32_t *indices = NULL;
    const float   *data    = NULL;
    int64_t m_pairs = 0;

    if (n_keep > 0 && n_keep == lp->n_edges) {
        float *w = SAG_NEW(a, float, n_keep);
        for (int64_t i = 0; i < n_keep; ++i) w[i] = (float)lp->edge_x[i];
        m_pairs = n_keep;
        pair_u = lp->edge_u; pair_v = lp->edge_v; pair_w = w;
        dense = lp->dense; indptr = lp->indptr; indices = lp->indices; data = lp->data_f32;
    } else if (n_keep > 0) {
        int64_t *ru = SAG_NEW(a, int64_t, n_keep);
        int64_t *rv = SAG_NEW(a, int64_t, n_keep);
        float   *rw = SAG_NEW(a, float,   n_keep);
        int64_t t = 0;
        for (int64_t i = 0; i < lp->n_edges; ++i) {
            if (lp->edge_x[i] < lp->threshold) continue;
            ru[t] = lp->edge_u[i]; rv[t] = lp->edge_v[i];
            rw[t] = (float)lp->edge_x[i];
            ++t;
        }
        float *d = SAG_NEW0(a, float, n * n);
        for (int64_t i = 0; i < t; ++i) d[ru[i] * n + rv[i]] = rw[i];
        for (int64_t i = 0; i < t; ++i) d[rv[i] * n + ru[i]] = rw[i];
        int64_t *ip = NULL; int32_t *ix = NULL; float *d32 = NULL; double *d64 = NULL;
        int64_t nnz = 0;
        csr_from_dense(a, d, n, 2 * t, &ip, &ix, &d32, &d64, &nnz);
        m_pairs = t;
        pair_u = ru; pair_v = rv; pair_w = rw;
        dense = d; indptr = ip; indices = ix; data = d32;
    } else {
        if (!sup_u || !sup_v || !sup_x)
            sag_die("features: the fallback branch needs the raw support arcs sup_* "
                    "(on the python side support_edge_* is a mandatory parameter and substituting the lp COO is not accepted)");
        const int64_t *ru = sup_u, *rv = sup_v;
        const double  *rx = sup_x;
        const int64_t  rm = sup_m;
        int64_t best = -1;
        float best_w = 0.0f;
        for (int64_t i = 0; i < rm; ++i) {
            if (ru[i] == rv[i]) continue;
            const float w = (float)rx[i];
            if (best < 0 || w > best_w) { best = i; best_w = w; }
        }
        if (best < 0) sag_die("features: Graph must contain at least one edge");
        const int64_t bu = ru[best], bv = rv[best];
        if (bu < 0 || bv < 0 || bu >= n || bv >= n)
            sag_die("features: fallback edge (%lld,%lld) is out of range for n=%lld",
                    (long long)bu, (long long)bv, (long long)n);
        printf("[eval-policy] lp_support_edge_threshold=%g removed all edges; "
               "keeping max-weight edge (%lld, %lld)=%.6f as fallback.\n",
               lp->threshold, (long long)(bu + 1), (long long)(bv + 1), (double)best_w);

        int64_t *fu = SAG_NEW(a, int64_t, 1);
        int64_t *fv = SAG_NEW(a, int64_t, 1);
        float   *fw = SAG_NEW(a, float,   1);
        fu[0] = bu; fv[0] = bv; fw[0] = best_w;
        float *d = SAG_NEW0(a, float, n * n);
        d[bu * n + bv] = best_w;
        d[bv * n + bu] = best_w;
        int64_t *ip = NULL; int32_t *ix = NULL; float *d32 = NULL; double *d64 = NULL;
        int64_t nnz = 0;
        csr_from_dense(a, d, n, 2, &ip, &ix, &d32, &d64, &nnz);
        m_pairs = 1;
        pair_u = fu; pair_v = fv; pair_w = fw;
        dense = d; indptr = ip; indices = ix; data = d32;
    }

    int64_t *sc_src = NULL, *sc_dst = NULL;
    float   *sc_val = NULL;
    const int64_t n_sc = two_hop_shortcuts(a, dense, n, indptr, indices, data,
                                           cross_depot,
                                           &sc_src, &sc_dst, &sc_val);

    bool *to_depot   = SAG_NEW0(a, bool, n);
    bool *from_depot = SAG_NEW0(a, bool, n);
    for (int64_t i = 0; i < n; ++i) {
        to_depot[i]   = dense[i * n + 0] > 0.0f;
        from_depot[i] = dense[0 * n + i] > 0.0f;
    }
    for (int64_t e = 0; e < n_sc; ++e) {
        if (sc_dst[e] == 0) to_depot[sc_src[e]] = true;
        if (sc_src[e] == 0) from_depot[sc_dst[e]] = true;
    }
    int64_t n_dep = 0;
    for (int64_t node = 1; node < n; ++node) {
        if (!to_depot[node]) ++n_dep;
        if (!from_depot[node]) ++n_dep;
    }

    const int64_t n_sup = 2 * m_pairs;
    const int64_t e_total = n_sup + n_sc + n_dep;
    if (e_total <= 0)
        sag_die("features: graph must contain at least one edge");

    int64_t *edge_index = SAG_NEW(a, int64_t, 2 * e_total);
    float   *edge_attr  = SAG_NEW0(a, float, e_total * SAG_EDGE_DIM);
    int64_t *src = edge_index;
    int64_t *dst = edge_index + e_total;

    for (int64_t i = 0; i < m_pairs; ++i) {
        src[2 * i]     = pair_u[i]; dst[2 * i]     = pair_v[i];
        src[2 * i + 1] = pair_v[i]; dst[2 * i + 1] = pair_u[i];
        edge_attr[(2 * i) * SAG_EDGE_DIM]         = pair_w[i];
        edge_attr[(2 * i) * SAG_EDGE_DIM + 1]     = SAG_HOP_ORIGINAL;
        edge_attr[(2 * i + 1) * SAG_EDGE_DIM]     = pair_w[i];
        edge_attr[(2 * i + 1) * SAG_EDGE_DIM + 1] = SAG_HOP_ORIGINAL;
    }
    for (int64_t e = 0; e < n_sc; ++e) {
        const int64_t p = n_sup + e;
        src[p] = sc_src[e]; dst[p] = sc_dst[e];
        edge_attr[p * SAG_EDGE_DIM]     = sc_val[e];
        edge_attr[p * SAG_EDGE_DIM + 1] = SAG_HOP_SHORTCUT;
    }
    int64_t p = n_sup + n_sc;
    for (int64_t node = 1; node < n; ++node) {
        if (!to_depot[node]) {
            src[p] = node; dst[p] = 0;
            edge_attr[p * SAG_EDGE_DIM]     = 0.0f;
            edge_attr[p * SAG_EDGE_DIM + 1] = SAG_HOP_DEPOT_FILL;
            ++p;
        }
        if (!from_depot[node]) {
            src[p] = 0; dst[p] = node;
            edge_attr[p * SAG_EDGE_DIM]     = 0.0f;
            edge_attr[p * SAG_EDGE_DIM + 1] = SAG_HOP_DEPOT_FILL;
            ++p;
        }
    }
    if (p != e_total)
        sag_die("features: depot fill-in edge count is inconsistent %lld != %lld",
                (long long)p, (long long)e_total);

    float cap32 = (float)inst->capacity;
    if (cap32 < 1e-6f) cap32 = 1e-6f;
    const double scale_span = (double)((k - 1) > 1 ? (k - 1) : 1);
    float *node_attr = SAG_NEW(a, float, k * n * SAG_NODE_DIM);
    for (int64_t v = 0; v < k; ++v) {
        const float scaled_m = (float)((double)v / scale_span);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t base = (v * n + i) * SAG_NODE_DIM;
            node_attr[base]     = (float)inst->demand[i] / cap32;
            node_attr[base + 1] = scaled_m;
        }
    }

    out->n_nodes         = n;
    out->vehicle_count   = k;
    out->n_edges         = e_total;
    out->edge_index      = edge_index;
    out->edge_attr       = edge_attr;
    out->node_attr       = node_attr;
    out->n_support_arcs  = n_sup;
    out->n_shortcut_arcs = n_sc;
    out->n_depot_arcs    = n_dep;
}
