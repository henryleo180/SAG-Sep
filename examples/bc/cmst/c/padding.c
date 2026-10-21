#include "padding.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "formulation.h"
#include "util.h"

void bc_padding_init(bc_padding *p, int64_t n_nodes, int64_t n_edge_vars)
{
    if (n_nodes < 2)
        bc_die("padding workspace needs at least one customer, got %lld nodes",
               (long long)n_nodes);
    memset(p, 0, sizeof *p);
    p->n_nodes = n_nodes;
    const size_t edge_slots = (size_t)(n_edge_vars > 0 ? n_edge_vars : 1);
    p->u = (int64_t *)bc_malloc(edge_slots * sizeof *p->u);
    p->v = (int64_t *)bc_malloc(edge_slots * sizeof *p->v);
    p->x = (double *)bc_malloc(edge_slots * sizeof *p->x);
    p->gather = (double *)bc_malloc(edge_slots * sizeof *p->gather);
    p->pair_degree =
        (double *)bc_calloc((size_t)n_nodes, sizeof *p->pair_degree);
    p->in_subset = (char *)bc_calloc((size_t)n_nodes, sizeof *p->in_subset);
}

void bc_padding_free(bc_padding *p)
{
    free(p->u);
    free(p->v);
    free(p->x);
    free(p->gather);
    free(p->pair_degree);
    free(p->in_subset);
    memset(p, 0, sizeof *p);
}

void bc_padding_build(bc_padding *p, const double *dense, bool padded)
{
    const int64_t n = p->n_nodes;
    p->m = 0;
    p->high_degree = 0;
    p->max_clip = 0.0;
    p->pair_degree[0] = 0.0;

    if (!padded) {
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = i + 1; j < n; ++j) {
                const double value = dense[i * n + j];
                if (!(value > BC_PADDED_SUPPORT_THRESHOLD)) continue;
                p->u[p->m] = i;
                p->v[p->m] = j;
                p->x[p->m] = value;
                p->m += 1;
            }
        }
        for (int64_t i = 1; i < n; ++i) p->pair_degree[i] = 0.0;
        return;
    }

    for (int64_t i = 1; i < n; ++i)
        p->pair_degree[i] = bc_reduce_sum(dense + i * n + 1, n - 1);

    for (int64_t i = 1; i < n; ++i) {
        double value = BC_PADDING_TARGET_DEGREE - p->pair_degree[i];
        if (value < 0.0) {
            const double clip = -value;
            p->high_degree += 1;
            if (clip > p->max_clip) p->max_clip = clip;
            value = 0.0;
        }
        if (!(value > BC_PADDED_SUPPORT_THRESHOLD)) continue;
        p->u[p->m] = 0;
        p->v[p->m] = i;
        p->x[p->m] = value;
        p->m += 1;
    }
    for (int64_t i = 1; i < n; ++i) {
        for (int64_t j = i + 1; j < n; ++j) {
            const double value = dense[i * n + j];
            if (!(value > BC_PADDED_SUPPORT_THRESHOLD)) continue;
            p->u[p->m] = i;
            p->v[p->m] = j;
            p->x[p->m] = value;
            p->m += 1;
        }
    }
}

double bc_padding_cut_value(bc_padding *p, const int32_t *nodes, int64_t count)
{
    if (count <= 0) return 0.0;
    for (int64_t a = 0; a < count; ++a) p->in_subset[nodes[a]] = 1;
    int64_t gathered = 0;
    for (int64_t e = 0; e < p->m; ++e) {
        if (p->in_subset[p->u[e]] == p->in_subset[p->v[e]]) continue;
        p->gather[gathered++] = p->x[e];
    }
    for (int64_t a = 0; a < count; ++a) p->in_subset[nodes[a]] = 0;
    return bc_reduce_sum(p->gather, gathered);
}

double bc_padding_identity_gap(bc_padding *p, const int32_t *nodes, int64_t count,
                               int64_t rhs, double inner_violation)
{
    const double cvrp_violation =
        2.0 * (double)rhs - bc_padding_cut_value(p, nodes, count);
    return 0.5 * cvrp_violation - inner_violation;
}
