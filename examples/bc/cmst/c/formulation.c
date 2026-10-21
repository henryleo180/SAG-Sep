#include "formulation.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

void bc_grb_check(int error, GRBmodel *model, const char *what)
{
    if (error == 0) return;
    GRBenv *env = model ? GRBgetenv(model) : NULL;
    const char *message = env ? GRBgeterrormsg(env) : NULL;
    bc_die("Gurobi C API error %d in %s%s%s", error, what,
           message ? ": " : "", message ? message : "");
}

/* ===================================================================== */
/* numpy-compatible float64 reduction                                     */
/* ===================================================================== */

#define BC_PAIRWISE_BLOCKSIZE 128
#define BC_REDUCE_BUFSIZE     8192

static double pairwise_sum(const double *a, int64_t n)
{
    if (n < 8) {
        double result = 0.0;
        for (int64_t i = 0; i < n; ++i) result += a[i];
        return result;
    }
    if (n <= BC_PAIRWISE_BLOCKSIZE) {
        double r[8];
        for (int64_t k = 0; k < 8; ++k) r[k] = a[k];
        int64_t i = 8;
        for (; i < n - (n % 8); i += 8) {
            for (int64_t k = 0; k < 8; ++k) r[k] += a[i + k];
        }
        double result = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) result += a[i];
        return result;
    }
    int64_t half = n / 2;
    half -= half % 8;
    return pairwise_sum(a, half) + pairwise_sum(a + half, n - half);
}

double bc_reduce_sum(const double *a, int64_t n)
{
    double result = 0.0;
    for (int64_t offset = 0; offset < n; offset += BC_REDUCE_BUFSIZE) {
        const int64_t chunk = (n - offset < BC_REDUCE_BUFSIZE) ? n - offset : BC_REDUCE_BUFSIZE;
        result += pairwise_sum(a + offset, chunk);
    }
    return result;
}

/* ===================================================================== */
/* Model construction                                                     */
/* ===================================================================== */

int64_t bc_min_root_degree(const bc_instance *inst)
{
    __int128 total = 0;
    for (int64_t i = 0; i < inst->n_nodes; ++i) total += inst->demand[i];
    const __int128 capacity = inst->capacity;
    const int64_t k = (int64_t)(total / capacity + (total % capacity != 0));
    return k > 1 ? k : 1;
}

void bc_model_build(GRBenv *env, const bc_instance *inst, bc_model *out)
{
    const int64_t n = inst->n_nodes;
    const int64_t n_edges = n * (n - 1) / 2;
    if (n_edges > INT32_MAX)
        bc_die("instance %s has %lld edges, more than one LP can address",
               inst->name, (long long)n_edges);

    memset(out, 0, sizeof *out);
    out->inst = inst;
    out->root_degree = bc_min_root_degree(inst);
    out->n_edge_vars = (int)n_edges;
    out->edge_i = (int32_t *)bc_malloc((size_t)n_edges * sizeof *out->edge_i);
    out->edge_j = (int32_t *)bc_malloc((size_t)n_edges * sizeof *out->edge_j);
    out->var_index = (int32_t *)bc_malloc((size_t)(n * n) * sizeof *out->var_index);

    double *obj = (double *)bc_malloc((size_t)n_edges * sizeof *obj);
    double *lb = (double *)bc_calloc((size_t)n_edges, sizeof *lb);
    double *ub = (double *)bc_malloc((size_t)n_edges * sizeof *ub);
    char *vtypes = (char *)bc_malloc((size_t)n_edges * sizeof *vtypes);

    for (int64_t i = 0; i < n; ++i) out->var_index[i * n + i] = -1;
    int64_t column = 0;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i + 1; j < n; ++j, ++column) {
            out->edge_i[column] = (int32_t)i;
            out->edge_j[column] = (int32_t)j;
            out->var_index[i * n + j] = (int32_t)column;
            out->var_index[j * n + i] = (int32_t)column;
            obj[column] = inst->weight[i * n + j];
            const bool incompatible =
                i > 0 && inst->demand[i] > inst->capacity - inst->demand[j];
            if (incompatible) out->fixed_pairs += 1;
            ub[column] = incompatible ? 0.0 : 1.0;
            vtypes[column] = GRB_BINARY;
        }
    }

    bc_grb_check(GRBnewmodel(env, &out->model, "cmst_tree", 0, NULL, NULL, NULL, NULL, NULL),
                 NULL, "GRBnewmodel");
    bc_grb_check(GRBaddvars(out->model, out->n_edge_vars, 0, NULL, NULL, NULL,
                            obj, lb, ub, vtypes, NULL),
                 out->model, "GRBaddvars");
    bc_grb_check(GRBsetintattr(out->model, GRB_INT_ATTR_MODELSENSE, GRB_MINIMIZE),
                 out->model, "set ModelSense");
    bc_grb_check(GRBupdatemodel(out->model), out->model, "GRBupdatemodel");
    free(obj);
    free(lb);
    free(ub);
    free(vtypes);

    int *indices = (int *)bc_malloc((size_t)n_edges * sizeof *indices);
    double *ones = (double *)bc_malloc((size_t)n_edges * sizeof *ones);
    for (int64_t k = 0; k < n_edges; ++k) ones[k] = 1.0;

    for (int64_t k = 0; k < n_edges; ++k) indices[k] = (int)k;
    bc_grb_check(GRBaddconstr(out->model, out->n_edge_vars, indices, ones, GRB_EQUAL,
                              (double)(n - 1), "edge_count"),
                 out->model, "GRBaddconstr (edge count)");
    for (int64_t other = 1; other < n; ++other) indices[other - 1] = out->var_index[other];
    bc_grb_check(GRBaddconstr(out->model, (int)(n - 1), indices, ones, GRB_GREATER_EQUAL,
                              (double)out->root_degree, "root_degree"),
                 out->model, "GRBaddconstr (root degree)");
    char name[64];
    for (int64_t node = 1; node < n; ++node) {
        int count = 0;
        for (int64_t other = 0; other < n; ++other) {
            if (other == node) continue;
            indices[count++] = out->var_index[node * n + other];
        }
        snprintf(name, sizeof name, "deg_%lld", (long long)(node + 1));
        bc_grb_check(GRBaddconstr(out->model, count, indices, ones, GRB_GREATER_EQUAL,
                                  1.0, name),
                     out->model, "GRBaddconstr (customer degree)");
    }
    bc_grb_check(GRBupdatemodel(out->model), out->model, "GRBupdatemodel");
    free(indices);
    free(ones);

    const int64_t n_customers = n - 1;
    out->dense = (double *)bc_malloc((size_t)(n * n) * sizeof *out->dense);
    out->gather = (double *)bc_malloc((size_t)(n_customers * n_customers) * sizeof *out->gather);
    out->in_subset = (char *)bc_calloc((size_t)n, sizeof *out->in_subset);
}

void bc_model_free(bc_model *m)
{
    if (m->model) GRBfreemodel(m->model);
    free(m->edge_i);
    free(m->edge_j);
    free(m->var_index);
    free(m->dense);
    free(m->gather);
    free(m->in_subset);
    memset(m, 0, sizeof *m);
}

static void set_vtypes(bc_model *m, char vtype)
{
    char *vtypes = (char *)bc_malloc((size_t)m->n_edge_vars);
    memset(vtypes, vtype, (size_t)m->n_edge_vars);
    bc_grb_check(GRBsetcharattrarray(m->model, GRB_CHAR_ATTR_VTYPE, 0, m->n_edge_vars, vtypes),
                 m->model, "set VType");
    bc_grb_check(GRBupdatemodel(m->model), m->model, "GRBupdatemodel");
    free(vtypes);
}

void bc_model_relax_to_continuous(bc_model *m) { set_vtypes(m, GRB_CONTINUOUS); }

void bc_model_restore_integer_types(bc_model *m) { set_vtypes(m, GRB_BINARY); }

/* ===================================================================== */
/* Cut rows                                                               */
/* ===================================================================== */

void bc_row_init(const bc_model *m, bc_row *row)
{
    const int64_t n = m->inst->n_nodes;
    const int64_t n_customers = n - 1;
    const int64_t inner = n_customers * (n_customers - 1) / 2;
    const int64_t cutset = (n / 2) * (n - n / 2);
    const int64_t capacity = inner > cutset ? inner : cutset;
    row->cols = (int *)bc_malloc((size_t)capacity * sizeof *row->cols);
    row->coeffs = (double *)bc_malloc((size_t)capacity * sizeof *row->coeffs);
    row->nnz = 0;
    row->sense = GRB_LESS_EQUAL;
    row->bound = 0.0;
}

void bc_row_free(bc_row *row)
{
    free(row->cols);
    free(row->coeffs);
    memset(row, 0, sizeof *row);
}

void bc_build_inner_row(const bc_model *m, const int32_t *nodes, int64_t count, int64_t rhs,
                        bc_row *out)
{
    const int64_t n = m->inst->n_nodes;
    out->nnz = 0;
    out->sense = GRB_LESS_EQUAL;
    out->bound = (double)count - (double)rhs;
    for (int64_t a = 0; a < count; ++a) {
        for (int64_t b = a + 1; b < count; ++b) {
            const int32_t column = m->var_index[(int64_t)nodes[a] * n + nodes[b]];
            if (column < 0) continue;
            out->cols[out->nnz] = column;
            out->coeffs[out->nnz] = 1.0;
            out->nnz += 1;
        }
    }
}

void bc_build_cutset_row(const bc_model *m, const int32_t *nodes, int64_t count, int64_t rhs,
                         bc_row *out)
{
    const int64_t n = m->inst->n_nodes;
    out->nnz = 0;
    out->sense = GRB_GREATER_EQUAL;
    out->bound = (double)rhs;
    for (int64_t a = 0; a < count; ++a) m->in_subset[nodes[a]] = 1;
    for (int64_t a = 0; a < count; ++a) {
        const int64_t row = (int64_t)nodes[a] * n;
        for (int64_t b = 0; b < n; ++b) {
            if (m->in_subset[b]) continue;
            const int32_t column = m->var_index[row + b];
            if (column < 0) continue;
            out->cols[out->nnz] = column;
            out->coeffs[out->nnz] = 1.0;
            out->nnz += 1;
        }
    }
    for (int64_t a = 0; a < count; ++a) m->in_subset[nodes[a]] = 0;
}

void bc_check_subset(const bc_model *m, const int32_t *nodes, int64_t count)
{
    const int64_t n = m->inst->n_nodes;
    if (count <= 0) bc_die("separator proposed an empty set");
    for (int64_t a = 0; a < count; ++a) {
        if (nodes[a] == 0)
            bc_die("separator proposed a set containing the root; capacity "
                   "inequalities are indexed by root-free sets only");
        if (nodes[a] < 0 || nodes[a] >= n)
            bc_die("separator proposed node id %d, outside 1..%lld",
                   nodes[a], (long long)(n - 1));
        if (a > 0 && nodes[a] <= nodes[a - 1])
            bc_die("separator proposed a set that is not strictly ascending "
                   "(%d after %d)", nodes[a], nodes[a - 1]);
    }
}

int64_t bc_capacity_rhs(const bc_model *m, const int32_t *nodes, int64_t count)
{
    __int128 total = 0;
    for (int64_t a = 0; a < count; ++a) total += m->inst->demand[nodes[a]];
    const __int128 capacity = m->inst->capacity;
    return (int64_t)(total / capacity + (total % capacity != 0));
}

/* ===================================================================== */
/* Support, violation                                                     */
/* ===================================================================== */

void bc_dense_support(bc_model *m, const double *values)
{
    const int64_t n = m->inst->n_nodes;
    memset(m->dense, 0, (size_t)(n * n) * sizeof *m->dense);
    for (int k = 0; k < m->n_edge_vars; ++k) {
        const int64_t i = m->edge_i[k], j = m->edge_j[k];
        m->dense[i * n + j] = values[k];
        m->dense[j * n + i] = values[k];
    }
}

double bc_inner_violation_from_dense(bc_model *m, const int32_t *nodes, int64_t count,
                                     int64_t rhs)
{
    if (count == 0) return 0.0;
    const int64_t n = m->inst->n_nodes;
    double *block = m->gather;
    for (int64_t a = 0; a < count; ++a) {
        const int64_t row = (int64_t)nodes[a] * n;
        for (int64_t b = 0; b < count; ++b) block[a * count + b] = m->dense[row + nodes[b]];
    }
    const double inside = bc_reduce_sum(block, count * count);
    return 0.5 * inside - (double)count + (double)rhs;
}

double bc_cutset_violation_from_dense(bc_model *m, const int32_t *nodes, int64_t count,
                                      int64_t rhs)
{
    if (count == 0) return 0.0;
    const int64_t n = m->inst->n_nodes;
    double *block = m->gather;
    for (int64_t a = 0; a < count; ++a) m->in_subset[nodes[a]] = 1;
    int64_t gathered = 0;
    for (int64_t a = 0; a < count; ++a) {
        const int64_t row = (int64_t)nodes[a] * n;
        for (int64_t b = 0; b < n; ++b) {
            if (m->in_subset[b]) continue;
            block[gathered++] = m->dense[row + b];
        }
    }
    for (int64_t a = 0; a < count; ++a) m->in_subset[nodes[a]] = 0;
    return (double)rhs - bc_reduce_sum(block, gathered);
}

/* ===================================================================== */
/* Integer solution decomposition                                         */
/* ===================================================================== */

void bc_setlist_init(bc_setlist *list)
{
    memset(list, 0, sizeof *list);
    list->indptr = (int64_t *)bc_malloc(sizeof *list->indptr);
    list->indptr[0] = 0;
}

void bc_setlist_free(bc_setlist *list)
{
    free(list->indptr);
    free(list->nodes);
    free(list->rhs);
    free(list->inner);
    memset(list, 0, sizeof *list);
}

static void setlist_clear(bc_setlist *list)
{
    list->count = 0;
    list->indptr[0] = 0;
}

static void setlist_append(bc_setlist *list, const int32_t *nodes, int64_t count,
                           int64_t rhs, bool inner)
{
    if (list->count + 1 > list->capacity) {
        list->capacity = list->capacity ? list->capacity * 2 : 16;
        list->indptr = (int64_t *)bc_realloc(list->indptr,
                                             (size_t)(list->capacity + 1) * sizeof *list->indptr);
        list->rhs = (int64_t *)bc_realloc(list->rhs, (size_t)list->capacity * sizeof *list->rhs);
        list->inner = (char *)bc_realloc(list->inner, (size_t)list->capacity * sizeof *list->inner);
    }
    const int64_t start = list->indptr[list->count];
    if (start + count > list->node_capacity) {
        while (list->node_capacity < start + count)
            list->node_capacity = list->node_capacity ? list->node_capacity * 2 : 256;
        list->nodes = (int32_t *)bc_realloc(list->nodes,
                                            (size_t)list->node_capacity * sizeof *list->nodes);
    }
    memcpy(list->nodes + start, nodes, (size_t)count * sizeof *nodes);
    list->indptr[list->count + 1] = start + count;
    list->rhs[list->count] = rhs;
    list->inner[list->count] = inner ? 1 : 0;
    list->count += 1;
}

typedef struct {
    int64_t *start;
    int32_t *nodes;
    int64_t *cursor;
} bc_adjacency;

static void adjacency_build(const bc_model *m, const double *values, bc_adjacency *adj)
{
    const int64_t n = m->inst->n_nodes;
    adj->start = (int64_t *)bc_calloc((size_t)(n + 1), sizeof *adj->start);
    int64_t used = 0;
    for (int k = 0; k < m->n_edge_vars; ++k) {
        if (values[k] <= BC_INTEGRALITY_TOLERANCE) continue;
        adj->start[m->edge_i[k] + 1] += 1;
        adj->start[m->edge_j[k] + 1] += 1;
        used += 1;
    }
    for (int64_t u = 0; u < n; ++u) adj->start[u + 1] += adj->start[u];
    adj->nodes = (int32_t *)bc_malloc((size_t)(2 * used + 1) * sizeof *adj->nodes);
    adj->cursor = (int64_t *)bc_malloc((size_t)n * sizeof *adj->cursor);
    for (int64_t u = 0; u < n; ++u) adj->cursor[u] = adj->start[u];
    for (int k = 0; k < m->n_edge_vars; ++k) {
        if (values[k] <= BC_INTEGRALITY_TOLERANCE) continue;
        adj->nodes[adj->cursor[m->edge_i[k]]++] = m->edge_j[k];
        adj->nodes[adj->cursor[m->edge_j[k]]++] = m->edge_i[k];
    }
}

static void adjacency_free(bc_adjacency *adj)
{
    free(adj->start);
    free(adj->nodes);
    free(adj->cursor);
    memset(adj, 0, sizeof *adj);
}

static int compare_int32(const void *lhs, const void *rhs)
{
    const int32_t a = *(const int32_t *)lhs, b = *(const int32_t *)rhs;
    return (a > b) - (a < b);
}

static int64_t component_of(const bc_adjacency *adj, int32_t start, char *visited,
                            int32_t *component, int32_t *stack)
{
    int64_t size = 0, top = 0;
    stack[top++] = start;
    while (top > 0) {
        const int32_t node = stack[--top];
        if (visited[node]) continue;
        visited[node] = 1;
        component[size++] = node;
        for (int64_t e = adj->start[node]; e < adj->start[node + 1]; ++e) {
            if (!visited[adj->nodes[e]]) stack[top++] = adj->nodes[e];
        }
    }
    qsort(component, (size_t)size, sizeof *component, compare_int32);
    return size;
}

int64_t bc_detect_violations(bc_model *m, const double *values, bc_setlist *out)
{
    const int64_t n = m->inst->n_nodes;
    setlist_clear(out);

    bc_adjacency adj;
    adjacency_build(m, values, &adj);
    char *visited = (char *)bc_calloc((size_t)n, sizeof *visited);
    int32_t *component = (int32_t *)bc_malloc((size_t)n * sizeof *component);
    int32_t *stack = (int32_t *)bc_malloc((size_t)(adj.start[n] + 1) * sizeof *stack);

    for (int64_t start = 0; start < n; ++start) {
        if (visited[start]) continue;
        const int64_t size = component_of(&adj, (int32_t)start, visited, component, stack);
        if (component[0] == 0) continue;
        setlist_append(out, component, size, bc_capacity_rhs(m, component, size), false);
    }

    if (out->count == 0) {
        memset(visited, 0, (size_t)n * sizeof *visited);
        visited[0] = 1;
        for (int64_t e = adj.start[0]; e < adj.start[1]; ++e) {
            const int32_t first = adj.nodes[e];
            if (visited[first]) continue;
            const int64_t size = component_of(&adj, first, visited, component, stack);
            const int64_t rhs = bc_capacity_rhs(m, component, size);
            if (rhs > 1) setlist_append(out, component, size, rhs, true);
        }
    }

    free(stack);
    free(component);
    free(visited);
    adjacency_free(&adj);
    return out->count;
}
