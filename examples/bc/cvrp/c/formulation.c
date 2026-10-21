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

static double reduce_sum(const double *a, int64_t n)
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

void bc_model_build(GRBenv *env, const bc_instance *inst, int64_t vehicles, bc_model *out)
{
    if (vehicles <= 0) bc_die("vehicles must be positive, got %lld", (long long)vehicles);
    const int64_t n = inst->n_nodes;
    const int64_t n_edges = n * (n - 1) / 2;
    if (n_edges > INT32_MAX)
        bc_die("instance %s has %lld edges, more than one LP can address",
               inst->name, (long long)n_edges);

    memset(out, 0, sizeof *out);
    out->inst = inst;
    out->vehicles = vehicles;
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
            ub[column] = (i == 0) ? 2.0 : 1.0;
            vtypes[column] = (i == 0) ? GRB_INTEGER : GRB_BINARY;
        }
    }

    bc_grb_check(GRBnewmodel(env, &out->model, "cvrp_two_index", 0, NULL, NULL, NULL, NULL, NULL),
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

    int *indices = (int *)bc_malloc((size_t)n * sizeof *indices);
    double *ones = (double *)bc_malloc((size_t)n * sizeof *ones);
    for (int64_t i = 0; i < n; ++i) ones[i] = 1.0;
    char name[64];
    for (int64_t node = 1; node < n; ++node) {
        int count = 0;
        for (int64_t other = 0; other < n; ++other) {
            if (other == node) continue;
            indices[count++] = out->var_index[node * n + other];
        }
        snprintf(name, sizeof name, "deg_%lld", (long long)(node + 1));
        bc_grb_check(GRBaddconstr(out->model, count, indices, ones, GRB_EQUAL, 2.0, name),
                     out->model, "GRBaddconstr (customer degree)");
    }
    for (int64_t other = 1; other < n; ++other) indices[other - 1] = out->var_index[other];
    bc_grb_check(GRBaddconstr(out->model, (int)(n - 1), indices, ones, GRB_EQUAL,
                              2.0 * (double)vehicles, "deg_depot"),
                 out->model, "GRBaddconstr (fleet)");
    bc_grb_check(GRBupdatemodel(out->model), out->model, "GRBupdatemodel");
    free(indices);
    free(ones);

    const int64_t n_customers = n - 1;
    out->dense = (double *)bc_malloc((size_t)(n * n) * sizeof *out->dense);
    out->degree = (double *)bc_malloc((size_t)n * sizeof *out->degree);
    out->gather = (double *)bc_malloc((size_t)(n_customers * n_customers + n_customers)
                                      * sizeof *out->gather);
    out->in_subset = (char *)bc_calloc((size_t)n, sizeof *out->in_subset);
    out->complement = (int32_t *)bc_malloc((size_t)n * sizeof *out->complement);
}

void bc_model_free(bc_model *m)
{
    if (m->model) GRBfreemodel(m->model);
    free(m->edge_i);
    free(m->edge_j);
    free(m->var_index);
    free(m->dense);
    free(m->degree);
    free(m->gather);
    free(m->in_subset);
    free(m->complement);
    memset(m, 0, sizeof *m);
}

static void set_vtypes(bc_model *m, const char *vtypes)
{
    bc_grb_check(GRBsetcharattrarray(m->model, GRB_CHAR_ATTR_VTYPE, 0, m->n_edge_vars,
                                     (char *)vtypes),
                 m->model, "set VType");
    bc_grb_check(GRBupdatemodel(m->model), m->model, "GRBupdatemodel");
}

void bc_model_relax_to_continuous(bc_model *m)
{
    char *vtypes = (char *)bc_malloc((size_t)m->n_edge_vars);
    memset(vtypes, GRB_CONTINUOUS, (size_t)m->n_edge_vars);
    set_vtypes(m, vtypes);
    free(vtypes);
}

void bc_model_restore_integer_types(bc_model *m)
{
    char *vtypes = (char *)bc_malloc((size_t)m->n_edge_vars);
    for (int k = 0; k < m->n_edge_vars; ++k)
        vtypes[k] = (m->edge_i[k] == 0) ? GRB_INTEGER : GRB_BINARY;
    set_vtypes(m, vtypes);
    free(vtypes);
}

/* ===================================================================== */
/* Cut rows                                                               */
/* ===================================================================== */

void bc_row_init(const bc_model *m, bc_row *row)
{
    const int64_t n_customers = m->inst->n_nodes - 1;
    const int64_t capacity = n_customers * (n_customers - 1) / 2 + n_customers;
    row->cols = (int *)bc_malloc((size_t)capacity * sizeof *row->cols);
    row->coeffs = (double *)bc_malloc((size_t)capacity * sizeof *row->coeffs);
    row->nnz = 0;
    row->bound = 0.0;
}

void bc_row_free(bc_row *row)
{
    free(row->cols);
    free(row->coeffs);
    memset(row, 0, sizeof *row);
}

static void append_inner_edges(const bc_model *m, const int32_t *subset, int64_t count,
                               bc_row *row)
{
    if (count < 2) return;
    const int64_t n = m->inst->n_nodes;
    for (int64_t a = 0; a < count; ++a) {
        for (int64_t b = a + 1; b < count; ++b) {
            const int32_t column = m->var_index[(int64_t)subset[a] * n + subset[b]];
            if (column < 0) continue;
            row->cols[row->nnz] = column;
            row->coeffs[row->nnz] = 1.0;
            row->nnz += 1;
        }
    }
}

static void append_depot_edges(const bc_model *m, const int32_t *subset, int64_t count,
                               double coefficient, bc_row *row)
{
    for (int64_t a = 0; a < count; ++a) {
        const int32_t column = m->var_index[subset[a]];
        if (column < 0) continue;
        row->cols[row->nnz] = column;
        row->coeffs[row->nnz] = coefficient;
        row->nnz += 1;
    }
}

void bc_build_cut_row(const bc_model *m, const int32_t *nodes, int64_t count, double rhs,
                      bc_row *out)
{
    const int64_t n = m->inst->n_nodes;
    const int64_t n_customers = n - 1;
    out->nnz = 0;

    if ((double)count > (double)n_customers / 2.0) {
        for (int64_t a = 0; a < count; ++a) m->in_subset[nodes[a]] = 1;
        int64_t complement_size = 0;
        for (int64_t node = 1; node < n; ++node) {
            if (!m->in_subset[node]) m->complement[complement_size++] = (int32_t)node;
        }
        for (int64_t a = 0; a < count; ++a) m->in_subset[nodes[a]] = 0;

        if (complement_size > 0) {
            append_inner_edges(m, m->complement, complement_size, out);
            append_depot_edges(m, m->complement, complement_size, 0.5, out);
            append_depot_edges(m, nodes, count, -0.5, out);
            out->bound = (double)complement_size - rhs;
            return;
        }
    }

    append_inner_edges(m, nodes, count, out);
    out->bound = (double)count - rhs;
}

double bc_rcc_rhs(const bc_model *m, const int32_t *nodes, int64_t count)
{
    __int128 total = 0;
    for (int64_t a = 0; a < count; ++a) total += m->inst->demand[nodes[a]];
    const __int128 capacity = m->inst->capacity;
    return (double)(int64_t)(total / capacity + (total % capacity != 0));
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
    for (int64_t u = 0; u < n; ++u) {
        double accumulator = 0.0;
        for (int64_t v = u + 1; v < n; ++v) accumulator += m->dense[u * n + v];
        for (int64_t v = 0; v < u; ++v) accumulator += m->dense[u * n + v];
        m->degree[u] = accumulator;
    }
}

double bc_violation_from_dense(bc_model *m, const int32_t *nodes, int64_t count, double rhs)
{
    if (count == 0) return 0.0;
    const int64_t n = m->inst->n_nodes;
    double *block = m->gather;
    for (int64_t a = 0; a < count; ++a) {
        const int64_t row = (int64_t)nodes[a] * n;
        for (int64_t b = 0; b < count; ++b) block[a * count + b] = m->dense[row + nodes[b]];
    }
    const double inside = reduce_sum(block, count * count);
    double *degrees = m->gather + count * count;
    for (int64_t a = 0; a < count; ++a) degrees[a] = m->degree[nodes[a]];
    const double boundary = reduce_sum(degrees, count) - inside;
    return 2.0 * rhs - boundary;
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
    memset(list, 0, sizeof *list);
}

static void setlist_clear(bc_setlist *list)
{
    list->count = 0;
    list->indptr[0] = 0;
}

static void setlist_append(bc_setlist *list, const int32_t *nodes, int64_t count, double rhs)
{
    if (list->count + 1 > list->capacity) {
        list->capacity = list->capacity ? list->capacity * 2 : 16;
        list->indptr = (int64_t *)bc_realloc(list->indptr,
                                             (size_t)(list->capacity + 1) * sizeof *list->indptr);
        list->rhs = (double *)bc_realloc(list->rhs, (size_t)list->capacity * sizeof *list->rhs);
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
        int64_t size = 0, top = 0;
        bool has_depot = false;
        stack[top++] = (int32_t)start;
        while (top > 0) {
            const int32_t node = stack[--top];
            if (visited[node]) continue;
            visited[node] = 1;
            if (node == 0) has_depot = true;
            component[size++] = node;
            for (int64_t e = adj.start[node]; e < adj.start[node + 1]; ++e) {
                if (!visited[adj.nodes[e]]) stack[top++] = adj.nodes[e];
            }
        }
        if (has_depot) continue;
        qsort(component, (size_t)size, sizeof *component, compare_int32);
        setlist_append(out, component, size, bc_rcc_rhs(m, component, size));
    }

    memset(visited, 0, (size_t)n * sizeof *visited);
    for (int64_t e = adj.start[0]; e < adj.start[1]; ++e) {
        const int32_t first = adj.nodes[e];
        if (visited[first]) continue;
        int64_t size = 0;
        int32_t previous = 0, current = first;
        while (current != 0) {
            if (visited[current]) break;
            visited[current] = 1;
            component[size++] = current;
            int32_t following = -1;
            for (int64_t t = adj.start[current]; t < adj.start[current + 1]; ++t) {
                if (adj.nodes[t] != previous) { following = adj.nodes[t]; break; }
            }
            if (following < 0) break;
            previous = current;
            current = following;
        }
        if (size == 0) continue;
        const double rhs = bc_rcc_rhs(m, component, size);
        if (rhs <= 1.0) continue;
        qsort(component, (size_t)size, sizeof *component, compare_int32);
        setlist_append(out, component, size, rhs);
    }

    free(stack);
    free(component);
    free(visited);
    adjacency_free(&adj);
    return out->count;
}
