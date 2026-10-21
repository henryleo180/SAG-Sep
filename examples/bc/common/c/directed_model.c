#include "directed_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

void bcd_grb_check(int error, GRBmodel *model, const char *what)
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

void bcd_model_build(GRBenv *env, const bc_instance *inst, int64_t vehicles,
                     bool open_routes, bcd_model *out)
{
    if (vehicles <= 0) bc_die("vehicles must be positive, got %lld", (long long)vehicles);
    const int64_t n = inst->n_nodes;
    const int64_t n_customers = n - 1;
    if (vehicles > n_customers)
        bc_die("a fixed fleet of %lld cannot serve %lld customers: every route must "
               "visit at least one", (long long)vehicles, (long long)n_customers);
    const int64_t n_arcs = n * (n - 1);
    if (n_arcs > INT32_MAX)
        bc_die("instance %s has %lld arcs, more than one LP can address", inst->name,
               (long long)n_arcs);

    memset(out, 0, sizeof *out);
    out->inst = inst;
    out->vehicles = vehicles;
    out->open_routes = open_routes;
    out->n_model_nodes = open_routes ? n + 1 : n;
    const int64_t sink = open_routes ? n : -1;
    out->terminal = open_routes ? sink : 0;
    out->n_arc_vars = (int)n_arcs;
    out->arc_tail = (int32_t *)bc_malloc((size_t)n_arcs * sizeof *out->arc_tail);
    out->arc_head = (int32_t *)bc_malloc((size_t)n_arcs * sizeof *out->arc_head);
    out->arc_sep_tail = (int32_t *)bc_malloc((size_t)n_arcs * sizeof *out->arc_sep_tail);
    out->arc_sep_head = (int32_t *)bc_malloc((size_t)n_arcs * sizeof *out->arc_sep_head);

    double *obj = (double *)bc_malloc((size_t)n_arcs * sizeof *obj);
    double *lb = (double *)bc_calloc((size_t)n_arcs, sizeof *lb);
    double *ub = (double *)bc_malloc((size_t)n_arcs * sizeof *ub);
    char *vtypes = (char *)bc_malloc((size_t)n_arcs);

    int64_t column = 0;
    for (int64_t tail = 0; tail < n; ++tail) {
        for (int64_t head = 0; head < n; ++head) {
            if (head == tail || (open_routes && head == 0)) continue;
            out->arc_tail[column] = (int32_t)tail;
            out->arc_head[column] = (int32_t)head;
            obj[column] = inst->weight[tail * n + head];
            ub[column] = 1.0;
            vtypes[column] = GRB_BINARY;
            column += 1;
        }
        if (open_routes && tail != 0) {
            out->arc_tail[column] = (int32_t)tail;
            out->arc_head[column] = (int32_t)sink;
            obj[column] = 0.0;
            ub[column] = 1.0;
            vtypes[column] = GRB_BINARY;
            column += 1;
        }
    }
    if (column != n_arcs) bc_die("arc enumeration produced %lld of %lld columns",
                                 (long long)column, (long long)n_arcs);

    bcd_grb_check(GRBnewmodel(env, &out->model, "directed_two_index", 0,
                             NULL, NULL, NULL, NULL, NULL), NULL, "GRBnewmodel");
    bcd_grb_check(GRBaddvars(out->model, out->n_arc_vars, 0, NULL, NULL, NULL,
                            obj, lb, ub, vtypes, NULL), out->model, "GRBaddvars");
    bcd_grb_check(GRBsetintattr(out->model, GRB_INT_ATTR_MODELSENSE, GRB_MINIMIZE),
                 out->model, "set ModelSense");
    bcd_grb_check(GRBupdatemodel(out->model), out->model, "GRBupdatemodel");
    free(obj);
    free(lb);
    free(ub);
    free(vtypes);

    for (int64_t k = 0; k < n_arcs; ++k) {
        const int32_t tail = out->arc_tail[k], head = out->arc_head[k];
        out->arc_sep_tail[k] = (open_routes && tail == sink) ? 0 : tail;
        out->arc_sep_head[k] = (open_routes && head == sink) ? 0 : head;
    }

    int *indices = (int *)bc_malloc((size_t)n_arcs * sizeof *indices);
    double *ones = (double *)bc_malloc((size_t)n_arcs * sizeof *ones);
    for (int64_t k = 0; k < n_arcs; ++k) ones[k] = 1.0;
    char name[64];
    for (int64_t node = 1; node < n; ++node) {
        int count = 0;
        for (int64_t k = 0; k < n_arcs; ++k) {
            if (out->arc_tail[k] == node) indices[count++] = (int)k;
        }
        snprintf(name, sizeof name, "out_%lld", (long long)(node + 1));
        bcd_grb_check(GRBaddconstr(out->model, count, indices, ones, GRB_EQUAL, 1.0, name),
                     out->model, "GRBaddconstr (out-degree)");
        count = 0;
        for (int64_t k = 0; k < n_arcs; ++k) {
            if (out->arc_head[k] == node) indices[count++] = (int)k;
        }
        snprintf(name, sizeof name, "in_%lld", (long long)(node + 1));
        bcd_grb_check(GRBaddconstr(out->model, count, indices, ones, GRB_EQUAL, 1.0, name),
                     out->model, "GRBaddconstr (in-degree)");
    }
    int count = 0;
    for (int64_t k = 0; k < n_arcs; ++k) {
        if (out->arc_tail[k] == 0) indices[count++] = (int)k;
    }
    bcd_grb_check(GRBaddconstr(out->model, count, indices, ones, GRB_EQUAL,
                              (double)vehicles, "fleet_out"),
                 out->model, "GRBaddconstr (fleet out)");
    count = 0;
    for (int64_t k = 0; k < n_arcs; ++k) {
        if (out->arc_head[k] == out->terminal) indices[count++] = (int)k;
    }
    bcd_grb_check(GRBaddconstr(out->model, count, indices, ones, GRB_EQUAL,
                              (double)vehicles, "fleet_in"),
                 out->model, "GRBaddconstr (fleet in)");
    bcd_grb_check(GRBupdatemodel(out->model), out->model, "GRBupdatemodel");
    free(indices);
    free(ones);

    out->n_pairs = n * (n - 1) / 2;
    out->pair_u = (int32_t *)bc_malloc((size_t)out->n_pairs * sizeof *out->pair_u);
    out->pair_v = (int32_t *)bc_malloc((size_t)out->n_pairs * sizeof *out->pair_v);
    out->pair_forward = (int32_t *)bc_malloc((size_t)out->n_pairs * sizeof *out->pair_forward);
    out->pair_reverse = (int32_t *)bc_malloc((size_t)out->n_pairs * sizeof *out->pair_reverse);
    int64_t *pair_id = (int64_t *)bc_malloc((size_t)(n * n) * sizeof *pair_id);
    int64_t pair = 0;
    for (int64_t u = 0; u < n; ++u) {
        for (int64_t v = u + 1; v < n; ++v, ++pair) {
            out->pair_u[pair] = (int32_t)u;
            out->pair_v[pair] = (int32_t)v;
            pair_id[u * n + v] = pair;
            pair_id[v * n + u] = pair;
            out->pair_forward[pair] = -1;
            out->pair_reverse[pair] = -1;
        }
    }
    for (int64_t k = 0; k < n_arcs; ++k) {
        const int32_t s = out->arc_sep_tail[k], t = out->arc_sep_head[k];
        if (s == t) bc_die("an arc projects onto a single separator node");
        const int64_t p = pair_id[(int64_t)s * n + t];
        if (s < t) out->pair_forward[p] = (int32_t)k;
        else out->pair_reverse[p] = (int32_t)k;
    }
    for (int64_t p = 0; p < out->n_pairs; ++p) {
        if (out->pair_forward[p] < 0 || out->pair_reverse[p] < 0)
            bc_die("a separator pair is not covered by two arcs");
    }
    free(pair_id);

    out->y = (double *)bc_malloc((size_t)out->n_pairs * sizeof *out->y);
    out->dense = (double *)bc_malloc((size_t)(n * n) * sizeof *out->dense);
    out->degree = (double *)bc_malloc((size_t)n * sizeof *out->degree);
    out->gather = (double *)bc_malloc((size_t)(n_customers * n_customers + n_customers)
                                      * sizeof *out->gather);
    out->inside = (char *)bc_calloc((size_t)n, sizeof *out->inside);
    out->successor = (int64_t *)bc_malloc((size_t)out->n_model_nodes * sizeof *out->successor);
    out->visited = (char *)bc_calloc((size_t)out->n_model_nodes, sizeof *out->visited);
    out->walk = (int32_t *)bc_malloc((size_t)n * sizeof *out->walk);
}

void bcd_model_free(bcd_model *m)
{
    if (m->model) GRBfreemodel(m->model);
    free(m->arc_tail);
    free(m->arc_head);
    free(m->arc_sep_tail);
    free(m->arc_sep_head);
    free(m->pair_u);
    free(m->pair_v);
    free(m->pair_forward);
    free(m->pair_reverse);
    free(m->y);
    free(m->dense);
    free(m->degree);
    free(m->gather);
    free(m->inside);
    free(m->successor);
    free(m->visited);
    free(m->walk);
    memset(m, 0, sizeof *m);
}

static void set_vtypes(bcd_model *m, char type)
{
    char *vtypes = (char *)bc_malloc((size_t)m->n_arc_vars);
    memset(vtypes, type, (size_t)m->n_arc_vars);
    bcd_grb_check(GRBsetcharattrarray(m->model, GRB_CHAR_ATTR_VTYPE, 0, m->n_arc_vars,
                                     vtypes),
                 m->model, "set VType");
    bcd_grb_check(GRBupdatemodel(m->model), m->model, "GRBupdatemodel");
    free(vtypes);
}

void bcd_model_relax_to_continuous(bcd_model *m) { set_vtypes(m, GRB_CONTINUOUS); }
void bcd_model_restore_integer_types(bcd_model *m) { set_vtypes(m, GRB_BINARY); }

/* ===================================================================== */
/* Cut rows                                                               */
/* ===================================================================== */

void bcd_row_init(const bcd_model *m, bcd_row *row)
{
    row->cols = (int *)bc_malloc((size_t)m->n_arc_vars * sizeof *row->cols);
    row->coeffs = (double *)bc_malloc((size_t)m->n_arc_vars * sizeof *row->coeffs);
    row->nnz = 0;
    row->bound = 0.0;
}

void bcd_row_free(bcd_row *row)
{
    free(row->cols);
    free(row->coeffs);
    memset(row, 0, sizeof *row);
}

void bcd_model_cut_row(const bcd_model *m, const int32_t *nodes, int64_t count,
                       double rhs, bcd_row *out)
{
    for (int64_t a = 0; a < count; ++a) m->inside[nodes[a]] = 1;
    out->nnz = 0;
    for (int k = 0; k < m->n_arc_vars; ++k) {
        if (m->inside[m->arc_sep_tail[k]] == m->inside[m->arc_sep_head[k]]) continue;
        out->cols[out->nnz] = k;
        out->coeffs[out->nnz] = 1.0;
        out->nnz += 1;
    }
    for (int64_t a = 0; a < count; ++a) m->inside[nodes[a]] = 0;
    out->bound = 2.0 * rhs;
}

double bcd_rcc_rhs(const bcd_model *m, const int32_t *nodes, int64_t count)
{
    __int128 total = 0;
    for (int64_t a = 0; a < count; ++a) total += m->inst->demand[nodes[a]];
    const __int128 capacity = m->inst->capacity;
    const int64_t required = (int64_t)(total / capacity + (total % capacity != 0));
    return (double)(required > 1 ? required : 1);
}

void bcd_check_subset(const bcd_model *m, const int32_t *nodes, int64_t count)
{
    const int64_t n = m->inst->n_nodes;
    if (count <= 0) bc_die("separator proposed an empty subset");
    for (int64_t a = 0; a < count; ++a) {
        if (nodes[a] == 0)
            bc_die("separator proposed a subset containing the depot; a cut set is a "
                   "set of customers, and dropping the depot from it would silently "
                   "change which inequality is added");
        if (nodes[a] < 0 || nodes[a] >= n)
            bc_die("separator proposed node %d, outside the customer range 1..%lld",
                   nodes[a], (long long)(n - 1));
        if (a > 0 && nodes[a] <= nodes[a - 1])
            bc_die("separator proposed a subset that is not strictly ascending "
                   "(%d after %d)", nodes[a], nodes[a - 1]);
    }
}

/* ===================================================================== */
/* Projection, support, violation                                         */
/* ===================================================================== */

const double *bcd_model_project(bcd_model *m, const double *values)
{
    for (int64_t p = 0; p < m->n_pairs; ++p)
        m->y[p] = values[m->pair_forward[p]] + values[m->pair_reverse[p]];
    return m->y;
}

void bcd_model_dense_support(bcd_model *m)
{
    const int64_t n = m->inst->n_nodes;
    memset(m->dense, 0, (size_t)(n * n) * sizeof *m->dense);
    for (int64_t p = 0; p < m->n_pairs; ++p) {
        const int64_t u = m->pair_u[p], v = m->pair_v[p];
        m->dense[u * n + v] = m->y[p];
        m->dense[v * n + u] = m->y[p];
    }
    for (int64_t u = 0; u < n; ++u) {
        double accumulator = 0.0;
        for (int64_t v = u + 1; v < n; ++v) accumulator += m->dense[u * n + v];
        for (int64_t v = 0; v < u; ++v) accumulator += m->dense[u * n + v];
        m->degree[u] = accumulator;
    }
}

double bcd_violation_from_dense(bcd_model *m, const int32_t *nodes, int64_t count,
                               double rhs)
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
/* Integer solution audit                                                 */
/* ===================================================================== */

void bcd_setlist_init(bcd_setlist *list)
{
    memset(list, 0, sizeof *list);
    list->indptr = (int64_t *)bc_malloc(sizeof *list->indptr);
    list->indptr[0] = 0;
}

void bcd_setlist_free(bcd_setlist *list)
{
    free(list->indptr);
    free(list->nodes);
    free(list->rhs);
    memset(list, 0, sizeof *list);
}

static void bcd_setlist_append(bcd_setlist *list, const int32_t *nodes, int64_t count, double rhs)
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

static int compare_int32(const void *lhs, const void *rhs)
{
    const int32_t a = *(const int32_t *)lhs, b = *(const int32_t *)rhs;
    return (a > b) - (a < b);
}


int64_t bcd_model_detect_violations(bcd_model *m, const double *values, bcd_setlist *out)
{
    const int64_t n = m->inst->n_nodes;
    out->count = 0;
    out->indptr[0] = 0;

    for (int64_t node = 0; node < m->n_model_nodes; ++node) m->successor[node] = -1;
    memset(m->visited, 0, (size_t)m->n_model_nodes * sizeof *m->visited);
    for (int k = 0; k < m->n_arc_vars; ++k) {
        if (values[k] <= BCD_INTEGRALITY_TOLERANCE) continue;
        if (m->arc_tail[k] == 0) continue;
        m->successor[m->arc_tail[k]] = m->arc_head[k];
    }

    for (int k = 0; k < m->n_arc_vars; ++k) {
        if (values[k] <= BCD_INTEGRALITY_TOLERANCE || m->arc_tail[k] != 0) continue;
        int64_t node = m->arc_head[k];
        while (node != m->terminal && node != 0) {
            if (m->visited[node]) break;
            m->visited[node] = 1;
            node = m->successor[node];
            if (node < 0) break;
        }
    }

    for (int64_t start = 1; start < n; ++start) {
        if (m->visited[start]) continue;
        int64_t size = 0, walk = start;
        while (walk > 0 && walk < n && !m->visited[walk]) {
            m->visited[walk] = 1;
            m->walk[size++] = (int32_t)walk;
            walk = m->successor[walk];
        }
        if (size == 0) continue;
        qsort(m->walk, (size_t)size, sizeof *m->walk, compare_int32);
        bcd_setlist_append(out, m->walk, size, bcd_rcc_rhs(m, m->walk, size));
    }

    for (int k = 0; k < m->n_arc_vars; ++k) {
        if (values[k] <= BCD_INTEGRALITY_TOLERANCE || m->arc_tail[k] != 0) continue;
        int64_t size = 0, node = m->arc_head[k];
        while (node != m->terminal && node != 0 && size < n - 1) {
            m->walk[size++] = (int32_t)node;
            node = m->successor[node];
            if (node < 0) break;
        }
        if (size == 0) continue;
        const double rhs = bcd_rcc_rhs(m, m->walk, size);
        if (rhs <= 1.0) continue;
        qsort(m->walk, (size_t)size, sizeof *m->walk, compare_int32);
        bcd_setlist_append(out, m->walk, size, rhs);
    }
    return out->count;
}
