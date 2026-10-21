#include "formulation_tw.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

void tw_grb_check(int error, GRBmodel *model, const char *what)
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

#define TW_PAIRWISE_BLOCKSIZE 128
#define TW_REDUCE_BUFSIZE     8192

static double pairwise_sum(const double *a, int64_t n)
{
    if (n < 8) {
        double result = 0.0;
        for (int64_t i = 0; i < n; ++i) result += a[i];
        return result;
    }
    if (n <= TW_PAIRWISE_BLOCKSIZE) {
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
    for (int64_t offset = 0; offset < n; offset += TW_REDUCE_BUFSIZE) {
        const int64_t chunk = (n - offset < TW_REDUCE_BUFSIZE) ? n - offset : TW_REDUCE_BUFSIZE;
        result += pairwise_sum(a + offset, chunk);
    }
    return result;
}

/* ===================================================================== */
/* Model construction                                                     */
/* ===================================================================== */

static void build_aggregation(tw_model *m)
{
    const int64_t n_agg = m->n_agg_nodes;
    const int64_t n_model = m->arcs->n_model_nodes;
    const int64_t sink = n_model - 1;
    const int64_t bound = n_agg * (n_agg - 1) / 2;

    m->y_u = (int64_t *)bc_malloc((size_t)bound * sizeof *m->y_u);
    m->y_v = (int64_t *)bc_malloc((size_t)bound * sizeof *m->y_v);
    m->y_arc_a = (int32_t *)bc_malloc((size_t)bound * sizeof *m->y_arc_a);
    m->y_arc_b = (int32_t *)bc_malloc((size_t)bound * sizeof *m->y_arc_b);
    m->y_index = (int32_t *)bc_malloc((size_t)(n_agg * n_agg) * sizeof *m->y_index);
    for (int64_t i = 0; i < n_agg * n_agg; ++i) m->y_index[i] = -1;

    int64_t count = 0;
    for (int64_t u = 0; u < n_agg; ++u) {
        for (int64_t v = u + 1; v < n_agg; ++v) {
            int32_t first, second;
            if (u == 0) {
                first = m->arc_index[v];
                second = m->arc_index[v * n_model + sink];
            } else {
                first = m->arc_index[u * n_model + v];
                second = m->arc_index[v * n_model + u];
            }
            if (first < 0 && second < 0) continue;
            m->y_index[u * n_agg + v] = (int32_t)count;
            m->y_index[v * n_agg + u] = (int32_t)count;
            m->y_u[count] = u;
            m->y_v[count] = v;
            m->y_arc_a[count] = first;
            m->y_arc_b[count] = second;
            count += 1;
        }
    }
    m->n_y = count;
}

void tw_model_build(GRBenv *env, const tw_instance *inst, const tw_arcs *arcs,
                    int64_t vehicles, tw_model *out)
{
    if (vehicles <= 0) bc_die("vehicles must be positive, got %lld", (long long)vehicles);
    tw_check_customer_degrees(arcs, inst->n_customers);

    const int64_t n = inst->n_customers;
    const int64_t sink = n + 1;
    const int64_t n_model = arcs->n_model_nodes;
    const int64_t n_arcs = arcs->n_arcs;
    if (n_arcs + n_model > INT32_MAX)
        bc_die("instance %s needs %lld columns, more than one LP can address",
               inst->name, (long long)(n_arcs + n_model));

    memset(out, 0, sizeof *out);
    out->inst = inst;
    out->arcs = arcs;
    out->vehicles = vehicles;
    out->n_arc_vars = (int)n_arcs;
    out->n_time_vars = (int)n_model;
    out->n_agg_nodes = n + 1;

    /* --- columns: the arcs, then one time variable per model node --- */
    double *lb = (double *)bc_malloc((size_t)(n_arcs + n_model) * sizeof *lb);
    double *ub = (double *)bc_malloc((size_t)(n_arcs + n_model) * sizeof *ub);
    double *obj = (double *)bc_calloc((size_t)(n_arcs + n_model), sizeof *obj);
    char *vtypes = (char *)bc_malloc((size_t)(n_arcs + n_model) * sizeof *vtypes);
    for (int64_t k = 0; k < n_arcs; ++k) {
        lb[k] = 0.0;
        ub[k] = 1.0;
        obj[k] = arcs->cost[k];
        vtypes[k] = GRB_BINARY;
    }
    for (int64_t node = 0; node < n_model; ++node) {
        lb[n_arcs + node] = arcs->lower[node];
        ub[n_arcs + node] = arcs->upper[node];
        vtypes[n_arcs + node] = GRB_CONTINUOUS;
    }
    tw_grb_check(GRBnewmodel(env, &out->model, "vrptw_two_index", 0, NULL, NULL, NULL,
                             NULL, NULL),
                 NULL, "GRBnewmodel");
    tw_grb_check(GRBaddvars(out->model, (int)(n_arcs + n_model), 0, NULL, NULL, NULL,
                            obj, lb, ub, vtypes, NULL),
                 out->model, "GRBaddvars");
    tw_grb_check(GRBsetintattr(out->model, GRB_INT_ATTR_MODELSENSE, GRB_MINIMIZE),
                 out->model, "set ModelSense");
    tw_grb_check(GRBupdatemodel(out->model), out->model, "GRBupdatemodel");
    free(lb);
    free(ub);
    free(obj);
    free(vtypes);

    out->arc_index = (int32_t *)bc_malloc((size_t)(n_model * n_model) * sizeof *out->arc_index);
    for (int64_t i = 0; i < n_model * n_model; ++i) out->arc_index[i] = -1;
    for (int64_t k = 0; k < n_arcs; ++k)
        out->arc_index[(int64_t)arcs->tail[k] * n_model + arcs->head[k]] = (int32_t)k;

    /* --- 1/2. customer in- and out-degree, one each, ascending customer --- */
    int *indices = (int *)bc_malloc((size_t)(n_arcs > n_model ? n_arcs : n_model) * sizeof *indices);
    double *ones = (double *)bc_malloc((size_t)n_arcs * sizeof *ones);
    for (int64_t k = 0; k < n_arcs; ++k) ones[k] = 1.0;
    char name[64];
    for (int64_t customer = 1; customer <= n; ++customer) {
        for (int side = 0; side < 2; ++side) {
            int count = 0;
            for (int64_t k = 0; k < n_arcs; ++k) {
                const int64_t endpoint = side == 0 ? arcs->tail[k] : arcs->head[k];
                if (endpoint == customer) indices[count++] = (int)k;
            }
            snprintf(name, sizeof name, "deg_%s_%lld", side == 0 ? "out" : "in",
                     (long long)customer);
            tw_grb_check(GRBaddconstr(out->model, count, indices, ones, GRB_EQUAL, 1.0, name),
                         out->model, "GRBaddconstr (customer degree)");
        }
    }
    /* --- 3/4. the fixed fleet, as two equalities --- */
    for (int side = 0; side < 2; ++side) {
        int count = 0;
        for (int64_t k = 0; k < n_arcs; ++k) {
            const bool hit = side == 0 ? (arcs->tail[k] == 0) : (arcs->head[k] == sink);
            if (hit) indices[count++] = (int)k;
        }
        tw_grb_check(GRBaddconstr(out->model, count, indices, ones, GRB_EQUAL,
                                  (double)vehicles,
                                  side == 0 ? "fleet_source" : "fleet_sink"),
                     out->model, "GRBaddconstr (fleet)");
    }
    free(ones);

    {
        int pair_cols[2];
        double pair_vals[2] = { 1.0, 1.0 };
        for (int64_t i = 1; i <= n; ++i) {
            for (int64_t j = i + 1; j <= n; ++j) {
                const int32_t forward = out->arc_index[i * n_model + j];
                const int32_t backward = out->arc_index[j * n_model + i];
                if (forward < 0 || backward < 0) continue;
                pair_cols[0] = forward;
                pair_cols[1] = backward;
                tw_grb_check(GRBaddconstr(out->model, 2, pair_cols, pair_vals,
                                          GRB_LESS_EQUAL, 1.0, NULL),
                             out->model, "GRBaddconstr (antiparallel pair)");
            }
        }
    }

    {
        int cols[3];
        double vals[3];
        for (int64_t k = 0; k < n_arcs; ++k) {
            const int64_t tail = arcs->tail[k], head = arcs->head[k];
            const double big_m = arcs->big_m[k];
            const double bound = big_m - arcs->service[tail] - arcs->time[k];
            int nnz = 0;
            if (big_m > 0.0) {
                cols[nnz] = (int)k;
                vals[nnz] = big_m;
                nnz += 1;
            }
            cols[nnz] = (int)(n_arcs + tail);
            vals[nnz] = 1.0;
            nnz += 1;
            cols[nnz] = (int)(n_arcs + head);
            vals[nnz] = -1.0;
            nnz += 1;
            tw_grb_check(GRBaddconstr(out->model, nnz, cols, vals, GRB_LESS_EQUAL,
                                      bound, NULL),
                         out->model, "GRBaddconstr (time propagation)");
        }
    }
    tw_grb_check(GRBupdatemodel(out->model), out->model, "GRBupdatemodel");
    free(indices);

    build_aggregation(out);

    const int64_t n_agg = out->n_agg_nodes;
    out->y_values = (double *)bc_malloc((size_t)(out->n_y > 0 ? out->n_y : 1)
                                        * sizeof *out->y_values);
    out->dense = (double *)bc_malloc((size_t)(n_agg * n_agg) * sizeof *out->dense);
    out->degree = (double *)bc_malloc((size_t)n_agg * sizeof *out->degree);
    out->gather = (double *)bc_malloc((size_t)(n * n + n) * sizeof *out->gather);
    out->in_subset = (char *)bc_calloc((size_t)n_agg, sizeof *out->in_subset);
    out->complement = (int32_t *)bc_malloc((size_t)n_agg * sizeof *out->complement);
}

void tw_model_free(tw_model *m)
{
    if (m->model) GRBfreemodel(m->model);
    free(m->arc_index);
    free(m->y_u);
    free(m->y_v);
    free(m->y_arc_a);
    free(m->y_arc_b);
    free(m->y_index);
    free(m->y_values);
    free(m->dense);
    free(m->degree);
    free(m->gather);
    free(m->in_subset);
    free(m->complement);
    memset(m, 0, sizeof *m);
}

static void set_arc_vtypes(tw_model *m, char type)
{
    char *vtypes = (char *)bc_malloc((size_t)m->n_arc_vars);
    memset(vtypes, type, (size_t)m->n_arc_vars);
    tw_grb_check(GRBsetcharattrarray(m->model, GRB_CHAR_ATTR_VTYPE, 0, m->n_arc_vars, vtypes),
                 m->model, "set VType");
    tw_grb_check(GRBupdatemodel(m->model), m->model, "GRBupdatemodel");
    free(vtypes);
}

void tw_model_relax_to_continuous(tw_model *m) { set_arc_vtypes(m, GRB_CONTINUOUS); }
void tw_model_restore_integer_types(tw_model *m) { set_arc_vtypes(m, GRB_BINARY); }

/* ===================================================================== */
/* Cut rows                                                               */
/* ===================================================================== */

void tw_row_init(const tw_model *m, tw_row *row)
{
    const int64_t n = m->inst->n_customers;
    const int64_t capacity = n * (n - 1) + 2 * n + 2;
    row->cols = (int *)bc_malloc((size_t)capacity * sizeof *row->cols);
    row->coeffs = (double *)bc_malloc((size_t)capacity * sizeof *row->coeffs);
    row->nnz = 0;
    row->bound = 0.0;
}

void tw_row_free(tw_row *row)
{
    free(row->cols);
    free(row->coeffs);
    memset(row, 0, sizeof *row);
}

static void append_edge(const tw_model *m, int32_t edge, double coefficient, tw_row *row)
{
    const int32_t arcs[2] = { m->y_arc_a[edge], m->y_arc_b[edge] };
    for (int side = 0; side < 2; ++side) {
        if (arcs[side] < 0) continue;
        row->cols[row->nnz] = arcs[side];
        row->coeffs[row->nnz] = coefficient;
        row->nnz += 1;
    }
}

static void append_inner_edges(const tw_model *m, const int32_t *subset, int64_t count,
                               double coefficient, tw_row *row)
{
    if (count < 2) return;
    const int64_t n_agg = m->n_agg_nodes;
    for (int64_t a = 0; a < count; ++a) {
        for (int64_t b = a + 1; b < count; ++b) {
            const int32_t edge = m->y_index[(int64_t)subset[a] * n_agg + subset[b]];
            if (edge < 0) continue;
            append_edge(m, edge, coefficient, row);
        }
    }
}

static void append_depot_edges(const tw_model *m, const int32_t *subset, int64_t count,
                               double coefficient, tw_row *row)
{
    for (int64_t a = 0; a < count; ++a) {
        const int32_t edge = m->y_index[subset[a]];
        if (edge < 0) continue;
        append_edge(m, edge, coefficient, row);
    }
}

void tw_build_cut_row(const tw_model *m, const int32_t *customers, int64_t count,
                      double rhs, tw_row *out)
{
    const int64_t n_customers = m->inst->n_customers;
    out->nnz = 0;

    if ((double)count > (double)n_customers / 2.0) {
        for (int64_t a = 0; a < count; ++a) m->in_subset[customers[a]] = 1;
        int64_t complement_size = 0;
        for (int64_t node = 1; node <= n_customers; ++node) {
            if (!m->in_subset[node]) m->complement[complement_size++] = (int32_t)node;
        }
        for (int64_t a = 0; a < count; ++a) m->in_subset[customers[a]] = 0;

        append_inner_edges(m, m->complement, complement_size, 1.0, out);
        append_depot_edges(m, m->complement, complement_size, 0.5, out);
        append_depot_edges(m, customers, count, -0.5, out);
        out->bound = (double)complement_size - rhs;
        return;
    }

    append_inner_edges(m, customers, count, 1.0, out);
    out->bound = (double)count - rhs;
}

double tw_rcc_rhs(const tw_model *m, const int32_t *customers, int64_t count)
{
    __int128 total = 0;
    for (int64_t a = 0; a < count; ++a) total += m->inst->demand[customers[a]];
    const __int128 capacity = m->inst->capacity;
    const int64_t required = (int64_t)(total / capacity + (total % capacity != 0));
    return (double)(required > 1 ? required : 1);
}

void tw_validate_subset(const tw_model *m, const int32_t *customers, int64_t count)
{
    const int64_t n_customers = m->inst->n_customers;
    if (count <= 0) bc_die("separator returned an empty cut set");
    for (int64_t a = 0; a < count; ++a) {
        const int32_t node = customers[a];
        if (node == 0)
            bc_die("separator returned a cut set containing the depot; this driver's "
                   "contract is customer sets only");
        if (node < 1 || node > n_customers)
            bc_die("separator returned node id %d, outside 1..%lld", node,
                   (long long)n_customers);
        if (a > 0 && node <= customers[a - 1])
            bc_die("separator returned node ids that are not strictly ascending "
                   "(%d after %d)", node, customers[a - 1]);
    }
}

/* ===================================================================== */
/* Aggregation, support, violation                                        */
/* ===================================================================== */

void tw_aggregate(tw_model *m, const double *values)
{
    for (int64_t e = 0; e < m->n_y; ++e) {
        const int32_t a = m->y_arc_a[e], b = m->y_arc_b[e];
        m->y_values[e] = (a >= 0 ? values[a] : 0.0) + (b >= 0 ? values[b] : 0.0);
    }
}

void tw_dense_support(tw_model *m)
{
    const int64_t n_agg = m->n_agg_nodes;
    memset(m->dense, 0, (size_t)(n_agg * n_agg) * sizeof *m->dense);
    for (int64_t e = 0; e < m->n_y; ++e) {
        const int64_t u = m->y_u[e], v = m->y_v[e];
        m->dense[u * n_agg + v] = m->y_values[e];
        m->dense[v * n_agg + u] = m->y_values[e];
    }
    for (int64_t u = 0; u < n_agg; ++u) {
        double accumulator = 0.0;
        for (int64_t v = u + 1; v < n_agg; ++v) accumulator += m->dense[u * n_agg + v];
        for (int64_t v = 0; v < u; ++v) accumulator += m->dense[u * n_agg + v];
        m->degree[u] = accumulator;
    }
}

double tw_violation_from_dense(tw_model *m, const int32_t *customers, int64_t count,
                               double rhs)
{
    if (count == 0) return 0.0;
    const int64_t n_agg = m->n_agg_nodes;
    double *block = m->gather;
    for (int64_t a = 0; a < count; ++a) {
        const int64_t row = (int64_t)customers[a] * n_agg;
        for (int64_t b = 0; b < count; ++b) block[a * count + b] = m->dense[row + customers[b]];
    }
    const double inside = reduce_sum(block, count * count);
    double *degrees = m->gather + count * count;
    for (int64_t a = 0; a < count; ++a) degrees[a] = m->degree[customers[a]];
    const double boundary = reduce_sum(degrees, count) - inside;
    return 2.0 * rhs - boundary;
}

/* ===================================================================== */
/* Integer solution decomposition                                         */
/* ===================================================================== */

void tw_setlist_init(tw_setlist *list)
{
    memset(list, 0, sizeof *list);
    list->indptr = (int64_t *)bc_malloc(sizeof *list->indptr);
    list->indptr[0] = 0;
}

void tw_setlist_free(tw_setlist *list)
{
    free(list->indptr);
    free(list->nodes);
    free(list->rhs);
    memset(list, 0, sizeof *list);
}

static void setlist_clear(tw_setlist *list)
{
    list->count = 0;
    list->indptr[0] = 0;
}

static void setlist_append(tw_setlist *list, const int32_t *nodes, int64_t count, double rhs)
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

int64_t tw_detect_violations(tw_model *m, const double *values, tw_setlist *out)
{
    const int64_t n = m->inst->n_customers;
    const int64_t sink = n + 1;
    const tw_arcs *arcs = m->arcs;
    setlist_clear(out);

    int32_t *successor = (int32_t *)bc_malloc((size_t)(n + 2) * sizeof *successor);
    for (int64_t node = 0; node < n + 2; ++node) successor[node] = -1;
    int32_t *starts = (int32_t *)bc_malloc((size_t)(n + 1) * sizeof *starts);
    int64_t n_starts = 0;
    for (int64_t k = 0; k < arcs->n_arcs; ++k) {
        if (values[k] <= TW_INTEGRALITY_TOLERANCE) continue;
        if (arcs->tail[k] == 0) {
            if (n_starts < n + 1) starts[n_starts++] = arcs->head[k];
        } else {
            successor[arcs->tail[k]] = arcs->head[k];
        }
    }

    char *visited = (char *)bc_calloc((size_t)(n + 2), sizeof *visited);
    int32_t *route = (int32_t *)bc_malloc((size_t)(n + 1) * sizeof *route);
    for (int64_t s = 0; s < n_starts; ++s) {
        const int32_t first = starts[s];
        if (visited[first]) continue;
        int64_t size = 0;
        int64_t node = first;
        while (node != sink && node > 0) {
            if (visited[node]) break;
            visited[node] = 1;
            route[size++] = (int32_t)node;
            node = successor[node];
            if (node < 0) break;
        }
        if (size == 0) continue;
        const double rhs = tw_rcc_rhs(m, route, size);
        if (rhs <= 1.0) continue;
        qsort(route, (size_t)size, sizeof *route, compare_int32);
        setlist_append(out, route, size, rhs);
    }

    int64_t unreached = 0;
    for (int64_t customer = 1; customer <= n; ++customer) unreached += !visited[customer];
    if (unreached > 0) {
        char *seen = (char *)bc_calloc((size_t)(n + 2), sizeof *seen);
        int32_t *stack = (int32_t *)bc_malloc((size_t)(2 * n + 2) * sizeof *stack);
        for (int64_t customer = 1; customer <= n; ++customer) {
            if (visited[customer] || seen[customer]) continue;
            int64_t size = 0, top = 0;
            stack[top++] = (int32_t)customer;
            while (top > 0) {
                const int32_t node = stack[--top];
                if (seen[node]) continue;
                seen[node] = 1;
                route[size++] = node;
                for (int64_t k = 0; k < arcs->n_arcs; ++k) {
                    if (values[k] <= TW_INTEGRALITY_TOLERANCE) continue;
                    const int32_t tail = arcs->tail[k], head = arcs->head[k];
                    int32_t other = -1;
                    if (tail == node) other = head;
                    else if (head == node) other = tail;
                    if (other <= 0 || other > n) continue;
                    if (visited[other] || seen[other]) continue;
                    stack[top++] = other;
                }
            }
            qsort(route, (size_t)size, sizeof *route, compare_int32);
            setlist_append(out, route, size, tw_rcc_rhs(m, route, size));
        }
        free(stack);
        free(seen);
    }

    free(route);
    free(visited);
    free(starts);
    free(successor);
    return out->count;
}
