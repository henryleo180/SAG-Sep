#include "separator_cvrpsep.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cnstrmgr.h"
#include "cvrpsep_capi.h"
#include "util.h"

typedef struct {
    bc_separator base;
    const bc_instance *inst;
    int     max_cuts;
    double  min_violation;
    const char *tree_commit;
    double *demand;

    CnstrMgrPointer known;
    CnstrMgrPointer pending;
    int            *records;

    int64_t *indptr;
    int32_t *nodes;
    double  *rhs;
    int64_t  rows;
    int64_t  row_capacity;
    char    *member;

    int    *tail;
    int    *head;
    double *value;
    int64_t support_capacity;
} bc_cvrpsep;

static int cvrpsep_id(const bc_cvrpsep *sep, int64_t node)
{
    return (int)(node == 0 ? sep->inst->n_nodes : node);
}

static void reserve_rows(bc_cvrpsep *sep, int64_t rows)
{
    if (rows <= sep->row_capacity) return;
    while (sep->row_capacity < rows) sep->row_capacity = sep->row_capacity ? sep->row_capacity * 2 : 64;
    sep->indptr = (int64_t *)bc_realloc(sep->indptr,
                                        (size_t)(sep->row_capacity + 1) * sizeof *sep->indptr);
    sep->rhs = (double *)bc_realloc(sep->rhs, (size_t)sep->row_capacity * sizeof *sep->rhs);
    sep->records = (int *)bc_realloc(sep->records, (size_t)sep->row_capacity * sizeof *sep->records);
    sep->nodes = (int32_t *)bc_realloc(
        sep->nodes, (size_t)(sep->row_capacity * (sep->inst->n_nodes - 1)) * sizeof *sep->nodes);
}

static void discard_pending(bc_cvrpsep *sep)
{
    if (!sep->pending) return;
    cvrpsep_capi_free_cmgr(sep->pending);
    sep->pending = NULL;
}

static void cvrpsep_propose(bc_separator *base, int64_t m, const int64_t *u, const int64_t *v,
                            const double *x, bc_proposals *out)
{
    bc_cvrpsep *sep = (bc_cvrpsep *)base;
    discard_pending(sep);

    if (m > sep->support_capacity) {
        sep->support_capacity = m;
        sep->tail = (int *)bc_realloc(sep->tail, (size_t)(m + 1) * sizeof *sep->tail);
        sep->head = (int *)bc_realloc(sep->head, (size_t)(m + 1) * sizeof *sep->head);
        sep->value = (double *)bc_realloc(sep->value, (size_t)(m + 1) * sizeof *sep->value);
    }
    sep->tail[0] = 0;
    sep->head[0] = 0;
    sep->value[0] = 0.0;
    for (int64_t e = 0; e < m; ++e) {
        sep->tail[e + 1] = cvrpsep_id(sep, u[e]);
        sep->head[e + 1] = cvrpsep_id(sep, v[e]);
        sep->value[e + 1] = x[e];
    }

    CnstrMgrPointer batch = (CnstrMgrPointer)cvrpsep_capi_create_cmgr(sep->max_cuts);
    if (!batch) bc_die("CVRPSEP could not allocate its cut manager");
    char integer_and_feasible = 0;
    double max_violation = 0.0;
    cvrpsep_capi_separate_capacity_cuts(
        (int)(sep->inst->n_nodes - 1), sep->demand, (double)sep->inst->capacity, (int)m,
        sep->tail, sep->head, sep->value, sep->known, sep->max_cuts,
        1e-6, sep->min_violation,
        &integer_and_feasible, &max_violation, batch);

    const int64_t n = sep->inst->n_nodes;
    sep->rows = 0;
    reserve_rows(sep, batch->Size > 0 ? batch->Size : 1);
    sep->indptr[0] = 0;
    for (int index = 0; index < batch->Size; ++index) {
        const CnstrPointer record = batch->CPL[index];
        if (!record) continue;
        int64_t size = 0;
        for (int t = 1; t <= record->IntListSize; ++t) {
            const int token = record->IntList[t];
            if (token <= 0 || token >= n)
                bc_die("cvrpsep record %d contains node token %d, which is neither "
                       "a customer id in [1, %lld) nor anything this driver can "
                       "use (the depot is %lld)", index, token, (long long)n,
                       (long long)n);
            if (sep->member[token])
                bc_die("cvrpsep record %d lists customer %d twice", index, token);
            sep->member[token] = 1;
            size += 1;
        }
        if (size == 0)
            bc_die("cvrpsep record %d proposes an empty customer set", index);
        int64_t start = sep->indptr[sep->rows], written = start;
        for (int64_t node = 1; node < n; ++node) {
            if (!sep->member[node]) continue;
            sep->member[node] = 0;
            sep->nodes[written++] = (int32_t)node;
        }
        double rhs = (double)size - record->RHS;
        sep->rhs[sep->rows] = rhs > 1.0 ? rhs : 1.0;
        sep->records[sep->rows] = index;
        sep->rows += 1;
        sep->indptr[sep->rows] = written;
    }

    if (sep->rows > 0) {
        sep->pending = batch;
    } else {
        cvrpsep_capi_free_cmgr(batch);
    }
    out->rows = sep->rows;
    out->indptr = sep->indptr;
    out->nodes = sep->nodes;
    out->rhs = sep->rhs;
}

static void cvrpsep_commit(bc_separator *base, const int64_t *accepted, int64_t count,
                           bool retain)
{
    bc_cvrpsep *sep = (bc_cvrpsep *)base;
    if (!sep->pending) return;
    if (!retain) { discard_pending(sep); return; }
    for (int64_t i = 0; i < count; ++i) {
        const int64_t row = accepted[i];
        if (row < 0 || row >= sep->rows)
            bc_die("cvrpsep commit: row %lld is outside [0, %lld)",
                   (long long)row, (long long)sep->rows);
        const int record = sep->records[row];
        if (!sep->pending->CPL[record]) continue;
        cvrpsep_capi_move_cnstr(sep->pending, sep->known, record, 0);
    }
    discard_pending(sep);
}

static void cvrpsep_describe(bc_separator *base, char *buf, size_t cap)
{
    const bc_cvrpsep *sep = (const bc_cvrpsep *)base;
    snprintf(buf, cap,
             "cvrpsep(max_cuts=%d, min_violation=%g, known_cuts=%d, tree_commit=%s)",
             sep->max_cuts, sep->min_violation, sep->known ? sep->known->Size : 0,
             sep->tree_commit);
}

static void cvrpsep_destroy(bc_separator *base)
{
    bc_cvrpsep *sep = (bc_cvrpsep *)base;
    discard_pending(sep);
    if (sep->known) cvrpsep_capi_free_cmgr(sep->known);
    free(sep->demand);
    free(sep->indptr);
    free(sep->nodes);
    free(sep->rhs);
    free(sep->records);
    free(sep->member);
    free(sep->tail);
    free(sep->head);
    free(sep->value);
    free(sep);
}

bc_separator *bc_separator_cvrpsep_create(const bc_instance *inst, int max_cuts,
                                          double min_violation,
                                          const char *tree_commit)
{
    if (max_cuts <= 0) bc_die("cvrpsep: max_cuts must be positive, got %d", max_cuts);
    if (!(min_violation >= 0.0))
        bc_die("cvrpsep: min_violation must be non-negative, got %.17g", min_violation);
    bc_cvrpsep *sep = (bc_cvrpsep *)bc_calloc(1, sizeof *sep);
    sep->base.name = "cvrpsep";
    sep->base.propose = cvrpsep_propose;
    sep->base.commit = cvrpsep_commit;
    sep->base.describe = cvrpsep_describe;
    sep->base.destroy = cvrpsep_destroy;
    sep->inst = inst;
    sep->max_cuts = max_cuts;
    sep->min_violation = min_violation;
    sep->tree_commit = tree_commit ? tree_commit : "offered";

    sep->demand = (double *)bc_malloc((size_t)inst->n_nodes * sizeof *sep->demand);
    for (int64_t i = 0; i < inst->n_nodes; ++i) sep->demand[i] = (double)inst->demand[i];
    sep->member = (char *)bc_calloc((size_t)inst->n_nodes, sizeof *sep->member);

    sep->known = (CnstrMgrPointer)cvrpsep_capi_create_cmgr(max_cuts);
    if (!sep->known) bc_die("CVRPSEP could not allocate its constraint manager");

    reserve_rows(sep, 64);
    sep->indptr[0] = 0;
    return &sep->base;
}
