#ifndef TW_FORMULATION_H
#define TW_FORMULATION_H

#include <stdbool.h>
#include <stdint.h>

#include "gurobi_c.h"
#include "solomon.h"

#define TW_INTEGRALITY_TOLERANCE 0.5
#define TW_SUPPORT_THRESHOLD 1e-8

typedef struct {
    GRBmodel *model;
    const tw_instance *inst;
    const tw_arcs *arcs;
    int64_t vehicles;
    int      n_arc_vars;
    int      n_time_vars;
    int64_t  n_agg_nodes;

    int32_t *arc_index;

    int64_t  n_y;
    int64_t *y_u;
    int64_t *y_v;
    int32_t *y_arc_a;
    int32_t *y_arc_b;
    int32_t *y_index;

    double  *y_values;
    double  *dense;
    double  *degree;
    double  *gather;
    char    *in_subset;
    int32_t *complement;
} tw_model;

typedef struct {
    int    *cols;
    double *coeffs;
    int     nnz;
    double  bound;
} tw_row;

void tw_grb_check(int error, GRBmodel *model, const char *what);

void tw_model_build(GRBenv *env, const tw_instance *inst, const tw_arcs *arcs,
                    int64_t vehicles, tw_model *out);
void tw_model_free(tw_model *m);

void tw_model_relax_to_continuous(tw_model *m);
void tw_model_restore_integer_types(tw_model *m);

void tw_row_init(const tw_model *m, tw_row *row);
void tw_row_free(tw_row *row);

void tw_build_cut_row(const tw_model *m, const int32_t *customers, int64_t count,
                      double rhs, tw_row *out);

double tw_rcc_rhs(const tw_model *m, const int32_t *customers, int64_t count);

void tw_validate_subset(const tw_model *m, const int32_t *customers, int64_t count);

void tw_aggregate(tw_model *m, const double *values);

void tw_dense_support(tw_model *m);

double tw_violation_from_dense(tw_model *m, const int32_t *customers, int64_t count,
                               double rhs);

typedef struct {
    int64_t  count;
    int64_t *indptr;
    int32_t *nodes;
    double  *rhs;
    int64_t  capacity;
    int64_t  node_capacity;
} tw_setlist;

void tw_setlist_init(tw_setlist *list);
void tw_setlist_free(tw_setlist *list);

int64_t tw_detect_violations(tw_model *m, const double *values, tw_setlist *out);

#endif
