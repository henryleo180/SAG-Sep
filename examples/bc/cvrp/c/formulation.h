#ifndef BC_FORMULATION_H
#define BC_FORMULATION_H

#include <stdbool.h>
#include <stdint.h>

#include "gurobi_c.h"
#include "vrp_io.h"

#define BC_INTEGRALITY_TOLERANCE 0.5

typedef struct {
    GRBmodel *model;
    const bc_instance *inst;
    int64_t vehicles;
    int32_t *edge_i;
    int32_t *edge_j;
    int32_t *var_index;
    int      n_edge_vars;

    double  *dense;
    double  *degree;
    double  *gather;
    char    *in_subset;
    int32_t *complement;
} bc_model;

typedef struct {
    int    *cols;
    double *coeffs;
    int     nnz;
    double  bound;
} bc_row;

void bc_grb_check(int error, GRBmodel *model, const char *what);

void bc_model_build(GRBenv *env, const bc_instance *inst, int64_t vehicles, bc_model *out);
void bc_model_free(bc_model *m);

void bc_model_relax_to_continuous(bc_model *m);
void bc_model_restore_integer_types(bc_model *m);

void bc_row_init(const bc_model *m, bc_row *row);
void bc_row_free(bc_row *row);

void bc_build_cut_row(const bc_model *m, const int32_t *nodes, int64_t count, double rhs,
                      bc_row *out);

double bc_rcc_rhs(const bc_model *m, const int32_t *nodes, int64_t count);

void bc_dense_support(bc_model *m, const double *values);

double bc_violation_from_dense(bc_model *m, const int32_t *nodes, int64_t count, double rhs);

typedef struct {
    int64_t  count;
    int64_t *indptr;
    int32_t *nodes;
    double  *rhs;
    int64_t  capacity;
    int64_t  node_capacity;
} bc_setlist;

void bc_setlist_init(bc_setlist *list);
void bc_setlist_free(bc_setlist *list);

int64_t bc_detect_violations(bc_model *m, const double *values, bc_setlist *out);

#endif
