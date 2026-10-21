#ifndef BCD_DIRECTED_MODEL_H
#define BCD_DIRECTED_MODEL_H

#include <stdbool.h>
#include <stdint.h>

#include "gurobi_c.h"
#include "instance.h"

#define BCD_INTEGRALITY_TOLERANCE 0.5

typedef struct {
    GRBmodel *model;
    const bc_instance *inst;
    int64_t vehicles;
    bool    open_routes;
    int64_t n_model_nodes;
    int64_t terminal;
    int     n_arc_vars;

    int32_t *arc_tail;
    int32_t *arc_head;
    int32_t *arc_sep_tail;
    int32_t *arc_sep_head;

    int64_t  n_pairs;
    int32_t *pair_u;
    int32_t *pair_v;
    int32_t *pair_forward;
    int32_t *pair_reverse;

    double  *y;
    double  *dense;
    double  *degree;
    double  *gather;
    char    *inside;
    int64_t *successor;
    char    *visited;
    int32_t *walk;
} bcd_model;

typedef struct {
    int    *cols;
    double *coeffs;
    int     nnz;
    double  bound;
} bcd_row;

typedef struct {
    int64_t  count;
    int64_t *indptr;
    int32_t *nodes;
    double  *rhs;
    int64_t  capacity;
    int64_t  node_capacity;
} bcd_setlist;

void bcd_grb_check(int error, GRBmodel *model, const char *what);

void bcd_model_build(GRBenv *env, const bc_instance *inst, int64_t vehicles,
                     bool open_routes, bcd_model *out);
void bcd_model_free(bcd_model *m);

void bcd_model_relax_to_continuous(bcd_model *m);
void bcd_model_restore_integer_types(bcd_model *m);

void bcd_row_init(const bcd_model *m, bcd_row *row);
void bcd_row_free(bcd_row *row);

void bcd_model_cut_row(const bcd_model *m, const int32_t *nodes, int64_t count,
                       double rhs, bcd_row *out);

double bcd_rcc_rhs(const bcd_model *m, const int32_t *nodes, int64_t count);

void bcd_check_subset(const bcd_model *m, const int32_t *nodes, int64_t count);

const double *bcd_model_project(bcd_model *m, const double *values);

void bcd_model_dense_support(bcd_model *m);

double bcd_violation_from_dense(bcd_model *m, const int32_t *nodes, int64_t count,
                               double rhs);

void bcd_setlist_init(bcd_setlist *list);
void bcd_setlist_free(bcd_setlist *list);

int64_t bcd_model_detect_violations(bcd_model *m, const double *values, bcd_setlist *out);

#endif
