#ifndef BC_PADDING_H
#define BC_PADDING_H

#include <stdbool.h>
#include <stdint.h>

#define BC_PADDED_SUPPORT_THRESHOLD 1e-8

#define BC_PADDING_TARGET_DEGREE 2.0

typedef struct {
    int64_t  n_nodes;

    int64_t *u;
    int64_t *v;
    double  *x;
    int64_t  m;

    double  *pair_degree;
    int64_t  high_degree;
    double   max_clip;

    double  *gather;
    char    *in_subset;
} bc_padding;

void bc_padding_init(bc_padding *p, int64_t n_nodes, int64_t n_edge_vars);
void bc_padding_free(bc_padding *p);

void bc_padding_build(bc_padding *p, const double *dense, bool padded);

double bc_padding_cut_value(bc_padding *p, const int32_t *nodes, int64_t count);

double bc_padding_identity_gap(bc_padding *p, const int32_t *nodes, int64_t count,
                               int64_t rhs, double inner_violation);

#endif
