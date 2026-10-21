#ifndef SAGSEP_LP_H
#define SAGSEP_LP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "arena.h"
#include "sagsep.h"

typedef struct sag_lp sag_lp;

void sag_lp_version_string(char *out, size_t cap);

sag_lp *sag_lp_create(const sag_instance *inst, const sag_lp_config *cfg);
void    sag_lp_destroy(sag_lp *lp);

void    sag_lp_optimize(sag_lp *lp);
double  sag_lp_objective(const sag_lp *lp);
int     sag_lp_status(const sag_lp *lp);

int64_t sag_lp_num_constrs(const sag_lp *lp);
int64_t sag_lp_base_rows(const sag_lp *lp);
int64_t sag_lp_num_vars(const sag_lp *lp);
const int64_t *sag_lp_edge_u(const sag_lp *lp);
const int64_t *sag_lp_edge_v(const sag_lp *lp);

const double *sag_lp_solution(sag_lp *lp);

int64_t sag_lp_support_edges(sag_lp *lp, double threshold,
                             int64_t *eu, int64_t *ev, double *ex);

int64_t sag_lp_add_cuts(sag_lp *lp, const sag_materialized_cuts *cuts);

#endif
