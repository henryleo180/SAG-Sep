#ifndef BC_BC_H
#define BC_BC_H

#include <stdbool.h>
#include <stdint.h>

#include "gurobi_c.h"
#include "instance.h"
#include "separator.h"

typedef struct {
    double time_limit;
    double mip_gap;
    int    threads;
    int    seed;
    int    root_rounds;
    int    cuts_per_round;
    double min_violation;
    const char *tree_commit;
    int    usercut_max_nodes;
    bool   padded_support;
    bool   connectivity_only;
    bool   verbose;
} bc_options;

typedef struct {
    long mipnode_calls;
    long mipnode_cuts_added;
    long mipsol_calls;
    long lazy_cuts_added;
    long lazy_connectivity_cuts;
} bc_callback_stats;

typedef struct {
    const char *instance_name;
    int64_t     n_customers;
    int64_t     capacity;
    int64_t     root_degree;
    const char *separator;
    const char *separator_detail;
    int         status;
    char        status_name[24];
    bool        has_objective;   double objective;
    bool        has_best_bound;  double best_bound;
    bool        has_gap;         double gap;
    double      runtime;
    double      setup_time;
    double      node_count;
    int         sol_count;
    bool        has_root_bound;  double root_bound;
    char        root_termination[32];
    bool        root_bound_final;
    int         root_rounds;
    int         root_cuts;
    int         generator_inner_rows;
    int         generator_cutset_rows;
    long        connectivity_cuts;
    double      root_time;
    bool        padded_support;
    bool        connectivity_only;
    int64_t     padding_high_degree_max;
    double      padding_max_clip;
    double      padding_max_identity_gap;
    bc_callback_stats callback;
    bool        has_optimal;     double optimal_value;
    bool        has_matches_optimum;
    bool        matches_optimum;
} bc_result;

#define BC_DETAIL_MAX 512

void bc_solve(GRBenv *env, const bc_instance *inst, bc_separator *separator,
              const bc_options *options, bool has_optimal, double optimal_value,
              double setup_start, char detail_buf[BC_DETAIL_MAX], bc_result *out);

#endif
