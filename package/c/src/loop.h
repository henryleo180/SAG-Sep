#ifndef SAGSEP_LOOP_H
#define SAGSEP_LOOP_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "arena.h"
#include "families/tts/core.h"
#include "forward.h"
#include "json_out.h"
#include "spine.h"
#include "sagsep.h"

typedef void (*sag_loop_support_probe)(void *user, int64_t iteration,
                                       const int64_t *sup_u, const int64_t *sup_v,
                                       const double *sup_x, int64_t m);

typedef void (*sag_loop_bank_probe)(void *user, int64_t iteration, const sag_bank *snap);

typedef void (*sag_loop_spine_probe)(void *user, int64_t iteration,
                                     const sag_candidate_batch *sweep,
                                     const sag_candidate_batch *onion,
                                     const sag_candidate_batch *tts,
                                     const uint64_t *blocked, int64_t n_blocked,
                                     const sag_spine_result *spine);

typedef struct {
    sag_forward *forward;
    sag_loop_support_probe probe;
    sag_loop_bank_probe    probe_bank;
    sag_loop_spine_probe   probe_spine;
    void                  *probe_user;
    int64_t      max_iter;
    bool         emit_diagnostics;
    FILE        *progress;
    size_t       iter_arena_bytes;
    size_t       long_arena_bytes;
} sag_loop_options;

typedef struct {
    sag_json_root_result json;
    sag_json_iteration  *iterations;
    int64_t              n_iterations;

    double  lower_bound;
    double  runtime;
    int64_t total_cuts;
    int64_t iteration_count;
    bool    time_limit_hit;
} sag_root_result;

void sag_loop_options_init(sag_loop_options *opt);

void sag_run_instance(const sag_instance *inst, const sag_config *cfg,
                      const sag_loop_options *opt, sag_arena *result_arena,
                      sag_root_result *out);

double sag_loop_violation_from_dense(const double *dense, const double *degree, int64_t n_nodes,
                                     const int32_t *nodes_ext, int64_t size, double rhs,
                                     double *block, double *degbuf);

#endif
