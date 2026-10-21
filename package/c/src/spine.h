#ifndef SAGSEP_SPINE_H
#define SAGSEP_SPINE_H

#include <stdbool.h>
#include <stdint.h>

#include "arena.h"
#include "candidate_ctx.h"
#include "sagsep.h"

/* ===================================================================== */
/* dedup result (= the reference implementation's DedupResult)            */
/* ===================================================================== */

typedef struct {
    sag_candidate_batch unique;
    int64_t            *origin_indptr;
    int32_t            *origin_raw;
    int64_t             n_origins;
    int64_t             raw_count;
} sag_dedup_result;

static inline int64_t sag_dedup_duplicates(const sag_dedup_result *d)
{
    return d->raw_count - d->unique.rows;
}

/* ===================================================================== */
/* One-shot output of run_spine                                           */
/* ===================================================================== */

typedef struct {
    sag_candidate_batch   raw;
    sag_dedup_result      dedup;
    sag_dedup_result      pool;
    sag_scored_batch      scored;
    sag_selection         sel;
    sag_materialized_cuts cuts;
    sag_pipeline_diag     diag;
    int8_t                family_order[SAG_FAMILY_COUNT];
    int64_t               family_count;
} sag_spine_result;

/* ===================================================================== */
/* The individual stages (each one callable on its own)                   */
/* ===================================================================== */

double sag_f64_pairwise_sum(const double *a, int64_t n);

double sag_f64_sum_numpy(const double *a, int64_t n);

void sag_assemble(sag_arena *a, const sag_candidate_batch *sweep,
                  const sag_candidate_batch *onion, const sag_candidate_batch *tts,
                  int64_t vehicle_count, sag_candidate_batch *raw);

void sag_dedup(sag_arena *a, const sag_candidate_batch *raw, sag_dedup_result *out);

void sag_filter_blocked(sag_arena *a, const sag_dedup_result *in,
                        const uint64_t *blocked, int64_t n_blocked, int64_t words,
                        sag_dedup_result *out, int64_t *blocked_count);

void sag_score(sag_arena *a, const sag_candidate_batch *pool,
               const sag_candidate_ctx *ctx, sag_scored_batch *out);

void sag_select(sag_arena *a, const sag_scored_batch *scored,
                const sag_selection_budget *budget, sag_selection *out);

void sag_materialize(sag_arena *a, const sag_candidate_batch *pool,
                     const sag_scored_batch *scored, const sag_selection *sel,
                     bool compute_cut_values, sag_materialized_cuts *out);

int64_t sag_spine_family_order(const sag_candidate_batch *raw, const bool *has_elapsed,
                               int8_t *out);

/* ===================================================================== */
/* Orchestration                                                          */
/* ===================================================================== */

void sag_run_spine(sag_arena *a, const sag_candidate_ctx *ctx,
                   const sag_candidate_batch *sweep, const sag_candidate_batch *onion,
                   const sag_candidate_batch *tts,
                   const uint64_t *blocked, int64_t n_blocked,
                   const sag_selection_budget *budget,
                   const double *family_elapsed,
                   bool compute_cut_values,
                   sag_spine_result *out);

#endif
