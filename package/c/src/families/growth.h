#ifndef SAGSEP_FAMILIES_GROWTH_H
#define SAGSEP_FAMILIES_GROWTH_H

#include <stdint.h>

#include "arena.h"
#include "candidate_ctx.h"
#include "sagsep.h"

typedef struct {
    int64_t rollouts;
    int64_t seeds;
    int64_t steps;
    int64_t frontier_empty;
    int64_t certified;
    int64_t rej_dup;
    int64_t rej_gate;
    double  margin_sum;
    int64_t emitted;
    int64_t seeds_emitting;
    int64_t emit_per_seed_max;
    int64_t dup_vs_sweep;
    int64_t dup_vs_seed;
    int64_t final_distinct;
    int64_t final_collisions;
    int64_t attempts;
    int64_t restarts;
    int64_t pool_exhausted;
    int64_t ptabu_swaps;
    int64_t ptabu_dead;
    double  seconds;
} sag_growth_stats;

int64_t sag_growth_row_reserve(const sag_growth_config *g, int64_t n_customers);

void sag_generate_growth(sag_arena *a, const sag_candidate_ctx *ctx,
                         const sag_sweep_config *sweep, const sag_growth_config *g,
                         sag_candidate_batch *out,
                         uint64_t *taken, int64_t *n_taken, int64_t cap_taken,
                         int64_t words, sag_growth_stats *stats);

#endif
