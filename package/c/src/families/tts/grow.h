#ifndef SAGSEP_FAMILIES_TTS_GROW_H
#define SAGSEP_FAMILIES_TTS_GROW_H

#include <stdint.h>

#include "arena.h"
#include "rng.h"

typedef struct {
    uint64_t *masks;
    int64_t   rows;
    int64_t   words;
    int64_t   n_cust;

    int64_t  *seed_id;
    int64_t  *seed_raw;
    int64_t  *seed_unique;
    int64_t  *seed_steps;
    int64_t  *seed_pair_le3;
    int64_t  *seed_pairs;
    int32_t  *row_seed;
    int64_t   n_seeds_used;
    int64_t   n_seed_early;
} sag_tts_grow_out;

void sag_tts_grow_adj_from_aggregate(const float *aggregate, int64_t n_super, double *out);

void sag_tts_lp_grow(sag_arena *a,
                     const double  *adj,
                     const double  *demand,
                     const int64_t *sizes,
                     const double  *internal_x,
                     int64_t        m,
                     double         capacity,
                     int64_t        n_seeds,
                     sag_pcg64     *rng,
                     double         min_violation,
                     int64_t        max_prefix,
                     sag_tts_grow_out *out);

#endif
