#ifndef SAGSEP_FAMILIES_TTS_COMPRESS_H
#define SAGSEP_FAMILIES_TTS_COMPRESS_H

#include <stdbool.h>
#include <stdint.h>

#include "arena.h"
#include "sagsep.h"

typedef struct {
    int64_t        n_nodes;
    int64_t        n_super;
    const int64_t *labels;
    const bool    *frozen;
    const float   *member;
    const float   *aggregate;
    const float   *demand;
    const int64_t *sizes;
    const double  *internal_x;
    int64_t        one_edge_customer_supernodes;
} sag_tts_compression;

void sag_tts_build_compression(sag_arena *a, const float *adj_view, const float *demand_f32,
                               double capacity, const float *old_sets, int64_t n_old,
                               int64_t n_nodes, double eps, sag_tts_compression *out);

double sag_tts_f64_pairwise_sum(const double *a, int64_t n);

double sag_tts_f64_sum_numpy(const double *a, int64_t n);

#endif
