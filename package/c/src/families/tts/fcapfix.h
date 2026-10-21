#ifndef SAGSEP_FAMILIES_TTS_FCAPFIX_H
#define SAGSEP_FAMILIES_TTS_FCAPFIX_H

#include <stdint.h>

#include "arena.h"
#include "families/tts/compress.h"

typedef struct {
    uint64_t *masks;
    int64_t   rows;
    int64_t   words;
    int64_t   n_nodes;
} sag_tts_fcapfix_out;

void sag_tts_fcapfix(sag_arena *a,
                     int64_t                     n_nodes,
                     double                      capacity,
                     int64_t                     rounds,
                     int64_t                     seed_limit,
                     const float                *seed_scores,
                     const sag_tts_compression  *compression,
                     sag_tts_fcapfix_out        *out);

void sag_tts_super_scores(const sag_tts_compression *compression,
                          const float *node_scores, int64_t n_nodes, double *out);

#endif
