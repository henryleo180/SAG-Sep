#ifndef SAGSEP_CANDIDATE_CTX_H
#define SAGSEP_CANDIDATE_CTX_H

#include <stdint.h>

#include "sagsep.h"

typedef struct sag_bank sag_bank;

typedef struct {
    const float          *probs;
    const sag_lp_support *lp;
    const int64_t        *demand;
    int64_t               capacity;
    int64_t               n_nodes;
    int64_t               vehicle_count;
    int64_t               probs_rows;
    int64_t               iteration;
    int64_t               seed;
    int64_t               tts_seed;
    const char           *instance;
    const sag_bank       *bank;
} sag_candidate_ctx;

#endif
