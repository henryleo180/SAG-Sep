#ifndef SAGSEP_FAMILIES_TTS_OLD_REPAIR_H
#define SAGSEP_FAMILIES_TTS_OLD_REPAIR_H

#include <stdbool.h>
#include <stdint.h>

#include "arena.h"
#include "families/tts/proposal.h"
#include "families/tts/score.h"
#include "sagsep.h"

typedef struct {
    int64_t      n_super;
    int64_t      n_nodes;
    const float *member;
    const float *aggregate;
    const float *demand;
} sag_tts_compress_view;

typedef struct {
    int64_t bank_size;
    int64_t eligible_count;
    int64_t top_b_count;
    int64_t still_violated_count;
    int64_t repair_attempt_count;
    int64_t repair_success_count;
    int64_t candidate_count;
    int64_t compressed_supernodes;
    int64_t raw_pool_count;
    int64_t dedup_count;
    int64_t ls_max_steps;
    bool    ls_hit_max_flips;
    bool    has_raw_best_violation;
    bool    has_ls_best_violation;
    bool    has_ls_best_gain;
    double  raw_best_violation;
    double  ls_best_violation;
    double  ls_best_gain;
} sag_tts_old_repair_diag;

void sag_tts_old_repair_diag_init(sag_tts_old_repair_diag *d);

int64_t sag_tts_old_repair_proposals(sag_arena *out_arena, sag_arena *scratch,
                                     const float *bank_dense, int64_t bank_rows,
                                     int64_t n, const float *adj, const float *deg,
                                     const float *demand, double capacity,
                                     const sag_tts_score_ctx *sctx,
                                     const sag_tts_config *cfg,
                                     const sag_tts_compress_view *comp,
                                     sag_tts_old_repair_diag *diag,
                                     sag_tts_proposal *out, int64_t cap_out);

#endif
