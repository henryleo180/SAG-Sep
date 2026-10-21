#ifndef SAGSEP_FAMILIES_TTS_ADDDROP_H
#define SAGSEP_FAMILIES_TTS_ADDDROP_H

#include <stdbool.h>
#include <stdint.h>

#include "arena.h"
#include "families/tts/old_repair.h"
#include "families/tts/proposal.h"
#include "families/tts/score.h"
#include "sagsep.h"

typedef struct {
    int64_t bank_size;
    int64_t seed_count;
    int64_t mapped_count;
    int64_t remove_moves;
    int64_t add_moves;
    int64_t exchange_moves;
    int64_t candidate_count;
    bool    oversized_supernode;
    bool    has_best_violation;
    double  best_violation;
    int64_t compressed_supernodes;
    int64_t exact_candidate_count;
    int64_t seed_from_old_bank;
    int64_t raw_pool_count;
    int64_t dedup_count;
    bool    has_raw_best_violation;
    double  raw_best_violation;
} sag_tts_adddrop_diag;

void sag_tts_adddrop_diag_init(sag_tts_adddrop_diag *d);

int64_t sag_tts_adddrop_seeds_from_bank(sag_arena *out_arena, sag_arena *scratch,
                                        const float *bank_dense, int64_t bank_rows,
                                        int64_t n, const float *adj, const float *deg,
                                        const float *demand, double capacity,
                                        const sag_tts_score_ctx *sctx,
                                        float **out_seeds);

int64_t sag_tts_adddrop_bank_unique_count(sag_arena *scratch, const float *bank_dense,
                                          int64_t bank_rows, int64_t n);

bool sag_tts_adddrop_has_oversized(const sag_tts_compress_view *comp, double capacity);

int64_t sag_tts_native_adddrop_subsets(sag_arena *out_arena, sag_arena *scratch,
                                       const sag_tts_compress_view *comp, double capacity,
                                       const float *seeds, int64_t n_seeds,
                                       double min_violation,
                                       sag_tts_adddrop_diag *diag, float **out_super);

int64_t sag_tts_adddrop_proposals(sag_arena *out_arena, sag_arena *scratch,
                                  const float *seeds, int64_t n_seeds,
                                  int64_t n, const float *adj, const float *deg,
                                  const float *demand, double capacity,
                                  const sag_tts_score_ctx *sctx,
                                  const sag_tts_config *cfg,
                                  const sag_tts_keyset *seen_keys,
                                  const sag_tts_compress_view *comp,
                                  sag_tts_adddrop_diag *diag,
                                  sag_tts_proposal *out, int64_t cap_out);

#endif
