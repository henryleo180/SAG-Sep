#ifndef SAGSEP_FAMILIES_TTS_FLOW_H
#define SAGSEP_FAMILIES_TTS_FLOW_H

#include <stdbool.h>
#include <stdint.h>

#include "arena.h"
#include "families/tts/proposal.h"
#include "sagsep.h"

#define SAG_TTS_BLOCKER_UNKNOWN SAG_VARIANT_COUNT
#define SAG_TTS_PAIR_SLOTS      (SAG_VARIANT_COUNT + 1)

typedef struct {
    int64_t near_duplicate_hit_count;
    int64_t pair_counts[SAG_TTS_PAIR_SLOTS][SAG_TTS_PAIR_SLOTS];

    int64_t pair_stage[2][2];
    int64_t pair_same_seed[2][2];
    int64_t sel_by_source[SAG_TTS_PAIR_SLOTS];
    int64_t sel_by_stage[2];
    int64_t input_by_source[SAG_TTS_PAIR_SLOTS];
    int64_t input_by_stage[2];
} sag_tts_flow_diag;

typedef struct {
    int64_t         vehicle_count;
    int64_t         n_nodes;
    int64_t         words;
    int64_t         count;
    int64_t         emitted;
    uint64_t       *masks;
    int8_t         *sources;
    int8_t         *stages;
    int32_t        *seeds;
    int64_t         cap;
    sag_tts_keyset  seen;
} sag_tts_flow_state;

void sag_tts_flow_state_init(sag_arena *a, sag_tts_flow_state *fs, int64_t vehicle_count,
                             int64_t n_nodes, int64_t words, int64_t cap);

double  sag_tts_flow_scale(const sag_tts_config *cfg, int64_t vehicle_count);
int64_t sag_tts_scaled_limit(int64_t value, double scale);

int64_t sag_tts_apply_cut_flow(sag_arena *scratch, const sag_tts_proposal *props, int64_t n_props,
                               const sag_tts_config *cfg, sag_tts_flow_state *fs,
                               int64_t *out_idx, sag_tts_flow_diag *diag);

#endif
