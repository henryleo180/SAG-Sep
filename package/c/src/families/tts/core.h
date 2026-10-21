#ifndef SAGSEP_FAMILIES_TTS_CORE_H
#define SAGSEP_FAMILIES_TTS_CORE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "arena.h"
#include "candidate_ctx.h"
#include "families/tts/adddrop.h"
#include "families/tts/flow.h"
#include "families/tts/old_repair.h"
#include "sagsep.h"

#define SAG_TTS_SHRINK_EPS 0.01

/* ===================================================================== */
/* 1. old-cut bank snapshot (candidate_ctx.h only forward-declares sag_bank) */
/* ===================================================================== */

struct sag_bank {
    int64_t         n_nodes;
    int64_t         words;
    int64_t         rows;
    const uint64_t *mask;
    const int64_t  *indptr;
    const int32_t  *indices;
};

/* ===================================================================== */
/* 2. per-vehicle diagnostics (= the dict of the python reduced arm, including the key insertion order) */
/* ===================================================================== */

typedef enum {
    SAG_TTS_DK_COMPRESSION = 0,
    SAG_TTS_DK_OLD_REPAIR,
    SAG_TTS_DK_SOURCES,
    SAG_TTS_DK_ADDDROP,
    SAG_TTS_DK_FLOW,
    SAG_TTS_DK_COUNT
} sag_tts_diag_key;

typedef struct sag_tts_diag_vehicle {
    int8_t  key_order[SAG_TTS_DK_COUNT];
    int64_t n_keys;

    int64_t one_edge_customer_supernodes;
    int64_t full_customer_supernodes;
    int64_t frozen_customer_supernodes;

    bool                     has_old_repair;
    sag_tts_old_repair_diag  old_repair;
    bool                     has_adddrop;
    sag_tts_adddrop_diag     adddrop;

    bool    sources_present[2];
    int8_t  stats_order[2];
    int64_t n_stats;
    int64_t stats_emitted_count[2];
    int64_t old_repair_emitted_count;
    int64_t adddrop_emitted_count;

    int64_t           flow_input_count;
    int64_t           flow_output_count;
    int64_t           flow_global_emitted;
    double            flow_vehicle_scale;
    sag_tts_flow_diag flow;
} sag_tts_diag_vehicle;

typedef struct {
    sag_tts_diag_vehicle *vehicles;
    int64_t               count;
    int64_t               cap;
} sag_tts_diag;

void sag_tts_diag_write_json(FILE *out, const sag_tts_diag *diag);

/* ===================================================================== */
/* 3. family entry point                                                 */
/* ===================================================================== */

void sag_generate_tts(sag_arena *a, const sag_candidate_ctx *ctx, const sag_tts_config *cfg,
                      sag_candidate_batch *out, sag_tts_diag *diag);

#endif
