#ifndef SAGSEP_FAMILIES_TTS_PROPOSAL_H
#define SAGSEP_FAMILIES_TTS_PROPOSAL_H

#include <stdbool.h>
#include <stdint.h>

#include "arena.h"
#include "bitset.h"
#include "sagsep.h"

typedef struct {
    float    *selection;
    uint64_t *mask;
    int8_t    source;
    double    violation;
    double    score;
    int8_t    origin_stage;
    int32_t   origin_seed;
} sag_tts_proposal;

#define SAG_TTS_STAGE_RAW 0
#define SAG_TTS_STAGE_LS  1

typedef struct {
    uint64_t *keys;
    int32_t  *slots;
    int64_t   hash_cap;
    int64_t   cap;
    int64_t   count;
    int64_t   words;
} sag_tts_keyset;

void sag_tts_keyset_init(sag_arena *a, sag_tts_keyset *set, int64_t words, int64_t cap);
bool sag_tts_keyset_has(const sag_tts_keyset *set, const uint64_t *mask);
bool sag_tts_keyset_add(sag_tts_keyset *set, const uint64_t *mask);

void sag_tts_pack_selection(uint64_t *mask, const float *sel, int64_t n_nodes, int64_t words);

static inline int64_t sag_tts_mask_to_indices(const uint64_t *mask, int64_t words,
                                              int32_t *out, int64_t cap)
{
    return sag_bitset_to_indices(mask, words, out, cap);
}

int64_t sag_tts_first_seen_unique(sag_arena *scratch, const uint64_t *masks, int64_t rows,
                                  int64_t words, int64_t *keep);

#endif
