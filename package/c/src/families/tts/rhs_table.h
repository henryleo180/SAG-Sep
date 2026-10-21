#ifndef SAGSEP_FAMILIES_TTS_RHS_TABLE_H
#define SAGSEP_FAMILIES_TTS_RHS_TABLE_H

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "arena.h"

typedef struct {
    const float *table;
    int64_t      max_k;
    float        q32;
    bool         usable;
} sag_rhs_table;

void sag_rhs_table_build(sag_arena *a, const int64_t *demand, int64_t n,
                         int64_t capacity, sag_rhs_table *out);

static inline float sag_rhs_table_lookup(const sag_rhs_table *t, float new_d)
{
    if (!t->usable)
        return 2.0f * ceilf(new_d / t->q32);
    const int64_t idx = (int64_t)new_d;
    if (idx < 0 || idx > t->max_k || (float)idx != new_d)
        return 2.0f * ceilf(new_d / t->q32);
    assert((float)idx == new_d && idx >= 0 && idx <= t->max_k);
    return t->table[idx];
}

#endif
