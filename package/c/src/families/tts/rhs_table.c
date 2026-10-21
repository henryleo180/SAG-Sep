#include "families/tts/rhs_table.h"

#define SAG_RHS_INT24_LIMIT     ((int64_t)1 << 24)
#define SAG_RHS_TABLE_MAX_BYTES ((int64_t)64 << 20)

void sag_rhs_table_build(sag_arena *a, const int64_t *demand, int64_t n,
                         int64_t capacity, sag_rhs_table *out)
{
    const double q = ((double)capacity > 1e-9) ? (double)capacity : 1e-9;
    out->table  = NULL;
    out->max_k  = 0;
    out->q32    = (float)q;
    out->usable = false;

    if (capacity <= 0)
        return;

    int64_t total = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t d = demand[i];
        if (d < 0 || d >= SAG_RHS_INT24_LIMIT)
            return;
        total += d;
        if (total >= SAG_RHS_INT24_LIMIT)
            return;
    }

    const int64_t bytes = (total + 1) * (int64_t)sizeof(float);
    if (bytes > SAG_RHS_TABLE_MAX_BYTES)
        return;

    const float q32 = out->q32;
    float *table = SAG_NEW(a, float, total + 1);
    for (int64_t k = 0; k <= total; ++k)
        table[k] = 2.0f * ceilf((float)k / q32);

    out->table  = table;
    out->max_k  = total;
    out->usable = true;
}
