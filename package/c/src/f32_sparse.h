#ifndef SAGSEP_F32_SPARSE_H
#define SAGSEP_F32_SPARSE_H

#include <stdbool.h>
#include <stdint.h>

#include "arena.h"

/* =====================================================================
 * materialized sparse sum
 * ===================================================================== */
float sag_f32_sum_sparse(const float *dense, const int32_t *nz, int64_t nnz, int64_t n);

/* =====================================================================
 * scaled variant
 * ===================================================================== */
float sag_f32_sum_sparse_scaled(const float *a, const float *b,
                                const int32_t *nz, int64_t nnz, int64_t n);

#endif
