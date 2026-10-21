#ifndef SAGSEP_FAMILIES_SWEEP_H
#define SAGSEP_FAMILIES_SWEEP_H

#include <stdint.h>

#include "arena.h"
#include "candidate_ctx.h"
#include "sagsep.h"

void sag_generate_sweep(sag_arena *a, const sag_candidate_ctx *ctx,
                        const sag_sweep_config *cfg, const sag_growth_config *growth,
                        sag_candidate_batch *out);

/* ------------------------------------------------------------------ */
/* numpy semantic primitives                                          */
/* ------------------------------------------------------------------ */

void sag_sweep_stable_argsort(const double *key, int64_t n, int32_t *idx, int32_t *scratch);

int64_t sag_sweep_argmax_f64(const double *v, int64_t n);
int64_t sag_sweep_argmax_f32(const float *v, int64_t n);

#endif
