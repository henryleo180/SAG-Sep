#ifndef SAGSEP_FAMILIES_ONION_H
#define SAGSEP_FAMILIES_ONION_H

#include <stdint.h>

#include "arena.h"
#include "candidate_ctx.h"
#include "sagsep.h"

void sag_generate_onion(sag_arena *a, const sag_candidate_ctx *ctx,
                        const sag_onion_config *cfg, sag_candidate_batch *out);

#endif
