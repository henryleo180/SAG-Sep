#include "instance.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

void bc_instance_free(bc_instance *inst)
{
    free(inst->demand);
    free(inst->coords);
    free(inst->weight);
    memset(inst, 0, sizeof *inst);
}

#define BC_LEARNED_DEMAND_MAX ((int64_t)1 << 24)

int64_t bc_instance_validate(const bc_instance *inst, _Bool learned, const char *what)
{
    if (inst->n_nodes < 2)
        bc_die("%s: instance %s has %lld nodes; a depot and at least one customer "
               "are required", what, inst->name, (long long)inst->n_nodes);
    if (inst->capacity <= 0)
        bc_die("%s: instance %s has capacity %lld (must be positive)",
               what, inst->name, (long long)inst->capacity);
    if (inst->demand[0] != 0)
        bc_die("%s: instance %s has depot demand %lld (must be 0)",
               what, inst->name, (long long)inst->demand[0]);

    __int128 total = 0;
    for (int64_t i = 1; i < inst->n_nodes; ++i) {
        const int64_t d = inst->demand[i];
        if (d < 0)
            bc_die("%s: instance %s has demand %lld at customer %lld (must be "
                   "non-negative)", what, inst->name, (long long)d, (long long)i);
        if (d > inst->capacity)
            bc_die("%s: instance %s has demand %lld at customer %lld, above the "
                   "capacity %lld -- no vehicle can serve it, so the instance is "
                   "infeasible and its singleton capacity row has no nonzeros",
                   what, inst->name, (long long)d, (long long)i,
                   (long long)inst->capacity);
        if (learned && d >= BC_LEARNED_DEMAND_MAX)
            bc_die("%s: instance %s has demand %lld at customer %lld; the learned "
                   "separator's C API requires every demand below 2^24",
                   what, inst->name, (long long)d, (long long)i);
        total += d;
    }
    if (total > (__int128)INT64_MAX)
        bc_die("%s: instance %s has a total demand beyond int64_t", what, inst->name);
    if (learned && total >= (__int128)BC_LEARNED_DEMAND_MAX)
        bc_die("%s: instance %s has total demand %lld; the learned separator's C "
               "API requires it below 2^24", what, inst->name, (long long)total);
    return (int64_t)total;
}
