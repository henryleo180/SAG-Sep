#ifndef BC_SEPARATOR_SAGSEP_H
#define BC_SEPARATOR_SAGSEP_H

#include "separator.h"
#include "instance.h"

bc_separator *bc_separator_sagsep_create(const bc_instance *inst, const char *model_path,
                                         const char *device, const char *preset,
                                         int threads, const char *tree_commit);

int64_t bc_vehicle_replica_count(const bc_instance *inst);

const char *bc_preset_for_size(int64_t n_customers);

#endif
