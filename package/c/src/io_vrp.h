#ifndef SAGSEP_IO_VRP_H
#define SAGSEP_IO_VRP_H

#include <stdbool.h>
#include <stdint.h>

#include "arena.h"
#include "sagsep.h"

double sag_euc2d(double xi, double yi, double xj, double yj);

void sag_distance_matrix(const sag_instance *inst, double *out);

void sag_load_instance(sag_arena *a, const char *vrp_path, int64_t vehicles,
                       bool has_optimal, double optimal, sag_instance *out);

#endif
