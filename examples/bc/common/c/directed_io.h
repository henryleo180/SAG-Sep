#ifndef BC_DIRECTED_IO_H
#define BC_DIRECTED_IO_H

#include <stdint.h>

#include "instance.h"

typedef enum {
    BC_DISTANCE_EXACT = 0,
    BC_DISTANCE_ROUND = 1
} bc_distance_mode;

typedef struct {
    bc_instance inst;
    int64_t     vehicles;
    const char *distance_convention;
} bc_directed_instance;

void bc_directed_load(const char *path, bc_distance_mode distance,
                      bc_directed_instance *out);

void bc_directed_free(bc_directed_instance *out);

#endif
