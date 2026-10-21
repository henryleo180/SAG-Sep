#ifndef BC_INSTANCE_H
#define BC_INSTANCE_H

#include <stdint.h>

#define BC_NAME_MAX 128

typedef struct {
    char     name[BC_NAME_MAX];
    int64_t  n_nodes;
    int64_t  capacity;
    int64_t *demand;
    double  *coords;
    double  *weight;
} bc_instance;

void bc_instance_free(bc_instance *inst);

int64_t bc_instance_validate(const bc_instance *inst, _Bool learned, const char *what);

#endif
