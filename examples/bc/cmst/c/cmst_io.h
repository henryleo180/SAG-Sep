#ifndef BC_CMST_IO_H
#define BC_CMST_IO_H

#include <stdint.h>

#include "instance.h"

void bc_cmst_load(const char *path, int64_t capacity, bc_instance *out);

#endif
