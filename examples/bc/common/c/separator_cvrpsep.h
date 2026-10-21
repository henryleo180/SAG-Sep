#ifndef BC_SEPARATOR_CVRPSEP_H
#define BC_SEPARATOR_CVRPSEP_H

#include "separator.h"
#include "instance.h"

bc_separator *bc_separator_cvrpsep_create(const bc_instance *inst, int max_cuts,
                                          double min_violation,
                                          const char *tree_commit);

#endif
