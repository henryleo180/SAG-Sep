#ifndef BCD_DIRECTED_DRIVER_H
#define BCD_DIRECTED_DRIVER_H

#include <stdbool.h>

typedef struct {
    const char *key;
    const char *prog;
    const char *summary;
    const char *model_line;
    bool open_routes;
    bool distance_flag;
} bcd_problem;

int bcd_main(int argc, char **argv, const bcd_problem *problem);

#endif
