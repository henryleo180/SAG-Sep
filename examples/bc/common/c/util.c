#define _POSIX_C_SOURCE 200809L

#include "util.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

_Noreturn void bc_die(const char *fmt, ...)
{
    va_list args;
    fputs("bc: ", stderr);
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    exit(EXIT_FAILURE);
}

void *bc_malloc(size_t bytes)
{
    void *p = malloc(bytes ? bytes : 1);
    if (!p) bc_die("out of memory (%zu bytes)", bytes);
    return p;
}

void *bc_calloc(size_t count, size_t size)
{
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (!p) bc_die("out of memory (%zu x %zu bytes)", count, size);
    return p;
}

void *bc_realloc(void *ptr, size_t bytes)
{
    void *p = realloc(ptr, bytes ? bytes : 1);
    if (!p) bc_die("out of memory (%zu bytes)", bytes);
    return p;
}

double bc_seconds(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        bc_die("clock_gettime(CLOCK_MONOTONIC) failed");
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

void bc_format_double(char *buf, size_t cap, double value)
{
    if (!isfinite(value)) bc_die("cannot format a non-finite number");
    for (int precision = 15; precision <= 17; ++precision) {
        int written = snprintf(buf, cap, "%.*g", precision, value);
        if (written < 0 || (size_t)written >= cap)
            bc_die("number does not fit in a %zu byte buffer", cap);
        if (strtod(buf, NULL) == value) break;
    }
    if (strpbrk(buf, ".e") == NULL) {
        size_t len = strlen(buf);
        if (len + 3 > cap) bc_die("number does not fit in a %zu byte buffer", cap);
        memcpy(buf + len, ".0", 3);
    }
}
