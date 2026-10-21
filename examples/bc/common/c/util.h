#ifndef BC_UTIL_H
#define BC_UTIL_H

#include <stddef.h>

#if defined(__GNUC__)
#define BC_PRINTF(fmt_index, first_arg) \
    __attribute__((format(printf, fmt_index, first_arg)))
#else
#define BC_PRINTF(fmt_index, first_arg)
#endif

_Noreturn void bc_die(const char *fmt, ...) BC_PRINTF(1, 2);

void *bc_malloc(size_t bytes);
void *bc_calloc(size_t count, size_t size);
void *bc_realloc(void *ptr, size_t bytes);

double bc_seconds(void);

void bc_format_double(char *buf, size_t cap, double value);

#endif
