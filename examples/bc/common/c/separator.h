#ifndef BC_SEPARATOR_H
#define BC_SEPARATOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    int64_t rows;
    const int64_t *indptr;
    const int32_t *nodes;
    const double  *rhs;
} bc_proposals;

typedef struct bc_separator bc_separator;

struct bc_separator {
    const char *name;

    void (*propose)(bc_separator *self, int64_t m, const int64_t *u, const int64_t *v,
                    const double *x, bc_proposals *out);

    void (*commit)(bc_separator *self, const int64_t *accepted, int64_t count, bool retain);

    void (*describe)(bc_separator *self, char *buf, size_t cap);

    void (*destroy)(bc_separator *self);
};

#endif
