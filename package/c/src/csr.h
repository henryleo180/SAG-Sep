#ifndef SAGSEP_CSR_H
#define SAGSEP_CSR_H

#include <stdint.h>

#include "arena.h"
#include "bitset.h"
#include "sagsep.h"

/* ---------------- generic CSR (int64 indptr / int32 indices) ---------------- */

typedef struct {
    int64_t *indptr;
    int32_t *indices;
    int64_t  rows;
    int64_t  nnz;
    int64_t  cap_rows;
    int64_t  cap_nnz;
} sag_csr;

sag_csr sag_csr_alloc(sag_arena *a, int64_t cap_rows, int64_t cap_nnz);
void    sag_csr_reset(sag_csr *c);
void    sag_csr_push_row(sag_csr *c, const int32_t *idx, int64_t n);

static inline const int32_t *sag_csr_row(const sag_csr *c, int64_t i, int64_t *len)
{
    int64_t lo = c->indptr[i], hi = c->indptr[i + 1];
    if (len) *len = hi - lo;
    return c->indices + lo;
}

void sag_csr_validate(const sag_csr *c, int64_t n_nodes, int32_t min_id, const char *what);

/* ---------------- candidate batch ---------------- */

sag_candidate_batch sag_batch_alloc(sag_arena *a, int64_t n_nodes,
                                    int64_t cap_rows, int64_t cap_nnz);
void sag_batch_reset(sag_candidate_batch *b);

void sag_batch_push_indices(sag_candidate_batch *b, const int32_t *idx, int64_t n,
                            int8_t family, int32_t group_id, int8_t variant, double pscore);
void sag_batch_push_mask(sag_candidate_batch *b, const uint64_t *mask,
                         int8_t family, int32_t group_id, int8_t variant, double pscore);

static inline const int32_t *sag_batch_row(const sag_candidate_batch *b, int64_t i, int64_t *len)
{
    int64_t lo = b->indptr[i], hi = b->indptr[i + 1];
    if (len) *len = hi - lo;
    return b->indices + lo;
}

static inline const uint64_t *sag_batch_mask(const sag_candidate_batch *b, int64_t i)
{
    return b->mask + i * b->words;
}

static inline int64_t sag_batch_size(const sag_candidate_batch *b, int64_t i)
{
    return b->indptr[i + 1] - b->indptr[i];
}

void sag_batch_validate(const sag_candidate_batch *b, const char *what);

double sag_quiet_nan(void);

#endif
