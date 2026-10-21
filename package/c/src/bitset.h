#ifndef SAGSEP_BITSET_H
#define SAGSEP_BITSET_H

#include <stdint.h>
#include <string.h>

#include "arena.h"

#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#error "mask row bytes must be byte-for-byte identical to python mask_row.tobytes(): little-endian only"
#endif

static inline int64_t sag_mask_words(int64_t n_nodes)
{
    return (n_nodes + 63) >> 6;
}

static inline void sag_bit_set(uint64_t *w, int64_t i)
{
    w[i >> 6] |= (uint64_t)1 << (i & 63);
}

static inline void sag_bit_clear(uint64_t *w, int64_t i)
{
    w[i >> 6] &= ~((uint64_t)1 << (i & 63));
}

static inline int sag_bit_test(const uint64_t *w, int64_t i)
{
    return (int)((w[i >> 6] >> (i & 63)) & (uint64_t)1);
}

static inline void sag_bitset_zero(uint64_t *w, int64_t words)
{
    memset(w, 0, (size_t)words * sizeof(uint64_t));
}

static inline void sag_bitset_copy(uint64_t *dst, const uint64_t *src, int64_t words)
{
    memcpy(dst, src, (size_t)words * sizeof(uint64_t));
}

static inline int sag_bitset_equal(const uint64_t *a, const uint64_t *b, int64_t words)
{
    return memcmp(a, b, (size_t)words * sizeof(uint64_t)) == 0;
}

static inline int sag_bitset_cmp(const uint64_t *a, const uint64_t *b, int64_t words)
{
    return memcmp(a, b, (size_t)words * sizeof(uint64_t));
}

int64_t  sag_bitset_popcount(const uint64_t *a, int64_t words);
int64_t  sag_bitset_hamming(const uint64_t *a, const uint64_t *b, int64_t words);
int64_t  sag_bitset_intersection(const uint64_t *a, const uint64_t *b, int64_t words);
int64_t  sag_bitset_union_count(const uint64_t *a, const uint64_t *b, int64_t words);

void     sag_bitset_xor(uint64_t *dst, const uint64_t *a, const uint64_t *b, int64_t words);
void     sag_bitset_and(uint64_t *dst, const uint64_t *a, const uint64_t *b, int64_t words);
void     sag_bitset_or(uint64_t *dst, const uint64_t *a, const uint64_t *b, int64_t words);

uint64_t sag_bitset_hash(const uint64_t *a, int64_t words);

int64_t sag_bitset_from_indices(uint64_t *dst, int64_t words, const int32_t *idx,
                                int64_t n, int64_t n_nodes, const char *what);
int64_t sag_bitset_to_indices(const uint64_t *src, int64_t words, int32_t *out, int64_t cap);

#endif
