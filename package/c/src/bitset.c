#include "bitset.h"

#include <inttypes.h>

int64_t sag_bitset_popcount(const uint64_t *a, int64_t words)
{
    int64_t total = 0;
    for (int64_t w = 0; w < words; ++w)
        total += __builtin_popcountll(a[w]);
    return total;
}

int64_t sag_bitset_hamming(const uint64_t *a, const uint64_t *b, int64_t words)
{
    int64_t total = 0;
    for (int64_t w = 0; w < words; ++w)
        total += __builtin_popcountll(a[w] ^ b[w]);
    return total;
}

int64_t sag_bitset_intersection(const uint64_t *a, const uint64_t *b, int64_t words)
{
    int64_t total = 0;
    for (int64_t w = 0; w < words; ++w)
        total += __builtin_popcountll(a[w] & b[w]);
    return total;
}

int64_t sag_bitset_union_count(const uint64_t *a, const uint64_t *b, int64_t words)
{
    int64_t total = 0;
    for (int64_t w = 0; w < words; ++w)
        total += __builtin_popcountll(a[w] | b[w]);
    return total;
}

void sag_bitset_xor(uint64_t *dst, const uint64_t *a, const uint64_t *b, int64_t words)
{
    for (int64_t w = 0; w < words; ++w)
        dst[w] = a[w] ^ b[w];
}

void sag_bitset_and(uint64_t *dst, const uint64_t *a, const uint64_t *b, int64_t words)
{
    for (int64_t w = 0; w < words; ++w)
        dst[w] = a[w] & b[w];
}

void sag_bitset_or(uint64_t *dst, const uint64_t *a, const uint64_t *b, int64_t words)
{
    for (int64_t w = 0; w < words; ++w)
        dst[w] = a[w] | b[w];
}

uint64_t sag_bitset_hash(const uint64_t *a, int64_t words)
{
    uint64_t h = 1469598103934665603ull;
    for (int64_t w = 0; w < words; ++w) {
        h ^= a[w];
        h *= 1099511628211ull;
    }
    h ^= h >> 30; h *= 0xbf58476d1ce4e5b9ull;
    h ^= h >> 27; h *= 0x94d049bb133111ebull;
    h ^= h >> 31;
    return h;
}

int64_t sag_bitset_from_indices(uint64_t *dst, int64_t words, const int32_t *idx,
                                int64_t n, int64_t n_nodes, const char *what)
{
    sag_bitset_zero(dst, words);
    for (int64_t k = 0; k < n; ++k) {
        int64_t v = idx[k];
        if (v <= 0 || v >= n_nodes)
            sag_die("%s: index %" PRId64 " out of range (depot 0 forbidden, n_nodes=%" PRId64 ")",
                    what ? what : "bitset_from_indices", v, n_nodes);
        if (k > 0 && idx[k - 1] >= idx[k])
            sag_die("%s: indices are not strictly ascending at position %" PRId64
                    " (%" PRId32 " >= %" PRId32 ")",
                    what ? what : "bitset_from_indices", k, idx[k - 1], idx[k]);
        sag_bit_set(dst, v);
    }
    return n;
}

int64_t sag_bitset_to_indices(const uint64_t *src, int64_t words, int32_t *out, int64_t cap)
{
    int64_t count = 0;
    for (int64_t w = 0; w < words; ++w) {
        uint64_t bits = src[w];
        while (bits) {
            int64_t bit = __builtin_ctzll(bits);
            bits &= bits - 1;
            if (count >= cap)
                sag_die("bitset_to_indices: output capacity %" PRId64 " exceeded", cap);
            out[count++] = (int32_t)((w << 6) + bit);
        }
    }
    return count;
}
