#include "families/tts/proposal.h"

#include <string.h>

/* ===================================================================== */
/* keyset -- python `set[bytes]` fixed-capacity open-addressing stand-in */
/* ===================================================================== */

static int64_t pow2_at_least(int64_t v)
{
    int64_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

void sag_tts_keyset_init(sag_arena *a, sag_tts_keyset *set, int64_t words, int64_t cap)
{
    if (words <= 0) sag_die("tts/proposal: keyset words=%lld must be positive", (long long)words);
    if (cap <= 0)   sag_die("tts/proposal: keyset cap=%lld must be positive", (long long)cap);
    set->words    = words;
    set->cap      = cap;
    set->count    = 0;
    set->hash_cap = pow2_at_least(cap * 2);
    set->keys     = SAG_NEW(a, uint64_t, cap * words);
    set->slots    = SAG_NEW(a, int32_t, set->hash_cap);
    for (int64_t i = 0; i < set->hash_cap; ++i) set->slots[i] = -1;
}

static int32_t keyset_probe(const sag_tts_keyset *set, const uint64_t *mask, int64_t *free_slot)
{
    const uint64_t h    = sag_bitset_hash(mask, set->words);
    const int64_t  mask_bits = set->hash_cap - 1;
    int64_t        idx  = (int64_t)(h & (uint64_t)mask_bits);
    for (int64_t step = 0; step < set->hash_cap; ++step) {
        const int32_t slot = set->slots[idx];
        if (slot < 0) {
            if (free_slot) *free_slot = idx;
            return -1;
        }
        if (sag_bitset_equal(set->keys + (int64_t)slot * set->words, mask, set->words))
            return slot;
        idx = (idx + 1) & mask_bits;
    }
    sag_die("tts/proposal: keyset probe wrapped around (cap=%lld count=%lld) -- the capacity was computed wrong",
            (long long)set->cap, (long long)set->count);
}

bool sag_tts_keyset_has(const sag_tts_keyset *set, const uint64_t *mask)
{
    return keyset_probe(set, mask, NULL) >= 0;
}

bool sag_tts_keyset_add(sag_tts_keyset *set, const uint64_t *mask)
{
    int64_t free_slot = -1;
    if (keyset_probe(set, mask, &free_slot) >= 0) return false;
    if (set->count >= set->cap)
        sag_die("tts/proposal: keyset overflow (cap=%lld) -- the reserved amount was computed wrong (growing is forbidden)",
                (long long)set->cap);
    const int32_t row = (int32_t)set->count++;
    sag_bitset_copy(set->keys + (int64_t)row * set->words, mask, set->words);
    set->slots[free_slot] = row;
    return true;
}

/* ===================================================================== */
/* selection -> mask                                                      */
/* ===================================================================== */

void sag_tts_pack_selection(uint64_t *mask, const float *sel, int64_t n_nodes, int64_t words)
{
    sag_bitset_zero(mask, words);
    for (int64_t i = 0; i < n_nodes; ++i)
        if (sel[i] > 0.5f) sag_bit_set(mask, i);
}

/* ===================================================================== */
/* first-seen order-preserving dedup                                     */
/* ===================================================================== */

int64_t sag_tts_first_seen_unique(sag_arena *scratch, const uint64_t *masks, int64_t rows,
                                  int64_t words, int64_t *keep)
{
    if (rows <= 0) return 0;
    const size_t   mark = sag_arena_mark(scratch);
    sag_tts_keyset set;
    sag_tts_keyset_init(scratch, &set, words, rows);
    int64_t n_keep = 0;
    for (int64_t i = 0; i < rows; ++i) {
        if (sag_tts_keyset_add(&set, masks + i * words)) keep[n_keep++] = i;
    }
    sag_arena_release(scratch, mark);
    return n_keep;
}
