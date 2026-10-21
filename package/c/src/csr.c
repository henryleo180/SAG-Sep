#include "csr.h"

#include <inttypes.h>
#include <string.h>

double sag_quiet_nan(void)
{
    uint64_t bits = 0x7FF8000000000000ull;
    double out;
    memcpy(&out, &bits, sizeof out);
    return out;
}

/* ------------------------------------------------------------------ */
/* generic CSR                                                         */
/* ------------------------------------------------------------------ */

sag_csr sag_csr_alloc(sag_arena *a, int64_t cap_rows, int64_t cap_nnz)
{
    if (cap_rows < 0 || cap_nnz < 0)
        sag_die("csr_alloc: negative capacity (rows=%" PRId64 ", nnz=%" PRId64 ")",
                cap_rows, cap_nnz);
    sag_csr c;
    c.indptr = SAG_NEW(a, int64_t, cap_rows + 1);
    c.indices = SAG_NEW(a, int32_t, cap_nnz > 0 ? cap_nnz : 1);
    c.indptr[0] = 0;
    c.rows = 0;
    c.nnz = 0;
    c.cap_rows = cap_rows;
    c.cap_nnz = cap_nnz;
    return c;
}

void sag_csr_reset(sag_csr *c)
{
    c->rows = 0;
    c->nnz = 0;
    c->indptr[0] = 0;
}

void sag_csr_push_row(sag_csr *c, const int32_t *idx, int64_t n)
{
    if (c->rows >= c->cap_rows)
        sag_die("csr_push_row: row capacity %" PRId64 " exhausted", c->cap_rows);
    if (n > c->cap_nnz - c->nnz)
        sag_die("csr_push_row: nnz capacity %" PRId64 " exhausted (need %" PRId64
                " more, used %" PRId64 ")", c->cap_nnz, n, c->nnz);
    for (int64_t k = 0; k < n; ++k) {
        if (k > 0 && idx[k - 1] >= idx[k])
            sag_die("csr_push_row: row %" PRId64 " is not strictly ascending at %" PRId64
                    " (%" PRId32 " >= %" PRId32 ")", c->rows, k, idx[k - 1], idx[k]);
        c->indices[c->nnz + k] = idx[k];
    }
    c->nnz += n;
    c->rows += 1;
    c->indptr[c->rows] = c->nnz;
}

void sag_csr_validate(const sag_csr *c, int64_t n_nodes, int32_t min_id, const char *what)
{
    const char *tag = what ? what : "csr";
    if (c->indptr[0] != 0)
        sag_die("%s: indptr[0] = %" PRId64 " != 0", tag, c->indptr[0]);
    if (c->indptr[c->rows] != c->nnz)
        sag_die("%s: indptr[rows] = %" PRId64 " != nnz %" PRId64, tag, c->indptr[c->rows], c->nnz);
    for (int64_t i = 0; i < c->rows; ++i) {
        int64_t lo = c->indptr[i], hi = c->indptr[i + 1];
        if (hi < lo)
            sag_die("%s: indptr not monotone at row %" PRId64 " (%" PRId64 " -> %" PRId64 ")",
                    tag, i, lo, hi);
        for (int64_t k = lo; k < hi; ++k) {
            int32_t v = c->indices[k];
            if (v < min_id || (int64_t)v >= n_nodes)
                sag_die("%s: row %" PRId64 " has id %" PRId32 " outside [%" PRId32 ", %" PRId64 ")",
                        tag, i, v, min_id, n_nodes);
            if (k > lo && c->indices[k - 1] >= v)
                sag_die("%s: row %" PRId64 " is not strictly ascending at %" PRId64, tag, i, k);
        }
    }
}

/* ------------------------------------------------------------------ */
/* candidate batch                                                     */
/* ------------------------------------------------------------------ */

sag_candidate_batch sag_batch_alloc(sag_arena *a, int64_t n_nodes,
                                    int64_t cap_rows, int64_t cap_nnz)
{
    if (n_nodes <= 0)
        sag_die("batch_alloc: n_nodes must be positive, got %" PRId64, n_nodes);
    if (cap_rows < 0 || cap_nnz < 0)
        sag_die("batch_alloc: negative capacity (rows=%" PRId64 ", nnz=%" PRId64 ")",
                cap_rows, cap_nnz);

    sag_candidate_batch b;
    b.n_nodes = n_nodes;
    b.words = sag_mask_words(n_nodes);
    b.rows = 0;
    b.nnz = 0;
    b.cap_rows = cap_rows;
    b.cap_nnz = cap_nnz;
    b.indptr = SAG_NEW(a, int64_t, cap_rows + 1);
    b.indices = SAG_NEW(a, int32_t, cap_nnz > 0 ? cap_nnz : 1);
    b.mask = SAG_NEW0(a, uint64_t, sag_size_mul((size_t)(cap_rows > 0 ? cap_rows : 1),
                                                (size_t)b.words));
    b.family = SAG_NEW(a, int8_t, cap_rows > 0 ? cap_rows : 1);
    b.group_id = SAG_NEW(a, int32_t, cap_rows > 0 ? cap_rows : 1);
    b.variant = SAG_NEW(a, int8_t, cap_rows > 0 ? cap_rows : 1);
    b.pscore = SAG_NEW(a, double, cap_rows > 0 ? cap_rows : 1);
    b.indptr[0] = 0;
    return b;
}

void sag_batch_reset(sag_candidate_batch *b)
{
    b->rows = 0;
    b->nnz = 0;
    b->indptr[0] = 0;
}

static void batch_check_room(sag_candidate_batch *b, int64_t n)
{
    if (b->rows >= b->cap_rows)
        sag_die("batch_push: row capacity %" PRId64 " exhausted", b->cap_rows);
    if (n > b->cap_nnz - b->nnz)
        sag_die("batch_push: nnz capacity %" PRId64 " exhausted (need %" PRId64
                " more, used %" PRId64 ")", b->cap_nnz, n, b->nnz);
}

static void batch_check_provenance(int8_t family, int8_t variant)
{
    if (family < 0 || family >= SAG_FAMILY_COUNT)
        sag_die("batch_push: family code %d out of range", (int)family);
    if (variant < 0 || variant >= SAG_VARIANT_COUNT)
        sag_die("batch_push: variant code %d out of range", (int)variant);
}

static void batch_finish_row(sag_candidate_batch *b, int8_t family, int32_t group_id,
                             int8_t variant, double pscore)
{
    b->family[b->rows] = family;
    b->group_id[b->rows] = group_id;
    b->variant[b->rows] = variant;
    b->pscore[b->rows] = pscore;
    b->rows += 1;
    b->indptr[b->rows] = b->nnz;
}

void sag_batch_push_indices(sag_candidate_batch *b, const int32_t *idx, int64_t n,
                            int8_t family, int32_t group_id, int8_t variant, double pscore)
{
    batch_check_provenance(family, variant);
    batch_check_room(b, n);
    if (n <= 0)
        sag_die("batch_push_indices: empty candidate rejected (dev drops empty subsets "
                "before they reach a batch)");

    uint64_t *mask = b->mask + b->rows * b->words;
    sag_bitset_from_indices(mask, b->words, idx, n, b->n_nodes, "batch_push_indices");
    memcpy(b->indices + b->nnz, idx, (size_t)n * sizeof(int32_t));
    b->nnz += n;
    batch_finish_row(b, family, group_id, variant, pscore);
}

void sag_batch_push_mask(sag_candidate_batch *b, const uint64_t *mask,
                         int8_t family, int32_t group_id, int8_t variant, double pscore)
{
    batch_check_provenance(family, variant);
    if (mask[0] & 1u)
        sag_die("batch_push_mask: depot bit is set");
    int64_t n = sag_bitset_popcount(mask, b->words);
    batch_check_room(b, n);
    if (n <= 0)
        sag_die("batch_push_mask: empty candidate rejected");

    uint64_t *dst = b->mask + b->rows * b->words;
    sag_bitset_copy(dst, mask, b->words);
    int64_t got = sag_bitset_to_indices(mask, b->words, b->indices + b->nnz, b->cap_nnz - b->nnz);
    if (got != n)
        sag_die("batch_push_mask: popcount %" PRId64 " != expanded %" PRId64, n, got);
    if (b->indices[b->nnz] <= 0 || (int64_t)b->indices[b->nnz + n - 1] >= b->n_nodes)
        sag_die("batch_push_mask: expanded id out of range [1, %" PRId64 ")", b->n_nodes);
    b->nnz += n;
    batch_finish_row(b, family, group_id, variant, pscore);
}

void sag_batch_validate(const sag_candidate_batch *b, const char *what)
{
    const char *tag = what ? what : "batch";
    if (b->words != sag_mask_words(b->n_nodes))
        sag_die("%s: words %" PRId64 " != mask_words(%" PRId64 ")", tag, b->words, b->n_nodes);
    if (b->indptr[0] != 0)
        sag_die("%s: indptr[0] != 0", tag);
    if (b->indptr[b->rows] != b->nnz)
        sag_die("%s: indptr[rows] = %" PRId64 " != nnz %" PRId64, tag, b->indptr[b->rows], b->nnz);

    for (int64_t i = 0; i < b->rows; ++i) {
        int64_t lo = b->indptr[i], hi = b->indptr[i + 1];
        if (hi < lo)
            sag_die("%s: indptr not monotone at row %" PRId64, tag, i);
        if (hi == lo)
            sag_die("%s: row %" PRId64 " is empty (dev never emits empty subsets)", tag, i);
        const uint64_t *mask = sag_batch_mask(b, i);
        if (mask[0] & 1u)
            sag_die("%s: row %" PRId64 " has the depot bit set", tag, i);
        for (int64_t k = lo; k < hi; ++k) {
            int32_t v = b->indices[k];
            if (v <= 0 || (int64_t)v >= b->n_nodes)
                sag_die("%s: row %" PRId64 " id %" PRId32 " out of range [1, %" PRId64 ")",
                        tag, i, v, b->n_nodes);
            if (k > lo && b->indices[k - 1] >= v)
                sag_die("%s: row %" PRId64 " not strictly ascending at %" PRId64, tag, i, k);
            if (!sag_bit_test(mask, v))
                sag_die("%s: row %" PRId64 " mask/CSR disagree on id %" PRId32, tag, i, v);
        }
        int64_t pop = sag_bitset_popcount(mask, b->words);
        if (pop != hi - lo)
            sag_die("%s: row %" PRId64 " popcount %" PRId64 " != size %" PRId64,
                    tag, i, pop, hi - lo);
        batch_check_provenance(b->family[i], b->variant[i]);
    }
}
