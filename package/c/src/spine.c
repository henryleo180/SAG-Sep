#define _POSIX_C_SOURCE 200809L

#include "spine.h"

#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bitset.h"
#include "csr.h"

/* ===================================================================== */
/* 0. Timing (diagnostics only; timings are exempt from bit comparison)   */
/* ===================================================================== */

static double now_seconds(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        sag_die("spine: clock_gettime(CLOCK_MONOTONIC) failed");
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ===================================================================== */
/* 1. numpy DOUBLE_pairwise_sum                                           */
/* ===================================================================== */

#define SAG_PW_BLOCKSIZE 128

double sag_f64_pairwise_sum(const double *a, int64_t n)
{
    if (n < 8) {
        double res = 0.0;
        for (int64_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= SAG_PW_BLOCKSIZE) {
        double r[8];
        for (int k = 0; k < 8; ++k) r[k] = a[k];
        int64_t i = 8;
        const int64_t end = n - (n % 8);
        for (; i < end; i += 8) {
            r[0] += a[i + 0]; r[1] += a[i + 1]; r[2] += a[i + 2]; r[3] += a[i + 3];
            r[4] += a[i + 4]; r[5] += a[i + 5]; r[6] += a[i + 6]; r[7] += a[i + 7];
        }
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    int64_t n2 = n / 2;
    n2 -= n2 % 8;
    return sag_f64_pairwise_sum(a, n2) + sag_f64_pairwise_sum(a + n2, n - n2);
}

#define SAG_NUMPY_BUFSIZE 8192

double sag_f64_sum_numpy(const double *a, int64_t n)
{
    double res = 0.0;
    int64_t off = 0;
    while (off < n) {
        int64_t chunk = n - off;
        if (chunk > SAG_NUMPY_BUFSIZE) chunk = SAG_NUMPY_BUFSIZE;
        res += sag_f64_pairwise_sum(a + off, chunk);
        off += chunk;
    }
    return res;
}

/* ===================================================================== */
/* 2. Open-addressing mask set (locates only, decides no output order)    */
/* ===================================================================== */

typedef struct {
    int64_t        *slot;
    uint64_t       *hash;
    int64_t         cap;
    const uint64_t *rows;
    int64_t         words;
} sag_mask_set;

static int64_t next_pow2_at_least(int64_t v)
{
    int64_t c = 1;
    while (c < v) {
        c <<= 1;
        if (c <= 0) sag_die("spine: hash capacity overflow (v=%" PRId64 ")", v);
    }
    return c;
}

static sag_mask_set mask_set_make(sag_arena *a, int64_t expect,
                                  const uint64_t *rows, int64_t words)
{
    sag_mask_set s;
    s.cap = next_pow2_at_least((expect > 0 ? expect : 1) * 2);
    s.slot = SAG_NEW(a, int64_t, s.cap);
    s.hash = SAG_NEW(a, uint64_t, s.cap);
    for (int64_t i = 0; i < s.cap; ++i) { s.slot[i] = -1; s.hash[i] = 0; }
    s.rows = rows;
    s.words = words;
    return s;
}

static int64_t mask_set_find(const sag_mask_set *s, const uint64_t *key, uint64_t h,
                             int64_t *free_slot)
{
    const int64_t bits = s->cap - 1;
    int64_t i = (int64_t)(h & (uint64_t)bits);
    for (int64_t probe = 0; probe < s->cap; ++probe) {
        const int64_t v = s->slot[i];
        if (v < 0) {
            if (free_slot) *free_slot = i;
            return -1;
        }
        if (s->hash[i] == h && sag_bitset_equal(s->rows + v * s->words, key, s->words))
            return v;
        i = (i + 1) & bits;
    }
    sag_die("spine: mask hash table is full (cap=%" PRId64 ")", s->cap);
}

static void mask_set_put(sag_mask_set *s, int64_t slot, int64_t row, uint64_t h)
{
    s->slot[slot] = row;
    s->hash[slot] = h;
}

/* ===================================================================== */
/* 3. assemble -- the raw concatenation order of the decode pipeline      */
/* ===================================================================== */

static void validate_family_batch(const sag_candidate_batch *b, int8_t code,
                                  int64_t vehicle_count, const char *name)
{
    if (b->rows == 0) return;
    for (int64_t i = 0; i < b->rows; ++i) {
        if (b->family[i] != code)
            sag_die("assemble: %s batch row %" PRId64 " has foreign family code %d",
                    name, i, (int)b->family[i]);
        if (b->group_id[i] < 0 || (int64_t)b->group_id[i] >= vehicle_count)
            sag_die("assemble: %s batch group_id %" PRId32 " out of range [0, %" PRId64 ")",
                    name, b->group_id[i], vehicle_count);
    }
}

static void group_bucket(sag_arena *a, const sag_candidate_batch *b, int64_t k,
                         int64_t **out_off, int64_t **out_order)
{
    int64_t *off = SAG_NEW0(a, int64_t, k + 1);
    int64_t *order = SAG_NEW(a, int64_t, b->rows > 0 ? b->rows : 1);
    for (int64_t i = 0; i < b->rows; ++i) off[b->group_id[i] + 1] += 1;
    for (int64_t v = 0; v < k; ++v) off[v + 1] += off[v];
    int64_t *cursor = SAG_NEW(a, int64_t, k > 0 ? k : 1);
    for (int64_t v = 0; v < k; ++v) cursor[v] = off[v];
    for (int64_t i = 0; i < b->rows; ++i) order[cursor[b->group_id[i]]++] = i;
    *out_off = off;
    *out_order = order;
}

static void push_row_from(sag_candidate_batch *dst, const sag_candidate_batch *src, int64_t i)
{
    int64_t len = 0;
    const int32_t *idx = sag_batch_row(src, i, &len);
    sag_batch_push_indices(dst, idx, len, src->family[i], src->group_id[i],
                           src->variant[i], src->pscore[i]);
}

void sag_assemble(sag_arena *a, const sag_candidate_batch *sweep,
                  const sag_candidate_batch *onion, const sag_candidate_batch *tts,
                  int64_t vehicle_count, sag_candidate_batch *raw)
{
    const int64_t n_nodes = sweep->n_nodes;
    if (onion->n_nodes != n_nodes || tts->n_nodes != n_nodes)
        sag_die("assemble: n_nodes mismatch across family batches (%" PRId64 "/%" PRId64
                "/%" PRId64 ")", sweep->n_nodes, onion->n_nodes, tts->n_nodes);
    if (vehicle_count <= 0)
        sag_die("assemble: vehicle_count must be positive, got %" PRId64, vehicle_count);
    validate_family_batch(sweep, SAG_FAMILY_SWEEP, vehicle_count, "sweep");
    validate_family_batch(tts, SAG_FAMILY_TTS, vehicle_count, "tts");
    for (int64_t i = 0; i < onion->rows; ++i)
        if (onion->family[i] != SAG_FAMILY_ONION)
            sag_die("assemble: onion batch row %" PRId64 " has foreign family code %d",
                    i, (int)onion->family[i]);

    const int64_t rows = sweep->rows + tts->rows + onion->rows;
    const int64_t nnz = sweep->nnz + tts->nnz + onion->nnz;
    *raw = sag_batch_alloc(a, n_nodes, rows, nnz);

    const size_t mark = sag_arena_mark(a);
    int64_t *sw_off = NULL, *sw_order = NULL, *tt_off = NULL, *tt_order = NULL;
    group_bucket(a, sweep, vehicle_count, &sw_off, &sw_order);
    group_bucket(a, tts, vehicle_count, &tt_off, &tt_order);

    for (int64_t v = 0; v < vehicle_count; ++v) {
        for (int64_t k = sw_off[v]; k < sw_off[v + 1]; ++k)
            push_row_from(raw, sweep, sw_order[k]);
        for (int64_t k = tt_off[v]; k < tt_off[v + 1]; ++k)
            push_row_from(raw, tts, tt_order[k]);
    }
    for (int64_t i = 0; i < onion->rows; ++i)
        push_row_from(raw, onion, i);

    sag_arena_release(a, mark);
    if (raw->rows != rows)
        sag_die("assemble: emitted %" PRId64 " rows, expected %" PRId64, raw->rows, rows);
}

/* ===================================================================== */
/* 4. dedup -- deduplication of the raw candidates                        */
/* ===================================================================== */

static uint64_t origin_score_bits(double s)
{
    if (s == 0.0) s = 0.0;
    if (isnan(s)) return UINT64_C(0x7FF8000000000000);
    uint64_t bits;
    memcpy(&bits, &s, sizeof bits);
    return bits;
}

static bool same_origin(const sag_candidate_batch *raw, int64_t i, int64_t j)
{
    return raw->family[i] == raw->family[j]
        && raw->group_id[i] == raw->group_id[j]
        && raw->variant[i] == raw->variant[j]
        && origin_score_bits(raw->pscore[i]) == origin_score_bits(raw->pscore[j]);
}

void sag_dedup(sag_arena *a, const sag_candidate_batch *raw, sag_dedup_result *out)
{
    const int64_t b = raw->rows;
    out->raw_count = b;
    out->unique = sag_batch_alloc(a, raw->n_nodes, b, raw->nnz);
    out->origin_indptr = SAG_NEW0(a, int64_t, b + 1);
    out->origin_raw = SAG_NEW(a, int32_t, b > 0 ? b : 1);
    out->n_origins = 0;
    if (b == 0) return;

    const size_t mark = sag_arena_mark(a);
    int64_t *head = SAG_NEW(a, int64_t, b);
    int64_t *tail = SAG_NEW(a, int64_t, b);
    int64_t *next = SAG_NEW(a, int64_t, b);
    int64_t *count = SAG_NEW0(a, int64_t, b);
    sag_mask_set set = mask_set_make(a, b, out->unique.mask, out->unique.words);

    for (int64_t i = 0; i < b; ++i) {
        const uint64_t *m = sag_batch_mask(raw, i);
        const uint64_t h = sag_bitset_hash(m, raw->words);
        int64_t slot = -1;
        int64_t u = mask_set_find(&set, m, h, &slot);
        if (u < 0) {
            u = out->unique.rows;
            push_row_from(&out->unique, raw, i);
            mask_set_put(&set, slot, u, h);
            head[u] = tail[u] = -1;
        }

        bool dup = false;
        for (int64_t j = head[u]; j >= 0; j = next[j]) {
            if (same_origin(raw, i, j)) { dup = true; break; }
        }
        if (dup) continue;
        next[i] = -1;
        if (tail[u] < 0) head[u] = i; else next[tail[u]] = i;
        tail[u] = i;
        count[u] += 1;
    }

    const int64_t u_rows = out->unique.rows;
    for (int64_t u = 0; u < u_rows; ++u)
        out->origin_indptr[u + 1] = out->origin_indptr[u] + count[u];
    out->n_origins = out->origin_indptr[u_rows];
    int64_t w = 0;
    for (int64_t u = 0; u < u_rows; ++u)
        for (int64_t j = head[u]; j >= 0; j = next[j])
            out->origin_raw[w++] = (int32_t)j;
    if (w != out->n_origins)
        sag_die("dedup: origin flatten wrote %" PRId64 " of %" PRId64, w, out->n_origins);
    sag_arena_release(a, mark);
}

/* ===================================================================== */
/* 5. Blocked filter -- derivation of the unblocked list                  */
/* ===================================================================== */

void sag_filter_blocked(sag_arena *a, const sag_dedup_result *in,
                        const uint64_t *blocked, int64_t n_blocked, int64_t words,
                        sag_dedup_result *out, int64_t *blocked_count)
{
    const int64_t u = in->unique.rows;
    if (u == 0 || n_blocked <= 0) {
        *out = *in;
        if (blocked_count) *blocked_count = 0;
        return;
    }
    if (words != in->unique.words)
        sag_die("filter_blocked: words %" PRId64 " != batch words %" PRId64,
                words, in->unique.words);
    if (!blocked)
        sag_die("filter_blocked: n_blocked=%" PRId64 " but blocked pointer is NULL", n_blocked);

    bool *keep = SAG_NEW(a, bool, u);
    const size_t mark = sag_arena_mark(a);
    sag_mask_set set = mask_set_make(a, n_blocked, blocked, words);
    for (int64_t i = 0; i < n_blocked; ++i) {
        const uint64_t *m = blocked + i * words;
        const uint64_t h = sag_bitset_hash(m, words);
        int64_t slot = -1;
        if (mask_set_find(&set, m, h, &slot) < 0)
            mask_set_put(&set, slot, i, h);
    }
    int64_t kept = 0, kept_nnz = 0, kept_origins = 0;
    for (int64_t i = 0; i < u; ++i) {
        const uint64_t *m = sag_batch_mask(&in->unique, i);
        keep[i] = mask_set_find(&set, m, sag_bitset_hash(m, words), NULL) < 0;
        if (keep[i]) {
            kept += 1;
            kept_nnz += sag_batch_size(&in->unique, i);
            kept_origins += in->origin_indptr[i + 1] - in->origin_indptr[i];
        }
    }
    sag_arena_release(a, mark);
    if (kept == u) {
        *out = *in;
        if (blocked_count) *blocked_count = 0;
        return;
    }

    sag_dedup_result res;
    res.unique = sag_batch_alloc(a, in->unique.n_nodes, kept, kept_nnz);
    res.origin_indptr = SAG_NEW0(a, int64_t, kept + 1);
    res.origin_raw = SAG_NEW(a, int32_t, kept_origins > 0 ? kept_origins : 1);
    res.n_origins = kept_origins;
    res.raw_count = in->raw_count;
    int64_t w = 0;
    for (int64_t i = 0; i < u; ++i) {
        if (!keep[i]) continue;
        const int64_t r = res.unique.rows;
        push_row_from(&res.unique, &in->unique, i);
        for (int64_t k = in->origin_indptr[i]; k < in->origin_indptr[i + 1]; ++k)
            res.origin_raw[w++] = in->origin_raw[k];
        res.origin_indptr[r + 1] = w;
    }
    if (w != kept_origins)
        sag_die("filter_blocked: origin copy wrote %" PRId64 " of %" PRId64, w, kept_origins);
    *out = res;
    if (blocked_count) *blocked_count = u - kept;
}

/* ===================================================================== */
/* 6. Authoritative scoring                                               */
/* ===================================================================== */

void sag_score(sag_arena *a, const sag_candidate_batch *pool,
               const sag_candidate_ctx *ctx, sag_scored_batch *out)
{
    const int64_t n = ctx->n_nodes;
    if (pool->n_nodes != n || ctx->lp->n_nodes != n)
        sag_die("score: n_nodes mismatch (batch=%" PRId64 ", ctx=%" PRId64 ", lp=%" PRId64 ")",
                pool->n_nodes, n, ctx->lp->n_nodes);
    const int64_t b = pool->rows;
    const int64_t alloc = b > 0 ? b : 1;
    out->rows = b;
    out->rhs = SAG_NEW(a, double, alloc);
    out->boundary = SAG_NEW(a, double, alloc);
    out->violation = SAG_NEW(a, double, alloc);
    out->row_nnz = SAG_NEW(a, int64_t, alloc);
    out->coef_l2 = SAG_NEW(a, double, alloc);
    if (b == 0) return;

    for (int64_t i = 0; i < b; ++i)
        if (sag_batch_size(pool, i) <= 0)
            sag_die("score: candidate keys must contain only original 1-based customer ids "
                    "(row %" PRId64 " is empty)", i);
    for (int64_t k = 0; k < pool->nnz; ++k)
        if (pool->indices[k] < 1 || (int64_t)pool->indices[k] >= n)
            sag_die("score: candidate keys must contain only original 1-based customer ids "
                    "(id %" PRId32 " at nnz %" PRId64 ")", pool->indices[k], k);
    if (ctx->capacity <= 0)
        sag_die("score: capacity must be positive (int64 semantics), got %" PRId64,
                ctx->capacity);

    const sag_lp_support *lp = ctx->lp;
    const int64_t m = lp->n_edges;
    for (int64_t e = 0; e < m; ++e) {
        if (lp->edge_u[e] < 0 || lp->edge_u[e] >= n || lp->edge_v[e] < 0 || lp->edge_v[e] >= n)
            sag_die("score: lp_support edge %" PRId64 " has endpoints (%" PRId64 ",%" PRId64
                    ") outside [0,%" PRId64 ")", e, lp->edge_u[e], lp->edge_v[e], n);
    }
    const size_t mark = sag_arena_mark(a);
    double *cross = SAG_NEW(a, double, m > 0 ? m : 1);

    const int64_t n_cust = n - 1;
    const int64_t q = ctx->capacity;
    for (int64_t i = 0; i < b; ++i) {
        const uint64_t *mask = sag_batch_mask(pool, i);

        int64_t cnt = 0;
        for (int64_t e = 0; e < m; ++e) {
            const int su = sag_bit_test(mask, lp->edge_u[e]);
            const int sv = sag_bit_test(mask, lp->edge_v[e]);
            if (su != sv) cross[cnt++] = lp->edge_x[e];
        }
        const double boundary = sag_f64_sum_numpy(cross, cnt);

        int64_t dsum = 0;
        int64_t lo = pool->indptr[i], hi = pool->indptr[i + 1];
        for (int64_t k = lo; k < hi; ++k) dsum += ctx->demand[pool->indices[k]];
        const double rhs = (double)((dsum + q - 1) / q);

        const int64_t s = hi - lo;
        if (s > n_cust)
            sag_die("score: subset size %" PRId64 " exceeds customer count %" PRId64, s, n_cust);
        const bool small = (2 * s <= n_cust) || (s == n_cust);
        const int64_t comb_s = s * (s - 1) / 2;
        const int64_t comp = n_cust - s;
        const int64_t comb_c = comp * (comp - 1) / 2;
        const int64_t row_nnz = small ? comb_s : comb_c + n_cust;
        const double l2_sq = small ? (double)row_nnz
                                   : ((double)comb_c + 0.25 * (double)n_cust);

        out->boundary[i] = boundary;
        out->rhs[i] = rhs;
        out->violation[i] = 2.0 * rhs - boundary;
        out->row_nnz[i] = row_nnz;
        out->coef_l2[i] = sqrt(l2_sq);
    }
    sag_arena_release(a, mark);
}

/* ===================================================================== */
/* 7. Selector -- passthrough selection + budget validation               */
/* ===================================================================== */

void sag_select(sag_arena *a, const sag_scored_batch *scored,
                const sag_selection_budget *budget, sag_selection *out)
{
    const int64_t b = scored->rows;
    out->selected = SAG_NEW(a, int64_t, b > 0 ? b : 1);
    out->count = 0;
    for (int64_t i = 0; i < b; ++i)
        if (scored->violation[i] > 0.0) out->selected[out->count++] = i;
    out->nonviolated = b - out->count;

    if (budget) {
        if (budget->max_rows >= 0 && out->count > budget->max_rows)
            sag_die("select: selector exceeded max_rows budget (%" PRId64 " > %" PRId64 ")",
                    out->count, budget->max_rows);
        if (budget->max_nnz >= 0) {
            int64_t nnz = 0;
            for (int64_t k = 0; k < out->count; ++k) nnz += scored->row_nnz[out->selected[k]];
            if (nnz > budget->max_nnz)
                sag_die("select: selector exceeded max_nnz budget (%" PRId64 " > %" PRId64 ")",
                        nnz, budget->max_nnz);
        }
    }
}

/* ===================================================================== */
/* 8. materialize -- the tail section of the decode pipeline              */
/* ===================================================================== */

void sag_materialize(sag_arena *a, const sag_candidate_batch *pool,
                     const sag_scored_batch *scored, const sag_selection *sel,
                     bool compute_cut_values, sag_materialized_cuts *out)
{
    if (scored->rows != pool->rows)
        sag_die("materialize: scored rows %" PRId64 " != pool rows %" PRId64,
                scored->rows, pool->rows);
    const int64_t s = sel->count;
    int64_t nnz = 0;
    for (int64_t k = 0; k < s; ++k) {
        const int64_t r = sel->selected[k];
        if (r < 0 || r >= pool->rows)
            sag_die("materialize: selected row %" PRId64 " out of range [0, %" PRId64 ")",
                    r, pool->rows);
        nnz += sag_batch_size(pool, r);
    }
    out->rows = s;
    out->nnz = nnz;
    out->indptr = SAG_NEW(a, int64_t, s + 1);
    out->indices = SAG_NEW(a, int32_t, nnz > 0 ? nnz : 1);
    out->rhs = SAG_NEW(a, double, s > 0 ? s : 1);
    out->sizes = SAG_NEW(a, int64_t, s > 0 ? s : 1);
    out->pool_rows = SAG_NEW(a, int64_t, s > 0 ? s : 1);
    out->n_cut_values = compute_cut_values ? s : 0;
    out->cut_values = SAG_NEW(a, double, out->n_cut_values > 0 ? out->n_cut_values : 1);

    out->indptr[0] = 0;
    int64_t w = 0;
    for (int64_t k = 0; k < s; ++k) {
        const int64_t r = sel->selected[k];
        int64_t len = 0;
        const int32_t *idx = sag_batch_row(pool, r, &len);
        for (int64_t t = 0; t < len; ++t) out->indices[w++] = idx[t] + 1;
        out->indptr[k + 1] = w;
        out->sizes[k] = len;
        out->rhs[k] = scored->rhs[r];
        out->pool_rows[k] = r;
        if (compute_cut_values) out->cut_values[k] = scored->boundary[r];
    }
}

/* ===================================================================== */
/* 9. families output order + orchestration                               */
/* ===================================================================== */

int64_t sag_spine_family_order(const sag_candidate_batch *raw, const bool *has_elapsed,
                               int8_t *out)
{
    static const bool frozen[SAG_FAMILY_COUNT] = { true, true, false };
    const bool *keys = has_elapsed ? has_elapsed : frozen;

    bool seen[SAG_FAMILY_COUNT] = { false, false, false };
    int64_t count = 0;
    for (int64_t i = 0; i < raw->rows; ++i) {
        const int8_t f = raw->family[i];
        if (f < 0 || f >= SAG_FAMILY_COUNT)
            sag_die("family_order: raw row %" PRId64 " has family code %d", i, (int)f);
        if (!seen[f]) { seen[f] = true; out[count++] = f; }
    }
    for (int8_t f = 0; f < SAG_FAMILY_COUNT; ++f)
        if (keys[f] && !seen[f]) { seen[f] = true; out[count++] = f; }
    return count;
}

void sag_run_spine(sag_arena *a, const sag_candidate_ctx *ctx,
                   const sag_candidate_batch *sweep, const sag_candidate_batch *onion,
                   const sag_candidate_batch *tts,
                   const uint64_t *blocked, int64_t n_blocked,
                   const sag_selection_budget *budget,
                   const double *family_elapsed,
                   bool compute_cut_values,
                   sag_spine_result *out)
{
    memset(out, 0, sizeof *out);
    sag_assemble(a, sweep, onion, tts, ctx->vehicle_count, &out->raw);

    const double t0 = now_seconds();
    sag_dedup(a, &out->raw, &out->dedup);
    const double t1 = now_seconds();
    int64_t blocked_count = 0;
    sag_filter_blocked(a, &out->dedup, blocked, n_blocked, out->raw.words,
                       &out->pool, &blocked_count);
    const double t2 = now_seconds();
    sag_score(a, &out->pool.unique, ctx, &out->scored);
    const double t3 = now_seconds();
    sag_select(a, &out->scored, budget, &out->sel);
    const double t4 = now_seconds();
    sag_materialize(a, &out->pool.unique, &out->scored, &out->sel,
                    compute_cut_values, &out->cuts);
    const double t5 = now_seconds();

    sag_pipeline_diag *d = &out->diag;
    d->raw = out->raw.rows;
    d->exact_unique = out->dedup.unique.rows;
    d->blocked = blocked_count;
    d->nonviolated = out->sel.nonviolated;
    d->selector_pool = out->pool.unique.rows - out->sel.nonviolated;
    d->selected = out->sel.count;
    d->materialized = out->cuts.rows;
    d->added_to_lp = 0;
    d->budget = budget ? *budget : (sag_selection_budget){ -1, -1 };

    /* ---- families diagnostics (item by item, as in the reference) ---- */
    const size_t mark = sag_arena_mark(a);
    if (out->raw.rows > 0) {
        uint8_t *fam_bits = SAG_NEW0(a, uint8_t, out->raw.rows);
        int64_t *rep = SAG_NEW(a, int64_t, out->raw.rows);
        int64_t n_rep = 0;
        sag_mask_set set = mask_set_make(a, out->raw.rows, out->raw.mask, out->raw.words);
        for (int64_t i = 0; i < out->raw.rows; ++i) {
            const uint64_t *m = sag_batch_mask(&out->raw, i);
            const uint64_t h = sag_bitset_hash(m, out->raw.words);
            int64_t slot = -1;
            int64_t r = mask_set_find(&set, m, h, &slot);
            if (r < 0) { r = i; mask_set_put(&set, slot, i, h); rep[n_rep++] = i; }
            fam_bits[r] |= (uint8_t)(1u << out->raw.family[i]);
            d->families[out->raw.family[i]].raw += 1;
        }
        for (int64_t k = 0; k < n_rep; ++k)
            for (int8_t f = 0; f < SAG_FAMILY_COUNT; ++f)
                if (fam_bits[rep[k]] & (uint8_t)(1u << f))
                    d->families[f].within_family_unique += 1;
    }
    for (int64_t i = 0; i < out->dedup.unique.rows; ++i)
        d->families[out->dedup.unique.family[i]].contributed_global_unique += 1;
    if (out->pool.unique.rows > 0) {
        uint8_t *pool_bits = SAG_NEW0(a, uint8_t, out->pool.unique.rows);
        for (int64_t i = 0; i < out->pool.unique.rows; ++i)
            for (int64_t k = out->pool.origin_indptr[i]; k < out->pool.origin_indptr[i + 1]; ++k)
                pool_bits[i] |= (uint8_t)(1u << out->raw.family[out->pool.origin_raw[k]]);
        for (int64_t k = 0; k < out->sel.count; ++k) {
            const uint8_t bits = pool_bits[out->sel.selected[k]];
            for (int8_t f = 0; f < SAG_FAMILY_COUNT; ++f)
                if (bits & (uint8_t)(1u << f)) d->families[f].selected_with_origin += 1;
        }
    }
    sag_arena_release(a, mark);

    double wall = 0.0;
    for (int64_t f = 0; f < SAG_FAMILY_COUNT; ++f) {
        const double e = family_elapsed ? family_elapsed[f] : 0.0;
        d->families[f].cpu_elapsed = e;
        wall += e;
    }
    d->timings[SAG_PIPE_T_FAMILY_DECODE_WALL] = wall;
    d->timings[SAG_PIPE_T_DEDUP] = t1 - t0;
    d->timings[SAG_PIPE_T_SCORING] = t3 - t2;
    d->timings[SAG_PIPE_T_SELECTOR] = t4 - t3;
    d->timings[SAG_PIPE_T_MATERIALIZE] = t5 - t4;

    out->family_count = sag_spine_family_order(&out->raw, NULL, out->family_order);
}
