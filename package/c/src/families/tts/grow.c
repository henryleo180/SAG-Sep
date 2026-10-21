#include "families/tts/grow.h"

#include <math.h>
#include <string.h>

#include "bitset.h"

#if !defined(SAG_FP_CONTRACT_OFF)
#error "families/tts/grow.c must be compiled with -ffp-contract=off (the Makefile's \
FPFLAGS also define -DSAG_FP_CONTRACT_OFF=1): bit-for-bit parity with the reference \
argmax depends on the floating-point chain not being contracted"
#endif

/* ------------------------------------------------------------------ *
 * Open-addressing table keyed by mask
 * ------------------------------------------------------------------ */

typedef struct {
    uint64_t *keys;
    uint64_t *vals;
    uint8_t  *used;
    int64_t   cap;
    int64_t   words;
    int64_t   count;
} sag_grow_map;

static int64_t sag_grow_pow2_at_least(int64_t v)
{
    int64_t p = 8;
    while (p < v) {
        p <<= 1;
        if (p <= 0) sag_die("lp_grow: hash capacity overflow (requested %lld)", (long long)v);
    }
    return p;
}

static void sag_grow_map_init(sag_grow_map *map, sag_arena *a, int64_t entries,
                              int64_t words, int with_vals)
{
    map->cap = sag_grow_pow2_at_least(entries * 2 + 8);
    map->words = words;
    map->count = 0;
    map->keys = SAG_NEW(a, uint64_t, sag_size_mul((size_t)map->cap, (size_t)words));
    map->vals = with_vals
                    ? SAG_NEW0(a, uint64_t, sag_size_mul((size_t)map->cap, (size_t)words))
                    : NULL;
    map->used = SAG_NEW0(a, uint8_t, map->cap);
}

static int64_t sag_grow_map_slot(const sag_grow_map *map, const uint64_t *key)
{
    const uint64_t mask = (uint64_t)map->cap - 1u;
    uint64_t slot = sag_bitset_hash(key, map->words) & mask;
    for (;;) {
        if (!map->used[slot]) return (int64_t)slot;
        if (sag_bitset_equal(map->keys + (int64_t)slot * map->words, key, map->words))
            return (int64_t)slot;
        slot = (slot + 1u) & mask;
    }
}

static int sag_grow_map_insert(sag_grow_map *map, const uint64_t *key)
{
    int64_t slot = sag_grow_map_slot(map, key);
    if (map->used[slot]) return 0;
    map->used[slot] = 1;
    sag_bitset_copy(map->keys + slot * map->words, key, map->words);
    map->count++;
    if (map->count * 2 > map->cap)
        sag_die("lp_grow: hash table overloaded (%lld/%lld) -- the reserved size was wrong",
                (long long)map->count, (long long)map->cap);
    return 1;
}

static void sag_grow_map_add_bit(sag_grow_map *map, const uint64_t *key, int64_t bit)
{
    int64_t slot = sag_grow_map_slot(map, key);
    if (!map->used[slot]) {
        map->used[slot] = 1;
        sag_bitset_copy(map->keys + slot * map->words, key, map->words);
        sag_bitset_zero(map->vals + slot * map->words, map->words);
        map->count++;
        if (map->count * 2 > map->cap)
            sag_die("lp_grow: anti_extensions overloaded (%lld/%lld)",
                    (long long)map->count, (long long)map->cap);
    }
    sag_bit_set(map->vals + slot * map->words, bit);
}

static const uint64_t *sag_grow_map_get(const sag_grow_map *map, const uint64_t *key)
{
    int64_t slot = sag_grow_map_slot(map, key);
    if (!map->used[slot]) return NULL;
    return map->vals + slot * map->words;
}

/* ------------------------------------------------------------------ *
 * Assembly helper
 * ------------------------------------------------------------------ */

void sag_tts_grow_adj_from_aggregate(const float *aggregate, int64_t n_super, double *out)
{
    if (n_super <= 1) return;
    const int64_t m = n_super - 1;
    for (int64_t i = 0; i < m; i++) {
        const float *src = aggregate + (i + 1) * n_super + 1;
        double *dst = out + i * m;
        for (int64_t j = 0; j < m; j++) dst[j] = (double)src[j];
        dst[i] = 0.0;
    }
}

/* ------------------------------------------------------------------ *
 * Growth state / environment
 * ------------------------------------------------------------------ */

typedef struct {
    uint8_t  *in_set;
    double   *connectivity;
    uint64_t *cur;
    int64_t  *order;
    int64_t   n_order;
    double    demand_sum;
    int64_t   original_size;
    double    x_in_set;
    int64_t   vehicles;
    uint64_t *src_out;
    int64_t   n_src_out;
    int64_t  *emit_at;
    int64_t  *cum;
    int64_t   previous_size;
    int64_t   previous_vehicles;
} grow_state;

typedef struct {
    const double  *adj;
    const double  *demand;
    const int64_t *sizes;
    const double  *internal_x;
    int64_t        m;
    int64_t        words;
    int64_t        path_cap;
    double         capacity;
    double         min_violation;
    int64_t        max_prefix;
    const sag_grow_map *anti;
    double        *viol;
    int64_t       *cand;
} grow_env;

/* ------------------------------------------------------------------ *
 * Single-step state advance
 * ------------------------------------------------------------------ */

static void grow_apply_move(grow_state *st, const grow_env *e,
                            int64_t chosen, double chosen_violation,
                            int64_t *pair_total, int64_t *pair_le3)
{
    st->in_set[chosen] = 1;
    sag_bit_set(st->cur, chosen);
    st->order[st->n_order] = chosen;
    st->cum[st->n_order + 1] = st->cum[st->n_order] + e->sizes[chosen];
    st->n_order += 1;
    st->demand_sum += e->demand[chosen];
    st->original_size += e->sizes[chosen];
    st->x_in_set += st->connectivity[chosen] + e->internal_x[chosen];
    st->vehicles = (int64_t)ceil(st->demand_sum / e->capacity);
    {
        const double *row = e->adj + chosen * e->m;
        for (int64_t j = 0; j < e->m; j++) st->connectivity[j] += row[j];
    }

    if (chosen_violation > e->min_violation) {
        if (e->sizes[chosen] == 1
            && st->n_order == st->previous_size + 1
            && st->vehicles == st->previous_vehicles + 1
            && st->n_src_out > 0) {
            st->n_src_out--;
        }
        if (st->n_src_out >= e->path_cap)
            sag_die("lp_grow: single-seed outputs exceed the upper bound (%lld)", (long long)e->path_cap);
        sag_bitset_copy(st->src_out + st->n_src_out * e->words, st->cur, e->words);
        if (st->n_src_out > 0 && pair_total) {
            *pair_total += 1;
            if (st->cum[st->n_order] - st->cum[st->emit_at[st->n_src_out - 1]] <= 3)
                *pair_le3 += 1;
        }
        st->emit_at[st->n_src_out] = st->n_order;
        st->n_src_out++;
        st->previous_size = st->n_order;
        st->previous_vehicles = st->vehicles;
    }
}

/* ------------------------------------------------------------------ *
 * Greedy growth main loop
 * ------------------------------------------------------------------ */
static void grow_walk(grow_state *st, const grow_env *e,
                      int64_t *pair_total, int64_t *pair_le3)
{
    for (;;) {
        if (e->max_prefix > 0 && e->max_prefix <= st->n_order) break;

        int64_t nc = 0;
        for (int64_t j = 0; j < e->m; j++)
            if (!st->in_set[j] && st->connectivity[j] > 0.0) e->cand[nc++] = j;
        if (nc == 0) break;

        const uint64_t *blocked = sag_grow_map_get(e->anti, st->cur);
        if (blocked) {
            int64_t keep = 0;
            for (int64_t i = 0; i < nc; i++)
                if (!sag_bit_test(blocked, e->cand[i])) e->cand[keep++] = e->cand[i];
            nc = keep;
            if (nc == 0) break;
        }

        for (int64_t i = 0; i < nc; i++) {
            const int64_t c = e->cand[i];
            const double  proposed_demand   = st->demand_sum + e->demand[c];
            const int64_t proposed_vehicles = (int64_t)ceil(proposed_demand / e->capacity);
            const int64_t proposed_size     = st->original_size + e->sizes[c];
            const double proposed_x = (st->x_in_set + st->connectivity[c]) + e->internal_x[c];
            e->viol[i] = 2.0 * ((proposed_x - (double)proposed_size)
                                + (double)proposed_vehicles);
        }

        int pref = 0;
        for (int64_t i = 0; i < nc; i++)
            if (e->sizes[e->cand[i]] == 1 && e->viol[i] > e->min_violation) { pref = 1; break; }

        int64_t local_index = -1;
        double  best = 0.0;
        for (int64_t i = 0; i < nc; i++) {
            if (pref && !(e->sizes[e->cand[i]] == 1 && e->viol[i] > e->min_violation))
                continue;
            if (local_index < 0) {
                local_index = i;
                best = e->viol[i];
                if (isnan(best)) break;
                continue;
            }
            if (!(e->viol[i] <= best)) {
                local_index = i;
                best = e->viol[i];
                if (isnan(best)) break;
            }
        }

        const int64_t chosen = e->cand[local_index];
        const double  chosen_violation = e->viol[local_index];

        grow_apply_move(st, e, chosen, chosen_violation, pair_total, pair_le3);
    }
}

/* ------------------------------------------------------------------ *
 * Main body
 * ------------------------------------------------------------------ */

void sag_tts_lp_grow(sag_arena *a,
                     const double  *adj,
                     const double  *demand,
                     const int64_t *sizes,
                     const double  *internal_x,
                     int64_t        m,
                     double         capacity,
                     int64_t        n_seeds,
                     sag_pcg64     *rng,
                     double         min_violation,
                     int64_t        max_prefix,
                     sag_tts_grow_out *out)
{
    const int64_t words = (m > 0) ? sag_mask_words(m) : 0;
    out->masks = (uint64_t *)sag_arena_alloc(a, 0, _Alignof(uint64_t));
    out->rows = 0;
    out->words = words;
    out->n_cust = m;
    out->seed_id = out->seed_raw = out->seed_unique = out->seed_steps = NULL;
    out->seed_pair_le3 = out->seed_pairs = NULL;
    out->row_seed = NULL;
    out->n_seeds_used = 0;
    out->n_seed_early = 0;

    if (m <= 0 || n_seeds <= 0 || capacity <= 0.0) return;

    for (int64_t i = 0; i < m; i++) {
        if (adj[i * m + i] != 0.0)
            sag_die("lp_grow: adj has a non-zero diagonal (i=%lld, %.17g) -- the matrix must have had fill_diagonal(0) applied",
                    (long long)i, adj[i * m + i]);
    }

    const int64_t path_cap = (max_prefix > 0 && max_prefix < m) ? max_prefix : m;
    const int64_t n_loop = (n_seeds >= m) ? m : n_seeds;
    const int64_t cap_rows = n_loop * path_cap;

    out->masks = SAG_NEW(a, uint64_t, sag_size_mul((size_t)cap_rows, (size_t)words));
    out->seed_id     = SAG_NEW0(a, int64_t, n_loop);
    out->seed_raw    = SAG_NEW0(a, int64_t, n_loop);
    out->seed_unique = SAG_NEW0(a, int64_t, n_loop);
    out->seed_steps  = SAG_NEW0(a, int64_t, n_loop);
    out->seed_pair_le3 = SAG_NEW0(a, int64_t, n_loop);
    out->seed_pairs    = SAG_NEW0(a, int64_t, n_loop);
    out->row_seed      = SAG_NEW0(a, int32_t, cap_rows);
    out->n_seeds_used = n_loop;

    const size_t mark = sag_arena_mark(a);

    int64_t *seeds = SAG_NEW(a, int64_t, n_loop);
    if (n_seeds >= m) {
        for (int64_t i = 0; i < m; i++) seeds[i] = i;
    } else {
        if (!rng) sag_die("lp_grow: n_seeds(%lld) < n_cust(%lld) needs an RNG, but NULL was passed",
                          (long long)n_seeds, (long long)m);
        sag_rng_choice(rng, m, n_seeds, seeds);
    }

    uint8_t  *blocked_sources = SAG_NEW0(a, uint8_t, m);
    uint64_t *prefix          = SAG_NEW(a, uint64_t, words);

    grow_state st;
    st.in_set       = SAG_NEW(a, uint8_t, m);
    st.connectivity = SAG_NEW(a, double, m);
    st.cur          = SAG_NEW(a, uint64_t, words);
    st.order        = SAG_NEW(a, int64_t, path_cap);
    st.src_out      = SAG_NEW(a, uint64_t, sag_size_mul((size_t)path_cap, (size_t)words));
    st.emit_at      = SAG_NEW(a, int64_t, path_cap);
    st.cum          = SAG_NEW(a, int64_t, path_cap + 1);

    sag_grow_map anti, okeys;
    sag_grow_map_init(&anti, a, n_loop * path_cap, words, 1);
    sag_grow_map_init(&okeys, a, cap_rows, words, 0);

    grow_env env;
    env.adj = adj;                 env.demand = demand;
    env.sizes = sizes;             env.internal_x = internal_x;
    env.m = m;                     env.words = words;
    env.path_cap = path_cap;       env.capacity = capacity;
    env.min_violation = min_violation;
    env.max_prefix = max_prefix;
    env.anti = &anti;
    env.viol = SAG_NEW(a, double, m);
    env.cand = SAG_NEW(a, int64_t, m);

    for (int64_t si = 0; si < n_loop; si++) {
        const int64_t seed = seeds[si];
        if (seed < 0 || seed >= m)
            sag_die("lp_grow: seed out of range (%lld / %lld)", (long long)seed, (long long)m);
        out->seed_id[si] = seed;
        if (blocked_sources[seed]) continue;

        memset(st.in_set, 0, (size_t)m);
        st.in_set[seed] = 1;
        sag_bitset_zero(st.cur, words);
        sag_bit_set(st.cur, seed);
        st.order[0] = seed;
        st.n_order = 1;
        st.cum[0] = 0;
        st.cum[1] = sizes[seed];
        memcpy(st.connectivity, adj + seed * m, (size_t)m * sizeof(double));
        st.demand_sum        = demand[seed];
        st.original_size     = sizes[seed];
        st.x_in_set          = internal_x[seed];
        st.vehicles          = (int64_t)ceil(st.demand_sum / capacity);
        st.n_src_out         = 0;
        st.previous_size     = -1;
        st.previous_vehicles = -1;

        const double seed_violation =
            2.0 * ((st.x_in_set - (double)st.original_size) + (double)st.vehicles);
        if (seed_violation > min_violation) {
            sag_bitset_copy(st.src_out, st.cur, words);
            st.emit_at[0] = st.n_order;
            st.n_src_out = 1;
            out->n_seed_early += 1;
        } else {
            grow_walk(&st, &env, &out->seed_pairs[si], &out->seed_pair_le3[si]);

            /* ---- anti-set / blocked_sources writeback ---- */
            if (st.n_order == 1) {
                blocked_sources[st.order[0]] = 1;
            } else {
                sag_bitset_zero(prefix, words);
                sag_bit_set(prefix, st.order[0]);
                for (int64_t k = 1; k < st.n_order; k++) {
                    sag_grow_map_add_bit(&anti, prefix, st.order[k]);
                    sag_bit_set(prefix, st.order[k]);
                }
            }
        }

        out->seed_raw[si]   = st.n_src_out;
        out->seed_steps[si] = st.n_order - 1;
        for (int64_t k = 0; k < st.n_src_out; k++) {
            const uint64_t *row = st.src_out + k * words;
            if (!sag_grow_map_insert(&okeys, row)) continue;
            if (out->rows >= cap_rows)
                sag_die("lp_grow: output rows exceed the upper bound (%lld)", (long long)cap_rows);
            sag_bitset_copy(out->masks + out->rows * words, row, words);
            out->row_seed[out->rows] = (int32_t)si;
            out->rows++;
            out->seed_unique[si] += 1;
        }
    }

    sag_arena_release(a, mark);
}
