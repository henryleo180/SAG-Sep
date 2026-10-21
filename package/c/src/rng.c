#include "rng.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "arena.h"

/* =================================================================== *
 * L1  payload
 * =================================================================== */

static size_t payload_put(char *out, size_t cap, size_t pos, const void *src, size_t n,
                          const char *what)
{
    if (pos + n + 1u > cap)
        sag_die("sag_rng_payload: buffer too small (cap=%zu, need>%zu) while appending %s",
                cap, pos + n, what);
    memcpy(out + pos, src, n);
    return pos + n;
}

static size_t payload_put_i64(char *out, size_t cap, size_t pos, int64_t value,
                              const char *what)
{
    char tmp[24];
    int m = snprintf(tmp, sizeof tmp, "%" PRId64, value);
    if (m < 0 || (size_t)m >= sizeof tmp)
        sag_die("sag_rng_payload: cannot format %s = %" PRId64, what, value);
    return payload_put(out, cap, pos, tmp, (size_t)m, what);
}

size_t sag_rng_payload(char *out, size_t cap, int64_t base_seed, const char *instance,
                       int64_t iteration, int64_t vehicle, const char *source)
{
    if (!out || cap == 0) sag_die("sag_rng_payload: null/zero output buffer");
    if (!instance) sag_die("sag_rng_payload: instance is NULL (must be the instance name verbatim)");
    if (!source) sag_die("sag_rng_payload: source is NULL");

    static const char nul = '\0';
    size_t pos = 0;
    pos = payload_put_i64(out, cap, pos, base_seed, "base_seed");
    pos = payload_put(out, cap, pos, &nul, 1, "sep1");
    pos = payload_put(out, cap, pos, instance, strlen(instance), "instance");
    pos = payload_put(out, cap, pos, &nul, 1, "sep2");
    pos = payload_put_i64(out, cap, pos, iteration, "iteration");
    pos = payload_put(out, cap, pos, &nul, 1, "sep3");
    pos = payload_put_i64(out, cap, pos, vehicle, "vehicle");
    pos = payload_put(out, cap, pos, &nul, 1, "sep4");
    pos = payload_put(out, cap, pos, source, strlen(source), "source");
    out[pos] = '\0';
    return pos;
}

/* =================================================================== *
 * L1  BLAKE2b (RFC 7693, self-contained, no external dependency)
 * =================================================================== */

static const uint64_t BLAKE2B_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL,
    0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

static const uint8_t BLAKE2B_SIGMA[12][16] = {
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 },
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 }
};

static uint64_t load64_le(const uint8_t *p)
{
    return (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16)
         | ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40)
         | ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

static void store64_le(uint8_t *p, uint64_t v)
{
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i));
}

static uint64_t rotr64(uint64_t x, unsigned r)
{
    return (x >> r) | (x << ((64u - r) & 63u));
}

#define BLAKE2B_G(a, b, c, d, x, y)          \
    do {                                     \
        v[a] = v[a] + v[b] + (x);            \
        v[d] = rotr64(v[d] ^ v[a], 32);      \
        v[c] = v[c] + v[d];                  \
        v[b] = rotr64(v[b] ^ v[c], 24);      \
        v[a] = v[a] + v[b] + (y);            \
        v[d] = rotr64(v[d] ^ v[a], 16);      \
        v[c] = v[c] + v[d];                  \
        v[b] = rotr64(v[b] ^ v[c], 63);      \
    } while (0)

static void blake2b_compress(uint64_t h[8], const uint8_t block[SAG_BLAKE2B_BLOCKBYTES],
                             uint64_t t_lo, uint64_t t_hi, int last)
{
    uint64_t v[16], m[16];
    for (int i = 0; i < 16; ++i) m[i] = load64_le(block + 8 * i);
    for (int i = 0; i < 8; ++i) v[i] = h[i];
    for (int i = 0; i < 8; ++i) v[i + 8] = BLAKE2B_IV[i];
    v[12] ^= t_lo;
    v[13] ^= t_hi;
    if (last) v[14] = ~v[14];

    for (int r = 0; r < 12; ++r) {
        const uint8_t *s = BLAKE2B_SIGMA[r];
        BLAKE2B_G(0, 4,  8, 12, m[s[ 0]], m[s[ 1]]);
        BLAKE2B_G(1, 5,  9, 13, m[s[ 2]], m[s[ 3]]);
        BLAKE2B_G(2, 6, 10, 14, m[s[ 4]], m[s[ 5]]);
        BLAKE2B_G(3, 7, 11, 15, m[s[ 6]], m[s[ 7]]);
        BLAKE2B_G(0, 5, 10, 15, m[s[ 8]], m[s[ 9]]);
        BLAKE2B_G(1, 6, 11, 12, m[s[10]], m[s[11]]);
        BLAKE2B_G(2, 7,  8, 13, m[s[12]], m[s[13]]);
        BLAKE2B_G(3, 4,  9, 14, m[s[14]], m[s[15]]);
    }
    for (int i = 0; i < 8; ++i) h[i] ^= v[i] ^ v[i + 8];
}

void sag_blake2b_128(const void *data, size_t len, uint8_t digest[16])
{
    uint64_t h[8];
    for (int i = 0; i < 8; ++i) h[i] = BLAKE2B_IV[i];
    h[0] ^= 0x01010000ULL ^ (uint64_t)SAG_BLAKE2B_OUTLEN;

    const uint8_t *p = (const uint8_t *)data;
    size_t remaining = len;
    uint64_t t = 0;

    while (remaining > (size_t)SAG_BLAKE2B_BLOCKBYTES) {
        t += SAG_BLAKE2B_BLOCKBYTES;
        blake2b_compress(h, p, t, 0, 0);
        p += SAG_BLAKE2B_BLOCKBYTES;
        remaining -= SAG_BLAKE2B_BLOCKBYTES;
    }
    uint8_t block[SAG_BLAKE2B_BLOCKBYTES];
    memset(block, 0, sizeof block);
    if (remaining) memcpy(block, p, remaining);
    t += (uint64_t)remaining;
    blake2b_compress(h, block, t, 0, 1);

    store64_le(digest + 0, h[0]);
    store64_le(digest + 8, h[1]);
}

void sag_rng_words_from_digest(const uint8_t digest[16], uint32_t words[4])
{
    for (int k = 0; k < 4; ++k) {
        words[k] = (uint32_t)digest[4 * k]
                 | ((uint32_t)digest[4 * k + 1] << 8)
                 | ((uint32_t)digest[4 * k + 2] << 16)
                 | ((uint32_t)digest[4 * k + 3] << 24);
    }
}

/* =================================================================== *
 * L3  numpy SeedSequence (hashmix / mix / generate_state from bit_generator.pyx)
 * =================================================================== */

#define SS_POOL_SIZE 4
#define SS_XSHIFT    16

static const uint32_t SS_INIT_A     = 0x43b0d7e5u;
static const uint32_t SS_MULT_A     = 0x931e8875u;
static const uint32_t SS_INIT_B     = 0x8b51f9ddu;
static const uint32_t SS_MULT_B     = 0x58f38dedu;
static const uint32_t SS_MIX_MULT_L = 0xca01f9ddu;
static const uint32_t SS_MIX_MULT_R = 0x4973f715u;

static uint32_t ss_hashmix(uint32_t value, uint32_t *hash_const)
{
    value ^= *hash_const;
    *hash_const *= SS_MULT_A;
    value *= *hash_const;
    value ^= value >> SS_XSHIFT;
    return value;
}

static uint32_t ss_mix(uint32_t x, uint32_t y)
{
    uint32_t result = SS_MIX_MULT_L * x - SS_MIX_MULT_R * y;
    result ^= result >> SS_XSHIFT;
    return result;
}

void sag_seed_sequence_u64(const uint32_t words[4], uint64_t state[4])
{
    uint32_t pool[SS_POOL_SIZE];
    uint32_t hash_const = SS_INIT_A;
    for (int i = 0; i < SS_POOL_SIZE; ++i) pool[i] = ss_hashmix(words[i], &hash_const);
    for (int i_src = 0; i_src < SS_POOL_SIZE; ++i_src)
        for (int i_dst = 0; i_dst < SS_POOL_SIZE; ++i_dst)
            if (i_src != i_dst)
                pool[i_dst] = ss_mix(pool[i_dst], ss_hashmix(pool[i_src], &hash_const));

    uint32_t out32[8];
    uint32_t hc = SS_INIT_B;
    for (int i = 0; i < 8; ++i) {
        uint32_t data_val = pool[i % SS_POOL_SIZE];
        data_val ^= hc;
        hc *= SS_MULT_B;
        data_val *= hc;
        data_val ^= data_val >> SS_XSHIFT;
        out32[i] = data_val;
    }
    for (int i = 0; i < 4; ++i)
        state[i] = (uint64_t)out32[2 * i] | ((uint64_t)out32[2 * i + 1] << 32);
}

/* =================================================================== *
 * L4  numpy PCG64 (XSL-RR 128/64)
 * =================================================================== */

#define SAG_PCG_MULT_HI 0x2360ED051FC65DA4ULL
#define SAG_PCG_MULT_LO 0x4385DF649FCCF645ULL

static void mul64_wide(uint64_t a, uint64_t b, uint64_t *hi, uint64_t *lo)
{
    const uint64_t M32 = 0xFFFFFFFFULL;
    uint64_t a0 = a & M32, a1 = a >> 32;
    uint64_t b0 = b & M32, b1 = b >> 32;
    uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint64_t mid = (p00 >> 32) + (p01 & M32) + (p10 & M32);
    *lo = (mid << 32) | (p00 & M32);
    *hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

static void pcg_step(sag_pcg64 *rng)
{
    uint64_t hi, lo;
    mul64_wide(rng->state_lo, SAG_PCG_MULT_LO, &hi, &lo);
    hi += rng->state_lo * SAG_PCG_MULT_HI + rng->state_hi * SAG_PCG_MULT_LO;
    uint64_t nlo = lo + rng->inc_lo;
    uint64_t carry = (nlo < lo) ? 1u : 0u;
    rng->state_lo = nlo;
    rng->state_hi = hi + rng->inc_hi + carry;
}

static uint64_t pcg_output(uint64_t hi, uint64_t lo)
{
    unsigned rot = (unsigned)(hi >> 58) & 63u;
    return rotr64(hi ^ lo, rot);
}

void sag_pcg64_seed(sag_pcg64 *rng, const uint64_t state_words[4])
{
    uint64_t initstate_hi = state_words[0], initstate_lo = state_words[1];
    uint64_t initseq_hi   = state_words[2], initseq_lo   = state_words[3];

    rng->state_hi = 0;
    rng->state_lo = 0;
    rng->inc_hi = (initseq_hi << 1) | (initseq_lo >> 63);
    rng->inc_lo = (initseq_lo << 1) | 1ULL;
    pcg_step(rng);
    uint64_t nlo = rng->state_lo + initstate_lo;
    uint64_t carry = (nlo < rng->state_lo) ? 1u : 0u;
    rng->state_lo = nlo;
    rng->state_hi += initstate_hi + carry;
    pcg_step(rng);

    rng->has_uint32 = 0;
    rng->uinteger = 0;
}

uint64_t sag_pcg64_next64(sag_pcg64 *rng)
{
    pcg_step(rng);
    return pcg_output(rng->state_hi, rng->state_lo);
}

uint32_t sag_pcg64_next32(sag_pcg64 *rng)
{
    if (rng->has_uint32) {
        rng->has_uint32 = 0;
        return rng->uinteger;
    }
    uint64_t next = sag_pcg64_next64(rng);
    rng->has_uint32 = 1;
    rng->uinteger = (uint32_t)(next >> 32);
    return (uint32_t)(next & 0xFFFFFFFFULL);
}

/* =================================================================== *
 * L5  bounded / choice
 * =================================================================== */

uint64_t sag_rng_bounded(sag_pcg64 *rng, uint64_t k)
{
    if (k == 0) return 0;
    if (k >= 0xFFFFFFFFULL)
        sag_die("sag_rng_bounded: k=%" PRIu64 " is outside the uint32 Lemire range; SAG-Sep's "
                "pop bound is ~1e3 so this is unreachable; uint64 Lemire is not implemented", k);

    const uint32_t rng_excl = (uint32_t)k + 1u;
    uint64_t m = (uint64_t)sag_pcg64_next32(rng) * (uint64_t)rng_excl;
    uint32_t leftover = (uint32_t)(m & 0xFFFFFFFFULL);
    if (leftover < rng_excl) {
        const uint32_t threshold = (0xFFFFFFFFu - (uint32_t)k) % rng_excl;
        while (leftover < threshold) {
            m = (uint64_t)sag_pcg64_next32(rng) * (uint64_t)rng_excl;
            leftover = (uint32_t)(m & 0xFFFFFFFFULL);
        }
    }
    return m >> 32;
}

static uint64_t gen_mask(uint64_t max_val)
{
    uint64_t mask = max_val;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;
    return mask;
}

#define SAG_RNG_CHOICE_SET_CAP 2048u

void sag_rng_choice_traced(sag_pcg64 *rng, int64_t pop, int64_t size, int64_t *out,
                           int64_t *trace_bound, int64_t *trace_val)
{
    if (!(size > 0 && size < pop))
        sag_die("sag_rng_choice: only 0 < size < pop is supported (n_seeds >= n_cust takes "
                "the arange path and consumes no RNG); got pop=%" PRId64 " size=%" PRId64, pop, size);
    if (size > SAG_RNG_CHOICE_MAX_SIZE)
        sag_die("sag_rng_choice: size=%" PRId64 " exceeds compile-time cap %d (frozen recipe lp_grow_seeds=16)",
                size, SAG_RNG_CHOICE_MAX_SIZE);

    uint64_t set_size = (uint64_t)(1.2 * (double)size);
    uint64_t mask = gen_mask(set_size);
    set_size = mask + 1;
    if (set_size > SAG_RNG_CHOICE_SET_CAP)
        sag_die("sag_rng_choice: hash set %" PRIu64 " > cap %u", set_size,
                (unsigned)SAG_RNG_CHOICE_SET_CAP);

    uint64_t hash_set[SAG_RNG_CHOICE_SET_CAP];
    for (uint64_t i = 0; i < set_size; ++i) hash_set[i] = UINT64_MAX;

    int64_t n_draw = 0;
    for (int64_t j = pop - size; j < pop; ++j) {
        uint64_t val = sag_rng_bounded(rng, (uint64_t)j);
        if (val > (uint64_t)j)
            sag_die("sag_rng_choice: bounded(%" PRId64 ") returned %" PRIu64 " (out of range)", j, val);
        if (trace_bound) trace_bound[n_draw] = j;
        if (trace_val) trace_val[n_draw] = (int64_t)val;
        ++n_draw;

        uint64_t loc = val & mask;
        while (hash_set[loc] != UINT64_MAX && hash_set[loc] != val) loc = (loc + 1) & mask;
        if (hash_set[loc] == UINT64_MAX) {
            hash_set[loc] = val;
            out[j - pop + size] = (int64_t)val;
        } else {
            loc = (uint64_t)j & mask;
            while (hash_set[loc] != UINT64_MAX) loc = (loc + 1) & mask;
            hash_set[loc] = (uint64_t)j;
            out[j - pop + size] = j;
        }
    }

    for (int64_t i = size - 1; i >= 1; --i) {
        int64_t j = (int64_t)sag_rng_bounded(rng, (uint64_t)i);
        if (j < 0 || j > i)
            sag_die("sag_rng_choice: shuffle bounded(%" PRId64 ") returned %" PRId64 " (out of range)", i, j);
        if (trace_bound) trace_bound[n_draw] = i;
        if (trace_val) trace_val[n_draw] = j;
        ++n_draw;
        int64_t tmp = out[j];
        out[j] = out[i];
        out[i] = tmp;
    }
}

void sag_rng_choice(sag_pcg64 *rng, int64_t pop, int64_t size, int64_t *out)
{
    sag_rng_choice_traced(rng, pop, size, out, NULL, NULL);
}

/* =================================================================== *
 * End-to-end derivation
 * =================================================================== */

sag_pcg64 sag_derive_source_rng(int64_t base_seed, const char *instance,
                                int64_t iteration, int64_t vehicle, const char *source)
{
    char payload[SAG_RNG_PAYLOAD_CAP];
    size_t n = sag_rng_payload(payload, sizeof payload, base_seed, instance,
                               iteration, vehicle, source);
    uint8_t digest[16];
    sag_blake2b_128(payload, n, digest);
    uint32_t words[4];
    sag_rng_words_from_digest(digest, words);
    uint64_t state[4];
    sag_seed_sequence_u64(words, state);
    sag_pcg64 rng;
    sag_pcg64_seed(&rng, state);
    return rng;
}
