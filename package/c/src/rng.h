#ifndef SAGSEP_RNG_H
#define SAGSEP_RNG_H

#include <stddef.h>
#include <stdint.h>

/* ------------------------------------------------------------------ *
 * L1  payload + blake2b-128
 * ------------------------------------------------------------------ */

#define SAG_BLAKE2B_OUTLEN     16u
#define SAG_BLAKE2B_BLOCKBYTES 128u

#define SAG_RNG_PAYLOAD_CAP 512u

size_t sag_rng_payload(char *out, size_t cap, int64_t base_seed, const char *instance,
                       int64_t iteration, int64_t vehicle, const char *source);

void sag_blake2b_128(const void *data, size_t len, uint8_t digest[16]);

void sag_rng_words_from_digest(const uint8_t digest[16], uint32_t words[4]);

/* ------------------------------------------------------------------ *
 * L3  numpy SeedSequence (pool_size=4, entropy is exactly 4 uint32 words)
 * ------------------------------------------------------------------ */

void sag_seed_sequence_u64(const uint32_t words[4], uint64_t state[4]);

/* ------------------------------------------------------------------ *
 * L4  numpy PCG64 (XSL-RR 128-bit state / 64-bit output)
 * ------------------------------------------------------------------ */

typedef struct {
    uint64_t state_hi, state_lo;
    uint64_t inc_hi, inc_lo;
    int      has_uint32;
    uint32_t uinteger;
} sag_pcg64;

void sag_pcg64_seed(sag_pcg64 *rng, const uint64_t state_words[4]);

uint64_t sag_pcg64_next64(sag_pcg64 *rng);
uint32_t sag_pcg64_next32(sag_pcg64 *rng);

/* ------------------------------------------------------------------ *
 * L5  bounded / choice
 * ------------------------------------------------------------------ */

uint64_t sag_rng_bounded(sag_pcg64 *rng, uint64_t k);

#define SAG_RNG_CHOICE_MAX_SIZE 1024

void sag_rng_choice(sag_pcg64 *rng, int64_t pop, int64_t size, int64_t *out);

void sag_rng_choice_traced(sag_pcg64 *rng, int64_t pop, int64_t size, int64_t *out,
                           int64_t *trace_bound, int64_t *trace_val);

/* ------------------------------------------------------------------ *
 * End-to-end derivation (= derive_source_rng in the Python reference)
 * ------------------------------------------------------------------ */

sag_pcg64 sag_derive_source_rng(int64_t base_seed, const char *instance,
                                int64_t iteration, int64_t vehicle, const char *source);

#endif
