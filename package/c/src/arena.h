#ifndef SAGSEP_ARENA_H
#define SAGSEP_ARENA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#error "The SAG-Sep C version is pure C11: C++ translation units are not accepted"
#endif

void sag_die(const char *fmt, ...) __attribute__((format(printf, 1, 2), noreturn));

size_t sag_size_mul(size_t a, size_t b);

#define SAG_ARENA_ALIGN 64u

typedef struct {
    unsigned char *base;
    size_t         cap;
    size_t         used;
    size_t         high_water;
    const char    *name;
} sag_arena;

void   sag_arena_init(sag_arena *a, size_t bytes, const char *name);
void   sag_arena_destroy(sag_arena *a);

void  *sag_arena_alloc(sag_arena *a, size_t bytes, size_t align);
void  *sag_arena_zalloc(sag_arena *a, size_t bytes, size_t align);

void   sag_arena_reset(sag_arena *a);

size_t sag_arena_mark(const sag_arena *a);
void   sag_arena_release(sag_arena *a, size_t mark);

static inline size_t sag_arena_used(const sag_arena *a)       { return a->used; }
static inline size_t sag_arena_capacity(const sag_arena *a)   { return a->cap; }
static inline size_t sag_arena_remaining(const sag_arena *a)  { return a->cap - a->used; }
static inline size_t sag_arena_high_water(const sag_arena *a) { return a->high_water; }

#define SAG_NEW(a, T, n)  ((T *)sag_arena_alloc((a),  sag_size_mul(sizeof(T), (size_t)(n)), _Alignof(T)))
#define SAG_NEW0(a, T, n) ((T *)sag_arena_zalloc((a), sag_size_mul(sizeof(T), (size_t)(n)), _Alignof(T)))

#endif
