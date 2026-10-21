#include "arena.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void sag_die(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fflush(stdout);
    fprintf(stderr, "FATAL: ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(1);
}

size_t sag_size_mul(size_t a, size_t b)
{
    if (a != 0 && b > SIZE_MAX / a)
        sag_die("size_t overflow: %zu * %zu", a, b);
    return a * b;
}

static size_t round_up(size_t value, size_t align)
{
    return (value + align - 1u) & ~(align - 1u);
}

void sag_arena_init(sag_arena *a, size_t bytes, const char *name)
{
    size_t cap = round_up(bytes ? bytes : SAG_ARENA_ALIGN, SAG_ARENA_ALIGN);
    void *base = aligned_alloc(SAG_ARENA_ALIGN, cap);
    if (!base)
        sag_die("arena '%s': cannot reserve %zu bytes", name ? name : "?", cap);
    a->base = (unsigned char *)base;
    a->cap = cap;
    a->used = 0;
    a->high_water = 0;
    a->name = name ? name : "arena";
}

void sag_arena_destroy(sag_arena *a)
{
    free(a->base);
    a->base = NULL;
    a->cap = a->used = a->high_water = 0;
    a->name = NULL;
}

void *sag_arena_alloc(sag_arena *a, size_t bytes, size_t align)
{
    if (!a->base)
        sag_die("arena '%s': allocate before init", a->name ? a->name : "?");
    if (align == 0 || (align & (align - 1u)) != 0)
        sag_die("arena '%s': alignment %zu is not a power of two", a->name, align);

    size_t offset = round_up(a->used, align);
    if (offset > a->cap || bytes > a->cap - offset)
        sag_die("arena '%s' exhausted: need %zu bytes (align %zu) at offset %zu, "
                "capacity %zu (high_water %zu); raise the reservation instead of "
                "falling back to malloc",
                a->name, bytes, align, offset, a->cap, a->high_water);

    a->used = offset + bytes;
    if (a->used > a->high_water)
        a->high_water = a->used;
    return a->base + offset;
}

void *sag_arena_zalloc(sag_arena *a, size_t bytes, size_t align)
{
    void *p = sag_arena_alloc(a, bytes, align);
    if (bytes)
        memset(p, 0, bytes);
    return p;
}

void sag_arena_reset(sag_arena *a)
{
    a->used = 0;
}

size_t sag_arena_mark(const sag_arena *a)
{
    return a->used;
}

void sag_arena_release(sag_arena *a, size_t mark)
{
    if (mark > a->used)
        sag_die("arena '%s': release mark %zu is beyond used %zu", a->name, mark, a->used);
    a->used = mark;
}
