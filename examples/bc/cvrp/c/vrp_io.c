#define _POSIX_C_SOURCE 200809L

#include "vrp_io.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "util.h"

typedef struct {
    double *value;
    char   *present;
    int64_t stride;
    int64_t capacity;
} bc_table;

static void table_init(bc_table *t, int64_t stride)
{
    t->value = NULL;
    t->present = NULL;
    t->stride = stride;
    t->capacity = 0;
}

static void table_free(bc_table *t)
{
    free(t->value);
    free(t->present);
    table_init(t, t->stride);
}

static void table_reserve(bc_table *t, int64_t id)
{
    if (id < t->capacity) return;
    int64_t capacity = t->capacity ? t->capacity : 64;
    while (capacity <= id) capacity *= 2;
    t->value = (double *)bc_realloc(t->value, (size_t)(capacity * t->stride) * sizeof *t->value);
    t->present = (char *)bc_realloc(t->present, (size_t)capacity * sizeof *t->present);
    memset(t->present + t->capacity, 0, (size_t)(capacity - t->capacity) * sizeof *t->present);
    t->capacity = capacity;
}

static bool table_has(const bc_table *t, int64_t id)
{
    return id >= 0 && id < t->capacity && t->present[id];
}

/* --- lexing helpers, matching Python's str.strip / str.split / int / float --- */

static char *strip(char *s)
{
    while (*s && isspace((unsigned char)*s)) ++s;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) --end;
    *end = '\0';
    return s;
}

static bool starts_with_upper(const char *line, const char *prefix)
{
    for (size_t i = 0; prefix[i]; ++i) {
        if (toupper((unsigned char)line[i]) != prefix[i]) return false;
    }
    return true;
}

static char *next_field(char **cursor)
{
    char *s = *cursor;
    while (*s && isspace((unsigned char)*s)) ++s;
    if (!*s) return NULL;
    char *start = s;
    while (*s && !isspace((unsigned char)*s)) ++s;
    if (*s) *s++ = '\0';
    *cursor = s;
    return start;
}

static int64_t parse_int(const char *text, const char *path, const char *what)
{
    char *end = NULL;
    errno = 0;
    long long value = strtoll(text, &end, 10);
    if (end == text || (end && *end != '\0') || errno != 0)
        bc_die("%s: %s is not an integer: \"%s\"", path, what, text);
    return (int64_t)value;
}

static double parse_double(const char *text, const char *path, const char *what)
{
    char *end = NULL;
    errno = 0;
    double value = strtod(text, &end);
    if (end == text || (end && *end != '\0'))
        bc_die("%s: %s is not a number: \"%s\"", path, what, text);
    return value;
}

static void set_name(bc_instance *out, const char *name, const char *path)
{
    if (strlen(name) >= BC_NAME_MAX)
        bc_die("%s: instance name \"%s\" is longer than %d bytes; the separator hashes it "
               "into its RNG payload and cannot truncate it", path, name, BC_NAME_MAX - 1);
    memcpy(out->name, name, strlen(name) + 1);
}

static void name_from_stem(bc_instance *out, const char *path)
{
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    const char *dot = strrchr(base, '.');
    size_t length = dot && dot != base ? (size_t)(dot - base) : strlen(base);
    if (length >= BC_NAME_MAX)
        bc_die("%s: file stem is longer than %d bytes", path, BC_NAME_MAX - 1);
    memcpy(out->name, base, length);
    out->name[length] = '\0';
}

static void build_weight(bc_instance *inst)
{
    const int64_t n = inst->n_nodes;
    inst->weight = (double *)bc_calloc((size_t)(n * n), sizeof *inst->weight);
    for (int64_t i = 0; i < n; ++i) {
        const double xi = inst->coords[2 * i];
        const double yi = inst->coords[2 * i + 1];
        for (int64_t j = i + 1; j < n; ++j) {
            const double distance = hypot(xi - inst->coords[2 * j], yi - inst->coords[2 * j + 1]);
            const double rounded = (double)(long long)(distance + 0.5);
            inst->weight[i * n + j] = rounded;
            inst->weight[j * n + i] = rounded;
        }
    }
}

void bc_instance_load(const char *path, bc_instance *out)
{
    FILE *file = fopen(path, "r");
    if (!file) bc_die("VRP file not found: %s", path);

    memset(out, 0, sizeof *out);
    name_from_stem(out, path);

    int64_t dimension = -1, depot = 1;
    double capacity = 0.0;
    bool has_capacity = false;
    char edge_weight_type[64] = "EUC_2D";

    bc_table coords, demand;
    table_init(&coords, 2);
    table_init(&demand, 1);

    enum { SECTION_NONE, SECTION_COORDS, SECTION_DEMAND, SECTION_DEPOT } section = SECTION_NONE;
    char *raw = NULL;
    size_t raw_capacity = 0;
    ssize_t length;
    while ((length = getline(&raw, &raw_capacity, file)) != -1) {
        (void)length;
        char *line = strip(raw);
        if (!*line) continue;
        if (starts_with_upper(line, "NODE_COORD_SECTION")) { section = SECTION_COORDS; continue; }
        if (starts_with_upper(line, "DEMAND_SECTION")) { section = SECTION_DEMAND; continue; }
        if (starts_with_upper(line, "DEPOT_SECTION")) { section = SECTION_DEPOT; continue; }
        if (starts_with_upper(line, "EOF")) break;

        if (section == SECTION_COORDS) {
            char *cursor = line;
            char *id_text = next_field(&cursor);
            char *x_text = next_field(&cursor);
            char *y_text = next_field(&cursor);
            if (!id_text || !x_text || !y_text) continue;
            const int64_t id = parse_int(id_text, path, "node id");
            if (id < 0) bc_die("%s: negative node id %lld", path, (long long)id);
            table_reserve(&coords, id);
            coords.value[2 * id] = parse_double(x_text, path, "coordinate");
            coords.value[2 * id + 1] = parse_double(y_text, path, "coordinate");
            coords.present[id] = 1;
        } else if (section == SECTION_DEMAND) {
            char *cursor = line;
            char *id_text = next_field(&cursor);
            char *demand_text = next_field(&cursor);
            if (!id_text || !demand_text) continue;
            const int64_t id = parse_int(id_text, path, "node id");
            if (id < 0) bc_die("%s: negative node id %lld", path, (long long)id);
            table_reserve(&demand, id);
            demand.value[id] = parse_double(demand_text, path, "demand");
            demand.present[id] = 1;
        } else if (section == SECTION_DEPOT) {
            if (line[0] == '-' && line[1] == '1') { section = SECTION_NONE; continue; }
            depot = parse_int(line, path, "depot id");
        } else {
            char *colon = strchr(line, ':');
            if (!colon) continue;
            *colon = '\0';
            char *key = strip(line);
            char *value = strip(colon + 1);
            for (char *p = key; *p; ++p) *p = (char)toupper((unsigned char)*p);
            if (strcmp(key, "NAME") == 0) {
                set_name(out, value, path);
            } else if (strcmp(key, "DIMENSION") == 0) {
                dimension = parse_int(value, path, "DIMENSION");
            } else if (strcmp(key, "CAPACITY") == 0) {
                capacity = parse_double(value, path, "CAPACITY");
                has_capacity = true;
            } else if (strcmp(key, "EDGE_WEIGHT_TYPE") == 0) {
                if (strlen(value) >= sizeof edge_weight_type)
                    bc_die("%s: EDGE_WEIGHT_TYPE value is too long", path);
                memcpy(edge_weight_type, value, strlen(value) + 1);
            }
        }
    }
    free(raw);
    fclose(file);

    if (dimension < 0 || !has_capacity)
        bc_die("Failed to parse dimension/capacity from %s", path);
    for (char *p = edge_weight_type; *p; ++p) *p = (char)toupper((unsigned char)*p);
    if (strcmp(edge_weight_type, "EUC_2D") != 0)
        bc_die("only EDGE_WEIGHT_TYPE=EUC_2D is supported, got '%s' in %s",
               edge_weight_type, path);
    if (depot != 1)
        bc_die("the depot must be external node 1, got %lld in %s", (long long)depot, path);
    if (dimension < 2)
        bc_die("%s: DIMENSION = %lld (need the depot plus at least one customer)",
               path, (long long)dimension);

    out->n_nodes = dimension;
    out->coords = (double *)bc_calloc((size_t)(2 * dimension), sizeof *out->coords);
    out->demand = (int64_t *)bc_calloc((size_t)dimension, sizeof *out->demand);
    for (int64_t id = 1; id <= dimension; ++id) {
        if (!table_has(&coords, id))
            bc_die("Missing coordinates for node %lld in %s", (long long)id, path);
        if (!table_has(&demand, id))
            bc_die("Missing demand for node %lld in %s", (long long)id, path);
        out->coords[2 * (id - 1)] = coords.value[2 * id];
        out->coords[2 * (id - 1) + 1] = coords.value[2 * id + 1];
        const double raw_demand = demand.value[id];
        if (raw_demand != floor(raw_demand))
            bc_die("Non-integer demand %g for node %lld in %s (demands are carried as int64)",
                   raw_demand, (long long)id, path);
        out->demand[id - 1] = (int64_t)raw_demand;
    }
    table_free(&coords);
    table_free(&demand);

    if (capacity != floor(capacity))
        bc_die("Non-integer capacity %g in %s (capacity is carried as int64)", capacity, path);
    out->capacity = (int64_t)capacity;
    if (out->capacity <= 0)
        bc_die("%s: CAPACITY = %lld (must be positive)", path, (long long)out->capacity);
    if (out->demand[0] != 0)
        bc_die("Depot demand must be 0, got %lld in %s", (long long)out->demand[0], path);

    build_weight(out);
}
