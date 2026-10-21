#define _POSIX_C_SOURCE 200809L

#include "directed_io.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

/* ===================================================================== */
/* Growable helpers                                                       */
/* ===================================================================== */

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

typedef struct {
    double *value;
    int64_t count;
    int64_t capacity;
} bc_vector;

static void vector_push(bc_vector *v, double x)
{
    if (v->count == v->capacity) {
        v->capacity = v->capacity ? v->capacity * 2 : 1024;
        v->value = (double *)bc_realloc(v->value, (size_t)v->capacity * sizeof *v->value);
    }
    v->value[v->count++] = x;
}

/* ===================================================================== */
/* Lexing, matching Python's str.strip / str.split / int / float          */
/* ===================================================================== */

static char *strip(char *s)
{
    while (*s && isspace((unsigned char)*s)) ++s;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) --end;
    *end = '\0';
    return s;
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
    const long long value = strtoll(text, &end, 10);
    if (end == text || (end && *end != '\0') || errno != 0)
        bc_die("%s: %s is not an integer: \"%s\"", path, what, text);
    return (int64_t)value;
}

static double parse_double(const char *text, const char *path, const char *what)
{
    char *end = NULL;
    double value = strtod(text, &end);
    if (end == text || (end && *end != '\0'))
        bc_die("%s: %s is not a number: \"%s\"", path, what, text);
    return value;
}

static void upcase(char *s)
{
    for (; *s; ++s) *s = (char)toupper((unsigned char)*s);
}

static void name_from_stem(bc_instance *out, const char *path)
{
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    const char *dot = strrchr(base, '.');
    const size_t length = (dot && dot != base) ? (size_t)(dot - base) : strlen(base);
    if (length >= BC_NAME_MAX) bc_die("%s: file stem is longer than %d bytes", path,
                                      BC_NAME_MAX - 1);
    memcpy(out->name, base, length);
    out->name[length] = '\0';
}

/* ===================================================================== */
/* The reader                                                             */
/* ===================================================================== */

enum {
    SECTION_NONE = 0,
    SECTION_COORDS,
    SECTION_DEMAND,
    SECTION_DEPOT,
    SECTION_WEIGHTS
};

static int section_of(const char *keyword)
{
    if (strcmp(keyword, "NODE_COORD_SECTION") == 0) return SECTION_COORDS;
    if (strcmp(keyword, "DEMAND_SECTION") == 0) return SECTION_DEMAND;
    if (strcmp(keyword, "DEPOT_SECTION") == 0) return SECTION_DEPOT;
    if (strcmp(keyword, "EDGE_WEIGHT_SECTION") == 0) return SECTION_WEIGHTS;
    return -1;
}

void bc_directed_load(const char *path, bc_distance_mode distance,
                      bc_directed_instance *out)
{
    FILE *file = fopen(path, "r");
    if (!file) bc_die("instance file not found: %s", path);

    memset(out, 0, sizeof *out);
    name_from_stem(&out->inst, path);

    int64_t dimension = -1;
    double capacity = 0.0;
    bool has_capacity = false;
    char weight_type[64] = "EUC_2D";
    char weight_format[64] = "";
    int64_t depot = 0;
    int64_t n_depots = 0;

    bc_table coords, demand;
    table_init(&coords, 2);
    table_init(&demand, 1);
    bc_vector weights = {NULL, 0, 0};

    int section = SECTION_NONE;
    char *raw = NULL;
    size_t raw_capacity = 0;
    while (getline(&raw, &raw_capacity, file) != -1) {
        char *line = strip(raw);
        if (!*line) continue;

        char keyword[128];
        const char *colon = strchr(line, ':');
        const size_t keyword_length = colon ? (size_t)(colon - line) : strlen(line);
        if (keyword_length < sizeof keyword) {
            memcpy(keyword, line, keyword_length);
            keyword[keyword_length] = '\0';
            char *trimmed = strip(keyword);
            upcase(trimmed);
            const int next = section_of(trimmed);
            if (next >= 0) { section = next; continue; }
            if (strcmp(trimmed, "EOF") == 0) break;
        }

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
            char *cursor = line, *token = NULL;
            while ((token = next_field(&cursor)) != NULL) {
                const int64_t node = parse_int(token, path, "depot id");
                if (node < 0) { section = SECTION_NONE; break; }
                depot = node;
                n_depots += 1;
            }
        } else if (section == SECTION_WEIGHTS) {
            char *cursor = line, *token = NULL;
            while ((token = next_field(&cursor)) != NULL)
                vector_push(&weights, parse_double(token, path, "edge weight"));
        } else if (colon) {
            char key[128];
            const size_t key_length = (size_t)(colon - line);
            if (key_length >= sizeof key) continue;
            memcpy(key, line, key_length);
            key[key_length] = '\0';
            char *trimmed_key = strip(key);
            upcase(trimmed_key);
            char *value = strip((char *)colon + 1);
            if (strcmp(trimmed_key, "NAME") == 0) {
                if (strlen(value) >= BC_NAME_MAX)
                    bc_die("%s: instance name \"%s\" is longer than %d bytes; the "
                           "separator hashes it into its RNG payload and cannot "
                           "truncate it", path, value, BC_NAME_MAX - 1);
                memcpy(out->inst.name, value, strlen(value) + 1);
            } else if (strcmp(trimmed_key, "DIMENSION") == 0) {
                dimension = parse_int(value, path, "DIMENSION");
            } else if (strcmp(trimmed_key, "CAPACITY") == 0) {
                capacity = parse_double(value, path, "CAPACITY");
                has_capacity = true;
            } else if (strcmp(trimmed_key, "VEHICLES") == 0) {
                out->vehicles = parse_int(value, path, "VEHICLES");
            } else if (strcmp(trimmed_key, "EDGE_WEIGHT_TYPE") == 0) {
                if (strlen(value) >= sizeof weight_type)
                    bc_die("%s: EDGE_WEIGHT_TYPE value is too long", path);
                memcpy(weight_type, value, strlen(value) + 1);
            } else if (strcmp(trimmed_key, "EDGE_WEIGHT_FORMAT") == 0) {
                if (strlen(value) >= sizeof weight_format)
                    bc_die("%s: EDGE_WEIGHT_FORMAT value is too long", path);
                memcpy(weight_format, value, strlen(value) + 1);
            }
        }
    }
    free(raw);
    fclose(file);

    if (dimension < 0 || !has_capacity)
        bc_die("%s: could not read DIMENSION and CAPACITY", path);
    if (dimension < 2)
        bc_die("%s: DIMENSION = %lld (need the depot plus at least one customer)",
               path, (long long)dimension);
    if (capacity != floor(capacity))
        bc_die("%s: non-integer CAPACITY %g (capacity is carried as int64)", path, capacity);
    if ((int64_t)capacity <= 0)
        bc_die("%s: CAPACITY = %lld (must be positive)", path, (long long)capacity);
    if (n_depots > 1)
        bc_die("%s: %lld depots; these examples model one", path, (long long)n_depots);
    if (n_depots == 0) depot = 1;
    if (depot < 1 || depot > dimension)
        bc_die("%s: depot %lld is outside 1..%lld", path, (long long)depot,
               (long long)dimension);
    upcase(weight_type);
    upcase(weight_format);

    int64_t *order = (int64_t *)bc_malloc((size_t)dimension * sizeof *order);
    order[0] = depot;
    int64_t written = 1;
    for (int64_t node = 1; node <= dimension; ++node) {
        if (node != depot) order[written++] = node;
    }

    out->inst.n_nodes = dimension;
    out->inst.capacity = (int64_t)capacity;
    out->inst.demand = (int64_t *)bc_calloc((size_t)dimension, sizeof *out->inst.demand);
    for (int64_t internal = 0; internal < dimension; ++internal) {
        const int64_t external = order[internal];
        if (!table_has(&demand, external))
            bc_die("%s: missing demand for node %lld", path, (long long)external);
        const double value = demand.value[external];
        if (!isfinite(value) || value != floor(value))
            bc_die("%s: demand %g for node %lld is not a finite integer (demands are "
                   "carried as int64)", path, value, (long long)external);
        if (value < 0.0)
            bc_die("%s: negative demand %g for node %lld", path, value,
                   (long long)external);
        out->inst.demand[internal] = (int64_t)value;
    }
    if (out->inst.demand[0] != 0)
        bc_die("%s: depot demand must be 0, got %lld", path,
               (long long)out->inst.demand[0]);
    for (int64_t internal = 1; internal < dimension; ++internal) {
        if (out->inst.demand[internal] > out->inst.capacity)
            bc_die("%s: a customer demands more than one vehicle carries", path);
    }
    if (out->vehicles != 0 && (out->vehicles < 1 || out->vehicles > dimension - 1))
        bc_die("%s: VEHICLES = %lld; a fixed fleet must be between 1 and the customer "
               "count %lld, since every route visits a customer", path,
               (long long)out->vehicles, (long long)(dimension - 1));

    out->inst.weight = (double *)bc_calloc((size_t)(dimension * dimension),
                                           sizeof *out->inst.weight);
    if (strcmp(weight_type, "EXPLICIT") == 0) {
        if (strcmp(weight_format, "FULL_MATRIX") != 0)
            bc_die("%s: EDGE_WEIGHT_FORMAT must be FULL_MATRIX for an explicit matrix, "
                   "got %s; a triangular format cannot hold an asymmetric cost matrix",
                   path, *weight_format ? weight_format : "(absent)");
        if (weights.count != dimension * dimension)
            bc_die("%s: EDGE_WEIGHT_SECTION has %lld entries, expected %lld", path,
                   (long long)weights.count, (long long)(dimension * dimension));
        for (int64_t k = 0; k < weights.count; ++k) {
            if (!isfinite(weights.value[k]))
                bc_die("%s: EDGE_WEIGHT_SECTION holds a non-finite cost", path);
        }
        for (int64_t i = 0; i < dimension; ++i) {
            const int64_t source_row = (order[i] - 1) * dimension;
            for (int64_t j = 0; j < dimension; ++j)
                out->inst.weight[i * dimension + j] = weights.value[source_row + order[j] - 1];
        }
        out->distance_convention = "explicit";
    } else if (strcmp(weight_type, "EUC_2D") == 0) {
        out->inst.coords = (double *)bc_calloc((size_t)(2 * dimension),
                                               sizeof *out->inst.coords);
        for (int64_t internal = 0; internal < dimension; ++internal) {
            const int64_t external = order[internal];
            if (!table_has(&coords, external))
                bc_die("%s: missing coordinates for node %lld", path, (long long)external);
            out->inst.coords[2 * internal] = coords.value[2 * external];
            out->inst.coords[2 * internal + 1] = coords.value[2 * external + 1];
            if (!isfinite(out->inst.coords[2 * internal])
                || !isfinite(out->inst.coords[2 * internal + 1]))
                bc_die("%s: NODE_COORD_SECTION holds a non-finite coordinate", path);
        }
        for (int64_t i = 0; i < dimension; ++i) {
            const double xi = out->inst.coords[2 * i];
            const double yi = out->inst.coords[2 * i + 1];
            for (int64_t j = i + 1; j < dimension; ++j) {
                const double raw_distance = hypot(xi - out->inst.coords[2 * j],
                                                  yi - out->inst.coords[2 * j + 1]);
                const double value = (distance == BC_DISTANCE_ROUND)
                                     ? (double)(long long)(raw_distance + 0.5)
                                     : raw_distance;
                out->inst.weight[i * dimension + j] = value;
                out->inst.weight[j * dimension + i] = value;
            }
        }
        out->distance_convention = (distance == BC_DISTANCE_ROUND) ? "round" : "exact";
    } else {
        bc_die("%s: EDGE_WEIGHT_TYPE must be EXPLICIT or EUC_2D, got %s", path,
               weight_type);
    }

    free(order);
    free(weights.value);
    table_free(&coords);
    table_free(&demand);
}

void bc_directed_free(bc_directed_instance *out)
{
    bc_instance_free(&out->inst);
    memset(out, 0, sizeof *out);
}
