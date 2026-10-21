#define _POSIX_C_SOURCE 200809L

#include "cmst_io.h"

#include <ctype.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

#define BC_MAX_CAPACITIES 16

static char *strip(char *s)
{
    while (*s && isspace((unsigned char)*s)) ++s;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) --end;
    *end = '\0';
    return s;
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

static void upcase(char *text)
{
    for (char *p = text; *p; ++p) *p = (char)toupper((unsigned char)*p);
}

static void require_value(const char *path, const char *key, const char *value,
                          const char *expected)
{
    char buffer[64];
    snprintf(buffer, sizeof buffer, "%s", value);
    upcase(buffer);
    if (strcmp(buffer, expected) != 0)
        bc_die("%s: %s must be %s, got \"%s\"", path, key, expected, value);
}

static void name_from_stem(bc_instance *out, const char *path)
{
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    const char *dot = strrchr(base, '.');
    const size_t length = (dot && dot != base) ? (size_t)(dot - base) : strlen(base);
    if (length >= BC_NAME_MAX) bc_die("%s: file stem is longer than %d bytes",
                                      path, BC_NAME_MAX - 1);
    memcpy(out->name, base, length);
    out->name[length] = '\0';
}

void bc_cmst_load(const char *path, int64_t capacity, bc_instance *out)
{
    FILE *file = fopen(path, "r");
    if (!file) bc_die("CMST file not found: %s", path);

    memset(out, 0, sizeof *out);
    name_from_stem(out, path);

    int64_t dimension = -1, root = 1;
    int64_t capacities[BC_MAX_CAPACITIES];
    int n_capacities = 0;
    int64_t node = 0, weights_read = 0;
    enum { SECTION_HEADER, SECTION_DEMAND, SECTION_WEIGHT, SECTION_DONE } section = SECTION_HEADER;

    char *raw = NULL;
    size_t raw_capacity = 0;
    while (getline(&raw, &raw_capacity, file) != -1) {
        char *line = strip(raw);
        if (!*line) continue;

        if (section == SECTION_HEADER && strcmp(line, "DEMAND_SECTION") == 0) {
            if (dimension < 0) bc_die("%s: no DIMENSION header", path);
            if (n_capacities == 0) bc_die("%s: no CAPACITIES header", path);
            if (root != 1) bc_die("%s: the root must be external node 1, got %lld",
                                  path, (long long)root);
            out->n_nodes = dimension;
            out->demand = (int64_t *)bc_calloc((size_t)dimension, sizeof *out->demand);
            out->weight = (double *)bc_calloc((size_t)(dimension * dimension),
                                              sizeof *out->weight);
            section = SECTION_DEMAND;
            continue;
        }
        if (section == SECTION_DEMAND && strcmp(line, "EDGE_WEIGHT_SECTION") == 0) {
            if (node != dimension)
                bc_die("%s: DEMAND_SECTION has %lld rows, expected %lld",
                       path, (long long)node, (long long)dimension);
            section = SECTION_WEIGHT;
            continue;
        }
        if (section == SECTION_WEIGHT && strcmp(line, "EOF") == 0) {
            section = SECTION_DONE;
            break;
        }

        if (section == SECTION_HEADER) {
            char *colon = strchr(line, ':');
            if (!colon) bc_die("%s: unexpected line \"%s\"", path, line);
            *colon = '\0';
            char *key = strip(line);
            char *value = strip(colon + 1);
            upcase(key);
            if (strcmp(key, "NAME") == 0) {
                if (strlen(value) >= BC_NAME_MAX)
                    bc_die("%s: instance name \"%s\" is longer than %d bytes; the "
                           "separator hashes it into its RNG payload and cannot "
                           "truncate it", path, value, BC_NAME_MAX - 1);
                memcpy(out->name, value, strlen(value) + 1);
            } else if (strcmp(key, "TYPE") == 0) {
                require_value(path, "TYPE", value, "CMST");
            } else if (strcmp(key, "DIMENSION") == 0) {
                dimension = parse_int(value, path, "DIMENSION");
            } else if (strcmp(key, "ROOT") == 0) {
                root = parse_int(value, path, "ROOT");
            } else if (strcmp(key, "EDGE_WEIGHT_TYPE") == 0) {
                require_value(path, "EDGE_WEIGHT_TYPE", value, "EXPLICIT");
            } else if (strcmp(key, "EDGE_WEIGHT_FORMAT") == 0) {
                require_value(path, "EDGE_WEIGHT_FORMAT", value, "FULL_MATRIX");
            } else if (strcmp(key, "CAPACITIES") == 0) {
                char *cursor = value, *token = NULL;
                while ((token = next_field(&cursor)) != NULL) {
                    if (n_capacities == BC_MAX_CAPACITIES)
                        bc_die("%s: more than %d capacities", path, BC_MAX_CAPACITIES);
                    capacities[n_capacities++] = parse_int(token, path, "capacity");
                }
            }
        } else if (section == SECTION_DEMAND) {
            char *cursor = line;
            char *id_text = next_field(&cursor);
            char *demand_text = next_field(&cursor);
            if (!id_text || !demand_text || next_field(&cursor) != NULL)
                bc_die("%s: bad demand row \"%s\"", path, line);
            if (parse_int(id_text, path, "node id") != node + 1)
                bc_die("%s: demand rows must be node ids 1..DIMENSION in order", path);
            out->demand[node] = parse_int(demand_text, path, "demand");
            node += 1;
        } else if (section == SECTION_WEIGHT) {
            char *cursor = line, *token = NULL;
            while ((token = next_field(&cursor)) != NULL) {
                if (weights_read == dimension * dimension)
                    bc_die("%s: more than %lld costs", path,
                           (long long)(dimension * dimension));
                out->weight[weights_read++] = (double)parse_int(token, path, "cost");
            }
        }
    }
    free(raw);
    fclose(file);

    if (section != SECTION_DONE && section != SECTION_WEIGHT)
        bc_die("%s: truncated file (no EDGE_WEIGHT_SECTION)", path);
    if (weights_read != dimension * dimension)
        bc_die("%s: expected %lld costs, got %lld", path,
               (long long)(dimension * dimension), (long long)weights_read);
    if (dimension < 2)
        bc_die("%s: DIMENSION = %lld (need the root and at least one customer)",
               path, (long long)dimension);

    bool listed = false;
    for (int i = 0; i < n_capacities; ++i) listed = listed || capacities[i] == capacity;
    if (capacity <= 0) {
        if (n_capacities > 1) {
            fprintf(stderr, "bc: %s is posed with capacities", path);
            for (int i = 0; i < n_capacities; ++i)
                fprintf(stderr, " %lld", (long long)capacities[i]);
            fprintf(stderr, "\n");
            bc_die("--capacity is required for this instance");
        }
        capacity = capacities[0];
    } else if (!listed) {
        fprintf(stderr, "bc: %s is posed with capacities", path);
        for (int i = 0; i < n_capacities; ++i)
            fprintf(stderr, " %lld", (long long)capacities[i]);
        fprintf(stderr, "\n");
        bc_die("capacity %lld is not one of them", (long long)capacity);
    }
    if (capacity <= 0)
        bc_die("%s: capacity must be positive, got %lld", path, (long long)capacity);
    out->capacity = capacity;

    if (out->demand[0] != 0)
        bc_die("%s: the root's demand must be 0, got %lld",
               path, (long long)out->demand[0]);
    for (int64_t i = 1; i < dimension; ++i) {
        if (out->demand[i] <= 0)
            bc_die("%s: customer demands must be positive, node %lld has %lld",
                   path, (long long)(i + 1), (long long)out->demand[i]);
        if (out->demand[i] > capacity)
            bc_die("%s: customer %lld has demand %lld > capacity %lld; the instance "
                   "is infeasible", path, (long long)(i + 1),
                   (long long)out->demand[i], (long long)capacity);
    }
    for (int64_t i = 0; i < dimension; ++i) {
        if (out->weight[i * dimension + i] != 0.0)
            bc_die("%s: the cost matrix has a non-zero diagonal at node %lld",
                   path, (long long)(i + 1));
        for (int64_t j = i + 1; j < dimension; ++j) {
            if (out->weight[i * dimension + j] != out->weight[j * dimension + i])
                bc_die("%s: the cost matrix is not symmetric", path);
        }
    }
}
