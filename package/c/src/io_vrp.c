#define _POSIX_C_SOURCE 200809L

#include "io_vrp.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VRP_I64_LIMIT 9223372036854775808.0

/* ------------------------------------------------------------------ */
/* Minimal reproduction of Python string semantics                    */
/* ------------------------------------------------------------------ */

static int vrp_isspace(unsigned char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'
        || c == 0x1c || c == 0x1d || c == 0x1e || c == 0x1f;
}

static char *vrp_strip(char *s, size_t len, size_t *out_len)
{
    size_t b = 0;
    while (b < len && vrp_isspace((unsigned char)s[b])) ++b;
    size_t e = len;
    while (e > b && vrp_isspace((unsigned char)s[e - 1])) --e;
    s[e] = '\0';
    *out_len = e - b;
    return s + b;
}

static bool vrp_starts_with_upper(const char *s, size_t len, const char *prefix)
{
    size_t n = strlen(prefix);
    if (len < n) return false;
    for (size_t i = 0; i < n; ++i) {
        unsigned char c = (unsigned char)s[i];
        if (c >= 'a' && c <= 'z') c = (unsigned char)(c - 'a' + 'A');
        if ((char)c != prefix[i]) return false;
    }
    return true;
}

static bool vrp_equals_upper(const char *s, const char *upper_literal)
{
    size_t i = 0;
    for (; s[i] != '\0' && upper_literal[i] != '\0'; ++i) {
        unsigned char c = (unsigned char)s[i];
        if (c >= 'a' && c <= 'z') c = (unsigned char)(c - 'a' + 'A');
        if ((char)c != upper_literal[i]) return false;
    }
    return s[i] == '\0' && upper_literal[i] == '\0';
}

static size_t vrp_split(char *s, size_t len, char **parts, size_t max_parts)
{
    size_t count = 0, i = 0;
    while (i < len) {
        while (i < len && vrp_isspace((unsigned char)s[i])) ++i;
        if (i >= len) break;
        size_t start = i;
        while (i < len && !vrp_isspace((unsigned char)s[i])) ++i;
        if (count < max_parts) {
            parts[count] = s + start;
            if (i < len) s[i] = '\0';
        }
        ++count;
        if (i < len) ++i;
    }
    return count;
}

static int64_t vrp_parse_i64(const char *token, const char *what, const char *path)
{
    errno = 0;
    char *end = NULL;
    long long v = strtoll(token, &end, 10);
    if (end == token || *end != '\0' || errno == ERANGE)
        sag_die("%s: %s '%s' is not a valid integer (Python int() would raise ValueError)", path, what, token);
    return (int64_t)v;
}

static double vrp_parse_f64(const char *token, const char *what, const char *path)
{
    errno = 0;
    char *end = NULL;
    double v = strtod(token, &end);
    if (end == token || *end != '\0')
        sag_die("%s: %s '%s' is not a valid float (Python float() would raise ValueError)", path, what, token);
    return v;
}

static bool vrp_is_integer(double v)
{
    return isfinite(v) && floor(v) == v;
}

/* ------------------------------------------------------------------ */
/* Distance                                                           */
/* ------------------------------------------------------------------ */

double sag_euc2d(double xi, double yi, double xj, double yj)
{
    double dist = hypot(xi - xj, yi - yj);
    double shifted = dist + 0.5;
    if (!(shifted >= 0.0) || shifted >= VRP_I64_LIMIT)
        sag_die("euclidean distance out of int64 range: hypot=%.17g "
                "(coords %.17g,%.17g -> %.17g,%.17g)", dist, xi, yi, xj, yj);
    return (double)(int64_t)shifted;
}

void sag_distance_matrix(const sag_instance *inst, double *out)
{
    int64_t n = inst->n_nodes;
    const double *c = inst->coords;
    if (n < 0 || c == NULL) sag_die("sag_distance_matrix: instance has no coords/n_nodes filled in");
    memset(out, 0, (size_t)n * (size_t)n * sizeof(double));
    for (int64_t i = 0; i < n; ++i) {
        double xi = c[2 * i], yi = c[2 * i + 1];
        for (int64_t j = i + 1; j < n; ++j) {
            double rounded = sag_euc2d(xi, yi, c[2 * j], c[2 * j + 1]);
            out[i * n + j] = rounded;
            out[j * n + i] = rounded;
        }
    }
}

/* ------------------------------------------------------------------ */
/* .vrp parsing                                                       */
/* ------------------------------------------------------------------ */

typedef enum {
    VRP_SEC_NONE = 0, VRP_SEC_COORDS, VRP_SEC_DEMAND, VRP_SEC_DEPOT
} vrp_section;

static void vrp_path_stem(const char *path, char *out, size_t cap)
{
    const char *slash = strrchr(path, '/');
    const char *base = slash ? slash + 1 : path;
    const char *dot = strrchr(base, '.');
    size_t len = (dot != NULL && dot != base) ? (size_t)(dot - base) : strlen(base);
    if (len + 1 > cap)
        sag_die("instance stem too long (%zu bytes > %zu): %s", len, cap - 1, path);
    memcpy(out, base, len);
    out[len] = '\0';
}

void sag_load_instance(sag_arena *a, const char *vrp_path, int64_t vehicles,
                       bool has_optimal, double optimal, sag_instance *out)
{
    FILE *fp = fopen(vrp_path, "r");
    if (!fp) sag_die("VRP file not found: %s", vrp_path);

    char name[SAG_NAME_MAX];
    vrp_path_stem(vrp_path, name, sizeof name);

    int64_t dimension = -1;
    bool    have_dimension = false;
    double  capacity = 0.0;
    bool    have_capacity = false;
    char    edge_weight_type[64] = "EUC_2D";
    int64_t depot = 1;

    double  *coords = NULL;
    int64_t *demand = NULL;
    bool    *have_coord = NULL;
    bool    *have_demand = NULL;
    int64_t  alloc_dim = -1;
    size_t   scratch_mark = 0;
    bool     scratch_taken = false;

    vrp_section section = VRP_SEC_NONE;
    char   *raw = NULL;
    size_t  raw_cap = 0;
    ssize_t got;

    while ((got = getline(&raw, &raw_cap, fp)) != -1) {
        size_t len = 0;
        char *line = vrp_strip(raw, (size_t)got, &len);
        if (len == 0) continue;

        bool switched = true;
        if (vrp_starts_with_upper(line, len, "NODE_COORD_SECTION"))      section = VRP_SEC_COORDS;
        else if (vrp_starts_with_upper(line, len, "DEMAND_SECTION"))     section = VRP_SEC_DEMAND;
        else if (vrp_starts_with_upper(line, len, "DEPOT_SECTION"))      section = VRP_SEC_DEPOT;
        else if (vrp_starts_with_upper(line, len, "EOF"))                 break;
        else switched = false;

        if (switched) {
            if (section != VRP_SEC_DEPOT && alloc_dim < 0) {
                if (!have_dimension)
                    sag_die("%s: the DIMENSION header must appear before "
                            "NODE_COORD_SECTION / DEMAND_SECTION", vrp_path);
                if (dimension <= 0)
                    sag_die("%s: DIMENSION=%lld is not positive", vrp_path, (long long)dimension);
                coords = SAG_NEW0(a, double, dimension * 2);
                demand = SAG_NEW0(a, int64_t, dimension);
                scratch_mark  = sag_arena_mark(a);
                have_coord    = SAG_NEW0(a, bool, dimension);
                have_demand   = SAG_NEW0(a, bool, dimension);
                scratch_taken = true;
                alloc_dim = dimension;
            }
            continue;
        }

        if (section == VRP_SEC_COORDS) {
            char *parts[3];
            if (vrp_split(line, len, parts, 3) < 3) continue;
            int64_t idx = vrp_parse_i64(parts[0], "node id", vrp_path);
            if (idx < 1 || idx > alloc_dim) continue;
            coords[(idx - 1) * 2 + 0] = vrp_parse_f64(parts[1], "coord x", vrp_path);
            coords[(idx - 1) * 2 + 1] = vrp_parse_f64(parts[2], "coord y", vrp_path);
            have_coord[idx - 1] = true;
        } else if (section == VRP_SEC_DEMAND) {
            char *parts[2];
            if (vrp_split(line, len, parts, 2) < 2) continue;
            int64_t idx = vrp_parse_i64(parts[0], "node id", vrp_path);
            if (idx < 1 || idx > alloc_dim) continue;
            double raw_demand = vrp_parse_f64(parts[1], "demand", vrp_path);
            if (!vrp_is_integer(raw_demand) || fabs(raw_demand) >= VRP_I64_LIMIT)
                sag_die("Non-integer demand %.17g for node %lld in %s "
                        "(demands are carried as int64)", raw_demand, (long long)idx, vrp_path);
            demand[idx - 1] = (int64_t)raw_demand;
            have_demand[idx - 1] = true;
        } else if (section == VRP_SEC_DEPOT) {
            if (line[0] == '-' && line[1] == '1') { section = VRP_SEC_NONE; continue; }
            depot = vrp_parse_i64(line, "depot id", vrp_path);
        } else {
            char *colon = strchr(line, ':');
            if (colon == NULL) continue;
            size_t key_len = (size_t)(colon - line);
            size_t klen = 0, vlen = 0;
            char *value = vrp_strip(colon + 1, len - key_len - 1, &vlen);
            char *key   = vrp_strip(line, key_len, &klen);
            if (vrp_equals_upper(key, "NAME")) {
                if (vlen + 1 > sizeof name)
                    sag_die("%s: NAME '%s' exceeds SAG_NAME_MAX=%d", vrp_path, value, SAG_NAME_MAX);
                for (size_t t = 0; t < vlen; ++t) {
                    const unsigned char ch = (unsigned char)value[t];
                    if (ch < 0x20 || ch >= 0x7f)
                        sag_die("%s: NAME contains a non-printable ASCII byte 0x%02x (position %zu) -- "
                                "Python's str.strip() strips multi-byte unicode whitespace and byte-wise C "
                                "does not, so the two arms' RNG payloads diverge silently. Rename the instance to pure ASCII",
                                vrp_path, (unsigned)ch, t);
                }
                memcpy(name, value, vlen + 1);
            } else if (vrp_equals_upper(key, "DIMENSION")) {
                int64_t d = vrp_parse_i64(value, "DIMENSION", vrp_path);
                if (alloc_dim >= 0 && d != alloc_dim)
                    sag_die("%s: DIMENSION rewritten after the data sections (%lld -> %lld)",
                            vrp_path, (long long)alloc_dim, (long long)d);
                dimension = d;
                have_dimension = true;
            } else if (vrp_equals_upper(key, "CAPACITY")) {
                capacity = vrp_parse_f64(value, "CAPACITY", vrp_path);
                have_capacity = true;
            } else if (vrp_equals_upper(key, "EDGE_WEIGHT_TYPE")) {
                if (vlen + 1 > sizeof edge_weight_type)
                    sag_die("%s: EDGE_WEIGHT_TYPE '%s' is too long", vrp_path, value);
                memcpy(edge_weight_type, value, vlen + 1);
            }
        }
    }
    free(raw);
    if (ferror(fp)) sag_die("%s: read error", vrp_path);
    fclose(fp);

    /* ---- Python's post-parse checks, item by item in the same order ---- */
    if (!have_dimension || !have_capacity)
        sag_die("Failed to parse dimension/capacity from %s", vrp_path);
    {
        size_t sl = 0;
        char *ewt = vrp_strip(edge_weight_type, strlen(edge_weight_type), &sl);
        if (!vrp_equals_upper(ewt, "EUC_2D"))
            sag_die("SAG-Sep only supports EDGE_WEIGHT_TYPE=EUC_2D (frozen scope), "
                    "got '%s' in %s", ewt, vrp_path);
    }
    if (depot != 1)
        sag_die("SAG-Sep expects depot at external node 1, got %lld in %s",
                (long long)depot, vrp_path);
    if (alloc_dim < 0)
        sag_die("%s: missing NODE_COORD_SECTION / DEMAND_SECTION (Python crashes while building the arrays)",
                vrp_path);

    for (int64_t idx = 1; idx <= alloc_dim; ++idx) {
        if (!have_coord[idx - 1])
            sag_die("Missing coordinates for node %lld in %s", (long long)idx, vrp_path);
        if (!have_demand[idx - 1])
            sag_die("Missing demand for node %lld in %s", (long long)idx, vrp_path);
    }
    if (!vrp_is_integer(capacity) || fabs(capacity) >= VRP_I64_LIMIT)
        sag_die("Non-integer capacity %.17g in %s (capacity is carried as int64)",
                capacity, vrp_path);
    if (demand[0] != 0)
        sag_die("Depot demand must be 0, got %lld in %s", (long long)demand[0], vrp_path);
    for (int64_t idx = 1; idx < dimension; ++idx)
        if (demand[idx] < 0)
            sag_die("Negative demand %lld for node %lld in %s",
                    (long long)demand[idx], (long long)(idx + 1), vrp_path);

    if (scratch_taken) sag_arena_release(a, scratch_mark);

    double *weight = SAG_NEW(a, double, alloc_dim * alloc_dim);

    memset(out, 0, sizeof *out);
    memcpy(out->name, name, strlen(name) + 1);
    out->n_nodes       = alloc_dim;
    out->depot         = SAG_DEPOT;
    out->vehicles      = vehicles;
    out->vehicle_count = -1;
    out->capacity      = (int64_t)capacity;
    out->demand        = demand;
    out->coords        = coords;
    out->weight        = weight;
    out->optimal       = has_optimal ? optimal : 0.0;
    out->has_optimal   = has_optimal;

    sag_distance_matrix(out, weight);
}
