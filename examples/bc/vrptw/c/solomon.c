#define _POSIX_C_SOURCE 200809L

#include "solomon.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

#define TW_FEASIBILITY_EPSILON 1e-9

/* ===================================================================== */
/* Distance conventions                                                   */
/* ===================================================================== */

_Bool tw_distance_parse(const char *text, tw_distance *out)
{
    if (!text) return false;
    char lowered[64];
    size_t length = strlen(text);
    if (length >= sizeof lowered) return false;
    for (size_t i = 0; i <= length; ++i) lowered[i] = (char)tolower((unsigned char)text[i]);

    if (strcmp(lowered, "trunc1") == 0 || strcmp(lowered, "truncated_1dp") == 0) {
        *out = TW_DISTANCE_TRUNC1;
        return true;
    }
    if (strcmp(lowered, "exact") == 0 || strcmp(lowered, "euclidean") == 0
        || strcmp(lowered, "full_precision") == 0) {
        *out = TW_DISTANCE_EXACT;
        return true;
    }
    return false;
}

const char *tw_distance_name(tw_distance convention)
{
    return convention == TW_DISTANCE_EXACT ? "exact" : "trunc1";
}

/* ===================================================================== */
/* Parsing                                                                */
/* ===================================================================== */

static char *strip(char *s)
{
    while (*s && isspace((unsigned char)*s)) ++s;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) --end;
    *end = '\0';
    return s;
}

static int split_fields(char *line, char **fields, int cap)
{
    int count = 0;
    char *cursor = line;
    while (count < cap) {
        while (*cursor && isspace((unsigned char)*cursor)) ++cursor;
        if (!*cursor) break;
        fields[count++] = cursor;
        while (*cursor && !isspace((unsigned char)*cursor)) ++cursor;
        if (*cursor) *cursor++ = '\0';
    }
    return count;
}

static bool parse_number(const char *text, double *out)
{
    char *end = NULL;
    errno = 0;
    const double value = strtod(text, &end);
    if (end == text || (end && *end != '\0')) return false;
    if (!isfinite(value)) return false;
    *out = value;
    return true;
}

static int64_t require_integral(double value, const char *path, const char *what)
{
    if (value != floor(value))
        bc_die("%s: %s must be an integer, got %g", path, what, value);
    return (int64_t)value;
}

static bool all_numeric(char **fields, int count, double *values)
{
    for (int i = 0; i < count; ++i) {
        if (!parse_number(fields[i], &values[i])) return false;
    }
    return true;
}

typedef struct {
    double *value;
    int64_t count;
    int64_t capacity;
} tw_records;

static void records_push(tw_records *records, const double *row)
{
    if (records->count == records->capacity) {
        records->capacity = records->capacity ? records->capacity * 2 : 128;
        records->value = (double *)bc_realloc(records->value,
                                              (size_t)(records->capacity * 7) * sizeof *records->value);
    }
    memcpy(records->value + records->count * 7, row, 7 * sizeof *row);
    records->count += 1;
}

static int compare_records(const void *lhs, const void *rhs)
{
    const double a = *(const double *)lhs, b = *(const double *)rhs;
    return (a > b) - (a < b);
}

static void build_travel(tw_instance *inst)
{
    const int64_t n = inst->n_customers + 1;
    inst->travel = (double *)bc_calloc((size_t)(n * n), sizeof *inst->travel);
    for (int64_t i = 0; i < n; ++i) {
        const double xi = inst->coords[2 * i], yi = inst->coords[2 * i + 1];
        for (int64_t j = i + 1; j < n; ++j) {
            double distance = hypot(xi - inst->coords[2 * j], yi - inst->coords[2 * j + 1]);
            if (inst->convention == TW_DISTANCE_TRUNC1)
                distance = floor(distance * 10.0 + 1e-6) / 10.0;
            inst->travel[i * n + j] = distance;
            inst->travel[j * n + i] = distance;
        }
    }
}

void tw_instance_load(const char *path, tw_distance convention, tw_instance *out)
{
    FILE *file = fopen(path, "r");
    if (!file) bc_die("Solomon file not found: %s", path);

    memset(out, 0, sizeof *out);
    out->convention = convention;

    bool have_name = false, have_capacity = false;
    tw_records records;
    memset(&records, 0, sizeof records);

    char *raw = NULL;
    size_t raw_capacity = 0;
    while (getline(&raw, &raw_capacity, file) != -1) {
        char *line = strip(raw);
        if (!*line) continue;
        if (!have_name) {
            if (strlen(line) >= TW_NAME_MAX)
                bc_die("%s: instance name \"%s\" is longer than %d bytes; the separator "
                       "hashes it into its RNG payload and cannot truncate it",
                       path, line, TW_NAME_MAX - 1);
            memcpy(out->name, line, strlen(line) + 1);
            have_name = true;
            continue;
        }
        char *fields[16];
        double values[16];
        const int count = split_fields(line, fields, 16);
        if (count == 0 || !all_numeric(fields, count, values)) continue;
        if (count == 2 && !have_capacity) {
            out->available_vehicles = require_integral(values[0], path, "the NUMBER field");
            out->capacity = require_integral(values[1], path, "CAPACITY");
            have_capacity = true;
        } else if (count >= 7) {
            (void)require_integral(values[0], path, "a node id");
            records_push(&records, values);
        }
    }
    free(raw);
    fclose(file);

    if (!have_name) bc_die("%s: the file is empty", path);
    if (!have_capacity) bc_die("%s: no 'NUMBER CAPACITY' record found", path);
    if (records.count == 0) bc_die("%s: no customer records found", path);
    if (out->capacity <= 0)
        bc_die("%s: CAPACITY = %lld (must be positive)", path, (long long)out->capacity);
    if (records.count < 2)
        bc_die("%s: need the depot plus at least one customer", path);

    qsort(records.value, (size_t)records.count, 7 * sizeof *records.value, compare_records);
    for (int64_t i = 0; i < records.count; ++i) {
        if ((int64_t)records.value[i * 7] != i)
            bc_die("%s: node ids must be 0..%lld without gaps or repeats; row %lld "
                   "has id %g", path, (long long)(records.count - 1), (long long)i,
                   records.value[i * 7]);
    }

    const int64_t n_nodes = records.count;
    out->n_customers = n_nodes - 1;
    out->demand = (int64_t *)bc_calloc((size_t)n_nodes, sizeof *out->demand);
    out->coords = (double *)bc_calloc((size_t)(2 * n_nodes), sizeof *out->coords);
    out->ready = (double *)bc_calloc((size_t)n_nodes, sizeof *out->ready);
    out->due = (double *)bc_calloc((size_t)n_nodes, sizeof *out->due);
    out->service = (double *)bc_calloc((size_t)n_nodes, sizeof *out->service);
    for (int64_t i = 0; i < n_nodes; ++i) {
        const double *row = records.value + i * 7;
        char what[64];
        snprintf(what, sizeof what, "the demand of node %lld", (long long)i);
        const int64_t quantity = require_integral(row[3], path, what);
        if (i == 0 && quantity != 0)
            bc_die("%s: the depot's demand must be 0, got %lld", path,
                   (long long)quantity);
        if (i > 0 && quantity <= 0)
            bc_die("%s: customer %lld has demand %lld; this example requires strictly "
                   "positive customer demand (see solomon.h: it is what makes the "
                   "MIPSOL check complete)", path, (long long)i, (long long)quantity);
        if (quantity > out->capacity)
            bc_die("%s: node %lld has demand %lld > capacity %lld; no feasible route "
                   "exists", path, (long long)i, (long long)quantity,
                   (long long)out->capacity);
        if (row[6] < 0.0)
            bc_die("%s: node %lld has negative service time %g", path, (long long)i, row[6]);
        if (row[4] > row[5])
            bc_die("%s: node %lld has an empty time window", path, (long long)i);
        out->coords[2 * i] = row[1];
        out->coords[2 * i + 1] = row[2];
        out->demand[i] = quantity;
        out->ready[i] = row[4];
        out->due[i] = row[5];
        out->service[i] = row[6];
    }
    free(records.value);

    build_travel(out);
}

void tw_instance_free(tw_instance *out)
{
    free(out->demand);
    free(out->coords);
    free(out->ready);
    free(out->due);
    free(out->service);
    free(out->travel);
    memset(out, 0, sizeof *out);
}

/* ===================================================================== */
/* Preprocessing                                                          */
/* ===================================================================== */

static int64_t physical(int64_t node, int64_t sink)
{
    return (node == 0 || node == sink) ? 0 : node;
}

static void dijkstra(const double *weight, int64_t n, int64_t source, double *dist)
{
    char *settled = (char *)bc_calloc((size_t)n, sizeof *settled);
    for (int64_t v = 0; v < n; ++v) dist[v] = INFINITY;
    dist[source] = 0.0;
    for (int64_t round = 0; round < n; ++round) {
        int64_t node = -1;
        double best = INFINITY;
        for (int64_t v = 0; v < n; ++v) {
            if (settled[v] || !(dist[v] < best)) continue;
            best = dist[v];
            node = v;
        }
        if (node < 0) break;
        settled[node] = 1;
        const double *row = weight + node * n;
        for (int64_t v = 0; v < n; ++v) {
            const double relaxed = best + row[v];
            if (relaxed < dist[v]) dist[v] = relaxed;
        }
    }
    free(settled);
}

void tw_preprocess_arcs(const tw_instance *inst, tw_arcs *out)
{
    const int64_t n = inst->n_customers;
    const int64_t sink = n + 1;
    const int64_t n_model_nodes = n + 2;
    const int64_t n_physical = n + 1;

    memset(out, 0, sizeof *out);
    out->n_model_nodes = n_model_nodes;
    out->lower = (double *)bc_malloc((size_t)n_model_nodes * sizeof *out->lower);
    out->upper = (double *)bc_malloc((size_t)n_model_nodes * sizeof *out->upper);
    out->service = (double *)bc_malloc((size_t)n_model_nodes * sizeof *out->service);

    for (int64_t node = 0; node <= n; ++node) {
        out->lower[node] = inst->ready[node];
        out->upper[node] = inst->due[node];
        out->service[node] = inst->service[node];
    }
    out->lower[sink] = inst->ready[0];
    out->upper[sink] = inst->due[0];
    out->service[sink] = 0.0;

    /* --- 1/2. structural and pair-capacity candidates -------------------- */
    char *candidate = (char *)bc_calloc((size_t)(n_model_nodes * n_model_nodes),
                                        sizeof *candidate);
    for (int64_t tail = 0; tail <= n; ++tail) {
        for (int64_t head = 1; head <= n + 1; ++head) {
            if (tail == head) continue;
            if (tail == 0 && head == sink) continue;
            out->n_arcs_before += 1;
            if (tail != 0 && head != sink
                && inst->demand[tail] > inst->capacity - inst->demand[head]) {
                out->removed_pair_capacity += 1;
                continue;
            }
            candidate[tail * n_model_nodes + head] = 1;
        }
    }

    /* --- 3. shortest-path bound tightening ------------------------------- */
    double *weight = (double *)bc_malloc((size_t)(n_model_nodes * n_model_nodes)
                                         * sizeof *weight);
    double *reverse = (double *)bc_malloc((size_t)(n_model_nodes * n_model_nodes)
                                          * sizeof *reverse);
    for (int64_t u = 0; u < n_model_nodes; ++u) {
        for (int64_t v = 0; v < n_model_nodes; ++v) {
            double length = INFINITY;
            if (candidate[u * n_model_nodes + v]) {
                const int64_t pu = physical(u, sink), pv = physical(v, sink);
                length = out->service[u] + inst->travel[pu * n_physical + pv];
            }
            weight[u * n_model_nodes + v] = length;
            reverse[v * n_model_nodes + u] = length;
        }
    }
    double *from_source = (double *)bc_malloc((size_t)n_model_nodes * sizeof *from_source);
    double *to_sink = (double *)bc_malloc((size_t)n_model_nodes * sizeof *to_sink);
    dijkstra(weight, n_model_nodes, 0, from_source);
    dijkstra(reverse, n_model_nodes, sink, to_sink);
    free(weight);
    free(reverse);

    for (int64_t node = 1; node <= n; ++node) {
        if (!isfinite(from_source[node]) || !isfinite(to_sink[node]))
            bc_die("%s: customer %lld cannot be reached from the depot and returned "
                   "from within the surviving arcs", inst->name, (long long)node);
        const double earliest = inst->ready[0] + from_source[node];
        const double latest = out->upper[sink] - to_sink[node];
        if (earliest > out->lower[node]) out->lower[node] = earliest;
        if (latest < out->upper[node]) out->upper[node] = latest;
    }
    free(from_source);
    free(to_sink);
    out->upper[0] = out->lower[0];
    for (int64_t node = 0; node < n_model_nodes; ++node) {
        if (out->lower[node] > out->upper[node] + TW_FEASIBILITY_EPSILON)
            bc_die("%s: node %lld cannot be served at all (tightened window "
                   "[%g, %g] is empty)", inst->name, (long long)node,
                   out->lower[node], out->upper[node]);
    }

    /* --- 4. time-window elimination, then the per-arc big-M -------------- */
    const int64_t capacity_bound = (n + 1) * (n + 1);
    out->tail = (int32_t *)bc_malloc((size_t)capacity_bound * sizeof *out->tail);
    out->head = (int32_t *)bc_malloc((size_t)capacity_bound * sizeof *out->head);
    out->cost = (double *)bc_malloc((size_t)capacity_bound * sizeof *out->cost);
    out->time = (double *)bc_malloc((size_t)capacity_bound * sizeof *out->time);
    out->big_m = (double *)bc_malloc((size_t)capacity_bound * sizeof *out->big_m);

    int64_t count = 0;
    for (int64_t tail = 0; tail <= n; ++tail) {
        for (int64_t head = 1; head <= n + 1; ++head) {
            if (!candidate[tail * n_model_nodes + head]) continue;
            const double travel_time =
                inst->travel[physical(tail, sink) * n_physical + physical(head, sink)];
            if (out->lower[tail] + out->service[tail] + travel_time
                > out->upper[head] + TW_FEASIBILITY_EPSILON) {
                out->removed_time_window += 1;
                continue;
            }
            out->tail[count] = (int32_t)tail;
            out->head[count] = (int32_t)head;
            out->cost[count] = travel_time;
            out->time[count] = travel_time;
            double big_m = out->upper[tail] + out->service[tail] + travel_time
                           - out->lower[head];
            out->big_m[count] = big_m > 0.0 ? big_m : 0.0;
            count += 1;
        }
    }
    free(candidate);
    if (count == 0) bc_die("%s: preprocessing removed every arc", inst->name);
    out->n_arcs = count;
}

void tw_arcs_free(tw_arcs *arcs)
{
    free(arcs->tail);
    free(arcs->head);
    free(arcs->cost);
    free(arcs->time);
    free(arcs->big_m);
    free(arcs->lower);
    free(arcs->upper);
    free(arcs->service);
    memset(arcs, 0, sizeof *arcs);
}

void tw_check_customer_degrees(const tw_arcs *arcs, int64_t n_customers)
{
    int64_t *out_degree = (int64_t *)bc_calloc((size_t)arcs->n_model_nodes, sizeof *out_degree);
    int64_t *in_degree = (int64_t *)bc_calloc((size_t)arcs->n_model_nodes, sizeof *in_degree);
    for (int64_t k = 0; k < arcs->n_arcs; ++k) {
        out_degree[arcs->tail[k]] += 1;
        in_degree[arcs->head[k]] += 1;
    }
    for (int64_t node = 1; node <= n_customers; ++node) {
        if (out_degree[node] == 0 || in_degree[node] == 0) {
            free(out_degree);
            free(in_degree);
            bc_die("preprocessing left customer %lld without an incoming or outgoing "
                   "arc; the instance is infeasible as modelled", (long long)node);
        }
    }
    free(out_degree);
    free(in_degree);
}
