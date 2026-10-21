#define _POSIX_C_SOURCE 200809L

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "bc_tw.h"
#include "formulation_tw.h"
#include "instance.h"
#include "separator_cvrpsep.h"
#include "separator_sagsep.h"
#include "solomon.h"
#include "util.h"

typedef struct {
    char   *name;
    int64_t vehicles;
    double  optimal;
    bool    has_optimal;
    char   *convention;
} tw_manifest_entry;

typedef struct {
    tw_manifest_entry *entries;
    int64_t count;
    int64_t capacity;
} tw_manifest;

typedef struct {
    const char *separator;
    const char **instances;
    int          n_instances;
    const char *instances_root;
    const char *manifest;
    int64_t     vehicles;
    double      optimal;
    bool        has_optimal;
    const char *distance;
    const char *onnx;
    const char *device;
    const char *size_preset;
    const char *out;
    tw_options  options;
} tw_args;

#define TW_DEFAULT_DISTANCE "trunc1"

/* ===================================================================== */
/* Command line                                                           */
/* ===================================================================== */

static void usage(FILE *stream)
{
    fputs(
"usage: bc_vrptw --separator {cvrpsep,sagseptts} --instance NAME [NAME ...] [options]\n"
"\n"
"Branch-and-cut for the VRPTW with rounded capacity cut separation. The two runs\n"
"share the model, the fleet, the Gurobi configuration, the root and tree\n"
"schedule, the exact rescoring and the admission rule; what changes is the\n"
"separator, which keeps its own proposal budget and internal gates.\n"
"\n"
"  --separator {cvrpsep,sagseptts}  cut source (required)\n"
"  --instance NAME [NAME ...]       instance names resolved under --instances-root,\n"
"                                   or paths to Solomon files (required)\n"
"  --instances-root DIR             directory holding the Solomon files\n"
"                                   (default: $SAGSEP_INSTANCES_ROOT)\n"
"  --manifest FILE                  CSV with instance,n,k,distance,status,\n"
"                                   distance_convention columns (default:\n"
"                                   manifest.csv next to the instances)\n"
"  --vehicles K                     fleet size, applied to every instance listed;\n"
"                                   overrides the manifest. Required when the\n"
"                                   manifest gives none: K is fixed by an equality,\n"
"                                   so a different K is a different problem\n"
"  --optimal VALUE                  known value of a single instance\n"
"  --distance {trunc1,exact}        travel-distance convention: trunc1 =\n"
"                                   floor(10*d + 1e-6)/10, exact = full precision\n"
"                                   (default: the manifest's, else " TW_DEFAULT_DISTANCE ")\n"
"  --onnx PATH                      sagseptts only: exported policy graph\n"
"                                   (default: $SAGSEP_ONNX)\n"
"  --device NAME                    sagseptts only: cpu | cuda | cuda:<id>\n"
"                                   (default: cpu)\n"
"  --size-preset TAG                sagseptts only: n50 ... n1000, or a *.toml\n"
"                                   config (default: by instance size)\n"
"  --time-limit SECONDS             budget per instance, root loop included\n"
"                                   (default: 1800; 0 = none)\n"
"  --mip-gap G                      relative MIP gap (default: 0.0001)\n"
"  --threads N                      Gurobi Threads, and the learned separator's\n"
"                                   ONNX Runtime intra-op threads (default: 1)\n"
"  --seed N                         Gurobi Seed (default: 0)\n"
"  --root-rounds N                  maximum root cutting-plane rounds\n"
"                                   (default: 100; 0 skips the root phase)\n"
"  --cuts-per-round N               cap on rows added per round; also CVRPSEP's\n"
"                                   own MaxNoOfCuts (default: 100)\n"
"  --min-violation V                the violation a row must exceed, applied to\n"
"                                   both cut sources after exact rescoring; also\n"
"                                   CVRPSEP's own EpsViolation (default: 1e-06)\n"
"  --tree-commit {offered,root}     what a MIPNODE batch tells the separators:\n"
"                                   offered = the rows the driver offered through\n"
"                                   GRBcbcut, which is CVRPSEP's own convention;\n"
"                                   root = nothing, so each records only the\n"
"                                   permanent root rows. Applied to both sources\n"
"                                   (default: offered)\n"
"  --usercut-max-nodes N            separate user cuts while the node count is at\n"
"                                   most this (default: 15; 0 = off)\n"
"  --out DIR                        directory for per-instance result JSON files\n"
"  --quiet                          suppress the Gurobi log and the root trace\n"
"  -h, --help                       this message\n",
    stream);
}

static const char *need_value(int argc, char **argv, int *index)
{
    if (*index + 1 >= argc) bc_die("%s needs a value", argv[*index]);
    *index += 1;
    return argv[*index];
}

static long parse_long_arg(const char *text, const char *flag)
{
    char *end = NULL;
    const long value = strtol(text, &end, 10);
    if (end == text || *end != '\0') bc_die("%s expects an integer, got \"%s\"", flag, text);
    return value;
}

static double parse_double_arg(const char *text, const char *flag)
{
    char *end = NULL;
    const double value = strtod(text, &end);
    if (end == text || *end != '\0') bc_die("%s expects a number, got \"%s\"", flag, text);
    return value;
}

static void parse_args(int argc, char **argv, tw_args *args)
{
    memset(args, 0, sizeof *args);
    args->instances_root = getenv("SAGSEP_INSTANCES_ROOT");
    args->onnx = getenv("SAGSEP_ONNX");
    args->device = "cpu";
    args->options.time_limit = 1800.0;
    args->options.mip_gap = 1e-4;
    args->options.threads = 1;
    args->options.seed = 0;
    args->options.root_rounds = 100;
    args->options.cuts_per_round = 100;
    args->options.min_violation = 1e-6;
    args->options.usercut_max_nodes = 15;
    args->options.tree_commit = "offered";
    args->options.verbose = true;
    args->instances = (const char **)bc_malloc((size_t)argc * sizeof *args->instances);

    for (int i = 1; i < argc; ++i) {
        const char *flag = argv[i];
        if (strcmp(flag, "-h") == 0 || strcmp(flag, "--help") == 0) {
            usage(stdout);
            free(args->instances);
            exit(EXIT_SUCCESS);
        } else if (strcmp(flag, "--separator") == 0) {
            args->separator = need_value(argc, argv, &i);
        } else if (strcmp(flag, "--instance") == 0) {
            if (i + 1 >= argc) bc_die("--instance needs at least one value");
            while (i + 1 < argc && strncmp(argv[i + 1], "--", 2) != 0)
                args->instances[args->n_instances++] = argv[++i];
        } else if (strcmp(flag, "--instances-root") == 0) {
            args->instances_root = need_value(argc, argv, &i);
        } else if (strcmp(flag, "--manifest") == 0) {
            args->manifest = need_value(argc, argv, &i);
        } else if (strcmp(flag, "--vehicles") == 0) {
            args->vehicles = parse_long_arg(need_value(argc, argv, &i), flag);
        } else if (strcmp(flag, "--optimal") == 0) {
            args->optimal = parse_double_arg(need_value(argc, argv, &i), flag);
            args->has_optimal = true;
        } else if (strcmp(flag, "--distance") == 0) {
            args->distance = need_value(argc, argv, &i);
        } else if (strcmp(flag, "--onnx") == 0) {
            args->onnx = need_value(argc, argv, &i);
        } else if (strcmp(flag, "--device") == 0) {
            args->device = need_value(argc, argv, &i);
        } else if (strcmp(flag, "--size-preset") == 0) {
            args->size_preset = need_value(argc, argv, &i);
        } else if (strcmp(flag, "--out") == 0) {
            args->out = need_value(argc, argv, &i);
        } else if (strcmp(flag, "--time-limit") == 0) {
            args->options.time_limit = parse_double_arg(need_value(argc, argv, &i), flag);
        } else if (strcmp(flag, "--mip-gap") == 0) {
            args->options.mip_gap = parse_double_arg(need_value(argc, argv, &i), flag);
        } else if (strcmp(flag, "--threads") == 0) {
            args->options.threads = (int)parse_long_arg(need_value(argc, argv, &i), flag);
        } else if (strcmp(flag, "--seed") == 0) {
            args->options.seed = (int)parse_long_arg(need_value(argc, argv, &i), flag);
        } else if (strcmp(flag, "--root-rounds") == 0) {
            args->options.root_rounds = (int)parse_long_arg(need_value(argc, argv, &i), flag);
        } else if (strcmp(flag, "--cuts-per-round") == 0) {
            args->options.cuts_per_round = (int)parse_long_arg(need_value(argc, argv, &i), flag);
        } else if (strcmp(flag, "--min-violation") == 0) {
            args->options.min_violation = parse_double_arg(need_value(argc, argv, &i), flag);
        } else if (strcmp(flag, "--tree-commit") == 0) {
            const char *value = need_value(argc, argv, &i);
            if (strcmp(value, "root") != 0 && strcmp(value, "offered") != 0)
                bc_die("--tree-commit must be root or offered, got \"%s\"", value);
            args->options.tree_commit = value;
        } else if (strcmp(flag, "--usercut-max-nodes") == 0) {
            args->options.usercut_max_nodes = (int)parse_long_arg(need_value(argc, argv, &i), flag);
        } else if (strcmp(flag, "--quiet") == 0) {
            args->options.verbose = false;
        } else {
            usage(stderr);
            bc_die("unrecognised argument: %s", flag);
        }
    }

    if (!args->separator) bc_die("--separator is required (cvrpsep or sagseptts)");
    if (strcmp(args->separator, "cvrpsep") != 0 && strcmp(args->separator, "sagseptts") != 0)
        bc_die("--separator must be cvrpsep or sagseptts, got \"%s\"", args->separator);
    if (args->n_instances == 0) bc_die("--instance is required");
    if (args->has_optimal && args->n_instances > 1)
        bc_die("--optimal applies to a single instance; use the manifest for a batch");
    if (args->options.cuts_per_round <= 0)
        bc_die("--cuts-per-round must be positive, got %d", args->options.cuts_per_round);
    if (args->options.root_rounds < 0)
        bc_die("--root-rounds must be non-negative, got %d", args->options.root_rounds);
    if (args->options.usercut_max_nodes < 0)
        bc_die("--usercut-max-nodes must be non-negative, got %d",
               args->options.usercut_max_nodes);
    if (args->options.threads < 1)
        bc_die("--threads must be at least 1, got %d", args->options.threads);
    if (!(args->options.min_violation >= 0.0))
        bc_die("--min-violation must be non-negative, got %g", args->options.min_violation);
    if (!(args->options.mip_gap >= 0.0))
        bc_die("--mip-gap must be non-negative, got %g", args->options.mip_gap);
    if (!(args->options.time_limit >= 0.0))
        bc_die("--time-limit must be non-negative (0 = no limit), got %g",
               args->options.time_limit);
    if (args->vehicles < 0)
        bc_die("--vehicles must be at least 1, got %lld", (long long)args->vehicles);
    if (args->distance) {
        tw_distance parsed;
        if (!tw_distance_parse(args->distance, &parsed))
            bc_die("--distance must be trunc1 or exact, got \"%s\"", args->distance);
    }
}

/* ===================================================================== */
/* Instance resolution and the manifest                                   */
/* ===================================================================== */

static bool file_exists(const char *path)
{
    struct stat info;
    return stat(path, &info) == 0 && S_ISREG(info.st_mode);
}

static char *join_path(const char *directory, const char *name, const char *suffix)
{
    const size_t length = strlen(directory) + 1 + strlen(name) + strlen(suffix);
    char *path = (char *)bc_malloc(length + 1);
    snprintf(path, length + 1, "%s/%s%s", directory, name, suffix);
    return path;
}

static char *resolve_instance_path(const char *name, const char *instances_root)
{
    if (file_exists(name)) {
        char *path = (char *)bc_malloc(strlen(name) + 1);
        memcpy(path, name, strlen(name) + 1);
        return path;
    }
    if (instances_root && *instances_root) {
        static const char *suffixes[] = { "", ".txt" };
        for (size_t s = 0; s < sizeof suffixes / sizeof suffixes[0]; ++s) {
            char *path = join_path(instances_root, name, suffixes[s]);
            if (file_exists(path)) return path;
            free(path);
        }
    }
    bc_die("cannot resolve instance \"%s\": pass a path to a Solomon file, set "
           "--instances-root, or set SAGSEP_INSTANCES_ROOT", name);
}

static char *path_stem(const char *path)
{
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    const char *dot = strrchr(base, '.');
    const size_t length = (dot && dot != base) ? (size_t)(dot - base) : strlen(base);
    char *stem = (char *)bc_malloc(length + 1);
    memcpy(stem, base, length);
    stem[length] = '\0';
    return stem;
}

static char *trim(char *text)
{
    while (*text == ' ' || *text == '\t' || *text == '\r' || *text == '\n') ++text;
    char *end = text + strlen(text);
    while (end > text && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' || end[-1] == '\n'))
        --end;
    *end = '\0';
    return text;
}

static char *duplicate(const char *text)
{
    char *copy = (char *)bc_malloc(strlen(text) + 1);
    memcpy(copy, text, strlen(text) + 1);
    return copy;
}

static void manifest_read(const char *path, tw_manifest *manifest)
{
    memset(manifest, 0, sizeof *manifest);
    FILE *file = fopen(path, "r");
    if (!file) bc_die("cannot open manifest %s", path);

    char *line = NULL;
    size_t line_capacity = 0;
    int instance_column = -1, vehicles_column = -1, k_column = -1;
    int optimal_column = -1, distance_column = -1, convention_column = -1;
    int status_column = -1;
    bool header_seen = false;
    while (getline(&line, &line_capacity, file) != -1) {
        char *cursor = line;
        char *fields[32];
        int n_fields = 0;
        while (n_fields < 32) {
            char *comma = strchr(cursor, ',');
            if (comma) *comma = '\0';
            fields[n_fields++] = trim(cursor);
            if (!comma) break;
            cursor = comma + 1;
        }
        if (!header_seen) {
            header_seen = true;
            for (int c = 0; c < n_fields; ++c) {
                if (strcmp(fields[c], "instance") == 0) instance_column = c;
                else if (strcmp(fields[c], "vehicles") == 0) vehicles_column = c;
                else if (strcmp(fields[c], "k") == 0) k_column = c;
                else if (strcmp(fields[c], "optimal") == 0) optimal_column = c;
                else if (strcmp(fields[c], "distance") == 0) distance_column = c;
                else if (strcmp(fields[c], "distance_convention") == 0) convention_column = c;
                else if (strcmp(fields[c], "convention") == 0) convention_column = c;
                else if (strcmp(fields[c], "status") == 0) status_column = c;
            }
            if (instance_column < 0)
                bc_die("%s: manifest has no \"instance\" column", path);
            if (vehicles_column < 0) vehicles_column = k_column;
            if (optimal_column < 0) optimal_column = distance_column;
            continue;
        }
        if (instance_column >= n_fields || !*fields[instance_column]) continue;
        if (manifest->count == manifest->capacity) {
            manifest->capacity = manifest->capacity ? manifest->capacity * 2 : 64;
            manifest->entries = (tw_manifest_entry *)bc_realloc(
                manifest->entries, (size_t)manifest->capacity * sizeof *manifest->entries);
        }
        tw_manifest_entry *entry = &manifest->entries[manifest->count++];
        memset(entry, 0, sizeof *entry);
        entry->name = duplicate(fields[instance_column]);
        if (vehicles_column >= 0 && vehicles_column < n_fields && *fields[vehicles_column])
            entry->vehicles = parse_long_arg(fields[vehicles_column], "manifest fleet size");
        const bool proved = status_column < 0 || status_column >= n_fields
                            || !*fields[status_column]
                            || strcmp(fields[status_column], "proved") == 0;
        if (proved && optimal_column >= 0 && optimal_column < n_fields
            && *fields[optimal_column]) {
            entry->optimal = parse_double_arg(fields[optimal_column], "manifest value");
            entry->has_optimal = true;
        }
        if (convention_column >= 0 && convention_column < n_fields
            && *fields[convention_column])
            entry->convention = duplicate(fields[convention_column]);
    }
    free(line);
    fclose(file);
}

static void manifest_free(tw_manifest *manifest)
{
    for (int64_t i = 0; i < manifest->count; ++i) {
        free(manifest->entries[i].name);
        free(manifest->entries[i].convention);
    }
    free(manifest->entries);
    memset(manifest, 0, sizeof *manifest);
}

static const tw_manifest_entry *manifest_lookup(const tw_manifest *manifest, const char *name)
{
    for (int64_t i = 0; i < manifest->count; ++i) {
        if (strcmp(manifest->entries[i].name, name) == 0) return &manifest->entries[i];
    }
    return NULL;
}

/* ===================================================================== */
/* The aggregated instance the separators consume                         */
/* ===================================================================== */

static void aggregated_instance(const tw_instance *inst, bc_instance *agg)
{
    memset(agg, 0, sizeof *agg);
    if (strlen(inst->name) >= BC_NAME_MAX)
        bc_die("instance name \"%s\" does not fit the separator's name field", inst->name);
    memcpy(agg->name, inst->name, strlen(inst->name) + 1);
    const int64_t n = inst->n_customers + 1;
    agg->n_nodes = n;
    agg->capacity = inst->capacity;
    agg->demand = (int64_t *)bc_malloc((size_t)n * sizeof *agg->demand);
    memcpy(agg->demand, inst->demand, (size_t)n * sizeof *agg->demand);
    agg->coords = (double *)bc_malloc((size_t)(2 * n) * sizeof *agg->coords);
    memcpy(agg->coords, inst->coords, (size_t)(2 * n) * sizeof *agg->coords);
    agg->weight = (double *)bc_malloc((size_t)(n * n) * sizeof *agg->weight);
    memcpy(agg->weight, inst->travel, (size_t)(n * n) * sizeof *agg->weight);
}

/* ===================================================================== */
/* Reporting                                                              */
/* ===================================================================== */

static const char *match_label(const tw_result *result)
{
    if (!result->has_matches_optimum) return "val-unknown";
    return result->matches_optimum ? "val-match" : "VALUE-MISMATCH";
}

static void print_summary(const tw_result *result)
{
    char objective[32] = "-", bound[32] = "-", root[32] = "-";
    if (result->has_objective) snprintf(objective, sizeof objective, "%.2f", result->objective);
    if (result->has_best_bound) snprintf(bound, sizeof bound, "%.2f", result->best_bound);
    if (result->has_root_bound) snprintf(root, sizeof root, "%.2f", result->root_bound);
    printf("%s  n=%lld  K=%lld  sep=%s  %s  obj=%s (%s)  bound=%s  root=%s  nodes=%.0f  "
           "lazy=%ld  usercuts=%ld  rootcuts=%d  arcs=%lld/%lld  %.2fs\n",
           result->instance_name, (long long)result->n_customers,
           (long long)result->vehicles, result->separator, result->status_name, objective,
           match_label(result), bound, root, result->node_count,
           result->callback.lazy_cuts_added, result->callback.mipnode_cuts_added,
           result->root_cuts, (long long)result->arcs_after_preprocessing,
           (long long)result->arcs_before_preprocessing, result->runtime);
    fflush(stdout);
}

static void json_string(FILE *file, const char *text)
{
    fputc('"', file);
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
        if (*p == '"' || *p == '\\') fprintf(file, "\\%c", *p);
        else if (*p < 0x20) fprintf(file, "\\u%04x", *p);
        else fputc((int)*p, file);
    }
    fputc('"', file);
}

static void json_double(FILE *file, bool present, double value)
{
    if (!present) { fputs("null", file); return; }
    char text[64];
    bc_format_double(text, sizeof text, value);
    fputs(text, file);
}

static void make_directories(const char *path)
{
    char *buffer = duplicate(path);
    for (char *slash = buffer + 1; ; ++slash) {
        const bool last = (*slash == '\0');
        if (*slash != '/' && !last) continue;
        *slash = '\0';
        if (*buffer && mkdir(buffer, 0777) != 0) {
            struct stat info;
            if (stat(buffer, &info) != 0 || !S_ISDIR(info.st_mode))
                bc_die("cannot create output directory %s", buffer);
        }
        if (last) break;
        *slash = '/';
    }
    free(buffer);
}

static void write_result_json(const char *directory, const char *stem, const tw_result *result)
{
    make_directories(directory);
    const size_t length = strlen(directory) + strlen(stem) + strlen(result->separator) + 16;
    char *path = (char *)bc_malloc(length);
    snprintf(path, length, "%s/bc_%s_%s.json", directory, stem, result->separator);
    FILE *file = fopen(path, "w");
    if (!file) bc_die("cannot write %s", path);

    fputs("{\n", file);
    fputs("  \"instance_name\": ", file); json_string(file, result->instance_name);
    fprintf(file, ",\n  \"n_customers\": %lld", (long long)result->n_customers);
    fprintf(file, ",\n  \"vehicles\": %lld", (long long)result->vehicles);
    fputs(",\n  \"distance\": ", file); json_string(file, result->distance);
    fprintf(file, ",\n  \"arcs_before_preprocessing\": %lld",
            (long long)result->arcs_before_preprocessing);
    fprintf(file, ",\n  \"arcs_after_preprocessing\": %lld",
            (long long)result->arcs_after_preprocessing);
    fputs(",\n  \"separator\": ", file); json_string(file, result->separator);
    fputs(",\n  \"separator_detail\": ", file); json_string(file, result->separator_detail);
    fprintf(file, ",\n  \"status\": %d", result->status);
    fputs(",\n  \"status_name\": ", file); json_string(file, result->status_name);
    fputs(",\n  \"objective\": ", file); json_double(file, result->has_objective, result->objective);
    fputs(",\n  \"best_bound\": ", file);
    json_double(file, result->has_best_bound, result->best_bound);
    fputs(",\n  \"gap\": ", file); json_double(file, result->has_gap, result->gap);
    fputs(",\n  \"runtime\": ", file); json_double(file, true, result->runtime);
    fputs(",\n  \"setup_time\": ", file); json_double(file, true, result->setup_time);
    fputs(",\n  \"node_count\": ", file); json_double(file, true, result->node_count);
    fprintf(file, ",\n  \"sol_count\": %d", result->sol_count);
    fputs(",\n  \"root_bound\": ", file);
    json_double(file, result->has_root_bound, result->root_bound);
    fputs(",\n  \"root_termination\": ", file);
    json_string(file, result->root_termination);
    fprintf(file, ",\n  \"root_bound_final\": %s",
            result->root_bound_final ? "true" : "false");
    fprintf(file, ",\n  \"root_rounds\": %d", result->root_rounds);
    fprintf(file, ",\n  \"root_cuts\": %d", result->root_cuts);
    fputs(",\n  \"root_time\": ", file); json_double(file, true, result->root_time);
    fprintf(file, ",\n  \"callback\": {\n    \"mipnode_calls\": %ld,\n"
                  "    \"mipnode_cuts_added\": %ld,\n    \"mipsol_calls\": %ld,\n"
                  "    \"lazy_cuts_added\": %ld\n  }",
            result->callback.mipnode_calls, result->callback.mipnode_cuts_added,
            result->callback.mipsol_calls, result->callback.lazy_cuts_added);
    fputs(",\n  \"optimal_value\": ", file);
    json_double(file, result->has_optimal, result->optimal_value);
    fputs(",\n  \"matches_optimum\": ", file);
    if (!result->has_matches_optimum) fputs("null", file);
    else fputs(result->matches_optimum ? "true" : "false", file);
    fputs("\n}", file);
    fclose(file);
    free(path);
}

/* ===================================================================== */
/* Driver                                                                 */
/* ===================================================================== */

static bool run_instance(const tw_args *args, GRBenv *env, const char *name,
                         const tw_manifest *manifest)
{
    char *path = resolve_instance_path(name, args->instances_root);
    char *stem = path_stem(path);
    const tw_manifest_entry *entry = manifest_lookup(manifest, stem);

    bool has_optimal = args->has_optimal;
    double optimal = args->optimal;
    if (!has_optimal && entry && entry->has_optimal) {
        has_optimal = true;
        optimal = entry->optimal;
    }

    const char *named = args->distance;
    if (!named && entry && entry->convention) named = entry->convention;
    if (!named) named = TW_DEFAULT_DISTANCE;
    tw_distance convention;
    if (!tw_distance_parse(named, &convention))
        bc_die("%s: distance convention \"%s\" is not one this loader implements "
               "(trunc1, exact). A value computed under another convention cannot be "
               "compared against this run", stem, named);

    int64_t vehicles = args->vehicles;
    if (vehicles <= 0 && entry) vehicles = entry->vehicles;
    if (vehicles <= 0)
        bc_die("no fleet size for %s: pass --vehicles, or give the instance a "
               "\"vehicles\" (or \"k\") column in manifest.csv. The fleet is fixed by "
               "an equality in this model, so it cannot be inferred", stem);

    tw_instance instance;
    tw_instance_load(path, convention, &instance);
    if (vehicles < 1 || vehicles > instance.n_customers)
        bc_die("%s: K = %lld is outside 1..%lld; every route must serve at least one "
               "customer and every customer is served once", stem, (long long)vehicles,
               (long long)instance.n_customers);
    tw_arcs arcs;
    tw_preprocess_arcs(&instance, &arcs);

    bc_instance aggregated;
    aggregated_instance(&instance, &aggregated);
    const bool learned = strcmp(args->separator, "sagseptts") == 0;
    bc_instance_validate(&aggregated, learned, "bc_vrptw");

    const double setup_start = bc_seconds();
    bc_separator *separator;
    if (!learned) {
        separator = bc_separator_cvrpsep_create(&aggregated, args->options.cuts_per_round,
                                                args->options.min_violation,
                                                args->options.tree_commit);
    } else {
        separator = bc_separator_sagsep_create(&aggregated, args->onnx, args->device,
                                               args->size_preset, args->options.threads,
                                               args->options.tree_commit);
    }

    char detail[TW_DETAIL_MAX];
    tw_result result;
    tw_solve(env, &instance, &arcs, vehicles, separator, &args->options, has_optimal,
             optimal, setup_start, detail, &result);
    print_summary(&result);
    if (args->out) write_result_json(args->out, stem, &result);

    const bool mismatch = result.has_matches_optimum && !result.matches_optimum;
    separator->destroy(separator);
    bc_instance_free(&aggregated);
    tw_arcs_free(&arcs);
    tw_instance_free(&instance);
    free(stem);
    free(path);
    return mismatch;
}

int main(int argc, char **argv)
{
    tw_args args;
    parse_args(argc, argv, &args);

    tw_manifest manifest;
    memset(&manifest, 0, sizeof manifest);
    char *default_manifest = NULL;
    const char *manifest_path = args.manifest;
    if (!manifest_path && args.instances_root && *args.instances_root) {
        default_manifest = join_path(args.instances_root, "manifest.csv", "");
        if (file_exists(default_manifest)) manifest_path = default_manifest;
    }
    if (manifest_path) manifest_read(manifest_path, &manifest);

    GRBenv *env = NULL;
    tw_grb_check(GRBloadenv(&env, NULL), NULL, "GRBloadenv");

    int mismatches = 0;
    for (int i = 0; i < args.n_instances; ++i) {
        if (run_instance(&args, env, args.instances[i], &manifest)) mismatches += 1;
    }

    GRBfreeenv(env);
    manifest_free(&manifest);
    free(default_manifest);
    free(args.instances);
    return mismatches ? 1 : 0;
}
