#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "bc.h"
#include "cmst_io.h"
#include "formulation.h"
#include "instance.h"
#include "separator_cvrpsep.h"
#include "separator_sagsep.h"
#include "util.h"

typedef struct {
    char   *name;
    int64_t capacity;
    double  optimum;
    bool    proved;
} bc_manifest_entry;

typedef struct {
    bc_manifest_entry *entries;
    int64_t count;
    int64_t capacity;
} bc_manifest;

typedef struct {
    const char *separator;
    const char **instances;
    int          n_instances;
    const char *instances_root;
    const char *manifest;
    int64_t     capacity;
    double      optimal;
    bool        has_optimal;
    const char *onnx;
    const char *device;
    const char *size_preset;
    const char *out;
    bc_options  options;
} bc_args;

/* ===================================================================== */
/* Command line                                                           */
/* ===================================================================== */

static void usage(FILE *stream)
{
    fputs(
"usage: bc_cmst --separator {cvrpsep,sagseptts} --instance NAME [NAME ...] [options]\n"
"\n"
"Branch-and-cut for the capacitated minimum spanning tree, with the capacity\n"
"cut as the only cut family. The two runs share the model, the Gurobi\n"
"configuration, the root and tree schedule, the exact rescoring and the\n"
"admission rule; what changes is the generator, which keeps its own proposal\n"
"budget and internal gates. Where the\n"
"heuristic candidate sets come from; connectivity is separated exactly in both.\n"
"\n"
"  --separator {cvrpsep,sagseptts}  cut source (required)\n"
"  --instance NAME [NAME ...]       instance names resolved under --instances-root,\n"
"                                   or paths to .cmst files (required)\n"
"  --instances-root DIR             directory holding the .cmst files\n"
"                                   (default: $SAGSEP_CMST_INSTANCES)\n"
"  --capacity Q                     one of the instance's CAPACITIES; required\n"
"                                   when it lists several\n"
"  --manifest FILE                  CSV with instance,capacity,optimum,status\n"
"                                   columns (default: manifest.csv next to the\n"
"                                   instances)\n"
"  --optimal VALUE                  known optimum of a single instance\n"
"  --onnx PATH                      sagseptts only: exported policy graph\n"
"                                   (default: $SAGSEP_ONNX)\n"
"  --device NAME                    sagseptts only: cpu | cuda | cuda:<id>\n"
"                                   (default: cpu)\n"
"  --size-preset TAG                sagseptts only: n50 ... n1000, or a *.toml\n"
"                                   config (default: by instance size)\n"
"  --time-limit SECONDS             budget per instance, root loop included\n"
"                                   (default: 300; 0 = none)\n"
"  --mip-gap G                      relative MIP gap (default: 0.0001)\n"
"  --threads N                      Gurobi Threads, and the learned separator's\n"
"                                   ONNX Runtime intra-op threads (default: 1)\n"
"  --seed N                         Gurobi Seed (default: 0)\n"
"  --root-rounds N                  maximum root cutting-plane rounds\n"
"                                   (default: 100; 0 skips the root phase)\n"
"  --cuts-per-round N               cap on heuristic rows per round, applied to\n"
"                                   both cut sources; also CVRPSEP's own\n"
"                                   MaxNoOfCuts. The exact max-flow rows are not\n"
"                                   capped (default: 100)\n"
"  --min-violation V                the CMST violation a row must exceed, applied\n"
"                                   to both cut sources after rescoring; also\n"
"                                   CVRPSEP's own EpsViolation, which it scores\n"
"                                   with the CVRP violation (default: 1e-06)\n"
"  --tree-commit {offered,root}     what a MIPNODE batch tells the separators:\n"
"                                   offered = the rows the driver offered through\n"
"                                   GRBcbcut, which is CVRPSEP's own convention;\n"
"                                   root = nothing, so each records only the\n"
"                                   permanent root rows. Applied to both sources\n"
"                                   (default: offered)\n"
"  --usercut-max-nodes N            separate user cuts while the node count is at\n"
"                                   most this (default: 15; 0 = off)\n"
"  --padded-support                 hand the generator the padded CVRP-shaped\n"
"                                   support instead of the CMST point itself\n"
"  --connectivity-only              never call the generator: exact connectivity\n"
"                                   separation and the MIPSOL audit only\n"
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
    if (end == text || *end != '\0')
        bc_die("%s expects an integer, got \"%s\"", flag, text);
    return value;
}

static double parse_double_arg(const char *text, const char *flag)
{
    char *end = NULL;
    const double value = strtod(text, &end);
    if (end == text || *end != '\0')
        bc_die("%s expects a number, got \"%s\"", flag, text);
    return value;
}

static void parse_args(int argc, char **argv, bc_args *args)
{
    memset(args, 0, sizeof *args);
    args->instances_root = getenv("SAGSEP_CMST_INSTANCES");
    args->onnx = getenv("SAGSEP_ONNX");
    args->device = "cpu";
    args->options.time_limit = 300.0;
    args->options.mip_gap = 1e-4;
    args->options.threads = 1;
    args->options.seed = 0;
    args->options.root_rounds = 100;
    args->options.cuts_per_round = 100;
    args->options.min_violation = 1e-6;
    args->options.usercut_max_nodes = 15;
    args->options.tree_commit = "offered";
    args->options.padded_support = false;
    args->options.connectivity_only = false;
    args->options.verbose = true;
    args->instances =
        (const char **)bc_malloc((size_t)argc * sizeof *args->instances);

    for (int i = 1; i < argc; ++i) {
        const char *flag = argv[i];
        if (strcmp(flag, "-h") == 0 || strcmp(flag, "--help") == 0) {
            usage(stdout);
            free(args->instances);
            exit(EXIT_SUCCESS);
        } else if (strcmp(flag, "--separator") == 0) {
            args->separator = need_value(argc, argv, &i);
        } else if (strcmp(flag, "--instance") == 0) {
            if (i + 1 >= argc)
                bc_die("--instance needs at least one value");
            while (i + 1 < argc && strncmp(argv[i + 1], "--", 2) != 0)
                args->instances[args->n_instances++] = argv[++i];
        } else if (strcmp(flag, "--instances-root") == 0) {
            args->instances_root = need_value(argc, argv, &i);
        } else if (strcmp(flag, "--capacity") == 0) {
            args->capacity =
                parse_long_arg(need_value(argc, argv, &i), flag);
        } else if (strcmp(flag, "--manifest") == 0) {
            args->manifest = need_value(argc, argv, &i);
        } else if (strcmp(flag, "--optimal") == 0) {
            args->optimal =
                parse_double_arg(need_value(argc, argv, &i), flag);
            args->has_optimal = true;
        } else if (strcmp(flag, "--onnx") == 0) {
            args->onnx = need_value(argc, argv, &i);
        } else if (strcmp(flag, "--device") == 0) {
            args->device = need_value(argc, argv, &i);
        } else if (strcmp(flag, "--size-preset") == 0) {
            args->size_preset = need_value(argc, argv, &i);
        } else if (strcmp(flag, "--out") == 0) {
            args->out = need_value(argc, argv, &i);
        } else if (strcmp(flag, "--time-limit") == 0) {
            args->options.time_limit =
                parse_double_arg(need_value(argc, argv, &i), flag);
        } else if (strcmp(flag, "--mip-gap") == 0) {
            args->options.mip_gap =
                parse_double_arg(need_value(argc, argv, &i), flag);
        } else if (strcmp(flag, "--threads") == 0) {
            args->options.threads =
                (int)parse_long_arg(need_value(argc, argv, &i), flag);
        } else if (strcmp(flag, "--seed") == 0) {
            args->options.seed =
                (int)parse_long_arg(need_value(argc, argv, &i), flag);
        } else if (strcmp(flag, "--root-rounds") == 0) {
            args->options.root_rounds =
                (int)parse_long_arg(need_value(argc, argv, &i), flag);
        } else if (strcmp(flag, "--cuts-per-round") == 0) {
            args->options.cuts_per_round =
                (int)parse_long_arg(need_value(argc, argv, &i), flag);
        } else if (strcmp(flag, "--min-violation") == 0) {
            args->options.min_violation =
                parse_double_arg(need_value(argc, argv, &i), flag);
        } else if (strcmp(flag, "--tree-commit") == 0) {
            const char *value = need_value(argc, argv, &i);
            if (strcmp(value, "root") != 0 && strcmp(value, "offered") != 0)
                bc_die("--tree-commit must be root or offered, got \"%s\"", value);
            args->options.tree_commit = value;
        } else if (strcmp(flag, "--usercut-max-nodes") == 0) {
            args->options.usercut_max_nodes =
                (int)parse_long_arg(need_value(argc, argv, &i), flag);
        } else if (strcmp(flag, "--padded-support") == 0) {
            args->options.padded_support = true;
        } else if (strcmp(flag, "--connectivity-only") == 0) {
            args->options.connectivity_only = true;
        } else if (strcmp(flag, "--quiet") == 0) {
            args->options.verbose = false;
        } else {
            usage(stderr);
            bc_die("unrecognised argument: %s", flag);
        }
    }

    if (!args->separator)
        bc_die("--separator is required (cvrpsep or sagseptts)");
    if (strcmp(args->separator, "cvrpsep") != 0
        && strcmp(args->separator, "sagseptts") != 0)
        bc_die(
            "--separator must be cvrpsep or sagseptts, got \"%s\"",
            args->separator);
    if (args->n_instances == 0)
        bc_die("--instance is required");
    if (args->has_optimal && args->n_instances > 1)
        bc_die(
            "--optimal applies to a single instance; "
            "use the manifest for a batch");
    if (args->options.time_limit < 0.0)
        bc_die(
            "--time-limit must be non-negative (0 = none), got %g",
            args->options.time_limit);
    if (!(args->options.mip_gap >= 0.0
          && args->options.mip_gap < 1.0))
        bc_die(
            "--mip-gap must be in [0, 1), got %g",
            args->options.mip_gap);
    if (args->options.threads < 1)
        bc_die(
            "--threads must be at least 1, got %d",
            args->options.threads);
    if (args->options.root_rounds < 0)
        bc_die(
            "--root-rounds must be non-negative, got %d",
            args->options.root_rounds);
    if (args->options.cuts_per_round <= 0)
        bc_die(
            "--cuts-per-round must be positive, got %d",
            args->options.cuts_per_round);
    if (args->options.min_violation < 0.0)
        bc_die(
            "--min-violation must be non-negative, got %g",
            args->options.min_violation);
    if (args->options.usercut_max_nodes < 0)
        bc_die(
            "--usercut-max-nodes must be non-negative, got %d",
            args->options.usercut_max_nodes);
    if (args->capacity < 0)
        bc_die(
            "--capacity must be positive, got %lld",
            (long long)args->capacity);
}

/* ===================================================================== */
/* Instance resolution and the manifest                                   */
/* ===================================================================== */

static bool has_cmst_suffix(const char *name)
{
    const size_t length = strlen(name);
    return length >= 5 && strcmp(name + length - 5, ".cmst") == 0;
}

static char *resolve_instance_path(
    const char *name,
    const char *instances_root
)
{
    if (has_cmst_suffix(name)) {
        char *path = (char *)bc_malloc(strlen(name) + 1);
        memcpy(path, name, strlen(name) + 1);
        return path;
    }
    if (!instances_root || !*instances_root)
        bc_die(
            "cannot resolve instance \"%s\": pass a path to a .cmst file, set "
            "--instances-root, or set SAGSEP_CMST_INSTANCES",
            name);
    const size_t length =
        strlen(instances_root) + 1 + strlen(name) + 5;
    char *path = (char *)bc_malloc(length + 1);
    snprintf(path, length + 1, "%s/%s.cmst", instances_root, name);
    return path;
}

static char *trim(char *text)
{
    while (*text == ' ' || *text == '\t'
           || *text == '\r' || *text == '\n')
        ++text;
    char *end = text + strlen(text);
    while (end > text
           && (end[-1] == ' ' || end[-1] == '\t'
               || end[-1] == '\r' || end[-1] == '\n'))
        --end;
    *end = '\0';
    return text;
}

static void manifest_read(const char *path, bc_manifest *manifest)
{
    memset(manifest, 0, sizeof *manifest);
    FILE *file = fopen(path, "r");
    if (!file) bc_die("cannot open manifest %s", path);

    char *line = NULL;
    size_t line_capacity = 0;
    int instance_column = -1;
    int capacity_column = -1;
    int optimum_column = -1;
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
                if (strcmp(fields[c], "instance") == 0)
                    instance_column = c;
                else if (strcmp(fields[c], "capacity") == 0)
                    capacity_column = c;
                else if (strcmp(fields[c], "optimum") == 0)
                    optimum_column = c;
                else if (strcmp(fields[c], "status") == 0)
                    status_column = c;
            }
            if (instance_column < 0 || capacity_column < 0)
                bc_die(
                    "%s: manifest needs \"instance\" and "
                    "\"capacity\" columns",
                    path);
            continue;
        }
        if (instance_column >= n_fields
            || !*fields[instance_column])
            continue;
        if (capacity_column >= n_fields
            || !*fields[capacity_column])
            continue;

        if (manifest->count == manifest->capacity) {
            manifest->capacity =
                manifest->capacity ? manifest->capacity * 2 : 64;
            manifest->entries = (bc_manifest_entry *)bc_realloc(
                manifest->entries,
                (size_t)manifest->capacity
                    * sizeof *manifest->entries);
        }
        bc_manifest_entry *entry =
            &manifest->entries[manifest->count++];
        memset(entry, 0, sizeof *entry);
        const size_t name_length =
            strlen(fields[instance_column]);
        entry->name = (char *)bc_malloc(name_length + 1);
        memcpy(
            entry->name, fields[instance_column], name_length + 1);
        entry->capacity = parse_long_arg(
            fields[capacity_column], "manifest capacity");

        const bool proved =
            status_column < 0 || status_column >= n_fields
            || !*fields[status_column]
            || strcmp(fields[status_column], "proved") == 0;
        if (proved
            && optimum_column >= 0
            && optimum_column < n_fields
            && *fields[optimum_column]) {
            entry->optimum = parse_double_arg(
                fields[optimum_column], "manifest optimum");
            entry->proved = true;
        }
    }
    free(line);
    fclose(file);
}

static void manifest_free(bc_manifest *manifest)
{
    for (int64_t i = 0; i < manifest->count; ++i)
        free(manifest->entries[i].name);
    free(manifest->entries);
    memset(manifest, 0, sizeof *manifest);
}

static const bc_manifest_entry *manifest_lookup(
    const bc_manifest *manifest,
    const char *name,
    int64_t capacity
)
{
    for (int64_t i = 0; i < manifest->count; ++i) {
        if (manifest->entries[i].capacity == capacity
            && strcmp(manifest->entries[i].name, name) == 0)
            return &manifest->entries[i];
    }
    return NULL;
}

/* ===================================================================== */
/* Reporting                                                              */
/* ===================================================================== */

static const char *match_label(const bc_result *result)
{
    if (!result->has_matches_optimum)
        return "opt-unknown";
    return result->matches_optimum ? "opt-match" : "OPT-MISMATCH";
}

static void print_summary(const bc_result *result)
{
    char objective[32] = "-";
    char bound[32] = "-";
    char root[32] = "-";
    if (result->has_objective)
        snprintf(
            objective, sizeof objective, "%.1f", result->objective);
    if (result->has_best_bound)
        snprintf(
            bound, sizeof bound, "%.1f", result->best_bound);
    if (result->has_root_bound)
        snprintf(
            root, sizeof root, "%.1f", result->root_bound);

    printf(
        "%s-Q%lld  sep=%s  %s  obj=%s (%s)  bound=%s  root=%s  "
        "nodes=%.0f  lazy=%ld  usercuts=%ld  rootcuts=%d  "
        "connectivity_cuts=%ld  padding=%s%s  %.2fs\n",
        result->instance_name, (long long)result->capacity,
        result->separator, result->status_name, objective,
        match_label(result), bound, root, result->node_count,
        result->callback.lazy_cuts_added,
        result->callback.mipnode_cuts_added,
        result->root_cuts, result->connectivity_cuts,
        result->padded_support ? "on" : "off",
        result->connectivity_only ? "  connectivity-only" : "",
        result->runtime);
    fflush(stdout);
}

static void json_string(FILE *file, const char *text)
{
    fputc('"', file);
    for (const unsigned char *p =
             (const unsigned char *)text;
         *p;
         ++p) {
        if (*p == '"' || *p == '\\')
            fprintf(file, "\\%c", *p);
        else if (*p < 0x20)
            fprintf(file, "\\u%04x", *p);
        else
            fputc((int)*p, file);
    }
    fputc('"', file);
}

static void json_double(FILE *file, bool present, double value)
{
    if (!present) {
        fputs("null", file);
        return;
    }
    char text[64];
    bc_format_double(text, sizeof text, value);
    fputs(text, file);
}

static void make_directories(const char *path)
{
    char *buffer = (char *)bc_malloc(strlen(path) + 1);
    memcpy(buffer, path, strlen(path) + 1);
    for (char *slash = buffer + 1; ; ++slash) {
        const bool last = (*slash == '\0');
        if (*slash != '/' && !last) continue;
        *slash = '\0';
        if (*buffer && mkdir(buffer, 0777) != 0) {
            struct stat info;
            if (stat(buffer, &info) != 0
                || !S_ISDIR(info.st_mode))
                bc_die(
                    "cannot create output directory %s", buffer);
        }
        if (last) break;
        *slash = '/';
    }
    free(buffer);
}

static void write_result_json(
    const char *directory,
    const bc_result *result
)
{
    make_directories(directory);
    const size_t length =
        strlen(directory) + strlen(result->instance_name)
        + strlen(result->separator) + 48;
    char *path = (char *)bc_malloc(length);
    snprintf(
        path, length, "%s/bc_%s_q%lld_%s.json",
        directory, result->instance_name,
        (long long)result->capacity, result->separator);
    FILE *file = fopen(path, "w");
    if (!file) bc_die("cannot write %s", path);

    fputs("{\n", file);
    fputs("  \"instance_name\": ", file);
    json_string(file, result->instance_name);
    fprintf(
        file, ",\n  \"n_customers\": %lld",
        (long long)result->n_customers);
    fprintf(
        file, ",\n  \"capacity\": %lld",
        (long long)result->capacity);
    fprintf(
        file, ",\n  \"root_degree\": %lld",
        (long long)result->root_degree);
    fputs(",\n  \"separator\": ", file);
    json_string(file, result->separator);
    fputs(",\n  \"separator_detail\": ", file);
    json_string(file, result->separator_detail);
    fprintf(file, ",\n  \"status\": %d", result->status);
    fputs(",\n  \"status_name\": ", file);
    json_string(file, result->status_name);
    fputs(",\n  \"objective\": ", file);
    json_double(
        file, result->has_objective, result->objective);
    fputs(",\n  \"best_bound\": ", file);
    json_double(
        file, result->has_best_bound, result->best_bound);
    fputs(",\n  \"gap\": ", file);
    json_double(file, result->has_gap, result->gap);
    fputs(",\n  \"runtime\": ", file);
    json_double(file, true, result->runtime);
    fputs(",\n  \"setup_time\": ", file);
    json_double(file, true, result->setup_time);
    fputs(",\n  \"node_count\": ", file);
    json_double(file, true, result->node_count);
    fprintf(file, ",\n  \"sol_count\": %d", result->sol_count);
    fputs(",\n  \"root_bound\": ", file);
    json_double(
        file, result->has_root_bound, result->root_bound);
    fputs(",\n  \"root_termination\": ", file);
    json_string(file, result->root_termination);
    fprintf(
        file, ",\n  \"root_bound_final\": %s",
        result->root_bound_final ? "true" : "false");
    fprintf(
        file, ",\n  \"root_rounds\": %d", result->root_rounds);
    fprintf(
        file, ",\n  \"root_cuts\": %d", result->root_cuts);
    fprintf(
        file, ",\n  \"generator_inner_rows\": %d",
        result->generator_inner_rows);
    fprintf(
        file, ",\n  \"generator_cutset_rows\": %d",
        result->generator_cutset_rows);
    fprintf(
        file, ",\n  \"connectivity_cuts\": %ld",
        result->connectivity_cuts);
    fputs(",\n  \"root_time\": ", file);
    json_double(file, true, result->root_time);
    fprintf(
        file, ",\n  \"padded_support\": %s",
        result->padded_support ? "true" : "false");
    fprintf(
        file, ",\n  \"connectivity_only\": %s",
        result->connectivity_only ? "true" : "false");
    fprintf(
        file, ",\n  \"padding_high_degree_max\": %lld",
        (long long)result->padding_high_degree_max);
    fputs(",\n  \"padding_max_clip\": ", file);
    json_double(file, true, result->padding_max_clip);
    fputs(",\n  \"padding_max_identity_gap\": ", file);
    json_double(file, true, result->padding_max_identity_gap);
    fprintf(
        file,
        ",\n  \"callback\": {\n"
        "    \"mipnode_calls\": %ld,\n"
        "    \"mipnode_cuts_added\": %ld,\n"
        "    \"mipsol_calls\": %ld,\n"
        "    \"lazy_cuts_added\": %ld,\n"
        "    \"lazy_connectivity_cuts\": %ld\n"
        "  }",
        result->callback.mipnode_calls,
        result->callback.mipnode_cuts_added,
        result->callback.mipsol_calls,
        result->callback.lazy_cuts_added,
        result->callback.lazy_connectivity_cuts);
    fputs(",\n  \"optimal_value\": ", file);
    json_double(
        file, result->has_optimal, result->optimal_value);
    fputs(",\n  \"matches_optimum\": ", file);
    if (!result->has_matches_optimum)
        fputs("null", file);
    else
        fputs(
            result->matches_optimum ? "true" : "false", file);
    fputs("\n}", file);
    fclose(file);
    free(path);
}

/* ===================================================================== */
/* Driver                                                                 */
/* ===================================================================== */

static bool run_instance(
    const bc_args *args,
    GRBenv *env,
    const char *name,
    const bc_manifest *manifest
)
{
    char *path =
        resolve_instance_path(name, args->instances_root);

    bc_instance instance;
    bc_cmst_load(path, args->capacity, &instance);
    const bool learned = strcmp(args->separator, "sagseptts") == 0;
    bc_instance_validate(&instance, learned, "bc_cmst");

    bool has_optimal = args->has_optimal;
    double optimal = args->optimal;
    if (!has_optimal) {
        const bc_manifest_entry *entry =
            manifest_lookup(
                manifest, instance.name, instance.capacity);
        if (entry && entry->proved) {
            has_optimal = true;
            optimal = entry->optimum;
        }
    }

    const double setup_start = bc_seconds();
    bc_separator *separator;
    if (!learned) {
        separator = bc_separator_cvrpsep_create(
            &instance, args->options.cuts_per_round,
            args->options.min_violation, args->options.tree_commit);
    } else {
        separator = bc_separator_sagsep_create(
            &instance, args->onnx, args->device,
            args->size_preset, args->options.threads, args->options.tree_commit);
    }

    char detail[BC_DETAIL_MAX];
    bc_result result;
    bc_solve(
        env, &instance, separator, &args->options,
        has_optimal, optimal, setup_start, detail, &result);
    print_summary(&result);
    if (args->out)
        write_result_json(args->out, &result);

    const bool mismatch =
        result.has_matches_optimum && !result.matches_optimum;
    separator->destroy(separator);
    bc_instance_free(&instance);
    free(path);
    return mismatch;
}

int main(int argc, char **argv)
{
    bc_args args;
    parse_args(argc, argv, &args);

    bc_manifest manifest;
    memset(&manifest, 0, sizeof manifest);
    char *default_manifest = NULL;
    const char *manifest_path = args.manifest;
    if (!manifest_path
        && args.instances_root
        && *args.instances_root) {
        const size_t length =
            strlen(args.instances_root) + 16;
        default_manifest = (char *)bc_malloc(length);
        snprintf(
            default_manifest, length,
            "%s/manifest.csv", args.instances_root);
        FILE *probe = fopen(default_manifest, "r");
        if (probe) {
            fclose(probe);
            manifest_path = default_manifest;
        }
    }
    if (manifest_path)
        manifest_read(manifest_path, &manifest);

    GRBenv *env = NULL;
    bc_grb_check(
        GRBloadenv(&env, NULL), NULL, "GRBloadenv");

    int mismatches = 0;
    for (int i = 0; i < args.n_instances; ++i) {
        if (run_instance(
                &args, env, args.instances[i], &manifest))
            mismatches += 1;
    }

    GRBfreeenv(env);
    manifest_free(&manifest);
    free(default_manifest);
    free(args.instances);
    return mismatches ? 1 : 0;
}
