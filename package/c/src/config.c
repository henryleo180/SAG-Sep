#include "sagsep.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Name tables                                                         */
/* ------------------------------------------------------------------ */

const char *const sag_family_names[SAG_FAMILY_COUNT] = { "sweep", "onion", "tts" };

const char *const sag_variant_names[SAG_VARIANT_COUNT] = {
    "winner",
    "threshold_fallback",
    "band_first",
    "lp_grow",
    "fcapfix",
    "old_repair",
    "adddrop",
    "growth"
};

int8_t sag_variant_from_str(const char *s, size_t len)
{
    for (int i = 0; i < SAG_VARIANT_COUNT; ++i) {
        size_t n = strlen(sag_variant_names[i]);
        if (n == len && memcmp(s, sag_variant_names[i], n) == 0)
            return (int8_t)i;
    }
    sag_die("unknown variant string '%.*s' -- the engine emits 8 variants "
            "(winner/threshold_fallback/band_first/lp_grow/fcapfix/old_repair/adddrop/growth); a "
            "new string means semantic drift; align first, then extend", (int)len, s);
}

const char *const sag_pipeline_timing_names[SAG_PIPE_T_COUNT] = {
    "family_decode_wall", "dedup", "scoring", "selector", "materialize"
};

const char *const sag_loop_timing_names[] = {
    "lp_solve", "solution_extract", "support_extract", "policy_call", "cvrpsep_wait",
    "iteration_hook", "candidate_prepare", "candidate_violation", "candidate_limit",
    "constraint_build", "constraint_add", "model_update", "iteration_total", NULL
};

const char *const sag_policy_timing_names[] = {
    "total", "prepare_total", "graph_build", "batch_materialize", "vehicle_unpack",
    "forward", "prob_export", "threshold", "subset_materialize", "rhs", "cut_value",
    "unified_prepare", "unified_sweep", "unified_onion", "unified_decode_wall",
    "unified_dedup", "unified_scoring", "unified_selector", "unified_build",
    "postprocess_total", NULL
};

/* ------------------------------------------------------------------ */
/* Frozen constants                                                     */
/* ------------------------------------------------------------------ */

static const char *sag_sota_run_dir(void)
{
    const char *dir = getenv("SAGSEP_RUN_DIR");
    return dir ? dir : "";
}

const char *const sag_preset_sizes[] = {
    "n50", "n75", "n100", "n200", "n300", "n400", "n500", "n750", "n1000", NULL
};

static const sag_source_quota QUOTAS_FROZEN[4] = {
    { "fcapfix", 4 }, { "old_repair", 40 }, { "lp_grow", 80 }, { "adddrop", 40 }
};

typedef struct {
    const char *size;
    int64_t     max_prefix;
} preset_row;

static const preset_row PRESET_ROWS[] = {
    { "n50",    90 },
    { "n75",    90 },
    { "n100",   90 },
    { "n200",  180 },
    { "n300",  270 },
    { "n400",  360 },
    { "n500",  360 },
    { "n750",  360 },
    { "n1000", 360 },
};

/* ------------------------------------------------------------------ */
/* Construction                                                         */
/* ------------------------------------------------------------------ */

static void set_str(char *dst, size_t cap, const char *src)
{
    size_t n = strlen(src);
    if (n + 1 > cap)
        sag_die("config: string '%s' does not fit in %zu bytes", src, cap);
    memcpy(dst, src, n + 1);
}

static sag_tts_config tts_frozen(int64_t max_prefix)
{
    sag_tts_config t;
    memset(&t, 0, sizeof t);
    t.enabled = true;
    t.max_prefix = max_prefix;
    t.min_size = 2;
    t.lp_grow_seeds = 16;
    t.lp_grow_native_rules = true;
    t.compress_mode = SAG_COMPRESS_FULL;
    t.fcapfix_rounds = 3;
    t.fcapfix_seed_limit = 32;
    t.fcapfix_seed_mode = SAG_FCAPFIX_SEED_MODEL;
    t.old_repair_bank_size = 100000;
    t.old_repair_top_b = 16;
    t.adddrop_seed_old_bank = true;
    t.adddrop_seed_source_count = 0;
    t.adddrop_seed_limit = 0;
    t.score_mode = SAG_SCORE_EFFICACY;
    t.min_violation = 1e-4;
    t.top_b = 120;
    t.top_l = 24;
    t.max_flips = 40;
    t.min_gain = 1e-6;
    t.ls_neighborhood = SAG_LS_BOTH;
    t.ls_swap = true;
    t.ls_swap_max_pairs = 250000;
    t.max_size_frac = 1.0;
    t.seed = 19;
    t.rng_mode = SAG_RNG_DERIVED;
    t.flow_enabled = true;
    memcpy(t.source_quotas, QUOTAS_FROZEN, sizeof QUOTAS_FROZEN);
    t.source_quota_count = (int64_t)(sizeof QUOTAS_FROZEN / sizeof QUOTAS_FROZEN[0]);
    t.per_vehicle_quota = 200;
    t.global_quota = 200;
    t.near_duplicate_hamming = 3;
    t.diversity_weight = 0.15;
    t.vehicle_tighten_start = 8;
    t.vehicle_tighten_power = 0.75;
    t.vehicle_tighten_min_scale = 0.4;
    t.max_replicas = 1;
    return t;
}

static sag_growth_config growth_frozen(void)
{
    sag_growth_config g;
    memset(&g, 0, sizeof g);
    g.enabled = false;
    g.rollouts = 48;
    g.horizon = 400;
    g.bands[0] = 2; g.bands[1] = 3; g.n_bands = 2;
    g.prefix_tabu = 3;
    g.restart_tabu = true;
    g.max_rows = 0;
    g.diag = false;
    return g;
}

int64_t sag_preset_max_prefix_for_customers(int64_t n_customers)
{
    const preset_row *best = NULL;
    int64_t best_gap = 0;
    for (size_t i = 0; i < sizeof PRESET_ROWS / sizeof PRESET_ROWS[0]; ++i) {
        const int64_t size = strtoll(PRESET_ROWS[i].size + 1, NULL, 10);
        const int64_t gap = n_customers > size ? n_customers - size : size - n_customers;
        if (!best || gap < best_gap) { best = &PRESET_ROWS[i]; best_gap = gap; }
    }
    return best->max_prefix;
}

sag_config sag_config_preset(const char *size)
{
    const preset_row *row = NULL;
    for (size_t i = 0; i < sizeof PRESET_ROWS / sizeof PRESET_ROWS[0]; ++i) {
        if (strcmp(PRESET_ROWS[i].size, size) == 0) { row = &PRESET_ROWS[i]; break; }
    }
    if (!row) {
        fprintf(stderr, "valid presets:");
        for (const char *const *s = sag_preset_sizes; *s; ++s) fprintf(stderr, " %s", *s);
        fputc('\n', stderr);
        sag_die("unknown size preset '%s'", size);
    }

    sag_config cfg;
    memset(&cfg, 0, sizeof cfg);

    set_str(cfg.model.run_dir, sizeof cfg.model.run_dir, sag_sota_run_dir());
    set_str(cfg.model.device, sizeof cfg.model.device, "cuda:0");

    cfg.features.partial_connect_cross_depot = true;

    cfg.lp.gurobi_method = 1;
    cfg.lp.gurobi_threads = 1;
    cfg.lp.lp_support_edge_threshold = 1e-6;
    cfg.lp.time_limit = 3600.0;

    cfg.loop.max_iter = -1;
    cfg.loop.min_violation = 1e-6;
    cfg.loop.cut_violation_ratio = 0.0;
    cfg.loop.dup_check = true;

    cfg.sweep.threshold = 0.5;
    cfg.sweep.min_size = 2;
    cfg.sweep.min_violation = 1e-6;

    cfg.growth = growth_frozen();

    cfg.onion.mode = SAG_ONION_MODE_MEAN;
    cfg.onion.m_max_cap = 0;
    cfg.onion.min_violation = 1e-4;
    cfg.onion.threshold = 0.5;
    cfg.onion.skip_below_M = 0;
    cfg.onion.cutpoint = SAG_ONION_CUT_BAND_FIRST;

    cfg.tts = tts_frozen(row->max_prefix);

    cfg.budget.max_rows = -1;
    cfg.budget.max_nnz = -1;
    cfg.compute_cut_values = false;
    cfg.mbp_seed = 0;
    set_str(cfg.size, sizeof cfg.size, row->size);

    sag_config_check_frozen(&cfg);
    return cfg;
}

void sag_config_check_frozen(const sag_config *cfg)
{
    if (cfg->onion.mode != SAG_ONION_MODE_MEAN)
        sag_die("config: onion mode is frozen to 'mean' (no other mode is implemented)");
    if (cfg->onion.cutpoint != SAG_ONION_CUT_BAND_FIRST)
        sag_die("config: onion cutpoint is frozen to 'band_first'");
    if (cfg->tts.compress_mode != SAG_COMPRESS_FULL)
        sag_die("config: tts.compress_mode is frozen to 'full'");
    if (cfg->tts.score_mode != SAG_SCORE_EFFICACY)
        sag_die("config: tts.score_mode is frozen to 'efficacy'");
    if (cfg->tts.ls_neighborhood != SAG_LS_BOTH)
        sag_die("config: tts.ls_neighborhood is frozen to 'both'");
    if (cfg->tts.rng_mode != SAG_RNG_DERIVED)
        sag_die("config: tts.rng_mode is frozen to 'derived'");
    if (cfg->tts.fcapfix_seed_mode != SAG_FCAPFIX_SEED_MODEL)
        sag_die("config: tts.fcapfix_seed_mode is frozen to 'model'");
    if (cfg->tts.adddrop_seed_source_count != 0)
        sag_die("config: tts.adddrop_seed_sources is frozen to the empty tuple "
                "(adddrop seeds come from the old-cut bank only)");
    if (!cfg->tts.ls_swap)
        sag_die("config: tts.ls_swap must be true in SAG-Sep (no-swap local search deleted)");
    if (!cfg->tts.flow_enabled)
        sag_die("config: tts.flow_enabled must be true in SAG-Sep (flow bypass deleted)");
    if (cfg->tts.old_repair_bank_size <= 0)
        sag_die("config: tts.old_repair_bank_size must be > 0 in SAG-Sep "
                "(the empty-bank branch is deleted), got %" PRId64,
                cfg->tts.old_repair_bank_size);
    if (cfg->tts.old_repair_top_b < 0)
        sag_die("config: tts.old_repair_top_b must be >= 0, got %" PRId64,
                cfg->tts.old_repair_top_b);
    if (cfg->tts.adddrop_seed_limit != 0)
        sag_die("config: tts.adddrop_seed_limit must be 0 (full pool) in SAG-Sep, got %" PRId64,
                cfg->tts.adddrop_seed_limit);
    if (!(cfg->tts.max_size_frac > 0.0 && cfg->tts.max_size_frac <= 1.0))
        sag_die("config: tts.max_size_frac must be in (0,1], got %g", cfg->tts.max_size_frac);
    if (!(cfg->tts.vehicle_tighten_min_scale > 0.0 && cfg->tts.vehicle_tighten_min_scale <= 1.0))
        sag_die("config: tts.vehicle_tighten_min_scale must be in (0,1], got %g",
                cfg->tts.vehicle_tighten_min_scale);
    if (cfg->compute_cut_values)
        sag_die("config: compute_cut_values is frozen to false (the reference arm hardcodes it)");
    if (cfg->lp.gurobi_method != 1 || cfg->lp.gurobi_threads != 1)
        sag_die("config: LP is frozen to Method=1 / Threads=1 (bit-for-bit reproducibility)");
    if (cfg->tts.source_quota_count > SAG_MAX_SOURCE_QUOTAS)
        sag_die("config: too many source quotas (%" PRId64 ")", cfg->tts.source_quota_count);
    if (cfg->tts.max_replicas < 0)
        sag_die("config: tts.max_replicas must be >= 0 (0 = all K), got %" PRId64,
                cfg->tts.max_replicas);
}

/* ------------------------------------------------------------------ */
/* Printing (smoke test: compare against the reference preset)          */
/* ------------------------------------------------------------------ */

static const char *yn(bool b) { return b ? "true" : "false"; }

static void print_opt_i64(FILE *out, const char *key, int64_t value)
{
    if (value < 0) fprintf(out, "  %-28s None\n", key);
    else fprintf(out, "  %-28s %" PRId64 "\n", key, value);
}

void sag_config_print(FILE *out, const sag_config *cfg)
{
    fprintf(out, "[preset] %s\n", cfg->size);
    fprintf(out, "[model]\n");
    fprintf(out, "  %-28s %s\n", "run_dir", cfg->model.run_dir);
    fprintf(out, "  %-28s %s\n", "device", cfg->model.device);

    fprintf(out, "[features]\n");
    fprintf(out, "  %-28s %s\n", "partial_connect_cross_depot",
            yn(cfg->features.partial_connect_cross_depot));

    fprintf(out, "[lp]\n");
    fprintf(out, "  %-28s %" PRId64 "\n", "gurobi_method", cfg->lp.gurobi_method);
    fprintf(out, "  %-28s %" PRId64 "\n", "gurobi_threads", cfg->lp.gurobi_threads);
    fprintf(out, "  %-28s %g\n", "lp_support_edge_threshold", cfg->lp.lp_support_edge_threshold);
    fprintf(out, "  %-28s %g\n", "time_limit", cfg->lp.time_limit);

    fprintf(out, "[loop]\n");
    print_opt_i64(out, "max_iter", cfg->loop.max_iter);
    fprintf(out, "  %-28s %g\n", "min_violation", cfg->loop.min_violation);
    fprintf(out, "  %-28s %g\n", "cut_violation_ratio", cfg->loop.cut_violation_ratio);
    fprintf(out, "  %-28s %s\n", "dup_check", yn(cfg->loop.dup_check));

    fprintf(out, "[sweep]\n");
    fprintf(out, "  %-28s %g\n", "threshold", cfg->sweep.threshold);
    fprintf(out, "  %-28s %" PRId64 "\n", "min_size", cfg->sweep.min_size);
    fprintf(out, "  %-28s %g\n", "min_violation", cfg->sweep.min_violation);

    fprintf(out, "[growth]\n");
    fprintf(out, "  %-28s %s\n", "enabled", yn(cfg->growth.enabled));
    fprintf(out, "  %-28s %" PRId64 "\n", "rollouts", cfg->growth.rollouts);
    fprintf(out, "  %-28s %" PRId64 "\n", "horizon", cfg->growth.horizon);
    fprintf(out, "  %-28s ", "bands");
    for (int64_t i = 0; i < cfg->growth.n_bands; ++i)
        fprintf(out, "%s%" PRId64, i ? "," : "", cfg->growth.bands[i]);
    fputc('\n', out);
    fprintf(out, "  %-28s %" PRId64 "\n", "prefix_tabu", cfg->growth.prefix_tabu);
    fprintf(out, "  %-28s %s\n", "restart_tabu", yn(cfg->growth.restart_tabu));
    fprintf(out, "  %-28s %" PRId64 "\n", "max_rows", cfg->growth.max_rows);
    fprintf(out, "  %-28s %s\n", "diag", yn(cfg->growth.diag));

    fprintf(out, "[onion]\n");
    fprintf(out, "  %-28s %s\n", "mode", "mean");
    fprintf(out, "  %-28s %s\n", "cutpoint", "band_first");
    fprintf(out, "  %-28s %" PRId64 "\n", "m_max_cap", cfg->onion.m_max_cap);
    fprintf(out, "  %-28s %g\n", "min_violation", cfg->onion.min_violation);
    fprintf(out, "  %-28s %g\n", "threshold", cfg->onion.threshold);
    fprintf(out, "  %-28s %" PRId64 "\n", "skip_below_M", cfg->onion.skip_below_M);

    const sag_tts_config *t = &cfg->tts;
    fprintf(out, "[tts]\n");
    fprintf(out, "  %-28s %s\n", "enabled", yn(t->enabled));
    fprintf(out, "  %-28s %" PRId64 "\n", "max_prefix", t->max_prefix);
    fprintf(out, "  %-28s %" PRId64 "\n", "min_size", t->min_size);
    fprintf(out, "  %-28s %" PRId64 "\n", "lp_grow_seeds", t->lp_grow_seeds);
    fprintf(out, "  %-28s %s\n", "lp_grow_native_rules", yn(t->lp_grow_native_rules));
    fprintf(out, "  %-28s %s\n", "compress_mode", "full");
    fprintf(out, "  %-28s %" PRId64 "\n", "fcapfix_rounds", t->fcapfix_rounds);
    fprintf(out, "  %-28s %" PRId64 "\n", "fcapfix_seed_limit", t->fcapfix_seed_limit);
    fprintf(out, "  %-28s %s\n", "fcapfix_seed_mode", "model");
    fprintf(out, "  %-28s %" PRId64 "\n", "old_repair_bank_size", t->old_repair_bank_size);
    fprintf(out, "  %-28s %" PRId64 "\n", "old_repair_top_b", t->old_repair_top_b);
    fprintf(out, "  %-28s %s\n", "adddrop_seed_old_bank", yn(t->adddrop_seed_old_bank));
    fprintf(out, "  %-28s %" PRId64 "\n", "adddrop_seed_limit", t->adddrop_seed_limit);
    fprintf(out, "  %-28s %s\n", "score_mode", "efficacy");
    fprintf(out, "  %-28s %g\n", "min_violation", t->min_violation);
    fprintf(out, "  %-28s %" PRId64 "\n", "top_b", t->top_b);
    fprintf(out, "  %-28s %" PRId64 "\n", "top_l", t->top_l);
    fprintf(out, "  %-28s %" PRId64 "\n", "max_flips", t->max_flips);
    fprintf(out, "  %-28s %g\n", "min_gain", t->min_gain);
    fprintf(out, "  %-28s %s\n", "ls_neighborhood", "both");
    fprintf(out, "  %-28s %s\n", "ls_swap", yn(t->ls_swap));
    fprintf(out, "  %-28s %" PRId64 "\n", "ls_swap_max_pairs", t->ls_swap_max_pairs);
    fprintf(out, "  %-28s %g\n", "max_size_frac", t->max_size_frac);
    fprintf(out, "  %-28s %" PRId64 "\n", "seed", t->seed);
    fprintf(out, "  %-28s %s\n", "rng_mode", "derived");
    fprintf(out, "  %-28s %s\n", "flow_enabled", yn(t->flow_enabled));
    fprintf(out, "  %-28s ", "source_quotas");
    for (int64_t i = 0; i < t->source_quota_count; ++i)
        fprintf(out, "%s%s=%" PRId64, i ? ", " : "", t->source_quotas[i].name,
                t->source_quotas[i].limit);
    fputc('\n', out);
    fprintf(out, "  %-28s %" PRId64 "\n", "per_vehicle_quota", t->per_vehicle_quota);
    fprintf(out, "  %-28s %" PRId64 "\n", "global_quota", t->global_quota);
    fprintf(out, "  %-28s %" PRId64 "\n", "near_duplicate_hamming", t->near_duplicate_hamming);
    fprintf(out, "  %-28s %g\n", "diversity_weight", t->diversity_weight);
    fprintf(out, "  %-28s %" PRId64 "\n", "vehicle_tighten_start", t->vehicle_tighten_start);
    fprintf(out, "  %-28s %g\n", "vehicle_tighten_power", t->vehicle_tighten_power);
    fprintf(out, "  %-28s %g\n", "vehicle_tighten_min_scale", t->vehicle_tighten_min_scale);
    fprintf(out, "  %-28s %" PRId64 "\n", "max_replicas", t->max_replicas);

    fprintf(out, "  %-28s %s\n", "similarity_metric", "jaccard");
    fprintf(out, "  %-28s %s\n", "norm", "l2");

    fprintf(out, "[misc]\n");
    print_opt_i64(out, "budget.max_rows", cfg->budget.max_rows);
    print_opt_i64(out, "budget.max_nnz", cfg->budget.max_nnz);
    fprintf(out, "  %-28s %s\n", "compute_cut_values", yn(cfg->compute_cut_values));
    fprintf(out, "  %-28s %" PRId64 "\n", "mbp_seed", cfg->mbp_seed);
    fprintf(out, "  %-28s %s\n", "cvrpsep", "out of scope (not implemented)");
}

/* ===================================================================== */
/* TOML configuration reader (semantics follow the reference loader)     */
/* ===================================================================== */

#include <ctype.h>
#include <stdlib.h>

#define TOML_MAX_KV   256
#define TOML_VAL_MAX  512

enum { TOML_SCOPE_TOP = 0, TOML_SCOPE_FEATURES, TOML_SCOPE_TTS, TOML_SCOPE_GROWTH, TOML_SCOPE_COUNT };

typedef struct {
    int  scope;
    char key[64];
    char val[TOML_VAL_MAX];
    bool used;
    int  line;
} toml_kv;

typedef struct {
    toml_kv kv[TOML_MAX_KV];
    int64_t n;
    const char *path;
} toml_doc;

static char *toml_trim(char *s)
{
    while (*s && isspace((unsigned char)*s)) ++s;
    size_t n = strlen(s);
    while (n > 0 && isspace((unsigned char)s[n - 1])) s[--n] = '\0';
    return s;
}

static void toml_strip_comment(char *s)
{
    bool in_str = false;
    for (char *p = s; *p; ++p) {
        if (*p == '"') in_str = !in_str;
        else if (*p == '#' && !in_str) { *p = '\0'; return; }
    }
}

static void toml_put(toml_doc *d, int scope, const char *key, const char *val, int line)
{
    if (d->n >= TOML_MAX_KV)
        sag_die("%s:%d: more than %d TOML keys (this reader only serves the reference configs)",
                d->path, line, TOML_MAX_KV);
    for (int64_t i = 0; i < d->n; ++i)
        if (d->kv[i].scope == scope && strcmp(d->kv[i].key, key) == 0)
            sag_die("%s:%d: duplicate key '%s'", d->path, line, key);
    toml_kv *e = &d->kv[d->n++];
    e->scope = scope;
    e->used = false;
    e->line = line;
    if (strlen(key) + 1 > sizeof e->key) sag_die("%s:%d: key name too long", d->path, line);
    if (strlen(val) + 1 > sizeof e->val) sag_die("%s:%d: value too long (key '%s')", d->path, line, key);
    snprintf(e->key, sizeof e->key, "%s", key);
    snprintf(e->val, sizeof e->val, "%s", val);
}

static const char *toml_take(toml_doc *d, int scope, const char *key)
{
    for (int64_t i = 0; i < d->n; ++i) {
        if (d->kv[i].scope != scope || strcmp(d->kv[i].key, key) != 0) continue;
        d->kv[i].used = true;
        return d->kv[i].val;
    }
    return NULL;
}

static bool toml_has(const toml_doc *d, int scope, const char *key)
{
    for (int64_t i = 0; i < d->n; ++i)
        if (d->kv[i].scope == scope && strcmp(d->kv[i].key, key) == 0) return true;
    return false;
}

/* ---- Value type conversion (out of range / malformed always dies) ---- */

static void toml_unquote(const char *v, char *out, size_t cap, const char *what)
{
    size_t n = strlen(v);
    if (n < 2 || v[0] != '"' || v[n - 1] != '"')
        sag_die("TOML: %s expects a string (in double quotes), got %s", what, v);
    if (n - 1 > cap) sag_die("TOML: the string for %s is too long", what);
    memcpy(out, v + 1, n - 2);
    out[n - 2] = '\0';
}

static int64_t toml_i64(const char *v, const char *what)
{
    char *end = NULL;
    long long x = strtoll(v, &end, 10);
    if (end == v) sag_die("TOML: %s expects an integer, got '%s'", what, v);
    while (*end && isspace((unsigned char)*end)) ++end;
    if (*end != '\0') sag_die("TOML: %s expects an integer, got '%s'", what, v);
    return (int64_t)x;
}

static double toml_f64(const char *v, const char *what)
{
    char *end = NULL;
    double x = strtod(v, &end);
    if (end == v) sag_die("TOML: %s expects a number, got '%s'", what, v);
    while (*end && isspace((unsigned char)*end)) ++end;
    if (*end != '\0') sag_die("TOML: %s expects a number, got '%s'", what, v);
    return x;
}

static bool toml_bool(const char *v, const char *what)
{
    if (strcmp(v, "true") == 0) return true;
    if (strcmp(v, "false") == 0) return false;
    sag_die("TOML: %s expects true/false, got '%s'", what, v);
}

static bool toml_empty_array(const char *v)
{
    if (v[0] != '[') return false;
    const char *p = v + 1;
    while (*p && isspace((unsigned char)*p)) ++p;
    return *p == ']';
}

/* ---- Frozen-value checks (mirrors the reference validation) ---- */

static void frozen_str(toml_doc *d, int scope, const char *key, const char *want)
{
    const char *v = toml_take(d, scope, key);
    if (!v) return;
    char buf[128];
    toml_unquote(v, buf, sizeof buf, key);
    for (char *p = buf; *p; ++p) *p = (char)tolower((unsigned char)*p);
    if (strcmp(buf, want) != 0)
        sag_die("TOML: %s = \"%s\" deviates from the frozen recipe (which requires \"%s\"); "
                "unsupported semantics are rejected, not silently dropped", key, buf, want);
}

static void frozen_bool(toml_doc *d, int scope, const char *key, bool want)
{
    const char *v = toml_take(d, scope, key);
    if (!v) return;
    if (toml_bool(v, key) != want)
        sag_die("TOML: %s = %s deviates from the frozen recipe (requires %s)", key, v, want ? "true" : "false");
}

static void frozen_num(toml_doc *d, int scope, const char *key, double want)
{
    const char *v = toml_take(d, scope, key);
    if (!v) return;
    if (toml_f64(v, key) != want)
        sag_die("TOML: %s = %s deviates from the frozen recipe (requires %g)", key, v, want);
}

static void frozen_empty_array(toml_doc *d, int scope, const char *key)
{
    const char *v = toml_take(d, scope, key);
    if (!v) return;
    if (!toml_empty_array(v))
        sag_die("TOML: %s = %s deviates from the frozen recipe (requires an empty array [])", key, v);
}

static void require_explicit(const toml_doc *d, int scope, const char *const *keys,
                             const char *scope_name)
{
    for (const char *const *k = keys; *k; ++k) {
        if (toml_has(d, scope, *k)) continue;
        sag_die("TOML: %s is missing the required key '%s' -- its frozen value differs "
                "from the loader default, and leaving it out would silently run the two "
                "arms as two different pipelines; state it explicitly", scope_name, *k);
    }
}

static const char *const TOP_REQUIRED[] = {
    "policy_decode_mode", "policy_sweep_coexist",
    "policy_device", "mbp_onion_mode", NULL
};
static const char *const TTS_REQUIRED[] = {
    "enabled", "samples_per_tau", "threshold_eps", "pl_enabled", "jitter_enabled",
    "max_prefix", "lp_grow_seeds", "lp_grow_native_rules", "compress_mode",
    "fcapfix_enabled", "fcapfix_seed_limit", "fcapfix_seed_mode", "old_repair_enabled",
    "old_repair_bank_size", "adddrop_enabled", "adddrop_seed_old_bank",
    "adddrop_seed_limit", "adddrop_seed_sources", "score_mode", "top_b", "top_l",
    "max_flips", "ls_swap", "max_size_frac", "seed", "rng_mode", "flow_enabled",
    "source_quotas", "per_vehicle_quota", "global_quota", "near_duplicate_hamming",
    "diversity_weight", "vehicle_tighten_start", "vehicle_tighten_power",
    "vehicle_tighten_min_scale", NULL
};

static void parse_source_quotas(const char *v, sag_tts_config *t)
{
    if (v[0] != '{') sag_die("TOML: tts.source_quotas expects an inline table, got '%s'", v);
    const char *p = v + 1;
    t->source_quota_count = 0;
    for (;;) {
        while (*p && (isspace((unsigned char)*p) || *p == ',')) ++p;
        if (*p == '}' || *p == '\0') break;
        const char *ks = p;
        while (*p && *p != '=' && !isspace((unsigned char)*p)) ++p;
        const size_t klen = (size_t)(p - ks);
        while (*p && (isspace((unsigned char)*p) || *p == '=')) ++p;
        const char *vs = p;
        while (*p && *p != ',' && *p != '}') ++p;
        char vbuf[64];
        const size_t vlen = (size_t)(p - vs);
        if (vlen + 1 > sizeof vbuf) sag_die("TOML: a source_quotas value is too long");
        memcpy(vbuf, vs, vlen);
        vbuf[vlen] = '\0';
        if (t->source_quota_count >= SAG_MAX_SOURCE_QUOTAS)
            sag_die("TOML: more than %d source_quotas entries", SAG_MAX_SOURCE_QUOTAS);
        sag_source_quota *q = &t->source_quotas[t->source_quota_count++];
        if (klen + 1 > sizeof q->name) sag_die("TOML: a source_quotas key is too long");
        memcpy(q->name, ks, klen);
        q->name[klen] = '\0';
        q->limit = toml_i64(toml_trim(vbuf), "tts.source_quotas value");
    }
}

void sag_config_from_toml(const char *path, sag_toml_config *out)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) sag_die("cannot open config file %s", path);

    memset(out, 0, sizeof *out);
    toml_doc doc;
    memset(&doc, 0, sizeof doc);
    doc.path = path;

    char line[1024];
    int lineno = 0, scope = TOML_SCOPE_TOP;
    int64_t inst = -1;
    while (fgets(line, sizeof line, fp)) {
        lineno += 1;
        toml_strip_comment(line);
        char *s = toml_trim(line);
        if (*s == '\0') continue;
        if (strcmp(s, "[[instances]]") == 0) {
            if (out->n_instances >= SAG_TOML_MAX_INSTANCES)
                sag_die("%s:%d: more than %d [[instances]] entries", path, lineno,
                        SAG_TOML_MAX_INSTANCES);
            inst = out->n_instances++;
            out->instances[inst].vehicles = -1;
            out->instances[inst].max_iter = -1;
            out->instances[inst].has_optimal = false;
            scope = -1;
            continue;
        }
        if (s[0] == '[') {
            if (strcmp(s, "[features]") == 0) scope = TOML_SCOPE_FEATURES;
            else if (strcmp(s, "[tts]") == 0) scope = TOML_SCOPE_TTS;
            else if (strcmp(s, "[growth]") == 0) scope = TOML_SCOPE_GROWTH;
            else if (strcmp(s, "[cut_manager]") == 0)
                sag_die("%s:%d: cut_manager is not part of the released recipe",
                        path, lineno);
            else if (strcmp(s, "[cvrpsep]") == 0)
                sag_die("%s:%d: [cvrpsep] asks for the CVRPSEP tail -- SAG-Sep does not "
                        "implement it at all; drop the section", path, lineno);
            else sag_die("%s:%d: unknown table header '%s' (only [features] / [tts] / [growth] / "
                         "[[instances]] are recognised)", path, lineno, s);
            inst = -1;
            continue;
        }
        char *eq = strchr(s, '=');
        if (!eq) sag_die("%s:%d: unparsable line '%s'", path, lineno, s);
        *eq = '\0';
        char *key = toml_trim(s);
        char *val = toml_trim(eq + 1);
        if (*val == '\0') sag_die("%s:%d: key '%s' has no value (this reader has no multi-line values)",
                                  path, lineno, key);
        if (scope < 0) {
            sag_instance_spec *sp = &out->instances[inst];
            if (strcmp(key, "name") == 0) toml_unquote(val, sp->name, sizeof sp->name, "instances.name");
            else if (strcmp(key, "vehicles") == 0) sp->vehicles = toml_i64(val, "instances.vehicles");
            else if (strcmp(key, "max_iter") == 0) {
                sp->max_iter = toml_i64(val, "instances.max_iter");
                if (sp->max_iter < 0)
                    sag_die("%s:%d: instances.max_iter does not accept a negative value (got %" PRId64 ")",
                            path, lineno, sp->max_iter);
            }
            else if (strcmp(key, "optimal") == 0) {
                sp->optimal = toml_f64(val, "instances.optimal");
                sp->has_optimal = true;
            } else sag_die("%s:%d: unknown instance key '%s' (only name/vehicles/optimal/max_iter)",
                           path, lineno, key);
            continue;
        }
        toml_put(&doc, scope, key, val, lineno);
    }
    fclose(fp);

    /* ---- Required keys (stop loader defaults silently diverging from frozen values) ---- */
    require_explicit(&doc, TOML_SCOPE_TOP, TOP_REQUIRED, "the top level");
    bool tts_enabled = true;
    {
        const char *ev = toml_take(&doc, TOML_SCOPE_TTS, "enabled");
        if (ev) tts_enabled = toml_bool(ev, "tts.enabled");
    }
    if (tts_enabled) require_explicit(&doc, TOML_SCOPE_TTS, TTS_REQUIRED, "[tts]");

    /* ---- Frozen keys (if present, they must equal the frozen value) ---- */
    frozen_str(&doc, TOML_SCOPE_TOP, "decode_pipeline", "unified");
    frozen_str(&doc, TOML_SCOPE_TOP, "method", "learned");
    frozen_str(&doc, TOML_SCOPE_TOP, "policy_decode_mode", "sweep_viol");
    {
        const char *cv = toml_take(&doc, TOML_SCOPE_TOP, "policy_sweep_coexist");
        if (cv && !toml_bool(cv, "policy_sweep_coexist") && tts_enabled)
            sag_die("TOML: policy_sweep_coexist = false deviates from the frozen recipe while "
                    "the test-time cut-refinement layer is on (requires true); it is accepted only together "
                    "with [tts] enabled = false, where it is inert");
    }
    frozen_str(&doc, TOML_SCOPE_TOP, "root_mode", "off");
    frozen_num(&doc, TOML_SCOPE_TOP, "mbp_delta", 0.0);
    frozen_empty_array(&doc, TOML_SCOPE_TOP, "mbp_strategies");
    frozen_num(&doc, TOML_SCOPE_TOP, "mbp_temperature", 1.0);
    frozen_str(&doc, TOML_SCOPE_TOP, "mbp_gate", "always");
    frozen_num(&doc, TOML_SCOPE_TOP, "mbp_kbest_max_iters", 0.0);
    frozen_str(&doc, TOML_SCOPE_TOP, "mbp_onion_mode", "mean");
    frozen_str(&doc, TOML_SCOPE_TOP, "mbp_onion_cutpoint", "band_first");
    frozen_bool(&doc, TOML_SCOPE_TOP, "policy_emit_base_subsets", true);
    frozen_num(&doc, TOML_SCOPE_TOP, "cut_violation_ratio", 0.0);
    frozen_bool(&doc, TOML_SCOPE_TOP, "dup_check", true);
    frozen_num(&doc, TOML_SCOPE_TOP, "mbp_onion_m_max", 0.0);
    frozen_num(&doc, TOML_SCOPE_TOP, "mbp_onion_skip_below_M", 0.0);
    {
        const char *v = toml_take(&doc, TOML_SCOPE_TOP, "mbp_m");
        if (v) {
            const int64_t m = toml_i64(v, "mbp_m");
            if (m != 0 && m != 1) sag_die("TOML: mbp_m can only be 0 or 1 in the frozen recipe, got %lld",
                                          (long long)m);
        }
    }

    frozen_num(&doc, TOML_SCOPE_TTS, "samples_per_tau", 0.0);
    frozen_empty_array(&doc, TOML_SCOPE_TTS, "threshold_eps");
    frozen_bool(&doc, TOML_SCOPE_TTS, "pl_enabled", false);
    frozen_bool(&doc, TOML_SCOPE_TTS, "jitter_enabled", false);
    frozen_bool(&doc, TOML_SCOPE_TTS, "shrink_one_edges", false);
    frozen_num(&doc, TOML_SCOPE_TTS, "sweep_topk", 0.0);
    frozen_bool(&doc, TOML_SCOPE_TTS, "lp_grow_enabled", true);
    frozen_bool(&doc, TOML_SCOPE_TTS, "fcapfix_enabled", true);
    frozen_bool(&doc, TOML_SCOPE_TTS, "old_repair_enabled", true);
    frozen_bool(&doc, TOML_SCOPE_TTS, "adddrop_enabled", true);
    (void)toml_take(&doc, TOML_SCOPE_TTS, "taus");

    /* ---- Map onto sag_config (start from the n100 preset, then override field by field) ---- */
    sag_config cfg = sag_config_preset("n100");
    cfg.size[0] = '\0';
    cfg.tts.max_replicas = 0;

    const char *v;
    if ((v = toml_take(&doc, TOML_SCOPE_TOP, "policy_run_dir")))
        toml_unquote(v, cfg.model.run_dir, sizeof cfg.model.run_dir, "policy_run_dir");
    if ((v = toml_take(&doc, TOML_SCOPE_TOP, "policy_device")))
        toml_unquote(v, cfg.model.device, sizeof cfg.model.device, "policy_device");

    if ((v = toml_take(&doc, TOML_SCOPE_TOP, "gurobi_method")))
        cfg.lp.gurobi_method = toml_i64(v, "gurobi_method");
    if ((v = toml_take(&doc, TOML_SCOPE_TOP, "gurobi_threads")))
        cfg.lp.gurobi_threads = toml_i64(v, "gurobi_threads");
    if ((v = toml_take(&doc, TOML_SCOPE_TOP, "lp_support_edge_threshold")))
        cfg.lp.lp_support_edge_threshold = toml_f64(v, "lp_support_edge_threshold");
    cfg.lp.time_limit = 0.0;
    if ((v = toml_take(&doc, TOML_SCOPE_TOP, "time_limit"))) {
        const double tl = toml_f64(v, "time_limit");
        cfg.lp.time_limit = tl > 0.0 ? tl : 0.0;
    }
    if ((v = toml_take(&doc, TOML_SCOPE_TOP, "max_iter"))) {
        const int64_t mi = toml_i64(v, "max_iter");
        if (mi < 0)
            sag_die("TOML: max_iter must not be negative (got %" PRId64 ") -- there a negative "
                    "means 'finite and already reached', here 'unlimited'; omit the key", mi);
        cfg.loop.max_iter = mi;
    } else cfg.loop.max_iter = -1;
    if ((v = toml_take(&doc, TOML_SCOPE_TOP, "policy_threshold"))) {
        cfg.sweep.threshold = toml_f64(v, "policy_threshold");
        cfg.onion.threshold = cfg.sweep.threshold;
    }
    if ((v = toml_take(&doc, TOML_SCOPE_TOP, "mbp_onion_min_violation")))
        cfg.onion.min_violation = toml_f64(v, "mbp_onion_min_violation");
    if ((v = toml_take(&doc, TOML_SCOPE_TOP, "mbp_seed")))
        cfg.mbp_seed = toml_i64(v, "mbp_seed");
    if ((v = toml_take(&doc, TOML_SCOPE_TOP, "instances_root")))
        toml_unquote(v, out->instances_root, sizeof out->instances_root, "instances_root");
    if ((v = toml_take(&doc, TOML_SCOPE_TOP, "output_dir")))
        toml_unquote(v, out->output_dir, sizeof out->output_dir, "output_dir");

    if ((v = toml_take(&doc, TOML_SCOPE_FEATURES, "partial_connect_cross_depot")))
        cfg.features.partial_connect_cross_depot =
            toml_bool(v, "features.partial_connect_cross_depot");

    sag_tts_config *t = &cfg.tts;
    t->enabled = tts_enabled;
#define TTS_I64(k, field) if ((v = toml_take(&doc, TOML_SCOPE_TTS, k))) t->field = toml_i64(v, "tts." k)
#define TTS_F64(k, field) if ((v = toml_take(&doc, TOML_SCOPE_TTS, k))) t->field = toml_f64(v, "tts." k)
#define TTS_BOOL(k, field) if ((v = toml_take(&doc, TOML_SCOPE_TTS, k))) t->field = toml_bool(v, "tts." k)
    TTS_I64("max_prefix", max_prefix);
    TTS_I64("min_size", min_size);
    TTS_I64("lp_grow_seeds", lp_grow_seeds);
    TTS_I64("fcapfix_rounds", fcapfix_rounds);
    TTS_I64("fcapfix_seed_limit", fcapfix_seed_limit);
    TTS_I64("old_repair_bank_size", old_repair_bank_size);
    TTS_I64("old_repair_top_b", old_repair_top_b);
    TTS_I64("adddrop_seed_limit", adddrop_seed_limit);
    TTS_I64("top_b", top_b);
    TTS_I64("top_l", top_l);
    TTS_I64("max_flips", max_flips);
    TTS_I64("ls_swap_max_pairs", ls_swap_max_pairs);
    TTS_I64("seed", seed);
    TTS_I64("per_vehicle_quota", per_vehicle_quota);
    TTS_I64("global_quota", global_quota);
    TTS_I64("near_duplicate_hamming", near_duplicate_hamming);
    TTS_I64("vehicle_tighten_start", vehicle_tighten_start);
    TTS_I64("max_replicas", max_replicas);
    TTS_F64("min_violation", min_violation);
    TTS_F64("min_gain", min_gain);
    TTS_F64("max_size_frac", max_size_frac);
    TTS_F64("diversity_weight", diversity_weight);
    TTS_F64("vehicle_tighten_power", vehicle_tighten_power);
    TTS_F64("vehicle_tighten_min_scale", vehicle_tighten_min_scale);
    TTS_BOOL("lp_grow_native_rules", lp_grow_native_rules);
    TTS_BOOL("adddrop_seed_old_bank", adddrop_seed_old_bank);
    TTS_BOOL("ls_swap", ls_swap);
    TTS_BOOL("flow_enabled", flow_enabled);
#undef TTS_I64
#undef TTS_F64
#undef TTS_BOOL
    {
        char buf[32];
        struct { const char *key, *want; } modes[] = {
            { "compress_mode", "full" }, { "fcapfix_seed_mode", "model" },
            { "score_mode", "efficacy" }, { "ls_neighborhood", "both" },
            { "rng_mode", "derived" }
        };
        for (size_t i = 0; i < sizeof modes / sizeof modes[0]; ++i) {
            if (!(v = toml_take(&doc, TOML_SCOPE_TTS, modes[i].key))) continue;
            toml_unquote(v, buf, sizeof buf, modes[i].key);
            for (char *p = buf; *p; ++p) *p = (char)tolower((unsigned char)*p);
            if (strcmp(buf, modes[i].want) != 0)
                sag_die("TOML: tts.%s = \"%s\" is outside the implementation surface (frozen value \"%s\")",
                        modes[i].key, buf, modes[i].want);
        }
    }
    if ((v = toml_take(&doc, TOML_SCOPE_TTS, "adddrop_seed_sources"))) {
        if (!toml_empty_array(v))
            sag_die("TOML: tts.adddrop_seed_sources is frozen to the empty array (adddrop "
                    "is seeded from the old-cut bank only), got %s", v);
        t->adddrop_seed_source_count = 0;
    }
    if ((v = toml_take(&doc, TOML_SCOPE_TTS, "source_quotas"))) parse_source_quotas(v, t);

    {
        sag_growth_config *g = &cfg.growth;
        if ((v = toml_take(&doc, TOML_SCOPE_GROWTH, "enabled"))) g->enabled = toml_bool(v, "growth.enabled");
        if ((v = toml_take(&doc, TOML_SCOPE_GROWTH, "rollouts"))) g->rollouts = toml_i64(v, "growth.rollouts");
        if ((v = toml_take(&doc, TOML_SCOPE_GROWTH, "horizon"))) g->horizon = toml_i64(v, "growth.horizon");
        if ((v = toml_take(&doc, TOML_SCOPE_GROWTH, "prefix_tabu"))) g->prefix_tabu = toml_i64(v, "growth.prefix_tabu");
        if ((v = toml_take(&doc, TOML_SCOPE_GROWTH, "restart_tabu"))) g->restart_tabu = toml_bool(v, "growth.restart_tabu");
        if ((v = toml_take(&doc, TOML_SCOPE_GROWTH, "max_rows"))) g->max_rows = toml_i64(v, "growth.max_rows");
        if ((v = toml_take(&doc, TOML_SCOPE_GROWTH, "diag"))) g->diag = toml_bool(v, "growth.diag");
        if ((v = toml_take(&doc, TOML_SCOPE_GROWTH, "bands"))) {
            char buf[64];
            toml_unquote(v, buf, sizeof buf, "growth.bands");
            g->n_bands = 0;
            const char *q = buf;
            while (*q != '\0') {
                char *end = NULL;
                const long long k = strtoll(q, &end, 10);
                if (end == q) sag_die("TOML: growth.bands cannot parse \"%s\"", buf);
                if (k < 1) sag_die("TOML: growth.bands: a band k must be >= 1, got %lld", k);
                if (g->n_bands >= SAG_GROWTH_MAX_BANDS)
                    sag_die("TOML: growth.bands lists more than %d bands", SAG_GROWTH_MAX_BANDS);
                g->bands[g->n_bands++] = (int64_t)k;
                q = end;
                while (*q == ',' || *q == ' ') ++q;
            }
            if (g->n_bands == 0) sag_die("TOML: growth.bands selects no band: \"%s\"", buf);
        }
        if (g->rollouts < 0 || g->horizon < 0 || g->prefix_tabu < 0 || g->max_rows < 0)
            sag_die("TOML: [growth] rollouts, horizon, prefix_tabu and max_rows must be non-negative");
    }

    /* ---- Unconsumed keys = unknown keys (the reference raises here as well) ---- */
    for (int64_t i = 0; i < doc.n; ++i) {
        if (doc.kv[i].used) continue;
        static const char *const SCOPE_NAME[TOML_SCOPE_COUNT] = {
            "top level", "[features]", "[tts]", "[growth]"
        };
        sag_die("%s:%d: %s has an unknown key '%s' -- this reader rejects unknown keys "
                "(config drift guard); compare against the reference config", path, doc.kv[i].line,
                SCOPE_NAME[doc.kv[i].scope], doc.kv[i].key);
    }
    if (out->n_instances == 0)
        sag_die("%s: no [[instances]] section (at least one is required)", path);

    sag_config_check_frozen(&cfg);
    out->cfg = cfg;
}
