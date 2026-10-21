#include "json_out.h"

#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Low-level writer                                                     */
/* ------------------------------------------------------------------ */

static void jw(sag_json *j, const char *s)
{
    if (fputs(s, j->out) == EOF) sag_die("json_out: write failed");
}

static void jw_indent(sag_json *j)
{
    for (int i = 0; i < j->depth; ++i) jw(j, "  ");
}

static void jw_before_value(sag_json *j)
{
    if (j->pending_key) { j->pending_key = false; return; }
    if (j->depth == 0) return;
    int top = j->depth - 1;
    if (j->kind[top] == 'o')
        sag_die("json_out: a value inside an object must be preceded by a key");
    jw(j, j->nonempty[top] ? ",\n" : "\n");
    j->nonempty[top] = true;
    jw_indent(j);
}

static void jw_push(sag_json *j, char kind, const char *open)
{
    jw_before_value(j);
    if (j->depth >= SAG_JSON_MAX_DEPTH)
        sag_die("json_out: nesting deeper than SAG_JSON_MAX_DEPTH=%d", SAG_JSON_MAX_DEPTH);
    jw(j, open);
    j->kind[j->depth] = kind;
    j->nonempty[j->depth] = false;
    j->depth++;
}

static void jw_pop(sag_json *j, char kind, const char *close)
{
    if (j->depth <= 0) sag_die("json_out: the container is already closed");
    int top = j->depth - 1;
    if (j->kind[top] != kind)
        sag_die("json_out: container type mismatch (opened '%c', closed '%c')", j->kind[top], kind);
    if (j->pending_key) sag_die("json_out: key with no value after it");
    bool had = j->nonempty[top];
    j->depth--;
    if (had) { jw(j, "\n"); jw_indent(j); }
    jw(j, close);
}

void sag_json_init(sag_json *j, FILE *out)
{
    memset(j, 0, sizeof *j);
    j->out = out;
}

void sag_json_finish(sag_json *j)
{
    if (j->depth != 0) sag_die("json_out: %d container level(s) left unclosed", j->depth);
    if (j->pending_key) sag_die("json_out: key with no value after it");
    if (fflush(j->out) != 0) sag_die("json_out: flush failed");
}

void sag_json_obj_begin(sag_json *j) { jw_push(j, 'o', "{"); }
void sag_json_obj_end(sag_json *j)   { jw_pop(j, 'o', "}"); }
void sag_json_arr_begin(sag_json *j) { jw_push(j, 'a', "["); }
void sag_json_arr_end(sag_json *j)   { jw_pop(j, 'a', "]"); }

static void jw_quoted(sag_json *j, const char *s)
{
    jw(j, "\"");
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        unsigned char c = *p;
        switch (c) {
        case '"':  jw(j, "\\\""); break;
        case '\\': jw(j, "\\\\"); break;
        case '\b': jw(j, "\\b");  break;
        case '\f': jw(j, "\\f");  break;
        case '\n': jw(j, "\\n");  break;
        case '\r': jw(j, "\\r");  break;
        case '\t': jw(j, "\\t");  break;
        default:
            if (c < 0x20) {
                char buf[16];
                snprintf(buf, sizeof buf, "\\u%04x", (unsigned)c);
                jw(j, buf);
            } else if (c >= 0x80) {
                sag_die("json_out: non-ASCII byte 0x%02x -- the reference writer escapes "
                        "it as \\uXXXX; UTF-8 escaping is outside this writer's minimal surface", c);
            } else {
                if (fputc((int)c, j->out) == EOF) sag_die("json_out: write failed");
            }
        }
    }
    jw(j, "\"");
}

void sag_json_key(sag_json *j, const char *key)
{
    if (j->depth <= 0 || j->kind[j->depth - 1] != 'o')
        sag_die("json_out: a key may only appear inside an object");
    if (j->pending_key) sag_die("json_out: two consecutive keys");
    int top = j->depth - 1;
    jw(j, j->nonempty[top] ? ",\n" : "\n");
    j->nonempty[top] = true;
    jw_indent(j);
    jw_quoted(j, key);
    jw(j, ": ");
    j->pending_key = true;
}

void sag_json_str(sag_json *j, const char *s)
{
    jw_before_value(j);
    jw_quoted(j, s);
}

void sag_json_i64(sag_json *j, int64_t v)
{
    jw_before_value(j);
    char buf[24];
    snprintf(buf, sizeof buf, "%lld", (long long)v);
    jw(j, buf);
}

void sag_json_f64(sag_json *j, double v)
{
    jw_before_value(j);
    if (isnan(v))      { jw(j, "NaN"); return; }
    if (isinf(v))      { jw(j, v > 0 ? "Infinity" : "-Infinity"); return; }
    char buf[40];
    snprintf(buf, sizeof buf, "%.17g", v);
    if (strpbrk(buf, ".eEnN") == NULL) {
        size_t n = strlen(buf);
        if (n + 3 > sizeof buf) sag_die("json_out: buffer too small for the double literal: %s", buf);
        buf[n] = '.'; buf[n + 1] = '0'; buf[n + 2] = '\0';
    }
    jw(j, buf);
}

void sag_json_bool(sag_json *j, bool v)
{
    jw_before_value(j);
    jw(j, v ? "true" : "false");
}

void sag_json_null(sag_json *j)
{
    jw_before_value(j);
    jw(j, "null");
}

void sag_json_kv_str(sag_json *j, const char *key, const char *v)  { sag_json_key(j, key); sag_json_str(j, v); }
void sag_json_kv_i64(sag_json *j, const char *key, int64_t v)      { sag_json_key(j, key); sag_json_i64(j, v); }
void sag_json_kv_f64(sag_json *j, const char *key, double v)       { sag_json_key(j, key); sag_json_f64(j, v); }
void sag_json_kv_bool(sag_json *j, const char *key, bool v)        { sag_json_key(j, key); sag_json_bool(j, v); }

void sag_json_kv_i64_or_null(sag_json *j, const char *key, int64_t v)
{
    sag_json_key(j, key);
    if (v < 0) sag_json_null(j); else sag_json_i64(j, v);
}

void sag_json_kv_f64_or_null(sag_json *j, const char *key, bool has, double v)
{
    sag_json_key(j, key);
    if (!has) sag_json_null(j); else sag_json_f64(j, v);
}

/* ------------------------------------------------------------------ */
/* schema: rcc_pipeline                                                */
/* ------------------------------------------------------------------ */

static void jw_raw_value(sag_json *j, const char *raw)
{
    jw_before_value(j);
    for (const char *p = raw; *p; ++p) {
        if (fputc((int)(unsigned char)*p, j->out) == EOF)
            sag_die("json_out: write failed");
        if (*p == '\n') jw_indent(j);
    }
}

static void jw_family_diag(sag_json *j, const sag_family_diag *d)
{
    sag_json_obj_begin(j);
    sag_json_kv_i64(j, "raw", d->raw);
    sag_json_kv_i64(j, "within_family_unique", d->within_family_unique);
    sag_json_kv_i64(j, "contributed_global_unique", d->contributed_global_unique);
    sag_json_kv_i64(j, "selected_with_origin", d->selected_with_origin);
    sag_json_kv_f64(j, "cpu_elapsed", d->cpu_elapsed);
    sag_json_obj_end(j);
}

static void jw_pipeline_diag(sag_json *j, const sag_json_iter_diag *diag)
{
    const sag_pipeline_diag *p = diag->pipeline;
    if (p == NULL) sag_die("json_out: diagnostics.rcc_pipeline has a NULL pipeline pointer");

    sag_json_obj_begin(j);

    sag_json_key(j, "global");
    sag_json_obj_begin(j);
    sag_json_kv_i64(j, "raw", p->raw);
    sag_json_kv_i64(j, "exact_unique", p->exact_unique);
    sag_json_kv_i64(j, "duplicates", p->raw - p->exact_unique);
    sag_json_kv_i64(j, "blocked", p->blocked);
    sag_json_kv_i64(j, "nonviolated", p->nonviolated);
    sag_json_kv_i64(j, "selector_pool", p->selector_pool);
    sag_json_kv_i64(j, "selected", p->selected);
    sag_json_kv_i64(j, "materialized", p->materialized);
    sag_json_kv_i64(j, "added_to_lp", p->added_to_lp);
    sag_json_obj_end(j);

    sag_json_key(j, "families");
    sag_json_obj_begin(j);
    if (diag->family_count < 0 || diag->family_count > SAG_FAMILY_COUNT)
        sag_die("json_out: family_count=%lld out of range [0,%d]",
                (long long)diag->family_count, SAG_FAMILY_COUNT);
    for (int64_t i = 0; i < diag->family_count; ++i) {
        int8_t code = diag->family_order[i];
        if (code < 0 || code >= SAG_FAMILY_COUNT)
            sag_die("json_out: family_order[%lld]=%d out of range", (long long)i, (int)code);
        sag_json_key(j, sag_family_names[code]);
        jw_family_diag(j, &p->families[code]);
    }
    sag_json_obj_end(j);

    sag_json_key(j, "selector");
    sag_json_obj_begin(j);
    sag_json_kv_str(j, "name", "passthrough");
    sag_json_kv_i64_or_null(j, "max_rows", p->budget.max_rows);
    sag_json_kv_i64_or_null(j, "max_nnz", p->budget.max_nnz);
    sag_json_obj_end(j);

    sag_json_key(j, "timings");
    sag_json_obj_begin(j);
    for (int i = 0; i < SAG_PIPE_T_COUNT; ++i)
        sag_json_kv_f64(j, sag_pipeline_timing_names[i], p->timings[i]);
    sag_json_obj_end(j);

    sag_json_obj_end(j);
}

/* ------------------------------------------------------------------ */
/* schema: iterations[] and the top level                               */
/* ------------------------------------------------------------------ */

static void jw_iteration(sag_json *j, const sag_json_iteration *it)
{
    sag_json_obj_begin(j);
    sag_json_kv_i64(j, "iteration", it->iteration);
    sag_json_kv_f64(j, "objective", it->objective);
    sag_json_kv_i64(j, "cuts_added", it->cuts_added);
    sag_json_kv_f64(j, "max_violation", it->max_violation);

    sag_json_key(j, "policy_info");
    sag_json_arr_begin(j);
    for (int64_t i = 0; i < it->policy_info_count; ++i)
        sag_json_f64(j, it->policy_info[i]);
    sag_json_arr_end(j);

    sag_json_kv_f64(j, "solve_time", it->solve_time);
    sag_json_kv_f64(j, "cut_generation_time", it->cut_generation_time);
    sag_json_kv_i64(j, "constraints", it->constraints);

    sag_json_key(j, "violations");
    sag_json_arr_begin(j);
    for (int64_t i = 0; i < it->violation_count; ++i)
        sag_json_f64(j, it->violations[i]);
    sag_json_arr_end(j);

    sag_json_key(j, "counts");
    sag_json_obj_begin(j);
    for (int64_t i = 0; i < it->counts_count; ++i)
        sag_json_kv_i64(j, it->counts[i].name, it->counts[i].value);
    sag_json_obj_end(j);

    sag_json_key(j, "timings");
    sag_json_obj_begin(j);
    for (int64_t i = 0; i < it->timings_count; ++i)
        sag_json_kv_f64(j, it->timings[i].name, it->timings[i].value);
    sag_json_obj_end(j);

    if (it->diag != NULL) {
        sag_json_key(j, "diagnostics");
        sag_json_obj_begin(j);
        sag_json_key(j, "rcc_pipeline");
        jw_pipeline_diag(j, it->diag);
        if (it->diag->tts_json != NULL) {
            sag_json_key(j, "tts");
            jw_raw_value(j, it->diag->tts_json);
        }
        sag_json_obj_end(j);
    }

    sag_json_obj_end(j);
}

void sag_json_write_root_result(FILE *out, const sag_json_root_result *res)
{
    sag_json j;
    sag_json_init(&j, out);

    sag_json_obj_begin(&j);
    sag_json_kv_str(&j, "instance", res->instance);
    sag_json_kv_i64(&j, "vehicles", res->vehicles);
    sag_json_kv_f64(&j, "lower_bound", res->lower_bound);
    sag_json_kv_f64(&j, "runtime", res->runtime);

    sag_json_key(&j, "iterations");
    sag_json_arr_begin(&j);
    for (int64_t i = 0; i < res->iterations_count; ++i)
        jw_iteration(&j, &res->iterations[i]);
    sag_json_arr_end(&j);

    sag_json_kv_i64(&j, "iteration_count", res->iteration_count);
    sag_json_kv_i64(&j, "total_cuts", res->total_cuts);
    sag_json_kv_i64(&j, "last_cuts_added", res->last_cuts_added);
    sag_json_kv_f64(&j, "last_max_violation", res->last_max_violation);
    sag_json_kv_f64_or_null(&j, "optimal", res->has_optimal, res->optimal);
    sag_json_kv_f64_or_null(&j, "time_limit", res->has_time_limit, res->time_limit);
    sag_json_kv_bool(&j, "time_limit_hit", res->time_limit_hit);

    sag_json_key(&j, "timing_totals");
    sag_json_obj_begin(&j);
    for (int64_t i = 0; i < res->timing_totals_count; ++i)
        sag_json_kv_f64(&j, res->timing_totals[i].name, res->timing_totals[i].value);
    sag_json_obj_end(&j);

    sag_json_obj_end(&j);
    sag_json_finish(&j);
}

void sag_json_write_result_file(const char *out_dir, const char *file_stem,
                                const sag_json_root_result *res)
{
    char path[1024];
    int n = snprintf(path, sizeof path, "%s/root_relax_%s_learned_fb.json",
                     out_dir, file_stem);
    if (n < 0 || (size_t)n >= sizeof path)
        sag_die("json_out: output path too long (dir=%s stem=%s)", out_dir, file_stem);
    FILE *fp = fopen(path, "w");
    if (!fp) sag_die("json_out: cannot write %s", path);
    sag_json_write_root_result(fp, res);
    if (fclose(fp) != 0) sag_die("json_out: close failed for %s", path);
}

