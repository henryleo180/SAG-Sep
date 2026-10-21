#ifndef SAGSEP_JSON_OUT_H
#define SAGSEP_JSON_OUT_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "sagsep.h"

/* ===================================================================== */
/* 1. Low-level streaming writer                                         */
/* ===================================================================== */

#define SAG_JSON_MAX_DEPTH 16

typedef struct {
    FILE *out;
    int   depth;
    char  kind[SAG_JSON_MAX_DEPTH];
    bool  nonempty[SAG_JSON_MAX_DEPTH];
    bool  pending_key;
} sag_json;

void sag_json_init(sag_json *j, FILE *out);
void sag_json_finish(sag_json *j);

void sag_json_obj_begin(sag_json *j);
void sag_json_obj_end(sag_json *j);
void sag_json_arr_begin(sag_json *j);
void sag_json_arr_end(sag_json *j);

void sag_json_key(sag_json *j, const char *key);

void sag_json_str(sag_json *j, const char *s);
void sag_json_i64(sag_json *j, int64_t v);
void sag_json_f64(sag_json *j, double v);
void sag_json_bool(sag_json *j, bool v);
void sag_json_null(sag_json *j);

void sag_json_kv_str(sag_json *j, const char *key, const char *v);
void sag_json_kv_i64(sag_json *j, const char *key, int64_t v);
void sag_json_kv_f64(sag_json *j, const char *key, double v);
void sag_json_kv_bool(sag_json *j, const char *key, bool v);
void sag_json_kv_i64_or_null(sag_json *j, const char *key, int64_t v);
void sag_json_kv_f64_or_null(sag_json *j, const char *key, bool has, double v);

/* ===================================================================== */
/* 2. schema: root_relax_<stem>_learned_fb.json                          */
/* ===================================================================== */

typedef struct { const char *name; int64_t value; } sag_json_i64_field;
typedef struct { const char *name; double  value; } sag_json_f64_field;

typedef struct {
    const sag_pipeline_diag *pipeline;
    int8_t                   family_order[SAG_FAMILY_COUNT];
    int64_t                  family_count;
    const char              *tts_json;
} sag_json_iter_diag;

typedef struct {
    int64_t       iteration;
    double        objective;
    int64_t       cuts_added;
    double        max_violation;
    const double *policy_info;   int64_t policy_info_count;
    double        solve_time;
    double        cut_generation_time;
    int64_t       constraints;
    const double *violations;    int64_t violation_count;
    const sag_json_i64_field *counts;   int64_t counts_count;
    const sag_json_f64_field *timings;  int64_t timings_count;
    const sag_json_iter_diag *diag;
} sag_json_iteration;

typedef struct {
    const char *instance;
    int64_t     vehicles;
    double      lower_bound;
    double      runtime;
    const sag_json_iteration *iterations;  int64_t iterations_count;
    int64_t     iteration_count;
    int64_t     total_cuts;
    int64_t     last_cuts_added;
    double      last_max_violation;
    bool        has_optimal;     double optimal;
    bool        has_time_limit;  double time_limit;
    bool        time_limit_hit;
    const sag_json_f64_field *timing_totals; int64_t timing_totals_count;
} sag_json_root_result;

void sag_json_write_root_result(FILE *out, const sag_json_root_result *res);

void sag_json_write_result_file(const char *out_dir, const char *file_stem,
                                const sag_json_root_result *res);


#endif
