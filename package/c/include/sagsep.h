#ifndef SAGSEP_H
#define SAGSEP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "arena.h"

/* ===================================================================== */
/* 0. Constants, families, variant string table                          */
/* ===================================================================== */

#define SAG_DEPOT 0

enum {
    SAG_FAMILY_SWEEP = 0,
    SAG_FAMILY_ONION = 1,
    SAG_FAMILY_TTS   = 2,
    SAG_FAMILY_COUNT = 3
};
extern const char *const sag_family_names[SAG_FAMILY_COUNT];

typedef enum {
    SAG_VAR_WINNER = 0,
    SAG_VAR_THRESHOLD_FALLBACK,
    SAG_VAR_BAND_FIRST,
    SAG_VAR_LP_GROW,
    SAG_VAR_FCAPFIX,
    SAG_VAR_OLD_REPAIR,
    SAG_VAR_ADDDROP,
    SAG_VAR_GROWTH,
    SAG_VARIANT_COUNT
} sag_variant;

extern const char *const sag_variant_names[SAG_VARIANT_COUNT];

int8_t sag_variant_from_str(const char *s, size_t len);

/* ===================================================================== */
/* 1. instance                                                            */
/* ===================================================================== */

#define SAG_NAME_MAX 128

typedef struct {
    char     name[SAG_NAME_MAX];
    int64_t  n_nodes;
    int64_t  depot;
    int64_t  vehicles;
    int64_t  vehicle_count;
    int64_t  capacity;
    const int64_t *demand;
    const double  *coords;
    const double  *weight;
    double   optimal;
    bool     has_optimal;
} sag_instance;

/* ===================================================================== */
/* 2. LP support (x* support-graph snapshot)                             */
/* ===================================================================== */

typedef struct {
    int64_t        n_nodes;
    double         threshold;
    const int64_t *indptr;
    const int32_t *indices;
    const float   *data_f32;
    const double  *data_f64;
    int64_t        nnz;
    const float   *dense;
    const float   *degree;
    const int64_t *edge_u;
    const int64_t *edge_v;
    const double  *edge_x;
    int64_t        n_edges;
} sag_lp_support;

/* ===================================================================== */
/* 3. candidate batch (columnar: CSR + bitmask + provenance)             */
/* ===================================================================== */

typedef struct {
    int64_t   n_nodes;
    int64_t   words;
    int64_t   rows;
    int64_t   nnz;
    int64_t  *indptr;
    int32_t  *indices;
    uint64_t *mask;
    int8_t   *family;
    int32_t  *group_id;
    int8_t   *variant;
    double   *pscore;
    int64_t   cap_rows;
    int64_t   cap_nnz;
} sag_candidate_batch;

/* ===================================================================== */
/* 4. scoring / selection / materialize                                   */
/* ===================================================================== */

typedef struct {
    int64_t  rows;
    double  *rhs;
    double  *boundary;
    double  *violation;
    int64_t *row_nnz;
    double  *coef_l2;
} sag_scored_batch;

typedef struct {
    int64_t max_rows;
    int64_t max_nnz;
} sag_selection_budget;

typedef struct {
    int64_t *selected;
    int64_t  count;
    int64_t  nonviolated;
} sag_selection;

typedef struct {
    int64_t  rows;
    int64_t  nnz;
    int64_t *indptr;
    int32_t *indices;
    double  *rhs;
    int64_t *sizes;
    double  *cut_values;
    int64_t  n_cut_values;
    int64_t *pool_rows;
} sag_materialized_cuts;

/* ===================================================================== */
/* 6. Frozen config (the frozen recipe + presets, copied 1:1)            */
/* ===================================================================== */

typedef enum { SAG_COMPRESS_FULL      = 0 } sag_compress_mode;
typedef enum { SAG_SCORE_EFFICACY     = 0 } sag_score_mode;
typedef enum { SAG_LS_BOTH            = 0 } sag_ls_neighborhood;
typedef enum { SAG_RNG_DERIVED        = 0 } sag_rng_mode;
typedef enum { SAG_FCAPFIX_SEED_MODEL = 0 } sag_fcapfix_seed_mode;
typedef enum { SAG_ONION_MODE_MEAN    = 0 } sag_onion_mode;
typedef enum { SAG_ONION_CUT_BAND_FIRST = 0 } sag_onion_cutpoint;

typedef struct {
    char run_dir[512];
    char device[16];
} sag_model_config;

typedef struct {
    bool partial_connect_cross_depot;
} sag_features_config;

typedef struct {
    int64_t gurobi_method;
    int64_t gurobi_threads;
    double  lp_support_edge_threshold;
    double  time_limit;
} sag_lp_config;

typedef struct {
    int64_t max_iter;
    double  min_violation;
    double  cut_violation_ratio;
    bool    dup_check;
} sag_loop_config;

typedef struct {
    double  threshold;
    int64_t min_size;
    double  min_violation;
} sag_sweep_config;

#define SAG_GROWTH_MAX_BANDS 8

typedef struct {
    bool    enabled;
    int64_t rollouts;
    int64_t horizon;
    int64_t bands[SAG_GROWTH_MAX_BANDS];
    int64_t n_bands;
    int64_t prefix_tabu;
    bool    restart_tabu;
    int64_t max_rows;
    bool    diag;
} sag_growth_config;

typedef struct {
    sag_onion_mode     mode;
    int64_t            m_max_cap;
    double             min_violation;
    double             threshold;
    int64_t            skip_below_M;
    sag_onion_cutpoint cutpoint;
} sag_onion_config;

#define SAG_MAX_SOURCE_QUOTAS 8

typedef struct {
    char    name[16];
    int64_t limit;
} sag_source_quota;

typedef struct {
    bool                  enabled;
    int64_t               max_prefix;
    int64_t               min_size;
    int64_t               lp_grow_seeds;
    bool                  lp_grow_native_rules;
    sag_compress_mode     compress_mode;
    int64_t               fcapfix_rounds;
    int64_t               fcapfix_seed_limit;
    sag_fcapfix_seed_mode fcapfix_seed_mode;
    int64_t               old_repair_bank_size;
    int64_t               old_repair_top_b;
    bool                  adddrop_seed_old_bank;
    int64_t               adddrop_seed_source_count;
    int64_t               adddrop_seed_limit;
    sag_score_mode        score_mode;
    double                min_violation;
    int64_t               top_b;
    int64_t               top_l;
    int64_t               max_flips;
    double                min_gain;
    sag_ls_neighborhood   ls_neighborhood;
    bool                  ls_swap;
    int64_t               ls_swap_max_pairs;
    double                max_size_frac;
    int64_t               seed;
    sag_rng_mode          rng_mode;
    bool                  flow_enabled;
    sag_source_quota      source_quotas[SAG_MAX_SOURCE_QUOTAS];
    int64_t               source_quota_count;
    int64_t               per_vehicle_quota;
    int64_t               global_quota;
    int64_t               near_duplicate_hamming;
    double                diversity_weight;
    int64_t               vehicle_tighten_start;
    double                vehicle_tighten_power;
    double                vehicle_tighten_min_scale;
    int64_t               max_replicas;

} sag_tts_config;

typedef struct {
    sag_model_config     model;
    sag_features_config  features;
    sag_lp_config        lp;
    sag_loop_config      loop;
    sag_sweep_config     sweep;
    sag_growth_config    growth;
    sag_onion_config     onion;
    sag_tts_config       tts;
    sag_selection_budget budget;
    bool                 compute_cut_values;
    int64_t              mbp_seed;
    char                 size[8];
} sag_config;

/* ===================================================================== */
/* 6b. reference TOML (minimal reader)                                   */
/* ===================================================================== */

#define SAG_TOML_MAX_INSTANCES 64

typedef struct {
    char    name[SAG_NAME_MAX];
    int64_t vehicles;
    int64_t max_iter;
    double  optimal;
    bool    has_optimal;
} sag_instance_spec;

typedef struct {
    sag_config        cfg;
    char              instances_root[512];
    char              output_dir[512];
    sag_instance_spec instances[SAG_TOML_MAX_INSTANCES];
    int64_t           n_instances;
} sag_toml_config;

void sag_config_from_toml(const char *path, sag_toml_config *out);

extern const char *const sag_preset_sizes[];

sag_config sag_config_preset(const char *size);

int64_t sag_preset_max_prefix_for_customers(int64_t n_customers);
void       sag_config_check_frozen(const sag_config *cfg);
void       sag_config_print(FILE *out, const sag_config *cfg);

/* ===================================================================== */
/* 7. Diagnostics / timing (a JSON schema contract)                      */
/* ===================================================================== */

enum {
    SAG_DIAG_RAW = 0,
    SAG_DIAG_EXACT_UNIQUE,
    SAG_DIAG_BLOCKED,
    SAG_DIAG_NONVIOLATED,
    SAG_DIAG_SELECTOR_POOL,
    SAG_DIAG_SELECTED,
    SAG_DIAG_MATERIALIZED,
    SAG_DIAG_COUNTS
};

typedef struct {
    int64_t raw;
    int64_t within_family_unique;
    int64_t contributed_global_unique;
    int64_t selected_with_origin;
    double  cpu_elapsed;
} sag_family_diag;

enum {
    SAG_PIPE_T_FAMILY_DECODE_WALL = 0,
    SAG_PIPE_T_DEDUP,
    SAG_PIPE_T_SCORING,
    SAG_PIPE_T_SELECTOR,
    SAG_PIPE_T_MATERIALIZE,
    SAG_PIPE_T_COUNT
};
extern const char *const sag_pipeline_timing_names[SAG_PIPE_T_COUNT];

typedef struct {
    int64_t              raw;
    int64_t              exact_unique;
    int64_t              blocked;
    int64_t              nonviolated;
    int64_t              selector_pool;
    int64_t              selected;
    int64_t              materialized;
    int64_t              added_to_lp;
    sag_family_diag      families[SAG_FAMILY_COUNT];
    sag_selection_budget budget;
    double               timings[SAG_PIPE_T_COUNT];
} sag_pipeline_diag;

extern const char *const sag_loop_timing_names[];
extern const char *const sag_policy_timing_names[];

#endif
