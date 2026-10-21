#ifndef SAGSEP_FAMILIES_TTS_SCORE_H
#define SAGSEP_FAMILIES_TTS_SCORE_H

#include <stdbool.h>
#include <stdint.h>

#include "arena.h"
#include "families/tts/rhs_table.h"
#include "sagsep.h"

/* ===================================================================== */
/* 0. Sparse identity-acceleration context                               */
/* ===================================================================== */

typedef struct {
    const int64_t       *adj_indptr;
    const int32_t       *adj_indices;
    const sag_rhs_table *rhs;
} sag_tts_score_ctx;

void sag_tts_build_adj_csr(sag_arena *a, const float *adj, int64_t n,
                           const int64_t **out_indptr, const int32_t **out_indices,
                           bool *out_clean);

/* ===================================================================== */
/* 1. numpy primitives                                                    */
/* ===================================================================== */

int64_t sag_tts_argmax_f32(const float *v, int64_t n);

void sag_tts_stable_argsort_f32(const float *key, int64_t n, int32_t *idx, int32_t *scratch);

/* ===================================================================== */
/* 2. score_subsets                                                       */
/* ===================================================================== */

void sag_tts_score_subsets(const float *subsets, int64_t rows, int64_t n,
                           const float *adj, const float *deg, const float *demand,
                           double capacity, const sag_tts_score_ctx *sctx,
                           float *viol, float *lhs, float *rhs, float *d_S,
                           float *scratch);

float sag_tts_score_one(const float *sel, int64_t n, const float *adj, const float *deg,
                        const float *demand, double capacity, const sag_tts_score_ctx *sctx,
                        float *scratch);

/* ===================================================================== */
/* 3. local_search_1flip                                                  */
/* ===================================================================== */

typedef struct {
    int64_t max_flips;
    double  min_gain;
    double  max_size_frac;
    int64_t swap_max_pairs;
    int64_t min_keep;
} sag_tts_ls_params;

typedef struct {
    double  initial_violation;
    double  best_violation;
    double  gain;
    int64_t steps;
    bool    hit_max_flips;
} sag_tts_ls_diag;

typedef struct {
    float   *deg_in_S;
    float   *prod;
    int32_t *in_idx;
    int32_t *out_idx;
    float   *demand_in;
    float   *rem_in;
    float   *demand_out;
    float   *add_out;
    int64_t  n_nodes;
} sag_tts_ls_work;

void sag_tts_ls_work_init(sag_arena *a, sag_tts_ls_work *w, int64_t n_nodes);

double sag_tts_local_search_1flip(float *sel, int64_t n, const float *adj, const float *deg,
                                  const float *demand, double capacity,
                                  const sag_tts_score_ctx *sctx,
                                  const sag_tts_ls_params *p, sag_tts_ls_diag *diag,
                                  sag_tts_ls_work *work);

double sag_tts_local_search_1flip_scalar_ref(float *sel, int64_t n, const float *adj,
                                             const float *deg, const float *demand,
                                             double capacity, const sag_tts_score_ctx *sctx,
                                             const sag_tts_ls_params *p, sag_tts_ls_diag *diag,
                                             sag_tts_ls_work *work);

#endif
