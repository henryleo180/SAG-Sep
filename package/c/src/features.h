#ifndef SAGSEP_FEATURES_H
#define SAGSEP_FEATURES_H

#include <stdint.h>

#include "arena.h"
#include "sagsep.h"

#define SAG_PARTIAL_CONNECT       2
#define SAG_PARTIAL_EDGE_COEF     1.0
#define SAG_NODE_DIM              2
#define SAG_EDGE_DIM              2
#define SAG_HOP_ORIGINAL          1.0f
#define SAG_HOP_SHORTCUT          2.0f
#define SAG_HOP_DEPOT_FILL        0.0f

typedef struct {
    int64_t  n_nodes;
    int64_t  vehicle_count;
    int64_t  n_edges;
    int64_t *edge_index;
    float   *edge_attr;
    float   *node_attr;
    int64_t  n_support_arcs;
    int64_t  n_shortcut_arcs;
    int64_t  n_depot_arcs;
} sag_graph_features;

static inline const int64_t *sag_gf_src(const sag_graph_features *g) { return g->edge_index; }
static inline const int64_t *sag_gf_dst(const sag_graph_features *g) { return g->edge_index + g->n_edges; }

float sag_f32_pairwise_sum(const float *a, int64_t n);

float sag_f32_sum_numpy(const float *a, int64_t n);

int64_t sag_vehicle_replica_count(const sag_instance *inst);

void sag_build_lp_support(sag_arena *a,
                          const int64_t *edge_u, const int64_t *edge_v,
                          const double *edge_x, int64_t m,
                          int64_t n_nodes, double threshold,
                          sag_lp_support *out);

void sag_build_graph_features(sag_arena *a, const sag_instance *inst,
                              const sag_lp_support *lp, int64_t vehicle_count,
                              const int64_t *sup_u, const int64_t *sup_v,
                              const double *sup_x, int64_t sup_m,
                              bool cross_depot,
                              sag_graph_features *out);

#endif
