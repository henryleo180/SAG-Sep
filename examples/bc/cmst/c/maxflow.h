#ifndef BC_MAXFLOW_H
#define BC_MAXFLOW_H

#include <stdint.h>

#define BC_CONNECTIVITY_SUPPORT_THRESHOLD  1e-9
#define BC_CONNECTIVITY_RESIDUAL_THRESHOLD 1e-12

typedef struct {
    int n_nodes;
    int max_undirected_edges;
    int max_arcs;
    int arc_count;

    int *head;
    int *to;
    int *next;
    int *level;
    int *queue;
    int *work;

    double *base_capacity;
    double *residual;
    double *aggregate;
    int    *touched;
    char   *pending;
    char   *reachable;
} bc_maxflow;

void bc_maxflow_init(bc_maxflow *flow, int n_nodes, int max_undirected_edges);
void bc_maxflow_free(bc_maxflow *flow);

void bc_maxflow_build_undirected(
    bc_maxflow *flow,
    int edge_count,
    const int32_t *edge_i,
    const int32_t *edge_j,
    const double *values,
    double threshold
);

void bc_maxflow_sink_shore(
    bc_maxflow *flow,
    int source,
    int sink,
    char *sink_side
);

#endif
