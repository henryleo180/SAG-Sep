#include "maxflow.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

static void add_raw_arc(bc_maxflow *flow, int from, int to, double capacity)
{
    if (flow->arc_count >= flow->max_arcs)
        bc_die("max-flow graph exceeded its %d-arc allocation", flow->max_arcs);
    const int arc = flow->arc_count++;
    flow->to[arc] = to;
    flow->base_capacity[arc] = capacity;
    flow->next[arc] = flow->head[from];
    flow->head[from] = arc;
}

static void add_directed_arc(bc_maxflow *flow, int from, int to, double capacity)
{
    add_raw_arc(flow, from, to, capacity);
    add_raw_arc(flow, to, from, 0.0);
}

void bc_maxflow_init(bc_maxflow *flow, int n_nodes, int max_undirected_edges)
{
    if (n_nodes <= 0)
        bc_die("max-flow workspace needs a positive node count, got %d", n_nodes);
    if (max_undirected_edges < 0)
        bc_die("max-flow workspace got a negative edge capacity");
    if (max_undirected_edges > INT_MAX / 4)
        bc_die("max-flow graph is too large for int arc indices");

    memset(flow, 0, sizeof *flow);
    flow->n_nodes = n_nodes;
    flow->max_undirected_edges = max_undirected_edges;
    flow->max_arcs = 4 * max_undirected_edges;

    const size_t node_slots = (size_t)n_nodes;
    const size_t arc_slots = (size_t)(flow->max_arcs > 0 ? flow->max_arcs : 1);
    flow->head = (int *)bc_malloc(node_slots * sizeof *flow->head);
    flow->to = (int *)bc_malloc(arc_slots * sizeof *flow->to);
    flow->next = (int *)bc_malloc(arc_slots * sizeof *flow->next);
    flow->level = (int *)bc_malloc(node_slots * sizeof *flow->level);
    flow->queue = (int *)bc_malloc(node_slots * sizeof *flow->queue);
    flow->work = (int *)bc_malloc(node_slots * sizeof *flow->work);
    flow->base_capacity =
        (double *)bc_malloc(arc_slots * sizeof *flow->base_capacity);
    flow->residual = (double *)bc_malloc(arc_slots * sizeof *flow->residual);
    flow->aggregate = (double *)bc_calloc(node_slots, sizeof *flow->aggregate);
    flow->touched = (int *)bc_malloc(node_slots * sizeof *flow->touched);
    flow->pending = (char *)bc_calloc(node_slots, sizeof *flow->pending);
    flow->reachable = (char *)bc_malloc(node_slots * sizeof *flow->reachable);
}

void bc_maxflow_free(bc_maxflow *flow)
{
    free(flow->head);
    free(flow->to);
    free(flow->next);
    free(flow->level);
    free(flow->queue);
    free(flow->work);
    free(flow->base_capacity);
    free(flow->residual);
    free(flow->aggregate);
    free(flow->touched);
    free(flow->pending);
    free(flow->reachable);
    memset(flow, 0, sizeof *flow);
}

void bc_maxflow_build_undirected(
    bc_maxflow *flow,
    int edge_count,
    const int32_t *edge_i,
    const int32_t *edge_j,
    const double *values,
    double threshold
)
{
    if (edge_count < 0 || edge_count > flow->max_undirected_edges)
        bc_die("max-flow build got %d edges for a %d-edge workspace",
               edge_count, flow->max_undirected_edges);
    if (!isfinite(threshold) || threshold < 0.0)
        bc_die("max-flow support threshold must be finite and non-negative");

    for (int node = 0; node < flow->n_nodes; ++node) flow->head[node] = -1;
    flow->arc_count = 0;

    for (int edge = 0; edge < edge_count; ++edge) {
        const int from = edge_i[edge];
        const int to = edge_j[edge];
        const double capacity = values[edge];
        if (!(0 <= from && from < to && to < flow->n_nodes))
            bc_die("max-flow edge (%d,%d) is outside 0..%d",
                   from, to, flow->n_nodes - 1);
        if (!isfinite(capacity))
            bc_die("max-flow support contains a non-finite edge value");
        if (capacity <= threshold) continue;

        add_directed_arc(flow, from, to, capacity);
        add_directed_arc(flow, to, from, capacity);
    }
}

static bool build_levels(bc_maxflow *flow, int source, int sink)
{
    for (int node = 0; node < flow->n_nodes; ++node) flow->level[node] = -1;
    int first = 0;
    int last = 0;
    flow->queue[last++] = source;
    flow->level[source] = 0;

    while (first < last) {
        const int node = flow->queue[first++];
        for (int arc = flow->head[node]; arc != -1; arc = flow->next[arc]) {
            const int next = flow->to[arc];
            if (flow->residual[arc] <= 0.0) continue;
            if (flow->level[next] >= 0) continue;
            flow->level[next] = flow->level[node] + 1;
            flow->queue[last++] = next;
        }
    }
    return flow->level[sink] >= 0;
}

static double send_flow(bc_maxflow *flow, int node, int sink, double available)
{
    if (node == sink) return available;
    if (available <= 0.0) return 0.0;

    for (int *cursor = &flow->work[node]; *cursor != -1;
         *cursor = flow->next[*cursor]) {
        const int arc = *cursor;
        const int next = flow->to[arc];
        if (flow->level[next] != flow->level[node] + 1) continue;
        if (flow->residual[arc] <= 0.0) continue;

        const double limit =
            available < flow->residual[arc] ? available : flow->residual[arc];
        const double sent = send_flow(flow, next, sink, limit);
        if (sent <= 0.0) continue;
        flow->residual[arc] -= sent;
        flow->residual[arc ^ 1] += sent;
        return sent;
    }
    return 0.0;
}

static void mark_sink_side(bc_maxflow *flow, int source, char *sink_side)
{
    memset(flow->reachable, 0, (size_t)flow->n_nodes * sizeof *flow->reachable);
    int first = 0;
    int last = 0;
    flow->queue[last++] = source;
    flow->reachable[source] = 1;

    while (first < last) {
        const int node = flow->queue[first++];
        int touched = 0;
        for (int arc = flow->head[node]; arc != -1; arc = flow->next[arc]) {
            const int next = flow->to[arc];
            if (flow->reachable[next]) continue;
            if (!flow->pending[next]) {
                flow->pending[next] = 1;
                flow->touched[touched++] = next;
            }
            flow->aggregate[next] += flow->residual[arc];
        }
        for (int k = 0; k < touched; ++k) {
            const int next = flow->touched[k];
            const double capacity = flow->aggregate[next];
            flow->pending[next] = 0;
            flow->aggregate[next] = 0.0;
            if (capacity <= BC_CONNECTIVITY_RESIDUAL_THRESHOLD) continue;
            flow->reachable[next] = 1;
            flow->queue[last++] = next;
        }
    }

    for (int node = 0; node < flow->n_nodes; ++node)
        sink_side[node] = (char)!flow->reachable[node];
}

void bc_maxflow_sink_shore(
    bc_maxflow *flow,
    int source,
    int sink,
    char *sink_side
)
{
    if (!(0 <= source && source < flow->n_nodes))
        bc_die("max-flow source %d is outside 0..%d", source, flow->n_nodes - 1);
    if (!(0 <= sink && sink < flow->n_nodes) || sink == source)
        bc_die("invalid max-flow sink %d for source %d", sink, source);

    memcpy(flow->residual, flow->base_capacity,
           (size_t)flow->arc_count * sizeof *flow->residual);

    while (build_levels(flow, source, sink)) {
        memcpy(flow->work, flow->head,
               (size_t)flow->n_nodes * sizeof *flow->work);
        for (;;) {
            if (send_flow(flow, source, sink, DBL_MAX) <= 0.0) break;
        }
    }

    mark_sink_side(flow, source, sink_side);
    if (!sink_side[sink] || sink_side[source])
        bc_die("the residual graph of a maximum flow reaches the sink");
}
