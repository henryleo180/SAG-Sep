#ifndef CVRPSEP_CAPI_H
#define CVRPSEP_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

void *cvrpsep_capi_create_cmgr(int dim);

void cvrpsep_capi_free_cmgr(void *cmgr);

void cvrpsep_capi_move_cnstr(void *source, void *sink, int source_index,
                             int sink_index);

void cvrpsep_capi_separate_capacity_cuts(
    int num_customers, const double *demand, double capacity, int num_edges,
    const int *edge_tail, const int *edge_head, const double *edge_x,
    void *existing_cuts, int max_num_cuts, double eps_integrality,
    double eps_violation, char *integer_and_feasible, double *max_violation,
    void *cuts);

#ifdef __cplusplus
}
#endif

#endif
