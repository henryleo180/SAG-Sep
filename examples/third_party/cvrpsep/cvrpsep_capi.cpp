#include "cvrpsep_capi.h"

#include <cstddef>
#include <cstdio>

#include "capsep.h"
#include "cnstrmgr.h"

extern "C" void *cvrpsep_capi_create_cmgr(int dim) {
  if (dim <= 0) {
    return NULL;
  }
  CnstrMgrPointer cmgr = NULL;
  CMGR_CreateCMgr(&cmgr, dim);
  return static_cast<void *>(cmgr);
}

extern "C" void cvrpsep_capi_free_cmgr(void *cmgr) {
  if (cmgr == NULL) {
    return;
  }
  CnstrMgrPointer handle = static_cast<CnstrMgrPointer>(cmgr);
  CMGR_FreeMemCMgr(&handle);
}

extern "C" void cvrpsep_capi_move_cnstr(void *source, void *sink,
                                        int source_index, int sink_index) {
  if (source == NULL || sink == NULL) {
    return;
  }
  CMGR_MoveCnstr(static_cast<CnstrMgrPointer>(source),
                 static_cast<CnstrMgrPointer>(sink), source_index, sink_index);
}

extern "C" void cvrpsep_capi_separate_capacity_cuts(
    int num_customers, const double *demand, double capacity, int num_edges,
    const int *edge_tail, const int *edge_head, const double *edge_x,
    void *existing_cuts, int max_num_cuts, double eps_integrality,
    double eps_violation, char *integer_and_feasible, double *max_violation,
    void *cuts) {
  CAPSEP_SeparateCapCuts(num_customers, demand, capacity, num_edges, edge_tail,
                         edge_head, edge_x,
                         static_cast<CnstrMgrPointer>(existing_cuts),
                         max_num_cuts, eps_integrality, eps_violation,
                         integer_and_feasible, max_violation,
                         static_cast<CnstrMgrPointer>(cuts));
}
