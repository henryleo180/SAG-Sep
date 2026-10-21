#include "bc.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "formulation.h"
#include "maxflow.h"
#include "padding.h"
#include "util.h"

/* ===================================================================== */
/* Cut identity: the mask key of a customer set                           */
/* ===================================================================== */

typedef struct {
    int64_t   words;
    uint64_t *keys;
    char     *used;
    int64_t   capacity;
    int64_t   count;
} bc_key_set;

static void key_set_init(bc_key_set *set, int64_t n_nodes)
{
    set->words = (n_nodes + 63) / 64;
    set->capacity = 1024;
    set->count = 0;
    set->keys = (uint64_t *)bc_calloc(
        (size_t)(set->capacity * set->words), sizeof *set->keys);
    set->used = (char *)bc_calloc((size_t)set->capacity, sizeof *set->used);
}

static void key_set_free(bc_key_set *set)
{
    free(set->keys);
    free(set->used);
    memset(set, 0, sizeof *set);
}

static void key_set_clear(bc_key_set *set)
{
    memset(set->used, 0, (size_t)set->capacity * sizeof *set->used);
    set->count = 0;
}

static uint64_t key_hash(const uint64_t *key, int64_t words)
{
    uint64_t hash = 1469598103934665603u;
    for (int64_t w = 0; w < words; ++w) {
        hash ^= key[w];
        hash *= 1099511628211u;
    }
    return hash;
}

static int64_t key_set_slot(const bc_key_set *set, const uint64_t *key, bool *found)
{
    int64_t slot =
        (int64_t)(key_hash(key, set->words) & (uint64_t)(set->capacity - 1));
    while (set->used[slot]) {
        if (memcmp(set->keys + slot * set->words, key,
                   (size_t)set->words * sizeof *key) == 0) {
            *found = true;
            return slot;
        }
        slot = (slot + 1) & (set->capacity - 1);
    }
    *found = false;
    return slot;
}

static bool key_set_contains(const bc_key_set *set, const uint64_t *key)
{
    bool found = false;
    (void)key_set_slot(set, key, &found);
    return found;
}

static void key_set_add(bc_key_set *set, const uint64_t *key)
{
    if ((set->count + 1) * 10 > set->capacity * 7) {
        bc_key_set grown;
        grown.words = set->words;
        grown.capacity = set->capacity * 2;
        grown.count = 0;
        grown.keys = (uint64_t *)bc_calloc(
            (size_t)(grown.capacity * grown.words), sizeof *grown.keys);
        grown.used =
            (char *)bc_calloc((size_t)grown.capacity, sizeof *grown.used);
        for (int64_t slot = 0; slot < set->capacity; ++slot) {
            if (!set->used[slot]) continue;
            bool found = false;
            const int64_t target =
                key_set_slot(&grown, set->keys + slot * set->words, &found);
            memcpy(grown.keys + target * grown.words,
                   set->keys + slot * set->words,
                   (size_t)set->words * sizeof *set->keys);
            grown.used[target] = 1;
            grown.count += 1;
        }
        key_set_free(set);
        *set = grown;
    }
    bool found = false;
    const int64_t slot = key_set_slot(set, key, &found);
    if (found) return;
    memcpy(set->keys + slot * set->words, key,
           (size_t)set->words * sizeof *key);
    set->used[slot] = 1;
    set->count += 1;
}

/* ===================================================================== */
/* Solver state                                                           */
/* ===================================================================== */

typedef struct {
    int64_t proposal;
    int64_t rhs;
    double  violation;
    double  inner_violation;
    double  cutset_violation;
    bool    add_inner;
    bool    add_cutset;
} bc_candidate;

typedef struct {
    double  violation;
    int64_t index;
    int     form;
} bc_row_rank;

static int compare_row_ranks(const void *lhs, const void *rhs)
{
    const bc_row_rank *a = (const bc_row_rank *)lhs;
    const bc_row_rank *b = (const bc_row_rank *)rhs;
    if (a->violation > b->violation) return -1;
    if (a->violation < b->violation) return 1;
    if (a->index != b->index) return a->index < b->index ? -1 : 1;
    return (a->form > b->form) - (a->form < b->form);
}

typedef struct {
    int64_t n_nodes;
    int64_t count;
    int64_t node_count;
    int64_t node_capacity;
    int64_t *indptr;
    int32_t *nodes;
    int64_t *rhs;
    double *cut_value;
    char *covered;
    char *shore;
} bc_connectivity_list;

static void connectivity_list_init(bc_connectivity_list *list, int64_t n_nodes)
{
    if (n_nodes < 1)
        bc_die("connectivity workspace needs at least the root");
    const int64_t customers = n_nodes - 1;
    if (customers > 0 && customers > INT64_MAX / customers)
        bc_die("connectivity set workspace is too large");

    memset(list, 0, sizeof *list);
    list->n_nodes = n_nodes;
    list->node_capacity = customers * customers;
    const size_t set_slots = (size_t)(customers > 0 ? customers : 1);
    const size_t node_slots =
        (size_t)(list->node_capacity > 0 ? list->node_capacity : 1);
    list->indptr =
        (int64_t *)bc_calloc(set_slots + 1, sizeof *list->indptr);
    list->nodes = (int32_t *)bc_malloc(node_slots * sizeof *list->nodes);
    list->rhs = (int64_t *)bc_malloc(set_slots * sizeof *list->rhs);
    list->cut_value =
        (double *)bc_malloc(set_slots * sizeof *list->cut_value);
    list->covered =
        (char *)bc_calloc((size_t)n_nodes, sizeof *list->covered);
    list->shore = (char *)bc_malloc((size_t)n_nodes * sizeof *list->shore);
}

static void connectivity_list_clear(bc_connectivity_list *list)
{
    list->count = 0;
    list->node_count = 0;
    list->indptr[0] = 0;
    memset(list->covered, 0, (size_t)list->n_nodes * sizeof *list->covered);
}

static void connectivity_list_free(bc_connectivity_list *list)
{
    free(list->indptr);
    free(list->nodes);
    free(list->rhs);
    free(list->cut_value);
    free(list->covered);
    free(list->shore);
    memset(list, 0, sizeof *list);
}

typedef struct {
    bc_model      model;
    bc_separator *sep;
    bc_options    opt;
    bc_callback_stats stats;
    double        start_time;

    double  *values;

    bc_proposals  proposals;
    bc_candidate *candidates;
    int64_t       n_candidates;
    int64_t      *accepted;
    int64_t       candidate_capacity;
    bc_row_rank  *row_ranks;
    int64_t       row_rank_capacity;

    bc_row     row;
    bc_setlist violations;

    int    *batch_begins;
    int    *batch_cols;
    double *batch_vals;
    double *batch_rhs;
    char   *batch_senses;
    int64_t batch_row_capacity;
    int64_t batch_nnz_capacity;

    bc_key_set root_inner_rows;
    bc_key_set root_cutset_rows;
    bc_key_set round_seen;
    bc_key_set root_connectivity_rows;
    bc_key_set connectivity_seen;
    uint64_t  *key_scratch;

    bc_maxflow maxflow;
    bc_connectivity_list connectivity;
    double  *cut_gather;

    bc_padding padding;
    double  padding_gap;
    int64_t padding_high_degree_max;
    double  padding_max_clip;
    double  padding_max_identity_gap;

    int64_t round;
    bool    proposed;
    long    connectivity_cuts;
    int64_t generator_inner_rows;
    int64_t generator_cutset_rows;
} bc_solver;

static void solver_reserve_candidates(bc_solver *s, int64_t rows)
{
    if (rows <= s->candidate_capacity) return;
    while (s->candidate_capacity < rows)
        s->candidate_capacity =
            s->candidate_capacity ? s->candidate_capacity * 2 : 256;
    s->candidates = (bc_candidate *)bc_realloc(
        s->candidates,
        (size_t)s->candidate_capacity * sizeof *s->candidates);
    s->accepted = (int64_t *)bc_realloc(
        s->accepted, (size_t)s->candidate_capacity * sizeof *s->accepted);
}

static void solver_cut_key(bc_solver *s, const int32_t *nodes, int64_t count)
{
    memset(s->key_scratch, 0,
           (size_t)s->root_inner_rows.words * sizeof *s->key_scratch);
    for (int64_t a = 0; a < count; ++a) {
        const int32_t node = nodes[a];
        s->key_scratch[node >> 6] |= (uint64_t)1 << (node & 63);
    }
}

static void proposal_nodes(const bc_proposals *proposals, int64_t p,
                           const int32_t **nodes, int64_t *count)
{
    const int64_t start = proposals->indptr[p];
    *nodes = proposals->nodes + start;
    *count = proposals->indptr[p + 1] - start;
}

static void connectivity_nodes(const bc_solver *s, int64_t index,
                               const int32_t **nodes, int64_t *count,
                               int64_t *rhs)
{
    const int64_t start = s->connectivity.indptr[index];
    *nodes = s->connectivity.nodes + start;
    *count = s->connectivity.indptr[index + 1] - start;
    *rhs = s->connectivity.rhs[index];
}

/* ===================================================================== */
/* Separation                                                             */
/* ===================================================================== */

static int compare_candidates(const void *lhs, const void *rhs)
{
    const bc_candidate *a = (const bc_candidate *)lhs;
    const bc_candidate *b = (const bc_candidate *)rhs;
    if (a->violation > b->violation) return -1;
    if (a->violation < b->violation) return 1;
    return (a->proposal > b->proposal) - (a->proposal < b->proposal);
}

static double connectivity_cut_value(
    bc_solver *s,
    const double *values,
    const char *shore
)
{
    int64_t gathered = 0;
    for (int k = 0; k < s->model.n_edge_vars; ++k) {
        const double value = values[k];
        if (!(value > BC_CONNECTIVITY_SUPPORT_THRESHOLD)) continue;
        if (shore[s->model.edge_i[k]] == shore[s->model.edge_j[k]]) continue;
        s->cut_gather[gathered++] = value;
    }
    return bc_reduce_sum(s->cut_gather, gathered);
}

static int64_t bc_separate_connectivity(bc_solver *s, const double *values)
{
    connectivity_list_clear(&s->connectivity);
    key_set_clear(&s->connectivity_seen);

    const double cutoff = 1.0 - s->opt.min_violation;
    if (cutoff <= 0.0) return 0;

    bc_maxflow_build_undirected(
        &s->maxflow,
        s->model.n_edge_vars,
        s->model.edge_i,
        s->model.edge_j,
        values,
        BC_CONNECTIVITY_SUPPORT_THRESHOLD);

    const int64_t n_nodes = s->model.inst->n_nodes;
    for (int target = 1; target < (int)n_nodes; ++target) {
        if (s->connectivity.covered[target]) continue;

        bc_maxflow_sink_shore(&s->maxflow, 0, target, s->connectivity.shore);
        const double cut_value =
            connectivity_cut_value(s, values, s->connectivity.shore);
        if (!(cut_value < cutoff)) continue;

        const int64_t start = s->connectivity.node_count;
        int64_t count = 0;
        for (int node = 1; node < (int)n_nodes; ++node) {
            if (!s->connectivity.shore[node]) continue;
            if (start + count >= s->connectivity.node_capacity)
                bc_die("connectivity set workspace exhausted");
            s->connectivity.nodes[start + count] = node;
            count += 1;
        }
        if (count == 0)
            bc_die("minimum-cut target shore is unexpectedly empty");

        solver_cut_key(s, s->connectivity.nodes + start, count);
        const bool duplicate =
            key_set_contains(&s->connectivity_seen, s->key_scratch)
            || key_set_contains(&s->root_connectivity_rows, s->key_scratch);
        if (duplicate) {
            for (int64_t a = 0; a < count; ++a)
                s->connectivity.covered[
                    s->connectivity.nodes[start + a]] = 1;
            continue;
        }

        if (s->connectivity.count >= n_nodes - 1)
            bc_die("connectivity separator emitted more than one set per customer");
        const int64_t row = s->connectivity.count;
        int64_t rhs = bc_capacity_rhs(
            &s->model, s->connectivity.nodes + start, count);
        if (rhs < 1) rhs = 1;

        s->connectivity.indptr[row] = start;
        s->connectivity.rhs[row] = rhs;
        s->connectivity.cut_value[row] = cut_value;
        s->connectivity.count += 1;
        s->connectivity.node_count = start + count;
        s->connectivity.indptr[s->connectivity.count] =
            s->connectivity.node_count;
        key_set_add(&s->connectivity_seen, s->key_scratch);

        for (int64_t a = 0; a < count; ++a)
            s->connectivity.covered[
                s->connectivity.nodes[start + a]] = 1;
    }
    return s->connectivity.count;
}

static void measure_padding_identity(bc_solver *s, int64_t count)
{
    if (!s->opt.padded_support) return;
    double worst = 0.0;
    for (int64_t i = 0; i < count; ++i) {
        const int32_t *nodes = NULL;
        int64_t size = 0;
        proposal_nodes(&s->proposals, s->candidates[i].proposal, &nodes, &size);
        const double gap = fabs(bc_padding_identity_gap(
            &s->padding, nodes, size, s->candidates[i].rhs,
            s->candidates[i].inner_violation));
        if (gap > worst) worst = gap;
    }
    s->padding_gap = worst;
    if (worst > s->padding_max_identity_gap)
        s->padding_max_identity_gap = worst;
}

static void apply_row_cap(bc_solver *s)
{
    if (s->n_candidates == 0) return;
    const int64_t capacity = 2 * s->n_candidates;
    if (capacity > s->row_rank_capacity) {
        s->row_rank_capacity = capacity;
        s->row_ranks = (bc_row_rank *)bc_realloc(
            s->row_ranks, (size_t)capacity * sizeof *s->row_ranks);
    }
    int64_t n_rows = 0;
    for (int64_t i = 0; i < s->n_candidates; ++i) {
        if (s->candidates[i].add_inner) {
            s->row_ranks[n_rows].violation = s->candidates[i].inner_violation;
            s->row_ranks[n_rows].index = i;
            s->row_ranks[n_rows].form = 0;
            n_rows += 1;
        }
        if (s->candidates[i].add_cutset) {
            s->row_ranks[n_rows].violation = s->candidates[i].cutset_violation;
            s->row_ranks[n_rows].index = i;
            s->row_ranks[n_rows].form = 1;
            n_rows += 1;
        }
    }
    qsort(s->row_ranks, (size_t)n_rows, sizeof *s->row_ranks, compare_row_ranks);
    if (n_rows > s->opt.cuts_per_round) n_rows = s->opt.cuts_per_round;

    for (int64_t i = 0; i < s->n_candidates; ++i) {
        s->candidates[i].add_inner = false;
        s->candidates[i].add_cutset = false;
        s->candidates[i].violation = -HUGE_VAL;
    }
    for (int64_t r = 0; r < n_rows; ++r) {
        bc_candidate *c = &s->candidates[s->row_ranks[r].index];
        if (s->row_ranks[r].form == 0) c->add_inner = true;
        else c->add_cutset = true;
        if (s->row_ranks[r].violation > c->violation)
            c->violation = s->row_ranks[r].violation;
    }
    int64_t kept = 0;
    for (int64_t i = 0; i < s->n_candidates; ++i) {
        if (!s->candidates[i].add_inner && !s->candidates[i].add_cutset) continue;
        s->candidates[kept++] = s->candidates[i];
    }
    s->n_candidates = kept;
}

static int64_t bc_separate(bc_solver *s, const double *values)
{
    s->round += 1;
    s->n_candidates = 0;
    s->padding_gap = 0.0;

    bc_dense_support(&s->model, values);
    bc_padding_build(&s->padding, s->model.dense, s->opt.padded_support);
    if (s->padding.high_degree > s->padding_high_degree_max)
        s->padding_high_degree_max = s->padding.high_degree;
    if (s->padding.max_clip > s->padding_max_clip)
        s->padding_max_clip = s->padding.max_clip;

    const int64_t m = s->padding.m;
    if (m == 0) {
        s->proposed = false;
        return 0;
    }
    s->proposed = true;
    s->sep->propose(
        s->sep, m, s->padding.u, s->padding.v, s->padding.x, &s->proposals);
    if (s->proposals.rows == 0) return 0;

    solver_reserve_candidates(s, s->proposals.rows);
    key_set_clear(&s->round_seen);
    for (int64_t p = 0; p < s->proposals.rows; ++p) {
        const int32_t *nodes = NULL;
        int64_t count = 0;
        proposal_nodes(&s->proposals, p, &nodes, &count);
        bc_check_subset(&s->model, nodes, count);
        const int64_t rhs = bc_capacity_rhs(&s->model, nodes, count);
        if (rhs <= 1) continue;
        solver_cut_key(s, nodes, count);
        if (key_set_contains(&s->round_seen, s->key_scratch)) continue;
        const double inner =
            bc_inner_violation_from_dense(&s->model, nodes, count, rhs);
        const double cutset =
            bc_cutset_violation_from_dense(&s->model, nodes, count, rhs);
        const bool add_inner =
            inner > s->opt.min_violation
            && !key_set_contains(&s->root_inner_rows, s->key_scratch);
        const bool add_cutset =
            cutset > s->opt.min_violation
            && !key_set_contains(&s->root_cutset_rows, s->key_scratch)
            && !key_set_contains(&s->root_connectivity_rows, s->key_scratch);
        if (!add_inner && !add_cutset) continue;
        key_set_add(&s->round_seen, s->key_scratch);
        s->candidates[s->n_candidates].proposal = p;
        s->candidates[s->n_candidates].rhs = rhs;
        s->candidates[s->n_candidates].violation =
            inner > cutset ? inner : cutset;
        s->candidates[s->n_candidates].inner_violation = inner;
        s->candidates[s->n_candidates].cutset_violation = cutset;
        s->candidates[s->n_candidates].add_inner = add_inner;
        s->candidates[s->n_candidates].add_cutset = add_cutset;
        s->n_candidates += 1;
    }

    apply_row_cap(s);
    qsort(s->candidates, (size_t)s->n_candidates,
          sizeof *s->candidates, compare_candidates);
    measure_padding_identity(s, s->n_candidates);
    return s->n_candidates;
}

static void bc_commit(bc_solver *s, int64_t count, bool permanent)
{
    if (!s->proposed) return;
    s->sep->commit(s->sep, s->accepted, count, permanent);
    s->proposed = false;
}

static void candidate_nodes(const bc_solver *s, int64_t index,
                            const int32_t **nodes, int64_t *count, int64_t *rhs)
{
    proposal_nodes(
        &s->proposals, s->candidates[index].proposal, nodes, count);
    *rhs = s->candidates[index].rhs;
}

/* ===================================================================== */
/* Root cutting-plane loop                                                */
/* ===================================================================== */

static const char *status_name(int status);

static void lp_termination(int status, char out[32])
{
    if (status == GRB_TIME_LIMIT) { snprintf(out, 32, "time_limit"); return; }
    const char *name = status_name(status);
    if (!name) { snprintf(out, 32, "lp_status_%d", status); return; }
    char lowered[24];
    size_t i = 0;
    for (; name[i] && i + 1 < sizeof lowered; ++i)
        lowered[i] = (name[i] >= 'A' && name[i] <= 'Z') ? (char)(name[i] - 'A' + 'a')
                                                        : name[i];
    lowered[i] = '\0';
    snprintf(out, 32, "lp_%s", lowered);
}

static double remaining_time(const bc_solver *s, bool *has_limit)
{
    *has_limit = s->opt.time_limit > 0.0;
    if (!*has_limit) return 0.0;
    return s->opt.time_limit - (bc_seconds() - s->start_time);
}

static bool out_of_time(const bc_solver *s)
{
    bool has_limit = false;
    const double remaining = remaining_time(s, &has_limit);
    return has_limit && remaining <= 0.0;
}

static bool apply_time_budget(bc_solver *s)
{
    bool has_limit = false;
    const double remaining = remaining_time(s, &has_limit);
    if (!has_limit) return true;
    if (remaining <= 0.0) return false;
    bc_grb_check(
        GRBsetdblparam(
            GRBgetenv(s->model.model), GRB_DBL_PAR_TIMELIMIT, remaining),
        s->model.model, "set TimeLimit (root LP)");
    return true;
}

static void reserve_batch(bc_solver *s, int64_t rows, int64_t nnz)
{
    if (rows > s->batch_row_capacity) {
        while (s->batch_row_capacity < rows)
            s->batch_row_capacity =
                s->batch_row_capacity ? s->batch_row_capacity * 2 : 128;
        s->batch_begins = (int *)bc_realloc(
            s->batch_begins,
            (size_t)s->batch_row_capacity * sizeof *s->batch_begins);
        s->batch_rhs = (double *)bc_realloc(
            s->batch_rhs,
            (size_t)s->batch_row_capacity * sizeof *s->batch_rhs);
        s->batch_senses = (char *)bc_realloc(
            s->batch_senses,
            (size_t)s->batch_row_capacity * sizeof *s->batch_senses);
    }
    if (nnz > s->batch_nnz_capacity) {
        while (s->batch_nnz_capacity < nnz)
            s->batch_nnz_capacity =
                s->batch_nnz_capacity ? s->batch_nnz_capacity * 2 : 4096;
        s->batch_cols = (int *)bc_realloc(
            s->batch_cols,
            (size_t)s->batch_nnz_capacity * sizeof *s->batch_cols);
        s->batch_vals = (double *)bc_realloc(
            s->batch_vals,
            (size_t)s->batch_nnz_capacity * sizeof *s->batch_vals);
    }
}

static int64_t add_root_connectivity_rows(bc_solver *s)
{
    int64_t added = 0;
    for (int64_t i = 0; i < s->connectivity.count; ++i) {
        const int32_t *nodes = NULL;
        int64_t count = 0;
        int64_t rhs = 0;
        connectivity_nodes(s, i, &nodes, &count, &rhs);
        bc_build_cutset_row(&s->model, nodes, count, rhs, &s->row);
        if (s->row.nnz == 0) continue;
        bc_grb_check(
            GRBaddconstr(
                s->model.model, s->row.nnz, s->row.cols, s->row.coeffs,
                s->row.sense, s->row.bound, NULL),
            s->model.model, "GRBaddconstr (root connectivity)");

        solver_cut_key(s, nodes, count);
        key_set_add(&s->root_connectivity_rows, s->key_scratch);
        added += 1;
    }
    if (added > 0)
        bc_grb_check(
            GRBupdatemodel(s->model.model),
            s->model.model, "GRBupdatemodel (root connectivity)");
    return added;
}

static void print_padding_line(const bc_solver *s, int round_index)
{
    printf(
        "[pad] round %d: c>2 customers=%lld, max clip=%.6g, "
        "max |v_cvrp/2 - v_in|=%.6g\n",
        round_index, (long long)s->padding.high_degree,
        s->padding.max_clip, s->padding_gap);
    fflush(stdout);
}

static void run_root_loop(
    bc_solver *s,
    bool *has_bound,
    double *bound,
    char termination[32],
    bool *bound_final,
    int *rounds,
    int *cuts,
    int *connectivity_cuts
)
{
    bc_model_relax_to_continuous(&s->model);
    *has_bound = false;
    *bound = 0.0;
    snprintf(termination, 32, "round_limit");
    *rounds = 0;
    *cuts = 0;
    *connectivity_cuts = 0;
    bool cuts_pending = false;

    for (int iteration = 0; iteration < s->opt.root_rounds; ++iteration) {
        if (!apply_time_budget(s)) { snprintf(termination, 32, "time_limit"); break; }
        bc_grb_check(
            GRBoptimize(s->model.model),
            s->model.model, "GRBoptimize (root LP)");
        int status = 0;
        bc_grb_check(
            GRBgetintattr(s->model.model, GRB_INT_ATTR_STATUS, &status),
            s->model.model, "get Status");
        if (status != GRB_OPTIMAL) {
            if (s->opt.verbose) {
                printf("[root] LP stopped with status %d\n", status);
                fflush(stdout);
            }
            lp_termination(status, termination);
            break;
        }

        *rounds += 1;
        bc_grb_check(
            GRBgetdblattr(s->model.model, GRB_DBL_ATTR_OBJVAL, bound),
            s->model.model, "get ObjVal");
        *has_bound = true;
        cuts_pending = false;
        bc_grb_check(
            GRBgetdblattrarray(
                s->model.model, GRB_DBL_ATTR_X, 0,
                s->model.n_edge_vars, s->values),
            s->model.model, "get X");

        if (out_of_time(s)) { snprintf(termination, 32, "time_limit"); break; }

        (void)bc_separate_connectivity(s, s->values);
        const int64_t exact_added = add_root_connectivity_rows(s);
        if (exact_added > INT_MAX - *connectivity_cuts)
            bc_die("too many root connectivity cuts to report");
        *connectivity_cuts += (int)exact_added;
        s->connectivity_cuts += (long)exact_added;

        int64_t found = 0;
        if (!s->opt.connectivity_only) {
            found = bc_separate(s, s->values);
            if (s->opt.verbose && s->opt.padded_support)
                print_padding_line(s, *rounds);
        }
        int64_t round_inner = 0;
        int64_t round_cutset = 0;
        for (int64_t i = 0; i < found; ++i) {
            if (s->candidates[i].add_inner) round_inner += 1;
            if (s->candidates[i].add_cutset) round_cutset += 1;
        }
        const int64_t round_rows = round_inner + round_cutset;
        if (found > 0) {
            int64_t nnz = 0;
            int64_t r = 0;
            reserve_batch(s, round_rows, 0);
            for (int64_t i = 0; i < found; ++i) {
                const int32_t *nodes = NULL;
                int64_t count = 0;
                int64_t rhs = 0;
                candidate_nodes(s, i, &nodes, &count, &rhs);
                for (int form = 0; form < 2; ++form) {
                    if (form == 0) {
                        if (!s->candidates[i].add_inner) continue;
                        bc_build_inner_row(
                            &s->model, nodes, count, rhs, &s->row);
                    } else {
                        if (!s->candidates[i].add_cutset) continue;
                        bc_build_cutset_row(
                            &s->model, nodes, count, rhs, &s->row);
                    }
                    reserve_batch(s, round_rows, nnz + s->row.nnz);
                    s->batch_begins[r] = (int)nnz;
                    s->batch_senses[r] = s->row.sense;
                    s->batch_rhs[r] = s->row.bound;
                    memcpy(
                        s->batch_cols + nnz, s->row.cols,
                        (size_t)s->row.nnz * sizeof *s->row.cols);
                    memcpy(
                        s->batch_vals + nnz, s->row.coeffs,
                        (size_t)s->row.nnz * sizeof *s->row.coeffs);
                    nnz += s->row.nnz;
                    r += 1;
                }
            }
            if (nnz > INT32_MAX)
                bc_die(
                    "root round %d produced %lld nonzeros, "
                    "more than one Gurobi call takes",
                    *rounds, (long long)nnz);
            bc_grb_check(
                GRBaddconstrs(
                    s->model.model, (int)round_rows, (int)nnz,
                    s->batch_begins, s->batch_cols, s->batch_vals,
                    s->batch_senses, s->batch_rhs, NULL),
                s->model.model, "GRBaddconstrs (root heuristic cuts)");
            bc_grb_check(
                GRBupdatemodel(s->model.model),
                s->model.model, "GRBupdatemodel");

            int64_t committed = 0;
            for (int64_t i = 0; i < found; ++i) {
                const int32_t *nodes = NULL;
                int64_t count = 0;
                int64_t rhs = 0;
                candidate_nodes(s, i, &nodes, &count, &rhs);
                solver_cut_key(s, nodes, count);
                if (s->candidates[i].add_inner) {
                    key_set_add(&s->root_inner_rows, s->key_scratch);
                    s->accepted[committed++] = s->candidates[i].proposal;
                }
                if (s->candidates[i].add_cutset)
                    key_set_add(&s->root_cutset_rows, s->key_scratch);
            }
            bc_commit(s, committed, true);
            if (round_rows > INT_MAX - *cuts)
                bc_die("too many root heuristic rows to report");
            *cuts += (int)round_rows;
            s->generator_inner_rows += round_inner;
            s->generator_cutset_rows += round_cutset;
        } else if (!s->opt.connectivity_only) {
            bc_commit(s, 0, true);
        }

        if (exact_added == 0 && found == 0) {
            snprintf(termination, 32, "closure");
            if (s->opt.verbose) {
                printf(
                    "[root] round %d: bound %.4f, exact pass and separator "
                    "found nothing \xe2\x80\x94 root converged\n",
                    *rounds, *bound);
                fflush(stdout);
            }
            break;
        }

        cuts_pending = true;
        if (s->opt.verbose) {
            if (found > 0) {
                printf(
                    "[root] round %d: bound %.4f, "
                    "+%lld connectivity cuts, +%lld heuristic rows "
                    "(%lld inner, %lld cutset), "
                    "max heuristic violation %.6f\n",
                    *rounds, *bound, (long long)exact_added,
                    (long long)round_rows, (long long)round_inner,
                    (long long)round_cutset, s->candidates[0].violation);
            } else {
                double max_violation = 0.0;
                for (int64_t i = 0; i < s->connectivity.count; ++i) {
                    const double violation =
                        (double)s->connectivity.rhs[i]
                        - s->connectivity.cut_value[i];
                    if (i == 0 || violation > max_violation)
                        max_violation = violation;
                }
                printf(
                    "[root] round %d: bound %.4f, "
                    "+%lld connectivity cuts, +0 heuristic rows "
                    "(0 inner, 0 cutset), "
                    "max connectivity violation %.6f\n",
                    *rounds, *bound, (long long)exact_added, max_violation);
            }
            fflush(stdout);
        }
    }

    if (cuts_pending && apply_time_budget(s)) {
        bc_grb_check(
            GRBoptimize(s->model.model),
            s->model.model, "GRBoptimize (root LP, final batch)");
        int status = 0;
        bc_grb_check(
            GRBgetintattr(s->model.model, GRB_INT_ATTR_STATUS, &status),
            s->model.model, "get Status");
        if (status == GRB_OPTIMAL) {
            bc_grb_check(
                GRBgetdblattr(
                    s->model.model, GRB_DBL_ATTR_OBJVAL, bound),
                s->model.model, "get ObjVal");
            *has_bound = true;
            cuts_pending = false;
            if (s->opt.verbose) {
                printf("[root] final batch priced: bound %.4f\n", *bound);
                fflush(stdout);
            }
        }
    }
    *bound_final = !cuts_pending;
}

static bool tree_commit_retains(const bc_options *options)
{
    return options->tree_commit && strcmp(options->tree_commit, "offered") == 0;
}

/* ===================================================================== */
/* Callbacks                                                              */
/* ===================================================================== */

static bool relaxation_is_integral(const double *values, int count)
{
    for (int k = 0; k < count; ++k) {
        const double nearest = nearbyint(values[k]);
        if (fabs(values[k] - nearest)
            > 1e-6 + 1e-5 * fabs(nearest))
            return false;
    }
    return true;
}

static bool all_finite(const double *values, int count)
{
    for (int k = 0; k < count; ++k) {
        if (!isfinite(values[k])) return false;
    }
    return true;
}

static void on_mipnode(bc_solver *s, void *cbdata)
{
    if (s->opt.usercut_max_nodes <= 0) return;
    int node_status = 0;
    bc_grb_check(
        GRBcbget(
            cbdata, GRB_CB_MIPNODE,
            GRB_CB_MIPNODE_STATUS, &node_status),
        NULL, "cbget MIPNODE_STATUS");
    if (node_status != GRB_OPTIMAL) return;

    double node_count = 0.0;
    bc_grb_check(
        GRBcbget(
            cbdata, GRB_CB_MIPNODE,
            GRB_CB_MIPNODE_NODCNT, &node_count),
        NULL, "cbget MIPNODE_NODCNT");
    if ((long long)node_count > s->opt.usercut_max_nodes) return;

    s->stats.mipnode_calls += 1;
    bc_grb_check(
        GRBcbget(
            cbdata, GRB_CB_MIPNODE,
            GRB_CB_MIPNODE_REL, s->values),
        NULL, "cbget MIPNODE_REL");
    if (!all_finite(s->values, s->model.n_edge_vars)) return;

    const int64_t exact = bc_separate_connectivity(s, s->values);
    for (int64_t i = 0; i < exact; ++i) {
        const int32_t *nodes = NULL;
        int64_t count = 0;
        int64_t rhs = 0;
        connectivity_nodes(s, i, &nodes, &count, &rhs);
        bc_build_cutset_row(&s->model, nodes, count, rhs, &s->row);
        if (s->row.nnz == 0) continue;
        bc_grb_check(
            GRBcbcut(
                cbdata, s->row.nnz, s->row.cols, s->row.coeffs,
                s->row.sense, s->row.bound),
            NULL, "GRBcbcut (connectivity)");
        s->connectivity_cuts += 1;
    }

    if (relaxation_is_integral(s->values, s->model.n_edge_vars))
        return;
    if (s->opt.connectivity_only) return;

    const int64_t found = bc_separate(s, s->values);
    int64_t accepted = 0;
    for (int64_t i = 0; i < found; ++i) {
        const int32_t *nodes = NULL;
        int64_t count = 0;
        int64_t rhs = 0;
        candidate_nodes(s, i, &nodes, &count, &rhs);
        bool added = false;
        for (int form = 0; form < 2; ++form) {
            if (form == 0) {
                if (!s->candidates[i].add_inner) continue;
                bc_build_inner_row(&s->model, nodes, count, rhs, &s->row);
            } else {
                if (!s->candidates[i].add_cutset) continue;
                bc_build_cutset_row(&s->model, nodes, count, rhs, &s->row);
            }
            if (s->row.nnz == 0) continue;
            bc_grb_check(
                GRBcbcut(
                    cbdata, s->row.nnz, s->row.cols, s->row.coeffs,
                    s->row.sense, s->row.bound),
                NULL, "GRBcbcut");
            s->stats.mipnode_cuts_added += 1;
            added = true;
        }
        if (added) s->accepted[accepted++] = s->candidates[i].proposal;
    }
    bc_commit(s, accepted, tree_commit_retains(&s->opt));
}

static void on_mipsol(bc_solver *s, void *cbdata)
{
    s->stats.mipsol_calls += 1;
    bc_grb_check(
        GRBcbget(
            cbdata, GRB_CB_MIPSOL,
            GRB_CB_MIPSOL_SOL, s->values),
        NULL, "cbget MIPSOL_SOL");
    const int64_t violated =
        bc_detect_violations(&s->model, s->values, &s->violations);
    for (int64_t i = 0; i < violated; ++i) {
        const int64_t start = s->violations.indptr[i];
        const int64_t count =
            s->violations.indptr[i + 1] - start;
        const int32_t *nodes = s->violations.nodes + start;
        if (s->violations.inner[i]) {
            bc_build_inner_row(
                &s->model, nodes, count,
                s->violations.rhs[i], &s->row);
        } else {
            bc_build_cutset_row(
                &s->model, nodes, count,
                s->violations.rhs[i], &s->row);
            s->stats.lazy_connectivity_cuts += 1;
        }
        if (s->row.nnz == 0) continue;
        bc_grb_check(
            GRBcblazy(
                cbdata, s->row.nnz, s->row.cols, s->row.coeffs,
                s->row.sense, s->row.bound),
            NULL, "GRBcblazy");
        s->stats.lazy_cuts_added += 1;
    }
}

static int callback(GRBmodel *model, void *cbdata, int where, void *usrdata)
{
    (void)model;
    bc_solver *s = (bc_solver *)usrdata;
    if (where == GRB_CB_MIPNODE) on_mipnode(s, cbdata);
    else if (where == GRB_CB_MIPSOL) on_mipsol(s, cbdata);
    return 0;
}

/* ===================================================================== */
/* Driver                                                                 */
/* ===================================================================== */

static const char *status_name(int status)
{
    switch (status) {
    case GRB_OPTIMAL:         return "OPTIMAL";
    case GRB_INFEASIBLE:      return "INFEASIBLE";
    case GRB_UNBOUNDED:       return "UNBOUNDED";
    case GRB_CUTOFF:          return "CUTOFF";
    case GRB_ITERATION_LIMIT: return "ITERATION_LIMIT";
    case GRB_NODE_LIMIT:      return "NODE_LIMIT";
    case GRB_TIME_LIMIT:      return "TIME_LIMIT";
    case GRB_SOLUTION_LIMIT:  return "SOLUTION_LIMIT";
    case GRB_INTERRUPTED:     return "INTERRUPTED";
    case GRB_INF_OR_UNBD:     return "INF_OR_UNBD";
    case GRB_NUMERIC:         return "NUMERIC";
    case GRB_USER_OBJ_LIMIT:  return "USER_OBJ_LIMIT";
    case GRB_WORK_LIMIT:      return "WORK_LIMIT";
    case GRB_MEM_LIMIT:       return "MEM_LIMIT";
    case GRB_SUBOPTIMAL:      return "SUBOPTIMAL";
    default:                  return NULL;
    }
}

static bool is_reportable(double value)
{
    return isfinite(value) && fabs(value) < 1e99;
}

static bool try_get_double(GRBmodel *model, const char *name, double *out)
{
    return GRBgetdblattr(model, name, out) == 0;
}

static void set_param_int(GRBmodel *model, const char *name, int value)
{
    bc_grb_check(
        GRBsetintparam(GRBgetenv(model), name, value), model, name);
}

static void set_param_dbl(GRBmodel *model, const char *name, double value)
{
    bc_grb_check(
        GRBsetdblparam(GRBgetenv(model), name, value), model, name);
}

static void validate_options(const bc_options *options)
{
    if (options->cuts_per_round <= 0)
        bc_die("--cuts-per-round must be greater than 0 (got %d)",
               options->cuts_per_round);
    if (options->root_rounds < 0)
        bc_die("--root-rounds must be nonnegative (got %d)", options->root_rounds);
    if (!(options->min_violation >= 0.0))
        bc_die("--min-violation must be nonnegative (got %.17g)",
               options->min_violation);
    if (!(options->time_limit >= 0.0))
        bc_die("--time-limit must be non-negative, 0 meaning no limit (got %.17g)",
               options->time_limit);
    if (options->threads < 1)
        bc_die("--threads must be at least 1 (got %d)", options->threads);
    if (options->usercut_max_nodes < 0)
        bc_die("--usercut-max-nodes must be nonnegative (got %d)",
               options->usercut_max_nodes);
}

void bc_solve(GRBenv *env, const bc_instance *inst, bc_separator *separator,
              const bc_options *options, bool has_optimal, double optimal_value,
              double setup_start, char detail_buf[BC_DETAIL_MAX], bc_result *out)
{
    validate_options(options);

    bc_solver s;
    memset(&s, 0, sizeof s);
    s.sep = separator;
    s.opt = *options;

    bc_model_build(env, inst, &s.model);
    set_param_int(
        s.model.model, GRB_INT_PAR_OUTPUTFLAG,
        options->verbose ? 1 : 0);
    set_param_int(
        s.model.model, GRB_INT_PAR_THREADS,
        options->threads > 1 ? options->threads : 1);
    set_param_int(s.model.model, GRB_INT_PAR_SEED, options->seed);
    s.start_time = bc_seconds();
    const double setup_time = s.start_time - setup_start;

    if (inst->n_nodes > INT_MAX)
        bc_die("instance has too many nodes for the max-flow implementation");
    s.values = (double *)bc_malloc(
        (size_t)s.model.n_edge_vars * sizeof *s.values);
    s.cut_gather = (double *)bc_malloc(
        (size_t)s.model.n_edge_vars * sizeof *s.cut_gather);
    bc_row_init(&s.model, &s.row);
    bc_setlist_init(&s.violations);
    key_set_init(&s.root_inner_rows, inst->n_nodes);
    key_set_init(&s.root_cutset_rows, inst->n_nodes);
    key_set_init(&s.round_seen, inst->n_nodes);
    key_set_init(&s.root_connectivity_rows, inst->n_nodes);
    key_set_init(&s.connectivity_seen, inst->n_nodes);
    s.key_scratch = (uint64_t *)bc_calloc(
        (size_t)s.root_inner_rows.words, sizeof *s.key_scratch);
    bc_maxflow_init(
        &s.maxflow, (int)inst->n_nodes, s.model.n_edge_vars);
    connectivity_list_init(&s.connectivity, inst->n_nodes);
    bc_padding_init(&s.padding, inst->n_nodes, s.model.n_edge_vars);
    solver_reserve_candidates(&s, 256);

    bool has_root_bound = false;
    double root_bound = 0.0;
    char root_termination[32] = "round_limit";
    bool root_bound_final = true;
    int root_rounds = 0;
    int root_cuts = 0;
    int root_connectivity_cuts = 0;
    run_root_loop(
        &s, &has_root_bound, &root_bound, root_termination, &root_bound_final,
        &root_rounds, &root_cuts, &root_connectivity_cuts);
    const double root_time = bc_seconds() - s.start_time;

    if (options->verbose) {
        char bound_text[64];
        if (has_root_bound)
            bc_format_double(bound_text, sizeof bound_text, root_bound);
        else
            snprintf(bound_text, sizeof bound_text, "None");
        printf(
            "[root] done: bound=%s, rounds=%d, cuts=%d, "
            "connectivity_cuts=%d, %.2fs\n",
            bound_text, root_rounds, root_cuts,
            root_connectivity_cuts, root_time);
        fflush(stdout);
    }

    bool has_limit = false;
    double remaining = remaining_time(&s, &has_limit);
    bool mip_ran = !has_limit || remaining > 0.0;

    if (mip_ran) {
        bc_model_restore_integer_types(&s.model);
        set_param_dbl(
            s.model.model, GRB_DBL_PAR_MIPGAP, options->mip_gap);
        set_param_int(
            s.model.model, GRB_INT_PAR_LAZYCONSTRAINTS, 1);
        if (options->usercut_max_nodes > 0)
            set_param_int(
                s.model.model, GRB_INT_PAR_PRECRUSH, 1);
        bc_grb_check(
            GRBsetcallbackfunc(s.model.model, callback, &s),
            s.model.model, "GRBsetcallbackfunc");

        remaining = remaining_time(&s, &has_limit);
        if (has_limit && remaining <= 0.0) {
            mip_ran = false;
        } else {
            if (has_limit)
                set_param_dbl(
                    s.model.model, GRB_DBL_PAR_TIMELIMIT, remaining);
            if (options->verbose) {
                printf("[mip] branch and bound \xe2\x80\xa6\n");
                fflush(stdout);
            }
            bc_grb_check(
                GRBoptimize(s.model.model),
                s.model.model, "GRBoptimize (branch and bound)");
        }
    }

    memset(out, 0, sizeof *out);
    out->instance_name = inst->name;
    out->n_customers = inst->n_nodes - 1;
    out->capacity = inst->capacity;
    out->root_degree = s.model.root_degree;
    out->separator = separator->name;
    separator->describe(separator, detail_buf, BC_DETAIL_MAX);
    out->separator_detail = detail_buf;

    if (mip_ran) {
        bc_grb_check(
            GRBgetintattr(
                s.model.model, GRB_INT_ATTR_STATUS, &out->status),
            s.model.model, "get Status");
        out->has_best_bound = try_get_double(
            s.model.model, GRB_DBL_ATTR_OBJBOUND, &out->best_bound);
        if (!try_get_double(
                s.model.model, GRB_DBL_ATTR_NODECOUNT, &out->node_count))
            out->node_count = 0.0;
        if (GRBgetintattr(
                s.model.model, GRB_INT_ATTR_SOLCOUNT, &out->sol_count) != 0)
            out->sol_count = 0;
        if (out->sol_count > 0) {
            out->has_objective = try_get_double(
                s.model.model, GRB_DBL_ATTR_OBJVAL, &out->objective);
            out->has_gap = try_get_double(
                s.model.model, GRB_DBL_ATTR_MIPGAP, &out->gap);
        }
    } else {
        out->status = GRB_TIME_LIMIT;
        out->has_best_bound = has_root_bound;
        out->best_bound = root_bound;
    }
    if (!out->has_best_bound && has_root_bound) {
        out->has_best_bound = true;
        out->best_bound = root_bound;
    }

    out->has_objective =
        out->has_objective && is_reportable(out->objective);
    out->has_best_bound =
        out->has_best_bound && is_reportable(out->best_bound);
    out->has_gap = out->has_gap && is_reportable(out->gap);
    out->has_root_bound =
        has_root_bound && is_reportable(root_bound);
    const char *name = status_name(out->status);
    if (name)
        snprintf(out->status_name, sizeof out->status_name, "%s", name);
    else
        snprintf(
            out->status_name, sizeof out->status_name,
            "STATUS_%d", out->status);

    out->runtime = bc_seconds() - s.start_time;
    out->setup_time = setup_time;
    out->root_bound = root_bound;
    snprintf(out->root_termination, sizeof out->root_termination, "%s",
             root_termination);
    out->root_bound_final = root_bound_final;
    out->root_rounds = root_rounds;
    out->root_cuts = root_cuts;
    out->generator_inner_rows = (int)s.generator_inner_rows;
    out->generator_cutset_rows = (int)s.generator_cutset_rows;
    out->connectivity_cuts = s.connectivity_cuts;
    out->root_time = root_time;
    out->padded_support = options->padded_support;
    out->connectivity_only = options->connectivity_only;
    out->padding_high_degree_max = s.padding_high_degree_max;
    out->padding_max_clip = s.padding_max_clip;
    out->padding_max_identity_gap = s.padding_max_identity_gap;
    out->callback = s.stats;
    out->has_optimal = has_optimal;
    out->optimal_value = optimal_value;
    out->has_matches_optimum =
        has_optimal && out->has_objective && out->status == GRB_OPTIMAL;
    out->matches_optimum =
        out->has_matches_optimum
        && fabs(out->objective - optimal_value) < 0.5;

    free(s.values);
    free(s.cut_gather);
    free(s.candidates);
    free(s.accepted);
    free(s.row_ranks);
    free(s.batch_begins);
    free(s.batch_cols);
    free(s.batch_vals);
    free(s.batch_rhs);
    free(s.batch_senses);
    free(s.key_scratch);
    key_set_free(&s.root_inner_rows);
    key_set_free(&s.root_cutset_rows);
    key_set_free(&s.round_seen);
    key_set_free(&s.root_connectivity_rows);
    key_set_free(&s.connectivity_seen);
    bc_maxflow_free(&s.maxflow);
    connectivity_list_free(&s.connectivity);
    bc_padding_free(&s.padding);
    bc_row_free(&s.row);
    bc_setlist_free(&s.violations);
    bc_model_free(&s.model);
}
