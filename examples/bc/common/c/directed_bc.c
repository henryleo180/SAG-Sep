#include "directed_bc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "directed_model.h"
#include "util.h"

#define BCD_SUPPORT_THRESHOLD 1e-8

/* ===================================================================== */
/* Cut identity: the mask key of a customer set                           */
/* ===================================================================== */

typedef struct {
    int64_t   words;
    uint64_t *keys;
    char     *used;
    int64_t   capacity;
    int64_t   count;
} bcd_key_set;

static void key_set_init(bcd_key_set *set, int64_t n_nodes)
{
    set->words = (n_nodes + 63) / 64;
    set->capacity = 1024;
    set->count = 0;
    set->keys = (uint64_t *)bc_calloc((size_t)(set->capacity * set->words), sizeof *set->keys);
    set->used = (char *)bc_calloc((size_t)set->capacity, sizeof *set->used);
}

static void key_set_free(bcd_key_set *set)
{
    free(set->keys);
    free(set->used);
    memset(set, 0, sizeof *set);
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

static int64_t key_set_slot(const bcd_key_set *set, const uint64_t *key, bool *found)
{
    int64_t slot = (int64_t)(key_hash(key, set->words) & (uint64_t)(set->capacity - 1));
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

static void key_set_clear(bcd_key_set *set)
{
    memset(set->used, 0, (size_t)set->capacity * sizeof *set->used);
    set->count = 0;
}

static bool key_set_contains(const bcd_key_set *set, const uint64_t *key)
{
    bool found = false;
    (void)key_set_slot(set, key, &found);
    return found;
}

static void key_set_add(bcd_key_set *set, const uint64_t *key)
{
    if ((set->count + 1) * 10 > set->capacity * 7) {
        bcd_key_set grown;
        grown.words = set->words;
        grown.capacity = set->capacity * 2;
        grown.count = 0;
        grown.keys = (uint64_t *)bc_calloc((size_t)(grown.capacity * grown.words),
                                           sizeof *grown.keys);
        grown.used = (char *)bc_calloc((size_t)grown.capacity, sizeof *grown.used);
        for (int64_t slot = 0; slot < set->capacity; ++slot) {
            if (!set->used[slot]) continue;
            bool found = false;
            const int64_t target = key_set_slot(&grown, set->keys + slot * set->words, &found);
            memcpy(grown.keys + target * grown.words, set->keys + slot * set->words,
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
    memcpy(set->keys + slot * set->words, key, (size_t)set->words * sizeof *key);
    set->used[slot] = 1;
    set->count += 1;
}

/* ===================================================================== */
/* Solver state                                                           */
/* ===================================================================== */

typedef struct {
    int64_t proposal;
    double  rhs;
    double  violation;
} bcd_candidate;

typedef struct {
    bcd_model     model;
    bc_separator *sep;
    bcd_options   opt;
    bcd_callback_stats stats;
    double        start_time;

    double  *values;
    int64_t *support_u;
    int64_t *support_v;
    double  *support_x;

    bc_proposals   proposals;
    bcd_candidate *candidates;
    int64_t        n_candidates;
    int64_t       *accepted;
    int64_t        candidate_capacity;

    bcd_row     row;
    bcd_setlist violations;

    int    *batch_begins;
    int    *batch_cols;
    double *batch_vals;
    double *batch_rhs;
    char   *batch_senses;
    int64_t batch_row_capacity;
    int64_t batch_nnz_capacity;

    bcd_key_set root_rows;
    bcd_key_set round_rows;
    uint64_t   *key_scratch;

    int64_t round;
    bool    proposed;
} bcd_solver;

static void solver_reserve_candidates(bcd_solver *s, int64_t rows)
{
    if (rows <= s->candidate_capacity) return;
    while (s->candidate_capacity < rows)
        s->candidate_capacity = s->candidate_capacity ? s->candidate_capacity * 2 : 256;
    s->candidates = (bcd_candidate *)bc_realloc(
        s->candidates, (size_t)s->candidate_capacity * sizeof *s->candidates);
    s->accepted = (int64_t *)bc_realloc(
        s->accepted, (size_t)s->candidate_capacity * sizeof *s->accepted);
}

static void solver_cut_key(bcd_solver *s, const int32_t *nodes, int64_t count)
{
    memset(s->key_scratch, 0, (size_t)s->root_rows.words * sizeof *s->key_scratch);
    for (int64_t a = 0; a < count; ++a) {
        const int32_t node = nodes[a];
        s->key_scratch[node >> 6] |= (uint64_t)1 << (node & 63);
    }
}

/* ===================================================================== */
/* Separation                                                             */
/* ===================================================================== */

static int compare_candidates(const void *lhs, const void *rhs)
{
    const bcd_candidate *a = (const bcd_candidate *)lhs;
    const bcd_candidate *b = (const bcd_candidate *)rhs;
    if (a->violation > b->violation) return -1;
    if (a->violation < b->violation) return 1;
    return (a->proposal > b->proposal) - (a->proposal < b->proposal);
}

static int64_t bcd_separate(bcd_solver *s, const double *values)
{
    s->round += 1;
    s->n_candidates = 0;

    const double *y = bcd_model_project(&s->model, values);
    int64_t m = 0;
    for (int64_t p = 0; p < s->model.n_pairs; ++p) {
        if (y[p] <= BCD_SUPPORT_THRESHOLD) continue;
        s->support_u[m] = s->model.pair_u[p];
        s->support_v[m] = s->model.pair_v[p];
        s->support_x[m] = y[p];
        m += 1;
    }
    if (m == 0) {
        s->proposed = false;
        return 0;
    }
    s->proposed = true;
    s->sep->propose(s->sep, m, s->support_u, s->support_v, s->support_x, &s->proposals);
    if (s->proposals.rows == 0) return 0;

    bcd_model_dense_support(&s->model);
    solver_reserve_candidates(s, s->proposals.rows);
    key_set_clear(&s->round_rows);
    if (s->proposals.indptr[0] != 0)
        bc_die("separator returned a CSR whose first offset is %lld, not 0",
               (long long)s->proposals.indptr[0]);
    for (int64_t p = 0; p < s->proposals.rows; ++p) {
        const int64_t start = s->proposals.indptr[p];
        const int64_t count = s->proposals.indptr[p + 1] - start;
        if (count < 0)
            bc_die("separator returned a CSR whose offsets decrease at row %lld",
                   (long long)p);
        const int32_t *nodes = s->proposals.nodes + start;
        bcd_check_subset(&s->model, nodes, count);
        solver_cut_key(s, nodes, count);
        if (key_set_contains(&s->root_rows, s->key_scratch)) continue;
        if (key_set_contains(&s->round_rows, s->key_scratch)) continue;
        key_set_add(&s->round_rows, s->key_scratch);
        const double rhs = bcd_rcc_rhs(&s->model, nodes, count);
        const double violation =
            bcd_violation_from_dense(&s->model, nodes, count, rhs);
        if (violation <= s->opt.min_violation) continue;
        s->candidates[s->n_candidates].proposal = p;
        s->candidates[s->n_candidates].rhs = rhs;
        s->candidates[s->n_candidates].violation = violation;
        s->n_candidates += 1;
    }

    qsort(s->candidates, (size_t)s->n_candidates, sizeof *s->candidates, compare_candidates);
    if (s->n_candidates > s->opt.cuts_per_round) s->n_candidates = s->opt.cuts_per_round;
    return s->n_candidates;
}

static void bcd_commit(bcd_solver *s, int64_t count, bool permanent)
{
    if (!s->proposed) return;
    s->sep->commit(s->sep, s->accepted, count, permanent);
    s->proposed = false;
}

static void candidate_nodes(const bcd_solver *s, int64_t index,
                            const int32_t **nodes, int64_t *count, double *rhs)
{
    const int64_t p = s->candidates[index].proposal;
    const int64_t start = s->proposals.indptr[p];
    *nodes = s->proposals.nodes + start;
    *count = s->proposals.indptr[p + 1] - start;
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

static double remaining_time(const bcd_solver *s)
{
    if (s->opt.time_limit <= 0.0) return HUGE_VAL;
    return s->opt.time_limit - (bc_seconds() - s->start_time);
}

static bool out_of_time(const bcd_solver *s)
{
    return remaining_time(s) <= 0.0;
}

static bool apply_time_budget(bcd_solver *s)
{
    const double remaining = remaining_time(s);
    if (remaining == HUGE_VAL) return true;
    if (remaining <= 0.0) return false;
    bcd_grb_check(GRBsetdblparam(GRBgetenv(s->model.model), GRB_DBL_PAR_TIMELIMIT,
                                 remaining),
                  s->model.model, "set TimeLimit");
    return true;
}

static bool solve_root_lp(bcd_solver *s, char termination[32])
{
    if (!apply_time_budget(s)) {
        if (termination) snprintf(termination, 32, "time_limit");
        return false;
    }
    bcd_grb_check(GRBoptimize(s->model.model), s->model.model, "GRBoptimize (root LP)");
    int status = 0;
    bcd_grb_check(GRBgetintattr(s->model.model, GRB_INT_ATTR_STATUS, &status),
                  s->model.model, "get Status");
    if (status != GRB_OPTIMAL) {
        if (s->opt.verbose) {
            printf("[root] LP stopped with status %d\n", status);
            fflush(stdout);
        }
        if (termination) lp_termination(status, termination);
        return false;
    }
    return true;
}

static void reserve_batch(bcd_solver *s, int64_t rows, int64_t nnz)
{
    if (rows > s->batch_row_capacity) {
        while (s->batch_row_capacity < rows)
            s->batch_row_capacity = s->batch_row_capacity ? s->batch_row_capacity * 2 : 128;
        s->batch_begins = (int *)bc_realloc(s->batch_begins,
                                            (size_t)s->batch_row_capacity * sizeof *s->batch_begins);
        s->batch_rhs = (double *)bc_realloc(s->batch_rhs,
                                            (size_t)s->batch_row_capacity * sizeof *s->batch_rhs);
        s->batch_senses = (char *)bc_realloc(s->batch_senses,
                                             (size_t)s->batch_row_capacity * sizeof *s->batch_senses);
    }
    if (nnz > s->batch_nnz_capacity) {
        while (s->batch_nnz_capacity < nnz)
            s->batch_nnz_capacity = s->batch_nnz_capacity ? s->batch_nnz_capacity * 2 : 4096;
        s->batch_cols = (int *)bc_realloc(s->batch_cols,
                                          (size_t)s->batch_nnz_capacity * sizeof *s->batch_cols);
        s->batch_vals = (double *)bc_realloc(s->batch_vals,
                                             (size_t)s->batch_nnz_capacity * sizeof *s->batch_vals);
    }
}

static void run_root_loop(bcd_solver *s, bool *has_bound, double *bound,
                          char termination[32], bool *bound_final,
                          int *rounds, int *cuts)
{
    bcd_model_relax_to_continuous(&s->model);
    *has_bound = false;
    *bound = 0.0;
    snprintf(termination, 32, "round_limit");
    *rounds = 0;
    *cuts = 0;

    bool cuts_pending = false;
    for (int iteration = 0; iteration < s->opt.root_rounds; ++iteration) {
        if (!solve_root_lp(s, termination)) break;
        *rounds += 1;
        cuts_pending = false;
        bcd_grb_check(GRBgetdblattr(s->model.model, GRB_DBL_ATTR_OBJVAL, bound),
                      s->model.model, "get ObjVal");
        *has_bound = true;
        bcd_grb_check(GRBgetdblattrarray(s->model.model, GRB_DBL_ATTR_X, 0,
                                         s->model.n_arc_vars, s->values),
                      s->model.model, "get X");

        if (out_of_time(s)) { snprintf(termination, 32, "time_limit"); break; }
        const int64_t found = bcd_separate(s, s->values);
        if (found == 0) {
            snprintf(termination, 32, "closure");
            bcd_commit(s, 0, true);
            if (s->opt.verbose) {
                printf("[root] round %d: bound %.4f, "
                       "separator found nothing \xe2\x80\x94 root converged\n", *rounds, *bound);
                fflush(stdout);
            }
            break;
        }

        int64_t nnz = 0;
        reserve_batch(s, found, 0);
        for (int64_t i = 0; i < found; ++i) {
            const int32_t *nodes = NULL;
            int64_t count = 0;
            double rhs = 0.0;
            candidate_nodes(s, i, &nodes, &count, &rhs);
            bcd_model_cut_row(&s->model, nodes, count, rhs, &s->row);
            reserve_batch(s, found, nnz + s->row.nnz);
            s->batch_begins[i] = (int)nnz;
            s->batch_senses[i] = GRB_GREATER_EQUAL;
            s->batch_rhs[i] = s->row.bound;
            memcpy(s->batch_cols + nnz, s->row.cols, (size_t)s->row.nnz * sizeof *s->row.cols);
            memcpy(s->batch_vals + nnz, s->row.coeffs, (size_t)s->row.nnz * sizeof *s->row.coeffs);
            nnz += s->row.nnz;
        }
        if (nnz > INT32_MAX)
            bc_die("root round %d produced %lld nonzeros, more than one Gurobi call takes",
                   *rounds, (long long)nnz);
        bcd_grb_check(GRBaddconstrs(s->model.model, (int)found, (int)nnz, s->batch_begins,
                                    s->batch_cols, s->batch_vals, s->batch_senses,
                                    s->batch_rhs, NULL),
                      s->model.model, "GRBaddconstrs (root cuts)");
        bcd_grb_check(GRBupdatemodel(s->model.model), s->model.model, "GRBupdatemodel");

        for (int64_t i = 0; i < found; ++i) {
            const int32_t *nodes = NULL;
            int64_t count = 0;
            double rhs = 0.0;
            candidate_nodes(s, i, &nodes, &count, &rhs);
            solver_cut_key(s, nodes, count);
            key_set_add(&s->root_rows, s->key_scratch);
            s->accepted[i] = s->candidates[i].proposal;
        }
        bcd_commit(s, found, true);
        *cuts += (int)found;
        cuts_pending = true;

        if (s->opt.verbose) {
            printf("[root] round %d: bound %.4f, +%lld cuts, max violation %.6f\n",
                   *rounds, *bound, (long long)found, s->candidates[0].violation);
            fflush(stdout);
        }
    }

    if (cuts_pending) {
        if (apply_time_budget(s)) {
            bcd_grb_check(GRBoptimize(s->model.model), s->model.model,
                          "GRBoptimize (root re-solve)");
            int status = 0;
            bcd_grb_check(GRBgetintattr(s->model.model, GRB_INT_ATTR_STATUS, &status),
                          s->model.model, "get Status");
            if (status == GRB_OPTIMAL) {
                bcd_grb_check(GRBgetdblattr(s->model.model, GRB_DBL_ATTR_OBJVAL, bound),
                              s->model.model, "get ObjVal");
                *has_bound = true;
                cuts_pending = false;
                if (s->opt.verbose) {
                    printf("[root] final re-solve: bound %.4f\n", *bound);
                    fflush(stdout);
                }
            } else if (s->opt.verbose) {
                printf("[root] final re-solve did not reach optimality; the bound "
                       "reported is the last certified one\n");
                fflush(stdout);
            }
        } else if (s->opt.verbose) {
            printf("[root] no budget left for the final re-solve; the bound reported "
                   "is the last certified one\n");
            fflush(stdout);
        }
    }
    *bound_final = !cuts_pending;
}

static bool tree_commit_retains(const bcd_options *options)
{
    return options->tree_commit && strcmp(options->tree_commit, "offered") == 0;
}

/* ===================================================================== */
/* Callbacks                                                              */
/* ===================================================================== */

static bool relaxation_is_integral(const double *values, int count)
{
    for (int k = 0; k < count; ++k) {
        if (fabs(values[k] - nearbyint(values[k])) > 1e-6) return false;
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

static void on_mipnode(bcd_solver *s, void *cbdata)
{
    if (s->opt.usercut_max_nodes <= 0) return;
    int node_status = 0;
    bcd_grb_check(GRBcbget(cbdata, GRB_CB_MIPNODE, GRB_CB_MIPNODE_STATUS, &node_status),
                  NULL, "cbget MIPNODE_STATUS");
    if (node_status != GRB_OPTIMAL) return;
    double node_count = 0.0;
    bcd_grb_check(GRBcbget(cbdata, GRB_CB_MIPNODE, GRB_CB_MIPNODE_NODCNT, &node_count),
                  NULL, "cbget MIPNODE_NODCNT");
    if ((long long)node_count > s->opt.usercut_max_nodes) return;

    s->stats.mipnode_calls += 1;
    bcd_grb_check(GRBcbget(cbdata, GRB_CB_MIPNODE, GRB_CB_MIPNODE_REL, s->values),
                  NULL, "cbget MIPNODE_REL");
    if (!all_finite(s->values, s->model.n_arc_vars)) return;
    if (relaxation_is_integral(s->values, s->model.n_arc_vars)) return;

    const int64_t found = bcd_separate(s, s->values);
    int64_t accepted = 0;
    for (int64_t i = 0; i < found; ++i) {
        const int32_t *nodes = NULL;
        int64_t count = 0;
        double rhs = 0.0;
        candidate_nodes(s, i, &nodes, &count, &rhs);
        bcd_model_cut_row(&s->model, nodes, count, rhs, &s->row);
        if (s->row.nnz == 0) continue;
        bcd_grb_check(GRBcbcut(cbdata, s->row.nnz, s->row.cols, s->row.coeffs,
                               GRB_GREATER_EQUAL, s->row.bound),
                      NULL, "GRBcbcut");
        s->accepted[accepted++] = s->candidates[i].proposal;
        s->stats.mipnode_cuts_added += 1;
    }
    bcd_commit(s, accepted, tree_commit_retains(&s->opt));
}

static void on_mipsol(bcd_solver *s, void *cbdata)
{
    s->stats.mipsol_calls += 1;
    bcd_grb_check(GRBcbget(cbdata, GRB_CB_MIPSOL, GRB_CB_MIPSOL_SOL, s->values),
                  NULL, "cbget MIPSOL_SOL");
    const int64_t violated = bcd_model_detect_violations(&s->model, s->values, &s->violations);
    for (int64_t i = 0; i < violated; ++i) {
        const int64_t start = s->violations.indptr[i];
        const int64_t count = s->violations.indptr[i + 1] - start;
        bcd_model_cut_row(&s->model, s->violations.nodes + start, count,
                          s->violations.rhs[i], &s->row);
        if (s->row.nnz == 0) continue;
        bcd_grb_check(GRBcblazy(cbdata, s->row.nnz, s->row.cols, s->row.coeffs,
                                GRB_GREATER_EQUAL, s->row.bound),
                      NULL, "GRBcblazy");
        s->stats.lazy_cuts_added += 1;
    }
}

static int callback(GRBmodel *model, void *cbdata, int where, void *usrdata)
{
    (void)model;
    bcd_solver *s = (bcd_solver *)usrdata;
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
    case GRB_SUBOPTIMAL:      return "SUBOPTIMAL";
    case GRB_INF_OR_UNBD:     return "INF_OR_UNBD";
    case GRB_NUMERIC:         return "NUMERIC";
    case GRB_USER_OBJ_LIMIT:  return "USER_OBJ_LIMIT";
    case GRB_WORK_LIMIT:      return "WORK_LIMIT";
    case GRB_MEM_LIMIT:       return "MEM_LIMIT";
    default:                  return NULL;
    }
}

static bool try_get_double(GRBmodel *model, const char *name, double *out)
{
    return GRBgetdblattr(model, name, out) == 0;
}

static void set_param_int(GRBmodel *model, const char *name, int value)
{
    bcd_grb_check(GRBsetintparam(GRBgetenv(model), name, value), model, name);
}

static void set_param_dbl(GRBmodel *model, const char *name, double value)
{
    bcd_grb_check(GRBsetdblparam(GRBgetenv(model), name, value), model, name);
}

void bcd_solve(GRBenv *env, const bc_instance *inst, int64_t vehicles, bool open_routes,
               bc_separator *separator, const bcd_options *options,
               bool has_optimal, double optimal_value, double optimal_tolerance,
               double setup_start, char detail_buf[BCD_DETAIL_MAX], bcd_result *out)
{
    bcd_solver s;
    memset(&s, 0, sizeof s);
    s.sep = separator;
    s.opt = *options;

    bcd_model_build(env, inst, vehicles, open_routes, &s.model);
    set_param_int(s.model.model, GRB_INT_PAR_OUTPUTFLAG, options->verbose ? 1 : 0);
    set_param_int(s.model.model, GRB_INT_PAR_THREADS, options->threads > 1 ? options->threads : 1);
    set_param_int(s.model.model, GRB_INT_PAR_SEED, options->seed);
    s.start_time = bc_seconds();
    const double setup_time = s.start_time - setup_start;

    s.values = (double *)bc_malloc((size_t)s.model.n_arc_vars * sizeof *s.values);
    s.support_u = (int64_t *)bc_malloc((size_t)s.model.n_pairs * sizeof *s.support_u);
    s.support_v = (int64_t *)bc_malloc((size_t)s.model.n_pairs * sizeof *s.support_v);
    s.support_x = (double *)bc_malloc((size_t)s.model.n_pairs * sizeof *s.support_x);
    bcd_row_init(&s.model, &s.row);
    bcd_setlist_init(&s.violations);
    key_set_init(&s.root_rows, inst->n_nodes);
    key_set_init(&s.round_rows, inst->n_nodes);
    s.key_scratch = (uint64_t *)bc_calloc((size_t)s.root_rows.words, sizeof *s.key_scratch);
    solver_reserve_candidates(&s, 256);

    bool has_root_bound = false;
    double root_bound = 0.0;
    char root_termination[32] = "round_limit";
    bool root_bound_final = true;
    int root_rounds = 0, root_cuts = 0;
    run_root_loop(&s, &has_root_bound, &root_bound, root_termination,
                  &root_bound_final, &root_rounds, &root_cuts);
    const double root_time = bc_seconds() - s.start_time;

    if (options->verbose) {
        char bound_text[64];
        if (has_root_bound) bc_format_double(bound_text, sizeof bound_text, root_bound);
        else snprintf(bound_text, sizeof bound_text, "None");
        printf("[root] done: bound=%s, rounds=%d, cuts=%d, %.2fs\n",
               bound_text, root_rounds, root_cuts, root_time);
        fflush(stdout);
    }

    bool mip_ran = false;
    if (remaining_time(&s) > 0.0) {
        bcd_model_restore_integer_types(&s.model);
        set_param_dbl(s.model.model, GRB_DBL_PAR_MIPGAP, options->mip_gap);
        set_param_int(s.model.model, GRB_INT_PAR_LAZYCONSTRAINTS, 1);
        if (options->usercut_max_nodes > 0)
            set_param_int(s.model.model, GRB_INT_PAR_PRECRUSH, 1);
        bcd_grb_check(GRBsetcallbackfunc(s.model.model, callback, &s), s.model.model,
                      "GRBsetcallbackfunc");
        if (options->verbose) {
            printf("[mip] branch and bound \xe2\x80\xa6\n");
            fflush(stdout);
        }
        const double remaining = remaining_time(&s);
        if (remaining > 0.0) {
            if (isfinite(remaining))
                set_param_dbl(s.model.model, GRB_DBL_PAR_TIMELIMIT, remaining);
            mip_ran = true;
            bcd_grb_check(GRBoptimize(s.model.model), s.model.model,
                          "GRBoptimize (branch and bound)");
        }
    }

    memset(out, 0, sizeof *out);
    out->instance_name = inst->name;
    out->n_customers = inst->n_nodes - 1;
    out->vehicles = vehicles;
    out->separator = separator->name;
    separator->describe(separator, detail_buf, BCD_DETAIL_MAX);
    out->separator_detail = detail_buf;

    if (mip_ran) {
        bcd_grb_check(GRBgetintattr(s.model.model, GRB_INT_ATTR_STATUS, &out->status),
                      s.model.model, "get Status");
        out->has_best_bound = try_get_double(s.model.model, GRB_DBL_ATTR_OBJBOUND,
                                             &out->best_bound);
        if (!try_get_double(s.model.model, GRB_DBL_ATTR_NODECOUNT, &out->node_count))
            out->node_count = 0.0;
        if (GRBgetintattr(s.model.model, GRB_INT_ATTR_SOLCOUNT, &out->sol_count) != 0)
            out->sol_count = 0;
        if (out->sol_count > 0) {
            out->has_objective = try_get_double(s.model.model, GRB_DBL_ATTR_OBJVAL,
                                                &out->objective);
            out->has_gap = try_get_double(s.model.model, GRB_DBL_ATTR_MIPGAP, &out->gap);
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
    if (out->has_objective && !isfinite(out->objective)) out->has_objective = false;
    if (out->has_best_bound && !isfinite(out->best_bound)) out->has_best_bound = false;
    if (out->has_gap && !isfinite(out->gap)) out->has_gap = false;
    if (!isfinite(out->node_count)) out->node_count = 0.0;

    const char *name = status_name(out->status);
    if (name) snprintf(out->status_name, sizeof out->status_name, "%s", name);
    else snprintf(out->status_name, sizeof out->status_name, "STATUS_%d", out->status);
    out->runtime = bc_seconds() - s.start_time;
    out->setup_time = setup_time;
    out->has_root_bound = has_root_bound && isfinite(root_bound);
    out->root_bound = root_bound;
    snprintf(out->root_termination, sizeof out->root_termination, "%s", root_termination);
    out->root_bound_final = root_bound_final;
    out->root_rounds = root_rounds;
    out->root_cuts = root_cuts;
    out->root_time = root_time;
    out->callback = s.stats;
    out->has_optimal = has_optimal;
    out->optimal_value = optimal_value;
    out->optimal_tolerance = optimal_tolerance;
    out->has_matches_optimum = has_optimal && out->has_objective
                               && out->status == GRB_OPTIMAL;
    out->matches_optimum = out->has_matches_optimum
                           && fabs(out->objective - optimal_value) <= optimal_tolerance;

    free(s.values);
    free(s.support_u);
    free(s.support_v);
    free(s.support_x);
    free(s.candidates);
    free(s.accepted);
    free(s.batch_begins);
    free(s.batch_cols);
    free(s.batch_vals);
    free(s.batch_rhs);
    free(s.batch_senses);
    free(s.key_scratch);
    key_set_free(&s.root_rows);
    key_set_free(&s.round_rows);
    bcd_row_free(&s.row);
    bcd_setlist_free(&s.violations);
    bcd_model_free(&s.model);
}
