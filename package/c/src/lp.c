#include "lp.h"

#ifndef HAVE_GUROBI
#error "src/lp.c requires the Gurobi C API: set GUROBI_HOME to a Gurobi installation with include/gurobi_c.h and lib/libgurobi*.so and a valid license"
#endif

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gurobi_c.h"

/* ===================================================================== */
/* Handle                                                                */
/* ===================================================================== */

struct sag_lp {
    GRBenv   *env;
    GRBmodel *model;

    int64_t   n_nodes;
    int64_t   n_customers;
    int64_t   n_vars;
    int64_t   base_rows;
    int64_t   cut_rows;

    int32_t  *var_index;
    int64_t  *edge_u;
    int64_t  *edge_v;

    double   *solution;
    bool      have_solution;
    bool      optimized;
    int       status;
    double    objective;

    int      *s_ind;    size_t s_ind_cap;
    double   *s_val;    size_t s_val_cap;
    int      *s_beg;    size_t s_beg_cap;
    double   *s_rhs;    size_t s_rhs_cap;
    char     *s_sense;  size_t s_sense_cap;
    int32_t  *s_comp;   size_t s_comp_cap;
    uint8_t  *s_member;

    int64_t   vehicles;

};

/* ===================================================================== */
/* Small helpers                                                         */
/* ===================================================================== */

static void lp_check(const sag_lp *lp, int rc, const char *what)
{
    if (rc == 0) return;
    GRBenv *err_env = NULL;
    if (lp) {
        if (lp->model) err_env = GRBgetenv(lp->model);
        if (!err_env) err_env = lp->env;
    }
    const char *msg = err_env ? GRBgeterrormsg(err_env) : NULL;
    if (msg && !*msg) msg = NULL;
    sag_die("Gurobi C API error %d in %s%s%s", rc, what,
            msg ? ": " : "", msg ? msg : "");
}

static void *lp_xmalloc(size_t bytes, const char *what)
{
    void *p = malloc(bytes ? bytes : 1);
    if (!p) sag_die("lp: out of memory allocating %s (%zu bytes)", what, bytes);
    return p;
}

static void *lp_reserve(void **buf, size_t *cap, size_t need, size_t elem, const char *what)
{
    if (*cap >= need) return *buf;
    size_t next = *cap ? *cap : 64;
    while (next < need) {
        if (next > (size_t)-1 / 2) sag_die("lp: scratch %s overflow (need %zu)", what, need);
        next *= 2;
    }
    void *p = realloc(*buf, sag_size_mul(next, elem));
    if (!p) sag_die("lp: out of memory growing scratch %s to %zu elems", what, next);
    *buf = p;
    *cap = next;
    return p;
}

static int lp_int(int64_t v, const char *what)
{
    if (v < INT_MIN || v > INT_MAX)
        sag_die("lp: %s = %lld does not fit in the Gurobi C API int", what, (long long)v);
    return (int)v;
}

/* ===================================================================== */
/* Model construction (same build order as the reference implementation) */
/* ===================================================================== */

static void lp_build(sag_lp *lp, const sag_instance *inst)
{
    const int64_t n = lp->n_nodes;
    const int64_t n_vars = lp->n_vars;

    double *obj = lp_xmalloc(sag_size_mul(sizeof(double), (size_t)n_vars), "obj");
    double *lb  = lp_xmalloc(sag_size_mul(sizeof(double), (size_t)n_vars), "lb");
    double *ub  = lp_xmalloc(sag_size_mul(sizeof(double), (size_t)n_vars), "ub");
    char   *vt  = lp_xmalloc((size_t)n_vars, "vtype");

    int64_t k = 0;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i + 1; j < n; ++j) {
            lp->edge_u[k] = i;
            lp->edge_v[k] = j;
            obj[k] = inst->weight[i * n + j];
            lb[k]  = 0.0;
            ub[k]  = (i == SAG_DEPOT || j == SAG_DEPOT) ? 2.0 : 1.0;
            vt[k]  = GRB_CONTINUOUS;
            lp->var_index[i * n + j] = (int32_t)k;
            lp->var_index[j * n + i] = (int32_t)k;
            ++k;
        }
    }
    if (k != n_vars) sag_die("lp: enumerated %lld vars, expected %lld", (long long)k,
                             (long long)n_vars);

    lp_check(lp, GRBaddvars(lp->model, lp_int(n_vars, "n_vars"), 0, NULL, NULL, NULL,
                            obj, lb, ub, vt, NULL), "GRBaddvars");
    lp_check(lp, GRBsetintattr(lp->model, GRB_INT_ATTR_MODELSENSE, GRB_MINIMIZE),
             "set ModelSense");
    lp_check(lp, GRBupdatemodel(lp->model), "GRBupdatemodel(vars)");

    free(obj); free(lb); free(ub); free(vt);

    int    *cind = lp_reserve((void **)&lp->s_ind, &lp->s_ind_cap, (size_t)(n - 1), sizeof(int), "cind");
    double *cval = lp_reserve((void **)&lp->s_val, &lp->s_val_cap, (size_t)(n - 1), sizeof(double), "cval");
    for (int64_t i = 0; i < n - 1; ++i) cval[i] = 1.0;

    char name[32];
    for (int64_t node = 1; node < n; ++node) {
        int64_t m = 0;
        for (int64_t other = 0; other < n; ++other) {
            if (other == node) continue;
            int32_t col = lp->var_index[node * n + other];
            if (col < 0) sag_die("lp: missing var for degree pair (%lld,%lld)",
                                 (long long)node, (long long)other);
            cind[m++] = (int)col;
        }
        snprintf(name, sizeof name, "deg_%lld", (long long)(node + 1));
        lp_check(lp, GRBaddconstr(lp->model, lp_int(m, "degree nnz"), cind, cval,
                                  GRB_EQUAL, 2.0, name), "GRBaddconstr(deg)");
    }
    {
        int64_t m = 0;
        for (int64_t other = 1; other < n; ++other)
            cind[m++] = (int)lp->var_index[SAG_DEPOT * n + other];
        lp_check(lp, GRBaddconstr(lp->model, lp_int(m, "depot degree nnz"), cind, cval,
                                  GRB_EQUAL, 2.0 * (double)inst->vehicles, "deg_depot"),
                 "GRBaddconstr(deg_depot)");
    }
    lp_check(lp, GRBupdatemodel(lp->model), "GRBupdatemodel(degree)");

    lp->base_rows = n;
}

/* ===================================================================== */
/* Version (part of the parity contract; see lp.h)                       */
/* ===================================================================== */

void sag_lp_version_string(char *out, size_t cap)
{
    if (!out || cap == 0) sag_die("sag_lp_version_string: NULL/0 buffer");
    int major = 0, minor = 0, tech = 0;
    GRBversion(&major, &minor, &tech);
    snprintf(out, cap, "%d.%d.%d", major, minor, tech);
}

/* ===================================================================== */
/* Lifecycle                                                             */
/* ===================================================================== */

sag_lp *sag_lp_create(const sag_instance *inst, const sag_lp_config *cfg)
{
    if (!inst || !cfg) sag_die("sag_lp_create: NULL argument");
    if (!inst->weight) sag_die("sag_lp_create: instance.weight is NULL");
    const int64_t n = inst->n_nodes;
    if (n < 3) sag_die("sag_lp_create: n_nodes = %lld (need >= 3)", (long long)n);
    if (inst->depot != SAG_DEPOT)
        sag_die("sag_lp_create: depot must be internal node 0, got %lld",
                (long long)inst->depot);
    if (inst->vehicles <= 0)
        sag_die("sag_lp_create: vehicles = %lld must be positive", (long long)inst->vehicles);

    sag_lp *lp = lp_xmalloc(sizeof *lp, "sag_lp");
    memset(lp, 0, sizeof *lp);
    lp->n_nodes = n;
    lp->n_customers = n - 1;
    lp->n_vars = n * (n - 1) / 2;
    lp->status = -1;
    lp->objective = 0.0;
    lp->vehicles = inst->vehicles;

    lp->var_index = lp_xmalloc(sag_size_mul(sizeof(int32_t), (size_t)(n * n)), "var_index");
    for (int64_t i = 0; i < n * n; ++i) lp->var_index[i] = -1;
    lp->edge_u   = lp_xmalloc(sag_size_mul(sizeof(int64_t), (size_t)lp->n_vars), "edge_u");
    lp->edge_v   = lp_xmalloc(sag_size_mul(sizeof(int64_t), (size_t)lp->n_vars), "edge_v");
    lp->solution = lp_xmalloc(sag_size_mul(sizeof(double), (size_t)lp->n_vars), "solution");
    lp->s_member = lp_xmalloc((size_t)n, "member flags");
    memset(lp->s_member, 0, (size_t)n);

    lp_check(lp, GRBloadenv(&lp->env, NULL), "GRBloadenv");
    lp_check(lp, GRBnewmodel(lp->env, &lp->model, "cvrp_root_relaxation", 0,
                             NULL, NULL, NULL, NULL, NULL), "GRBnewmodel");

    GRBenv *menv = GRBgetenv(lp->model);
    if (!menv) sag_die("lp: GRBgetenv returned NULL");
    lp_check(lp, GRBsetintparam(menv, "OutputFlag", 0), "set OutputFlag");
    lp_check(lp, GRBsetintparam(menv, "Method", lp_int(cfg->gurobi_method, "gurobi_method")),
             "set Method");
    if (cfg->gurobi_threads > 0)
        lp_check(lp, GRBsetintparam(menv, "Threads", lp_int(cfg->gurobi_threads, "gurobi_threads")),
                 "set Threads");

    lp_build(lp, inst);
    return lp;
}

void sag_lp_destroy(sag_lp *lp)
{
    if (!lp) return;
    if (lp->model) GRBfreemodel(lp->model);
    if (lp->env) GRBfreeenv(lp->env);
    free(lp->var_index);
    free(lp->edge_u);
    free(lp->edge_v);
    free(lp->solution);
    free(lp->s_ind);
    free(lp->s_val);
    free(lp->s_beg);
    free(lp->s_rhs);
    free(lp->s_sense);
    free(lp->s_comp);
    free(lp->s_member);
    free(lp);
}

/* ===================================================================== */
/* Solve / read back                                                     */
/* ===================================================================== */

void sag_lp_optimize(sag_lp *lp)
{
    lp_check(lp, GRBoptimize(lp->model), "GRBoptimize");
    int status = 0;
    lp_check(lp, GRBgetintattr(lp->model, GRB_INT_ATTR_STATUS, &status), "get Status");
    lp->status = status;
    if (status != GRB_OPTIMAL && status != GRB_SUBOPTIMAL)
        sag_die("Gurobi terminated with status %d", status);
    lp_check(lp, GRBgetdblattr(lp->model, GRB_DBL_ATTR_OBJVAL, &lp->objective), "get ObjVal");
    lp->have_solution = false;
    lp->optimized = true;
}

double sag_lp_objective(const sag_lp *lp)
{
    if (!lp->optimized) sag_die("sag_lp_objective: call sag_lp_optimize first");
    return lp->objective;
}

int sag_lp_status(const sag_lp *lp)
{
    if (!lp->optimized) sag_die("sag_lp_status: call sag_lp_optimize first");
    return lp->status;
}

int64_t sag_lp_num_constrs(const sag_lp *lp)
{
    int value = 0;
    lp_check(lp, GRBgetintattr(lp->model, GRB_INT_ATTR_NUMCONSTRS, &value), "get NumConstrs");
    return (int64_t)value;
}

int64_t sag_lp_base_rows(const sag_lp *lp) { return lp->base_rows; }
int64_t sag_lp_num_vars(const sag_lp *lp)  { return lp->n_vars; }
const int64_t *sag_lp_edge_u(const sag_lp *lp) { return lp->edge_u; }
const int64_t *sag_lp_edge_v(const sag_lp *lp) { return lp->edge_v; }

const double *sag_lp_solution(sag_lp *lp)
{
    if (!lp->optimized) sag_die("sag_lp_solution: call sag_lp_optimize first");
    if (!lp->have_solution) {
        lp_check(lp, GRBgetdblattrarray(lp->model, GRB_DBL_ATTR_X, 0,
                                        lp_int(lp->n_vars, "n_vars"), lp->solution),
                 "get X");
        lp->have_solution = true;
    }
    return lp->solution;
}

int64_t sag_lp_support_edges(sag_lp *lp, double threshold,
                             int64_t *eu, int64_t *ev, double *ex)
{
    if (!eu || !ev || !ex) sag_die("sag_lp_support_edges: NULL output array");
    const double *x = sag_lp_solution(lp);
    int64_t m = 0;
    for (int64_t k = 0; k < lp->n_vars; ++k) {
        if (x[k] > threshold) {
            eu[m] = lp->edge_u[k];
            ev[m] = lp->edge_v[k];
            ex[m] = x[k];
            ++m;
        }
    }
    return m;
}

/* ===================================================================== */
/* Cut row construction + batch add                                      */
/* ===================================================================== */

/* ===========================================================================
 * RCC row form
 * --------------------------------------------------------------------------- */

static int64_t lp_row_nnz(int64_t s, int64_t n_cust)
{
    const int64_t c = n_cust - s;
    if (2 * s <= n_cust || s == n_cust) return s * (s - 1) / 2;
    return c * (c - 1) / 2 + n_cust;
}

int64_t sag_lp_add_cuts(sag_lp *lp, const sag_materialized_cuts *cuts)
{
    if (!cuts || cuts->rows <= 0) return 0;
    if (!cuts->indptr || !cuts->indices || !cuts->rhs)
        sag_die("sag_lp_add_cuts: materialized cuts missing indptr/indices/rhs");

    const int64_t n = lp->n_nodes;
    const int64_t n_cust = lp->n_customers;
    const int64_t nr = cuts->rows;

    /* ---- 1) predict the total nnz and reserve once (no malloc in steady state) ---- */
    int64_t total_nnz = 0;
    for (int64_t r = 0; r < nr; ++r) {
        const int64_t s = cuts->indptr[r + 1] - cuts->indptr[r];
        if (s <= 0) sag_die("sag_lp_add_cuts: row %lld is empty (never emitted upstream)",
                            (long long)r);
        if (s > n_cust) sag_die("sag_lp_add_cuts: row %lld has %lld nodes > %lld customers",
                                (long long)r, (long long)s, (long long)n_cust);
        total_nnz += lp_row_nnz(s, n_cust);
    }

    const size_t nnz_cap = total_nnz > 0 ? (size_t)total_nnz : 1u;
    int    *cind  = lp_reserve((void **)&lp->s_ind,   &lp->s_ind_cap,   nnz_cap,           sizeof(int),     "cind");
    double *cval  = lp_reserve((void **)&lp->s_val,   &lp->s_val_cap,   nnz_cap,           sizeof(double),  "cval");
    int    *cbeg  = lp_reserve((void **)&lp->s_beg,   &lp->s_beg_cap,   (size_t)nr,        sizeof(int),     "cbeg");
    double *crhs  = lp_reserve((void **)&lp->s_rhs,   &lp->s_rhs_cap,   (size_t)nr,        sizeof(double),  "rhs");
    char   *sense = lp_reserve((void **)&lp->s_sense, &lp->s_sense_cap, (size_t)nr,        sizeof(char),    "sense");
    int32_t *comp = lp_reserve((void **)&lp->s_comp,  &lp->s_comp_cap,  (size_t)n_cust,    sizeof(int32_t), "complement");
    uint8_t *member = lp->s_member;

    memset(sense, GRB_LESS_EQUAL, (size_t)nr);

    /* ---- 2) build row by row ---- */
    int64_t pos = 0;
    for (int64_t r = 0; r < nr; ++r) {
        const int64_t lo = cuts->indptr[r], hi = cuts->indptr[r + 1];
        const int64_t s = hi - lo;
        cbeg[r] = lp_int(pos, "cbeg");
        const int64_t row_start = pos;

        int32_t prev = 0;
        for (int64_t t = lo; t < hi; ++t) {
            const int32_t ext = cuts->indices[t];
            if (ext < 2 || (int64_t)ext > n)
                sag_die("sag_lp_add_cuts: row %lld external id %d out of range [2,%lld]",
                        (long long)r, ext, (long long)n);
            if (t > lo && ext <= prev)
                sag_die("sag_lp_add_cuts: row %lld indices are not strictly ascending "
                        "(%d after %d)", (long long)r, ext, prev);
            prev = ext;
            member[ext - 1] = 1;
        }

        bool use_complement = (2 * s > n_cust);
        int64_t n_comp = 0;
        if (use_complement) {
            for (int64_t v = 1; v < n; ++v)
                if (!member[v]) comp[n_comp++] = (int32_t)v;
            if (n_comp != n_cust - s)
                sag_die("sag_lp_add_cuts: complement size %lld != %lld",
                        (long long)n_comp, (long long)(n_cust - s));
            if (n_comp == 0) use_complement = false;
        }

        double bound;
        if (!use_complement) {
            for (int64_t a = 0; a < s; ++a) {
                const int64_t ia = cuts->indices[lo + a] - 1;
                for (int64_t b = a + 1; b < s; ++b) {
                    const int64_t ib = cuts->indices[lo + b] - 1;
                    const int32_t col = lp->var_index[ia * n + ib];
                    if (col < 0) sag_die("lp: missing var for cut pair (%lld,%lld)",
                                         (long long)ia, (long long)ib);
                    cind[pos] = (int)col;
                    cval[pos] = 1.0;
                    ++pos;
                }
            }
            bound = (double)s - cuts->rhs[r];
        } else {
            for (int64_t a = 0; a < n_comp; ++a) {
                const int64_t ia = comp[a];
                for (int64_t b = a + 1; b < n_comp; ++b) {
                    const int64_t ib = comp[b];
                    const int32_t col = lp->var_index[ia * n + ib];
                    if (col < 0) sag_die("lp: missing var for cut pair (%lld,%lld)",
                                         (long long)ia, (long long)ib);
                    cind[pos] = (int)col;
                    cval[pos] = 1.0;
                    ++pos;
                }
            }
            for (int64_t a = 0; a < n_comp; ++a) {
                cind[pos] = (int)lp->var_index[SAG_DEPOT * n + comp[a]];
                cval[pos] = 0.5;
                ++pos;
            }
            for (int64_t a = 0; a < s; ++a) {
                const int64_t ia = cuts->indices[lo + a] - 1;
                cind[pos] = (int)lp->var_index[SAG_DEPOT * n + ia];
                cval[pos] = -0.5;
                ++pos;
            }
            bound = (double)n_comp - cuts->rhs[r];
        }
        crhs[r] = bound;

        for (int64_t t = lo; t < hi; ++t) member[cuts->indices[t] - 1] = 0;

        const int64_t got = pos - row_start;
        const int64_t want = lp_row_nnz(s, n_cust);
        if (got != want)
            sag_die("sag_lp_add_cuts: row %lld emitted %lld nnz, lp_row_nnz "
                    "predicted %lld (|S|=%lld, n_cust=%lld)", (long long)r,
                    (long long)got, (long long)want, (long long)s, (long long)n_cust);
    }
    if (pos != total_nnz)
        sag_die("sag_lp_add_cuts: emitted %lld nnz, reserved %lld", (long long)pos,
                (long long)total_nnz);

    /* ---- 3) one GRBaddconstrs call; batch add is always on, constrname = NULL ---- */
    lp_check(lp, GRBaddconstrs(lp->model, lp_int(nr, "numconstrs"),
                               lp_int(total_nnz, "numnz"), cbeg, cind, cval,
                               sense, crhs, NULL), "GRBaddconstrs");
    lp->cut_rows += nr;
    return total_nnz;
}

