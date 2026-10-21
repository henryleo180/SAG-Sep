#ifndef SAGSEP_SEP_API_H
#define SAGSEP_SEP_API_H

/*
 * sep_api.h -- the separate-once public API of SAG-Sep.
 *
 * Use this when **the caller owns the LP**: a branch-and-cut written against
 * the Gurobi C API, RouteOpt's C++ branch-price-and-cut, or any other solver
 * that wants rounded capacity cuts from its own cut callback. Hand it the
 * current fractional two-index support x*, get back the cuts one round of the
 * SAG-Sep pipeline proposes, tell it which of them you accepted, repeat.
 *
 * The other entry point, `sag_run_instance` (src/loop.h), owns the
 * whole root LP loop itself and is the right one for standalone root
 * relaxations. Both drive the identical per-round engine (src/round.c), so this
 * API is the production pipeline rather than a re-implementation of it.
 *
 * The canonical caller is `examples/bc/common/c/separator_sagsep.c`, shared by
 * the Gurobi C branch-and-cut examples under `examples/bc/`, whose rounded
 * capacity cuts come either from CVRPSEP or from this API; it is the worked
 * example to read alongside this header. `examples/bpc` wires the same
 * API into RouteOpt's C++ branch-price-and-cut.
 *
 * This header is deliberately free of internal SAG types: fixed-width scalars,
 * one opaque handle and borrowed output views only, so it can be included from
 * C++ (the internal headers cannot).
 *
 * ---------------------------------------------------------------------------
 * Node numbering
 * ---------------------------------------------------------------------------
 * Throughout this API the depot is node 0 and the customers are 1..n_nodes-1,
 * both on the way in (support arcs, demand) and on the way out (cut subsets).
 * This is the engine's own internal convention and also RouteOpt's.
 *
 * ---------------------------------------------------------------------------
 * The cut the API returns
 * ---------------------------------------------------------------------------
 * A row is a customer subset S (`customers[indptr[r] .. indptr[r+1])`, strictly
 * ascending, never containing the depot) together with
 *
 *     rhs[r] = ceil(d(S) / Q)          -- the rounded capacity bound, k(S)
 *
 * which is the canonical form the pipeline materialises. Both textbook
 * spellings of the same rounded capacity cut follow from it:
 *
 *     x(delta(S)) >= 2 * rhs[r]        (crossing-edge form)
 *     x(E(S))     <= |S| - rhs[r]      (inner-edge form)
 *
 * The two are equivalent under the customer degree equalities x(delta(i)) = 2.
 * The reference root loop adds the **inner-edge form**, and for a large S it
 * writes the algebraically identical complement row instead, purely to keep the
 * matrix sparse (see `sag_lp_add_cuts` in src/lp.c). The
 * subset and rhs are exported here so the caller can pick whichever spelling
 * its model wants; no coefficient vector is exported, because the right one
 * depends on the caller's variable layout.
 *
 * `boundary[r]` is x*(delta(S)) and `violation[r]` is `2*rhs[r] - boundary[r]`,
 * both from the pipeline's authoritative scorer, in its exact float64
 * summation order. Every returned row has `violation[r] > 0` (that gate is the
 * spine's selector), so the rows are always violated by the x* passed in.
 *
 * ---------------------------------------------------------------------------
 * The separate / commit protocol
 * ---------------------------------------------------------------------------
 * `sag_separator_separate` returns **proposals**, and the caller decides which
 * of them actually enter its model -- it may drop rows for reasons the
 * separator cannot see (already present in its pool, rejected by its own
 * numerical filters, over a per-round cut budget). So each separate call must
 * be followed by exactly one `sag_separator_commit` naming the accepted rows,
 * `count = 0` included. Two calls in a row without a commit in between is a
 * fatal error rather than a silent state corruption.
 *
 * The commit feeds the **blocked set**: the keys of the committed subsets, i.e.
 * the cuts the caller's model now holds. The pipeline skips a candidate whose
 * key is blocked, which is what stops it from re-proposing cuts that are
 * already in the model. In the reference loop the blocked set is exactly "the
 * cuts currently in the LP"; the commit protocol is how an external caller
 * reproduces that, since the separator cannot observe the caller's model.
 *
 * Commits alone only track a set that grows. That is enough for a caller whose
 * pool is monotone, and it is not enough for one whose is not: a
 * branch-price-and-cut purges rows whose dual collapses, and moving to another
 * node changes which cuts are active. Reconstructing "what the model holds" by
 * subtraction is not possible from this side, and dropping the whole blocked set
 * when one cut disappears also un-blocks every cut that is still there -- so the
 * pipeline spends the next round re-proposing cuts the model already has.
 * `sag_separator_set_blocked` is the answer: a caller that can enumerate its
 * current cuts states the set outright, and the commit after the next separate
 * then adds that round's accepted rows on top of it.
 *
 * The **old-cut bank** is a different, larger container: it is written from the
 * whole proposed batch (not just the committed rows), immediately after the
 * pipeline materialises it, and it seeds the tts family's old_repair/adddrop
 * components on later rounds. That is the reference timing, and it is not
 * affected by what the caller commits.
 *
 * ---------------------------------------------------------------------------
 * What this API does NOT do
 * ---------------------------------------------------------------------------
 * It never touches an LP, never solves anything and never allocates a solver
 * environment; it does not link Gurobi on the caller's behalf. It also does not
 * apply the reference loop's own LP-admission gate. That gate recomputes each
 * proposal's violation from the loop's dense x* matrix and keeps the rows above
 * `min_violation` (reported in `sag_separator_stats::min_violation`). It cannot
 * be reproduced here bit for bit, because that dense matrix is accumulated over
 * **all** LP columns whereas this API only ever sees the support arcs above
 * 1e-8. If you want the loop's admission policy, apply
 * `violation[r] > stats.min_violation` to the returned rows: the two violation
 * chains are the same quantity computed in two summation orders and agree to
 * float64 rounding.
 *
 * Nothing here ever removes a cut from the caller's model: the pipeline only
 * ever proposes. Cut ageing and eviction are the caller's business.
 *
 * ---------------------------------------------------------------------------
 * Failure mode
 * ---------------------------------------------------------------------------
 * Every misuse is fail-loud: the engine prints a diagnostic to stderr and
 * exits. There are no error return codes, and nothing degrades silently to a
 * different configuration. This matches the rest of the engine; a separator
 * that quietly changes recipe would produce a valid-looking but wrong bound.
 *
 * ---------------------------------------------------------------------------
 * Thread safety
 * ---------------------------------------------------------------------------
 * A handle is not thread-safe; it holds per-round arenas and cross-round state.
 * Serialise calls on one handle, or create one handle per thread.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque per-instance separator state. */
typedef struct sag_separator sag_separator;

/*
 * Everything the separator needs for one instance.
 *
 * The forward backend is `model_path` + `provider`: the exported ONNX policy
 * graph, and the execution provider it runs on. `provider` is "cpu"
 * (bit-for-bit reproducible), "cuda" or "cuda:<id>" (much faster, but the GPU
 * reduction order is not fixed, so runs are not bit-for-bit reproducible).
 * Anything else is fatal.
 *
 * `instance_name` must be the instance's exact name **verbatim**, the same
 * string the reference implementation used (the VRP NAME header, ".vrp" suffix
 * included if the file carries one). It is not cosmetic: it is hashed into the
 * RNG payload character by character, so normalising it silently changes which
 * cuts come out.
 *
 * `preset` selects the recipe: one of the frozen per-size preset names ("n50",
 * "n75", "n100", "n200", "n300", "n400", "n500", "n750", "n1000"), or a path to
 * a TOML config file (recognised by a ".toml" suffix). Pick the preset by the
 * size of the instances the separator will see, not by the size of the
 * branch-and-bound node.
 *
 * `demand` is (n_nodes,) with `demand[0] == 0`; it is copied into the handle,
 * so the caller's array need not outlive the call. Every value and their sum
 * must be below 2^24 so the float32 chains built from them stay exact.
 *
 * `cross_depot` selects the two-hop shortcut rule; see
 * `sag_separator_cross_depot` just below.
 *
 * The arena sizes are optional tuning; 0 selects a built-in default sized for
 * the largest frozen preset. The per-round arena dominates and grows with
 * n_nodes.
 */

/*
 * ---------------------------------------------------------------------------
 * Two-hop shortcut arcs and the depot
 * ---------------------------------------------------------------------------
 * The graph the policy reads is not the support alone. On top of every support
 * edge it carries "partial connect" shortcut arcs: for a pair (s, c) that the
 * support does not join directly, an arc s -> c is added whenever some node b
 * is a support neighbour of both, weighted max over b of min(x*(s,b), x*(b,c)).
 *
 * `cross_depot` decides whether the depot may play those roles:
 *
 *   ENABLED   the depot is an ordinary node: it may be the intermediate b, and
 *             it may be the far endpoint c. This is the released recipe.
 *   DISABLED  for every start s other than the depot, a shortcut is dropped
 *             when it would run through the depot (b == 0) and when it would
 *             end at the depot (c == 0). Shortcuts that start at the depot
 *             (s == 0) are untouched, and so is everything else about the
 *             graph: the emission order of the arcs that survive, their values,
 *             the depot completion pass that follows, all identical.
 *
 *   DEFAULT   leave the choice to the recipe: the preset, or the
 *             `partial_connect_cross_depot` key of the `[features]` table of a
 *             TOML config. Every preset carries `true`, so DEFAULT and ENABLED
 *             coincide unless a TOML config says otherwise.
 *
 * This is a property of the **trained policy**, not a tuning knob: a model has
 * only ever seen graphs built one way, and scoring the other kind is scoring an
 * input distribution it was never trained on. Run a model with the setting its
 * training data was generated with. The released general-purpose CVRP model
 * wants ENABLED; the branch-price-and-cut fine-tuned model wants DISABLED.
 */
typedef enum {
    SAG_CROSS_DEPOT_DEFAULT  = 0,  /* take the preset's / the TOML config's value */
    SAG_CROSS_DEPOT_ENABLED  = 1,  /* the depot takes part in the shortcuts */
    SAG_CROSS_DEPOT_DISABLED = 2   /* drop the depot-mediated shortcuts */
} sag_separator_cross_depot;

typedef struct {
    const char *model_path;         /* the exported ONNX policy graph; required */
    const char *provider;           /* "cpu" | "cuda" | "cuda:<id>"; required with model_path */
    const char *instance_name;      /* verbatim; enters the RNG payload */
    const char *preset;             /* frozen preset name, or a *.toml path */
    sag_separator_cross_depot cross_depot;  /* 0 = whatever the recipe says */
    int64_t     n_nodes;            /* depot + customers */
    int64_t     capacity;           /* Q */
    const int64_t *demand;          /* (n_nodes,), demand[0] == 0; copied */
    size_t      iteration_arena_bytes;   /* 0 = default */
    size_t      persistent_arena_bytes;  /* 0 = default */
} sag_separator_options;

/*
 * The cuts of one round. All pointers are **borrowed** views into the
 * separator's per-round arena: they stay valid until the next
 * `sag_separator_separate` or `sag_separator_destroy`, and a `commit` in
 * between does not invalidate them. Copy anything you need to keep.
 *
 * None of the pointers is ever NULL, `rows == 0` included: `indptr` then holds
 * the single entry 0, so the usual `for (r = 0; r < rows; ++r)` loop needs no
 * empty-batch special case.
 */
typedef struct {
    int64_t rows;
    int64_t nnz;
    const int64_t *indptr;     /* (rows+1,) */
    const int32_t *customers;  /* (nnz,) ascending within a row; depot=0 numbering, never 0 */
    const double  *rhs;        /* (rows,) ceil(d(S)/Q) */
    const double  *boundary;   /* (rows,) x*(delta(S)) */
    const double  *violation;  /* (rows,) 2*rhs - boundary, always > 0 */
} sag_separator_cuts;

/*
 * Per-call counters and timings. Diagnostics only: nothing here feeds back into
 * a decision, and none of it appears in any JSON the CLI writes.
 */
typedef struct {
    int64_t iteration;               /* 1-based round counter of this handle */
    int64_t replicas;                /* K_replica = ceil(d(V)/Q) = probs rows */
    int64_t support_edges;           /* support arcs kept by the 1e-6 threshold chain */
    int64_t model_edges;             /* arcs in the policy input graph */
    int64_t sweep_candidates;        /* raw rows out of each family */
    int64_t onion_candidates;
    int64_t tts_candidates;
    int64_t raw_candidates;          /* after assembly */
    int64_t unique_candidates;       /* after exact-set dedup */
    int64_t blocked_candidates;      /* dropped because their key was committed earlier */
    int64_t nonviolated_candidates;  /* dropped by the selector's violation > 0 gate */
    int64_t materialized_cuts;       /* == cuts.rows */
    int64_t committed_cuts;          /* rows accepted by the previous commit */
    int64_t blocked_keys;            /* size of the blocked set entering this round */
    int64_t old_bank_rows;           /* size of the old-cut bank after the write-back */
    size_t  arena_used;
    size_t  arena_high_water;
    double  min_violation;           /* the reference loop's LP-admission threshold */
    double  load_seconds;            /* forward-backend open cost, reported on round 1 only */
    double  support_seconds;
    double  graph_seconds;
    double  forward_seconds;
    double  family_seconds;
    double  spine_seconds;
    double  total_seconds;
} sag_separator_stats;

/*
 * Create a separator for one instance. Opens the forward backend (an ONNX
 * session can take a moment; do it once, outside the branch-and-bound). Never
 * returns NULL -- any problem is fatal.
 */
sag_separator *sag_separator_create(const sag_separator_options *options);

/* Release everything, including the forward backend. NULL is a no-op. */
void sag_separator_destroy(sag_separator *separator);

/*
 * Run one separation round on the support of the current fractional solution.
 *
 * `u` / `v` / `x` are the support arcs of x*: `m` entries, `u[e] < v[e]`, both
 * in 0..n_nodes-1, values finite and **strictly greater than 1e-8**, sorted
 * lexicographically by (u, v) with no repeats. That is exactly the format the
 * reference loop extracts from its LP, and exactly the argument list of the
 * Python reference `LearnedSeparator.propose`. The arcs must be the raw ones:
 * do not pre-filter them at 1e-6, the graph builder needs the small ones for
 * its fallback rule. An empty support is fatal, since no policy input graph can
 * be built from it.
 *
 * `cuts` is filled with borrowed views (see `sag_separator_cuts`). `stats` may
 * be NULL.
 *
 * Returns the number of cut rows (== cuts->rows), which may be 0.
 *
 * Must be followed by exactly one `sag_separator_commit`.
 */
int64_t sag_separator_separate(sag_separator *separator,
                               int64_t m, const int64_t *u, const int64_t *v,
                               const double *x,
                               sag_separator_cuts *cuts,
                               sag_separator_stats *stats);

/*
 * Declare which rows of the immediately preceding `sag_separator_separate`
 * result actually entered the caller's model. `accepted_rows` holds `count`
 * row indices, strictly ascending; `count = 0` (with `accepted_rows` NULL) is
 * the legitimate "I took none of them" answer and still has to be made.
 *
 * The committed keys become blocked, so those subsets are not proposed again.
 */
void sag_separator_commit(sag_separator *separator,
                          const int64_t *accepted_rows, int64_t count);

/*
 * Forget every blocked key, so previously committed subsets become proposable
 * again. Use it when the caller's model discards its cuts -- for instance on
 * entering a branch-and-bound node that does not inherit the parent's pool.
 *
 * It does **not** touch the old-cut bank (that is a memory of what the
 * pipeline has proposed, not of what the model holds) and it does not reset the
 * round counter. Calling it while a separation is awaiting its commit is fatal.
 */
void sag_separator_reset_blocked(sag_separator *separator);

/*
 * Replace the blocked set wholesale with the given customer subsets. Use it when
 * the caller can enumerate the cuts its model currently holds, which is the only
 * way to stay in step with a pool that shrinks as well as grows (see the protocol
 * section above). The commit that follows the next separate still adds that
 * round's accepted rows on top.
 *
 * CSR in the depot=0 numbering, i.e. the same numbering
 * `sag_separator_cuts::customers` comes back in, so a row can be handed straight
 * back:
 *   `rows`      number of subsets;
 *   `indptr`    (rows+1,) with indptr[0] == 0, non-decreasing... and strictly so:
 *               an empty subset is not a cut and is rejected;
 *   `customers` (indptr[rows],) each in 1..n_nodes-1 -- never the depot -- and
 *               strictly ascending within a row.
 * Duplicate subsets are accepted and collapse, the container being a set.
 * `rows = 0` (with `indptr` and `customers` NULL, or `indptr` = {0}) is exactly
 * `sag_separator_reset_blocked`.
 *
 * The keys are built by the engine's own key function, the one `commit` uses, so
 * a subset blocked here is byte-identical to the same subset blocked by a commit.
 *
 * The whole argument is validated before anything is replaced, and, like the rest
 * of this API, a malformed one is fatal. Calling it while a separation is
 * awaiting its commit is fatal too: it would throw away the record of what that
 * round produced.
 */
void sag_separator_set_blocked(sag_separator *separator, int64_t rows,
                               const int64_t *indptr, const int32_t *customers);

/* One-line description of the configuration (model, provider, preset,
 * cross_depot, replicas), for a startup banner. Lifetime = the handle. */
const char *sag_separator_description(const sag_separator *separator);

#ifdef __cplusplus
}
#endif

#endif /* SAGSEP_SEP_API_H */
