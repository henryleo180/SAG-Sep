#define _POSIX_C_SOURCE 200809L

#include "loop.h"

#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "bitset.h"
#include "families/tts/core.h"
#include "features.h"
#include "lp.h"
#include "round.h"
#include "spine.h"

/* ===================================================================== */
/* 0. Timing                                                             */
/* ===================================================================== */

static double now_sec(void)
{
    return sag_round_now_seconds();
}

/* ===================================================================== */
/* 1. Cross-round state (blocked-key set + old-cut bank)                 */
/* ===================================================================== */

/* ===================================================================== */
/* 2. violation_from_dense                                               */
/* ===================================================================== */

typedef struct {
    const double *dense;
    const double *degree;
    int64_t       n_nodes;
    double       *block;
    double       *degbuf;
} viol_ctx;

double sag_loop_violation_from_dense(const double *dense, const double *degree, int64_t n,
                                     const int32_t *nodes_ext, int64_t size, double rhs,
                                     double *block, double *degbuf)
{
    if (size <= 0) return 0.0;
    int64_t pos = 0;
    for (int64_t a = 0; a < size; ++a) {
        const int64_t ia = (int64_t)nodes_ext[a] - 1;
        if (ia < 0 || ia >= n)
            sag_die("loop: violation node external id %d is out of range [1,%lld]", nodes_ext[a], (long long)n);
        const double *row = dense + ia * n;
        for (int64_t b = 0; b < size; ++b)
            block[pos++] = row[(int64_t)nodes_ext[b] - 1];
        degbuf[a] = degree[ia];
    }
    const double inside = sag_f64_sum_numpy(block, size * size);
    const double boundary = sag_f64_sum_numpy(degbuf, size) - inside;
    return 2.0 * rhs - boundary;
}

static double viol_from_dense(void *user, const uint64_t *mask, const int32_t *nodes_ext,
                              int64_t size, double rhs)
{
    (void)mask;
    viol_ctx *v = (viol_ctx *)user;
    return sag_loop_violation_from_dense(v->dense, v->degree, v->n_nodes, nodes_ext, size,
                                         rhs, v->block, v->degbuf);
}

/* ===================================================================== */
/* 3. Names of the JSON timing buckets (order = the JSON field contract)  */
/* ===================================================================== */

enum {
    LT_LP_SOLVE = 0, LT_SOLUTION_EXTRACT, LT_SUPPORT_EXTRACT, LT_POLICY_CALL,
    LT_CVRPSEP_WAIT, LT_ITERATION_HOOK, LT_CANDIDATE_PREPARE, LT_CANDIDATE_VIOLATION,
    LT_CANDIDATE_LIMIT, LT_CONSTRAINT_BUILD, LT_CONSTRAINT_ADD, LT_MODEL_UPDATE,
    LT_ITERATION_TOTAL, LT_COUNT
};

enum {
    PT_TOTAL = 0, PT_PREPARE_TOTAL, PT_GRAPH_BUILD, PT_BATCH_MATERIALIZE,
    PT_VEHICLE_UNPACK, PT_FORWARD, PT_PROB_EXPORT, PT_THRESHOLD, PT_SUBSET_MATERIALIZE,
    PT_RHS, PT_CUT_VALUE, PT_UNIFIED_PREPARE, PT_UNIFIED_SWEEP, PT_UNIFIED_ONION,
    PT_UNIFIED_DECODE_WALL, PT_UNIFIED_DEDUP, PT_UNIFIED_SCORING, PT_UNIFIED_SELECTOR,
    PT_UNIFIED_BUILD, PT_POSTPROCESS_TOTAL, PT_UNIFIED_TTS, PT_COUNT
};

static const char *const POLICY_JSON_NAMES[PT_COUNT] = {
    "policy_total", "policy_prepare_total", "policy_graph_build", "policy_batch_materialize",
    "policy_vehicle_unpack", "policy_forward", "policy_prob_export", "policy_threshold",
    "policy_subset_materialize", "policy_rhs", "policy_cut_value", "policy_unified_prepare",
    "policy_unified_sweep", "policy_unified_onion", "policy_unified_decode_wall",
    "policy_unified_dedup", "policy_unified_scoring", "policy_unified_selector",
    "policy_unified_build", "policy_postprocess_total", "policy_unified_tts"
};

static const char *const COUNT_NAMES[10] = {
    "support_edges", "proposed_cuts", "policy_proposed", "policy_added",
    "candidate_cuts", "cuts_added", "cuts_removed", "batch_nnz",
    "matrix_add_used", "constraints"
};

/* ===================================================================== */
/* 4. Small helpers                                                      */
/* ===================================================================== */

void sag_loop_options_init(sag_loop_options *opt)
{
    memset(opt, 0, sizeof *opt);
    opt->max_iter = -1;
    opt->emit_diagnostics = true;
}

static void totals_add(sag_json_f64_field *tot, int64_t *n_tot, int64_t cap,
                       const char *name, double value)
{
    for (int64_t i = 0; i < *n_tot; ++i) {
        if (tot[i].name == name || strcmp(tot[i].name, name) == 0) {
            tot[i].value += value;
            return;
        }
    }
    if (*n_tot >= cap) sag_die("loop: timing_totals exceeded %lld keys", (long long)cap);
    tot[*n_tot].name = name;
    tot[*n_tot].value = value;
    *n_tot += 1;
}

static const char *render_tts_json(sag_arena *a, const sag_tts_diag *diag)
{
    char  *buf = NULL;
    size_t len = 0;
    FILE  *mem = open_memstream(&buf, &len);
    if (!mem) sag_die("loop: open_memstream failed while rendering the tts diagnostics");
    sag_tts_diag_write_json(mem, diag);
    if (fclose(mem) != 0) sag_die("loop: writing the rendered tts diagnostics failed");
    char *dst = SAG_NEW(a, char, (int64_t)len + 1);
    memcpy(dst, buf, len);
    dst[len] = '\0';
    free(buf);
    return dst;
}

/* ===================================================================== */
/* 5. run_instance                                                        */
/* ===================================================================== */

void sag_run_instance(const sag_instance *inst, const sag_config *cfg,
                      const sag_loop_options *opt, sag_arena *result_arena,
                      sag_root_result *out)
{
    if (!inst || !cfg || !opt || !result_arena || !out)
        sag_die("sag_run_instance: NULL argument");
    if (!opt->forward)
        sag_die("sag_run_instance: a forward backend is required (--onnx <model.onnx>)");
    sag_config_check_frozen(cfg);

    const int64_t n = inst->n_nodes;
    const int64_t words = sag_mask_words(n);
    const int64_t vehicles = inst->vehicles;

    const bool   has_tl = cfg->lp.time_limit > 0.0;
    const double time_limit = cfg->lp.time_limit;
    if (opt->max_iter < -1)
        sag_die("loop: options.max_iter = %" PRId64 " -- only >=0 (a limit) or -1 (not given) is accepted; "
                "a negative max_iter means 'finite and already reached' in the reference, the opposite of 'unlimited' here",
                opt->max_iter);
    if (cfg->loop.max_iter < -1)
        sag_die("loop: cfg.loop.max_iter = %" PRId64 " -- only >=0 or -1 (not given) is accepted",
                cfg->loop.max_iter);
    const int64_t max_iter = opt->max_iter >= 0 ? opt->max_iter : cfg->loop.max_iter;
    const double min_violation_default = cfg->loop.min_violation;
    const bool   can_generate = (max_iter != 0);

    /* ---- arena ---- */
    size_t long_bytes = opt->long_arena_bytes;
    if (long_bytes == 0) {
        size_t est = (size_t)n * (size_t)n * sizeof(double) * 2u
                   + (size_t)(cfg->tts.old_repair_bank_size + 1) * (size_t)words * 8u
                   + (size_t)n * (size_t)n * 8u
                   + (128u << 20);
        long_bytes = est;
    }
    size_t iter_bytes = opt->iter_arena_bytes;
    if (iter_bytes == 0) iter_bytes = 1024u << 20;

    sag_arena longt, iter;
    sag_arena_init(&longt, long_bytes, "loop.long");
    sag_arena_init(&iter, iter_bytes, "loop.iter");

    /* ---- LP ---- */
    sag_lp *lp = sag_lp_create(inst, &cfg->lp);
    const int64_t base = sag_lp_base_rows(lp);
    const int64_t n_vars = sag_lp_num_vars(lp);
    const int64_t *edge_i = sag_lp_edge_u(lp);
    const int64_t *edge_j = sag_lp_edge_v(lp);

    /* ---- Buffers that live across iterations ---- */
    double *dense_buf = SAG_NEW0(&longt, double, n * n);
    double *degree_buf = SAG_NEW0(&longt, double, n);
    viol_ctx vctx;
    vctx.dense = dense_buf;
    vctx.degree = degree_buf;
    vctx.n_nodes = n;
    vctx.block = SAG_NEW(&longt, double, n * n);
    vctx.degbuf = SAG_NEW(&longt, double, n);

    int64_t *sup_u = SAG_NEW(&longt, int64_t, n_vars);
    int64_t *sup_v = SAG_NEW(&longt, int64_t, n_vars);
    double  *sup_x = SAG_NEW(&longt, double, n_vars);

    sag_old_bank bank;
    sag_old_bank_init(&bank, &longt, n, cfg->tts.old_repair_bank_size);

    sag_key_set seen;
    sag_key_set_init(&seen, words);

    const int64_t replica_count = sag_vehicle_replica_count(inst);

    sag_round_env round_env;
    round_env.inst = inst;
    round_env.cfg = cfg;
    round_env.forward = opt->forward;
    round_env.replica_count = replica_count;
    round_env.bank = &bank;

    /* ---- Result containers (on the result arena) ---- */
    const int64_t cap_iters = max_iter > 0 ? max_iter : 65536;
    sag_json_iteration *logs = SAG_NEW0(result_arena, sag_json_iteration, cap_iters);
    const int64_t TOTALS_CAP = LT_COUNT + PT_COUNT + 8;
    sag_json_f64_field *totals = SAG_NEW0(result_arena, sag_json_f64_field, TOTALS_CAP);
    int64_t n_totals = 0;

    /* ---- Loop state ---- */
    const double start_time = now_sec();
    int64_t iteration = 0, iteration_count = 0, total_cuts = 0, last_cuts_added = 0;
    double last_max_violation = 0.0;
    double final_objective = INFINITY;
    bool time_limit_hit = false;
    double load_time = 0.0;

    for (;;) {
        iteration += 1;
        if (iteration > cap_iters)
            sag_die("loop: iteration count exceeded the reserved limit %lld (use --max-iter, or raise the limit)",
                    (long long)cap_iters);
        sag_arena_reset(&iter);

        const double iteration_start = now_sec();
        const double solve_start = now_sec();
        sag_lp_optimize(lp);
        const double solve_time = now_sec() - solve_start;
        const double objective = sag_lp_objective(lp);

        /* ---- Solution extraction ---- */
        const double sol_start = now_sec();
        const double *values = sag_lp_solution(lp);
        memset(dense_buf, 0, (size_t)(n * n) * sizeof(double));
        for (int64_t k = 0; k < n_vars; ++k) {
            dense_buf[edge_i[k] * n + edge_j[k]] = values[k];
            dense_buf[edge_j[k] * n + edge_i[k]] = values[k];
        }
        memset(degree_buf, 0, (size_t)n * sizeof(double));
        for (int64_t k = 0; k < n_vars; ++k) degree_buf[edge_i[k]] += values[k];
        for (int64_t k = 0; k < n_vars; ++k) degree_buf[edge_j[k]] += values[k];
        const double solution_extract_time = now_sec() - sol_start;

        /* ---- policy ---- */
        double policy_time = 0.0, support_extract_time = 0.0;
        double policy_info[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
        int64_t policy_info_count = 0;
        double policy_timings[PT_COUNT];
        memset(policy_timings, 0, sizeof policy_timings);
        int64_t support_count = 0;
        int64_t policy_proposed_count = 0;
        const sag_materialized_cuts *cuts = NULL;
        sag_round_result round;
        memset(&round, 0, sizeof round);
        bool have_policy = false;

        if (can_generate) {
            const double sup_start = now_sec();
            support_count = sag_lp_support_edges(lp, 1e-8, sup_u, sup_v, sup_x);
            support_extract_time = now_sec() - sup_start;
            if (opt->probe)
                opt->probe(opt->probe_user, iteration, sup_u, sup_v, sup_x, support_count);

            const uint64_t *blocked = seen.masks;
            const int64_t   n_blocked = seen.count;

            const double policy_start = now_sec();

            sag_round_probes probes;
            probes.bank = opt->probe_bank;
            probes.spine = opt->probe_spine;
            probes.user = opt->probe_user;
            sag_round_execute(&round_env, &iter, iteration, sup_u, sup_v, sup_x,
                              support_count, blocked, n_blocked, &probes, &round);
            have_policy = true;
            cuts = &round.spine.cuts;
            policy_proposed_count = round.spine.cuts.rows;

            const double t_dedup = round.spine.diag.timings[SAG_PIPE_T_DEDUP];
            const double t_scoring = round.spine.diag.timings[SAG_PIPE_T_SCORING];
            const double t_selector = round.spine.diag.timings[SAG_PIPE_T_SELECTOR];
            const double t_mat = round.spine.diag.timings[SAG_PIPE_T_MATERIALIZE];

            policy_timings[PT_TOTAL] = round.total_seconds;
            policy_timings[PT_PREPARE_TOTAL] = round.graph_seconds;
            policy_timings[PT_GRAPH_BUILD] = round.graph_seconds;
            policy_timings[PT_FORWARD] = round.forward_seconds;
            policy_timings[PT_THRESHOLD] = round.decode_wall_seconds;
            policy_timings[PT_SUBSET_MATERIALIZE] = t_mat;
            policy_timings[PT_UNIFIED_PREPARE] = round.prepare_seconds;
            policy_timings[PT_UNIFIED_SWEEP] = round.sweep_seconds;
            policy_timings[PT_UNIFIED_ONION] = round.onion_seconds;
            policy_timings[PT_UNIFIED_DECODE_WALL] = round.decode_wall_seconds;
            policy_timings[PT_UNIFIED_DEDUP] = t_dedup;
            policy_timings[PT_UNIFIED_SCORING] = t_scoring;
            policy_timings[PT_UNIFIED_SELECTOR] = t_selector;
            policy_timings[PT_UNIFIED_BUILD] = t_mat;
            policy_timings[PT_POSTPROCESS_TOTAL] = round.prepare_seconds
                + round.decode_wall_seconds + t_dedup + t_scoring + t_selector + t_mat;
            policy_timings[PT_UNIFIED_TTS] = round.tts_seconds;

            policy_info[0] = round.total_seconds;
            policy_info[1] = load_time;
            policy_info[2] = round.graph_seconds;
            policy_info[3] = round.forward_seconds;
            policy_info[4] = round.mean_set_size;
            policy_info_count = 5;
            load_time = 0.0;
            /* --- end of proposal --- */

            policy_time = now_sec() - policy_start;
        }

        /* ---- Candidates -> LP ---- */
        double *iteration_violations = NULL;
        int64_t n_iteration_violations = 0;
        double raw_max_violation = 0.0, max_added_violation = 0.0;
        int64_t cuts_added = 0, policy_added_to_lp = 0, cuts_removed = 0;
        int64_t candidate_count = 0, batch_nnz = 0, matrix_add_used = 0;
        int64_t constraint_count = 0;
        double candidate_prepare_time = 0.0, candidate_violation_time = 0.0;
        double candidate_limit_time = 0.0, constraint_build_time = 0.0, constraint_add_time = 0.0;

        const int64_t n_sets = cuts ? cuts->rows : 0;

        {
            int64_t *cand_row = NULL;
            double  *cand_viol = NULL;
            uint64_t *cand_key = NULL;
            if (n_sets > 0) {
                cand_row = SAG_NEW(&iter, int64_t, n_sets);
                cand_viol = SAG_NEW(&iter, double, n_sets);
                cand_key = SAG_NEW0(&iter, uint64_t, n_sets * words);
                iteration_violations = SAG_NEW(result_arena, double, n_sets);
            }
            for (int64_t r = 0; r < n_sets; ++r) {
                const double prep0 = now_sec();
                const int64_t lo = cuts->indptr[r], hi = cuts->indptr[r + 1];
                uint64_t *key = cand_key + r * words;
                const int64_t size = sag_round_cut_key(cuts, r, n, words, key);
                if (size == 0) { candidate_prepare_time += now_sec() - prep0; continue; }
                const bool duplicate = sag_key_set_contains(&seen, key);
                candidate_prepare_time += now_sec() - prep0;

                const double viol0 = now_sec();
                const double raw_violation = viol_from_dense(&vctx, key, cuts->indices + lo,
                                                             hi - lo, cuts->rhs[r]);
                candidate_violation_time += now_sec() - viol0;

                if (!duplicate) iteration_violations[n_iteration_violations++] = raw_violation;
                const double violation = raw_violation > 0.0 ? raw_violation : 0.0;
                if (duplicate) continue;
                if (violation <= 0.0) continue;
                if (violation > raw_max_violation) raw_max_violation = violation;
                cand_row[candidate_count] = r;
                cand_viol[candidate_count] = violation;
                candidate_count += 1;
            }

            const double lim0 = now_sec();
            const double violation_threshold = min_violation_default;
            candidate_limit_time += now_sec() - lim0;

            int64_t n_sel = 0, sel_nnz = 0;
            for (int64_t i = 0; i < candidate_count; ++i) {
                if (!(cand_viol[i] > violation_threshold)) continue;
                const int64_t r = cand_row[i];
                sel_nnz += cuts->indptr[r + 1] - cuts->indptr[r];
                cand_row[n_sel] = r;
                cand_viol[n_sel] = cand_viol[i];
                n_sel += 1;
            }

            if (n_sel > 0) {
                sag_materialized_cuts rows;
                memset(&rows, 0, sizeof rows);
                rows.rows = n_sel;
                rows.nnz = sel_nnz;
                rows.indptr = SAG_NEW(&iter, int64_t, n_sel + 1);
                rows.indices = SAG_NEW(&iter, int32_t, sel_nnz > 0 ? sel_nnz : 1);
                rows.rhs = SAG_NEW(&iter, double, n_sel);
                rows.sizes = SAG_NEW(&iter, int64_t, n_sel);
                const double build0 = now_sec();
                rows.indptr[0] = 0;
                int64_t pos = 0;
                for (int64_t i = 0; i < n_sel; ++i) {
                    const int64_t r = cand_row[i];
                    for (int64_t t = cuts->indptr[r]; t < cuts->indptr[r + 1]; ++t)
                        rows.indices[pos++] = cuts->indices[t];
                    rows.rhs[i] = cuts->rhs[r];
                    rows.sizes[i] = cuts->sizes[r];
                    rows.indptr[i + 1] = pos;
                }
                constraint_build_time += now_sec() - build0;
                const double add0 = now_sec();
                batch_nnz = sag_lp_add_cuts(lp, &rows);
                constraint_add_time += now_sec() - add0;
                matrix_add_used = 1;
                for (int64_t i = 0; i < n_sel; ++i) {
                    sag_key_set_add(&seen, cand_key + cand_row[i] * words);
                    cuts_added += 1;
                    if (cand_viol[i] > max_added_violation) max_added_violation = cand_viol[i];
                }
                policy_added_to_lp = n_sel;
            }
            constraint_count = base + total_cuts + cuts_added;
        }

        double logged_max_violation = raw_max_violation;
        if (max_added_violation > logged_max_violation) logged_max_violation = max_added_violation;
        const double iteration_total_time = now_sec() - iteration_start;

        /* ---- counts / timings / diagnostics accumulated for the JSON ---- */
        const int64_t n_counts = 10;
        sag_json_i64_field *counts = SAG_NEW(result_arena, sag_json_i64_field, n_counts);
        const int64_t count_vals[10] = {
            support_count, n_sets, policy_proposed_count, policy_added_to_lp,
            candidate_count, cuts_added, cuts_removed, batch_nnz,
            matrix_add_used, constraint_count
        };
        for (int64_t i = 0; i < n_counts; ++i) {
            counts[i].name = COUNT_NAMES[i];
            counts[i].value = count_vals[i];
        }

        const double loop_vals[LT_COUNT] = {
            solve_time, solution_extract_time, support_extract_time, policy_time,
            0.0, 0.0,
            candidate_prepare_time, candidate_violation_time, candidate_limit_time,
            constraint_build_time, constraint_add_time, 0.0,
            iteration_total_time
        };
        sag_json_f64_field *timings = SAG_NEW(result_arena, sag_json_f64_field,
                                              LT_COUNT + PT_COUNT);
        int64_t n_timings = 0;
        for (int64_t i = 0; i < LT_COUNT; ++i) {
            if (!(loop_vals[i] > 0.0)) continue;
            timings[n_timings].name = sag_loop_timing_names[i];
            timings[n_timings].value = loop_vals[i];
            n_timings += 1;
        }
        for (int64_t i = 0; i < PT_COUNT; ++i) {
            if (!(policy_timings[i] > 0.0)) continue;
            timings[n_timings].name = POLICY_JSON_NAMES[i];
            timings[n_timings].value = policy_timings[i];
            n_timings += 1;
        }
        for (int64_t i = 0; i < n_timings; ++i)
            totals_add(totals, &n_totals, TOTALS_CAP, timings[i].name, timings[i].value);

        sag_json_iter_diag *diag_out = NULL;
        if (have_policy && opt->emit_diagnostics) {
            sag_pipeline_diag *pd = SAG_NEW(result_arena, sag_pipeline_diag, 1);
            *pd = round.spine.diag;
            pd->added_to_lp = policy_added_to_lp;
            diag_out = SAG_NEW0(result_arena, sag_json_iter_diag, 1);
            diag_out->pipeline = pd;
            if (round.spine.family_count < 0 || round.spine.family_count > SAG_FAMILY_COUNT)
                sag_die("loop: spine.family_count=%lld out of range [0,%d]",
                        (long long)round.spine.family_count, SAG_FAMILY_COUNT);
            diag_out->family_count = round.spine.family_count;
            for (int64_t i = 0; i < round.spine.family_count; ++i)
                diag_out->family_order[i] = round.spine.family_order[i];
            diag_out->tts_json = round.tts_diag.count > 0
                               ? render_tts_json(result_arena, &round.tts_diag) : NULL;
        }

        double *policy_info_copy = NULL;
        if (policy_info_count > 0) {
            policy_info_copy = SAG_NEW(result_arena, double, policy_info_count);
            memcpy(policy_info_copy, policy_info, (size_t)policy_info_count * sizeof(double));
        }

        sag_json_iteration *log = &logs[iteration - 1];
        memset(log, 0, sizeof *log);
        log->iteration = iteration;
        log->objective = objective;
        log->cuts_added = cuts_added;
        log->max_violation = logged_max_violation;
        log->policy_info = policy_info_copy;
        log->policy_info_count = policy_info_count;
        log->solve_time = solve_time;
        log->cut_generation_time = policy_time;
        log->constraints = constraint_count;
        log->violations = iteration_violations;
        log->violation_count = n_iteration_violations;
        log->counts = counts;
        log->counts_count = n_counts;
        log->timings = timings;
        log->timings_count = n_timings;
        log->diag = diag_out;

        iteration_count = iteration;
        total_cuts += cuts_added;
        last_cuts_added = cuts_added;
        last_max_violation = logged_max_violation;
        final_objective = objective;

        if (opt->progress) {
            fprintf(opt->progress,
                    "  it%-4lld obj=%.10f cuts=%-5lld rows=%-6lld viol=%.3e t=%.2fs\n",
                    (long long)iteration, objective, (long long)cuts_added,
                    (long long)constraint_count, logged_max_violation,
                    now_sec() - start_time);
            fflush(opt->progress);
        }

        /* ---- Termination checks ---- */
        const double elapsed = now_sec() - start_time;
        bool should_stop = false;
        if (cuts_added == 0) should_stop = true;
        else if (max_iter >= 0 && iteration >= max_iter) should_stop = true;
        if (has_tl && elapsed >= time_limit) { time_limit_hit = true; should_stop = true; }
        if (should_stop) break;
    }

    const double total_time = now_sec() - start_time;

    /* ---- Result ---- */
    memset(out, 0, sizeof *out);
    out->iterations = logs;
    out->n_iterations = iteration_count;
    out->lower_bound = final_objective;
    out->runtime = total_time;
    out->total_cuts = total_cuts;
    out->iteration_count = iteration_count;
    out->time_limit_hit = time_limit_hit;

    out->json.instance = inst->name;
    out->json.vehicles = vehicles;
    out->json.lower_bound = final_objective;
    out->json.runtime = total_time;
    out->json.iterations = logs;
    out->json.iterations_count = iteration_count;
    out->json.iteration_count = iteration_count;
    out->json.total_cuts = total_cuts;
    out->json.last_cuts_added = last_cuts_added;
    out->json.last_max_violation = last_max_violation;
    out->json.has_optimal = inst->has_optimal;
    out->json.optimal = inst->optimal;
    out->json.has_time_limit = has_tl;
    out->json.time_limit = time_limit;
    out->json.time_limit_hit = time_limit_hit;
    out->json.timing_totals = totals;
    out->json.timing_totals_count = n_totals;

    /* ---- Teardown: dispose of the solver state and return freed memory to the OS ---- */
    sag_lp_destroy(lp);
    sag_key_set_free(&seen);
    sag_arena_destroy(&iter);
    sag_arena_destroy(&longt);
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}
