#include "families/tts/flow.h"

#include <math.h>
#include <string.h>

#include "bitset.h"

void sag_tts_flow_state_init(sag_arena *a, sag_tts_flow_state *fs, int64_t vehicle_count,
                             int64_t n_nodes, int64_t words, int64_t cap)
{
    if (cap <= 0) sag_die("tts/flow: flow_state cap=%lld must be positive", (long long)cap);
    fs->vehicle_count = vehicle_count;
    fs->n_nodes       = n_nodes;
    fs->words         = words;
    fs->count         = 0;
    fs->emitted       = 0;
    fs->cap           = cap;
    fs->masks         = SAG_NEW(a, uint64_t, cap * words);
    fs->sources       = SAG_NEW(a, int8_t, cap);
    fs->stages        = SAG_NEW(a, int8_t, cap);
    fs->seeds         = SAG_NEW(a, int32_t, cap);
    sag_tts_keyset_init(a, &fs->seen, words, cap);
}

double sag_tts_flow_scale(const sag_tts_config *cfg, int64_t vehicle_count)
{
    const int64_t start = cfg->vehicle_tighten_start;
    if (start <= 0 || vehicle_count <= start) return 1.0;
    const int64_t denom = (vehicle_count > 1) ? vehicle_count : 1;
    const double  raw   = pow((double)start / (double)denom, cfg->vehicle_tighten_power);
    const double  hi    = (raw < 1.0) ? raw : 1.0;
    return (cfg->vehicle_tighten_min_scale > hi) ? cfg->vehicle_tighten_min_scale : hi;
}

int64_t sag_tts_scaled_limit(int64_t value, double scale)
{
    if (value <= 0) return 0;
    const int64_t v = (int64_t)ceil((double)value * scale);
    return (v > 1) ? v : 1;
}

static int8_t source_slot(const char *name)
{
    static const int8_t candidates[4] = {
        SAG_VAR_LP_GROW, SAG_VAR_FCAPFIX, SAG_VAR_OLD_REPAIR, SAG_VAR_ADDDROP
    };
    for (int i = 0; i < 4; ++i)
        if (strcmp(name, sag_variant_names[candidates[i]]) == 0) return candidates[i];
    return -1;
}

int64_t sag_tts_apply_cut_flow(sag_arena *scratch, const sag_tts_proposal *props, int64_t n_props,
                               const sag_tts_config *cfg, sag_tts_flow_state *fs,
                               int64_t *out_idx, sag_tts_flow_diag *diag)
{
    if (diag) memset(diag, 0, sizeof *diag);
    if (n_props <= 0) return 0;

    const double  scale         = sag_tts_flow_scale(cfg, fs->vehicle_count);
    const int64_t vehicle_limit = sag_tts_scaled_limit(cfg->per_vehicle_quota, scale);
    const int64_t global_limit  = sag_tts_scaled_limit(cfg->global_quota, scale);
    const int64_t hamming_limit = cfg->near_duplicate_hamming;
    const int64_t n             = fs->n_nodes;
    const int64_t words         = fs->words;

    int64_t source_limits[SAG_VARIANT_COUNT];
    for (int i = 0; i < SAG_VARIANT_COUNT; ++i) source_limits[i] = -1;
    for (int64_t q = 0; q < cfg->source_quota_count; ++q) {
        if (cfg->source_quotas[q].limit <= 0) continue;
        const int8_t slot = source_slot(cfg->source_quotas[q].name);
        if (slot < 0) continue;
        source_limits[slot] = sag_tts_scaled_limit(cfg->source_quotas[q].limit, scale);
    }

    if (diag) {
        for (int64_t i = 0; i < n_props; ++i) {
            diag->input_by_source[props[i].source] += 1;
            diag->input_by_stage[props[i].origin_stage ? 1 : 0] += 1;
        }
    }

    const size_t mark = sag_arena_mark(scratch);

    int64_t *remaining = SAG_NEW(scratch, int64_t, n_props);
    int64_t  n_rem = 0;
    for (int64_t i = 0; i < n_props; ++i)
        if (!sag_tts_keyset_has(&fs->seen, props[i].mask)) remaining[n_rem++] = i;
    if (n_rem == 0) { sag_arena_release(scratch, mark); return 0; }

    int64_t *min_hamming = SAG_NEW(scratch, int64_t, n_rem);
    int8_t  *blocker     = SAG_NEW(scratch, int8_t, n_rem);
    int8_t  *blk_stage   = SAG_NEW(scratch, int8_t, n_rem);
    int32_t *blk_seed    = SAG_NEW(scratch, int32_t, n_rem);
    bool    *near_hit    = SAG_NEW0(scratch, bool, n_rem);
    bool    *active      = SAG_NEW(scratch, bool, n_rem);
    for (int64_t i = 0; i < n_rem; ++i) {
        min_hamming[i] = n;
        blocker[i]     = (int8_t)SAG_TTS_BLOCKER_UNKNOWN;
        blk_stage[i]   = 0;
        blk_seed[i]    = -1;
        active[i]      = true;
    }

    if (fs->count > 0) {
        for (int64_t i = 0; i < n_rem; ++i) {
            const uint64_t *m = props[remaining[i]].mask;
            int64_t best = n;
            int8_t  who  = (int8_t)SAG_TTS_BLOCKER_UNKNOWN;
            int8_t  wst  = 0;
            int32_t wsd  = -1;
            for (int64_t r = 0; r < fs->count; ++r) {
                const int64_t d = sag_bitset_hamming(m, fs->masks + r * words, words);
                if (d < best) { best = d; who = fs->sources[r];
                                wst = fs->stages[r]; wsd = fs->seeds[r]; }
            }
            min_hamming[i] = best;
            if (hamming_limit > 0 && diag) {
                blocker[i] = who; blk_stage[i] = wst; blk_seed[i] = wsd;
            }
        }
        if (hamming_limit > 0)
            for (int64_t i = 0; i < n_rem; ++i) near_hit[i] = (min_hamming[i] <= hamming_limit);
    }

    int64_t local_counts[SAG_VARIANT_COUNT];
    for (int i = 0; i < SAG_VARIANT_COUNT; ++i) local_counts[i] = 0;
    const double denom = (double)((n - 1) > 1 ? (n - 1) : 1);
    int64_t n_selected = 0;

    for (;;) {
        bool any_active = false;
        for (int64_t i = 0; i < n_rem; ++i) if (active[i]) { any_active = true; break; }
        if (!any_active) break;
        if (vehicle_limit > 0 && n_selected >= vehicle_limit) break;
        if (global_limit > 0 && fs->emitted + n_selected >= global_limit) break;

        const bool apply_hamming = ((fs->count > 0 || n_selected > 0) && hamming_limit > 0);
        int64_t best_idx = -1;
        double  best_rank = 0.0;
        for (int64_t i = 0; i < n_rem; ++i) {
            if (!active[i]) continue;
            const int8_t src = props[remaining[i]].source;
            if (source_limits[src] >= 0 && local_counts[src] >= source_limits[src]) continue;
            if (apply_hamming && !(min_hamming[i] > hamming_limit)) continue;
            const double rank = props[remaining[i]].score
                              + cfg->diversity_weight * ((double)min_hamming[i] / denom);
            if (best_idx < 0 || !(rank <= best_rank)) {
                best_idx = i; best_rank = rank;
                if (isnan(rank)) break;
            }
        }
        if (best_idx < 0) break;

        const int64_t chosen = remaining[best_idx];
        active[best_idx] = false;
        out_idx[n_selected++] = chosen;
        local_counts[props[chosen].source] += 1;

        const uint64_t *cm = props[chosen].mask;
        for (int64_t i = 0; i < n_rem; ++i) {
            const int64_t d = sag_bitset_hamming(props[remaining[i]].mask, cm, words);
            if (hamming_limit > 0 && diag) {
                if (active[i] && !near_hit[i] && d <= hamming_limit) {
                    blocker[i]   = props[chosen].source;
                    blk_stage[i] = props[chosen].origin_stage;
                    blk_seed[i]  = props[chosen].origin_seed;
                    near_hit[i]  = true;
                }
            }
            if (d < min_hamming[i]) min_hamming[i] = d;
        }
    }

    for (int64_t k = 0; k < n_selected; ++k) {
        if (fs->count >= fs->cap)
            sag_die("tts/flow: flow_state reference set overflow (cap=%lld) -- the reserved amount was computed wrong",
                    (long long)fs->cap);
        sag_bitset_copy(fs->masks + fs->count * words, props[out_idx[k]].mask, words);
        fs->sources[fs->count] = props[out_idx[k]].source;
        fs->stages[fs->count]  = props[out_idx[k]].origin_stage;
        fs->seeds[fs->count]   = props[out_idx[k]].origin_seed;
        fs->count += 1;
        if (diag) {
            diag->sel_by_source[props[out_idx[k]].source] += 1;
            diag->sel_by_stage[props[out_idx[k]].origin_stage ? 1 : 0] += 1;
        }
        sag_tts_keyset_add(&fs->seen, props[out_idx[k]].mask);
    }
    fs->emitted += n_selected;

    if (diag) {
        for (int64_t i = 0; i < n_rem; ++i) {
            if (!near_hit[i]) continue;
            diag->near_duplicate_hit_count += 1;
            diag->pair_counts[props[remaining[i]].source][blocker[i]] += 1;
            {
                const int fs_ = props[remaining[i]].origin_stage ? 1 : 0;
                const int bs_ = blk_stage[i] ? 1 : 0;
                diag->pair_stage[fs_][bs_] += 1;
                if (props[remaining[i]].origin_seed >= 0 &&
                    props[remaining[i]].origin_seed == blk_seed[i])
                    diag->pair_same_seed[fs_][bs_] += 1;
            }
        }
    }
    sag_arena_release(scratch, mark);
    return n_selected;
}
