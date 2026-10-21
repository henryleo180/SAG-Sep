#ifndef SAGSEP_ROUND_H
#define SAGSEP_ROUND_H

#include <stdbool.h>
#include <stdint.h>

#include "arena.h"
#include "candidate_ctx.h"
#include "families/tts/core.h"
#include "forward.h"
#include "spine.h"
#include "sagsep.h"

/* ===================================================================== */
/* 1. Monotonic clock                                                    */
/* ===================================================================== */

double sag_round_now_seconds(void);

/* ===================================================================== */
/* 2. Blocked-key set                                                    */
/* ===================================================================== */

typedef struct {
    int64_t   words;
    uint64_t *masks;
    int64_t   count;
    int64_t   cap;
    int32_t  *hash;
    int64_t   hash_cap;
} sag_key_set;

void sag_key_set_init(sag_key_set *s, int64_t words);
void sag_key_set_free(sag_key_set *s);
void sag_key_set_clear(sag_key_set *s);
bool sag_key_set_contains(const sag_key_set *s, const uint64_t *mask);
bool sag_key_set_add(sag_key_set *s, const uint64_t *mask);

/* ===================================================================== */
/* 3. Old-cut bank: an LRU map keyed by cut mask                         */
/* ===================================================================== */

typedef struct {
    int64_t   n_nodes, words, limit, cap;
    uint64_t *masks;
    int32_t  *prev, *next;
    int32_t  *hash;
    int64_t   hash_cap, hash_live, hash_tomb;
    int32_t   head, tail;
    int32_t   free_head;
    int64_t   count;
    uint64_t *scratch;
} sag_old_bank;

void sag_old_bank_init(sag_old_bank *b, sag_arena *a, int64_t n_nodes, int64_t limit);

void sag_old_bank_remember(sag_old_bank *b, const sag_materialized_cuts *cuts);

void sag_old_bank_snapshot(const sag_old_bank *b, sag_arena *a, sag_bank *out);

static inline int64_t sag_old_bank_count(const sag_old_bank *b) { return b->count; }

/* ===================================================================== */
/* 4. Cut key of a materialized row                                      */
/* ===================================================================== */

int64_t sag_round_cut_key(const sag_materialized_cuts *cuts, int64_t row,
                          int64_t n_nodes, int64_t words, uint64_t *key);

/* ===================================================================== */
/* 5. The per-round engine                                               */
/* ===================================================================== */

typedef void (*sag_round_bank_probe)(void *user, int64_t iteration, const sag_bank *snap);
typedef void (*sag_round_spine_probe)(void *user, int64_t iteration,
                                      const sag_candidate_batch *sweep,
                                      const sag_candidate_batch *onion,
                                      const sag_candidate_batch *tts,
                                      const uint64_t *blocked, int64_t n_blocked,
                                      const sag_spine_result *spine);

typedef struct {
    sag_round_bank_probe  bank;
    sag_round_spine_probe spine;
    void                 *user;
} sag_round_probes;

typedef struct {
    const sag_instance *inst;
    const sag_config   *cfg;
    sag_forward        *forward;
    int64_t             replica_count;
    sag_old_bank       *bank;
} sag_round_env;

typedef struct {
    sag_spine_result spine;
    sag_tts_diag     tts_diag;
    double  mean_set_size;
    int64_t support_edges;
    int64_t model_edges;
    int64_t sweep_rows, onion_rows, tts_rows;
    double  prepare_lp_seconds, graph_seconds, forward_seconds, prepare_seconds;
    double  sweep_seconds, onion_seconds, tts_seconds, decode_wall_seconds;
    double  spine_seconds;
    double  total_seconds;
} sag_round_result;

void sag_round_execute(const sag_round_env *env, sag_arena *a, int64_t iteration,
                       const int64_t *sup_u, const int64_t *sup_v, const double *sup_x,
                       int64_t support_count,
                       const uint64_t *blocked, int64_t n_blocked,
                       const sag_round_probes *probes,
                       sag_round_result *out);

#endif
