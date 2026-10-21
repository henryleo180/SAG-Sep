#ifndef SAGSEP_FORWARD_H
#define SAGSEP_FORWARD_H

#include <stdbool.h>
#include <stdint.h>

#include "features.h"
#include "sagsep.h"

typedef enum {
    SAG_FORWARD_ONNX = 0
} sag_forward_backend;

typedef struct sag_forward sag_forward;

/* ---------------- Lifecycle --------------- */

sag_forward *sag_forward_open_onnx(const char *model_path, const char *provider);

void sag_forward_close(sag_forward *f);

/* ---------------- Queries ----------------- */

sag_forward_backend sag_forward_kind(const sag_forward *f);
bool                sag_forward_onnx_available(void);
const char         *sag_forward_describe(const sag_forward *f);

/* ---------------- The forward channel ---------------- */

const float *sag_forward_probs_graph(sag_forward *f, const sag_graph_features *g);

#endif
