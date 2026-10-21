#include "f32_sparse.h"

#define SAG_PW_BLOCKSIZE       128
#define SAG_F32_NUMPY_BUFSIZE  8192

/* =====================================================================
 * sparse_pw_dense
 * ===================================================================== */
static float sparse_pw_dense(const float *dense, const int32_t *nz, int64_t nnz,
                             int64_t lo, int64_t len, int64_t *cur)
{
    if (*cur >= nnz || (int64_t)nz[*cur] >= lo + len) return 0.0f;

    if (len < 8) {
        float res = 0.0f;
        while (*cur < nnz && (int64_t)nz[*cur] < lo + len) res += dense[nz[(*cur)++]];
        return res;
    }

    if (len <= SAG_PW_BLOCKSIZE) {
        float r[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const int64_t end = len - (len % 8);
        while (*cur < nnz && (int64_t)nz[*cur] - lo < end) {
            const int64_t p = (int64_t)nz[*cur] - lo;
            r[p & 7] += dense[nz[*cur]];
            ++(*cur);
        }
        float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while (*cur < nnz && (int64_t)nz[*cur] < lo + len) res += dense[nz[(*cur)++]];
        return res;
    }

    int64_t n2 = len / 2;
    n2 -= n2 % 8;
    const float L = (*cur < nnz && (int64_t)nz[*cur] < lo + n2)
                        ? sparse_pw_dense(dense, nz, nnz, lo, n2, cur) : 0.0f;
    const float R = (*cur < nnz && (int64_t)nz[*cur] < lo + len)
                        ? sparse_pw_dense(dense, nz, nnz, lo + n2, len - n2, cur) : 0.0f;
    return L + R;
}

float sag_f32_sum_sparse(const float *dense, const int32_t *nz, int64_t nnz, int64_t n)
{
    float res = 0.0f;
    int64_t cur = 0, off = 0;
    while (off < n) {
        int64_t chunk = n - off;
        if (chunk > SAG_F32_NUMPY_BUFSIZE) chunk = SAG_F32_NUMPY_BUFSIZE;
        res += sparse_pw_dense(dense, nz, nnz, off, chunk, &cur);
        off += chunk;
    }
    return res;
}

/* =====================================================================
 * sparse_pw_scaled
 * ===================================================================== */
static float sparse_pw_scaled(const float *a, const float *b, const int32_t *nz, int64_t nnz,
                              int64_t lo, int64_t len, int64_t *cur)
{
    if (*cur >= nnz || (int64_t)nz[*cur] >= lo + len) return 0.0f;

    if (len < 8) {
        float res = 0.0f;
        while (*cur < nnz && (int64_t)nz[*cur] < lo + len) {
            const int32_t j = nz[(*cur)++];
            res += a[j] * b[j];
        }
        return res;
    }

    if (len <= SAG_PW_BLOCKSIZE) {
        float r[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const int64_t end = len - (len % 8);
        while (*cur < nnz && (int64_t)nz[*cur] - lo < end) {
            const int32_t j = nz[*cur];
            const int64_t p = (int64_t)j - lo;
            r[p & 7] += a[j] * b[j];
            ++(*cur);
        }
        float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while (*cur < nnz && (int64_t)nz[*cur] < lo + len) {
            const int32_t j = nz[(*cur)++];
            res += a[j] * b[j];
        }
        return res;
    }

    int64_t n2 = len / 2;
    n2 -= n2 % 8;
    const float L = (*cur < nnz && (int64_t)nz[*cur] < lo + n2)
                        ? sparse_pw_scaled(a, b, nz, nnz, lo, n2, cur) : 0.0f;
    const float R = (*cur < nnz && (int64_t)nz[*cur] < lo + len)
                        ? sparse_pw_scaled(a, b, nz, nnz, lo + n2, len - n2, cur) : 0.0f;
    return L + R;
}

float sag_f32_sum_sparse_scaled(const float *a, const float *b,
                                const int32_t *nz, int64_t nnz, int64_t n)
{
    float res = 0.0f;
    int64_t cur = 0, off = 0;
    while (off < n) {
        int64_t chunk = n - off;
        if (chunk > SAG_F32_NUMPY_BUFSIZE) chunk = SAG_F32_NUMPY_BUFSIZE;
        res += sparse_pw_scaled(a, b, nz, nnz, off, chunk, &cur);
        off += chunk;
    }
    return res;
}
