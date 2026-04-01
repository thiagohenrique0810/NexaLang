/*
 * turboquant.c — NexaLang C runtime for TurboQuant vector compression.
 *
 * Pure C implementation of the TurboQuant MSE algorithm:
 *   1. Generate random Haar-distributed orthogonal rotation matrix (QR)
 *   2. Precompute Lloyd-Max codebook for Gaussian(0, 1/sqrt(d))
 *   3. Rotate → scalar quantize each coordinate → indices
 *   4. Lookup centroids → inverse rotate → reconstruct
 *
 * Reference: Zandieh, Daliri, Hadian, Mirrokni — arXiv 2504.19874, 2025.
 */

#include "turboquant.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ──────────────────────────────────────────────────────────────────────── */
/*  Simple xoshiro256** PRNG (deterministic, portable)                    */
/* ──────────────────────────────────────────────────────────────────────── */

typedef struct {
    uint64_t s[4];
} tq_rng;

static inline uint64_t tq_rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t tq_rng_next(tq_rng* rng) {
    uint64_t* s = rng->s;
    uint64_t result = tq_rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = tq_rotl(s[3], 45);
    return result;
}

static void tq_rng_seed(tq_rng* rng, uint64_t seed) {
    /* SplitMix64 to initialize state */
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng->s[i] = z ^ (z >> 31);
    }
}

/* Standard normal via Box-Muller */
static float tq_randn(tq_rng* rng) {
    double u1 = (double)(tq_rng_next(rng) >> 11) / (double)(1ULL << 53);
    double u2 = (double)(tq_rng_next(rng) >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));
}

/* ──────────────────────────────────────────────────────────────────────── */
/*  Internal context                                                       */
/* ──────────────────────────────────────────────────────────────────────── */

struct tq_ctx {
    int dim;
    int bits;
    int n_levels;       /* 2^bits */
    float* rotation;    /* dim x dim, row-major */
    float* rot_inv;     /* dim x dim, transposed rotation (= inverse for orthogonal) */
    float* centroids;   /* n_levels centroids from Lloyd-Max */
    float* buf;         /* scratch buffer for rotation, dim floats */
};

/* ──────────────────────────────────────────────────────────────────────── */
/*  Gaussian PDF for Lloyd-Max (sigma = 1/sqrt(d))                         */
/* ──────────────────────────────────────────────────────────────────────── */

static float gauss_pdf(float x, float sigma) {
    float z = x / sigma;
    return expf(-0.5f * z * z) / (sigma * sqrtf(2.0f * (float)M_PI));
}

/* ──────────────────────────────────────────────────────────────────────── */
/*  Lloyd-Max scalar quantizer (matches Python lloyd_max.py)               */
/* ──────────────────────────────────────────────────────────────────────── */

/* Simple trapezoidal numerical integration */
static void lloyd_max_integrate(float a, float b, float sigma, int n_steps,
                                float* out_num, float* out_den) {
    float h = (b - a) / (float)n_steps;
    float num = 0.0f, den = 0.0f;
    for (int i = 0; i <= n_steps; i++) {
        float x = a + i * h;
        float p = gauss_pdf(x, sigma);
        float w = (i == 0 || i == n_steps) ? 0.5f : 1.0f;
        num += w * x * p;
        den += w * p;
    }
    *out_num = num * h;
    *out_den = den * h;
}

static void lloyd_max(int dim, int bits, float* centroids) {
    int n_levels = 1 << bits;
    float sigma = 1.0f / sqrtf((float)dim);
    float lo = -5.0f * sigma;
    float hi =  5.0f * sigma;

    /* Initialize centroids uniformly */
    for (int i = 0; i < n_levels; i++) {
        centroids[i] = lo + (hi - lo) * ((float)i + 0.5f) / (float)n_levels;
    }

    float* bounds = (float*)malloc((n_levels + 1) * sizeof(float));

    for (int iter = 0; iter < 100; iter++) {
        /* Compute decision boundaries (midpoints) */
        bounds[0] = lo;
        for (int i = 0; i < n_levels - 1; i++) {
            bounds[i + 1] = 0.5f * (centroids[i] + centroids[i + 1]);
        }
        bounds[n_levels] = hi;

        /* Update centroids */
        float max_change = 0.0f;
        for (int i = 0; i < n_levels; i++) {
            float num, den;
            lloyd_max_integrate(bounds[i], bounds[i + 1], sigma, 200, &num, &den);
            float new_c = (den > 1e-12f) ? num / den : centroids[i];
            float diff = fabsf(new_c - centroids[i]);
            if (diff > max_change) max_change = diff;
            centroids[i] = new_c;
        }

        if (max_change < 1e-6f) break;
    }

    free(bounds);

    /* Sort centroids (insertion sort, n_levels is small) */
    for (int i = 1; i < n_levels; i++) {
        float key = centroids[i];
        int j = i - 1;
        while (j >= 0 && centroids[j] > key) {
            centroids[j + 1] = centroids[j];
            j--;
        }
        centroids[j + 1] = key;
    }
}

/* ──────────────────────────────────────────────────────────────────────── */
/*  QR decomposition (Gram-Schmidt) for rotation matrix                    */
/* ──────────────────────────────────────────────────────────────────────── */

static void generate_rotation(int d, tq_rng* rng, float* Q) {
    /* Fill d x d matrix with standard normal entries */
    float* A = (float*)malloc(d * d * sizeof(float));
    for (int i = 0; i < d * d; i++) {
        A[i] = tq_randn(rng);
    }

    /* Modified Gram-Schmidt QR */
    for (int j = 0; j < d; j++) {
        /* Copy column j of A into column j of Q */
        for (int i = 0; i < d; i++) {
            Q[i * d + j] = A[i * d + j];
        }
        /* Orthogonalize against previous columns */
        for (int k = 0; k < j; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) {
                dot += Q[i * d + k] * Q[i * d + j];
            }
            for (int i = 0; i < d; i++) {
                Q[i * d + j] -= dot * Q[i * d + k];
            }
        }
        /* Normalize */
        float norm = 0.0f;
        for (int i = 0; i < d; i++) {
            norm += Q[i * d + j] * Q[i * d + j];
        }
        norm = sqrtf(norm);
        if (norm > 1e-12f) {
            for (int i = 0; i < d; i++) {
                Q[i * d + j] /= norm;
            }
        }
    }

    free(A);
}

/* ──────────────────────────────────────────────────────────────────────── */
/*  Public API                                                             */
/* ──────────────────────────────────────────────────────────────────────── */

tq_ctx* tq_create(int dim, int bits, int seed) {
    tq_ctx* ctx = (tq_ctx*)calloc(1, sizeof(tq_ctx));
    ctx->dim = dim;
    ctx->bits = bits;
    ctx->n_levels = 1 << bits;

    /* Rotation matrix */
    ctx->rotation = (float*)malloc(dim * dim * sizeof(float));
    ctx->rot_inv  = (float*)malloc(dim * dim * sizeof(float));
    tq_rng rng;
    tq_rng_seed(&rng, (uint64_t)seed);
    generate_rotation(dim, &rng, ctx->rotation);

    /* Transpose = inverse for orthogonal matrix */
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            ctx->rot_inv[i * dim + j] = ctx->rotation[j * dim + i];
        }
    }

    /* Lloyd-Max codebook */
    ctx->centroids = (float*)malloc(ctx->n_levels * sizeof(float));
    lloyd_max(dim, bits, ctx->centroids);

    ctx->buf = (float*)malloc(dim * sizeof(float));
    return ctx;
}

void tq_destroy(tq_ctx* ctx) {
    if (!ctx) return;
    free(ctx->rotation);
    free(ctx->rot_inv);
    free(ctx->centroids);
    free(ctx->buf);
    free(ctx);
}

void tq_quantize(tq_ctx* ctx, const float* in, uint16_t* out, int n_vectors) {
    int d = ctx->dim;
    int nl = ctx->n_levels;
    const float* Rt = ctx->rot_inv;  /* R^T: rotate forward */
    const float* C = ctx->centroids;

    for (int v = 0; v < n_vectors; v++) {
        const float* x = in + v * d;
        uint16_t* idx = out + v * d;

        /* Rotate: buf = x @ R^T  (buf[j] = sum_k x[k] * R^T[k][j]) */
        for (int j = 0; j < d; j++) {
            float s = 0.0f;
            for (int k = 0; k < d; k++) {
                s += x[k] * Rt[k * d + j];
            }
            ctx->buf[j] = s;
        }

        /* Scalar quantize each coordinate */
        for (int j = 0; j < d; j++) {
            float val = ctx->buf[j];
            int best = 0;
            float best_dist = fabsf(val - C[0]);
            for (int c = 1; c < nl; c++) {
                float dist = fabsf(val - C[c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best = c;
                }
            }
            idx[j] = (uint16_t)best;
        }
    }
}

void tq_dequantize(tq_ctx* ctx, const uint16_t* in, float* out, int n_vectors) {
    int d = ctx->dim;
    const float* R = ctx->rotation;  /* R: inverse rotate */
    const float* C = ctx->centroids;

    for (int v = 0; v < n_vectors; v++) {
        const uint16_t* idx = in + v * d;
        float* xhat = out + v * d;

        /* Lookup centroids */
        for (int j = 0; j < d; j++) {
            ctx->buf[j] = C[idx[j]];
        }

        /* Inverse rotate: xhat = buf @ R */
        for (int i = 0; i < d; i++) {
            float s = 0.0f;
            for (int k = 0; k < d; k++) {
                s += ctx->buf[k] * R[k * d + i];
            }
            xhat[i] = s;
        }
    }
}

float tq_mse(tq_ctx* ctx, const float* x, int n_vectors) {
    int d = ctx->dim;
    int total = n_vectors * d;
    uint16_t* idx = (uint16_t*)malloc(total * sizeof(uint16_t));
    float*    rec = (float*)malloc(total * sizeof(float));

    tq_quantize(ctx, x, idx, n_vectors);
    tq_dequantize(ctx, idx, rec, n_vectors);

    double mse = 0.0;
    for (int v = 0; v < n_vectors; v++) {
        double sq = 0.0;
        for (int j = 0; j < d; j++) {
            double diff = (double)x[v * d + j] - (double)rec[v * d + j];
            sq += diff * diff;
        }
        mse += sq;
    }

    free(idx);
    free(rec);
    return (float)(mse / (double)n_vectors);
}

float tq_upper_bound(tq_ctx* ctx) {
    /* Theorem 1: E[‖x − x̂‖²] ≤ (√(3π)/2) · 4^{-b} */
    return sqrtf(3.0f * (float)M_PI) / 2.0f * powf(4.0f, -(float)ctx->bits);
}

float tq_lower_bound(tq_ctx* ctx) {
    /* Theorem 3: no quantizer can do better than 4^{-b} */
    return powf(4.0f, -(float)ctx->bits);
}
