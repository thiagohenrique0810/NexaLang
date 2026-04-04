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
#include <pthread.h>

/* SIMD: ARM NEON on Apple Silicon / aarch64 */
#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define TQ_HAS_NEON 1
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define TQ_HAS_SSE 1
#endif

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
    float* signs;       /* dim random ±1 signs for SRHT */
    float* centroids;   /* n_levels centroids from Lloyd-Max */
    float* boundaries;  /* n_levels-1 decision boundaries (midpoints) */
    float* buf;         /* scratch buffer, dim floats */
    int log2_dim;       /* log2(dim) for Hadamard stages */
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
/*  Fast Walsh-Hadamard Transform (in-place, O(d log d))                   */
/*  Replaces dense O(d²) rotation with Subsampled Randomized Hadamard.     */
/* ──────────────────────────────────────────────────────────────────────── */

static int tq_log2i(int n) {
    int r = 0;
    while ((1 << r) < n) r++;
    return r;
}

/* In-place unnormalized Walsh-Hadamard transform.  d must be power of 2. */
static void fwht_inplace(float* x, int d) {
    for (int half = 1; half < d; half <<= 1) {
        for (int i = 0; i < d; i += half << 1) {
            for (int j = i; j < i + half; j++) {
                float a = x[j];
                float b = x[j + half];
                x[j]        = a + b;
                x[j + half]  = a - b;
            }
        }
    }
}

#if TQ_HAS_NEON
/* NEON-accelerated FWHT for half >= 4 */
static void fwht_inplace_neon(float* x, int d) {
    /* Small stages: scalar */
    for (int half = 1; half < 4; half <<= 1) {
        for (int i = 0; i < d; i += half << 1) {
            for (int j = i; j < i + half; j++) {
                float a = x[j], b = x[j + half];
                x[j] = a + b;
                x[j + half] = a - b;
            }
        }
    }
    /* Larger stages: NEON */
    for (int half = 4; half < d; half <<= 1) {
        for (int i = 0; i < d; i += half << 1) {
            for (int j = i; j < i + half; j += 4) {
                float32x4_t a = vld1q_f32(&x[j]);
                float32x4_t b = vld1q_f32(&x[j + half]);
                vst1q_f32(&x[j],        vaddq_f32(a, b));
                vst1q_f32(&x[j + half], vsubq_f32(a, b));
            }
        }
    }
}
#define FWHT(x, d) fwht_inplace_neon(x, d)
#else
#define FWHT(x, d) fwht_inplace(x, d)
#endif

/* Apply random sign flip + FWHT + normalize = randomized Hadamard rotation */
static void srht_forward(const float* restrict signs, float* restrict buf, int d, int log2d) {
    /* Step 1: multiply by random diagonal D (±1) */
    #if TQ_HAS_NEON
    for (int i = 0; i + 4 <= d; i += 4) {
        float32x4_t b = vld1q_f32(&buf[i]);
        float32x4_t s = vld1q_f32(&signs[i]);
        vst1q_f32(&buf[i], vmulq_f32(b, s));
    }
    #else
    for (int i = 0; i < d; i++) buf[i] *= signs[i];
    #endif

    /* Step 2: Walsh-Hadamard transform */
    FWHT(buf, d);

    /* Step 3: normalize by 1/sqrt(d) */
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    #if TQ_HAS_NEON
    float32x4_t norm = vdupq_n_f32(inv_sqrt_d);
    for (int i = 0; i + 4 <= d; i += 4) {
        float32x4_t b = vld1q_f32(&buf[i]);
        vst1q_f32(&buf[i], vmulq_f32(b, norm));
    }
    #else
    for (int i = 0; i < d; i++) buf[i] *= inv_sqrt_d;
    #endif
}

/* Inverse = same operation (Hadamard is self-inverse, D² = I) */
static void srht_inverse(const float* restrict signs, float* restrict buf, int d, int log2d) {
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    #if TQ_HAS_NEON
    float32x4_t norm = vdupq_n_f32(inv_sqrt_d);
    for (int i = 0; i + 4 <= d; i += 4) {
        float32x4_t b = vld1q_f32(&buf[i]);
        vst1q_f32(&buf[i], vmulq_f32(b, norm));
    }
    #else
    for (int i = 0; i < d; i++) buf[i] *= inv_sqrt_d;
    #endif

    FWHT(buf, d);

    #if TQ_HAS_NEON
    for (int i = 0; i + 4 <= d; i += 4) {
        float32x4_t b = vld1q_f32(&buf[i]);
        float32x4_t s = vld1q_f32(&signs[i]);
        vst1q_f32(&buf[i], vmulq_f32(b, s));
    }
    #else
    for (int i = 0; i < d; i++) buf[i] *= signs[i];
    #endif
}

/* ──────────────────────────────────────────────────────────────────────── */
/*  Public API                                                             */
/* ──────────────────────────────────────────────────────────────────────── */

tq_ctx* tq_create(int dim, int bits, int seed) {
    tq_ctx* ctx = (tq_ctx*)calloc(1, sizeof(tq_ctx));
    ctx->dim = dim;
    ctx->bits = bits;
    ctx->n_levels = 1 << bits;
    ctx->log2_dim = tq_log2i(dim);

    /* Random sign vector for SRHT */
    ctx->signs = (float*)malloc(dim * sizeof(float));
    tq_rng rng;
    tq_rng_seed(&rng, (uint64_t)seed);
    for (int i = 0; i < dim; i++) {
        ctx->signs[i] = (tq_rng_next(&rng) & 1) ? 1.0f : -1.0f;
    }

    /* Lloyd-Max codebook */
    ctx->centroids = (float*)malloc(ctx->n_levels * sizeof(float));
    lloyd_max(dim, bits, ctx->centroids);

    /* Pre-compute decision boundaries (midpoints between centroids) */
    ctx->boundaries = (float*)malloc((ctx->n_levels - 1) * sizeof(float));
    for (int i = 0; i < ctx->n_levels - 1; i++) {
        ctx->boundaries[i] = 0.5f * (ctx->centroids[i] + ctx->centroids[i + 1]);
    }

    ctx->buf = (float*)malloc(dim * sizeof(float));
    return ctx;
}

void tq_destroy(tq_ctx* ctx) {
    if (!ctx) return;
    free(ctx->signs);
    free(ctx->centroids);
    free(ctx->boundaries);
    free(ctx->buf);
    free(ctx);
}

void tq_quantize(tq_ctx* ctx, const float* restrict in, uint16_t* restrict out, int n_vectors) {
    const int d = ctx->dim;
    const int nl = ctx->n_levels;
    const int nb = nl - 1;
    const float* restrict signs = ctx->signs;
    const float* restrict B = ctx->boundaries;
    const int log2d = ctx->log2_dim;

    float buf[d];

    for (int v = 0; v < n_vectors; v++) {
        const float* restrict x = in + v * d;
        uint16_t* restrict idx = out + v * d;

        /* Copy input to buf */
        memcpy(buf, x, d * sizeof(float));

        /* SRHT forward: D·H·x / sqrt(d) */
        srht_forward(signs, buf, d, log2d);

        /* Quantize: scan boundaries (branchless-friendly for small nl) */
        for (int j = 0; j < d; j++) {
            float val = buf[j];
            int level = 0;
            for (int b = 0; b < nb; b++) {
                level += (val > B[b]);
            }
            idx[j] = (uint16_t)level;
        }
    }
}

void tq_dequantize(tq_ctx* ctx, const uint16_t* restrict in, float* restrict out, int n_vectors) {
    const int d = ctx->dim;
    const float* restrict signs = ctx->signs;
    const float* restrict C = ctx->centroids;
    const int log2d = ctx->log2_dim;

    for (int v = 0; v < n_vectors; v++) {
        const uint16_t* restrict idx = in + v * d;
        float* restrict xhat = out + v * d;

        /* Lookup centroids into output buffer directly */
        for (int j = 0; j < d; j++) {
            xhat[j] = C[idx[j]];
        }

        /* SRHT inverse: H·D·x / sqrt(d) = inverse rotation */
        srht_inverse(signs, xhat, d, log2d);
    }
}

float tq_mse(tq_ctx* ctx, const float* x, int n_vectors) {
    const int d = ctx->dim;
    const int total = n_vectors * d;
    uint16_t* idx = (uint16_t*)malloc(total * sizeof(uint16_t));
    float*    rec = (float*)malloc(total * sizeof(float));

    tq_quantize(ctx, x, idx, n_vectors);
    tq_dequantize(ctx, idx, rec, n_vectors);

    double mse = 0.0;
    for (int v = 0; v < n_vectors; v++) {
        const float* xv = x + v * d;
        const float* rv = rec + v * d;
#if TQ_HAS_NEON
        float32x4_t acc = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j + 4 <= d; j += 4) {
            float32x4_t a = vld1q_f32(&xv[j]);
            float32x4_t b = vld1q_f32(&rv[j]);
            float32x4_t diff = vsubq_f32(a, b);
            acc = vfmaq_f32(acc, diff, diff);
        }
        float sq = vaddvq_f32(acc);
        for (; j < d; j++) {
            float diff = xv[j] - rv[j];
            sq += diff * diff;
        }
        mse += (double)sq;
#else
        double sq = 0.0;
        for (int j = 0; j < d; j++) {
            double diff = (double)xv[j] - (double)rv[j];
            sq += diff * diff;
        }
        mse += sq;
#endif
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

/* ──────────────────────────────────────────────────────────────────────── */
/*  Parallel quantize / dequantize via pthreads                            */
/* ──────────────────────────────────────────────────────────────────────── */

#define TQ_MAX_THREADS 8

typedef struct {
    tq_ctx*         ctx;
    const float*    in_f;
    const uint16_t* in_u;
    float*          out_f;
    uint16_t*       out_u;
    int             start;
    int             count;
} tq_thread_arg;

static void* tq_quantize_worker(void* arg) {
    tq_thread_arg* a = (tq_thread_arg*)arg;
    tq_ctx* ctx = a->ctx;
    const int d = ctx->dim;

    /* Each thread needs its own local context copy for thread-safe buf */
    const int nl = ctx->n_levels;
    const int nb = nl - 1;
    const float* restrict signs = ctx->signs;
    const float* restrict B = ctx->boundaries;
    const int log2d = ctx->log2_dim;

    float* buf = (float*)malloc(d * sizeof(float));

    const float* restrict in = a->in_f + a->start * d;
    uint16_t* restrict out = a->out_u + a->start * d;

    for (int v = 0; v < a->count; v++) {
        const float* restrict x = in + v * d;
        uint16_t* restrict idx = out + v * d;

        memcpy(buf, x, d * sizeof(float));
        srht_forward(signs, buf, d, log2d);

        for (int j = 0; j < d; j++) {
            float val = buf[j];
            int level = 0;
            for (int b = 0; b < nb; b++) {
                level += (val > B[b]);
            }
            idx[j] = (uint16_t)level;
        }
    }

    free(buf);
    return NULL;
}

static void* tq_dequantize_worker(void* arg) {
    tq_thread_arg* a = (tq_thread_arg*)arg;
    tq_ctx* ctx = a->ctx;
    const int d = ctx->dim;
    const float* restrict signs = ctx->signs;
    const float* restrict C = ctx->centroids;
    const int log2d = ctx->log2_dim;

    const uint16_t* restrict in = a->in_u + a->start * d;
    float* restrict out = a->out_f + a->start * d;

    for (int v = 0; v < a->count; v++) {
        const uint16_t* restrict idx = in + v * d;
        float* restrict xhat = out + v * d;

        for (int j = 0; j < d; j++) {
            xhat[j] = C[idx[j]];
        }

        srht_inverse(signs, xhat, d, log2d);
    }

    return NULL;
}

void tq_quantize_parallel(tq_ctx* ctx, const float* in, uint16_t* out, int n_vectors) {
    if (n_vectors < 1000) {
        tq_quantize(ctx, in, out, n_vectors);
        return;
    }

    int n_threads = TQ_MAX_THREADS;
    if (n_vectors < n_threads * 100) n_threads = 2;

    pthread_t threads[TQ_MAX_THREADS];
    tq_thread_arg args[TQ_MAX_THREADS];

    int per_thread = n_vectors / n_threads;
    int remainder  = n_vectors % n_threads;

    int offset = 0;
    for (int t = 0; t < n_threads; t++) {
        args[t].ctx   = ctx;
        args[t].in_f  = in;
        args[t].out_u = out;
        args[t].start = offset;
        args[t].count = per_thread + (t < remainder ? 1 : 0);
        offset += args[t].count;
        pthread_create(&threads[t], NULL, tq_quantize_worker, &args[t]);
    }

    for (int t = 0; t < n_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}

void tq_dequantize_parallel(tq_ctx* ctx, const uint16_t* in, float* out, int n_vectors) {
    if (n_vectors < 1000) {
        tq_dequantize(ctx, in, out, n_vectors);
        return;
    }

    int n_threads = TQ_MAX_THREADS;
    if (n_vectors < n_threads * 100) n_threads = 2;

    pthread_t threads[TQ_MAX_THREADS];
    tq_thread_arg args[TQ_MAX_THREADS];

    int per_thread = n_vectors / n_threads;
    int remainder  = n_vectors % n_threads;

    int offset = 0;
    for (int t = 0; t < n_threads; t++) {
        args[t].ctx   = ctx;
        args[t].in_u  = in;
        args[t].out_f = out;
        args[t].start = offset;
        args[t].count = per_thread + (t < remainder ? 1 : 0);
        offset += args[t].count;
        pthread_create(&threads[t], NULL, tq_dequantize_worker, &args[t]);
    }

    for (int t = 0; t < n_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}
