/*
 * turboquant.c — NexaLang C runtime for TurboQuant vector compression.
 *
 * Pure C implementation of TurboQuant-style MSE quantization:
 *   1. Randomized orthogonal transform (SRHT)
 *   2. Precompute Lloyd-Max codebook for Gaussian(0, 1/sqrt(d))
 *   3. Transform -> scalar quantize each coordinate -> indices
 *   4. Lookup centroids -> inverse transform -> reconstruct
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

    /* TurboQuantProd support: (bits-1)-bit MSE base + 1-bit QJL residual */
    int n_levels_prod;
    float* centroids_prod;
    float* boundaries_prod;
    float* qjl_proj;    /* d x d Gaussian projection matrix, row-major */
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

static int tq_is_pow2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

static size_t tq_bits_packed_size(int total_coords, int bits_per_coord) {
    if (total_coords <= 0 || bits_per_coord <= 0) return 0;
    return ((size_t)total_coords * (size_t)bits_per_coord + 7u) >> 3;
}

static inline void tq_pack_value(uint8_t* dst, size_t bit_pos, int bits, uint32_t val) {
    size_t byte = bit_pos >> 3;
    int shift = (int)(bit_pos & 7u);
    uint32_t v = val & ((1u << bits) - 1u);
    dst[byte] |= (uint8_t)(v << shift);
    if (shift + bits > 8) {
        dst[byte + 1] |= (uint8_t)(v >> (8 - shift));
    }
}

static inline uint32_t tq_unpack_value(const uint8_t* src, size_t bit_pos, int bits) {
    size_t byte = bit_pos >> 3;
    int shift = (int)(bit_pos & 7u);
    uint32_t v = (uint32_t)src[byte] >> shift;
    if (shift + bits > 8) {
        v |= (uint32_t)src[byte + 1] << (8 - shift);
    }
    return v & ((1u << bits) - 1u);
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
    (void)log2d;
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
    (void)log2d;
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
    if (!tq_is_pow2(dim)) return NULL;
    if (bits < 1 || bits > 8) return NULL;

    tq_ctx* ctx = (tq_ctx*)calloc(1, sizeof(tq_ctx));
    if (!ctx) return NULL;
    ctx->dim = dim;
    ctx->bits = bits;
    ctx->n_levels = 1 << bits;
    ctx->log2_dim = tq_log2i(dim);

    /* Random sign vector for SRHT */
    ctx->signs = (float*)malloc(dim * sizeof(float));
    if (!ctx->signs) { tq_destroy(ctx); return NULL; }
    tq_rng rng;
    tq_rng_seed(&rng, (uint64_t)seed);
    for (int i = 0; i < dim; i++) {
        ctx->signs[i] = (tq_rng_next(&rng) & 1) ? 1.0f : -1.0f;
    }

    /* Lloyd-Max codebook */
    ctx->centroids = (float*)malloc(ctx->n_levels * sizeof(float));
    if (!ctx->centroids) { tq_destroy(ctx); return NULL; }
    lloyd_max(dim, bits, ctx->centroids);

    /* Pre-compute decision boundaries (midpoints between centroids) */
    ctx->boundaries = (float*)malloc((ctx->n_levels - 1) * sizeof(float));
    if (!ctx->boundaries) { tq_destroy(ctx); return NULL; }
    for (int i = 0; i < ctx->n_levels - 1; i++) {
        ctx->boundaries[i] = 0.5f * (ctx->centroids[i] + ctx->centroids[i + 1]);
    }

    ctx->buf = (float*)malloc(dim * sizeof(float));
    if (!ctx->buf) { tq_destroy(ctx); return NULL; }

    if (bits >= 2) {
        int bits_prod = bits - 1;
        ctx->n_levels_prod = 1 << bits_prod;
        ctx->centroids_prod = (float*)malloc(ctx->n_levels_prod * sizeof(float));
        ctx->boundaries_prod = (float*)malloc((ctx->n_levels_prod - 1) * sizeof(float));
        if (!ctx->centroids_prod || !ctx->boundaries_prod) { tq_destroy(ctx); return NULL; }
        lloyd_max(dim, bits_prod, ctx->centroids_prod);
        for (int i = 0; i < ctx->n_levels_prod - 1; i++) {
            ctx->boundaries_prod[i] = 0.5f * (ctx->centroids_prod[i] + ctx->centroids_prod[i + 1]);
        }
    }

    ctx->qjl_proj = (float*)malloc((size_t)dim * (size_t)dim * sizeof(float));
    if (!ctx->qjl_proj) { tq_destroy(ctx); return NULL; }
    for (int i = 0; i < dim * dim; i++) {
        ctx->qjl_proj[i] = tq_randn(&rng);
    }

    return ctx;
}

void tq_destroy(tq_ctx* ctx) {
    if (!ctx) return;
    free(ctx->signs);
    free(ctx->centroids);
    free(ctx->boundaries);
    free(ctx->buf);
    free(ctx->centroids_prod);
    free(ctx->boundaries_prod);
    free(ctx->qjl_proj);
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

size_t tq_packed_size(const tq_ctx* ctx, int n_vectors) {
    if (!ctx || n_vectors <= 0) return 0;
    return tq_bits_packed_size(n_vectors * ctx->dim, ctx->bits);
}

void tq_quantize_packed(tq_ctx* ctx, const float* in, uint8_t* out, int n_vectors) {
    if (!ctx || !in || !out || n_vectors <= 0) return;
    const int total = n_vectors * ctx->dim;
    size_t out_sz = tq_packed_size(ctx, n_vectors);
    memset(out, 0, out_sz);

    uint16_t* idx = (uint16_t*)malloc((size_t)total * sizeof(uint16_t));
    if (!idx) return;

    tq_quantize(ctx, in, idx, n_vectors);
    for (int i = 0; i < total; i++) {
        tq_pack_value(out, (size_t)i * (size_t)ctx->bits, ctx->bits, (uint32_t)idx[i]);
    }
    free(idx);
}

void tq_dequantize_packed(tq_ctx* ctx, const uint8_t* in, float* out, int n_vectors) {
    if (!ctx || !in || !out || n_vectors <= 0) return;
    const int total = n_vectors * ctx->dim;
    uint16_t* idx = (uint16_t*)malloc((size_t)total * sizeof(uint16_t));
    if (!idx) return;

    for (int i = 0; i < total; i++) {
        idx[i] = (uint16_t)tq_unpack_value(in, (size_t)i * (size_t)ctx->bits, ctx->bits);
    }
    tq_dequantize(ctx, idx, out, n_vectors);
    free(idx);
}

size_t tq_prod_idx_packed_size(const tq_ctx* ctx, int n_vectors) {
    if (!ctx || n_vectors <= 0 || ctx->bits < 2) return 0;
    return tq_bits_packed_size(n_vectors * ctx->dim, ctx->bits - 1);
}

size_t tq_prod_qjl_packed_size(const tq_ctx* ctx, int n_vectors) {
    if (!ctx || n_vectors <= 0) return 0;
    return tq_bits_packed_size(n_vectors * ctx->dim, 1);
}

int tq_quantize_prod(
    tq_ctx* ctx,
    const float* in,
    uint8_t* out_idx_packed,
    uint8_t* out_qjl_packed,
    float* out_gamma,
    int n_vectors
) {
    if (!ctx || !in || !out_idx_packed || !out_qjl_packed || !out_gamma || n_vectors <= 0) return -1;
    if (ctx->bits < 2 || !ctx->centroids_prod || !ctx->boundaries_prod || !ctx->qjl_proj) return -2;

    const int d = ctx->dim;
    const int nb = ctx->n_levels_prod - 1;
    const int bidx = ctx->bits - 1;
    const int log2d = ctx->log2_dim;

    memset(out_idx_packed, 0, tq_prod_idx_packed_size(ctx, n_vectors));
    memset(out_qjl_packed, 0, tq_prod_qjl_packed_size(ctx, n_vectors));

    float* rot = (float*)malloc((size_t)d * sizeof(float));
    float* xhat = (float*)malloc((size_t)d * sizeof(float));
    float* res = (float*)malloc((size_t)d * sizeof(float));
    if (!rot || !xhat || !res) {
        free(rot); free(xhat); free(res);
        return -3;
    }

    for (int v = 0; v < n_vectors; v++) {
        const float* x = in + (size_t)v * (size_t)d;

        memcpy(rot, x, (size_t)d * sizeof(float));
        srht_forward(ctx->signs, rot, d, log2d);

        for (int j = 0; j < d; j++) {
            float val = rot[j];
            int level = 0;
            for (int b = 0; b < nb; b++) level += (val > ctx->boundaries_prod[b]);
            tq_pack_value(out_idx_packed, ((size_t)v * (size_t)d + (size_t)j) * (size_t)bidx, bidx, (uint32_t)level);
            xhat[j] = ctx->centroids_prod[level];
        }

        srht_inverse(ctx->signs, xhat, d, log2d);

        float gamma_sq = 0.0f;
        for (int j = 0; j < d; j++) {
            float r = x[j] - xhat[j];
            res[j] = r;
            gamma_sq += r * r;
        }
        float gamma = sqrtf(gamma_sq);
        out_gamma[v] = gamma;

        for (int i = 0; i < d; i++) {
            const float* row = ctx->qjl_proj + (size_t)i * (size_t)d;
            float dot = 0.0f;
            for (int j = 0; j < d; j++) dot += row[j] * res[j];
            uint32_t bit = (dot >= 0.0f) ? 1u : 0u;
            tq_pack_value(out_qjl_packed, (size_t)v * (size_t)d + (size_t)i, 1, bit);
        }
    }

    free(rot); free(xhat); free(res);
    return 0;
}

int tq_dequantize_prod(
    tq_ctx* ctx,
    const uint8_t* in_idx_packed,
    const uint8_t* in_qjl_packed,
    const float* in_gamma,
    float* out,
    int n_vectors
) {
    if (!ctx || !in_idx_packed || !in_qjl_packed || !in_gamma || !out || n_vectors <= 0) return -1;
    if (ctx->bits < 2 || !ctx->centroids_prod || !ctx->qjl_proj) return -2;

    const int d = ctx->dim;
    const int bidx = ctx->bits - 1;
    const int log2d = ctx->log2_dim;
    const float qjl_scale = (float)M_PI / (2.0f * (float)d);

    float* xhat = (float*)malloc((size_t)d * sizeof(float));
    float* z = (float*)malloc((size_t)d * sizeof(float));
    if (!xhat || !z) {
        free(xhat); free(z);
        return -3;
    }

    for (int v = 0; v < n_vectors; v++) {
        for (int j = 0; j < d; j++) {
            uint32_t level = tq_unpack_value(in_idx_packed, ((size_t)v * (size_t)d + (size_t)j) * (size_t)bidx, bidx);
            xhat[j] = ctx->centroids_prod[level];
        }
        srht_inverse(ctx->signs, xhat, d, log2d);

        for (int i = 0; i < d; i++) {
            uint32_t bit = tq_unpack_value(in_qjl_packed, (size_t)v * (size_t)d + (size_t)i, 1);
            z[i] = bit ? 1.0f : -1.0f;
        }

        float* out_v = out + (size_t)v * (size_t)d;
        for (int j = 0; j < d; j++) {
            float stz = 0.0f;
            for (int i = 0; i < d; i++) {
                stz += ctx->qjl_proj[(size_t)i * (size_t)d + (size_t)j] * z[i];
            }
            float corr = in_gamma[v] * qjl_scale * stz;
            out_v[j] = xhat[j] + corr;
        }
    }

    free(xhat); free(z);
    return 0;
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
