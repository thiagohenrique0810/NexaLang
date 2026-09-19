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
#ifndef _WIN32
#include <pthread.h>
#endif
#include <limits.h>
#include <float.h>

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
    float* buf;         /* legacy reserved buffer; NULL in an MSE-only context */
    int log2_dim;       /* log2(dim) for Hadamard stages */

    /* TurboQuantProd support: (bits-1)-bit MSE base + 1-bit QJL residual */
    int n_levels_prod;
    float* centroids_prod;
    float* boundaries_prod;
    float* qjl_proj;    /* d x d Gaussian projection matrix, row-major */
    size_t memory_bytes; /* owned persistent allocation requests, including ctx */
};

_Static_assert(sizeof(tq_ctx) <= TQ_MSE_CONTEXT_STRUCT_RESERVE_BYTES,
               "update the public TQ context reserve before changing the ABI");

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

static int lloyd_max(int dim, int bits, float* centroids) {
    int n_levels = 1 << bits;
    float sigma = 1.0f / sqrtf((float)dim);
    float lo = -5.0f * sigma;
    float hi =  5.0f * sigma;

    /* Initialize centroids uniformly */
    for (int i = 0; i < n_levels; i++) {
        centroids[i] = lo + (hi - lo) * ((float)i + 0.5f) / (float)n_levels;
    }

    float* bounds = (float*)malloc((n_levels + 1) * sizeof(float));
    if (!bounds) return -1;

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
    return 0;
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

static size_t tq_bits_packed_size(size_t total_coords, int bits_per_coord) {
    if (!total_coords || bits_per_coord <= 0 ||
        total_coords > (SIZE_MAX - 7u) / (size_t)bits_per_coord) return 0;
    return (total_coords * (size_t)bits_per_coord + 7u) >> 3;
}

static int tq_valid_count(const tq_ctx* ctx, int n_vectors) {
    return ctx && n_vectors > 0 &&
        (size_t)n_vectors <= SIZE_MAX / sizeof(float) / (size_t)ctx->dim;
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
    for (int half = 1; half < 4 && half < d; half <<= 1) {
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
    if (d < 4) {
        for (int i = 0; i < d; i++) buf[i] *= signs[i];
        fwht_inplace(buf, d);
        for (int i = 0; i < d; i++) buf[i] /= sqrtf((float)d);
        return;
    }
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

/* Inverse reverses the order: H is self-inverse after normalization, D² = I. */
static void srht_inverse(const float* restrict signs, float* restrict buf, int d, int log2d) {
    (void)log2d;
    if (d < 4) {
        fwht_inplace(buf, d);
        for (int i = 0; i < d; i++) buf[i] *= signs[i] / sqrtf((float)d);
        return;
    }
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

static int tq_add_float_bytes(size_t* total, size_t count) {
    if (count > (SIZE_MAX - *total) / sizeof(float)) return 0;
    *total += count * sizeof(float);
    return 1;
}

/* Admission checks the complete persistent allocation sum before any malloc.
 * The MSE path deliberately never computes or validates a quadratic size. */
static int tq_context_size(int dim, int bits, int with_prod, size_t* total) {
    if (!tq_is_pow2(dim) || bits < 1 || bits > 8) return 0;
    const size_t d = (size_t)dim;
    const size_t levels = (size_t)1u << bits;
    *total = sizeof(tq_ctx);
    if (!tq_add_float_bytes(total, d) ||
        !tq_add_float_bytes(total, levels) ||
        !tq_add_float_bytes(total, levels - 1u)) return 0;
    if (with_prod) {
        if (!tq_add_float_bytes(total, d)) return 0;
        if (bits >= 2) {
            const size_t prod_levels = levels >> 1;
            if (!tq_add_float_bytes(total, prod_levels) ||
                !tq_add_float_bytes(total, prod_levels - 1u)) return 0;
        }
        if (d > SIZE_MAX / d || !tq_add_float_bytes(total, d * d)) return 0;
    }
    return 1;
}

static int tq_valid_codebook(const float* centroids, size_t count, int bits) {
    if (!centroids || bits < 1 || bits > 8 || count != (size_t)(1u << bits)) return 0;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(centroids[i])) return 0;
        if (i && (!(centroids[i - 1] < centroids[i]) ||
                  !isfinite(centroids[i - 1] + centroids[i]))) return 0;
    }
    return 1;
}

static tq_ctx* tq_create_context(int dim, int bits, int seed, int with_prod,
                                 const float* imported_centroids) {
    size_t memory_bytes;
    if (!tq_context_size(dim, bits, with_prod, &memory_bytes)) return NULL;

    tq_ctx* ctx = (tq_ctx*)calloc(1, sizeof(tq_ctx));
    if (!ctx) return NULL;
    ctx->dim = dim;
    ctx->bits = bits;
    ctx->n_levels = 1 << bits;
    ctx->log2_dim = tq_log2i(dim);
    ctx->memory_bytes = memory_bytes;

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
    if (imported_centroids) {
        memcpy(ctx->centroids, imported_centroids, (size_t)ctx->n_levels * sizeof(float));
    } else if (lloyd_max(dim, bits, ctx->centroids)) { tq_destroy(ctx); return NULL; }

    /* Pre-compute decision boundaries (midpoints between centroids) */
    ctx->boundaries = (float*)malloc((ctx->n_levels - 1) * sizeof(float));
    if (!ctx->boundaries) { tq_destroy(ctx); return NULL; }
    for (int i = 0; i < ctx->n_levels - 1; i++) {
        ctx->boundaries[i] = 0.5f * (ctx->centroids[i] + ctx->centroids[i + 1]);
    }

    if (with_prod) {
        ctx->buf = (float*)malloc(dim * sizeof(float));
        if (!ctx->buf) { tq_destroy(ctx); return NULL; }
    }

    if (with_prod && bits >= 2) {
        int bits_prod = bits - 1;
        ctx->n_levels_prod = 1 << bits_prod;
        ctx->centroids_prod = (float*)malloc(ctx->n_levels_prod * sizeof(float));
        ctx->boundaries_prod = (float*)malloc((ctx->n_levels_prod - 1) * sizeof(float));
        if (!ctx->centroids_prod || !ctx->boundaries_prod) { tq_destroy(ctx); return NULL; }
        if (lloyd_max(dim, bits_prod, ctx->centroids_prod)) { tq_destroy(ctx); return NULL; }
        for (int i = 0; i < ctx->n_levels_prod - 1; i++) {
            ctx->boundaries_prod[i] = 0.5f * (ctx->centroids_prod[i] + ctx->centroids_prod[i + 1]);
        }
    }

    if (with_prod) {
        ctx->qjl_proj = (float*)malloc((size_t)dim * (size_t)dim * sizeof(float));
        if (!ctx->qjl_proj) { tq_destroy(ctx); return NULL; }
        for (size_t i = 0; i < (size_t)dim * (size_t)dim; i++) {
            ctx->qjl_proj[i] = tq_randn(&rng);
        }
    }

    return ctx;
}

tq_ctx* tq_create(int dim, int bits, int seed) {
    return tq_create_context(dim, bits, seed, 1, NULL);
}

tq_ctx* tq_create_mse(int dim, int bits, int seed) {
    return tq_create_context(dim, bits, seed, 0, NULL);
}

size_t tq_mse_context_memory_size(int dim, int bits) {
    size_t bytes = 0;
    return tq_context_size(dim, bits, 0, &bytes) ? bytes : 0;
}

tq_ctx* tq_create_mse_from_codebook(int dim, int bits, int seed,
                                   const float* centroids, size_t count) {
    if (!tq_valid_codebook(centroids, count, bits)) return NULL;
    return tq_create_context(dim, bits, seed, 0, centroids);
}

size_t tq_context_memory_bytes(const tq_ctx* ctx) {
    return ctx ? ctx->memory_bytes : 0;
}

int tq_context_dim(const tq_ctx* ctx) { return ctx ? ctx->dim : 0; }
int tq_context_bits(const tq_ctx* ctx) { return ctx ? ctx->bits : 0; }

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
    if (!tq_valid_count(ctx, n_vectors) || !in || !out) return;
    const int d = ctx->dim;
    const int nl = ctx->n_levels;
    const int nb = nl - 1;
    const float* restrict signs = ctx->signs;
    const float* restrict B = ctx->boundaries;
    const int log2d = ctx->log2_dim;

    float* buf = (float*)malloc((size_t)d * sizeof(float));
    if (!buf) return;

    for (int v = 0; v < n_vectors; v++) {
        const float* restrict x = in + (size_t)v * (size_t)d;
        uint16_t* restrict idx = out + (size_t)v * (size_t)d;

        /* Copy input to buf */
        memcpy(buf, x, d * sizeof(float));

        /* SRHT forward: H·D·x / sqrt(d) */
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
    free(buf);
}

void tq_dequantize(tq_ctx* ctx, const uint16_t* restrict in, float* restrict out, int n_vectors) {
    if (!tq_valid_count(ctx, n_vectors) || !in || !out) return;
    const int d = ctx->dim;
    const float* restrict signs = ctx->signs;
    const float* restrict C = ctx->centroids;
    const int log2d = ctx->log2_dim;

    for (int v = 0; v < n_vectors; v++) {
        const uint16_t* restrict idx = in + (size_t)v * (size_t)d;
        float* restrict xhat = out + (size_t)v * (size_t)d;

        /* Lookup centroids into output buffer directly */
        for (int j = 0; j < d; j++) {
            xhat[j] = C[idx[j] < ctx->n_levels ? idx[j] : 0];
        }

        /* SRHT inverse: D·H·x / sqrt(d). */
        srht_inverse(signs, xhat, d, log2d);
    }
}

float tq_mse(tq_ctx* ctx, const float* x, int n_vectors) {
    if (!tq_valid_count(ctx, n_vectors) || !x) return NAN;
    size_t total = (size_t)n_vectors * (size_t)ctx->dim;
    size_t packed_size = tq_packed_size(ctx, n_vectors);
    if (!packed_size) return NAN;
    uint8_t* packed = (uint8_t*)malloc(packed_size);
    float* rec = (float*)malloc(total * sizeof(float));
    if (!packed || !rec) { free(packed); free(rec); return NAN; }
    if (tq_quantize_packed(ctx, x, packed, n_vectors) ||
        tq_dequantize_packed(ctx, packed, rec, n_vectors)) {
        free(packed); free(rec); return NAN;
    }
    double mse = 0.0;
    for (size_t i = 0; i < total; i++) {
        double diff = (double)x[i] - (double)rec[i];
        mse += diff * diff;
    }
    free(packed); free(rec);
    return (float)(mse / (double)n_vectors);
}

float tq_upper_bound(tq_ctx* ctx) {
    /* Theorem 1: E[‖x − x̂‖²] ≤ (√(3π)/2) · 4^{-b} */
    if (!ctx) return NAN;
    return sqrtf(3.0f * (float)M_PI) / 2.0f * powf(4.0f, -(float)ctx->bits);
}

float tq_lower_bound(tq_ctx* ctx) {
    /* Theorem 3: no quantizer can do better than 4^{-b} */
    if (!ctx) return NAN;
    return powf(4.0f, -(float)ctx->bits);
}

/* Versioned records: TQ01, a host-endian float32 norm, then indices. */
#define TQ_RECORD_HEADER 8u
static const uint8_t tq_record_magic[4] = {'T', 'Q', '0', '1'};
size_t tq_packed_size(const tq_ctx* ctx, int n_vectors) {
    if (!tq_valid_count(ctx, n_vectors)) return 0;
    size_t indices = tq_bits_packed_size((size_t)ctx->dim, ctx->bits);
    if (!indices || indices > SIZE_MAX - TQ_RECORD_HEADER) return 0;
    size_t stride = TQ_RECORD_HEADER + indices;
    if ((size_t)n_vectors > SIZE_MAX / stride) return 0;
    return (size_t)n_vectors * stride;
}

int tq_quantize_packed(tq_ctx* ctx, const float* in, uint8_t* out, int n_vectors) {
    size_t stride = tq_packed_size(ctx, 1);
    if (!in || !out || !stride || !tq_packed_size(ctx, n_vectors)) return -1;
    size_t d = (size_t)ctx->dim;
    float* buf = (float*)malloc((size_t)d * sizeof(float));
    if (!buf) return -2;
    for (int v = 0; v < n_vectors; v++) {
        const float* x = in + (size_t)v * d;
        uint8_t* record = out + (size_t)v * stride;
        double sum = 0.0;
        for (size_t j = 0; j < d; j++) {
            if (!isfinite(x[j])) { free(buf); return -1; }
            sum += (double)x[j] * (double)x[j];
        }
        double norm64 = sqrt(sum);
        if (norm64 > FLT_MAX) { free(buf); return -1; }
        float norm = (float)norm64;
        memcpy(record, tq_record_magic, 4);
        memcpy(record + 4, &norm, sizeof(norm));
        memset(record + TQ_RECORD_HEADER, 0, stride - TQ_RECORD_HEADER);
        if (norm == 0.0f) continue;
        for (size_t j = 0; j < d; j++) buf[j] = (float)((double)x[j] / norm64);
        srht_forward(ctx->signs, buf, ctx->dim, ctx->log2_dim);
        for (size_t j = 0; j < d; j++) {
            int level = 0;
            for (int b = 0; b < ctx->n_levels - 1; b++) level += buf[j] > ctx->boundaries[b];
            tq_pack_value(record + TQ_RECORD_HEADER, j * (size_t)ctx->bits, ctx->bits, (uint32_t)level);
        }
    }
    free(buf);
    return 0;
}

int tq_dequantize_packed(tq_ctx* ctx, const uint8_t* in, float* out, int n_vectors) {
    size_t stride = tq_packed_size(ctx, 1);
    if (!in || !out || !stride || !tq_packed_size(ctx, n_vectors)) return -1;
    size_t d = (size_t)ctx->dim;
    for (int v = 0; v < n_vectors; v++) {
        const uint8_t* record = in + (size_t)v * stride;
        float* xhat = out + (size_t)v * d;
        float norm;
        if (memcmp(record, tq_record_magic, 4) != 0) return -1;
        memcpy(&norm, record + 4, sizeof(norm));
        if (!isfinite(norm) || norm < 0.0f) return -1;
        if (norm == 0.0f) { memset(xhat, 0, d * sizeof(float)); continue; }
        for (size_t j = 0; j < d; j++) {
            uint32_t level = tq_unpack_value(record + TQ_RECORD_HEADER, j * (size_t)ctx->bits, ctx->bits);
            xhat[j] = ctx->centroids[level];
        }
        srht_inverse(ctx->signs, xhat, ctx->dim, ctx->log2_dim);
        for (size_t j = 0; j < d; j++) {
            double value = (double)xhat[j] * (double)norm;
            if (!isfinite(value) || fabs(value) > FLT_MAX) return -1;
            xhat[j] = (float)value;
        }
    }
    return 0;
}

/* Portable records. Integer shifts define byte order even on big-endian hosts. */
static uint32_t tq_read_le32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void tq_write_le32(uint8_t* p, uint32_t value) {
    for (unsigned i = 0; i < 4; i++) p[i] = (uint8_t)(value >> (8u * i));
}

/* A wrapped pointer interval is invalid and therefore always conflicts. */
static int tq_overlap(const void* a, size_t an, const void* b, size_t bn) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    if (an > UINTPTR_MAX - x || bn > UINTPTR_MAX - y) return 1;
    return an && bn && x < y + bn && y < x + an;
}

static int tq_overlaps_context(const tq_ctx* ctx, const void* p, size_t bytes) {
    size_t d = (size_t)ctx->dim, levels = (size_t)ctx->n_levels;
    if (tq_overlap(ctx, sizeof(*ctx), p, bytes) ||
        tq_overlap(ctx->signs, d * sizeof(float), p, bytes) ||
        tq_overlap(ctx->centroids, levels * sizeof(float), p, bytes) ||
        tq_overlap(ctx->boundaries, (levels - 1u) * sizeof(float), p, bytes)) return 1;
    if (ctx->buf && tq_overlap(ctx->buf, d * sizeof(float), p, bytes)) return 1;
    if (ctx->qjl_proj && tq_overlap(ctx->qjl_proj, d * d * sizeof(float), p, bytes)) return 1;
    if (ctx->centroids_prod && tq_overlap(ctx->centroids_prod,
            (size_t)ctx->n_levels_prod * sizeof(float), p, bytes)) return 1;
    if (ctx->boundaries_prod && tq_overlap(ctx->boundaries_prod,
            (size_t)(ctx->n_levels_prod - 1) * sizeof(float), p, bytes)) return 1;
    return 0;
}

int tq_context_buffer_disjoint(const tq_ctx* ctx, const void* data, size_t bytes) {
    return ctx && data && bytes <= UINTPTR_MAX - (uintptr_t)data &&
           !tq_overlaps_context(ctx, data, bytes);
}

int tq_export_mse_codebook(const tq_ctx* ctx, float* out, size_t count) {
    if (!ctx || !out) return -1;
    if (count != (size_t)ctx->n_levels) return -2;
    if (tq_overlaps_context(ctx, out, count * sizeof(float))) return -3;
    memcpy(out, ctx->centroids, count * sizeof(float));
    return 0;
}

static int tq_tq02_sizes(const tq_ctx* ctx, int n, size_t packed_bytes,
                         size_t float_count, size_t* total, size_t* stride) {
    if (sizeof(float) != 4 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128) return -1;
    if (!ctx || n <= 0) return -1;
    size_t required = tq_packed_size(ctx, n);
    if (!required || packed_bytes != required) return -2;
    *total = (size_t)n * (size_t)ctx->dim;
    if (float_count < *total) return -2;
    *stride = tq_packed_size(ctx, 1);
    return 0;
}

int tq_quantize_tq02(const tq_ctx* ctx, const float* in, size_t in_count,
                    uint8_t* packed, size_t packed_bytes, int n_vectors,
                    float* scratch, size_t scratch_count) {
    if (!in || !packed || !scratch) return -1;
    size_t total, stride;
    int status = tq_tq02_sizes(ctx, n_vectors, packed_bytes, in_count, &total, &stride);
    if (status) return status;
    size_t d = (size_t)ctx->dim, input_bytes = total * sizeof(float);
    size_t scratch_bytes = d * sizeof(float);
    if (scratch_count < d) return -2;
    if (tq_overlap(in, input_bytes, packed, packed_bytes) ||
        tq_overlap(in, input_bytes, scratch, scratch_bytes) ||
        tq_overlap(packed, packed_bytes, scratch, scratch_bytes) ||
        tq_overlaps_context(ctx, packed, packed_bytes) ||
        tq_overlaps_context(ctx, scratch, scratch_bytes) ||
        tq_overlaps_context(ctx, in, input_bytes)) return -3;
    for (int v = 0; v < n_vectors; v++) {
        const float* x = in + (size_t)v * d;
        uint8_t* record = packed + (size_t)v * stride;
        double sum = 0.0;
        for (size_t j = 0; j < d; j++) {
            if (!isfinite(x[j])) return -4;
            sum += (double)x[j] * (double)x[j];
        }
        double norm64 = sqrt(sum);
        if (!isfinite(norm64) || norm64 > FLT_MAX) return -4;
        float norm = (float)norm64;
        uint32_t norm_bits;
        memcpy(&norm_bits, &norm, sizeof(norm_bits));
        memcpy(record, "TQ02", 4);
        tq_write_le32(record + 4, norm_bits);
        memset(record + TQ_RECORD_HEADER, 0, stride - TQ_RECORD_HEADER);
        if (norm == 0.0f) continue;
        for (size_t j = 0; j < d; j++) scratch[j] = (float)((double)x[j] / norm64);
        srht_forward(ctx->signs, scratch, ctx->dim, ctx->log2_dim);
        for (size_t j = 0; j < d; j++) {
            int level = 0;
            for (int b = 0; b < ctx->n_levels - 1; b++) level += scratch[j] > ctx->boundaries[b];
            tq_pack_value(record + TQ_RECORD_HEADER, j * (size_t)ctx->bits,
                          ctx->bits, (uint32_t)level);
        }
    }
    return 0;
}

static int tq_validate_tq02_record(const uint8_t* record, size_t stride,
                                  size_t coordinates, int bits) {
    if (memcmp(record, "TQ02", 4)) return 0;
    uint32_t norm_bits = tq_read_le32(record + 4);
    /* Negative zero, negatives and every nonfinite encoding are forbidden. */
    if ((norm_bits & 0x80000000u) || (norm_bits & 0x7f800000u) == 0x7f800000u) return 0;
    unsigned tail_bits = (unsigned)((coordinates * (size_t)bits) & 7u);
    if (tail_bits && (record[stride - 1u] >> tail_bits)) return 0;
    if (!norm_bits) {
        for (size_t j = TQ_RECORD_HEADER; j < stride; j++) if (record[j]) return 0;
    }
    return 1;
}

int tq_dequantize_tq02(const tq_ctx* ctx, const uint8_t* packed,
                      size_t packed_bytes, float* out, size_t out_count,
                      int n_vectors) {
    if (!packed || !out) return -1;
    size_t total, stride;
    int status = tq_tq02_sizes(ctx, n_vectors, packed_bytes, out_count, &total, &stride);
    if (status) return status;
    size_t d = (size_t)ctx->dim, output_bytes = total * sizeof(float);
    if (tq_overlap(packed, packed_bytes, out, output_bytes) ||
        tq_overlaps_context(ctx, out, output_bytes) ||
        tq_overlaps_context(ctx, packed, packed_bytes)) return -3;
    /* Validate all structural records before writing any consumer output. */
    for (int v = 0; v < n_vectors; v++) {
        if (!tq_validate_tq02_record(packed + (size_t)v * stride, stride, d, ctx->bits)) return -1;
    }
    for (int v = 0; v < n_vectors; v++) {
        const uint8_t* record = packed + (size_t)v * stride;
        float* xhat = out + (size_t)v * d;
        uint32_t norm_bits = tq_read_le32(record + 4);
        if (!norm_bits) { memset(xhat, 0, d * sizeof(float)); continue; }
        float norm;
        memcpy(&norm, &norm_bits, sizeof(norm));
        for (size_t j = 0; j < d; j++) {
            uint32_t level = tq_unpack_value(record + TQ_RECORD_HEADER,
                                             j * (size_t)ctx->bits, ctx->bits);
            xhat[j] = ctx->centroids[level];
        }
        srht_inverse(ctx->signs, xhat, ctx->dim, ctx->log2_dim);
        for (size_t j = 0; j < d; j++) {
            double value = (double)xhat[j] * (double)norm;
            if (!isfinite(value) || fabs(value) > FLT_MAX) return -4;
            xhat[j] = (float)value;
        }
    }
    return 0;
}

size_t tq_prod_idx_packed_size(const tq_ctx* ctx, int n_vectors) {
    if (!tq_valid_count(ctx, n_vectors) || !ctx->qjl_proj || ctx->bits < 2) return 0;
    return tq_bits_packed_size((size_t)n_vectors * (size_t)ctx->dim, ctx->bits - 1);
}

size_t tq_prod_qjl_packed_size(const tq_ctx* ctx, int n_vectors) {
    if (!tq_valid_count(ctx, n_vectors) || !ctx->qjl_proj) return 0;
    return tq_bits_packed_size((size_t)n_vectors * (size_t)ctx->dim, 1);
}

int tq_quantize_prod(
    tq_ctx* ctx,
    const float* in,
    uint8_t* out_idx_packed,
    uint8_t* out_qjl_packed,
    float* out_gamma,
    int n_vectors
) {
    if (ctx && !ctx->qjl_proj) return -2;
    if (!tq_valid_count(ctx, n_vectors) || !in || !out_idx_packed || !out_qjl_packed || !out_gamma) return -1;
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

        double gamma_sq = 0.0;
        for (int j = 0; j < d; j++) {
            float r = x[j] - xhat[j];
            res[j] = r;
            if (!isfinite(r)) { free(rot); free(xhat); free(res); return -1; }
            gamma_sq += (double)r * (double)r;
        }
        double gamma64 = sqrt(gamma_sq);
        if (gamma64 > FLT_MAX) { free(rot); free(xhat); free(res); return -1; }
        float gamma = (float)gamma64;
        out_gamma[v] = gamma;

        for (int i = 0; i < d; i++) {
            const float* row = ctx->qjl_proj + (size_t)i * (size_t)d;
            double dot = 0.0;
            for (int j = 0; j < d; j++) dot += (double)row[j] * (double)res[j];
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
    if (ctx && !ctx->qjl_proj) return -2;
    if (!tq_valid_count(ctx, n_vectors) || !in_idx_packed || !in_qjl_packed || !in_gamma || !out) return -1;
    if (ctx->bits < 2 || !ctx->centroids_prod || !ctx->qjl_proj) return -2;

    const int d = ctx->dim;
    const int bidx = ctx->bits - 1;
    const int log2d = ctx->log2_dim;
    const float qjl_scale = sqrtf((float)M_PI / 2.0f) / (float)d;

    float* xhat = (float*)malloc((size_t)d * sizeof(float));
    float* z = (float*)malloc((size_t)d * sizeof(float));
    if (!xhat || !z) {
        free(xhat); free(z);
        return -3;
    }

    for (int v = 0; v < n_vectors; v++) {
        if (!isfinite(in_gamma[v]) || in_gamma[v] < 0) { free(xhat); free(z); return -1; }
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
            double value = (double)xhat[j] + (double)in_gamma[v] * (double)qjl_scale * (double)stz;
            if (!isfinite(value) || fabs(value) > FLT_MAX) { free(xhat); free(z); return -1; }
            out_v[j] = (float)value;
        }
    }

    free(xhat); free(z);
    return 0;
}

/* ──────────────────────────────────────────────────────────────────────── */
/*  Parallel quantize / dequantize via pthreads                            */
/* ──────────────────────────────────────────────────────────────────────── */

#ifndef _WIN32
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

    /* Context data is immutable; each worker owns its transient buffer. */
    const int nl = ctx->n_levels;
    const int nb = nl - 1;
    const float* restrict signs = ctx->signs;
    const float* restrict B = ctx->boundaries;
    const int log2d = ctx->log2_dim;

    float* buf = (float*)malloc((size_t)d * sizeof(float));
    if (!buf) return NULL;

    const float* restrict in = a->in_f + (size_t)a->start * (size_t)d;
    uint16_t* restrict out = a->out_u + (size_t)a->start * (size_t)d;

    for (int v = 0; v < a->count; v++) {
        const float* restrict x = in + (size_t)v * (size_t)d;
        uint16_t* restrict idx = out + (size_t)v * (size_t)d;

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

    const uint16_t* restrict in = a->in_u + (size_t)a->start * (size_t)d;
    float* restrict out = a->out_f + (size_t)a->start * (size_t)d;

    for (int v = 0; v < a->count; v++) {
        const uint16_t* restrict idx = in + (size_t)v * (size_t)d;
        float* restrict xhat = out + (size_t)v * (size_t)d;

        for (int j = 0; j < d; j++) {
            xhat[j] = C[idx[j] < ctx->n_levels ? idx[j] : 0];
        }

        srht_inverse(signs, xhat, d, log2d);
    }

    return NULL;
}

void tq_quantize_parallel(tq_ctx* ctx, const float* in, uint16_t* out, int n_vectors) {
    if (!tq_valid_count(ctx, n_vectors) || !in || !out) return;
    if (n_vectors < 1000) {
        tq_quantize(ctx, in, out, n_vectors);
        return;
    }

    int n_threads = TQ_MAX_THREADS;
    if (n_vectors < n_threads * 100) n_threads = 2;

    pthread_t threads[TQ_MAX_THREADS];
    int started[TQ_MAX_THREADS] = {0};
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
        started[t] = pthread_create(&threads[t], NULL, tq_quantize_worker, &args[t]) == 0;
        if (!started[t]) tq_quantize_worker(&args[t]);
    }

    for (int t = 0; t < n_threads; t++) {
        if (started[t]) pthread_join(threads[t], NULL);
    }
}

void tq_dequantize_parallel(tq_ctx* ctx, const uint16_t* in, float* out, int n_vectors) {
    if (!tq_valid_count(ctx, n_vectors) || !in || !out) return;
    if (n_vectors < 1000) {
        tq_dequantize(ctx, in, out, n_vectors);
        return;
    }

    int n_threads = TQ_MAX_THREADS;
    if (n_vectors < n_threads * 100) n_threads = 2;

    pthread_t threads[TQ_MAX_THREADS];
    int started[TQ_MAX_THREADS] = {0};
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
        started[t] = pthread_create(&threads[t], NULL, tq_dequantize_worker, &args[t]) == 0;
        if (!started[t]) tq_dequantize_worker(&args[t]);
    }

    for (int t = 0; t < n_threads; t++) {
        if (started[t]) pthread_join(threads[t], NULL);
    }
}

#else
void tq_quantize_parallel(tq_ctx* ctx, const float* in, uint16_t* out, int n_vectors) {
    tq_quantize(ctx, in, out, n_vectors);
}
void tq_dequantize_parallel(tq_ctx* ctx, const uint16_t* in, float* out, int n_vectors) {
    tq_dequantize(ctx, in, out, n_vectors);
}
#endif
