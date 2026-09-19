#include "turboquant.h"
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void roundtrip(int dim, int bits) {
    tq_ctx* ctx = tq_create(dim, bits, 42);
    assert(ctx);
    size_t stride = 8u + ((size_t)dim * (size_t)bits + 7u) / 8u;
    assert(tq_packed_size(ctx, 3) == stride * 3);
    float* x = calloc((size_t)dim * 3, sizeof(float));
    float* out = calloc((size_t)dim * 3, sizeof(float));
    uint8_t* packed = malloc(stride * 3);
    assert(x && out && packed);
    for (int i = 0; i < dim; i++) {
        x[i] = (float)(i + 1) / (float)dim;
        x[dim + i] = 64.0f * x[i];
    }
    assert(tq_quantize_packed(ctx, x, packed, 3) == 0);
    assert(memcmp(packed, "TQ01", 4) == 0);
    assert(tq_dequantize_packed(ctx, packed, out, 3) == 0);
    for (int i = 0; i < dim; i++) {
        assert(isfinite(out[i]));
        assert(fabsf(out[dim + i] - 64.0f * out[i]) < 1e-4f);
        assert(out[2 * dim + i] == 0.0f);
    }
    /* Independently indexed records match batched encoding. */
    assert(tq_dequantize_packed(ctx, packed + stride, out, 1) == 0);
    assert(memcmp(packed + 8, packed + stride + 8, stride - 8) == 0);
    packed[0] = 0;
    assert(tq_dequantize_packed(ctx, packed, out, 1) == -1);
    packed[0] = 'T';
    float invalid_norm = NAN;
    memcpy(packed + 4, &invalid_norm, sizeof(invalid_norm));
    assert(tq_dequantize_packed(ctx, packed, out, 1) == -1);
    x[0] = NAN;
    assert(tq_quantize_packed(ctx, x, packed, 1) == -1);
    assert(tq_packed_size(ctx, -1) == 0);
    assert(tq_quantize_packed(ctx, x, packed, 0) == -1);
    assert(tq_packed_size(ctx, INT_MAX) == stride * (size_t)INT_MAX);
    free(x); free(out); free(packed);
    tq_destroy(ctx);
}

static void raw_small_dimensions(void) {
    for (int dim = 1; dim <= 4; dim *= 2) {
        tq_ctx* ctx = tq_create(dim, 3, 42);
        float* x = malloc((size_t)dim * sizeof(float));
        float* out = malloc((size_t)dim * sizeof(float));
        uint16_t* idx = malloc((size_t)dim * sizeof(uint16_t));
        for (int i = 0; i < dim; i++) x[i] = 1.0f / sqrtf((float)dim);
        tq_quantize(ctx, x, idx, 1);
        tq_dequantize(ctx, idx, out, 1);
        for (int i = 0; i < dim; i++) assert(isfinite(out[i]));
        idx[0] = UINT16_MAX;
        tq_dequantize(ctx, idx, out, 1);
        free(x); free(out); free(idx); tq_destroy(ctx);
    }
}

static void parallel_matches_serial(void) {
    const int dim = 8, count = 1001;
    size_t total = (size_t)dim * count;
    tq_ctx* ctx = tq_create(dim, 3, 7);
    float* x = malloc(total * sizeof(float));
    float* a = malloc(total * sizeof(float));
    float* b = malloc(total * sizeof(float));
    uint16_t* ia = malloc(total * sizeof(uint16_t));
    uint16_t* ib = malloc(total * sizeof(uint16_t));
    assert(ctx && x && a && b && ia && ib);
    for (size_t i = 0; i < total; i++) x[i] = (float)(i % dim) / (float)dim;
    tq_quantize(ctx, x, ia, count);
    tq_quantize_parallel(ctx, x, ib, count);
    assert(memcmp(ia, ib, total * sizeof(uint16_t)) == 0);
    tq_dequantize(ctx, ia, a, count);
    tq_dequantize_parallel(ctx, ib, b, count);
    assert(memcmp(a, b, total * sizeof(float)) == 0);
    free(x); free(a); free(b); free(ia); free(ib); tq_destroy(ctx);
}

static void prod_roundtrip(void) {
    for (int dim = 1; dim <= 64; dim *= 2) {
        tq_ctx* ctx = tq_create(dim, 3, 42);
        float* x = malloc((size_t)dim * sizeof(float));
        float* out = malloc((size_t)dim * sizeof(float));
        uint8_t* idx = malloc(tq_prod_idx_packed_size(ctx, 1));
        uint8_t* qjl = malloc(tq_prod_qjl_packed_size(ctx, 1));
        float gamma = 0;
        for (int i = 0; i < dim; i++) x[i] = (float)(i + 1);
        assert(tq_quantize_prod(ctx, x, idx, qjl, &gamma, 1) == 0);
        assert(tq_dequantize_prod(ctx, idx, qjl, &gamma, out, 1) == 0);
        for (int i = 0; i < dim; i++) assert(isfinite(out[i]));
        assert(gamma >= 0 && isfinite(gamma));
        free(x); free(out); free(idx); free(qjl); tq_destroy(ctx);
    }
}

int main(void) {
    assert(!tq_create(0, 3, 42));
    assert(!tq_create(3, 3, 42));
    assert(!tq_create(4, 0, 42));
    assert(tq_packed_size(NULL, 1) == 0);
    tq_ctx* edge = tq_create(1, 2, 42);
    float maximum = FLT_MAX, decoded = 0.0f;
    uint8_t edge_packed[9];
    assert(tq_quantize_packed(edge, &maximum, edge_packed, 1) == 0);
    assert(tq_dequantize_packed(edge, edge_packed, &decoded, 1) == -1);
    tq_destroy(edge);
    raw_small_dimensions();
    const int dims[] = {1, 2, 4, 8, 64};
    for (size_t i = 0; i < sizeof(dims) / sizeof(dims[0]); i++) {
        roundtrip(dims[i], 1);
        roundtrip(dims[i], 3);
        roundtrip(dims[i], 8);
    }
    parallel_matches_serial();
    prod_roundtrip();
    puts("TurboQuant regressions passed");
    return 0;
}
