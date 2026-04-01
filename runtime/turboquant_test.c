/*
 * turboquant_test.c — Quick smoke test for the C TurboQuant runtime.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "turboquant.h"

int main(void) {
    int dim  = 64;
    int bits = 3;

    printf("TurboQuant C runtime test (dim=%d, bits=%d)\n", dim, bits);

    tq_ctx* ctx = tq_create(dim, bits, 42);

    /* Create a simple test vector and normalize to unit norm */
    float* x = (float*)malloc(dim * sizeof(float));
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) {
        x[i] = (float)(i + 1) * 0.1f;
        norm += x[i] * x[i];
    }
    norm = sqrtf(norm);
    for (int i = 0; i < dim; i++) {
        x[i] /= norm;
    }

    /* Quantize */
    uint16_t* idx = (uint16_t*)malloc(dim * sizeof(uint16_t));
    tq_quantize(ctx, x, idx, 1);

    /* Dequantize */
    float* x_hat = (float*)malloc(dim * sizeof(float));
    tq_dequantize(ctx, idx, x_hat, 1);

    /* Measure MSE */
    float mse = tq_mse(ctx, x, 1);
    float ub  = tq_upper_bound(ctx);
    float lb  = tq_lower_bound(ctx);

    printf("  MSE:         %.6f\n", mse);
    printf("  Upper bound: %.6f\n", ub);
    printf("  Lower bound: %.6f\n", lb);
    printf("  First 4 original:      %.4f %.4f %.4f %.4f\n", x[0], x[1], x[2], x[3]);
    printf("  First 4 reconstructed: %.4f %.4f %.4f %.4f\n", x_hat[0], x_hat[1], x_hat[2], x_hat[3]);

    int pass = (mse >= 0.0f && mse <= ub * 2.0f);
    printf("  Result: %s\n", pass ? "PASS" : "FAIL");

    free(x);
    free(idx);
    free(x_hat);
    tq_destroy(ctx);

    return pass ? 0 : 1;
}
