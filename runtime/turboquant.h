/*
 * turboquant.h — NexaLang C runtime shim for TurboQuant vector compression.
 *
 * Implements the core TurboQuant algorithm (Zandieh et al. 2025) in plain C
 * so NexaLang binaries can link against it via FFI without Python/NumPy.
 *
 * Pipeline:  x -> rotate(Π) -> lloyd_max_quantize -> indices (b bits/coord)
 *            indices -> lloyd_max_dequantize -> rotate_inv(Π) -> x̂
 *
 * API:
 *   tq_ctx*   tq_create(dim, bits, seed)
 *   void      tq_destroy(tq_ctx*)
 *   void      tq_quantize(ctx, in_f32*, out_u16*, n_vectors)
 *   void      tq_dequantize(ctx, in_u16*, out_f32*, n_vectors)
 *   float     tq_mse(ctx, original_f32*, n_vectors)
 *   float     tq_upper_bound(ctx)
 *   float     tq_lower_bound(ctx)
 */

#ifndef TURBOQUANT_H
#define TURBOQUANT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque context handle */
typedef struct tq_ctx tq_ctx;

/*
 * Create a TurboQuant MSE context.
 *   dim   — vector dimension (d)
 *   bits  — quantization bitwidth (1..4 typically)
 *   seed  — PRNG seed for the rotation matrix
 */
tq_ctx* tq_create(int dim, int bits, int seed);

/* Destroy context and free all internal buffers. */
void tq_destroy(tq_ctx* ctx);

/*
 * Quantize n_vectors of dimension ctx->dim.
 *   in:  float[n_vectors * dim]   (row-major, need NOT be unit-norm)
 *   out: uint16_t[n_vectors * dim] (index per coordinate, 0..2^bits-1)
 */
void tq_quantize(tq_ctx* ctx, const float* in, uint16_t* out, int n_vectors);

/*
 * Dequantize: reconstruct approximate vectors from indices.
 *   in:  uint16_t[n_vectors * dim]
 *   out: float[n_vectors * dim]
 */
void tq_dequantize(tq_ctx* ctx, const uint16_t* in, float* out, int n_vectors);

/*
 * Convenience: compute MSE between original and round-trip reconstruction.
 *   x:  float[n_vectors * dim]
 *   Returns: mean squared error across all vectors.
 */
float tq_mse(tq_ctx* ctx, const float* x, int n_vectors);

/* Theoretical upper bound: (sqrt(3π)/2) * 4^{-bits} */
float tq_upper_bound(tq_ctx* ctx);

/* Theoretical lower bound: 4^{-bits} */
float tq_lower_bound(tq_ctx* ctx);

/* Parallel quantize/dequantize using pthreads (multi-core) */
void tq_quantize_parallel(tq_ctx* ctx, const float* in, uint16_t* out, int n_vectors);
void tq_dequantize_parallel(tq_ctx* ctx, const uint16_t* in, float* out, int n_vectors);

#ifdef __cplusplus
}
#endif

#endif /* TURBOQUANT_H */
