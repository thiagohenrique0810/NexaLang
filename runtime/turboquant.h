/*
 * turboquant.h — NexaLang C runtime shim for TurboQuant vector compression.
 *
 * Implements core TurboQuant-style vector quantization (Zandieh et al. 2025) in plain C
 * so NexaLang binaries can link against it via FFI without Python/NumPy.
 *
 * Pipeline:  x -> randomized orthogonal transform -> lloyd_max_quantize -> indices
 *            indices -> lloyd_max_dequantize -> inverse transform -> x_hat
 *
 * API:
 *   tq_ctx*   tq_create(dim, bits, seed)
 *   void      tq_destroy(tq_ctx*)
 *   void      tq_quantize(ctx, in_f32*, out_u16*, n_vectors)
 *   void      tq_dequantize(ctx, in_u16*, out_f32*, n_vectors)
 *   void      tq_quantize_packed(ctx, in_f32*, out_u8*, n_vectors)
 *   void      tq_dequantize_packed(ctx, in_u8*, out_f32*, n_vectors)
 *   float     tq_mse(ctx, original_f32*, n_vectors)
 *   float     tq_upper_bound(ctx)
 *   float     tq_lower_bound(ctx)
 *   int       tq_quantize_prod(...)
 *   int       tq_dequantize_prod(...)
 */

#ifndef TURBOQUANT_H
#define TURBOQUANT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque context handle */
typedef struct tq_ctx tq_ctx;

/*
 * Create a TurboQuant MSE context.
 *   dim   — vector dimension (d)
 *   bits  — quantization bitwidth (1..8)
 *   seed  — PRNG seed for the rotation matrix
 *
 * Notes:
 * - dim must be a power of 2 (required by FWHT-based transform).
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

/* Packed bitstream helpers (stores exactly bits-per-coordinate). */
size_t tq_packed_size(const tq_ctx* ctx, int n_vectors);
void tq_quantize_packed(tq_ctx* ctx, const float* in, uint8_t* out, int n_vectors);
void tq_dequantize_packed(tq_ctx* ctx, const uint8_t* in, float* out, int n_vectors);

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

/*
 * TurboQuantProd-style API (inner-product oriented):
 * - Uses (bits-1)-bit MSE quantization for the base component.
 * - Uses 1-bit QJL-style residual sketch with per-vector gamma scaling.
 * - Returns 0 on success, non-zero on invalid arguments.
 */
size_t tq_prod_idx_packed_size(const tq_ctx* ctx, int n_vectors);
size_t tq_prod_qjl_packed_size(const tq_ctx* ctx, int n_vectors);
int tq_quantize_prod(
	tq_ctx* ctx,
	const float* in,
	uint8_t* out_idx_packed,
	uint8_t* out_qjl_packed,
	float* out_gamma,
	int n_vectors
);
int tq_dequantize_prod(
	tq_ctx* ctx,
	const uint8_t* in_idx_packed,
	const uint8_t* in_qjl_packed,
	const float* in_gamma,
	float* out,
	int n_vectors
);

/* Parallel quantize/dequantize using pthreads (multi-core) */
void tq_quantize_parallel(tq_ctx* ctx, const float* in, uint16_t* out, int n_vectors);
void tq_dequantize_parallel(tq_ctx* ctx, const uint16_t* in, float* out, int n_vectors);

#ifdef __cplusplus
}
#endif

#endif /* TURBOQUANT_H */
