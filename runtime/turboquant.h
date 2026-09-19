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
 *   tq_ctx*   tq_create_mse(dim, bits, seed)
 *   size_t    tq_context_memory_bytes(const tq_ctx*)
 *   void      tq_destroy(tq_ctx*)
 *   void      tq_quantize(ctx, in_f32*, out_u16*, n_vectors)
 *   void      tq_dequantize(ctx, in_u16*, out_f32*, n_vectors)
 *   int       tq_quantize_packed(ctx, in_f32*, out_u8*, n_vectors)
 *   int       tq_dequantize_packed(ctx, in_u8*, out_f32*, n_vectors)
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

#ifdef _WIN32
#define TQ_API __declspec(dllexport)
#else
#define TQ_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque context handle. The caller owns a successfully created context and
 * must destroy it once all operations using it have finished. Inputs/outputs
 * remain caller-owned and must not alias. Execution does not mutate context
 * data: independent calls may share a context with disjoint writable buffers.
 * Destroy must not run concurrently with any operation using that context. */
typedef struct tq_ctx tq_ctx;

/* Conservative ABI-independent reserve for planners that admit before loading
 * a native library. Actual allocation accounting remains available below. */
#define TQ_MSE_CONTEXT_STRUCT_RESERVE_BYTES 128u

/* Read-only metadata; NULL returns zero. Buffer validation returns one only
 * for a nonnull, nonwrapping span disjoint from every context-owned allocation.
 * It does not dereference the supplied span or allocate memory. */
TQ_API int tq_context_dim(const tq_ctx* ctx);
TQ_API int tq_context_bits(const tq_ctx* ctx);
TQ_API int tq_context_buffer_disjoint(const tq_ctx* ctx, const void* data, size_t bytes);

/*
 * Create a legacy TurboQuant context with MSE and Prod state.
 *   dim   — vector dimension (d)
 *   bits  — quantization bitwidth (1..8)
 *   seed  — PRNG seed for the rotation matrix
 *
 * Notes:
 * - dim must be a power of 2 (required by FWHT-based transform).
 * - Returns NULL for invalid dimensions/bits, size overflow or allocation failure.
 * - Preserves the legacy sign/QJL PRNG sequence and MSE/Prod results.
 * - Owns a dim*dim float QJL matrix and a legacy reserved dim-float buffer,
 *   including at bits=1 (where Prod execution itself remains unsupported).
 */
TQ_API tq_ctx* tq_create(int dim, int bits, int seed);

/* Create an MSE-only context with the same MSE results and input constraints.
 * Persistent data is linear in dim: signs plus the MSE codebook/boundaries.
 * It owns no QJL matrix, Prod codebook or persistent scratch buffer. Admission
 * checks only the allocations used by this mode, not a quadratic Prod size.
 * There is no lazy upgrade: Prod operations return -2 without allocations or
 * output writes for this context, and both Prod size helpers return 0.
 * Return/ownership rules otherwise match tq_create. */
TQ_API tq_ctx* tq_create_mse(int dim, int bits, int seed);

/* Portable MSE codec TQ_MSE_SRHT v1 (SRHT_XOSHIRO256SS_V1).
 * TQ02 rows contain magic, IEEE binary32 little-endian norm, then LSB-first
 * indices. Unused high bits are zero; a zero norm must be +0 with zero indices.
 * Persist the explicit codebook with dim/bits/signed-int32 seed; decoding must
 * import those centroids instead of recomputing a platform-dependent codebook.
 * Imported centroids must be finite, strictly increasing, and have finite
 * float32 sums for adjacent midpoint boundaries (legacy operation order).
 */
TQ_API size_t tq_mse_context_memory_size(int dim, int bits);
TQ_API tq_ctx* tq_create_mse_from_codebook(int dim, int bits, int seed,
                                        const float* centroids, size_t count);
TQ_API int tq_export_mse_codebook(const tq_ctx* ctx, float* out, size_t count);

/* No heap allocations. Quantization uses caller scratch of dim floats; decode
 * transforms directly in caller output. Float capacities count elements and
 * packed_bytes must equal tq_packed_size(ctx, n_vectors). Caller buffers must
 * not overlap each other or context-owned storage. Return 0 on success,
 * -1 invalid arguments/record, -2 capacity/overflow, -3 overlap, -4 arithmetic
 * range/nonfinite input. Errors may leave partial output (never context state).
 * Only IEEE binary32 hosts are supported. The byte format is endian independent.
 */
TQ_API int tq_quantize_tq02(const tq_ctx* ctx, const float* in, size_t in_count,
                          uint8_t* packed, size_t packed_bytes, int n_vectors,
                          float* scratch, size_t scratch_count);
TQ_API int tq_dequantize_tq02(const tq_ctx* ctx, const uint8_t* packed,
                            size_t packed_bytes, float* out, size_t out_count,
                            int n_vectors);

/* Sum of bytes requested by all persistent allocations owned by ctx, including
 * the opaque context itself. NULL returns 0. Excludes allocator overhead,
 * caller buffers, stack, thread runtime and transient workspaces; it is not RSS.
 * Both constructors validate the full persistent sum before allocating.
 * Construction also uses a temporary (2^bits + 1)-float Lloyd-Max buffer
 * (and, for legacy bits>=2, a separate (2^(bits-1) + 1)-float buffer).
 * These constructor buffers are sequential, never retained in the context. */
TQ_API size_t tq_context_memory_bytes(const tq_ctx* ctx);

/* Destroy context and free all owned buffers; NULL is accepted. */
TQ_API void tq_destroy(tq_ctx* ctx);

/*
 * Quantize n_vectors of dimension ctx->dim.
 *   in:  float[n_vectors * dim]   (row-major, unit-norm vectors; use packed API for arbitrary norms)
 *   out: uint16_t[n_vectors * dim] (index per coordinate, 0..2^bits-1)
 *   Transient heap workspace: dim*sizeof(float), reused for all vectors.
 */
TQ_API void tq_quantize(tq_ctx* ctx, const float* in, uint16_t* out, int n_vectors);

/*
 * Dequantize: reconstruct approximate vectors from indices.
 *   in:  uint16_t[n_vectors * dim]
 *   out: float[n_vectors * dim]
 *   No transient heap allocation; transforms directly in caller output.
 */
TQ_API void tq_dequantize(tq_ctx* ctx, const uint16_t* in, float* out, int n_vectors);

/* Scaled packed records: each vector stores the magic/version bytes TQ01,
 * sizeof(float) bytes of norm, then ceil(dim*bits/8) bytes of indices. Records are independently addressable.
 * This format is incompatible with the former indices-only packed format.
 * Packed operations return 0 on success, -1 on invalid input, -2 on allocation
 * failure. Norms are finite nonnegative host-endian IEEE float32 values.
 * Size helpers return 0 on invalid arguments or overflow. Serial quantization
 * allocates dim*sizeof(float) transient bytes independent of n_vectors;
 * dequantization performs no heap allocations. */
TQ_API size_t tq_packed_size(const tq_ctx* ctx, int n_vectors);
TQ_API int tq_quantize_packed(tq_ctx* ctx, const float* in, uint8_t* out, int n_vectors);
TQ_API int tq_dequantize_packed(tq_ctx* ctx, const uint8_t* in, float* out, int n_vectors);

/*
 * Convenience: compute MSE between original and round-trip reconstruction.
 *   x:  float[n_vectors * dim]
 *   Returns: mean squared error across all vectors.
 *   Diagnostic helper: allocates tq_packed_size(ctx, n_vectors) bytes plus
 *   n_vectors*dim*sizeof(float) reconstruction bytes, and calls packed
 *   quantization with its additional dim*sizeof(float) transient workspace.
 *   This batch diagnostic is not the bounded per-vector execution API.
 */
TQ_API float tq_mse(tq_ctx* ctx, const float* x, int n_vectors);

/* Unit-norm theoretical reference upper bound: (sqrt(3π)/2) * 4^{-bits} */
TQ_API float tq_upper_bound(tq_ctx* ctx);

/* Unit-norm theoretical reference lower bound: 4^{-bits} */
TQ_API float tq_lower_bound(tq_ctx* ctx);

/*
 * TurboQuantProd-style API (inner-product oriented):
 * - Uses (bits-1)-bit MSE quantization for the base component.
 * - Uses 1-bit QJL-style residual sketch with per-vector gamma scaling.
 * - Requires tq_create, bits>=2; no implicit upgrade of tq_create_mse contexts.
 * - Returns 0 on success, -1 for invalid input, -2 for unsupported context/mode,
 *   -3 for transient allocation failure. MSE-only mode returns -2 first.
 * - Quantization allocates 3*dim*sizeof(float) transient bytes;
 *   dequantization allocates 2*dim*sizeof(float) transient bytes.
 * - Size helpers return 0 for MSE-only contexts. For legacy bits=1 the index
 *   helper returns 0 and the QJL helper preserves its legacy nonzero size.
 */
TQ_API size_t tq_prod_idx_packed_size(const tq_ctx* ctx, int n_vectors);
TQ_API size_t tq_prod_qjl_packed_size(const tq_ctx* ctx, int n_vectors);
TQ_API int tq_quantize_prod(
	tq_ctx* ctx,
	const float* in,
	uint8_t* out_idx_packed,
	uint8_t* out_qjl_packed,
	float* out_gamma,
	int n_vectors
);
TQ_API int tq_dequantize_prod(
	tq_ctx* ctx,
	const uint8_t* in_idx_packed,
	const uint8_t* in_qjl_packed,
	const float* in_gamma,
	float* out,
	int n_vectors
);

/* Parallel quantize/dequantize using pthreads (multi-core). Quantization owns
 * dim*sizeof(float) transient bytes per active worker (up to 8); dequantization
 * owns no heap workspace. Thread runtime/stack costs are separate. */
TQ_API void tq_quantize_parallel(tq_ctx* ctx, const float* in, uint16_t* out, int n_vectors);
TQ_API void tq_dequantize_parallel(tq_ctx* ctx, const uint16_t* in, float* out, int n_vectors);

#ifdef __cplusplus
}
#endif

#endif /* TURBOQUANT_H */
