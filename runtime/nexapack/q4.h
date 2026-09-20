#ifndef NEXA_NEXAPACK_Q4_H
#define NEXA_NEXAPACK_Q4_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define NEXA_Q4_API __declspec(dllexport)
#else
#define NEXA_Q4_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Rows are independent. Each group occupies 4 + ceil(group_size / 2) bytes:
 * a little-endian IEEE binary32 scale, then low-nibble-first signed Q4 values.
 * Valid values are -7..7; the two's-complement -8 code is reserved/invalid.
 * Scale must be finite and nonnegative. A zero scale requires all-zero values.
 * All padding (including the spare nibble of odd group sizes) must be zero.
 * Shape and group size are supplied by the enclosing model/container format.
 */
enum nexa_q4_status {
    NEXA_Q4_OK = 0,
    NEXA_Q4_INVALID_ARGUMENT = -1,
    NEXA_Q4_BUFFER_TOO_SMALL = -2,
    NEXA_Q4_OVERFLOW = -3,
    NEXA_Q4_INVALID_DATA = -4,
    NEXA_Q4_NUMERIC_RANGE = -5
};

/* Zero dimensions/group sizes and size_t overflow return zero. */
NEXA_Q4_API size_t nexa_q4_row_size(size_t cols, size_t group_size);
NEXA_Q4_API size_t nexa_q4_size(size_t rows, size_t cols, size_t group_size);

/* Quantize weights[rows, cols], using max(abs(group))/7 rounded to float32.
 * Values are divided by that stored scale, rounded half away from zero and
 * clamped to [-7,7]. A nonzero group whose scale underflows to zero is rejected.
 * weight_count is a count of floats; packed_bytes is a count of bytes.
 * Input and output buffers must not overlap. No allocation is performed.
 */
NEXA_Q4_API int nexa_q4_quantize(
    const float *weights, size_t weight_count,
    size_t rows, size_t cols, size_t group_size,
    uint8_t *packed, size_t packed_bytes);

/* Compute inputs[batch,cols] @ weights[rows,cols]^T -> output[batch,rows].
 * The packed values are consumed directly, with double-precision accumulation;
 * no decoded matrix, workspace, or heap allocation is used. Counts of input
 * and output elements are float counts, not byte counts. Output must not
 * overlap inputs or packed weights. All dimensions must be positive.
 * Nonfinite inputs, malformed packed data and float32 output overflow fail.
 * Callers must discard output on error: numeric errors may leave partial data.
 * Buffers may have excess capacity; bytes/elements beyond the shape are unused.
 */
/* Q2_GROUPED v1: float32 scale then ternary codes -1, 0 and 1 in two bits,
 * packed from the least significant bit; code -2 is reserved. */
NEXA_Q4_API int nexa_q2_decode_row(
    const uint8_t *packed, size_t packed_bytes, size_t cols, size_t group_size,
    float *output, size_t output_count);

NEXA_Q4_API int nexa_q2_matmul(
    const float *inputs, size_t input_count, size_t batch,
    const uint8_t *packed, size_t packed_bytes,
    size_t rows, size_t cols, size_t group_size,
    float *output, size_t output_count);

/* Q3_GROUPED v1: float32 scale then three-bit signed codes packed from the
 * least significant bit, the same layout the paged KV cache stores. Code -4
 * and nonzero padding bits are rejected as invalid data. The row size helper
 * lives in the transformer kernels, which already store this layout. */
NEXA_Q4_API int nexa_q3_decode_row(
    const uint8_t *packed, size_t packed_bytes, size_t cols, size_t group_size,
    float *output, size_t output_count);

NEXA_Q4_API int nexa_q3_matmul(
    const float *inputs, size_t input_count, size_t batch,
    const uint8_t *packed, size_t packed_bytes,
    size_t rows, size_t cols, size_t group_size,
    float *output, size_t output_count);

/* Q8_GROUPED: float32 scale then one signed byte per coordinate, per group.
 * Contract matches nexa_q4_matmul; code -128 is rejected as invalid data. */
NEXA_Q4_API size_t nexa_q8_row_size(size_t cols, size_t group_size);

NEXA_Q4_API int nexa_q8_decode_row(
    const uint8_t *packed, size_t packed_bytes, size_t cols, size_t group_size,
    float *output, size_t output_count);

NEXA_Q4_API int nexa_q8_matmul(
    const float *inputs, size_t input_count, size_t batch,
    const uint8_t *packed, size_t packed_bytes,
    size_t rows, size_t cols, size_t group_size,
    float *output, size_t output_count);

NEXA_Q4_API int nexa_q4_matmul(
    const float *inputs, size_t input_count, size_t batch,
    const uint8_t *packed, size_t packed_bytes,
    size_t rows, size_t cols, size_t group_size,
    float *output, size_t output_count);

/* Decode exactly one already selected packed row to output[cols]. The caller
 * reads/selects the embedding row; no complete matrix or heap is allocated.
 * Applies the same packed metadata, padding and finite-range checks as matmul.
 * Output must not overlap packed. Discard output on error.
 */
NEXA_Q4_API int nexa_q4_decode_row(
    const uint8_t *packed, size_t packed_bytes, size_t cols, size_t group_size,
    float *output, size_t output_count);


/* ── qint<N> / PackedVector<N> storage ABI ────────────────────────────────
 * One kernel family for the widths the language exposes: bits in {2,3,4,8}.
 * A packed vector is `groups` records of 4 + ceil(bits*group_size/8) bytes:
 * a little-endian F32 scale, then bits-wide two's-complement codes packed
 * from the least significant bit. The last group encodes only the values it
 * actually has; its remaining code slots and the spare high bits stay zero.
 * Byte-for-byte this is Q2_GROUPED, Q3_GROUPED, Q4_GROUPED and Q8_GROUPED
 * version 1 as `runtime/nexapack/format.py` writes them.
 * Zero and unsupported widths return zero (sizes) or a negative status.
 */
NEXA_Q4_API size_t nexa_qpack_size(size_t bits, size_t count, size_t group_size);
NEXA_Q4_API size_t nexa_qpack_groups(size_t bits, size_t count, size_t group_size);

NEXA_Q4_API int nexa_qpack_pack(
    size_t bits, const float *values, size_t value_count,
    size_t count, size_t group_size,
    uint8_t *packed, size_t packed_bytes);

NEXA_Q4_API int nexa_qpack_unpack(
    size_t bits, const uint8_t *packed, size_t packed_bytes,
    size_t count, size_t group_size,
    float *output, size_t output_count);

/* Read one stored level code, or one group scale, without decoding the rest. */
NEXA_Q4_API int nexa_qpack_code(
    size_t bits, const uint8_t *packed, size_t packed_bytes,
    size_t count, size_t group_size, size_t index, int8_t *code);

NEXA_Q4_API int nexa_qpack_scale(
    size_t bits, const uint8_t *packed, size_t packed_bytes,
    size_t count, size_t group_size, size_t group, float *scale);

#ifdef __cplusplus
}
#endif
#endif
