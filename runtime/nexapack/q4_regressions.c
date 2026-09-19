#include "q4.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static void sizes_and_argument_checks(void) {
    assert(nexa_q4_row_size(5, 3) == 12);
    assert(nexa_q4_size(3, 5, 3) == 36);
    assert(nexa_q4_row_size(5, 5) == 7);
    assert(nexa_q4_row_size(1, 1) == 5);
    assert(nexa_q4_row_size(0, 3) == 0);
    assert(nexa_q4_row_size(3, 0) == 0);
    assert(nexa_q4_size(0, 3, 3) == 0);
    assert(nexa_q4_row_size(SIZE_MAX, 1) == 0);
    assert(nexa_q4_size(SIZE_MAX, 1, 1) == 0);
    float weight = 7.0f, output = 0.0f;
    uint8_t short_buffer[1] = {0};
    uint8_t packed[5] = {0};
    assert(nexa_q4_quantize(NULL, 1, 1, 1, 1, packed, sizeof(packed)) == NEXA_Q4_INVALID_ARGUMENT);
    assert(nexa_q4_quantize(&weight, 0, 1, 1, 1, packed, sizeof(packed)) == NEXA_Q4_BUFFER_TOO_SMALL);
    assert(nexa_q4_quantize(&weight, 1, 1, 1, 1, short_buffer, sizeof(short_buffer)) == NEXA_Q4_BUFFER_TOO_SMALL);
    assert(nexa_q4_quantize(&weight, 1, SIZE_MAX, 2, 1, packed, sizeof(packed)) == NEXA_Q4_OVERFLOW);
    assert(nexa_q4_quantize(&weight, 1, 1, 1, 1, (uint8_t*)&weight, sizeof(weight)) == NEXA_Q4_BUFFER_TOO_SMALL);
    assert(nexa_q4_matmul(&weight, 1, 1, short_buffer, sizeof(short_buffer), 1, 1, 1, &output, 1) == NEXA_Q4_BUFFER_TOO_SMALL);
    assert(nexa_q4_matmul(&weight, 1, SIZE_MAX, packed, sizeof(packed), 2, 2, 2, &output, 1) == NEXA_Q4_OVERFLOW);
    assert(nexa_q4_matmul(&weight, 1, 0, packed, sizeof(packed), 1, 1, 1, &output, 1) == NEXA_Q4_INVALID_ARGUMENT);
}

static void golden_bytes_and_half_rounding(void) {
    const float weights[5] = {7.0f, 0.5f, -0.5f, 1.5f, -1.5f};
    uint8_t packed[7];
    const uint8_t expected[7] = {0x00, 0x00, 0x80, 0x3f, 0x17, 0x2f, 0x0e};
    assert(nexa_q4_quantize(weights, 5, 1, 5, 5, packed, sizeof(packed)) == 0);
    assert(memcmp(packed, expected, sizeof(expected)) == 0);
    const float zero[5] = {0};
    assert(nexa_q4_quantize(zero, 5, 1, 5, 5, packed, sizeof(packed)) == 0);
    const uint8_t zeros[7] = {0};
    assert(memcmp(packed, zeros, sizeof(zeros)) == 0);
}

static void matrix_tails_and_batch(void) {
    const float weights[15] = {7, 0, -7, 7, -7, 1, 7, 0, -7, 1, -2, 3, 7, 0, 7};
    const float inputs[10] = {1, 2, 3, 4, 5, -1, 0.5f, 0, 2, -3};
    const float expected[6] = {-21, -8, 60, 28, -14.5f, -17.5f};
    uint8_t packed[36];
    float output[6];
    assert(nexa_q4_quantize(weights, 15, 3, 5, 3, packed, sizeof(packed)) == 0);
    assert(nexa_q4_matmul(inputs, 10, 2, packed, sizeof(packed), 3, 5, 3, output, 6) == 0);
    for (size_t i = 0; i < 6; i++) assert(output[i] == expected[i]);
    /* Last group in each row has only two live values and one zero padding byte. */
    for (size_t row = 0; row < 3; row++) assert(packed[row * 12 + 11] == 0);
    /* Odd group size has a spare upper nibble even in complete groups. */
    for (size_t row = 0; row < 3; row++) assert((packed[row * 12 + 5] & 0xf0) == 0);
}

static void malformed_records_and_numeric_ranges(void) {
    const float one = 1.0f;
    float output = 123.0f;
    uint8_t packed[5] = {0, 0, 0x80, 0x3f, 7}; /* scale=1, q=7 */
    assert(nexa_q4_matmul(&one, 1, 1, packed, 5, 1, 1, 1, &output, 1) == 0);
    assert(output == 7.0f);
    packed[4] = 8; /* -8 is not part of the codebook. */
    output = 123.0f;
    assert(nexa_q4_matmul(&one, 1, 1, packed, 5, 1, 1, 1, &output, 1) == NEXA_Q4_INVALID_DATA);
    assert(output == 123.0f);
    packed[4] = 0x17; /* Nonzero spare nibble. */
    assert(nexa_q4_matmul(&one, 1, 1, packed, 5, 1, 1, 1, &output, 1) == NEXA_Q4_INVALID_DATA);
    packed[4] = 7;
    const float pair[2] = {1, 1};
    const uint8_t bad_tail[6] = {0, 0, 0x80, 0x3f, 0x77, 0x01};
    assert(nexa_q4_matmul(pair, 2, 1, bad_tail, sizeof(bad_tail), 1, 2, 4, &output, 1) == NEXA_Q4_INVALID_DATA);
    packed[3] = 0xbf; /* Negative scale. */
    assert(nexa_q4_matmul(&one, 1, 1, packed, 5, 1, 1, 1, &output, 1) == NEXA_Q4_INVALID_DATA);
    packed[3] = 0x7f; /* Infinite scale. */
    assert(nexa_q4_matmul(&one, 1, 1, packed, 5, 1, 1, 1, &output, 1) == NEXA_Q4_INVALID_DATA);
    packed[2] = 0xc0; /* NaN scale. */
    assert(nexa_q4_matmul(&one, 1, 1, packed, 5, 1, 1, 1, &output, 1) == NEXA_Q4_INVALID_DATA);
    memset(packed, 0, 4); /* Zero scale with nonzero code. */
    assert(nexa_q4_matmul(&one, 1, 1, packed, 5, 1, 1, 1, &output, 1) == NEXA_Q4_INVALID_DATA);
    const float seven = 7.0f;
    assert(nexa_q4_quantize(&seven, 1, 1, 1, 1, packed, 5) == 0);
    const float huge = FLT_MAX;
    assert(nexa_q4_matmul(&huge, 1, 1, packed, 5, 1, 1, 1, &output, 1) == NEXA_Q4_NUMERIC_RANGE);
    const float nan = NAN;
    assert(nexa_q4_matmul(&nan, 1, 1, packed, 5, 1, 1, 1, &output, 1) == NEXA_Q4_INVALID_DATA);
    assert(nexa_q4_quantize(&nan, 1, 1, 1, 1, packed, 5) == NEXA_Q4_INVALID_DATA);
    uint32_t smallest_bits = 1;
    float smallest;
    memcpy(&smallest, &smallest_bits, sizeof(smallest));
    assert(nexa_q4_quantize(&smallest, 1, 1, 1, 1, packed, 5) == NEXA_Q4_NUMERIC_RANGE);
    assert(nexa_q4_quantize(&huge, 1, 1, 1, 1, packed, 5) == 0);
    const float half = 0.5f;
    assert(nexa_q4_matmul(&half, 1, 1, packed, 5, 1, 1, 1, &output, 1) == 0);
    assert(isfinite(output) && output > 0.0f);
    float overlap[4] = {7, 1, 2, 3};
    assert(nexa_q4_quantize(overlap, 1, 1, 1, 1, (uint8_t*)overlap, sizeof(overlap)) == NEXA_Q4_INVALID_ARGUMENT);
    assert(nexa_q4_matmul(overlap, 1, 1, packed, 5, 1, 1, 1, overlap, 1) == NEXA_Q4_INVALID_ARGUMENT);
}

int main(void) {
    sizes_and_argument_checks();
    golden_bytes_and_half_rounding();
    matrix_tails_and_batch();
    malformed_records_and_numeric_ranges();
    puts("Nexa Q4 CPU regressions passed");
    return 0;
}
