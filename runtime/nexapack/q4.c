#include "q4.h"

#include <float.h>
#include <math.h>
#include <string.h>

_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 &&
               FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
               "Nexa Q4 requires IEEE-754 binary32 floats");

static int checked_mul(size_t a, size_t b, size_t *result) {
    if (b && a > SIZE_MAX / b) return 0;
    *result = a * b;
    return 1;
}

static int layout(size_t cols, size_t group_size, size_t *groups,
                  size_t *group_bytes, size_t *row_bytes) {
    if (!cols || !group_size) return NEXA_Q4_INVALID_ARGUMENT;
    size_t nibble_bytes = group_size / 2 + group_size % 2;
    if (nibble_bytes > SIZE_MAX - 4) return NEXA_Q4_OVERFLOW;
    *group_bytes = 4 + nibble_bytes;
    *groups = cols / group_size + (cols % group_size != 0);
    if (!checked_mul(*groups, *group_bytes, row_bytes)) return NEXA_Q4_OVERFLOW;
    return NEXA_Q4_OK;
}

size_t nexa_q4_row_size(size_t cols, size_t group_size) {
    size_t groups, group_bytes, row_bytes;
    return layout(cols, group_size, &groups, &group_bytes, &row_bytes) == 0
        ? row_bytes : 0;
}

size_t nexa_q4_size(size_t rows, size_t cols, size_t group_size) {
    size_t row_bytes = nexa_q4_row_size(cols, group_size), total;
    if (!rows || !row_bytes || !checked_mul(rows, row_bytes, &total)) return 0;
    return total;
}

static void store_scale(uint8_t *dst, float scale) {
    uint32_t bits;
    memcpy(&bits, &scale, sizeof(bits));
    dst[0] = (uint8_t)bits;
    dst[1] = (uint8_t)(bits >> 8);
    dst[2] = (uint8_t)(bits >> 16);
    dst[3] = (uint8_t)(bits >> 24);
}

static float load_scale(const uint8_t *src) {
    uint32_t bits = (uint32_t)src[0] | ((uint32_t)src[1] << 8) |
                    ((uint32_t)src[2] << 16) | ((uint32_t)src[3] << 24);
    float scale;
    memcpy(&scale, &bits, sizeof(scale));
    return scale;
}

/* Check address wrap as well as aliasing before reading/writing user buffers. */
static int disjoint(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    uintptr_t aa = (uintptr_t)a, bb = (uintptr_t)b;
    if (a_bytes > UINTPTR_MAX - aa || b_bytes > UINTPTR_MAX - bb) return 0;
    return aa + a_bytes <= bb || bb + b_bytes <= aa;
}

static int nibble_value(uint8_t byte, size_t index) {
    unsigned int nibble = (byte >> ((index % 2) * 4)) & 15u;
    return nibble < 8u ? (int)nibble : (int)nibble - 16;
}

static int validate_packed(const uint8_t *packed, size_t rows, size_t cols,
                           size_t group_size, size_t groups,
                           size_t group_bytes, size_t row_bytes) {
    for (size_t row = 0; row < rows; row++) {
        const uint8_t *record = packed + row * row_bytes;
        size_t start = 0;
        for (size_t group = 0; group < groups; group++) {
            const uint8_t *data = record + group * group_bytes;
            float scale = load_scale(data);
            if (!isfinite(scale) || scale < 0.0f) return NEXA_Q4_INVALID_DATA;
            size_t valid = cols - start < group_size ? cols - start : group_size;
            for (size_t byte = 0; byte < group_bytes - 4; byte++) {
                int low = nibble_value(data[4 + byte], 0);
                int high = nibble_value(data[4 + byte], 1);
                size_t low_index = byte * 2;
                if (low == -8 || high == -8) return NEXA_Q4_INVALID_DATA;
                if ((low_index >= valid || scale == 0.0f) && low != 0)
                    return NEXA_Q4_INVALID_DATA;
                if ((low_index + 1 >= valid || scale == 0.0f) && high != 0)
                    return NEXA_Q4_INVALID_DATA;
            }
            start += valid;
        }
    }
    return NEXA_Q4_OK;
}

int nexa_q4_quantize(const float *weights, size_t weight_count,
                    size_t rows, size_t cols, size_t group_size,
                    uint8_t *packed, size_t packed_bytes) {
    if (!weights || !packed || !rows) return NEXA_Q4_INVALID_ARGUMENT;
    size_t groups, group_bytes, row_bytes, total_bytes, elements, input_bytes;
    int status = layout(cols, group_size, &groups, &group_bytes, &row_bytes);
    if (status) return status;
    if (!checked_mul(rows, row_bytes, &total_bytes) ||
        !checked_mul(rows, cols, &elements) ||
        !checked_mul(elements, sizeof(float), &input_bytes)) return NEXA_Q4_OVERFLOW;
    if (weight_count < elements || packed_bytes < total_bytes)
        return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(weights, input_bytes, packed, total_bytes))
        return NEXA_Q4_INVALID_ARGUMENT;
    for (size_t i = 0; i < elements; i++) {
        if (!isfinite(weights[i])) return NEXA_Q4_INVALID_DATA;
    }
    for (size_t row = 0; row < rows; row++) {
        size_t start = 0;
        for (size_t group = 0; group < groups; group++) {
            size_t valid = cols - start < group_size ? cols - start : group_size;
            const float *values = weights + row * cols + start;
            uint8_t *record = packed + row * row_bytes + group * group_bytes;
            float maximum = 0.0f;
            for (size_t i = 0; i < valid; i++) {
                float magnitude = fabsf(values[i]);
                if (magnitude > maximum) maximum = magnitude;
            }
            float scale = (float)((double)maximum / 7.0);
            if (maximum > 0.0f && scale == 0.0f) return NEXA_Q4_NUMERIC_RANGE;
            store_scale(record, scale);
            memset(record + 4, 0, group_bytes - 4);
            if (scale > 0.0f) {
                for (size_t i = 0; i < valid; i++) {
                    double value = (double)values[i] / (double)scale;
                    int quantized;
                    if (value >= 7.0) quantized = 7;
                    else if (value <= -7.0) quantized = -7;
                    else quantized = (int)(value >= 0.0 ? value + 0.5 : value - 0.5);
                    unsigned int code = (unsigned int)(quantized < 0 ? quantized + 16 : quantized);
                    record[4 + i / 2] |= (uint8_t)(code << ((i % 2) * 4));
                }
            }
            start += valid;
        }
    }
    return NEXA_Q4_OK;
}

static int q3_layout(size_t cols, size_t group_size, size_t *groups,
                     size_t *group_bytes, size_t *row_bytes) {
    if (!cols || !group_size) return NEXA_Q4_INVALID_ARGUMENT;
    if (group_size > (SIZE_MAX - 7) / 3) return NEXA_Q4_OVERFLOW;
    size_t payload = (3 * group_size + 7) / 8;
    if (payload > SIZE_MAX - 4) return NEXA_Q4_OVERFLOW;
    *group_bytes = 4 + payload;  /* scale plus three bits per value */
    *groups = cols / group_size + (cols % group_size != 0);
    if (!checked_mul(*groups, *group_bytes, row_bytes)) return NEXA_Q4_OVERFLOW;
    return NEXA_Q4_OK;
}

/* Two or three bits never span more than two bytes: the shift is at most 7. */
static int packed_code(const uint8_t *payload, size_t payload_bytes, size_t index,
                       unsigned int bits) {
    size_t bit = index * bits, byte = bit >> 3;
    unsigned int shift = (unsigned int)(bit & 7u);
    unsigned int mask = (1u << bits) - 1u;
    unsigned int value = payload[byte];
    if (byte + 1 < payload_bytes) value |= (unsigned int)payload[byte + 1] << 8;
    value = (value >> shift) & mask;
    unsigned int sign = 1u << (bits - 1u);
    return value < sign ? (int)value : (int)value - (int)(sign << 1u);
}

static int q3_code(const uint8_t *payload, size_t payload_bytes, size_t index) {
    return packed_code(payload, payload_bytes, index, 3u);
}

static int validate_packed_q3(const uint8_t *packed, size_t rows, size_t cols,
                              size_t group_size, size_t groups,
                              size_t group_bytes, size_t row_bytes) {
    size_t payload_bytes = group_bytes - 4;
    size_t used_bits = 3 * group_size;
    for (size_t row = 0; row < rows; row++) {
        const uint8_t *record = packed + row * row_bytes;
        size_t start = 0;
        for (size_t group = 0; group < groups; group++) {
            const uint8_t *data = record + group * group_bytes;
            float scale = load_scale(data);
            if (!isfinite(scale) || scale < 0.0f) return NEXA_Q4_INVALID_DATA;
            size_t valid = cols - start < group_size ? cols - start : group_size;
            for (size_t index = 0; index < group_size; index++) {
                int value = q3_code(data + 4, payload_bytes, index);
                if (value == -4) return NEXA_Q4_INVALID_DATA;
                if ((index >= valid || scale == 0.0f) && value != 0)
                    return NEXA_Q4_INVALID_DATA;
            }
            /* Bits past the last code must be zero, like the writer emits. */
            for (size_t bit = used_bits; bit < payload_bytes * 8; bit++) {
                if ((data[4 + (bit >> 3)] >> (bit & 7u)) & 1u) return NEXA_Q4_INVALID_DATA;
            }
            start += valid;
        }
    }
    return NEXA_Q4_OK;
}

int nexa_q3_decode_row(const uint8_t *packed, size_t packed_bytes,
                       size_t cols, size_t group_size,
                       float *output, size_t output_count) {
    if (!packed || !output) return NEXA_Q4_INVALID_ARGUMENT;
    size_t groups, group_bytes, row_bytes, output_bytes;
    int status = q3_layout(cols, group_size, &groups, &group_bytes, &row_bytes);
    if (status) return status;
    if (!checked_mul(cols, sizeof(float), &output_bytes)) return NEXA_Q4_OVERFLOW;
    if (packed_bytes < row_bytes || output_count < cols) return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(packed, row_bytes, output, output_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    status = validate_packed_q3(packed, 1, cols, group_size, groups, group_bytes, row_bytes);
    if (status) return status;
    size_t start = 0;
    for (size_t group = 0; group < groups; group++) {
        const uint8_t *data = packed + group * group_bytes;
        double scale = (double)load_scale(data);
        size_t valid = cols - start < group_size ? cols - start : group_size;
        for (size_t lane = 0; lane < valid; lane++) {
            double value = (double)q3_code(data + 4, group_bytes - 4, lane) * scale;
            if (!isfinite(value) || fabs(value) > FLT_MAX) return NEXA_Q4_NUMERIC_RANGE;
            output[start + lane] = (float)value;
        }
        start += valid;
    }
    return NEXA_Q4_OK;
}

/* Same contract and reduction order as the other packed matmuls. */
int nexa_q3_matmul(const float *inputs, size_t input_count, size_t batch,
                   const uint8_t *packed, size_t packed_bytes,
                   size_t rows, size_t cols, size_t group_size,
                   float *output, size_t output_count) {
    if (!inputs || !packed || !output || !rows || !batch)
        return NEXA_Q4_INVALID_ARGUMENT;
    size_t groups, group_bytes, row_bytes, total_bytes;
    size_t input_elements, output_elements, input_bytes, output_bytes;
    int status = q3_layout(cols, group_size, &groups, &group_bytes, &row_bytes);
    if (status) return status;
    if (!checked_mul(rows, row_bytes, &total_bytes) ||
        !checked_mul(batch, cols, &input_elements) ||
        !checked_mul(batch, rows, &output_elements) ||
        !checked_mul(input_elements, sizeof(float), &input_bytes) ||
        !checked_mul(output_elements, sizeof(float), &output_bytes)) return NEXA_Q4_OVERFLOW;
    if (input_count < input_elements || packed_bytes < total_bytes ||
        output_count < output_elements) return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(inputs, input_bytes, output, output_bytes) ||
        !disjoint(packed, total_bytes, output, output_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    for (size_t i = 0; i < input_elements; i++) {
        if (!isfinite(inputs[i])) return NEXA_Q4_INVALID_DATA;
    }
    status = validate_packed_q3(packed, rows, cols, group_size, groups, group_bytes, row_bytes);
    if (status) return status;
    for (size_t item = 0; item < batch; item++) {
        const float *input = inputs + item * cols;
        for (size_t row = 0; row < rows; row++) {
            const uint8_t *record = packed + row * row_bytes;
            double sum = 0.0;
            size_t start = 0;
            for (size_t group = 0; group < groups; group++) {
                const uint8_t *data = record + group * group_bytes;
                double scale = (double)load_scale(data);
                size_t valid = cols - start < group_size ? cols - start : group_size;
                for (size_t i = 0; i < valid; i++) {
                    int value = q3_code(data + 4, group_bytes - 4, i);
                    sum += (double)input[start + i] * ((double)value * scale);
                }
                start += valid;
            }
            if (!isfinite(sum) || fabs(sum) > FLT_MAX) return NEXA_Q4_NUMERIC_RANGE;
            output[item * rows + row] = (float)sum;
        }
    }
    return NEXA_Q4_OK;
}

static int q2_layout(size_t cols, size_t group_size, size_t *groups,
                     size_t *group_bytes, size_t *row_bytes) {
    if (!cols || !group_size) return NEXA_Q4_INVALID_ARGUMENT;
    if (group_size > (SIZE_MAX - 7) / 2) return NEXA_Q4_OVERFLOW;
    size_t payload = (2 * group_size + 7) / 8;
    if (payload > SIZE_MAX - 4) return NEXA_Q4_OVERFLOW;
    *group_bytes = 4 + payload;  /* scale plus two bits per value */
    *groups = cols / group_size + (cols % group_size != 0);
    if (!checked_mul(*groups, *group_bytes, row_bytes)) return NEXA_Q4_OVERFLOW;
    return NEXA_Q4_OK;
}

static int validate_packed_q2(const uint8_t *packed, size_t rows, size_t cols,
                              size_t group_size, size_t groups,
                              size_t group_bytes, size_t row_bytes) {
    size_t payload_bytes = group_bytes - 4, used_bits = 2 * group_size;
    for (size_t row = 0; row < rows; row++) {
        const uint8_t *record = packed + row * row_bytes;
        size_t start = 0;
        for (size_t group = 0; group < groups; group++) {
            const uint8_t *data = record + group * group_bytes;
            float scale = load_scale(data);
            if (!isfinite(scale) || scale < 0.0f) return NEXA_Q4_INVALID_DATA;
            size_t valid = cols - start < group_size ? cols - start : group_size;
            for (size_t index = 0; index < group_size; index++) {
                int value = packed_code(data + 4, payload_bytes, index, 2u);
                if (value == -2) return NEXA_Q4_INVALID_DATA;  /* reserved */
                if ((index >= valid || scale == 0.0f) && value != 0)
                    return NEXA_Q4_INVALID_DATA;
            }
            for (size_t bit = used_bits; bit < payload_bytes * 8; bit++) {
                if ((data[4 + (bit >> 3)] >> (bit & 7u)) & 1u) return NEXA_Q4_INVALID_DATA;
            }
            start += valid;
        }
    }
    return NEXA_Q4_OK;
}

int nexa_q2_decode_row(const uint8_t *packed, size_t packed_bytes,
                       size_t cols, size_t group_size,
                       float *output, size_t output_count) {
    if (!packed || !output) return NEXA_Q4_INVALID_ARGUMENT;
    size_t groups, group_bytes, row_bytes, output_bytes;
    int status = q2_layout(cols, group_size, &groups, &group_bytes, &row_bytes);
    if (status) return status;
    if (!checked_mul(cols, sizeof(float), &output_bytes)) return NEXA_Q4_OVERFLOW;
    if (packed_bytes < row_bytes || output_count < cols) return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(packed, row_bytes, output, output_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    status = validate_packed_q2(packed, 1, cols, group_size, groups, group_bytes, row_bytes);
    if (status) return status;
    size_t start = 0;
    for (size_t group = 0; group < groups; group++) {
        const uint8_t *data = packed + group * group_bytes;
        double scale = (double)load_scale(data);
        size_t valid = cols - start < group_size ? cols - start : group_size;
        for (size_t lane = 0; lane < valid; lane++) {
            double value = (double)packed_code(data + 4, group_bytes - 4, lane, 2u) * scale;
            if (!isfinite(value) || fabs(value) > FLT_MAX) return NEXA_Q4_NUMERIC_RANGE;
            output[start + lane] = (float)value;
        }
        start += valid;
    }
    return NEXA_Q4_OK;
}

int nexa_q2_matmul(const float *inputs, size_t input_count, size_t batch,
                   const uint8_t *packed, size_t packed_bytes,
                   size_t rows, size_t cols, size_t group_size,
                   float *output, size_t output_count) {
    if (!inputs || !packed || !output || !rows || !batch)
        return NEXA_Q4_INVALID_ARGUMENT;
    size_t groups, group_bytes, row_bytes, total_bytes;
    size_t input_elements, output_elements, input_bytes, output_bytes;
    int status = q2_layout(cols, group_size, &groups, &group_bytes, &row_bytes);
    if (status) return status;
    if (!checked_mul(rows, row_bytes, &total_bytes) ||
        !checked_mul(batch, cols, &input_elements) ||
        !checked_mul(batch, rows, &output_elements) ||
        !checked_mul(input_elements, sizeof(float), &input_bytes) ||
        !checked_mul(output_elements, sizeof(float), &output_bytes)) return NEXA_Q4_OVERFLOW;
    if (input_count < input_elements || packed_bytes < total_bytes ||
        output_count < output_elements) return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(inputs, input_bytes, output, output_bytes) ||
        !disjoint(packed, total_bytes, output, output_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    for (size_t i = 0; i < input_elements; i++) {
        if (!isfinite(inputs[i])) return NEXA_Q4_INVALID_DATA;
    }
    status = validate_packed_q2(packed, rows, cols, group_size, groups, group_bytes, row_bytes);
    if (status) return status;
    for (size_t item = 0; item < batch; item++) {
        const float *input = inputs + item * cols;
        for (size_t row = 0; row < rows; row++) {
            const uint8_t *record = packed + row * row_bytes;
            double sum = 0.0;
            size_t start = 0;
            for (size_t group = 0; group < groups; group++) {
                const uint8_t *data = record + group * group_bytes;
                double scale = (double)load_scale(data);
                size_t valid = cols - start < group_size ? cols - start : group_size;
                for (size_t i = 0; i < valid; i++) {
                    int value = packed_code(data + 4, group_bytes - 4, i, 2u);
                    sum += (double)input[start + i] * ((double)value * scale);
                }
                start += valid;
            }
            if (!isfinite(sum) || fabs(sum) > FLT_MAX) return NEXA_Q4_NUMERIC_RANGE;
            output[item * rows + row] = (float)sum;
        }
    }
    return NEXA_Q4_OK;
}

static int q8_layout(size_t cols, size_t group_size, size_t *groups,
                     size_t *group_bytes, size_t *row_bytes) {
    if (!cols || !group_size) return NEXA_Q4_INVALID_ARGUMENT;
    if (group_size > SIZE_MAX - 4) return NEXA_Q4_OVERFLOW;
    *group_bytes = 4 + group_size;  /* scale plus one signed byte per value */
    *groups = cols / group_size + (cols % group_size != 0);
    if (!checked_mul(*groups, *group_bytes, row_bytes)) return NEXA_Q4_OVERFLOW;
    return NEXA_Q4_OK;
}

size_t nexa_q8_row_size(size_t cols, size_t group_size) {
    size_t groups, group_bytes, row_bytes;
    return q8_layout(cols, group_size, &groups, &group_bytes, &row_bytes) == 0 ? row_bytes : 0;
}

static int validate_packed_q8(const uint8_t *packed, size_t rows, size_t cols,
                              size_t group_size, size_t groups,
                              size_t group_bytes, size_t row_bytes) {
    for (size_t row = 0; row < rows; row++) {
        const uint8_t *record = packed + row * row_bytes;
        size_t start = 0;
        for (size_t group = 0; group < groups; group++) {
            const uint8_t *data = record + group * group_bytes;
            float scale = load_scale(data);
            if (!isfinite(scale) || scale < 0.0f) return NEXA_Q4_INVALID_DATA;
            size_t valid = cols - start < group_size ? cols - start : group_size;
            for (size_t index = 0; index < group_size; index++) {
                int value = (int8_t)data[4 + index];
                /* -128 has no positive counterpart; the writer never emits it. */
                if (value == -128) return NEXA_Q4_INVALID_DATA;
                if ((index >= valid || scale == 0.0f) && value != 0)
                    return NEXA_Q4_INVALID_DATA;
            }
            start += valid;
        }
    }
    return NEXA_Q4_OK;
}

/* Same contract, reduction order and accumulator as nexa_q4_matmul, over
 * one signed byte per coordinate. Nothing is dequantized into a buffer. */
int nexa_q8_matmul(const float *inputs, size_t input_count, size_t batch,
                   const uint8_t *packed, size_t packed_bytes,
                   size_t rows, size_t cols, size_t group_size,
                   float *output, size_t output_count) {
    if (!inputs || !packed || !output || !rows || !batch)
        return NEXA_Q4_INVALID_ARGUMENT;
    size_t groups, group_bytes, row_bytes, total_bytes;
    size_t input_elements, output_elements, input_bytes, output_bytes;
    int status = q8_layout(cols, group_size, &groups, &group_bytes, &row_bytes);
    if (status) return status;
    if (!checked_mul(rows, row_bytes, &total_bytes) ||
        !checked_mul(batch, cols, &input_elements) ||
        !checked_mul(batch, rows, &output_elements) ||
        !checked_mul(input_elements, sizeof(float), &input_bytes) ||
        !checked_mul(output_elements, sizeof(float), &output_bytes)) return NEXA_Q4_OVERFLOW;
    if (input_count < input_elements || packed_bytes < total_bytes ||
        output_count < output_elements) return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(inputs, input_bytes, output, output_bytes) ||
        !disjoint(packed, total_bytes, output, output_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    for (size_t i = 0; i < input_elements; i++) {
        if (!isfinite(inputs[i])) return NEXA_Q4_INVALID_DATA;
    }
    status = validate_packed_q8(packed, rows, cols, group_size, groups, group_bytes, row_bytes);
    if (status) return status;
    for (size_t item = 0; item < batch; item++) {
        const float *input = inputs + item * cols;
        for (size_t row = 0; row < rows; row++) {
            const uint8_t *record = packed + row * row_bytes;
            double sum = 0.0;
            size_t start = 0;
            for (size_t group = 0; group < groups; group++) {
                const uint8_t *data = record + group * group_bytes;
                double scale = (double)load_scale(data);
                size_t valid = cols - start < group_size ? cols - start : group_size;
                for (size_t i = 0; i < valid; i++) {
                    int value = (int8_t)data[4 + i];
                    sum += (double)input[start + i] * ((double)value * scale);
                }
                start += valid;
            }
            if (!isfinite(sum) || fabs(sum) > FLT_MAX) return NEXA_Q4_NUMERIC_RANGE;
            output[item * rows + row] = (float)sum;
        }
    }
    return NEXA_Q4_OK;
}

int nexa_q4_matmul(const float *inputs, size_t input_count, size_t batch,
                  const uint8_t *packed, size_t packed_bytes,
                  size_t rows, size_t cols, size_t group_size,
                  float *output, size_t output_count) {
    if (!inputs || !packed || !output || !rows || !batch)
        return NEXA_Q4_INVALID_ARGUMENT;
    size_t groups, group_bytes, row_bytes, total_bytes;
    size_t input_elements, output_elements, input_bytes, output_bytes;
    int status = layout(cols, group_size, &groups, &group_bytes, &row_bytes);
    if (status) return status;
    if (!checked_mul(rows, row_bytes, &total_bytes) ||
        !checked_mul(batch, cols, &input_elements) ||
        !checked_mul(batch, rows, &output_elements) ||
        !checked_mul(input_elements, sizeof(float), &input_bytes) ||
        !checked_mul(output_elements, sizeof(float), &output_bytes)) return NEXA_Q4_OVERFLOW;
    if (input_count < input_elements || packed_bytes < total_bytes ||
        output_count < output_elements) return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(inputs, input_bytes, output, output_bytes) ||
        !disjoint(packed, total_bytes, output, output_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    for (size_t i = 0; i < input_elements; i++) {
        if (!isfinite(inputs[i])) return NEXA_Q4_INVALID_DATA;
    }
    status = validate_packed(packed, rows, cols, group_size, groups, group_bytes, row_bytes);
    if (status) return status;
    for (size_t item = 0; item < batch; item++) {
        const float *input = inputs + item * cols;
        for (size_t row = 0; row < rows; row++) {
            const uint8_t *record = packed + row * row_bytes;
            double sum = 0.0;
            size_t start = 0;
            for (size_t group = 0; group < groups; group++) {
                const uint8_t *data = record + group * group_bytes;
                double scale = (double)load_scale(data);
                size_t valid = cols - start < group_size ? cols - start : group_size;
                for (size_t i = 0; i < valid; i++) {
                    int value = nibble_value(data[4 + i / 2], i);
                    sum += (double)input[start + i] * ((double)value * scale);
                }
                start += valid;
            }
            if (!isfinite(sum) || fabs(sum) > FLT_MAX) return NEXA_Q4_NUMERIC_RANGE;
            output[item * rows + row] = (float)sum;
        }
    }
    return NEXA_Q4_OK;
}

int nexa_q8_decode_row(const uint8_t *packed, size_t packed_bytes,
                       size_t cols, size_t group_size,
                       float *output, size_t output_count) {
    if (!packed || !output) return NEXA_Q4_INVALID_ARGUMENT;
    size_t groups, group_bytes, row_bytes, output_bytes;
    int status = q8_layout(cols, group_size, &groups, &group_bytes, &row_bytes);
    if (status) return status;
    if (!checked_mul(cols, sizeof(float), &output_bytes)) return NEXA_Q4_OVERFLOW;
    if (packed_bytes < row_bytes || output_count < cols) return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(packed, row_bytes, output, output_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    status = validate_packed_q8(packed, 1, cols, group_size, groups, group_bytes, row_bytes);
    if (status) return status;
    size_t start = 0;
    for (size_t group = 0; group < groups; group++) {
        const uint8_t *data = packed + group * group_bytes;
        double scale = (double)load_scale(data);
        size_t valid = cols - start < group_size ? cols - start : group_size;
        for (size_t lane = 0; lane < valid; lane++) {
            double value = (double)(int8_t)data[4 + lane] * scale;
            if (!isfinite(value) || fabs(value) > FLT_MAX) return NEXA_Q4_NUMERIC_RANGE;
            output[start + lane] = (float)value;
        }
        start += valid;
    }
    return NEXA_Q4_OK;
}

int nexa_q4_decode_row(const uint8_t *packed, size_t packed_bytes,
                       size_t cols, size_t group_size,
                       float *output, size_t output_count) {
    if (!packed || !output) return NEXA_Q4_INVALID_ARGUMENT;
    size_t groups, group_bytes, row_bytes, output_bytes;
    int status = layout(cols, group_size, &groups, &group_bytes, &row_bytes);
    if (status) return status;
    if (!checked_mul(cols, sizeof(float), &output_bytes)) return NEXA_Q4_OVERFLOW;
    if (packed_bytes < row_bytes || output_count < cols) return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(packed, row_bytes, output, output_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    status = validate_packed(packed, 1, cols, group_size, groups, group_bytes, row_bytes);
    if (status) return status;
    size_t start = 0;
    for (size_t group = 0; group < groups; group++) {
        const uint8_t *data = packed + group * group_bytes;
        double scale = (double)load_scale(data);
        size_t valid = cols - start < group_size ? cols - start : group_size;
        for (size_t lane = 0; lane < valid; lane++) {
            double value = (double)nibble_value(data[4 + lane / 2], lane) * scale;
            if (!isfinite(value) || fabs(value) > FLT_MAX) return NEXA_Q4_NUMERIC_RANGE;
            output[start + lane] = (float)value;
        }
        start += valid;
    }
    return NEXA_Q4_OK;
}

/* ── qint<N>/PackedVector<N> storage: one layout for every grouped width ──
 *
 * Q2, Q3, Q4 and Q8 are not four layouts, they are one: a little-endian F32
 * scale followed by N-bit two's-complement codes packed from the least
 * significant bit, with the tail of the last group left at zero. Writing a
 * single kernel over `bits` is what lets the language guarantee that what it
 * emits is byte-identical to what the Python writers emit; four hand-written
 * packers would only be *likely* to agree.
 */

static int qpack_layout(size_t bits, size_t count, size_t group_size,
                        size_t *groups, size_t *group_bytes, size_t *total) {
    if (bits != 2 && bits != 3 && bits != 4 && bits != 8)
        return NEXA_Q4_INVALID_ARGUMENT;
    if (!count || !group_size) return NEXA_Q4_INVALID_ARGUMENT;
    if (group_size > (SIZE_MAX - 7) / bits) return NEXA_Q4_OVERFLOW;
    size_t payload = (bits * group_size + 7) / 8;
    if (payload > SIZE_MAX - 4) return NEXA_Q4_OVERFLOW;
    *group_bytes = 4 + payload;
    *groups = count / group_size + (count % group_size != 0);
    if (!checked_mul(*groups, *group_bytes, total)) return NEXA_Q4_OVERFLOW;
    return NEXA_Q4_OK;
}

/* The payload is pre-zeroed, so writing means setting the bits that are one. */
static void store_packed_code(uint8_t *payload, size_t index, unsigned int bits,
                              unsigned int code) {
    size_t bit = index * bits;
    for (unsigned int offset = 0; offset < bits; offset++) {
        if ((code >> offset) & 1u)
            payload[(bit + offset) >> 3] |= (uint8_t)(1u << ((bit + offset) & 7u));
    }
}

size_t nexa_qpack_size(size_t bits, size_t count, size_t group_size) {
    size_t groups, group_bytes, total;
    return qpack_layout(bits, count, group_size, &groups, &group_bytes, &total) == 0
        ? total : 0;
}

size_t nexa_qpack_groups(size_t bits, size_t count, size_t group_size) {
    size_t groups, group_bytes, total;
    return qpack_layout(bits, count, group_size, &groups, &group_bytes, &total) == 0
        ? groups : 0;
}

/* Scale is max|v| / (2^(N-1) - 1) rounded to F32, values are divided by that
 * stored scale and rounded half away from zero. The reserved -2^(N-1) code is
 * unreachable because the magnitude is clamped before the sign is applied.
 * That clamp is defensive only: with a scale derived from the group's own
 * maximum, the largest quotient is levels*(1 + 2^-24) and still rounds to
 * levels, so no finite input can drive the magnitude past it. */
int nexa_qpack_pack(size_t bits, const float *values, size_t value_count,
                    size_t count, size_t group_size,
                    uint8_t *packed, size_t packed_bytes) {
    if (!values || !packed) return NEXA_Q4_INVALID_ARGUMENT;
    size_t groups, group_bytes, total, value_bytes;
    int status = qpack_layout(bits, count, group_size, &groups, &group_bytes, &total);
    if (status) return status;
    if (!checked_mul(count, sizeof(float), &value_bytes)) return NEXA_Q4_OVERFLOW;
    if (value_count < count || packed_bytes < total) return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(values, value_bytes, packed, total)) return NEXA_Q4_INVALID_ARGUMENT;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(values[i])) return NEXA_Q4_INVALID_DATA;
    }
    double levels = (double)((1u << (bits - 1u)) - 1u);
    memset(packed, 0, total);
    size_t start = 0;
    for (size_t group = 0; group < groups; group++) {
        size_t valid = count - start < group_size ? count - start : group_size;
        const float *lane = values + start;
        uint8_t *record = packed + group * group_bytes;
        float maximum = 0.0f;
        for (size_t i = 0; i < valid; i++) {
            float magnitude = fabsf(lane[i]);
            if (magnitude > maximum) maximum = magnitude;
        }
        float scale = (float)((double)maximum / levels);
        if (maximum > 0.0f && scale == 0.0f) return NEXA_Q4_NUMERIC_RANGE;
        store_scale(record, scale);
        if (scale > 0.0f) {
            for (size_t i = 0; i < valid; i++) {
                double quotient = (double)lane[i] / (double)scale;
                double magnitude = floor(fabs(quotient) + 0.5);
                if (magnitude > levels) magnitude = levels;
                int quantized = (int)magnitude;
                if (quotient < 0.0) quantized = -quantized;
                unsigned int code = (unsigned int)quantized & ((1u << bits) - 1u);
                store_packed_code(record + 4, i, (unsigned int)bits, code);
            }
        }
        start += valid;
    }
    return NEXA_Q4_OK;
}

/* Decoding reuses the per-codec row kernels, so a language unpack and a
 * runtime decode cannot drift apart into two different validation rules. */
int nexa_qpack_unpack(size_t bits, const uint8_t *packed, size_t packed_bytes,
                      size_t count, size_t group_size,
                      float *output, size_t output_count) {
    switch (bits) {
        case 2: return nexa_q2_decode_row(packed, packed_bytes, count, group_size,
                                          output, output_count);
        case 3: return nexa_q3_decode_row(packed, packed_bytes, count, group_size,
                                          output, output_count);
        case 4: return nexa_q4_decode_row(packed, packed_bytes, count, group_size,
                                          output, output_count);
        case 8: return nexa_q8_decode_row(packed, packed_bytes, count, group_size,
                                          output, output_count);
        default: return NEXA_Q4_INVALID_ARGUMENT;
    }
}

int nexa_qpack_code(size_t bits, const uint8_t *packed, size_t packed_bytes,
                    size_t count, size_t group_size, size_t index, int8_t *code) {
    if (!packed || !code) return NEXA_Q4_INVALID_ARGUMENT;
    size_t groups, group_bytes, total;
    int status = qpack_layout(bits, count, group_size, &groups, &group_bytes, &total);
    if (status) return status;
    if (index >= count) return NEXA_Q4_INVALID_ARGUMENT;
    if (packed_bytes < total) return NEXA_Q4_BUFFER_TOO_SMALL;
    const uint8_t *record = packed + (index / group_size) * group_bytes;
    float scale = load_scale(record);
    if (!isfinite(scale) || scale < 0.0f) return NEXA_Q4_INVALID_DATA;
    int value = packed_code(record + 4, group_bytes - 4, index % group_size,
                            (unsigned int)bits);
    if (value == -(int)(1u << (bits - 1u))) return NEXA_Q4_INVALID_DATA;
    if (scale == 0.0f && value != 0) return NEXA_Q4_INVALID_DATA;
    *code = (int8_t)value;
    return NEXA_Q4_OK;
}

int nexa_qpack_scale(size_t bits, const uint8_t *packed, size_t packed_bytes,
                     size_t count, size_t group_size, size_t group, float *scale) {
    if (!packed || !scale) return NEXA_Q4_INVALID_ARGUMENT;
    size_t groups, group_bytes, total;
    int status = qpack_layout(bits, count, group_size, &groups, &group_bytes, &total);
    if (status) return status;
    if (group >= groups) return NEXA_Q4_INVALID_ARGUMENT;
    if (packed_bytes < total) return NEXA_Q4_BUFFER_TOO_SMALL;
    float value = load_scale(packed + group * group_bytes);
    if (!isfinite(value) || value < 0.0f) return NEXA_Q4_INVALID_DATA;
    *scale = value;
    return NEXA_Q4_OK;
}
