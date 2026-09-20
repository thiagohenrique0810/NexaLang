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
