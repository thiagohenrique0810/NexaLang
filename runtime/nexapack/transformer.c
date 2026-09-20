#include "transformer.h"

#include <float.h>
#include <math.h>
#include <string.h>

static int checked_mul(size_t left, size_t right, size_t *result) {
    if (right && left > SIZE_MAX / right) return 0;
    *result = left * right;
    return 1;
}

static int float_bytes(size_t count, size_t *bytes) {
    return checked_mul(count, sizeof(float), bytes);
}

static int disjoint(const void *left, size_t left_bytes,
                    const void *right, size_t right_bytes) {
    uintptr_t l = (uintptr_t)left, r = (uintptr_t)right;
    if (left_bytes > UINTPTR_MAX - l || right_bytes > UINTPTR_MAX - r) return 0;
    return l + left_bytes <= r || r + right_bytes <= l;
}

static int finite_input(const float *values, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(values[i])) return 0;
    }
    return 1;
}

static int write_float(float *output, double value) {
    if (!isfinite(value) || fabs(value) > FLT_MAX) return NEXA_Q4_NUMERIC_RANGE;
    *output = (float)value;
    return NEXA_Q4_OK;
}

static int binary_buffers(const float *left, size_t left_count,
                          const float *right, size_t right_count,
                          size_t elements, float *output, size_t output_count) {
    if (!left || !right || !output || !elements) return NEXA_Q4_INVALID_ARGUMENT;
    size_t bytes;
    if (!float_bytes(elements, &bytes)) return NEXA_Q4_OVERFLOW;
    if (left_count < elements || right_count < elements || output_count < elements)
        return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(left, bytes, output, bytes) || !disjoint(right, bytes, output, bytes))
        return NEXA_Q4_INVALID_ARGUMENT;
    if (!finite_input(left, elements) || !finite_input(right, elements))
        return NEXA_Q4_INVALID_DATA;
    return NEXA_Q4_OK;
}

/* IEEE-754 binary16 to float, without assuming the compiler has _Float16.
 * Subnormals and zero are handled explicitly; inf/NaN are rejected upstream. */
static float half_to_float(uint16_t bits) {
    unsigned int sign = (bits >> 15) & 1u;
    unsigned int exponent = (bits >> 10) & 0x1Fu;
    unsigned int mantissa = bits & 0x3FFu;
    double value;
    if (!exponent) {
        value = ldexp((double)mantissa, -24);
    } else if (exponent == 0x1Fu) {
        value = mantissa ? NAN : INFINITY;
    } else {
        value = ldexp((double)(mantissa | 0x400u), (int)exponent - 25);
    }
    return (float)(sign ? -value : value);
}

static uint16_t load_half(const uint8_t *source) {
    return (uint16_t)((unsigned int)source[0] | ((unsigned int)source[1] << 8));
}

/* Dense F16 weights: same contract and reduction order as the F32 kernel,
 * reading two bytes per coordinate without expanding the tile. */
int nexa_f16_matmul(const float *inputs, size_t input_count, size_t batch,
                    const uint8_t *weights, size_t weight_bytes,
                    size_t rows, size_t cols,
                    float *output, size_t output_count) {
    if (!inputs || !weights || !output || !rows || !cols || !batch)
        return NEXA_Q4_INVALID_ARGUMENT;
    size_t weight_elements, needed, input_elements, output_elements;
    size_t input_bytes, output_bytes;
    if (!checked_mul(rows, cols, &weight_elements) ||
        !checked_mul(weight_elements, 2, &needed) ||
        !checked_mul(batch, cols, &input_elements) ||
        !checked_mul(batch, rows, &output_elements) ||
        !float_bytes(input_elements, &input_bytes) ||
        !float_bytes(output_elements, &output_bytes)) return NEXA_Q4_OVERFLOW;
    if (input_count < input_elements || weight_bytes < needed ||
        output_count < output_elements) return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(inputs, input_bytes, output, output_bytes) ||
        !disjoint(weights, needed, output, output_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    if (!finite_input(inputs, input_elements)) return NEXA_Q4_INVALID_DATA;
    for (size_t i = 0; i < weight_elements; i++) {
        if (!isfinite(half_to_float(load_half(weights + i * 2)))) return NEXA_Q4_INVALID_DATA;
    }
    for (size_t item = 0; item < batch; item++) {
        const float *input = inputs + item * cols;
        for (size_t row = 0; row < rows; row++) {
            const uint8_t *record = weights + row * cols * 2;
            double sum = 0.0;
            for (size_t i = 0; i < cols; i++)
                sum += (double)input[i] * (double)half_to_float(load_half(record + i * 2));
            int status = write_float(output + item * rows + row, sum);
            if (status) return status;
        }
    }
    return NEXA_Q4_OK;
}

int nexa_f16_decode_row(const uint8_t *weights, size_t weight_bytes,
                        size_t cols, float *output, size_t output_count) {
    if (!weights || !output || !cols) return NEXA_Q4_INVALID_ARGUMENT;
    size_t needed, output_bytes;
    if (!checked_mul(cols, 2, &needed) || !float_bytes(cols, &output_bytes))
        return NEXA_Q4_OVERFLOW;
    if (weight_bytes < needed || output_count < cols) return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(weights, needed, output, output_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    for (size_t i = 0; i < cols; i++) {
        float value = half_to_float(load_half(weights + i * 2));
        if (!isfinite(value)) return NEXA_Q4_INVALID_DATA;
        output[i] = value;
    }
    return NEXA_Q4_OK;
}

/* Dense F32 weights, same reduction order and accumulator as nexa_q4_matmul,
 * so a tensor kept in F32 differs from its packed form only by quantization.
 * The weight tile is read from the bundle; nothing is dequantized or copied. */
int nexa_f32_matmul(const float *inputs, size_t input_count, size_t batch,
                    const float *weights, size_t weight_count,
                    size_t rows, size_t cols,
                    float *output, size_t output_count) {
    if (!inputs || !weights || !output || !rows || !cols || !batch)
        return NEXA_Q4_INVALID_ARGUMENT;
    size_t weight_elements, input_elements, output_elements;
    size_t weight_bytes, input_bytes, output_bytes;
    if (!checked_mul(rows, cols, &weight_elements) ||
        !checked_mul(batch, cols, &input_elements) ||
        !checked_mul(batch, rows, &output_elements) ||
        !float_bytes(weight_elements, &weight_bytes) ||
        !float_bytes(input_elements, &input_bytes) ||
        !float_bytes(output_elements, &output_bytes)) return NEXA_Q4_OVERFLOW;
    if (input_count < input_elements || weight_count < weight_elements ||
        output_count < output_elements) return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(inputs, input_bytes, output, output_bytes) ||
        !disjoint(weights, weight_bytes, output, output_bytes))
        return NEXA_Q4_INVALID_ARGUMENT;
    if (!finite_input(inputs, input_elements) || !finite_input(weights, weight_elements))
        return NEXA_Q4_INVALID_DATA;
    for (size_t item = 0; item < batch; item++) {
        const float *input = inputs + item * cols;
        for (size_t row = 0; row < rows; row++) {
            const float *record = weights + row * cols;
            double sum = 0.0;
            for (size_t i = 0; i < cols; i++) sum += (double)input[i] * (double)record[i];
            int status = write_float(output + item * rows + row, sum);
            if (status) return status;
        }
    }
    return NEXA_Q4_OK;
}

int nexa_add(const float *left, size_t left_count,
             const float *right, size_t right_count,
             size_t elements, float *output, size_t output_count) {
    int status = binary_buffers(left, left_count, right, right_count,
                                elements, output, output_count);
    if (status) return status;
    for (size_t i = 0; i < elements; i++) {
        status = write_float(output + i, (double)left[i] + (double)right[i]);
        if (status) return status;
    }
    return NEXA_Q4_OK;
}

int nexa_swiglu(const float *gate, size_t gate_count,
                const float *up, size_t up_count,
                size_t elements, float *output, size_t output_count) {
    int status = binary_buffers(gate, gate_count, up, up_count,
                                elements, output, output_count);
    if (status) return status;
    for (size_t i = 0; i < elements; i++) {
        double value = gate[i];
        double exponential = exp(value >= 0.0 ? -value : value);
        double sigmoid = value >= 0.0 ? 1.0 / (1.0 + exponential)
                                     : exponential / (1.0 + exponential);
        status = write_float(output + i, (value * sigmoid) * (double)up[i]);
        if (status) return status;
    }
    return NEXA_Q4_OK;
}

int nexa_rmsnorm(const float *input, size_t input_count,
                 const float *weight, size_t weight_count,
                 size_t sequence, size_t hidden, double epsilon,
                 float *output, size_t output_count) {
    if (!input || !weight || !output || !sequence || !hidden ||
        !isfinite(epsilon) || epsilon <= 0.0) return NEXA_Q4_INVALID_ARGUMENT;
    size_t elements, input_bytes, weight_bytes;
    if (!checked_mul(sequence, hidden, &elements) ||
        !float_bytes(elements, &input_bytes) || !float_bytes(hidden, &weight_bytes))
        return NEXA_Q4_OVERFLOW;
    if (input_count < elements || weight_count < hidden || output_count < elements)
        return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(input, input_bytes, output, input_bytes) ||
        !disjoint(weight, weight_bytes, output, input_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    if (!finite_input(input, elements) || !finite_input(weight, hidden)) return NEXA_Q4_INVALID_DATA;
    for (size_t row = 0; row < sequence; row++) {
        const float *values = input + row * hidden;
        double sum = 0.0;
        for (size_t i = 0; i < hidden; i++) sum += (double)values[i] * (double)values[i];
        double factor = 1.0 / sqrt(sum / (double)hidden + epsilon);
        for (size_t i = 0; i < hidden; i++) {
            int status = write_float(output + row * hidden + i,
                                     ((double)values[i] * factor) * (double)weight[i]);
            if (status) return status;
        }
    }
    return NEXA_Q4_OK;
}

int nexa_rope(const float *input, size_t input_count,
              size_t sequence, size_t heads, size_t head_dim, double theta,
              float *output, size_t output_count) {
    return nexa_rope_offset(input, input_count, sequence, heads, head_dim, theta,
                            0, output, output_count);
}

int nexa_rope_offset(const float *input, size_t input_count,
                     size_t sequence, size_t heads, size_t head_dim, double theta,
                     size_t position_offset, float *output, size_t output_count) {
    if (!input || !output || !sequence || !heads || !head_dim || head_dim % 2 ||
        !isfinite(theta) || theta <= 0.0) return NEXA_Q4_INVALID_ARGUMENT;
    if (position_offset > SIZE_MAX - (sequence - 1)) return NEXA_Q4_OVERFLOW;
    size_t width, elements, bytes;
    if (!checked_mul(heads, head_dim, &width) || !checked_mul(sequence, width, &elements) ||
        !float_bytes(elements, &bytes)) return NEXA_Q4_OVERFLOW;
    if (input_count < elements || output_count < elements) return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(input, bytes, output, bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    if (!finite_input(input, elements)) return NEXA_Q4_INVALID_DATA;
    size_t half = head_dim / 2;
    for (size_t lane = 0; lane < half; lane++) {
        double frequency = pow(theta, -2.0 * (double)lane / (double)head_dim);
        if (!isfinite(frequency)) return NEXA_Q4_NUMERIC_RANGE;
        for (size_t position = 0; position < sequence; position++) {
            double angle = (double)(position_offset + position) * frequency;
            if (!isfinite(angle)) return NEXA_Q4_NUMERIC_RANGE;
            double cosine = cos(angle), sine = sin(angle);
            for (size_t head = 0; head < heads; head++) {
                size_t offset = position * width + head * head_dim + lane;
                double first = input[offset], second = input[offset + half];
                int status = write_float(output + offset, first * cosine - second * sine);
                if (status) return status;
                status = write_float(output + offset + half, second * cosine + first * sine);
                if (status) return status;
            }
        }
    }
    return NEXA_Q4_OK;
}

static double attention_score(const float *query, const float *key,
                               size_t head_dim, double scale) {
    double dot = 0.0;
    for (size_t lane = 0; lane < head_dim; lane++) dot += (double)query[lane] * (double)key[lane];
    return dot * scale;
}

int nexa_causal_gqa_attention(
    const float *query, size_t query_count,
    const float *key, size_t key_count,
    const float *value, size_t value_count,
    size_t sequence, size_t query_heads, size_t kv_heads, size_t head_dim,
    float *scratch, size_t scratch_count,
    float *output, size_t output_count) {
    return nexa_causal_gqa_attention_cached(query, query_count, key, key_count,
                                            value, value_count, 0, sequence,
                                            query_heads, kv_heads, head_dim,
                                            scratch, scratch_count, output, output_count);
}

int nexa_causal_gqa_attention_cached(
    const float *query, size_t query_count,
    const float *key, size_t key_count,
    const float *value, size_t value_count,
    size_t past_length, size_t sequence, size_t query_heads, size_t kv_heads, size_t head_dim,
    float *scratch, size_t scratch_count,
    float *output, size_t output_count) {
    if (!query || !key || !value || !scratch || !output || !sequence || !query_heads ||
        !kv_heads || !head_dim || query_heads % kv_heads) return NEXA_Q4_INVALID_ARGUMENT;
    if (past_length > SIZE_MAX - sequence) return NEXA_Q4_OVERFLOW;
    size_t cache_length = past_length + sequence;
    size_t q_width, kv_width, q_elements, kv_elements, q_bytes, kv_bytes, scratch_bytes;
    if (!checked_mul(query_heads, head_dim, &q_width) ||
        !checked_mul(kv_heads, head_dim, &kv_width) ||
        !checked_mul(sequence, q_width, &q_elements) ||
        !checked_mul(cache_length, kv_width, &kv_elements) ||
        !float_bytes(q_elements, &q_bytes) || !float_bytes(kv_elements, &kv_bytes) ||
        !float_bytes(cache_length, &scratch_bytes)) return NEXA_Q4_OVERFLOW;
    if (query_count < q_elements || key_count < kv_elements || value_count < kv_elements ||
        output_count < q_elements || scratch_count < cache_length) return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(query, q_bytes, output, q_bytes) ||
        !disjoint(key, kv_bytes, output, q_bytes) || !disjoint(value, kv_bytes, output, q_bytes) ||
        !disjoint(query, q_bytes, scratch, scratch_bytes) ||
        !disjoint(key, kv_bytes, scratch, scratch_bytes) ||
        !disjoint(value, kv_bytes, scratch, scratch_bytes) ||
        !disjoint(output, q_bytes, scratch, scratch_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    if (!finite_input(query, q_elements) || !finite_input(key, kv_elements) ||
        !finite_input(value, kv_elements)) return NEXA_Q4_INVALID_DATA;
    double scale = 1.0 / sqrt((double)head_dim);
    size_t repeats = query_heads / kv_heads;
    for (size_t position = 0; position < sequence; position++) {
        size_t causal_end = past_length + position;
        for (size_t head = 0; head < query_heads; head++) {
            const float *q = query + position * q_width + head * head_dim;
            size_t kv_offset = (head / repeats) * head_dim;
            double maximum = -INFINITY;
            for (size_t past = 0; past <= causal_end; past++) {
                double score = attention_score(q, key + past * kv_width + kv_offset, head_dim, scale);
                if (!isfinite(score)) return NEXA_Q4_NUMERIC_RANGE;
                if (score > maximum) maximum = score;
            }
            double denominator = 0.0;
            for (size_t past = 0; past <= causal_end; past++) {
                double score = attention_score(q, key + past * kv_width + kv_offset, head_dim, scale);
                scratch[past] = (float)exp(score - maximum);
                denominator += (double)scratch[past];
            }
            for (size_t lane = 0; lane < head_dim; lane++) {
                double sum = 0.0;
                for (size_t past = 0; past <= causal_end; past++) {
                    sum += (double)scratch[past] * (double)value[past * kv_width + kv_offset + lane];
                }
                int status = write_float(output + position * q_width + head * head_dim + lane,
                                         sum / denominator);
                if (status) return status;
            }
        }
    }
    return NEXA_Q4_OK;
}

static const float *paged_row(const float *const *pages, size_t token,
                              size_t page_tokens, size_t width, size_t head_offset) {
    return pages[token / page_tokens] + (token % page_tokens) * width + head_offset;
}

int nexa_causal_gqa_attention_paged(
    const float *query, size_t query_count,
    const float *const *key_pages, size_t key_page_count,
    const float *const *value_pages, size_t value_page_count,
    size_t page_tokens, size_t page_floats,
    size_t past_length, size_t sequence, size_t query_heads, size_t kv_heads, size_t head_dim,
    float *scratch, size_t scratch_count,
    float *output, size_t output_count) {
    if (!query || !key_pages || !value_pages || !scratch || !output || !page_tokens ||
        !sequence || !query_heads || !kv_heads || !head_dim || query_heads % kv_heads)
        return NEXA_Q4_INVALID_ARGUMENT;
    if (past_length > SIZE_MAX - sequence) return NEXA_Q4_OVERFLOW;
    size_t cache_length = past_length + sequence;
    size_t needed_pages = cache_length / page_tokens + (cache_length % page_tokens != 0);
    size_t q_width, kv_width, q_elements, q_bytes, page_elements, page_bytes;
    size_t cache_elements, cache_bytes, table_bytes, scratch_bytes;
    if (!checked_mul(query_heads, head_dim, &q_width) ||
        !checked_mul(kv_heads, head_dim, &kv_width) ||
        !checked_mul(sequence, q_width, &q_elements) || !float_bytes(q_elements, &q_bytes) ||
        !checked_mul(page_tokens, kv_width, &page_elements) || !float_bytes(page_elements, &page_bytes) ||
        !checked_mul(cache_length, kv_width, &cache_elements) || !float_bytes(cache_elements, &cache_bytes) ||
        !checked_mul(needed_pages, sizeof(*key_pages), &table_bytes) ||
        !float_bytes(cache_length, &scratch_bytes)) return NEXA_Q4_OVERFLOW;
    if (query_count < q_elements || output_count < q_elements || scratch_count < cache_length ||
        key_page_count < needed_pages || value_page_count < needed_pages || page_floats < page_elements)
        return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(query, q_bytes, output, q_bytes) ||
        !disjoint(query, q_bytes, scratch, scratch_bytes) ||
        !disjoint(output, q_bytes, scratch, scratch_bytes) ||
        !disjoint(key_pages, table_bytes, output, q_bytes) ||
        !disjoint(value_pages, table_bytes, output, q_bytes) ||
        !disjoint(key_pages, table_bytes, scratch, scratch_bytes) ||
        !disjoint(value_pages, table_bytes, scratch, scratch_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    if (!finite_input(query, q_elements)) return NEXA_Q4_INVALID_DATA;
    size_t remaining = cache_length;
    for (size_t page = 0; page < needed_pages; page++) {
        const float *key = key_pages[page], *value = value_pages[page];
        if (!key || !value) return NEXA_Q4_INVALID_ARGUMENT;
        size_t visible_tokens = remaining < page_tokens ? remaining : page_tokens;
        /* Bounded by the checked page_elements/page_bytes products above. */
        size_t visible_elements = visible_tokens * kv_width;
        size_t visible_bytes = visible_elements * sizeof(float);
        if (!disjoint(key, visible_bytes, output, q_bytes) ||
            !disjoint(value, visible_bytes, output, q_bytes) ||
            !disjoint(key, visible_bytes, scratch, scratch_bytes) ||
            !disjoint(value, visible_bytes, scratch, scratch_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
        if (!finite_input(key, visible_elements) || !finite_input(value, visible_elements))
            return NEXA_Q4_INVALID_DATA;
        remaining -= visible_tokens;
    }
    double scale = 1.0 / sqrt((double)head_dim);
    size_t repeats = query_heads / kv_heads;
    for (size_t position = 0; position < sequence; position++) {
        size_t causal_end = past_length + position;
        for (size_t head = 0; head < query_heads; head++) {
            const float *q = query + position * q_width + head * head_dim;
            size_t kv_offset = (head / repeats) * head_dim;
            double maximum = -INFINITY;
            for (size_t past = 0; past <= causal_end; past++) {
                const float *key = paged_row(key_pages, past, page_tokens, kv_width, kv_offset);
                double score = attention_score(q, key, head_dim, scale);
                if (!isfinite(score)) return NEXA_Q4_NUMERIC_RANGE;
                if (score > maximum) maximum = score;
            }
            double denominator = 0.0;
            for (size_t past = 0; past <= causal_end; past++) {
                const float *key = paged_row(key_pages, past, page_tokens, kv_width, kv_offset);
                double score = attention_score(q, key, head_dim, scale);
                scratch[past] = (float)exp(score - maximum);
                denominator += (double)scratch[past];
            }
            for (size_t lane = 0; lane < head_dim; lane++) {
                double sum = 0.0;
                for (size_t past = 0; past <= causal_end; past++) {
                    const float *value = paged_row(value_pages, past, page_tokens, kv_width, kv_offset);
                    sum += (double)scratch[past] * (double)value[lane];
                }
                int status = write_float(output + position * q_width + head * head_dim + lane,
                                         sum / denominator);
                if (status) return status;
            }
        }
    }
    return NEXA_Q4_OK;
}

/* Byte loads preserve the canonical little-endian codec even for unaligned
 * page addresses. No temporary float32 vector is needed for packed KV. */
static float q4_scale(const uint8_t *record) {
    uint32_t bits = (uint32_t)record[0] | ((uint32_t)record[1] << 8) |
                    ((uint32_t)record[2] << 16) | ((uint32_t)record[3] << 24);
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static int q4_code(uint8_t byte, size_t lane) {
    unsigned int code = (byte >> (4 * (lane % 2))) & 15u;
    return code < 8 ? (int)code : (int)code - 16;
}

static int valid_q4_rows(const uint8_t *page, size_t rows, size_t cols,
                         size_t group_size, size_t group_bytes, size_t row_bytes) {
    for (size_t row = 0; row < rows; row++) {
        const uint8_t *record = page + row * row_bytes;
        size_t start = 0;
        for (size_t offset = 0; offset < row_bytes; offset += group_bytes) {
            const uint8_t *group = record + offset;
            float scale = q4_scale(group);
            if (!isfinite(scale) || scale < 0.0f) return 0;
            size_t valid = cols - start < group_size ? cols - start : group_size;
            for (size_t byte = 0; byte < group_bytes - 4; byte++) {
                int low = q4_code(group[4 + byte], 0), high = q4_code(group[4 + byte], 1);
                size_t index = byte * 2;
                if (low == -8 || high == -8 ||
                    ((index >= valid || scale == 0.0f) && low != 0) ||
                    ((index + 1 >= valid || scale == 0.0f) && high != 0)) return 0;
            }
            start += valid;
        }
    }
    return 1;
}

static const uint8_t *paged_q4_row(const uint8_t *const *pages, size_t token,
                                  size_t page_tokens, size_t token_bytes,
                                  size_t row_bytes, size_t head) {
    return pages[token / page_tokens] + (token % page_tokens) * token_bytes + head * row_bytes;
}

static double q4_attention_score(const float *query, const uint8_t *row,
                                 size_t cols, size_t group_size, size_t group_bytes,
                                 double scale) {
    double dot = 0.0;
    size_t start = 0, offset = 0;
    while (start < cols) {
        const uint8_t *group = row + offset;
        double magnitude = q4_scale(group);
        size_t valid = cols - start < group_size ? cols - start : group_size;
        for (size_t lane = 0; lane < valid; lane++) {
            double value = (double)q4_code(group[4 + lane / 2], lane) * magnitude;
            dot += (double)query[start + lane] * value;
        }
        start += valid;
        offset += group_bytes;
    }
    return dot * scale;
}

static double q4_lane(const uint8_t *row, size_t lane, size_t group_size, size_t group_bytes) {
    const uint8_t *group = row + (lane / group_size) * group_bytes;
    size_t within = lane % group_size;
    return (double)q4_code(group[4 + within / 2], within) * (double)q4_scale(group);
}

int nexa_causal_gqa_attention_paged_q4(
    const float *query, size_t query_count,
    const uint8_t *const *key_pages, size_t key_page_count,
    const uint8_t *const *value_pages, size_t value_page_count,
    size_t page_tokens, size_t page_bytes, size_t group_size,
    size_t past_length, size_t sequence, size_t query_heads, size_t kv_heads, size_t head_dim,
    float *scratch, size_t scratch_count,
    float *output, size_t output_count) {
    if (!query || !key_pages || !value_pages || !scratch || !output || !page_tokens || !group_size ||
        !sequence || !query_heads || !kv_heads || !head_dim || query_heads % kv_heads)
        return NEXA_Q4_INVALID_ARGUMENT;
    if (past_length > SIZE_MAX - sequence) return NEXA_Q4_OVERFLOW;
    size_t cache_length = past_length + sequence;
    size_t needed_pages = cache_length / page_tokens + (cache_length % page_tokens != 0);
    size_t row_bytes = nexa_q4_row_size(head_dim, group_size);
    if (!row_bytes) return NEXA_Q4_OVERFLOW;
    size_t group_bytes = 4 + group_size / 2 + group_size % 2;
    size_t q_width, q_elements, q_bytes, token_bytes, required_page_bytes, cache_bytes;
    size_t table_bytes, scratch_bytes;
    if (!checked_mul(query_heads, head_dim, &q_width) ||
        !checked_mul(sequence, q_width, &q_elements) || !float_bytes(q_elements, &q_bytes) ||
        !checked_mul(kv_heads, row_bytes, &token_bytes) ||
        !checked_mul(page_tokens, token_bytes, &required_page_bytes) ||
        !checked_mul(cache_length, token_bytes, &cache_bytes) ||
        !checked_mul(needed_pages, sizeof(*key_pages), &table_bytes) ||
        !float_bytes(cache_length, &scratch_bytes)) return NEXA_Q4_OVERFLOW;
    if (query_count < q_elements || output_count < q_elements || scratch_count < cache_length ||
        key_page_count < needed_pages || value_page_count < needed_pages || page_bytes < required_page_bytes)
        return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(query, q_bytes, output, q_bytes) ||
        !disjoint(query, q_bytes, scratch, scratch_bytes) ||
        !disjoint(output, q_bytes, scratch, scratch_bytes) ||
        !disjoint(key_pages, table_bytes, output, q_bytes) ||
        !disjoint(value_pages, table_bytes, output, q_bytes) ||
        !disjoint(key_pages, table_bytes, scratch, scratch_bytes) ||
        !disjoint(value_pages, table_bytes, scratch, scratch_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    if (!finite_input(query, q_elements)) return NEXA_Q4_INVALID_DATA;
    size_t remaining = cache_length;
    for (size_t page = 0; page < needed_pages; page++) {
        const uint8_t *key = key_pages[page], *value = value_pages[page];
        if (!key || !value) return NEXA_Q4_INVALID_ARGUMENT;
        size_t visible_tokens = remaining < page_tokens ? remaining : page_tokens;
        size_t visible_bytes = visible_tokens * token_bytes;
        if (!disjoint(key, visible_bytes, output, q_bytes) ||
            !disjoint(value, visible_bytes, output, q_bytes) ||
            !disjoint(key, visible_bytes, scratch, scratch_bytes) ||
            !disjoint(value, visible_bytes, scratch, scratch_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
        if (!valid_q4_rows(key, visible_tokens * kv_heads, head_dim, group_size, group_bytes, row_bytes) ||
            !valid_q4_rows(value, visible_tokens * kv_heads, head_dim, group_size, group_bytes, row_bytes))
            return NEXA_Q4_INVALID_DATA;
        remaining -= visible_tokens;
    }
    double scale = 1.0 / sqrt((double)head_dim);
    size_t repeats = query_heads / kv_heads;
    for (size_t position = 0; position < sequence; position++) {
        size_t causal_end = past_length + position;
        for (size_t head = 0; head < query_heads; head++) {
            const float *q = query + position * q_width + head * head_dim;
            size_t kv_head = head / repeats;
            double maximum = -INFINITY;
            for (size_t past = 0; past <= causal_end; past++) {
                const uint8_t *key = paged_q4_row(key_pages, past, page_tokens, token_bytes, row_bytes, kv_head);
                double score = q4_attention_score(q, key, head_dim, group_size, group_bytes, scale);
                if (!isfinite(score)) return NEXA_Q4_NUMERIC_RANGE;
                if (score > maximum) maximum = score;
            }
            double denominator = 0.0;
            for (size_t past = 0; past <= causal_end; past++) {
                const uint8_t *key = paged_q4_row(key_pages, past, page_tokens, token_bytes, row_bytes, kv_head);
                double score = q4_attention_score(q, key, head_dim, group_size, group_bytes, scale);
                scratch[past] = (float)exp(score - maximum);
                denominator += (double)scratch[past];
            }
            for (size_t lane = 0; lane < head_dim; lane++) {
                double sum = 0.0;
                for (size_t past = 0; past <= causal_end; past++) {
                    const uint8_t *value = paged_q4_row(value_pages, past, page_tokens, token_bytes, row_bytes, kv_head);
                    sum += (double)scratch[past] * q4_lane(value, lane, group_size, group_bytes);
                }
                int status = write_float(output + position * q_width + head * head_dim + lane, sum / denominator);
                if (status) return status;
            }
        }
    }
    return NEXA_Q4_OK;
}

static int q3_layout(size_t cols, size_t group_size, size_t *groups,
                     size_t *group_bytes, size_t *row_bytes) {
    if (!cols || !group_size) return NEXA_Q4_INVALID_ARGUMENT;
    size_t bits;
    if (!checked_mul(group_size, 3, &bits)) return NEXA_Q4_OVERFLOW;
    size_t code_bytes = bits / 8 + (bits % 8 != 0);
    if (code_bytes > SIZE_MAX - 4) return NEXA_Q4_OVERFLOW;
    *group_bytes = 4 + code_bytes;
    *groups = cols / group_size + (cols % group_size != 0);
    if (!checked_mul(*groups, *group_bytes, row_bytes)) return NEXA_Q4_OVERFLOW;
    return NEXA_Q4_OK;
}

size_t nexa_q3_row_size(size_t cols, size_t group_size) {
    size_t groups, group_bytes, row_bytes;
    return q3_layout(cols, group_size, &groups, &group_bytes, &row_bytes) == 0
        ? row_bytes : 0;
}

static void q3_store_scale(uint8_t *record, float scale) {
    uint32_t bits;
    memcpy(&bits, &scale, sizeof(bits));
    record[0] = (uint8_t)bits;
    record[1] = (uint8_t)(bits >> 8);
    record[2] = (uint8_t)(bits >> 16);
    record[3] = (uint8_t)(bits >> 24);
}

/* q3_layout checks 3*group_size before a lane bit offset can be formed. The
 * second byte exists exactly when a code crosses a byte boundary. */
static int q3_code(const uint8_t *codes, size_t lane) {
    size_t bit = lane * 3, byte = bit / 8;
    unsigned int shift = (unsigned int)(bit % 8);
    unsigned int code = (unsigned int)codes[byte] >> shift;
    if (shift > 5) code |= (unsigned int)codes[byte + 1] << (8 - shift);
    code &= 7u;
    return code < 4 ? (int)code : (int)code - 8;
}

int nexa_q3_quantize(const float *weights, size_t weight_count,
                    size_t rows, size_t cols, size_t group_size,
                    uint8_t *packed, size_t packed_bytes) {
    if (!weights || !packed || !rows) return NEXA_Q4_INVALID_ARGUMENT;
    size_t groups, group_bytes, row_bytes, total_bytes, elements, input_bytes;
    int status = q3_layout(cols, group_size, &groups, &group_bytes, &row_bytes);
    if (status) return status;
    if (!checked_mul(rows, row_bytes, &total_bytes) ||
        !checked_mul(rows, cols, &elements) || !float_bytes(elements, &input_bytes))
        return NEXA_Q4_OVERFLOW;
    if (weight_count < elements || packed_bytes < total_bytes)
        return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(weights, input_bytes, packed, total_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    if (!finite_input(weights, elements)) return NEXA_Q4_INVALID_DATA;
    for (size_t row = 0; row < rows; row++) {
        size_t start = 0;
        for (size_t group = 0; group < groups; group++) {
            size_t valid = cols - start < group_size ? cols - start : group_size;
            const float *values = weights + row * cols + start;
            uint8_t *record = packed + row * row_bytes + group * group_bytes;
            float maximum = 0.0f;
            for (size_t lane = 0; lane < valid; lane++) {
                float magnitude = fabsf(values[lane]);
                if (magnitude > maximum) maximum = magnitude;
            }
            float scale = (float)((double)maximum / 3.0);
            if (maximum > 0.0f && scale == 0.0f) return NEXA_Q4_NUMERIC_RANGE;
            q3_store_scale(record, scale);
            memset(record + 4, 0, group_bytes - 4);
            if (scale > 0.0f) {
                for (size_t lane = 0; lane < valid; lane++) {
                    double value = (double)values[lane] / (double)scale;
                    int quantized;
                    if (value >= 3.0) quantized = 3;
                    else if (value <= -3.0) quantized = -3;
                    else quantized = (int)(value >= 0.0 ? value + 0.5 : value - 0.5);
                    unsigned int code = (unsigned int)(quantized < 0 ? quantized + 8 : quantized);
                    size_t bit = lane * 3, byte = bit / 8;
                    unsigned int shift = (unsigned int)(bit % 8);
                    record[4 + byte] |= (uint8_t)(code << shift);
                    if (shift > 5) record[5 + byte] |= (uint8_t)(code >> (8 - shift));
                }
            }
            start += valid;
        }
    }
    return NEXA_Q4_OK;
}

int nexa_q8_quantize(const float *weights, size_t weight_count,
                    size_t rows, size_t cols, size_t group_size,
                    uint8_t *packed, size_t packed_bytes) {
    if (!weights || !packed || !rows || !cols || !group_size) return NEXA_Q4_INVALID_ARGUMENT;
    size_t groups, group_bytes, row_bytes, total_bytes, elements, input_bytes;
    if (group_size > SIZE_MAX - 4) return NEXA_Q4_OVERFLOW;
    group_bytes = 4 + group_size;
    groups = cols / group_size + (cols % group_size != 0);
    if (!checked_mul(groups, group_bytes, &row_bytes) ||
        !checked_mul(rows, row_bytes, &total_bytes) ||
        !checked_mul(rows, cols, &elements) || !float_bytes(elements, &input_bytes))
        return NEXA_Q4_OVERFLOW;
    if (weight_count < elements || packed_bytes < total_bytes)
        return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(weights, input_bytes, packed, total_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    if (!finite_input(weights, elements)) return NEXA_Q4_INVALID_DATA;
    for (size_t row = 0; row < rows; row++) {
        size_t start = 0;
        for (size_t group = 0; group < groups; group++) {
            size_t valid = cols - start < group_size ? cols - start : group_size;
            const float *values = weights + row * cols + start;
            uint8_t *record = packed + row * row_bytes + group * group_bytes;
            float maximum = 0.0f;
            for (size_t lane = 0; lane < valid; lane++) {
                float magnitude = fabsf(values[lane]);
                if (magnitude > maximum) maximum = magnitude;
            }
            float scale = (float)((double)maximum / 127.0);
            if (maximum > 0.0f && scale == 0.0f) return NEXA_Q4_NUMERIC_RANGE;
            q3_store_scale(record, scale);
            memset(record + 4, 0, group_bytes - 4);
            if (scale > 0.0f) {
                for (size_t lane = 0; lane < valid; lane++) {
                    double value = (double)values[lane] / (double)scale;
                    int quantized;
                    if (value >= 127.0) quantized = 127;
                    else if (value <= -127.0) quantized = -127;
                    else quantized = (int)(value >= 0.0 ? value + 0.5 : value - 0.5);
                    record[4 + lane] = (uint8_t)(quantized & 0xFF);
                }
            }
            start += valid;
        }
    }
    return NEXA_Q4_OK;
}

static double q8_lane(const uint8_t *row, size_t lane, size_t group_size, size_t group_bytes) {
    const uint8_t *group = row + (lane / group_size) * group_bytes;
    float scale;
    memcpy(&scale, group, sizeof(scale));
    return (double)scale * (double)(int8_t)group[4 + lane % group_size];
}

static int valid_q8_rows(const uint8_t *page, size_t rows, size_t cols,
                         size_t group_size, size_t group_bytes, size_t row_bytes) {
    size_t groups = cols / group_size + (cols % group_size != 0);
    for (size_t row = 0; row < rows; row++) {
        const uint8_t *record = page + row * row_bytes;
        size_t start = 0;
        for (size_t group = 0; group < groups; group++) {
            const uint8_t *data = record + group * group_bytes;
            float scale;
            memcpy(&scale, data, sizeof(scale));
            if (!isfinite(scale) || scale < 0.0f) return 0;
            size_t valid = cols - start < group_size ? cols - start : group_size;
            for (size_t lane = 0; lane < group_size; lane++) {
                int value = (int8_t)data[4 + lane];
                if (value == -128) return 0;
                if ((lane >= valid || scale == 0.0f) && value != 0) return 0;
            }
            start += valid;
        }
    }
    return 1;
}

static int valid_q3_rows(const uint8_t *page, size_t rows, size_t cols,
                         size_t group_size, size_t group_bytes, size_t row_bytes) {
    unsigned int tail_bits = (unsigned int)((group_size * 3) % 8);
    for (size_t row = 0; row < rows; row++) {
        const uint8_t *record = page + row * row_bytes;
        size_t start = 0;
        for (size_t offset = 0; offset < row_bytes; offset += group_bytes) {
            const uint8_t *group = record + offset;
            float scale = q4_scale(group); /* Shared unaligned little-endian F32 load. */
            if (!isfinite(scale) || scale < 0.0f) return 0;
            size_t valid = cols - start < group_size ? cols - start : group_size;
            for (size_t lane = 0; lane < group_size; lane++) {
                int code = q3_code(group + 4, lane);
                if (code == -4 || ((lane >= valid || scale == 0.0f) && code != 0)) return 0;
            }
            if (tail_bits && (group[group_bytes - 1] >> tail_bits) != 0) return 0;
            start += valid;
        }
    }
    return 1;
}

static double q3_attention_score(const float *query, const uint8_t *row,
                                 size_t cols, size_t group_size, size_t group_bytes,
                                 double scale) {
    double dot = 0.0;
    size_t start = 0, offset = 0;
    while (start < cols) {
        const uint8_t *group = row + offset;
        double magnitude = q4_scale(group);
        size_t valid = cols - start < group_size ? cols - start : group_size;
        for (size_t lane = 0; lane < valid; lane++) {
            double value = (double)q3_code(group + 4, lane) * magnitude;
            dot += (double)query[start + lane] * value;
        }
        start += valid;
        offset += group_bytes;
    }
    return dot * scale;
}

static double q3_lane(const uint8_t *row, size_t lane, size_t group_size, size_t group_bytes) {
    const uint8_t *group = row + (lane / group_size) * group_bytes;
    return (double)q3_code(group + 4, lane % group_size) * (double)q4_scale(group);
}

int nexa_causal_gqa_attention_paged_q3(
    const float *query, size_t query_count,
    const uint8_t *const *key_pages, size_t key_page_count,
    const uint8_t *const *value_pages, size_t value_page_count,
    size_t page_tokens, size_t page_bytes, size_t group_size,
    size_t past_length, size_t sequence, size_t query_heads, size_t kv_heads, size_t head_dim,
    float *scratch, size_t scratch_count,
    float *output, size_t output_count) {
    if (!query || !key_pages || !value_pages || !scratch || !output || !page_tokens || !group_size ||
        !sequence || !query_heads || !kv_heads || !head_dim || query_heads % kv_heads)
        return NEXA_Q4_INVALID_ARGUMENT;
    if (past_length > SIZE_MAX - sequence) return NEXA_Q4_OVERFLOW;
    size_t cache_length = past_length + sequence;
    size_t needed_pages = cache_length / page_tokens + (cache_length % page_tokens != 0);
    size_t groups, group_bytes, row_bytes;
    int status = q3_layout(head_dim, group_size, &groups, &group_bytes, &row_bytes);
    if (status) return status;
    size_t q_width, q_elements, q_bytes, token_bytes, required_page_bytes, cache_bytes;
    size_t table_bytes, scratch_bytes;
    if (!checked_mul(query_heads, head_dim, &q_width) ||
        !checked_mul(sequence, q_width, &q_elements) || !float_bytes(q_elements, &q_bytes) ||
        !checked_mul(kv_heads, row_bytes, &token_bytes) ||
        !checked_mul(page_tokens, token_bytes, &required_page_bytes) ||
        !checked_mul(cache_length, token_bytes, &cache_bytes) ||
        !checked_mul(needed_pages, sizeof(*key_pages), &table_bytes) ||
        !float_bytes(cache_length, &scratch_bytes)) return NEXA_Q4_OVERFLOW;
    if (query_count < q_elements || output_count < q_elements || scratch_count < cache_length ||
        key_page_count < needed_pages || value_page_count < needed_pages || page_bytes < required_page_bytes)
        return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(query, q_bytes, output, q_bytes) ||
        !disjoint(query, q_bytes, scratch, scratch_bytes) ||
        !disjoint(output, q_bytes, scratch, scratch_bytes) ||
        !disjoint(key_pages, table_bytes, output, q_bytes) ||
        !disjoint(value_pages, table_bytes, output, q_bytes) ||
        !disjoint(key_pages, table_bytes, scratch, scratch_bytes) ||
        !disjoint(value_pages, table_bytes, scratch, scratch_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    if (!finite_input(query, q_elements)) return NEXA_Q4_INVALID_DATA;
    size_t remaining = cache_length;
    for (size_t page = 0; page < needed_pages; page++) {
        const uint8_t *key = key_pages[page], *value = value_pages[page];
        if (!key || !value) return NEXA_Q4_INVALID_ARGUMENT;
        size_t visible_tokens = remaining < page_tokens ? remaining : page_tokens;
        size_t visible_bytes = visible_tokens * token_bytes;
        if (!disjoint(key, visible_bytes, output, q_bytes) ||
            !disjoint(value, visible_bytes, output, q_bytes) ||
            !disjoint(key, visible_bytes, scratch, scratch_bytes) ||
            !disjoint(value, visible_bytes, scratch, scratch_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
        if (!valid_q3_rows(key, visible_tokens * kv_heads, head_dim, group_size, group_bytes, row_bytes) ||
            !valid_q3_rows(value, visible_tokens * kv_heads, head_dim, group_size, group_bytes, row_bytes))
            return NEXA_Q4_INVALID_DATA;
        remaining -= visible_tokens;
    }
    double scale = 1.0 / sqrt((double)head_dim);
    size_t repeats = query_heads / kv_heads;
    for (size_t position = 0; position < sequence; position++) {
        size_t causal_end = past_length + position;
        for (size_t head = 0; head < query_heads; head++) {
            const float *q = query + position * q_width + head * head_dim;
            size_t kv_head = head / repeats;
            double maximum = -INFINITY;
            for (size_t past = 0; past <= causal_end; past++) {
                const uint8_t *key = paged_q4_row(key_pages, past, page_tokens, token_bytes, row_bytes, kv_head);
                double score = q3_attention_score(q, key, head_dim, group_size, group_bytes, scale);
                if (!isfinite(score)) return NEXA_Q4_NUMERIC_RANGE;
                if (score > maximum) maximum = score;
            }
            double denominator = 0.0;
            for (size_t past = 0; past <= causal_end; past++) {
                const uint8_t *key = paged_q4_row(key_pages, past, page_tokens, token_bytes, row_bytes, kv_head);
                double score = q3_attention_score(q, key, head_dim, group_size, group_bytes, scale);
                scratch[past] = (float)exp(score - maximum);
                denominator += (double)scratch[past];
            }
            for (size_t lane = 0; lane < head_dim; lane++) {
                double sum = 0.0;
                for (size_t past = 0; past <= causal_end; past++) {
                    const uint8_t *value = paged_q4_row(value_pages, past, page_tokens, token_bytes, row_bytes, kv_head);
                    sum += (double)scratch[past] * q3_lane(value, lane, group_size, group_bytes);
                }
                status = write_float(output + position * q_width + head * head_dim + lane, sum / denominator);
                if (status) return status;
            }
        }
    }
    return NEXA_Q4_OK;
}

/* Common scalar access for mixed pages and one-head reencoding. F32 payloads
 * deliberately use memcpy, allowing byte-addressed page storage without UB. */
static int kv_row_layout(int codec, size_t dim, size_t group_size,
                         size_t *row_bytes, size_t *group_bytes) {
    if (!dim || !group_size) return NEXA_Q4_INVALID_ARGUMENT;
    if (codec == 0) {
        *group_bytes = 0;
        return float_bytes(dim, row_bytes) ? NEXA_Q4_OK : NEXA_Q4_OVERFLOW;
    }
    if (codec == 4) {
        *row_bytes = nexa_q4_row_size(dim, group_size);
        if (!*row_bytes) return NEXA_Q4_OVERFLOW;
        *group_bytes = 4 + group_size / 2 + group_size % 2;
        return NEXA_Q4_OK;
    }
    if (codec == 3) {
        size_t groups;
        return q3_layout(dim, group_size, &groups, group_bytes, row_bytes);
    }
    if (codec == 8) {
        if (group_size > SIZE_MAX - 4) return NEXA_Q4_OVERFLOW;
        *group_bytes = 4 + group_size;
        size_t groups = dim / group_size + (dim % group_size != 0);
        return checked_mul(groups, *group_bytes, row_bytes) ? NEXA_Q4_OK : NEXA_Q4_OVERFLOW;
    }
    return NEXA_Q4_INVALID_ARGUMENT;
}

static double kv_lane(const uint8_t *row, int codec, size_t lane,
                       size_t group_size, size_t group_bytes) {
    if (codec == 4) return q4_lane(row, lane, group_size, group_bytes);
    if (codec == 3) return q3_lane(row, lane, group_size, group_bytes);
    if (codec == 8) return q8_lane(row, lane, group_size, group_bytes);
    float value;
    memcpy(&value, row + lane * sizeof(float), sizeof(value));
    return (double)value;
}

static int valid_kv_rows(const uint8_t *data, int codec, size_t rows, size_t dim,
                         size_t group_size, size_t group_bytes, size_t row_bytes) {
    if (codec == 4) return valid_q4_rows(data, rows, dim, group_size, group_bytes, row_bytes);
    if (codec == 3) return valid_q3_rows(data, rows, dim, group_size, group_bytes, row_bytes);
    if (codec == 8) return valid_q8_rows(data, rows, dim, group_size, group_bytes, row_bytes);
    for (size_t row = 0; row < rows; row++) {
        for (size_t lane = 0; lane < dim; lane++) {
            if (!isfinite(kv_lane(data + row * row_bytes, codec, lane, group_size, group_bytes)))
                return 0;
        }
    }
    return 1;
}

static double mixed_score(const float *query, const uint8_t *const *pages,
                           const uint8_t *codecs, const size_t *row_bytes,
                           const size_t *group_bytes, size_t token, size_t page_tokens,
                           size_t kv_heads, size_t head, size_t dim, size_t group_size,
                           double scale) {
    size_t page = token / page_tokens;
    int codec = codecs[page];
    const uint8_t *row = pages[page] + ((token % page_tokens) * kv_heads + head) * row_bytes[codec];
    double dot = 0.0;
    for (size_t lane = 0; lane < dim; lane++)
        dot += (double)query[lane] * kv_lane(row, codec, lane, group_size, group_bytes[codec]);
    return dot * scale;
}

/* One homogeneous codec across every page, read through the shared accessors.
 * The reduction order matches the per-codec kernels: maxima, then exponential
 * weights into the caller's scratch, then one value lane at a time. */
int nexa_causal_gqa_attention_paged_codec(
    const float *query, size_t query_count,
    const uint8_t *const *key_pages, size_t key_page_count,
    const uint8_t *const *value_pages, size_t value_page_count,
    int codec, size_t page_tokens, size_t page_bytes, size_t group_size,
    size_t past_length, size_t sequence, size_t query_heads, size_t kv_heads, size_t head_dim,
    float *scratch, size_t scratch_count,
    float *output, size_t output_count) {
    if (!query || !key_pages || !value_pages || !scratch || !output || !page_tokens || !group_size ||
        !sequence || !query_heads || !kv_heads || !head_dim || query_heads % kv_heads)
        return NEXA_Q4_INVALID_ARGUMENT;
    if (past_length > SIZE_MAX - sequence) return NEXA_Q4_OVERFLOW;
    size_t cache_length = past_length + sequence;
    size_t needed_pages = cache_length / page_tokens + (cache_length % page_tokens != 0);
    size_t row_bytes, group_bytes;
    int status = kv_row_layout(codec, head_dim, group_size, &row_bytes, &group_bytes);
    if (status) return status;
    size_t q_width, q_elements, q_bytes, token_bytes, required_page_bytes;
    size_t table_bytes, scratch_bytes;
    if (!checked_mul(query_heads, head_dim, &q_width) ||
        !checked_mul(sequence, q_width, &q_elements) || !float_bytes(q_elements, &q_bytes) ||
        !checked_mul(kv_heads, row_bytes, &token_bytes) ||
        !checked_mul(page_tokens, token_bytes, &required_page_bytes) ||
        !checked_mul(needed_pages, sizeof(*key_pages), &table_bytes) ||
        !float_bytes(cache_length, &scratch_bytes)) return NEXA_Q4_OVERFLOW;
    if (query_count < q_elements || output_count < q_elements || scratch_count < cache_length ||
        key_page_count < needed_pages || value_page_count < needed_pages ||
        page_bytes < required_page_bytes) return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(query, q_bytes, output, q_bytes) ||
        !disjoint(query, q_bytes, scratch, scratch_bytes) ||
        !disjoint(output, q_bytes, scratch, scratch_bytes) ||
        !disjoint(key_pages, table_bytes, output, q_bytes) ||
        !disjoint(value_pages, table_bytes, output, q_bytes) ||
        !disjoint(key_pages, table_bytes, scratch, scratch_bytes) ||
        !disjoint(value_pages, table_bytes, scratch, scratch_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    if (!finite_input(query, q_elements)) return NEXA_Q4_INVALID_DATA;
    size_t remaining = cache_length;
    for (size_t page = 0; page < needed_pages; page++) {
        const uint8_t *key = key_pages[page], *value = value_pages[page];
        if (!key || !value) return NEXA_Q4_INVALID_ARGUMENT;
        size_t visible_tokens = remaining < page_tokens ? remaining : page_tokens;
        size_t visible_bytes = visible_tokens * token_bytes;
        if (!disjoint(key, visible_bytes, output, q_bytes) ||
            !disjoint(value, visible_bytes, output, q_bytes) ||
            !disjoint(key, visible_bytes, scratch, scratch_bytes) ||
            !disjoint(value, visible_bytes, scratch, scratch_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
        if (!valid_kv_rows(key, codec, visible_tokens * kv_heads, head_dim, group_size,
                           group_bytes, row_bytes) ||
            !valid_kv_rows(value, codec, visible_tokens * kv_heads, head_dim, group_size,
                           group_bytes, row_bytes)) return NEXA_Q4_INVALID_DATA;
        remaining -= visible_tokens;
    }
    double scale = 1.0 / sqrt((double)head_dim);
    size_t repeats = query_heads / kv_heads;
    for (size_t position = 0; position < sequence; position++) {
        size_t causal_end = past_length + position;
        for (size_t head = 0; head < query_heads; head++) {
            const float *q = query + position * q_width + head * head_dim;
            size_t kv_head = head / repeats;
            double maximum = -INFINITY;
            for (size_t past = 0; past <= causal_end; past++) {
                const uint8_t *key = paged_q4_row(key_pages, past, page_tokens, token_bytes,
                                                  row_bytes, kv_head);
                double dot = 0.0;
                for (size_t lane = 0; lane < head_dim; lane++)
                    dot += (double)q[lane] * kv_lane(key, codec, lane, group_size, group_bytes);
                double score = dot * scale;
                if (!isfinite(score)) return NEXA_Q4_NUMERIC_RANGE;
                if (score > maximum) maximum = score;
            }
            double denominator = 0.0;
            for (size_t past = 0; past <= causal_end; past++) {
                const uint8_t *key = paged_q4_row(key_pages, past, page_tokens, token_bytes,
                                                  row_bytes, kv_head);
                double dot = 0.0;
                for (size_t lane = 0; lane < head_dim; lane++)
                    dot += (double)q[lane] * kv_lane(key, codec, lane, group_size, group_bytes);
                scratch[past] = (float)exp(dot * scale - maximum);
                denominator += (double)scratch[past];
            }
            for (size_t lane = 0; lane < head_dim; lane++) {
                double sum = 0.0;
                for (size_t past = 0; past <= causal_end; past++) {
                    const uint8_t *value = paged_q4_row(value_pages, past, page_tokens, token_bytes,
                                                        row_bytes, kv_head);
                    sum += (double)scratch[past] * kv_lane(value, codec, lane, group_size, group_bytes);
                }
                status = write_float(output + position * q_width + head * head_dim + lane,
                                     sum / denominator);
                if (status) return status;
            }
        }
    }
    return NEXA_Q4_OK;
}

int nexa_causal_gqa_attention_paged_mixed(
    const float *query, size_t query_count,
    const uint8_t *const *key_pages, size_t key_page_count,
    const uint8_t *const *value_pages, size_t value_page_count,
    const uint8_t *page_codecs, size_t codec_count,
    const size_t *page_bytes, size_t byte_count,
    size_t page_tokens, size_t group_size,
    size_t past_length, size_t sequence, size_t query_heads, size_t kv_heads, size_t head_dim,
    float *scratch, size_t scratch_count,
    float *output, size_t output_count) {
    if (!query || !key_pages || !value_pages || !page_codecs || !page_bytes || !scratch || !output ||
        !page_tokens || !group_size || !sequence || !query_heads || !kv_heads || !head_dim ||
        query_heads % kv_heads || (uintptr_t)query % _Alignof(float) ||
        (uintptr_t)scratch % _Alignof(float) || (uintptr_t)output % _Alignof(float) ||
        (uintptr_t)key_pages % _Alignof(const uint8_t *) ||
        (uintptr_t)value_pages % _Alignof(const uint8_t *) ||
        (uintptr_t)page_bytes % _Alignof(size_t)) return NEXA_Q4_INVALID_ARGUMENT;
    if (past_length > SIZE_MAX - sequence) return NEXA_Q4_OVERFLOW;
    size_t cache_length = past_length + sequence;
    size_t needed_pages = cache_length / page_tokens + (cache_length % page_tokens != 0);
    size_t q_width, q_elements, q_bytes, pointer_bytes, capacity_bytes, scratch_bytes;
    if (!checked_mul(query_heads, head_dim, &q_width) ||
        !checked_mul(sequence, q_width, &q_elements) || !float_bytes(q_elements, &q_bytes) ||
        !checked_mul(needed_pages, sizeof(*key_pages), &pointer_bytes) ||
        !checked_mul(needed_pages, sizeof(*page_bytes), &capacity_bytes) ||
        !float_bytes(cache_length, &scratch_bytes)) return NEXA_Q4_OVERFLOW;
    if (query_count < q_elements || output_count < q_elements || scratch_count < cache_length ||
        key_page_count < needed_pages || value_page_count < needed_pages ||
        codec_count < needed_pages || byte_count < needed_pages) return NEXA_Q4_BUFFER_TOO_SMALL;
    const void *tables[] = {key_pages, value_pages, page_codecs, page_bytes};
    size_t table_sizes[] = {pointer_bytes, pointer_bytes, needed_pages, capacity_bytes};
    if (!disjoint(query, q_bytes, output, q_bytes) ||
        !disjoint(query, q_bytes, scratch, scratch_bytes) ||
        !disjoint(output, q_bytes, scratch, scratch_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    for (size_t i = 0; i < 4; i++) {
        if (!disjoint(tables[i], table_sizes[i], output, q_bytes) ||
            !disjoint(tables[i], table_sizes[i], scratch, scratch_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    }
    if (!finite_input(query, q_elements)) return NEXA_Q4_INVALID_DATA;
    size_t row_sizes[5] = {0}, group_sizes[5] = {0};
    size_t remaining = cache_length;
    for (size_t page = 0; page < needed_pages; page++) {
        int codec = page_codecs[page];
        if (codec != 0 && codec != 3 && codec != 4) return NEXA_Q4_INVALID_ARGUMENT;
        if (!row_sizes[codec]) {
            int status = kv_row_layout(codec, head_dim, group_size, row_sizes + codec, group_sizes + codec);
            if (status) return status;
        }
        size_t token_bytes, required_bytes;
        if (!checked_mul(kv_heads, row_sizes[codec], &token_bytes) ||
            !checked_mul(page_tokens, token_bytes, &required_bytes)) return NEXA_Q4_OVERFLOW;
        if (page_bytes[page] < required_bytes) return NEXA_Q4_BUFFER_TOO_SMALL;
        const uint8_t *key = key_pages[page], *value = value_pages[page];
        if (!key || !value) return NEXA_Q4_INVALID_ARGUMENT;
        size_t visible_tokens = remaining < page_tokens ? remaining : page_tokens;
        size_t visible_bytes = visible_tokens * token_bytes;
        if (!disjoint(key, visible_bytes, output, q_bytes) ||
            !disjoint(value, visible_bytes, output, q_bytes) ||
            !disjoint(key, visible_bytes, scratch, scratch_bytes) ||
            !disjoint(value, visible_bytes, scratch, scratch_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
        if (!valid_kv_rows(key, codec, visible_tokens * kv_heads, head_dim, group_size,
                           group_sizes[codec], row_sizes[codec]) ||
            !valid_kv_rows(value, codec, visible_tokens * kv_heads, head_dim, group_size,
                           group_sizes[codec], row_sizes[codec])) return NEXA_Q4_INVALID_DATA;
        remaining -= visible_tokens;
    }
    double scale = 1.0 / sqrt((double)head_dim);
    size_t repeats = query_heads / kv_heads;
    for (size_t position = 0; position < sequence; position++) {
        size_t causal_end = past_length + position;
        for (size_t head = 0; head < query_heads; head++) {
            const float *q = query + position * q_width + head * head_dim;
            size_t kv_head = head / repeats;
            double maximum = -INFINITY;
            for (size_t past = 0; past <= causal_end; past++) {
                double score = mixed_score(q, key_pages, page_codecs, row_sizes, group_sizes,
                                          past, page_tokens, kv_heads, kv_head, head_dim, group_size, scale);
                if (!isfinite(score)) return NEXA_Q4_NUMERIC_RANGE;
                if (score > maximum) maximum = score;
            }
            double denominator = 0.0;
            for (size_t past = 0; past <= causal_end; past++) {
                double score = mixed_score(q, key_pages, page_codecs, row_sizes, group_sizes,
                                          past, page_tokens, kv_heads, kv_head, head_dim, group_size, scale);
                scratch[past] = (float)exp(score - maximum);
                denominator += (double)scratch[past];
            }
            for (size_t lane = 0; lane < head_dim; lane++) {
                double sum = 0.0;
                for (size_t past = 0; past <= causal_end; past++) {
                    size_t page = past / page_tokens;
                    int codec = page_codecs[page];
                    const uint8_t *row = value_pages[page] +
                        ((past % page_tokens) * kv_heads + kv_head) * row_sizes[codec];
                    sum += (double)scratch[past] * kv_lane(row, codec, lane, group_size, group_sizes[codec]);
                }
                int status = write_float(output + position * q_width + head * head_dim + lane,
                                         sum / denominator);
                if (status) return status;
            }
        }
    }
    return NEXA_Q4_OK;
}

int nexa_kv_reencode_rows(
    const uint8_t *input, size_t input_bytes, int source_codec, int target_codec,
    size_t rows, size_t head_dim, size_t group_size,
    uint8_t *output, size_t output_bytes,
    float *scratch, size_t scratch_count, double *stats, size_t stats_count) {
    if (!input || !output || !scratch || !stats || !rows ||
        (uintptr_t)scratch % _Alignof(float) || (uintptr_t)stats % _Alignof(double))
        return NEXA_Q4_INVALID_ARGUMENT;
    size_t source_row_bytes, source_group_bytes, target_row_bytes, target_group_bytes;
    int status = kv_row_layout(source_codec, head_dim, group_size, &source_row_bytes, &source_group_bytes);
    if (status) return status;
    status = kv_row_layout(target_codec, head_dim, group_size, &target_row_bytes, &target_group_bytes);
    if (status) return status;
    size_t source_bytes, target_bytes, scratch_bytes, elements;
    if (!checked_mul(rows, source_row_bytes, &source_bytes) ||
        !checked_mul(rows, target_row_bytes, &target_bytes) ||
        !float_bytes(head_dim, &scratch_bytes) || !checked_mul(rows, head_dim, &elements))
        return NEXA_Q4_OVERFLOW;
    if (input_bytes < source_bytes || output_bytes < target_bytes ||
        scratch_count < head_dim || stats_count < 3) return NEXA_Q4_BUFFER_TOO_SMALL;
    size_t stats_bytes = 3 * sizeof(double);
    if (!disjoint(input, source_bytes, output, target_bytes) ||
        !disjoint(input, source_bytes, scratch, scratch_bytes) ||
        !disjoint(input, source_bytes, stats, stats_bytes) ||
        !disjoint(output, target_bytes, scratch, scratch_bytes) ||
        !disjoint(output, target_bytes, stats, stats_bytes) ||
        !disjoint(scratch, scratch_bytes, stats, stats_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    if (!valid_kv_rows(input, source_codec, rows, head_dim, group_size, source_group_bytes, source_row_bytes))
        return NEXA_Q4_INVALID_DATA;
    double maximum = 0.0, sum_squared = 0.0;
    if (source_codec == target_codec) {
        memcpy(output, input, source_bytes);
    } else {
        for (size_t row = 0; row < rows; row++) {
            const uint8_t *source = input + row * source_row_bytes;
            uint8_t *target = output + row * target_row_bytes;
            for (size_t lane = 0; lane < head_dim; lane++) {
                status = write_float(scratch + lane,
                                     kv_lane(source, source_codec, lane, group_size, source_group_bytes));
                if (status) return status;
            }
            if (target_codec == 0) memcpy(target, scratch, scratch_bytes);
            else if (target_codec == 4)
                status = nexa_q4_quantize(scratch, head_dim, 1, head_dim, group_size, target, target_row_bytes);
            else
                status = nexa_q3_quantize(scratch, head_dim, 1, head_dim, group_size, target, target_row_bytes);
            if (status) return status;
            for (size_t lane = 0; lane < head_dim; lane++) {
                double error = kv_lane(source, source_codec, lane, group_size, source_group_bytes) -
                               kv_lane(target, target_codec, lane, group_size, target_group_bytes);
                if (fabs(error) > maximum) maximum = fabs(error);
                sum_squared += error * error;
                if (!isfinite(sum_squared)) return NEXA_Q4_NUMERIC_RANGE;
            }
        }
    }
    stats[0] = maximum;
    stats[1] = sum_squared;
    stats[2] = (double)elements;
    return NEXA_Q4_OK;
}

static int attention_accumulator_layout(size_t sequence, size_t heads, size_t dim,
                                        size_t *heads_total, size_t *elements,
                                        size_t *q_bytes, size_t *max_bytes,
                                        size_t *sum_count, size_t *sum_bytes) {
    if (!sequence || !heads || !dim) return NEXA_Q4_INVALID_ARGUMENT;
    if (dim == SIZE_MAX || !checked_mul(sequence, heads, heads_total) ||
        !checked_mul(*heads_total, dim, elements) || !float_bytes(*elements, q_bytes) ||
        !checked_mul(*heads_total, sizeof(double), max_bytes) ||
        !checked_mul(*heads_total, dim + 1, sum_count) ||
        !checked_mul(*sum_count, sizeof(double), sum_bytes)) return NEXA_Q4_OVERFLOW;
    return NEXA_Q4_OK;
}

static int valid_attention_accumulators(const double *maxima, const double *sums,
                                        size_t heads_total, size_t dim, int phase) {
    for (size_t head = 0; head < heads_total; head++) {
        double maximum = maxima[head];
        if (!isfinite(maximum) && !(phase == 0 && maximum == -INFINITY)) return 0;
        const double *sum = sums + head * (dim + 1);
        if (!isfinite(sum[0]) || sum[0] < 0.0 || (phase == 2 && sum[0] == 0.0)) return 0;
        for (size_t lane = 1; lane <= dim; lane++) {
            if (!isfinite(sum[lane])) return 0;
        }
    }
    return 1;
}

int nexa_causal_gqa_attention_page(
    const float *query, size_t query_count,
    const uint8_t *key, size_t key_bytes, const uint8_t *value, size_t value_bytes,
    int codec, size_t group_size, size_t page_start, size_t page_valid_tokens,
    size_t past_length, size_t sequence, size_t query_heads, size_t kv_heads, size_t head_dim,
    int phase, double *maxima, size_t maxima_count, double *sums, size_t sums_count) {
    if (!query || !key || !value || !maxima || !sums || !page_valid_tokens || !kv_heads ||
        query_heads % kv_heads || (phase != 0 && phase != 1) ||
        (uintptr_t)query % _Alignof(float) || (uintptr_t)maxima % _Alignof(double) ||
        (uintptr_t)sums % _Alignof(double)) return NEXA_Q4_INVALID_ARGUMENT;
    size_t heads_total, q_elements, q_bytes, max_bytes, sum_count, sum_bytes;
    int status = attention_accumulator_layout(sequence, query_heads, head_dim,
                                              &heads_total, &q_elements, &q_bytes,
                                              &max_bytes, &sum_count, &sum_bytes);
    if (status) return status;
    if (past_length > SIZE_MAX - sequence || page_start > SIZE_MAX - page_valid_tokens)
        return NEXA_Q4_OVERFLOW;
    size_t cache_length = past_length + sequence;
    if (page_start + page_valid_tokens > cache_length) return NEXA_Q4_INVALID_ARGUMENT;
    size_t row_bytes, group_bytes, rows, payload_bytes;
    status = kv_row_layout(codec, head_dim, group_size, &row_bytes, &group_bytes);
    if (status) return status;
    if (!checked_mul(page_valid_tokens, kv_heads, &rows) ||
        !checked_mul(rows, row_bytes, &payload_bytes)) return NEXA_Q4_OVERFLOW;
    if (query_count < q_elements || key_bytes < payload_bytes || value_bytes < payload_bytes ||
        maxima_count < heads_total || sums_count < sum_count) return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(maxima, max_bytes, sums, sum_bytes) ||
        !disjoint(maxima, max_bytes, query, q_bytes) ||
        !disjoint(sums, sum_bytes, query, q_bytes) ||
        !disjoint(maxima, max_bytes, key, payload_bytes) ||
        !disjoint(sums, sum_bytes, key, payload_bytes) ||
        !disjoint(maxima, max_bytes, value, payload_bytes) ||
        !disjoint(sums, sum_bytes, value, payload_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    if (!finite_input(query, q_elements) ||
        !valid_kv_rows(key, codec, rows, head_dim, group_size, group_bytes, row_bytes) ||
        !valid_kv_rows(value, codec, rows, head_dim, group_size, group_bytes, row_bytes) ||
        !valid_attention_accumulators(maxima, sums, heads_total, head_dim, phase))
        return NEXA_Q4_INVALID_DATA;
    double scale = 1.0 / sqrt((double)head_dim);
    size_t repeats = query_heads / kv_heads;
    for (size_t position = 0; position < sequence; position++) {
        size_t causal_end = past_length + position;
        if (page_start > causal_end) continue;
        size_t visible_tokens = causal_end - page_start + 1;
        if (visible_tokens > page_valid_tokens) visible_tokens = page_valid_tokens;
        for (size_t head = 0; head < query_heads; head++) {
            size_t head_index = position * query_heads + head;
            const float *q = query + head_index * head_dim;
            size_t kv_head = head / repeats;
            double *sum = sums + head_index * (head_dim + 1);
            for (size_t token = 0; token < visible_tokens; token++) {
                const uint8_t *k = key + (token * kv_heads + kv_head) * row_bytes;
                double dot = 0.0;
                for (size_t lane = 0; lane < head_dim; lane++)
                    dot += (double)q[lane] * kv_lane(k, codec, lane, group_size, group_bytes);
                double score = dot * scale;
                if (!isfinite(score)) return NEXA_Q4_NUMERIC_RANGE;
                if (phase == 0) {
                    if (score > maxima[head_index]) maxima[head_index] = score;
                } else {
                    /* A complete first pass must bound every score in phase 1.
                     * Detect changed pages or an incomplete maximum pass. */
                    if (score > maxima[head_index]) return NEXA_Q4_INVALID_DATA;
                    float weight = (float)exp(score - maxima[head_index]);
                    sum[0] += (double)weight;
                    if (!isfinite(sum[0])) return NEXA_Q4_NUMERIC_RANGE;
                    const uint8_t *v = value + (token * kv_heads + kv_head) * row_bytes;
                    for (size_t lane = 0; lane < head_dim; lane++) {
                        sum[lane + 1] += (double)weight * kv_lane(v, codec, lane, group_size, group_bytes);
                        if (!isfinite(sum[lane + 1])) return NEXA_Q4_NUMERIC_RANGE;
                    }
                }
            }
        }
    }
    return NEXA_Q4_OK;
}

int nexa_causal_gqa_attention_finish(
    const double *maxima, size_t maxima_count, const double *sums, size_t sums_count,
    size_t sequence, size_t query_heads, size_t head_dim,
    float *output, size_t output_count) {
    if (!maxima || !sums || !output || (uintptr_t)maxima % _Alignof(double) ||
        (uintptr_t)sums % _Alignof(double) || (uintptr_t)output % _Alignof(float))
        return NEXA_Q4_INVALID_ARGUMENT;
    size_t heads_total, elements, output_bytes, max_bytes, sum_count, sum_bytes;
    int status = attention_accumulator_layout(sequence, query_heads, head_dim,
                                              &heads_total, &elements, &output_bytes,
                                              &max_bytes, &sum_count, &sum_bytes);
    if (status) return status;
    if (maxima_count < heads_total || sums_count < sum_count || output_count < elements)
        return NEXA_Q4_BUFFER_TOO_SMALL;
    if (!disjoint(maxima, max_bytes, sums, sum_bytes) ||
        !disjoint(maxima, max_bytes, output, output_bytes) ||
        !disjoint(sums, sum_bytes, output, output_bytes)) return NEXA_Q4_INVALID_ARGUMENT;
    if (!valid_attention_accumulators(maxima, sums, heads_total, head_dim, 2))
        return NEXA_Q4_INVALID_DATA;
    for (size_t head = 0; head < heads_total; head++) {
        const double *sum = sums + head * (head_dim + 1);
        for (size_t lane = 0; lane < head_dim; lane++) {
            status = write_float(output + head * head_dim + lane, sum[lane + 1] / sum[0]);
            if (status) return status;
        }
    }
    return NEXA_Q4_OK;
}
