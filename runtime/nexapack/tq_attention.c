#include "tq_attention.h"

#include <float.h>
#include <math.h>
#include <string.h>

static int tq_attn_mul(size_t a, size_t b, size_t *out) {
    if (b && a > SIZE_MAX / b) return 0;
    *out = a * b;
    return 1;
}

static int tq_attn_disjoint(const void *a, size_t an, const void *b, size_t bn) {
    uintptr_t ap = (uintptr_t)a, bp = (uintptr_t)b;
    if (an > UINTPTR_MAX - ap || bn > UINTPTR_MAX - bp) return 0;
    return ap + an <= bp || bp + bn <= ap;
}

static int tq_attn_aligned(const void *p, size_t alignment) {
    return (uintptr_t)p % alignment == 0;
}

static int tq_attn_valid_row(const uint8_t *row, size_t bytes, size_t dim, int bits) {
    if (memcmp(row, "TQ02", 4)) return 0;
    uint32_t norm = (uint32_t)row[4] | ((uint32_t)row[5] << 8) |
                    ((uint32_t)row[6] << 16) | ((uint32_t)row[7] << 24);
    if ((norm & 0x80000000u) || (norm & 0x7f800000u) == 0x7f800000u) return 0;
    unsigned tail = (unsigned)((dim * (size_t)bits) & 7u);
    if (tail && (row[bytes - 1] >> tail)) return 0;
    if (!norm) for (size_t i = 8; i < bytes; i++) if (row[i]) return 0;
    return 1;
}

static const uint8_t *tq_attn_row(const uint8_t *const *pages, size_t token,
                                 size_t page_tokens, size_t token_bytes,
                                 size_t row_bytes, size_t head) {
    return pages[token / page_tokens] + (token % page_tokens) * token_bytes + head * row_bytes;
}

static int tq_attn_decode(const tq_ctx *ctx, const uint8_t *row, size_t row_bytes,
                          float *vector, size_t dim) {
    int result = tq_dequantize_tq02(ctx, row, row_bytes, vector, dim, 1);
    if (result == -4) return NEXA_Q4_NUMERIC_RANGE;
    if (result) return NEXA_Q4_INVALID_DATA; /* Shape/alias were already checked. */
    return NEXA_Q4_OK;
}

int nexa_causal_gqa_attention_paged_tq(
    const tq_ctx *ctx,
    const float *query, size_t query_count,
    const uint8_t *const *key_pages, size_t key_page_count,
    const uint8_t *const *value_pages, size_t value_page_count,
    size_t page_tokens, size_t page_bytes,
    size_t past_length, size_t sequence, size_t query_heads, size_t kv_heads, size_t head_dim,
    float *scores, size_t scores_count,
    float *vector_scratch, size_t vector_count,
    double *accumulator, size_t accumulator_count,
    float *output, size_t output_count) {
    if (!ctx || !query || !key_pages || !value_pages || !scores || !vector_scratch ||
        !accumulator || !output || !page_tokens || !sequence || !query_heads || !kv_heads ||
        !head_dim || query_heads % kv_heads || (size_t)tq_context_dim(ctx) != head_dim ||
        !tq_attn_aligned(query, _Alignof(float)) || !tq_attn_aligned(scores, _Alignof(float)) ||
        !tq_attn_aligned(vector_scratch, _Alignof(float)) || !tq_attn_aligned(output, _Alignof(float)) ||
        !tq_attn_aligned(accumulator, _Alignof(double)) ||
        !tq_attn_aligned(key_pages, _Alignof(const uint8_t *)) ||
        !tq_attn_aligned(value_pages, _Alignof(const uint8_t *))) return NEXA_Q4_INVALID_ARGUMENT;
    if (past_length > SIZE_MAX - sequence) return NEXA_Q4_OVERFLOW;
    size_t cache_length = past_length + sequence;
    size_t needed_pages = cache_length / page_tokens + (cache_length % page_tokens != 0);
    size_t row_bytes = tq_packed_size(ctx, 1);
    if (!row_bytes) return NEXA_Q4_INVALID_ARGUMENT;
    size_t width, elements, query_bytes, token_bytes, full_page_bytes, cache_bytes;
    size_t table_bytes, score_bytes, vector_bytes, accumulator_bytes;
    if (!tq_attn_mul(query_heads, head_dim, &width) ||
        !tq_attn_mul(sequence, width, &elements) || !tq_attn_mul(elements, sizeof(float), &query_bytes) ||
        !tq_attn_mul(kv_heads, row_bytes, &token_bytes) ||
        !tq_attn_mul(page_tokens, token_bytes, &full_page_bytes) ||
        !tq_attn_mul(cache_length, token_bytes, &cache_bytes) ||
        !tq_attn_mul(needed_pages, sizeof(*key_pages), &table_bytes) ||
        !tq_attn_mul(cache_length, sizeof(float), &score_bytes) ||
        !tq_attn_mul(head_dim, sizeof(float), &vector_bytes) ||
        !tq_attn_mul(head_dim, sizeof(double), &accumulator_bytes)) return NEXA_Q4_OVERFLOW;
    if (query_count < elements || output_count < elements || scores_count < cache_length ||
        vector_count < head_dim || accumulator_count < head_dim || page_bytes < full_page_bytes ||
        key_page_count < needed_pages || value_page_count < needed_pages) return NEXA_Q4_BUFFER_TOO_SMALL;

    const void *writable[] = {scores, vector_scratch, accumulator, output};
    const size_t write_bytes[] = {score_bytes, vector_bytes, accumulator_bytes, query_bytes};
    const void *readonly[] = {query, key_pages, value_pages};
    const size_t read_bytes[] = {query_bytes, table_bytes, table_bytes};
    for (size_t i = 0; i < 4; i++) {
        if (!tq_context_buffer_disjoint(ctx, writable[i], write_bytes[i])) return NEXA_Q4_INVALID_ARGUMENT;
        for (size_t j = 0; j < i; j++)
            if (!tq_attn_disjoint(writable[i], write_bytes[i], writable[j], write_bytes[j]))
                return NEXA_Q4_INVALID_ARGUMENT;
        for (size_t j = 0; j < 3; j++)
            if (!tq_attn_disjoint(writable[i], write_bytes[i], readonly[j], read_bytes[j]))
                return NEXA_Q4_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < 3; i++)
        if (!tq_context_buffer_disjoint(ctx, readonly[i], read_bytes[i])) return NEXA_Q4_INVALID_ARGUMENT;
    size_t remaining = cache_length;
    for (size_t page = 0; page < needed_pages; page++) {
        const uint8_t *key = key_pages[page], *value = value_pages[page];
        size_t visible = remaining < page_tokens ? remaining : page_tokens;
        size_t bytes = visible * token_bytes;
        if (!tq_context_buffer_disjoint(ctx, key, bytes) ||
            !tq_context_buffer_disjoint(ctx, value, bytes)) return NEXA_Q4_INVALID_ARGUMENT;
        for (size_t i = 0; i < 4; i++)
            if (!tq_attn_disjoint(key, bytes, writable[i], write_bytes[i]) ||
                !tq_attn_disjoint(value, bytes, writable[i], write_bytes[i])) return NEXA_Q4_INVALID_ARGUMENT;
        remaining -= visible;
    }
    for (size_t i = 0; i < elements; i++) if (!isfinite(query[i])) return NEXA_Q4_INVALID_DATA;
    remaining = cache_length;
    for (size_t page = 0; page < needed_pages; page++) {
        size_t visible = remaining < page_tokens ? remaining : page_tokens;
        for (size_t row = 0; row < visible * kv_heads; row++)
            if (!tq_attn_valid_row(key_pages[page] + row * row_bytes, row_bytes, head_dim, tq_context_bits(ctx)) ||
                !tq_attn_valid_row(value_pages[page] + row * row_bytes, row_bytes, head_dim, tq_context_bits(ctx)))
                return NEXA_Q4_INVALID_DATA;
        remaining -= visible;
    }

    double scale = 1.0 / sqrt((double)head_dim);
    size_t repeats = query_heads / kv_heads;
    for (size_t position = 0; position < sequence; position++) {
        size_t causal_end = past_length + position;
        for (size_t head = 0; head < query_heads; head++) {
            const float *q = query + position * width + head * head_dim;
            size_t kv_head = head / repeats;
            double maximum = -INFINITY;
            for (size_t past = 0; past <= causal_end; past++) {
                const uint8_t *row = tq_attn_row(key_pages, past, page_tokens, token_bytes, row_bytes, kv_head);
                int status = tq_attn_decode(ctx, row, row_bytes, vector_scratch, head_dim);
                if (status) return status;
                double dot = 0.0;
                for (size_t lane = 0; lane < head_dim; lane++) dot += (double)q[lane] * (double)vector_scratch[lane];
                double score = dot * scale;
                if (!isfinite(score)) return NEXA_Q4_NUMERIC_RANGE;
                if (score > maximum) maximum = score;
            }
            double denominator = 0.0;
            for (size_t past = 0; past <= causal_end; past++) {
                const uint8_t *row = tq_attn_row(key_pages, past, page_tokens, token_bytes, row_bytes, kv_head);
                int status = tq_attn_decode(ctx, row, row_bytes, vector_scratch, head_dim);
                if (status) return status;
                double dot = 0.0;
                for (size_t lane = 0; lane < head_dim; lane++) dot += (double)q[lane] * (double)vector_scratch[lane];
                scores[past] = (float)exp(dot * scale - maximum);
                denominator += (double)scores[past];
            }
            memset(accumulator, 0, accumulator_bytes);
            for (size_t past = 0; past <= causal_end; past++) {
                const uint8_t *row = tq_attn_row(value_pages, past, page_tokens, token_bytes, row_bytes, kv_head);
                int status = tq_attn_decode(ctx, row, row_bytes, vector_scratch, head_dim);
                if (status) return status;
                for (size_t lane = 0; lane < head_dim; lane++)
                    accumulator[lane] += (double)scores[past] * (double)vector_scratch[lane];
            }
            for (size_t lane = 0; lane < head_dim; lane++) {
                double value = accumulator[lane] / denominator;
                if (!isfinite(value) || fabs(value) > FLT_MAX) return NEXA_Q4_NUMERIC_RANGE;
                output[position * width + head * head_dim + lane] = (float)value;
            }
        }
    }
    return NEXA_Q4_OK;
}
