#ifndef NEXA_NEXAPACK_TQ_ATTENTION_H
#define NEXA_NEXAPACK_TQ_ATTENTION_H

#include "../turboquant.h"
#include "q4.h" /* Shared status values only; no Q4 library dependency. */

#ifdef __cplusplus
extern "C" {
#endif

/* Causal attention over TQ_MSE_SRHT v1 / TQ02 KV pages. Each page is
 * [page_tokens, kv_heads, row_bytes], row_bytes = 8 + ceil(D * bits / 8).
 * Context dimension must equal head_dim. The caller owns the immutable context
 * and supplies its original codebook/seed; the context is never changed.
 *
 * Scratch is caller-owned: scores[past_length + sequence] float elements,
 * vector_scratch[head_dim] float elements, accumulator[head_dim] doubles.
 * Decode reconstructs one K/V head at a time, never a page or cache prefix.
 * No heap allocations; key vectors are decoded twice for stable softmax and
 * each V once. Double dot products and accumulation preserve finite values
 * whose unshifted attention scores exceed F32 range.
 *
 * Only visible tokens are inspected; unused page slots and extra pointer-table
 * entries are ignored. page_bytes must accommodate a full physical page.
 * Consumed spans must not wrap addresses. Writable spans must be mutually
 * disjoint and disjoint from query, consumed tables/pages and context storage.
 * All float/double/table pointers require their natural alignment.
 * Counts refer to elements, except page_bytes. Extra capacity is untouched.
 * Status uses nexa_q4_status. Discard output/scratch on error: numeric failures
 * can leave partial results. Invalid structure is checked before output writes.
 */
NEXA_Q4_API int nexa_causal_gqa_attention_paged_tq(
    const tq_ctx *ctx,
    const float *query, size_t query_count,
    const uint8_t *const *key_pages, size_t key_page_count,
    const uint8_t *const *value_pages, size_t value_page_count,
    size_t page_tokens, size_t page_bytes,
    size_t past_length, size_t sequence, size_t query_heads, size_t kv_heads, size_t head_dim,
    float *scores, size_t scores_count,
    float *vector_scratch, size_t vector_count,
    double *accumulator, size_t accumulator_count,
    float *output, size_t output_count);

#ifdef __cplusplus
}
#endif
#endif
