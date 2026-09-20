#ifndef NEXA_NEXAPACK_TRANSFORMER_H
#define NEXA_NEXAPACK_TRANSFORMER_H

#include "q4.h"

#ifdef __cplusplus
extern "C" {
#endif

/* CPU float32 Transformer kernels. Counts are element capacities, not bytes.
 * All dimensions must be positive. Status codes are enum nexa_q4_status.
 * No heap allocation occurs. Output must be disjoint from every input.
 * Inputs must be finite. Discard output after any error (partial writes are
 * possible on numeric range failures). Excess buffer capacity is unused.
 */
NEXA_Q4_API int nexa_rmsnorm(
    const float *input, size_t input_count,
    const float *weight, size_t weight_count,
    size_t sequence, size_t hidden, double epsilon,
    float *output, size_t output_count);

/* Full-head HF RoPE: rotate halves [-x_second, x_first], not adjacent pairs.
 * Position is the sequence index 0..sequence-1. Frequency lane j is
 * theta^(-2*j/head_dim). head_dim must be even; theta must be positive/finite.
 */
NEXA_Q4_API int nexa_rope(
    const float *input, size_t input_count,
    size_t sequence, size_t heads, size_t head_dim, double theta,
    float *output, size_t output_count);

/* RoPE for an appended chunk: sequence position i uses position_offset+i.
 * Position addition is checked before accessing buffers. All other contracts
 * match nexa_rope; the existing API is equivalent to position_offset=0.
 */
NEXA_Q4_API int nexa_rope_offset(
    const float *input, size_t input_count,
    size_t sequence, size_t heads, size_t head_dim, double theta,
    size_t position_offset, float *output, size_t output_count);

/* output = SiLU(gate) * up; SiLU uses a sign-stable sigmoid evaluation. */
NEXA_Q4_API int nexa_swiglu(
    const float *gate, size_t gate_count,
    const float *up, size_t up_count,
    size_t elements, float *output, size_t output_count);

NEXA_Q4_API int nexa_add(
    const float *left, size_t left_count,
    const float *right, size_t right_count,
    size_t elements, float *output, size_t output_count);

/* Causal prefill attention, sequence-major [S,heads,head_dim]. Query head h
 * attends to KV head h/(query_heads/kv_heads); query_heads%kv_heads must be 0.
 * Scores use 1/sqrt(head_dim). Each position sees keys 0..position inclusive.
 * scratch holds sequence floats and must be disjoint from inputs and output.
 * It stores max-shifted exponential weights; reductions accumulate in double.
 * Neither a square attention matrix nor a decoded weight matrix is allocated.
 */
NEXA_Q4_API int nexa_causal_gqa_attention(
    const float *query, size_t query_count,
    const float *key, size_t key_count,
    const float *value, size_t value_count,
    size_t sequence, size_t query_heads, size_t kv_heads, size_t head_dim,
    float *scratch, size_t scratch_count,
    float *output, size_t output_count);

/* Incremental causal attention. query/output hold [sequence,query_heads,D].
 * key/value start at cache position zero and contain an initialized prefix of
 * past_length+sequence rows [row,kv_heads,D]. The caller writes the appended
 * K/V rows before this call. Query row i sees cache rows 0..past_length+i.
 * key_count/value_count may include extra capacity; unused suffix data is
 * neither read nor validated (it may contain NaNs left by an aborted append).
 * scratch requires past_length+sequence floats. Its used span and the output
 * must be disjoint from each other and all active input spans. Cache data is
 * never modified. Reductions/softmax match nexa_causal_gqa_attention exactly.
 */
NEXA_Q4_API int nexa_causal_gqa_attention_cached(
    const float *query, size_t query_count,
    const float *key, size_t key_count,
    const float *value, size_t value_count,
    size_t past_length, size_t sequence, size_t query_heads, size_t kv_heads, size_t head_dim,
    float *scratch, size_t scratch_count,
    float *output, size_t output_count);

/* Paged incremental F32 KV. Pointer tables are ordered by logical page; their
 * physical pages may be discontiguous or arranged in any order. Each page is
 * token-major [page_tokens,kv_heads,head_dim] and page_floats supplies the
 * uniform capacity of every page in floats (at least page_tokens*KVH*D).
 * Tables require ceil((past_length+sequence)/page_tokens) pointers. Only those
 * pointers and their visible prefix data are inspected; extra table entries
 * and the unused suffix of the last page are ignored. No concatenation or heap
 * allocation occurs. Scratch holds past_length+sequence floats. Output and
 * scratch must be disjoint from each other, query, used pointer-table spans,
 * and every active page span. Cache data and tables are never modified.
 */
/* Dense F16 weight tile, row-major, two bytes per coordinate; same contract
 * as nexa_f32_matmul. Nonfinite stored values are rejected. */
NEXA_Q4_API int nexa_f16_matmul(
    const float *inputs, size_t input_count, size_t batch,
    const uint8_t *weights, size_t weight_bytes,
    size_t rows, size_t cols,
    float *output, size_t output_count);

NEXA_Q4_API int nexa_f16_decode_row(
    const uint8_t *weights, size_t weight_bytes, size_t cols,
    float *output, size_t output_count);

/* Dense F32 weight tile of `rows` x `cols`, row-major, matching the packed
 * matmul contract: inputs are batch x cols and outputs batch x rows.
 * Returns NEXA_Q4_INVALID_DATA for nonfinite inputs or weights. */
NEXA_Q4_API int nexa_f32_matmul(
    const float *inputs, size_t input_count, size_t batch,
    const float *weights, size_t weight_count,
    size_t rows, size_t cols,
    float *output, size_t output_count);

NEXA_Q4_API int nexa_causal_gqa_attention_paged(
    const float *query, size_t query_count,
    const float *const *key_pages, size_t key_page_count,
    const float *const *value_pages, size_t value_page_count,
    size_t page_tokens, size_t page_floats,
    size_t past_length, size_t sequence, size_t query_heads, size_t kv_heads, size_t head_dim,
    float *scratch, size_t scratch_count,
    float *output, size_t output_count);

/* Paged Q4 KV using the existing nexa_q4_quantize row format. Each token holds
 * kv_heads independent packed rows of head_dim values. page_bytes supplies the
 * uniform byte capacity of each key or value page, at least
 * page_tokens*kv_heads*nexa_q4_row_size(head_dim,group_size).
 * Pointer-table order is logical; pages need not be physically contiguous.
 * Only the past_length+sequence prefix is validated/read. Missing/dirty suffix
 * entries are ignored. Scale/code/padding validation matches the Q4 codec.
 * Packed values are multiplied scale*code in double without an intermediate
 * float32 rounding. Dot products/weighted values reduce in double; exponential
 * scratch weights are float32 as in the F32 attention kernels. No decoded row,
 * cache concatenation, or heap allocation is performed. Aliasing, scratch and
 * status contracts match nexa_causal_gqa_attention_paged.
 */
NEXA_Q4_API int nexa_causal_gqa_attention_paged_q4(
    const float *query, size_t query_count,
    const uint8_t *const *key_pages, size_t key_page_count,
    const uint8_t *const *value_pages, size_t value_page_count,
    size_t page_tokens, size_t page_bytes, size_t group_size,
    size_t past_length, size_t sequence, size_t query_heads, size_t kv_heads, size_t head_dim,
    float *scratch, size_t scratch_count,
    float *output, size_t output_count);

/* Q3_GROUPED V1: independent rows, each group stores a little-endian binary32
 * scale followed by ceil(3*group_size/8) bytes. Signed two's-complement codes
 * -3..3 occupy bits 3*i..3*i+2, least-significant bit first across bytes.
 * Code 4 (-4) is reserved. Missing lanes and unused high bits must be zero;
 * a zero scale requires zero codes. Scales must be finite and nonnegative.
 * Shape/layout errors return zero from row_size, otherwise enum nexa_q4_status.
 */
NEXA_Q4_API size_t nexa_q3_row_size(size_t cols, size_t group_size);

/* Quantize weights[rows,cols] with float32(max(abs(group))/3), divide by that
 * stored scale, round half away from zero and clamp to [-3,3]. Nonzero groups
 * whose scale rounds to zero fail. Counts are float/byte capacities as named.
 * Input/output must not overlap. No allocation occurs; discard output on error.
 */
NEXA_Q4_API int nexa_q3_quantize(
    const float *weights, size_t weight_count,
    size_t rows, size_t cols, size_t group_size,
    uint8_t *packed, size_t packed_bytes);

/* Paged Q3 KV. Same capacities, causal/GQA, aliasing and numeric contracts as
 * paged_q4, using nexa_q3_row_size(head_dim,group_size) bytes per head. Only
 * the visible prefix is validated/read. Scale*code stays double, without a
 * decoded float32 vector, page concatenation or heap allocation.
 */
NEXA_Q4_API int nexa_causal_gqa_attention_paged_q3(
    const float *query, size_t query_count,
    const uint8_t *const *key_pages, size_t key_page_count,
    const uint8_t *const *value_pages, size_t value_page_count,
    size_t page_tokens, size_t page_bytes, size_t group_size,
    size_t past_length, size_t sequence, size_t query_heads, size_t kv_heads, size_t head_dim,
    float *scratch, size_t scratch_count,
    float *output, size_t output_count);

/* Mixed CPU pages: codec 0 = native F32, 4 = Q4_GROUPED V1, 3 = Q3_GROUPED V1.
 * K and V share each logical page's codec and capacity, supplied by parallel
 * page_codecs/page_bytes tables. Each capacity covers page_tokens complete
 * tokens, although only the visible prefix is accessed/validated. Extra table
 * entries and the final unused suffix are ignored. Packed scale*code stays
 * double, matching homogeneous attention. No decoded vectors or heap are used.
 * Query/output/scratch and typed tables require their natural alignment;
 * page payloads may be unaligned, including native-endian F32 pages.
 * Output/scratch must not alias any active payload or any used metadata table.
 */
NEXA_Q4_API int nexa_causal_gqa_attention_paged_mixed(
    const float *query, size_t query_count,
    const uint8_t *const *key_pages, size_t key_page_count,
    const uint8_t *const *value_pages, size_t value_page_count,
    const uint8_t *page_codecs, size_t codec_count,
    const size_t *page_bytes, size_t byte_count,
    size_t page_tokens, size_t group_size,
    size_t past_length, size_t sequence, size_t query_heads, size_t kv_heads, size_t head_dim,
    float *scratch, size_t scratch_count,
    float *output, size_t output_count);

/* Reencode independent heads with the same codec IDs as mixed attention.
 * All pairs are supported. Identity copies validated bytes without rounding.
 * Other pairs reconstruct one source head to caller-owned scratch[D] F32,
 * then encode/copy that head. F32 payloads use native endian and may be
 * unaligned; scratch/stats require their natural float/double alignment.
 * Stats has three doubles: max absolute error, sum squared error, value count.
 * Errors compare original and target scalar reconstructions in double (packed
 * scale*code), including any F32 bridge rounding. Stats is written only after
 * complete success. Input/output/scratch/stats spans must be disjoint. There
 * is no heap allocation. Discard output/scratch after any failure; numeric
 * range errors (including an unrepresentable source F32 bridge) may leave
 * partial output. Unused capacities are neither read nor written.
 */
NEXA_Q4_API int nexa_kv_reencode_rows(
    const uint8_t *input, size_t input_bytes, int source_codec, int target_codec,
    size_t rows, size_t head_dim, size_t group_size,
    uint8_t *output, size_t output_bytes,
    float *scratch, size_t scratch_count, double *stats, size_t stats_count);

/* Bounded two-pass attention over one loaded page at a time. The caller visits
 * every page in increasing logical token order, without gaps/duplicates, first
 * with phase=0 and then phase=1. This ordering/coverage is caller-owned state;
 * this stateless kernel cannot detect missing or repeated pages. The page's
 * positive token interval must lie in [0,past_length+sequence). Individual
 * queries only see positions <= past_length+query_position.
 *
 * Initialize maxima[sequence*query_heads] to -INFINITY and sums to zero before
 * phase 0. sums has [sequence,query_heads,head_dim+1] doubles: denominator first,
 * then weighted value sums. Phase 0 updates maxima and preserves sums. After
 * ALL phase-0 pages, phase 1 preserves maxima and adds F32 exponential weights
 * and weighted values in token order. Scalar arithmetic matches mixed paged
 * attention; there is no prefix-sized score array, reconstructed head or heap.
 * Codecs 0/4/3 and packed validation match the mixed attention API. Capacities
 * need only cover page_valid_tokens; excess payload bytes are ignored. Both
 * page payloads are validated in either phase. They may be unaligned; query,
 * maxima and sums require natural alignment. Mutable spans must be disjoint
 * from each other and inputs. On any error discard both accumulators, since
 * numeric failures may leave partial state. Counts are element capacities,
 * except explicitly named key_bytes/value_bytes.
 */
NEXA_Q4_API int nexa_causal_gqa_attention_page(
    const float *query, size_t query_count,
    const uint8_t *key, size_t key_bytes, const uint8_t *value, size_t value_bytes,
    int codec, size_t group_size, size_t page_start, size_t page_valid_tokens,
    size_t past_length, size_t sequence, size_t query_heads, size_t kv_heads, size_t head_dim,
    int phase, double *maxima, size_t maxima_count, double *sums, size_t sums_count);

/* Finalize after both complete passes. Every maximum must be finite, every
 * denominator positive/finite and every weighted sum finite. Caller-owned
 * state is read-only; output is disjoint from both used accumulator spans.
 * Discard output on error (F32 range failures can leave partial output).
 */
NEXA_Q4_API int nexa_causal_gqa_attention_finish(
    const double *maxima, size_t maxima_count, const double *sums, size_t sums_count,
    size_t sequence, size_t query_heads, size_t head_dim,
    float *output, size_t output_count);

#ifdef __cplusplus
}
#endif
#endif
