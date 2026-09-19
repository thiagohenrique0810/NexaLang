/* M1.07: public contracts, allocation failures, and exact MSE compatibility.
 * Compile this file alone: it instruments allocations in the included runtime.
 * Caller buffers and allocator bookkeeping are deliberately outside the ledger.
 */
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <pthread.h>
#endif

void* tq_test_malloc(size_t size);
void* tq_test_calloc(size_t count, size_t size);
void* tq_test_realloc(void* ptr, size_t size);
void tq_test_free(void* ptr);
#define malloc tq_test_malloc
#define calloc tq_test_calloc
#define realloc tq_test_realloc
#define free tq_test_free
#include "turboquant.c"
#undef malloc
#undef calloc
#undef realloc
#undef free

typedef struct { void* ptr; size_t size; } allocation;
static allocation ledger[128];
static size_t live_bytes, peak_bytes, live_blocks, allocation_calls, fail_at;
#ifndef _WIN32
static pthread_mutex_t ledger_lock = PTHREAD_MUTEX_INITIALIZER;
#define LOCK() assert(pthread_mutex_lock(&ledger_lock) == 0)
#define UNLOCK() assert(pthread_mutex_unlock(&ledger_lock) == 0)
#else
#define LOCK() ((void)0)
#define UNLOCK() ((void)0)
#endif

static void record(void* ptr, size_t size) {
    if (!ptr) return;
    for (size_t i = 0; i < 128; i++) if (!ledger[i].ptr) {
        ledger[i] = (allocation){ptr, size};
        live_blocks++; live_bytes += size;
        if (live_bytes > peak_bytes) peak_bytes = live_bytes;
        return;
    }
    assert(!"allocator ledger exhausted");
}
static void forget(void* ptr) {
    if (!ptr) return;
    for (size_t i = 0; i < 128; i++) if (ledger[i].ptr == ptr) {
        live_bytes -= ledger[i].size; live_blocks--;
        ledger[i] = (allocation){NULL, 0}; return;
    }
    assert(!"foreign pointer or double free");
}
void* tq_test_malloc(size_t size) {
    LOCK(); allocation_calls++;
    void* ptr = allocation_calls == fail_at ? NULL : malloc(size);
    record(ptr, size); UNLOCK(); return ptr;
}
void* tq_test_calloc(size_t count, size_t size) {
    LOCK(); allocation_calls++;
    void* ptr = allocation_calls == fail_at ? NULL : calloc(count, size);
    assert(!size || count <= SIZE_MAX / size);
    record(ptr, count * size); UNLOCK(); return ptr;
}
void* tq_test_realloc(void* ptr, size_t size) {
    LOCK(); allocation_calls++;
    if (allocation_calls == fail_at) { UNLOCK(); return NULL; }
    /* Runtime currently does not use realloc; preserve failure ownership anyway. */
    size_t old_size = 0;
    for (size_t i = 0; i < 128; i++) if (ledger[i].ptr == ptr) old_size = ledger[i].size;
    forget(ptr);
    void* replacement = realloc(ptr, size);
    if (!replacement && size) record(ptr, old_size);
    else record(replacement, size);
    UNLOCK(); return replacement;
}
void tq_test_free(void* ptr) {
    LOCK(); forget(ptr); free(ptr); UNLOCK();
}
static void measure(size_t failure) {
    allocation_calls = 0; fail_at = failure; peak_bytes = live_bytes;
}
static void empty(void) {
    assert(live_bytes == 0 && live_blocks == 0);
    measure(0);
}
static void input(float* x, size_t count) {
    for (size_t i = 0; i < count; i++) x[i] = ((int)((i * 7 + 3) % 23) - 11) / 16.0f;
}
static void near(float a, float b) {
    assert(isfinite(a) && isfinite(b));
    assert(fabsf(a - b) <= 2e-5f * (1.0f + fabsf(b)));
}

static void equivalence(void) {
    const int dimensions[] = {1, 2, 8, 16, 32, 64, 128, 256, 8, 64};
    const int bits[] = {1, 2, 3, 4, 5, 6, 7, 8, 8, 3};
    const int seeds[] = {0, 1, -7, 42, INT_MIN, INT_MAX, -9, 3, 0, 42};
    for (size_t c = 0; c < sizeof(dimensions) / sizeof(*dimensions); c++) {
        const int d = dimensions[c], n = 5;
        tq_ctx* legacy = tq_create(d, bits[c], seeds[c]);
        tq_ctx* mse = tq_create_mse(d, bits[c], seeds[c]);
        assert(legacy && mse);
        const size_t count = (size_t)d * n, bytes = count * sizeof(float);
        float* x = malloc(bytes); float* left = malloc(bytes); float* right = malloc(bytes);
        uint16_t* a = malloc(count * sizeof(*a)); uint16_t* b = malloc(count * sizeof(*b));
        assert(x && left && right && a && b);
        input(x, count);
        memset(x, 0, (size_t)d * sizeof(float));
        for (int j = 0; j < d; j++) x[d + j] = j & 1 ? -FLT_TRUE_MIN : FLT_TRUE_MIN;
        const size_t owned = live_bytes;
        measure(0); tq_quantize(mse, x, b, n);
        assert(allocation_calls == 1 && live_bytes == owned && peak_bytes == owned + (size_t)d * sizeof(float));
        tq_quantize(legacy, x, a, n);
        assert(memcmp(a, b, count * sizeof(*a)) == 0);
        tq_dequantize(legacy, a, left, n);
        measure(0); tq_dequantize(mse, b, right, n);
        assert(allocation_calls == 0 && live_bytes == owned);
        assert(memcmp(left, right, bytes) == 0);
        const size_t packed_size = tq_packed_size(mse, n);
        assert(packed_size == tq_packed_size(legacy, n));
        uint8_t* p = malloc(packed_size); uint8_t* q = malloc(packed_size);
        assert(p && q);
        assert(tq_quantize_packed(legacy, x, p, n) == 0);
        measure(0); assert(tq_quantize_packed(mse, x, q, n) == 0);
        assert(allocation_calls == 1 && peak_bytes == owned + (size_t)d * sizeof(float));
        assert(memcmp(p, q, packed_size) == 0);
        assert(tq_dequantize_packed(legacy, p, left, n) == 0);
        measure(0); assert(tq_dequantize_packed(mse, q, right, n) == 0);
        assert(allocation_calls == 0 && live_bytes == owned);
        assert(memcmp(left, right, bytes) == 0);
        float lm = tq_mse(legacy, x, n);
        measure(0); float mm = tq_mse(mse, x, n);
        assert(memcmp(&lm, &mm, sizeof(float)) == 0);
        assert(peak_bytes == owned + packed_size + bytes + (size_t)d * sizeof(float));
        assert(live_bytes == owned);
        assert(tq_lower_bound(legacy) == tq_lower_bound(mse));
        assert(tq_upper_bound(legacy) == tq_upper_bound(mse));
        free(x); free(left); free(right); free(a); free(b); free(p); free(q);
        tq_destroy(legacy); tq_destroy(mse); empty();
    }
    puts("MSE packed/raw/bounds/errors byte-identical to legacy across dimensions, bits, seeds, zeros and subnormals");
}

static void parallel_equivalence(void) {
    const int d = 64, n = 1009;
    const size_t count = (size_t)d * n;
    tq_ctx* legacy = tq_create(d, 3, -91); tq_ctx* mse = tq_create_mse(d, 3, -91);
    assert(legacy && mse);
    float* x = malloc(count * sizeof(float)); float* l = malloc(count * sizeof(float));
    float* r = malloc(count * sizeof(float));
    uint16_t* a = malloc(count * sizeof(uint16_t)); uint16_t* b = malloc(count * sizeof(uint16_t));
    assert(x && l && r && a && b); input(x, count);
    tq_quantize(legacy, x, a, n);
    const size_t owned = live_bytes;
    measure(0); tq_quantize_parallel(mse, x, b, n);
#ifndef _WIN32
    assert(allocation_calls == 8 && peak_bytes <= owned + 8u * d * sizeof(float));
#else
    assert(allocation_calls == 1);
#endif
    assert(live_bytes == owned && memcmp(a, b, count * sizeof(*a)) == 0);
    tq_quantize_parallel(legacy, x, b, n);
    assert(memcmp(a, b, count * sizeof(*a)) == 0);
    tq_dequantize(legacy, a, l, n);
    measure(0); tq_dequantize_parallel(mse, b, r, n);
    assert(allocation_calls == 0 && live_bytes == owned);
    assert(memcmp(l, r, count * sizeof(float)) == 0);
    /* Reusing a shared context after workers finish cannot retain scratch. */
    tq_quantize(mse, x, b, n); assert(memcmp(a, b, count * sizeof(*a)) == 0);
    tq_destroy(legacy); tq_destroy(mse);
    free(x); free(l); free(r); free(a); free(b); empty();
    puts("Parallel MSE uses per-worker bounded scratch and preserves byte-exact serial/legacy results");
}

static void creation_failures(void) {
    for (int mode = 0; mode < 2; mode++) {
        for (int bits = 1; bits <= 3; bits += 2) {
            measure(0);
            tq_ctx* ctx = mode ? tq_create(32, bits, 42) : tq_create_mse(32, bits, 42);
            assert(ctx && tq_context_memory_bytes(ctx) == live_bytes);
            const size_t calls = allocation_calls;
            tq_destroy(ctx); empty();
            for (size_t failure = 1; failure <= calls; failure++) {
                measure(failure);
                ctx = mode ? tq_create(32, bits, 42) : tq_create_mse(32, bits, 42);
                assert(!ctx); /* Includes the Lloyd-Max integration bounds allocation. */
                assert(allocation_calls >= failure);
                empty();
            }
            /* A failure in one construction never poisons later contexts. */
            ctx = mode ? tq_create(32, bits, 42) : tq_create_mse(32, bits, 42);
            assert(ctx && tq_context_memory_bytes(ctx) == live_bytes);
            tq_destroy(ctx); empty();
        }
    }
    puts("Every MSE and legacy creation allocation can fail with complete cleanup, including Lloyd-Max scratch");
}

static void operation_failures(void) {
    tq_ctx* ctx = tq_create_mse(16, 3, 42); assert(ctx);
    const size_t owned = live_bytes;
    float x[16], decoded[16]; uint16_t raw[16]; uint8_t packed[14], golden[14];
    input(x, 16); assert(tq_packed_size(ctx, 1) == sizeof(packed));
    assert(tq_quantize_packed(ctx, x, golden, 1) == 0);
    memset(packed, 0xA5, sizeof(packed)); measure(1);
    assert(tq_quantize_packed(ctx, x, packed, 1) == -2);
    for (size_t i = 0; i < sizeof(packed); i++) assert(packed[i] == 0xA5);
    assert(live_bytes == owned);
    measure(0); assert(tq_quantize_packed(ctx, x, packed, 1) == 0);
    assert(memcmp(packed, golden, sizeof(packed)) == 0);
    for (size_t i = 0; i < 16; i++) raw[i] = 0xCAFE;
    measure(1); tq_quantize(ctx, x, raw, 1);
    for (size_t i = 0; i < 16; i++) assert(raw[i] == 0xCAFE);
    assert(live_bytes == owned);
    for (size_t failure = 1; failure <= 3; failure++) {
        measure(failure); assert(isnan(tq_mse(ctx, x, 1)));
        assert(live_bytes == owned && tq_context_memory_bytes(ctx) == owned);
    }
    measure(0); tq_quantize(ctx, x, raw, 1); tq_dequantize(ctx, raw, decoded, 1);
    for (size_t i = 0; i < 16; i++) assert(isfinite(decoded[i]));
    assert(tq_quantize_packed(ctx, x, packed, 1) == 0 && memcmp(packed, golden, sizeof(packed)) == 0);
    tq_destroy(ctx); empty();
    puts("Allocation failure leaves packed/raw outputs untouched, releases scratch and permits context reuse");
}

static void unsupported_prod(void) {
    for (int bits = 1; bits <= 8; bits += 3) {
        tq_ctx* ctx = tq_create_mse(8, bits, 7); assert(ctx);
        const size_t owned = tq_context_memory_bytes(ctx);
        uint8_t idx[16], qjl[16], before_idx[16], before_qjl[16];
        float x[8], gamma[8], output[8], before_gamma[8], before_output[8];
        input(x, 8); memset(idx, 0xA5, sizeof(idx)); memset(qjl, 0x5A, sizeof(qjl));
        for (size_t i = 0; i < 8; i++) gamma[i] = output[i] = 1234.5f;
        memcpy(before_idx, idx, sizeof(idx)); memcpy(before_qjl, qjl, sizeof(qjl));
        memcpy(before_gamma, gamma, sizeof(gamma)); memcpy(before_output, output, sizeof(output));
        measure(1);
        assert(tq_prod_idx_packed_size(ctx, 1) == 0 && tq_prod_qjl_packed_size(ctx, 1) == 0);
        assert(tq_quantize_prod(ctx, x, idx, qjl, gamma, 1) == -2);
        assert(tq_dequantize_prod(ctx, idx, qjl, gamma, output, 1) == -2);
        assert(tq_quantize_prod(ctx, NULL, NULL, NULL, NULL, 0) == -2);
        assert(tq_dequantize_prod(ctx, NULL, NULL, NULL, NULL, -1) == -2);
        assert(allocation_calls == 0 && live_bytes == owned);
        assert(memcmp(idx, before_idx, sizeof(idx)) == 0 && memcmp(qjl, before_qjl, sizeof(qjl)) == 0);
        assert(memcmp(gamma, before_gamma, sizeof(gamma)) == 0 && memcmp(output, before_output, sizeof(output)) == 0);
        assert(tq_context_memory_bytes(ctx) == owned);
        measure(0); tq_destroy(ctx); empty();
    }
    puts("MSE-only Prod APIs consistently reject unsupported mode without writes or lazy allocations");
}

static void invalid_arguments(void) {
    const int bad_dims[] = {INT_MIN, -1, 0, 3, 63, INT_MAX};
    measure(1);
    for (size_t i = 0; i < sizeof(bad_dims)/sizeof(*bad_dims); i++) {
        assert(!tq_create_mse(bad_dims[i], 3, 0)); assert(!tq_create(bad_dims[i], 3, 0));
    }
    for (int bits = -1; bits <= 9; bits++) if (bits < 1 || bits > 8) {
        assert(!tq_create_mse(8, bits, 0)); assert(!tq_create(8, bits, 0));
    }
    assert(allocation_calls == 0 && tq_context_memory_bytes(NULL) == 0);
    tq_destroy(NULL);
    measure(0); tq_ctx* ctx = tq_create_mse(8, 3, 0); assert(ctx);
    float x[8], out[8]; uint8_t packed[11]; uint16_t raw[8]; input(x, 8);
    measure(1);
    for (int n = -1; n <= 0; n++) {
        assert(tq_packed_size(ctx, n) == 0);
        assert(tq_quantize_packed(ctx, x, packed, n) == -1);
        assert(tq_dequantize_packed(ctx, packed, out, n) == -1);
        tq_quantize(ctx, x, raw, n); tq_dequantize(ctx, raw, out, n);
        tq_quantize_parallel(ctx, x, raw, n); tq_dequantize_parallel(ctx, raw, out, n);
    }
    assert(tq_quantize_packed(NULL, x, packed, 1) == -1);
    assert(tq_quantize_packed(ctx, NULL, packed, 1) == -1);
    assert(tq_dequantize_packed(ctx, NULL, out, 1) == -1);
    assert(tq_dequantize_packed(ctx, packed, NULL, 1) == -1);
    assert(tq_packed_size(NULL, 1) == 0 && isnan(tq_mse(NULL, x, 1)));
    assert(allocation_calls == 0);
    /* Huge valid dimensions must be size-safe before their first real allocation.
     * On 64-bit these are representable: failing allocation 1 prevents overcommit.
     */
    tq_destroy(ctx); empty(); measure(1);
    assert(!tq_create_mse(1 << 30, 8, 0)); assert(allocation_calls <= 1);
    empty(); measure(1); assert(!tq_create(1 << 30, 8, 0)); assert(allocation_calls <= 1);
    empty();
    puts("Invalid dimensions, bits, counts and nulls reject before allocation; large dimensions use checked sizes");
}

static void memory_report(void) {
    puts("{\"scope\":\"requested runtime heap bytes; excludes allocator overhead, caller buffers and thread stacks\",\"bits\":3,\"seed\":42,\"measurements\":[");
    size_t previous_mse = 0; int previous_dim = 0;
    const int dims[] = {64, 1024};
    for (size_t i = 0; i < sizeof(dims)/sizeof(*dims); i++) {
        const int d = dims[i]; size_t state[2], creation_peak[2], scratch_peak[2];
        for (int mode = 0; mode < 2; mode++) {
            empty();
            tq_ctx* ctx = mode ? tq_create(d, 3, 42) : tq_create_mse(d, 3, 42); assert(ctx);
            state[mode] = live_bytes; creation_peak[mode] = peak_bytes;
            assert(tq_context_memory_bytes(ctx) == state[mode]);
            float* x = malloc((size_t)d * sizeof(float));
            uint8_t* packed = malloc(tq_packed_size(ctx, 1)); assert(x && packed); input(x, (size_t)d);
            measure(0); assert(tq_quantize_packed(ctx, x, packed, 1) == 0);
            scratch_peak[mode] = peak_bytes - state[mode];
            assert(scratch_peak[mode] == (size_t)d * sizeof(float));
            assert(live_bytes == state[mode]);
            free(x); free(packed); tq_destroy(ctx); empty();
        }
        if (previous_dim) assert(state[0] - previous_mse == (size_t)(d - previous_dim) * sizeof(float));
        previous_dim = d; previous_mse = state[0];
        assert(state[1] > state[0] + (size_t)d * d * sizeof(float));
        printf("%s{\"dimension\":%d,\"mse_state_bytes\":%zu,\"legacy_state_bytes\":%zu,\"mse_creation_peak_bytes\":%zu,\"legacy_creation_peak_bytes\":%zu,\"mse_quantize_scratch_bytes\":%zu,\"legacy_quantize_scratch_bytes\":%zu}",
               i ? ",\n" : "", d, state[0], state[1], creation_peak[0], creation_peak[1], scratch_peak[0], scratch_peak[1]);
    }
    puts("\n]}");
}

/* Captured from the pre-M1.07 runtime, not generated from the implementation
 * under test. Packed index and QJL bytes lock PRNG/sign/projection order.
 * Float results allow libm/SIMD rounding differences across supported hosts.
 */
static void legacy_golden(void) {
    {
        const int d = 8, bits = 3, seed = -7;
        static const uint8_t expected_idx[] = {124, 47, 166, 152, 166, 219};
        static const uint8_t expected_qjl[] = {130, 84, 139};
        static const float expected_gamma[] = {0.3238377571f, 0.4582218528f, 0.6168184876f};
        static const float expected_out[] = {-0.5838211179f, -0.2786460519f, 0.631141603f, -0.5268909931f, 0.045181714f, 0.3878622651f, 0.6052160859f, -0.1491892338f, 0.006592323072f, 0.760337472f, -0.4404384494f, -0.3906547725f, 0.4536780417f, -0.3915202916f, -0.2858043015f, 0.1222518682f, -0.6989250779f, -0.4857313335f, 0.4323652685f, 1.203101277f, -0.2702774107f, -0.1238852516f, 0.5567637682f, -0.08611562848f};
        static const uint8_t expected_mse_indices[] = {170, 103, 83, 28, 11, 170, 92, 121, 174};
        tq_ctx* ctx = tq_create(d, bits, seed); assert(ctx);
        float x[48], gamma[3], reconstructed[48]; uint8_t idx[48], qjl[6], packed[72];
        input(x, (size_t)d * 3);
        assert(tq_prod_idx_packed_size(ctx, 3) == sizeof(expected_idx));
        assert(tq_prod_qjl_packed_size(ctx, 3) == sizeof(expected_qjl));
        assert(tq_quantize_prod(ctx, x, idx, qjl, gamma, 3) == 0);
        assert(memcmp(idx, expected_idx, sizeof(expected_idx)) == 0);
        assert(memcmp(qjl, expected_qjl, sizeof(expected_qjl)) == 0);
        for (int i = 0; i < 3; i++) near(gamma[i], expected_gamma[i]);
        assert(tq_dequantize_prod(ctx, idx, qjl, gamma, reconstructed, 3) == 0);
        for (int i = 0; i < d * 3; i++) near(reconstructed[i], expected_out[i]);
        assert(tq_quantize_packed(ctx, x, packed, 3) == 0);
        const size_t stride = tq_packed_size(ctx, 1), index_size = stride - 8;
        for (size_t v = 0; v < 3; v++) {
            assert(memcmp(packed + v * stride, "TQ01", 4) == 0);
            assert(memcmp(packed + v * stride + 8, expected_mse_indices + v * index_size, index_size) == 0);
        }
        tq_destroy(ctx); empty();
    }
    {
        const int d = 16, bits = 2, seed = 42;
        static const uint8_t expected_idx[] = {116, 62, 92, 191, 22, 133};
        static const uint8_t expected_qjl[] = {153, 38, 168, 52, 37, 12};
        static const float expected_gamma[] = {1.085022092f, 1.152839065f, 1.086632013f};
        static const float expected_out[] = {-0.3853659332f, 0.3058390915f, 0.746915102f, -0.1690020263f, -0.3572300673f, 0.06928629428f, 0.9315373302f, -0.1831406504f, -0.03460629284f, 0.3276691139f, -0.6645261049f, 0.1125877276f, 0.3961111009f, -0.2564001083f, -0.4577132165f, 0.2635681927f, -0.6534779668f, 0.2957237065f, -0.1881045997f, 0.1688407362f, -0.4989091456f, 0.01292496547f, 0.8859873414f, -0.6297335625f, -0.01432307065f, 0.259924531f, -1.014041901f, -0.6116588712f, 0.288585335f, 0.9873666763f, -0.6717166901f, 0.07053249329f, 0.04787426069f, -0.1666520238f, -0.2791155577f, -0.08610831946f, -1.15226841f, -0.265339911f, 0.3195162117f, -1.214789391f, -0.2023928463f, 0.3208204508f, 0.8506533504f, -0.3386505544f, -0.03680384532f, 0.6005102992f, -0.7515027523f, -0.4319317341f};
        static const uint8_t expected_mse_indices[] = {117, 107, 237, 30, 245, 103, 171, 138, 105, 7, 102, 129};
        tq_ctx* ctx = tq_create(d, bits, seed); assert(ctx);
        float x[48], gamma[3], reconstructed[48]; uint8_t idx[48], qjl[6], packed[72];
        input(x, (size_t)d * 3);
        assert(tq_prod_idx_packed_size(ctx, 3) == sizeof(expected_idx));
        assert(tq_prod_qjl_packed_size(ctx, 3) == sizeof(expected_qjl));
        assert(tq_quantize_prod(ctx, x, idx, qjl, gamma, 3) == 0);
        assert(memcmp(idx, expected_idx, sizeof(expected_idx)) == 0);
        assert(memcmp(qjl, expected_qjl, sizeof(expected_qjl)) == 0);
        for (int i = 0; i < 3; i++) near(gamma[i], expected_gamma[i]);
        assert(tq_dequantize_prod(ctx, idx, qjl, gamma, reconstructed, 3) == 0);
        for (int i = 0; i < d * 3; i++) near(reconstructed[i], expected_out[i]);
        assert(tq_quantize_packed(ctx, x, packed, 3) == 0);
        const size_t stride = tq_packed_size(ctx, 1), index_size = stride - 8;
        for (size_t v = 0; v < 3; v++) {
            assert(memcmp(packed + v * stride, "TQ01", 4) == 0);
            assert(memcmp(packed + v * stride + 8, expected_mse_indices + v * index_size, index_size) == 0);
        }
        tq_destroy(ctx); empty();
    }
    {
        const int d = 8, bits = 8, seed = 0;
        static const uint8_t expected_idx[] = {55, 218, 50, 235, 229, 241, 154, 207, 35, 14, 152, 122, 166, 126, 174, 227, 119, 71, 125, 62, 137};
        static const uint8_t expected_qjl[] = {172, 84, 228};
        static const float expected_gamma[] = {0.0156607125f, 0.02217963897f, 0.02690288983f};
        static const float expected_out[] = {-0.4990717173f, -0.06483114511f, 0.3795906603f, -0.6193136573f, -0.2052131742f, 0.2533928156f, 0.6939532757f, -0.3078756332f, 0.1270631999f, 0.5643498898f, -0.4293997586f, -0.0006695790798f, 0.4514296055f, -0.5599195957f, -0.1272279322f, 0.3154487908f, -0.6659225225f, -0.2433704287f, 0.1992036402f, 0.6400213838f, -0.3875515461f, 0.04746514186f, 0.5076631308f, -0.4674710929f};
        static const uint8_t expected_mse_indices[] = {114, 109, 146, 168, 176, 122, 122, 149, 156, 141, 114, 129, 87, 156, 87, 126, 101, 139, 175, 121, 159, 152, 152, 134};
        tq_ctx* ctx = tq_create(d, bits, seed); assert(ctx);
        float x[48], gamma[3], reconstructed[48]; uint8_t idx[48], qjl[6], packed[72];
        input(x, (size_t)d * 3);
        assert(tq_prod_idx_packed_size(ctx, 3) == sizeof(expected_idx));
        assert(tq_prod_qjl_packed_size(ctx, 3) == sizeof(expected_qjl));
        assert(tq_quantize_prod(ctx, x, idx, qjl, gamma, 3) == 0);
        assert(memcmp(idx, expected_idx, sizeof(expected_idx)) == 0);
        assert(memcmp(qjl, expected_qjl, sizeof(expected_qjl)) == 0);
        for (int i = 0; i < 3; i++) near(gamma[i], expected_gamma[i]);
        assert(tq_dequantize_prod(ctx, idx, qjl, gamma, reconstructed, 3) == 0);
        for (int i = 0; i < d * 3; i++) near(reconstructed[i], expected_out[i]);
        assert(tq_quantize_packed(ctx, x, packed, 3) == 0);
        const size_t stride = tq_packed_size(ctx, 1), index_size = stride - 8;
        for (size_t v = 0; v < 3; v++) {
            assert(memcmp(packed + v * stride, "TQ01", 4) == 0);
            assert(memcmp(packed + v * stride + 8, expected_mse_indices + v * index_size, index_size) == 0);
        }
        tq_destroy(ctx); empty();
    }
    puts("Pre-change Prod and MSE goldens preserve PRNG, packed bytes and decoded values");
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "all";
    if (!strcmp(mode, "memory")) { memory_report(); return 0; }
    const int all = !strcmp(mode, "all");
    if (all || !strcmp(mode, "equivalence")) equivalence();
    if (all || !strcmp(mode, "parallel")) parallel_equivalence();
    if (all || !strcmp(mode, "creation_failures")) creation_failures();
    if (all || !strcmp(mode, "operation_failures")) operation_failures();
    if (all || !strcmp(mode, "unsupported_prod")) unsupported_prod();
    if (all || !strcmp(mode, "invalid")) invalid_arguments();
    if (all || !strcmp(mode, "golden")) legacy_golden();
    empty();
    return 0;
}
