"""TQ02 native ABI: legacy equivalence, strict bytes, ownership and no heap."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

HARNESS = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
static size_t allocations, live, live_bytes, fail_at;
static void *owners[128];
static size_t sizes[128];
static void *track_malloc(size_t size) {
    allocations++;
    if (fail_at && allocations == fail_at) return NULL;
    void *p = malloc(size);
    if (p) {
        size_t i = 0;
        while (i < 128 && owners[i]) i++;
        assert(i < 128);
        owners[i] = p; sizes[i] = size; live++; live_bytes += size;
    }
    return p;
}
static void *track_calloc(size_t n, size_t size) {
    assert(!n || size <= SIZE_MAX / n);
    void *p = track_malloc(n * size);
    if (p) memset(p, 0, n * size);
    return p;
}
static void track_free(void *p) {
    if (!p) return;
    size_t i = 0;
    while (i < 128 && owners[i] != p) i++;
    assert(i < 128);
    live--; live_bytes -= sizes[i]; owners[i] = NULL;
    free(p);
}
#define malloc track_malloc
#define calloc track_calloc
#define free track_free
#include "turboquant.c"
#undef malloc
#undef calloc
#undef free

static tq_ctx *custom(int d, int bits, int seed) {
    float book[256];
    for (int i = 0; i < (1 << bits); i++) book[i] = (float)(2 * i + 1 - (1 << bits)) / (float)(1 << bits);
    tq_ctx *ctx = tq_create_mse_from_codebook(d, bits, seed, book, (size_t)1 << bits);
    assert(ctx);
    return ctx;
}

static void equivalence(void) {
    const int dims[] = {1, 8, 64}, widths[] = {1, 3, 8};
    for (size_t a = 0; a < 3; a++) for (size_t b = 0; b < 3; b++) {
        int d = dims[a], bits = widths[b];
        tq_ctx *ctx = tq_create_mse(d, bits, -42);
        assert(ctx);
        float book[256], input[192], scratch[64], legacy_out[192], portable_out[192];
        uint8_t legacy[216], portable[216];
        assert(tq_export_mse_codebook(ctx, book, (size_t)1 << bits) == 0);
        tq_ctx *restored = tq_create_mse_from_codebook(d, bits, -42, book, (size_t)1 << bits);
        assert(restored);
        for (int i = 0; i < 3 * d; i++) input[i] = i < d ? 0.0f : (float)((i * 17) % 37 - 18) / 7.0f;
        size_t bytes = tq_packed_size(ctx, 3), row = tq_packed_size(ctx, 1);
        assert(bytes <= sizeof legacy);
        assert(tq_quantize_packed(ctx, input, legacy, 3) == 0);
        assert(tq_quantize_tq02(restored, input, 3u * (size_t)d, portable, bytes, 3, scratch, (size_t)d) == 0);
        for (int v = 0; v < 3; v++) {
            uint32_t norm_bits;
            memcpy(&norm_bits, legacy + (size_t)v * row + 4, 4);
            legacy[(size_t)v * row + 3] = '2';
            for (unsigned k = 0; k < 4; k++) legacy[(size_t)v * row + 4 + k] = (uint8_t)(norm_bits >> (8 * k));
        }
        assert(memcmp(legacy, portable, bytes) == 0);
        assert(tq_dequantize_tq02(ctx, portable, bytes, legacy_out, 3u * (size_t)d, 3) == 0);
        assert(tq_dequantize_tq02(restored, portable, bytes, portable_out, 3u * (size_t)d, 3) == 0);
        assert(memcmp(legacy_out, portable_out, 3u * (size_t)d * sizeof(float)) == 0);
        tq_destroy(restored); tq_destroy(ctx);
        assert(live == 0);
    }
}

static void imported(void) {
    float book[2] = {-0.25f, 0.75f}, exported[2];
    size_t before = allocations;
    size_t estimate = tq_mse_context_memory_size(32, 1);
    assert(estimate > 128 && allocations == before);
    tq_ctx *ctx = tq_create_mse_from_codebook(32, 1, INT_MIN, book, 2);
    assert(ctx && allocations == before + 4); /* no Lloyd-Max temporary */
    assert(live_bytes == estimate && tq_context_memory_bytes(ctx) == estimate);
    book[0] = 13.0f;
    assert(tq_export_mse_codebook(ctx, exported, 2) == 0);
    assert(exported[0] == -0.25f && exported[1] == 0.75f);
    assert(tq_export_mse_codebook(ctx, exported, 1) == -2);
    assert(tq_export_mse_codebook(ctx, ctx->centroids, 2) == -3);
    assert(tq_export_mse_codebook(NULL, exported, 2) == -1);
    assert(tq_export_mse_codebook(ctx, NULL, 2) == -1);
    tq_destroy(ctx);
    before = allocations;
    assert(!tq_mse_context_memory_size(3, 3));
    assert(!tq_mse_context_memory_size(8, 0));
    assert(!tq_mse_context_memory_size(8, 9));
    assert(!tq_create_mse_from_codebook(4, 1, 0, book, 1));
    assert(!tq_create_mse_from_codebook(4, 1, 0, book, 2));
    book[0] = -INFINITY;
    assert(!tq_create_mse_from_codebook(4, 1, 0, book, 2));
    book[0] = NAN;
    assert(!tq_create_mse_from_codebook(4, 1, 0, book, 2));
    book[0] = FLT_MAX / 2; book[1] = FLT_MAX;
    assert(!tq_create_mse_from_codebook(4, 1, 0, book, 2));
    assert(allocations == before && live == 0);
    book[0] = -1; book[1] = 1;
    for (size_t i = 1; i <= 4; i++) {
        fail_at = allocations + i;
        assert(!tq_create_mse_from_codebook(4, 1, 0, book, 2));
        assert(live == 0 && live_bytes == 0);
    }
    fail_at = 0;
}

static void noheap(void) {
    for (int bits = 1; bits <= 8; bits++) for (int d = 1; d <= 64; d *= 2) {
        tq_ctx *ctx = custom(d, bits, -17);
        float input[64], scratch[64], output[64];
        uint8_t packed[72];
        for (int i = 0; i < d; i++) input[i] = (float)(i - d / 2) / 7;
        size_t bytes = tq_packed_size(ctx, 1), before = allocations;
        fail_at = before + 1;
        assert(tq_quantize_tq02(ctx, input, (size_t)d, packed, bytes, 1, scratch, (size_t)d) == 0);
        assert(tq_dequantize_tq02(ctx, packed, bytes, output, (size_t)d, 1) == 0);
        assert(allocations == before);
        unsigned tail = ((unsigned)d * (unsigned)bits) % 8;
        assert(!tail || !(packed[bytes - 1] >> tail));
        for (int i = 0; i < d; i++) assert(isfinite(output[i]));
        fail_at = 0;
        tq_destroy(ctx);
    }
    assert(live == 0);
}

static void errors(void) {
    tq_ctx *ctx = custom(2, 3, 42);
    float input[2] = {1, -1}, scratch[2], output[2] = {91, 92};
    uint8_t packed[18], corrupted[18];
    assert(tq_quantize_tq02(ctx, input, 2, packed, 9, 1, scratch, 2) == 0);
    assert(tq_quantize_tq02(ctx, input, 1, packed, 9, 1, scratch, 2) == -2);
    assert(tq_quantize_tq02(ctx, input, 2, packed, 8, 1, scratch, 2) == -2);
    assert(tq_quantize_tq02(ctx, input, 2, packed, 10, 1, scratch, 2) == -2);
    assert(tq_quantize_tq02(ctx, input, 2, packed, 9, 1, scratch, 1) == -2);
    assert(tq_quantize_tq02(ctx, input, SIZE_MAX, packed, SIZE_MAX, INT_MAX, scratch, 2) == -2);
    assert(tq_quantize_tq02(ctx, input, 2, packed, 9, 0, scratch, 2) == -1);
    assert(tq_quantize_tq02(NULL, input, 2, packed, 9, 1, scratch, 2) == -1);
    assert(tq_quantize_tq02(ctx, NULL, 2, packed, 9, 1, scratch, 2) == -1);
    assert(tq_quantize_tq02(ctx, input, 2, (uint8_t*)input, 9, 1, scratch, 2) == -3);
    assert(tq_quantize_tq02(ctx, input, 2, packed, 9, 1, input, 2) == -3);
    assert(tq_quantize_tq02(ctx, input, 2, packed, 9, 1, (float*)packed, 2) == -3);
    assert(tq_quantize_tq02(ctx, input, 2, packed, 9, 1, ctx->signs, 2) == -3);
    assert(tq_quantize_tq02(ctx, (float*)(UINTPTR_MAX - 1u), 2, packed, 9, 1, scratch, 2) == -3);
    assert(tq_dequantize_tq02(ctx, packed, 9, (float*)packed, 2, 1) == -3);
    assert(tq_dequantize_tq02(ctx, packed, 9, ctx->signs, 2, 1) == -3);
    assert(tq_dequantize_tq02(ctx, packed, 9, output, 1, 1) == -2);
    assert(tq_dequantize_tq02(ctx, packed, 8, output, 2, 1) == -2);
    assert(tq_dequantize_tq02(ctx, packed, 10, output, 2, 1) == -2);
    const uint32_t invalid_norms[] = {0x80000000u, 0xbf800000u, 0x7f800000u, 0x7fc00000u};
    for (size_t i = 0; i < 4; i++) {
        memcpy(corrupted, packed, 9);
        tq_write_le32(corrupted + 4, invalid_norms[i]);
        assert(tq_dequantize_tq02(ctx, corrupted, 9, output, 2, 1) == -1);
        assert(output[0] == 91 && output[1] == 92);
    }
    memcpy(corrupted, packed, 9); corrupted[3] = '1';
    assert(tq_dequantize_tq02(ctx, corrupted, 9, output, 2, 1) == -1);
    memcpy(corrupted, packed, 9); corrupted[8] |= 0x80;
    assert(tq_dequantize_tq02(ctx, corrupted, 9, output, 2, 1) == -1);
    memcpy(corrupted, packed, 9); memset(corrupted + 4, 0, 4); corrupted[8] = 1;
    assert(tq_dequantize_tq02(ctx, corrupted, 9, output, 2, 1) == -1);
    /* Bad second row is rejected before writes to first row output. */
    memcpy(corrupted, packed, 9); memcpy(corrupted + 9, packed, 9); corrupted[12] = '1';
    float batch_out[4] = {91, 92, 93, 94};
    assert(tq_dequantize_tq02(ctx, corrupted, 18, batch_out, 4, 2) == -1);
    assert(batch_out[0] == 91 && batch_out[3] == 94);
    input[0] = INFINITY;
    assert(tq_quantize_tq02(ctx, input, 2, packed, 9, 1, scratch, 2) == -4);
    input[0] = NAN;
    assert(tq_quantize_tq02(ctx, input, 2, packed, 9, 1, scratch, 2) == -4);
    input[0] = input[1] = FLT_MAX;
    assert(tq_quantize_tq02(ctx, input, 2, packed, 9, 1, scratch, 2) == -4);
    input[0] = -0.0f; input[1] = 0.0f;
    assert(tq_quantize_tq02(ctx, input, 2, packed, 9, 1, scratch, 2) == 0);
    for (size_t i = 4; i < 9; i++) assert(packed[i] == 0);
    assert(tq_dequantize_tq02(ctx, packed, 9, output, 2, 1) == 0);
    assert(output[0] == 0 && !signbit(output[0]));
    tq_destroy(ctx);
    float extreme[2] = {-FLT_MAX, FLT_MAX};
    ctx = tq_create_mse_from_codebook(1, 1, 42, extreme, 2);
    assert(ctx);
    memcpy(packed, "TQ02", 4); tq_write_le32(packed + 4, 0x40000000u); packed[8] = 1;
    assert(tq_dequantize_tq02(ctx, packed, 9, output, 1, 1) == -4);
    tq_destroy(ctx);
    assert(live == 0);
}

int main(int argc, char** argv) {
    assert(argc == 2);
    if (!strcmp(argv[1], "equivalence")) equivalence();
    else if (!strcmp(argv[1], "imported")) imported();
    else if (!strcmp(argv[1], "noheap")) noheap();
    else if (!strcmp(argv[1], "errors")) errors();
    else return 2;
    assert(live == 0 && live_bytes == 0);
    puts("ok");
    return 0;
}
'''


class TQPortableKernelRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("clang") or shutil.which("cc")
        if not compiler:
            raise unittest.SkipTest("C compiler unavailable")
        cls.directory = tempfile.TemporaryDirectory(prefix="nexa-tq-portable-")
        cls.addClassCleanup(cls.directory.cleanup)
        source = Path(cls.directory.name) / "portable.c"
        source.write_text(HARNESS)
        cls.binary = Path(cls.directory.name) / ("portable.exe" if os.name == "nt" else "portable")
        command = [compiler, "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                   "-I", str(ROOT / "runtime"), str(source), "-o", str(cls.binary)]
        if os.name != "nt":
            command += ["-pthread", "-lm", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.environment = dict(os.environ)
        cls.environment["ASAN_OPTIONS"] = "detect_leaks=0" if sys.platform == "darwin" else "detect_leaks=1"
        cls.environment["UBSAN_OPTIONS"] = "halt_on_error=1"

    def run_mode(self, mode):
        result = subprocess.run([str(self.binary), mode], env=self.environment,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_tq02_preserves_legacy_indices_norms_and_imported_context_results(self):
        self.run_mode("equivalence")

    def test_codebook_import_preflight_ownership_failures_and_no_lloyd_max(self):
        self.run_mode("imported")

    def test_encode_decode_have_no_heap_calls_across_dimensions_and_bits(self):
        self.run_mode("noheap")

    def test_capacities_alias_overflow_padding_norm_and_arithmetic_validation(self):
        self.run_mode("errors")


if __name__ == "__main__":
    unittest.main()
