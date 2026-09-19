"""Paged TQ02 attention: independent oracle, hostile ABI inputs and no heap."""
import ctypes
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime.build_runtime import build_runtime
from test_tq_portable_format_regressions import codebook, f32, reference_decode, reference_pack


def attention_reference(query, keys, values, past, sequence, heads, kv_heads, dim):
    result = []
    for position in range(sequence):
        for head in range(heads):
            kv_head = head // (heads // kv_heads)
            q = query[(position * heads + head) * dim:(position * heads + head + 1) * dim]
            scores = []
            for token in range(past + position + 1):
                key = keys[token * kv_heads + kv_head]
                scores.append(sum(a * b for a, b in zip(q, key)) / math.sqrt(dim))
            maximum = max(scores)
            weights = [f32(math.exp(score - maximum)) for score in scores]
            denominator = sum(weights)
            for lane in range(dim):
                value = sum(weight * values[token * kv_heads + kv_head][lane]
                            for token, weight in enumerate(weights)) / denominator
                result.append(f32(value))
    return result


class TQKVOracleRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="nexa-tq-kv-oracle-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.lib = ctypes.CDLL(str(build_runtime("nexa_tq_attention", cls.temporary.name)))
        fp, bp, dp = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_double)
        sz, table = ctypes.c_size_t, ctypes.POINTER(bp)
        cls.lib.tq_create_mse_from_codebook.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, fp, sz]
        cls.lib.tq_create_mse_from_codebook.restype = ctypes.c_void_p
        cls.lib.tq_destroy.argtypes = [ctypes.c_void_p]
        cls.lib.nexa_causal_gqa_attention_paged_tq.argtypes = [ctypes.c_void_p, fp, sz, table, sz, table, sz,
            sz, sz, sz, sz, sz, sz, sz, fp, sz, fp, sz, dp, sz, fp, sz]
        cls.lib.nexa_causal_gqa_attention_paged_tq.restype = ctypes.c_int

    def exercise(self, dims, widths, layouts):
        bp = ctypes.POINTER(ctypes.c_uint8)
        for dim in dims:
            for bits in widths:
                centroids, _ = codebook(bits)
                book = (ctypes.c_float * len(centroids))(*centroids)
                seed = -2147483648 if bits % 2 else 42
                ctx = self.lib.tq_create_mse_from_codebook(dim, bits, seed, book, len(book))
                self.assertTrue(ctx)
                try:
                    for heads, kv_heads, page_tokens in layouts:
                        tokens = 5
                        query = [f32(((i * 13 + 5) % 37 - 18) / 9) for i in range(tokens * heads * dim)]
                        source = [[f32(((row * 7 + lane * 11 + offset) % 31 - 15) / 5)
                                   if row else 0.0 for lane in range(dim)]
                                  for offset in (0, 5) for row in range(tokens * kv_heads)]
                        records = [reference_pack(row, bits, seed, centroids) for row in source]
                        decoded = [reference_decode(row, dim, bits, seed, centroids) for row in records]
                        row_bytes = len(records[0])
                        page_bytes = page_tokens * kv_heads * row_bytes
                        owners, tables = [], []
                        for bank in range(2):
                            bank_owners, pointers = [], []
                            for start in range(0, tokens, page_tokens):
                                owner = (ctypes.c_uint8 * (page_bytes + 2))(*([203] * (page_bytes + 2)))
                                data = b"".join(records[bank * tokens * kv_heads + start * kv_heads:
                                                        bank * tokens * kv_heads + min(tokens, start + page_tokens) * kv_heads])
                                ctypes.memmove(ctypes.byref(owner, 1), data, len(data))
                                bank_owners.append(owner)
                                pointers.append(ctypes.cast(ctypes.byref(owner, 1), bp))
                            # Invalid unused entry must never be dereferenced.
                            pointers.append(ctypes.cast(ctypes.c_void_p(9), bp))
                            tables.append((bp * len(pointers))(*pointers))
                            owners.extend(bank_owners)
                        original = [bytes(owner) for owner in owners]
                        for past, count in ((0, 5), (0, 2), (2, 2), (4, 1)):
                            q = query[past * heads * dim:(past + count) * heads * dim]
                            expected = attention_reference(q, decoded[:tokens * kv_heads], decoded[tokens * kv_heads:],
                                                           past, count, heads, kv_heads, dim)
                            query_buffer = (ctypes.c_float * len(q))(*q)
                            scores = (ctypes.c_float * (past + count + 1))(*([77.0] * (past + count + 1)))
                            vector = (ctypes.c_float * (dim + 1))(*([77.0] * (dim + 1)))
                            accumulator = (ctypes.c_double * (dim + 1))(*([77.0] * (dim + 1)))
                            output = (ctypes.c_float * (len(q) + 1))(*([77.0] * (len(q) + 1)))
                            status = self.lib.nexa_causal_gqa_attention_paged_tq(ctx,
                                query_buffer, len(q), tables[0], len(tables[0]), tables[1], len(tables[1]),
                                page_tokens, page_bytes, past, count, heads, kv_heads, dim,
                                scores, len(scores), vector, len(vector), accumulator, len(accumulator), output, len(output))
                            self.assertEqual(status, 0, (dim, bits, heads, kv_heads, page_tokens, past, count))
                            for actual, want in zip(output, expected):
                                self.assertTrue(math.isclose(actual, want, abs_tol=3e-6, rel_tol=3e-6),
                                                (dim, bits, actual, want))
                            self.assertEqual((scores[-1], vector[-1], accumulator[-1], output[-1]), (77.0,) * 4)
                            self.assertEqual([bytes(owner) for owner in owners], original)
                finally:
                    self.lib.tq_destroy(ctx)

    def test_all_bits_tiny_heads_padding_and_partial_pages(self):
        self.exercise((1, 2, 4), range(1, 9), ((1, 1, 1), (4, 1, 2)))

    def test_gqa_mqa_mha_prefill_chunks_and_decode(self):
        self.exercise((8, 64), (1, 3, 8), ((4, 1, 4), (4, 2, 2), (4, 4, 3)))


HARNESS = r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
static size_t allocations;
static void *tracked_malloc(size_t n) { allocations++; return malloc(n); }
static void *tracked_calloc(size_t n, size_t width) { allocations++; return calloc(n, width); }
#define malloc tracked_malloc
#define calloc tracked_calloc
#include "turboquant.c"
#include "nexapack/tq_attention.c"
#undef malloc
#undef calloc

typedef struct {
    tq_ctx *ctx;
    const float *q;
    const uint8_t *const *kt, *const *vt;
    size_t qc, kc, vc, pt, pb, past, seq, heads, kvheads, dim;
    float *scores, *vector, *out;
    double *acc;
    size_t sc, vsc, ac, oc;
} args;

static int run(args a) {
    size_t before = allocations;
    int status = nexa_causal_gqa_attention_paged_tq(a.ctx, a.q, a.qc, a.kt, a.kc, a.vt, a.vc,
        a.pt, a.pb, a.past, a.seq, a.heads, a.kvheads, a.dim,
        a.scores, a.sc, a.vector, a.vsc, a.acc, a.ac, a.out, a.oc);
    assert(allocations == before);
    return status;
}

static tq_ctx *custom(int dim, int bits) {
    float book[256];
    for (int i = 0; i < (1 << bits); i++) book[i] = (float)(2 * i + 1 - (1 << bits)) / (1 << bits);
    tq_ctx *ctx = tq_create_mse_from_codebook(dim, bits, -42, book, (size_t)1 << bits);
    assert(ctx);
    return ctx;
}

static void errors(int mode) {
    tq_ctx *ctx = custom(2, 3);
    uint8_t k[36], v[36], records[27];
    memset(k, 203, sizeof k); memset(v, 203, sizeof v);
    float input[6] = {1,-2,3,-4,5,-6}, quant[2];
    assert(tq_quantize_tq02(ctx,input,6,records,27,3,quant,2)==0);
    memcpy(k,records,27); memcpy(v,records,27);
    const uint8_t *kt[3] = {k,k+18,NULL}, *vt[3] = {v,v+18,NULL};
    float query[8] = {1,-2,3,-4,5,-6,7,-8}, scores[3], vector[2], output[8];
    double acc[2];
    args base = {ctx,query,kt,vt,8,3,3,2,18,1,2,2,1,2,scores,vector,output,acc,3,2,2,8};
    assert(run(base)==0);
    args a;
#define BAD(field,value,expected) do { a=base; a.field=value; assert(run(a)==expected); } while(0)
    if (mode == 0) {
        BAD(qc,7,-2); BAD(kc,1,-2); BAD(vc,1,-2); BAD(pb,17,-2);
        BAD(sc,2,-2); BAD(vsc,1,-2); BAD(ac,1,-2); BAD(oc,7,-2);
        BAD(ctx,NULL,-1); BAD(q,NULL,-1); BAD(kt,NULL,-1); BAD(vt,NULL,-1);
        BAD(scores,NULL,-1); BAD(vector,NULL,-1); BAD(acc,NULL,-1); BAD(out,NULL,-1);
        BAD(pt,0,-1); BAD(seq,0,-1); BAD(heads,0,-1); BAD(kvheads,0,-1); BAD(dim,0,-1);
        BAD(dim,4,-1); BAD(kvheads,3,-1);
        BAD(past,SIZE_MAX,-3); BAD(pt,SIZE_MAX,-3); BAD(heads,SIZE_MAX,-3);
        assert(tq_context_dim(NULL)==0 && tq_context_bits(NULL)==0);
        assert(tq_context_dim(ctx)==2 && tq_context_bits(ctx)==3);
        assert(tq_context_memory_bytes(ctx)<=128+4*2+4*(2*8-1));
        assert(!tq_context_buffer_disjoint(NULL,k,sizeof k));
        assert(!tq_context_buffer_disjoint(ctx,NULL,sizeof k));
        assert(!tq_context_buffer_disjoint(ctx,(void*)(UINTPTR_MAX-1),8));
        kt[1]=NULL; assert(run(base)==-1); kt[1]=k+18;
        vt[1]=NULL; assert(run(base)==-1); vt[1]=v+18;
    } else if (mode == 1) {
        BAD(out,(float*)query,-1); BAD(scores,(float*)query,-1); BAD(vector,(float*)query,-1);
        BAD(acc,(double*)query,-1); BAD(scores,output,-1); BAD(vector,output,-1);
        BAD(acc,(double*)output,-1); BAD(scores,vector,-1); BAD(acc,(double*)vector,-1);
        BAD(vector,(float*)acc,-1);
        BAD(out,(float*)k,-1); BAD(scores,(float*)k,-1); BAD(vector,(float*)k,-1); BAD(acc,(double*)k,-1);
        BAD(out,(float*)v,-1); BAD(scores,(float*)v,-1); BAD(vector,(float*)v,-1); BAD(acc,(double*)v,-1);
        BAD(out,(float*)kt,-1); BAD(scores,(float*)kt,-1); BAD(vector,(float*)kt,-1); BAD(acc,(double*)kt,-1);
        BAD(out,(float*)vt,-1); BAD(scores,(float*)vt,-1); BAD(vector,(float*)vt,-1); BAD(acc,(double*)vt,-1);
        BAD(out,ctx->signs,-1); BAD(scores,ctx->centroids,-1); BAD(vector,ctx->boundaries,-1);
        BAD(acc,(double*)ctx,-1); BAD(q,ctx->signs,-1); BAD(kt,(const uint8_t*const*)ctx,-1);
        BAD(q,(float*)(uintptr_t)(UINTPTR_MAX-3),-1);
        BAD(out,(float*)(uintptr_t)(UINTPTR_MAX-3),-1);
        BAD(kt,(const uint8_t*const*)(uintptr_t)(UINTPTR_MAX-7),-1);
        BAD(q,(float*)((uint8_t*)query+1),-1);
        BAD(acc,(double*)((uint8_t*)acc+1),-1);
        kt[1]=(uint8_t*)(uintptr_t)(UINTPTR_MAX-2); assert(run(base)==-1); kt[1]=k+18;
        vt[1]=(uint8_t*)ctx->centroids; assert(run(base)==-1); vt[1]=v+18;
    } else {
        for (int i=0;i<8;i++) output[i]=123.0f;
        /* Corruption in the last visible V must fail before any output write. */
        v[21]='1'; assert(run(base)==-4);
        const uint32_t corrupt[] = {0x80000000u,0xbf800000u,0x7f800000u,0x7fc00000u};
        for (size_t i=0;i<sizeof corrupt/sizeof *corrupt;i++) {
            memcpy(v,records,27);
            for (unsigned b=0;b<4;b++) v[22+b]=(uint8_t)(corrupt[i]>>(8*b));
            assert(run(base)==-4);
            for (int j=0;j<8;j++) assert(output[j]==123.0f);
        }
        memcpy(v,records,27); v[26]|=0x80; assert(run(base)==-4); /* high index padding */
        memcpy(v,records,27); memset(v+22,0,4); v[26]=1; assert(run(base)==-4); /* zero norm indices */
        v[26]=0; assert(run(base)==0); /* canonical zero */
        memcpy(v,records,27); query[0]=NAN; assert(run(base)==-4);
        query[0]=INFINITY; assert(run(base)==-4);
    }
#undef BAD
    tq_destroy(ctx);
}

static void noheap(void) {
    for (int dim=1;dim<=64;dim*=2) for (int bits=1;bits<=8;bits++) {
        tq_ctx *ctx=custom(dim,bits);
        float input[3*64], q[2*4*64], quant[64], scores[3], vector[64], output[2*4*64];
        double acc[64]; uint8_t packed[3*72];
        for (int i=0;i<3*dim;i++) input[i]=(float)(i%13-6);
        for (int i=0;i<2*4*dim;i++) q[i]=(float)(i%19-9);
        size_t row=tq_packed_size(ctx,1);
        assert(tq_quantize_tq02(ctx,input,3*(size_t)dim,packed,3*row,3,quant,(size_t)dim)==0);
        const uint8_t *pages[]={packed,packed+2*row};
        args a={ctx,q,pages,pages,8*(size_t)dim,2,2,2,2*row,1,2,4,1,(size_t)dim,
                scores,vector,output,acc,3,(size_t)dim,(size_t)dim,8*(size_t)dim};
        assert(run(a)==0);
        tq_destroy(ctx);
    }
}

static void numeric(void) {
    tq_ctx *ctx=custom(1,1);
    uint8_t packed[18]={'T','Q','0','2',0xff,0xff,0x7f,0x7f,1,
                       'T','Q','0','2',0xff,0xff,0x7f,0x7f,1};
    const uint8_t *pages[]={packed};
    float query[]={FLT_MAX}, scores[2], vector[1], output[1]; double acc[1];
    args a={ctx,query,pages,pages,1,1,1,2,18,1,1,1,1,1,scores,vector,output,acc,2,1,1,1};
    /* Double score is finite but far beyond FLT_MAX; shifted softmax works. */
    assert(run(a)==0 && isfinite(output[0]));
    tq_destroy(ctx);
    float large[]={2.0f,3.0f};
    ctx=tq_create_mse_from_codebook(1,1,0,large,2); assert(ctx); a.ctx=ctx;
    assert(run(a)==-5); /* Decode itself would overflow a float lane. */
    tq_destroy(ctx);
}

int main(int argc,char **argv) {
    assert(argc==2);
    if (!strcmp(argv[1],"capacity")) errors(0);
    else if (!strcmp(argv[1],"aliases")) errors(1);
    else if (!strcmp(argv[1],"corruption")) errors(2);
    else if (!strcmp(argv[1],"noheap")) noheap();
    else if (!strcmp(argv[1],"numeric")) numeric();
    else return 2;
    puts("ok"); return 0;
}
'''


class TQKVKernelSafetyRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("clang") or shutil.which("cc")
        if not compiler:
            raise unittest.SkipTest("C compiler unavailable")
        cls.directory = tempfile.TemporaryDirectory(prefix="nexa-tq-kv-safety-")
        cls.addClassCleanup(cls.directory.cleanup)
        source = Path(cls.directory.name) / "safety.c"
        source.write_text(HARNESS)
        cls.binary = Path(cls.directory.name) / ("safety.exe" if os.name == "nt" else "safety")
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

    def test_capacities_dimensions_overflow_and_context_reserve(self):
        self.run_mode("capacity")

    def test_aliases_context_storage_alignment_and_wrapped_addresses(self):
        self.run_mode("aliases")

    def test_visible_corruption_norm_padding_nan_and_zero_records(self):
        self.run_mode("corruption")

    def test_no_heap_across_all_bit_widths_and_head_sizes(self):
        self.run_mode("noheap")

    def test_large_double_scores_and_reconstruction_float_overflow(self):
        self.run_mode("numeric")


if __name__ == "__main__":
    unittest.main()
