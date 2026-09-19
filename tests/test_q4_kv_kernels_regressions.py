"""Direct packed-Q4 KV attention without a decoded cache or head-sized vector."""
import ctypes
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime/nexapack"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_q4_runtime_regressions import reference_pack
from test_transformer_kernels_regressions import attention_oracle, equal, f32, floats


def bind(library):
    lib = ctypes.CDLL(str(library))
    fp, bp, sz = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
    table = ctypes.POINTER(bp)
    lib.nexa_q4_quantize.argtypes = [fp, sz, sz, sz, sz, bp, sz]
    lib.nexa_q4_quantize.restype = ctypes.c_int
    lib.nexa_q4_row_size.argtypes = [sz, sz]
    lib.nexa_q4_row_size.restype = sz
    lib.nexa_causal_gqa_attention_paged_q4.argtypes = [fp, sz, table, sz, table, sz,
                                                     sz, sz, sz, sz, sz, sz, sz, sz, fp, sz, fp, sz]
    lib.nexa_causal_gqa_attention_paged_q4.restype = ctypes.c_int
    return lib


def ctypes_probe(library):
    lib = bind(library)
    rng = random.Random(402)
    bp = ctypes.POINTER(ctypes.c_uint8)
    sequence = 7
    # D=6/G=5 gives 14-byte head rows: subsequent scales are unaligned.
    for query_heads, kv_heads, dim, group in ((4, 1, 6, 5), (4, 2, 5, 3), (2, 2, 7, 9), (1, 1, 1, 1)):
        qwidth, kvwidth = query_heads * dim, kv_heads * dim
        query = [f32(rng.uniform(-2, 2)) for _ in range(sequence * qwidth)]
        key = [f32(rng.uniform(-4, 4)) for _ in range(sequence * kvwidth)]
        value = [f32(rng.uniform(-5, 5)) for _ in range(sequence * kvwidth)]
        expected_key, decoded_key = reference_pack(key, sequence * kv_heads, dim, group)
        expected_value, decoded_value = reference_pack(value, sequence * kv_heads, dim, group)
        flat_key = [value for row in decoded_key for value in row]
        flat_value = [value for row in decoded_value for value in row]
        reference = attention_oracle(query, flat_key, flat_value, sequence, query_heads, kv_heads, dim)
        row_bytes = lib.nexa_q4_row_size(dim, group)
        token_bytes = kv_heads * row_bytes
        full_native = None
        for page_tokens in (1, 2, 3, sequence):
            page_bytes = page_tokens * token_bytes + 7
            pages = (sequence + page_tokens - 1) // page_tokens
            for chunk_size in (1, 2, 3, sequence):
                owners_k = [(ctypes.c_uint8 * (page_bytes + 2))(*([0xA5] + [0xFF] * page_bytes + [0x5A])) for _ in range(pages)]
                owners_v = [(ctypes.c_uint8 * (page_bytes + 2))(*([0xA5] + [0xFF] * page_bytes + [0x5A])) for _ in range(pages)]
                table_k, table_v = (bp * (pages + 1))(), (bp * (pages + 1))()
                permutation = list(reversed(range(pages)))
                collected = []
                for past in range(0, sequence, chunk_size):
                    current = min(chunk_size, sequence - past)
                    end = past + current
                    chunk_bytes = current * token_bytes
                    packed_k, packed_v = (ctypes.c_uint8 * chunk_bytes)(), (ctypes.c_uint8 * chunk_bytes)()
                    # Quantize only the appended tokens/heads, never the old prefix.
                    for values, packed in ((key, packed_k), (value, packed_v)):
                        part = values[past * kvwidth:end * kvwidth]
                        assert lib.nexa_q4_quantize(floats(part), len(part), current * kv_heads, dim, group,
                                                    packed, chunk_bytes) == 0
                    assert bytes(packed_k) == expected_key[past * token_bytes:end * token_bytes]
                    assert bytes(packed_v) == expected_value[past * token_bytes:end * token_bytes]
                    for position in range(past, end):
                        page, offset = divmod(position, page_tokens)
                        owner_k, owner_v = owners_k[permutation[page]], owners_v[permutation[page]]
                        table_k[page] = ctypes.cast(ctypes.byref(owner_k, 1), bp)
                        table_v[page] = ctypes.cast(ctypes.byref(owner_v, 1), bp)
                        destination = 1 + offset * token_bytes
                        start = (position - past) * token_bytes
                        owner_k[destination:destination + token_bytes] = packed_k[start:start + token_bytes]
                        owner_v[destination:destination + token_bytes] = packed_v[start:start + token_bytes]
                    before = [bytes(owner) for owner in owners_k + owners_v]
                    before_tables = bytes(table_k), bytes(table_v)
                    q = floats(query[past * qwidth:end * qwidth])
                    output, scratch = floats([0] * len(q) + [123]), floats([float("nan")] * end + [321])
                    status = lib.nexa_causal_gqa_attention_paged_q4(q, len(q), table_k, len(table_k), table_v, len(table_v),
                        page_tokens, page_bytes, group, past, current, query_heads, kv_heads, dim,
                        scratch, end, output, len(q))
                    assert status == 0, (status, page_tokens, chunk_size, past, dim, group)
                    equal(output[:-1], reference[past * qwidth:end * qwidth], "Q4 KV scalar reference")
                    assert [bytes(owner) for owner in owners_k + owners_v] == before
                    assert (bytes(table_k), bytes(table_v)) == before_tables
                    assert output[-1] == 123 and scratch[-1] == 321
                    collected.extend(output[:-1])
                if full_native is None:
                    full_native = collected
                assert collected == full_native, "paging/chunking changed native packed-Q4 arithmetic"
                assert all(owner[0] == 0xA5 and owner[-1] == 0x5A for owner in owners_k + owners_v)
    print("Packed Q4 KV matches independent codec/attention oracle across heads, odd groups, tails, pages and chunks")


C_REGRESSIONS = r'''
#include "transformer.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void test_codec_and_boundaries(void) {
    /* D5/G3: two six-byte groups per head; P2:24 bytes per page. */
    float q[20]={0}, keys_f32[15]={0};
    float values_f32[15]={7,0,-7,1,-1, 14,0,-14,2,-2, 21,0,-21,3,-3};
    uint8_t k[48],v[48]; memset(k,255,sizeof(k)); memset(v,255,sizeof(v));
    assert(nexa_q4_row_size(5,3)==12);
    assert(nexa_q4_quantize(keys_f32,15,3,5,3,k,36)==0);
    assert(nexa_q4_quantize(values_f32,15,3,5,3,v,36)==0);
    const uint8_t *kt[3]={k,k+24,NULL}, *vt[3]={v,v+24,NULL};
    float out[21]={0}, scratch[4]={0}; out[20]=987; scratch[3]=654;
    #define VALID nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,out,20)
    assert(VALID==0);
    assert(out[0]==10.5f && out[5]==10.5f && out[10]==14 && out[15]==14);
    assert(out[20]==987 && scratch[3]==654);
    uint8_t before_k[48],before_v[48]; memcpy(before_k,k,48); memcpy(before_v,v,48);
    assert(VALID==0 && memcmp(before_k,k,48)==0 && memcmp(before_v,v,48)==0);

    /* Reserved code, negative/nonfinite scale, zero-scale code, odd padding. */
    k[4]=8; assert(VALID==-4); k[4]=0;
    k[2]=128;k[3]=191; assert(VALID==-4); k[2]=0;k[3]=0;
    k[2]=192;k[3]=127; assert(VALID==-4); k[2]=0;k[3]=0;
    k[2]=128;k[3]=127; assert(VALID==-4); k[2]=0;k[3]=0;
    k[4]=1; assert(VALID==-4); k[4]=0;
    k[2]=128;k[3]=63;k[5]=16; assert(VALID==-4); k[2]=0;k[3]=0;k[5]=0;
    k[8]=128;k[9]=63;k[11]=1; assert(VALID==-4); k[8]=0;k[9]=0;k[11]=0;
    v[26]=192;v[27]=127; assert(VALID==-4); memcpy(v,before_v,48);
    q[19]=NAN; assert(VALID==-4); q[19]=0;
    assert(VALID==0); /* Unused fourth token is all 0xff and is never inspected. */

    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,1,vt,3,2,24,3,1,2,2,1,5,scratch,3,out,20)==-2);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,1,2,24,3,1,2,2,1,5,scratch,3,out,20)==-2);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,23,3,1,2,2,1,5,scratch,3,out,20)==-2);
    assert(nexa_causal_gqa_attention_paged_q4(q,19,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,out,20)==-2);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,2,out,20)==-2);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,out,19)==-2);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,0,24,3,1,2,2,1,5,scratch,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,0,1,2,2,1,5,scratch,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,3,1,0,2,1,5,scratch,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,3,1,2,3,2,5,scratch,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,3,1,2,2,0,5,scratch,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,3,1,2,2,1,0,scratch,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,3,SIZE_MAX,1,2,1,5,scratch,3,out,20)==-3);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,SIZE_MAX,SIZE_MAX,3,1,2,2,1,5,scratch,3,out,20)==-3);
    assert(nexa_causal_gqa_attention_paged_q4(q,SIZE_MAX,kt,3,vt,3,1,SIZE_MAX,1,0,1,1,1,SIZE_MAX,scratch,3,out,SIZE_MAX)==-3);
    assert(nexa_causal_gqa_attention_paged_q4(q,SIZE_MAX,kt,3,vt,3,1,SIZE_MAX,1,0,1,SIZE_MAX,1,2,scratch,3,out,SIZE_MAX)==-3);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,SIZE_MAX,vt,SIZE_MAX,1,5,1,SIZE_MAX/4,1,1,1,1,scratch,SIZE_MAX,out,20)==-3);

    assert(nexa_causal_gqa_attention_paged_q4(NULL,20,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,NULL,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,NULL,3,2,24,3,1,2,2,1,5,scratch,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,NULL,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,NULL,20)==-1);
    kt[1]=NULL; assert(VALID==-1); kt[1]=k+24;
    vt[0]=NULL; assert(VALID==-1); vt[0]=v;
    kt[1]=(const uint8_t *)(uintptr_t)(UINTPTR_MAX-1); assert(VALID==-1); kt[1]=k+24;
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,q,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,(float*)k,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,(float*)v,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,q,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,(float*)k,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,(float*)v,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,out+1,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,(float*)kt,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,(float*)vt,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,(float*)kt,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q4(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,(float*)vt,3,out,20)==-1);
    #undef VALID
}

static void test_double_decode_and_numeric_range(void) {
    /* Uniform attention: 0.1f*3 and -0.3f cancel only if prematurely rounded. */
    uint8_t zero[10]={0};
    uint8_t values[10]={0xcd,0xcc,0xcc,0x3d,3, 0x9a,0x99,0x99,0x3e,15};
    const uint8_t *kt[1]={zero},*vt[1]={values};
    float q[1]={0},scratch[2],out[1];
    assert(nexa_causal_gqa_attention_paged_q4(q,1,kt,1,vt,1,2,10,1,1,1,1,1,1,scratch,2,out,1)==0);
    assert(out[0]==-ldexpf(1,-28));
    /* Finite scale with decoded value 7*FLT_MAX: only the final output overflows. */
    uint8_t huge[5]={0xff,0xff,0x7f,0x7f,7}; vt[0]=huge;
    assert(nexa_causal_gqa_attention_paged_q4(q,1,kt,1,vt,1,1,5,1,0,1,1,1,1,scratch,1,out,1)==-5);
    /* Scores larger than FLT_MAX remain stable because dots/max are double. */
    uint8_t largekeys[10]={0xff,0xff,0x7f,0x7f,7, 0xff,0xff,0x7f,0x7f,9};
    uint8_t finitevalues[10]={0,0,128,63,3, 0,0,128,63,9};
    kt[0]=largekeys;vt[0]=finitevalues;q[0]=FLT_MAX;
    assert(nexa_causal_gqa_attention_paged_q4(q,1,kt,1,vt,1,2,10,1,1,1,1,1,1,scratch,2,out,1)==0);
    assert(out[0]==3);
}

int main(void) {
    test_codec_and_boundaries(); test_double_decode_and_numeric_range();
    puts("Packed Q4 KV sanitizer regressions passed");
    return 0;
}
'''


class Q4KVKernelRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("clang") or shutil.which("cc")
        if not cls.compiler:
            raise unittest.SkipTest("C compiler unavailable")

    def test_c_sanitizers_codec_boundaries_and_no_heap(self):
        with tempfile.TemporaryDirectory(prefix="nexa-q4-kv-native-") as directory:
            directory = Path(directory)
            source = directory / "regressions.c"
            source.write_text(C_REGRESSIONS)
            binary = directory / ("kernels.exe" if os.name == "nt" else "kernels")
            command = [self.compiler, "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                       "-I", str(SOURCE), str(SOURCE / "q4.c"), str(SOURCE / "transformer.c"),
                       str(source), "-o", str(binary)]
            command.extend(f"-D{name}=nexa_q4_kv_forbidden_{name}" for name in ("malloc", "calloc", "realloc", "free"))
            if os.name != "nt":
                command.extend(["-lm", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"])
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            env = dict(os.environ)
            env["ASAN_OPTIONS"] = "detect_leaks=0" if sys.platform == "darwin" else "detect_leaks=1"
            env["UBSAN_OPTIONS"] = "halt_on_error=1"
            result = subprocess.run([str(binary)], capture_output=True, text=True, env=env, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_independent_codec_and_attention_reference(self):
        with tempfile.TemporaryDirectory(prefix="nexa-q4-kv-reference-") as directory:
            extension = "dll" if os.name == "nt" else "dylib" if sys.platform == "darwin" else "so"
            library = Path(directory) / f"transformer.{extension}"
            command = [self.compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror"]
            if os.name != "nt":
                command.append("-fPIC")
            command.extend(["-dynamiclib" if sys.platform == "darwin" else "-shared",
                            str(SOURCE / "q4.c"), str(SOURCE / "transformer.c"), "-o", str(library)])
            if os.name != "nt":
                command.append("-lm")
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--ctypes-probe", str(library)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--ctypes-probe":
        ctypes_probe(sys.argv[2])
    else:
        unittest.main()
