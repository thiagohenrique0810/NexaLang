"""Incremental RoPE/GQA kernels: dirty cache suffixes and bounded scratch."""
import ctypes
import math
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
from test_transformer_kernels_regressions import (
    attention_oracle, bind as bind_full, equal, f32, floats,
)


def bind(library):
    lib = bind_full(library)
    fp, sz = ctypes.POINTER(ctypes.c_float), ctypes.c_size_t
    lib.nexa_rope_offset.argtypes = [fp, sz, sz, sz, sz, ctypes.c_double, sz, fp, sz]
    lib.nexa_rope_offset.restype = ctypes.c_int
    lib.nexa_causal_gqa_attention_cached.argtypes = [fp, sz, fp, sz, fp, sz, sz, sz, sz, sz, sz, fp, sz, fp, sz]
    lib.nexa_causal_gqa_attention_cached.restype = ctypes.c_int
    return lib


def rope_oracle(values, sequence, heads, dim, offset, theta=10000):
    expected = [0.0] * len(values)
    for position in range(sequence):
        for head in range(heads):
            start = (position * heads + head) * dim
            for lane in range(dim // 2):
                angle = (offset + position) * theta ** (-2 * lane / dim)
                cosine, sine = math.cos(angle), math.sin(angle)
                a, b = values[start + lane], values[start + lane + dim // 2]
                expected[start + lane] = f32(a * cosine - b * sine)
                expected[start + lane + dim // 2] = f32(b * cosine + a * sine)
    return expected


def ctypes_probe(library):
    lib = bind(library)
    rng = random.Random(400)
    for sequence, heads, dim, offset, theta in ((1, 1, 2, 7, 10000), (3, 4, 6, 11, 10000),
                                               (2, 2, 4, 8192, 2), (3, 1, 8, 0, 0.5)):
        values = [f32(rng.uniform(-3, 3)) for _ in range(sequence * heads * dim)]
        output = floats([0] * len(values) + [987])
        assert lib.nexa_rope_offset(floats(values), len(values), sequence, heads, dim, theta,
                                    offset, output, len(values)) == 0
        equal(output[:-1], rope_oracle(values, sequence, heads, dim, offset, theta), "rope offset")
        assert output[-1] == 987

    sequence, capacity = 7, 10
    for query_heads, kv_heads, dim in ((4, 2, 4), (4, 1, 6), (3, 3, 2)):
        qwidth, kvwidth = query_heads * dim, kv_heads * dim
        query = [f32(rng.uniform(-2, 2)) for _ in range(sequence * qwidth)]
        key = [f32(rng.uniform(-2, 2)) for _ in range(sequence * kvwidth)]
        value = [f32(rng.uniform(-4, 4)) for _ in range(sequence * kvwidth)]
        rotated_q, rotated_k = floats([0] * len(query)), floats([0] * len(key))
        assert lib.nexa_rope(floats(query), len(query), sequence, query_heads, dim, 10000, rotated_q, len(query)) == 0
        assert lib.nexa_rope(floats(key), len(key), sequence, kv_heads, dim, 10000, rotated_k, len(key)) == 0
        reference = attention_oracle(rotated_q, rotated_k, value, sequence, query_heads, kv_heads, dim)
        full_output, scratch = floats([0] * len(query)), floats([0] * sequence)
        assert lib.nexa_causal_gqa_attention(rotated_q, len(query), rotated_k, len(key), floats(value), len(value),
                                             sequence, query_heads, kv_heads, dim, scratch, sequence, full_output, len(query)) == 0
        equal(full_output, reference, "full attention oracle")
        for chunk_size in (1, 2, 3):
            kcache = floats([float("nan")] * (capacity * kvwidth))
            vcache = floats([float("nan")] * (capacity * kvwidth))
            collected = []
            for past in range(0, sequence, chunk_size):
                current = min(chunk_size, sequence - past)
                end = past + current
                chunkq = query[past * qwidth:end * qwidth]
                chunkk = key[past * kvwidth:end * kvwidth]
                qout, kout = floats([0] * len(chunkq)), floats([0] * len(chunkk))
                assert lib.nexa_rope_offset(floats(chunkq), len(chunkq), current, query_heads, dim,
                                             10000, past, qout, len(chunkq)) == 0
                assert lib.nexa_rope_offset(floats(chunkk), len(chunkk), current, kv_heads, dim,
                                             10000, past, kout, len(chunkk)) == 0
                equal(qout, rotated_q[past * qwidth:end * qwidth], "chunk query rope")
                equal(kout, rotated_k[past * kvwidth:end * kvwidth], "chunk key rope")
                kcache[past * kvwidth:end * kvwidth] = kout
                vcache[past * kvwidth:end * kvwidth] = value[past * kvwidth:end * kvwidth]
                before_k, before_v = bytes(kcache), bytes(vcache)
                output = floats([0] * len(chunkq) + [654])
                scratch = floats([float("nan")] * end + [321])
                assert lib.nexa_causal_gqa_attention_cached(qout, len(chunkq), kcache, len(kcache), vcache, len(vcache),
                                                            past, current, query_heads, kv_heads, dim,
                                                            scratch, end, output, len(chunkq)) == 0
                assert bytes(kcache) == before_k and bytes(vcache) == before_v, "kernel modified KV"
                assert output[-1] == 654 and scratch[-1] == 321
                equal(output[:-1], reference[past * qwidth:end * qwidth], "cached attention oracle")
                collected.extend(output[:-1])
            assert collected == list(full_output), "chunking changed native arithmetic"
    print("Cached attention chunks 1/2/3 match full causal MHA/MQA/GQA and scalar reference")


C_REGRESSIONS = r'''
#include "transformer.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void test_rope_offset(void) {
    float input[4]={1,2,3,4}, output[5]={0,0,0,0,987};
    assert(nexa_rope_offset(input,4,2,1,2,10000,7,output,4)==0);
    assert(fabs(output[0]-(cos(7)-2*sin(7)))<1e-6);
    assert(fabs(output[2]-(3*cos(8)-4*sin(8)))<1e-6);
    assert(output[4]==987);
    assert(nexa_rope_offset(input,4,2,1,2,10000,SIZE_MAX,output,4)==-3);
    assert(nexa_rope_offset(input,SIZE_MAX,SIZE_MAX,1,2,10000,0,output,SIZE_MAX)==-3);
    assert(nexa_rope_offset(input,4,2,1,1,10000,1,output,4)==-1);
    assert(nexa_rope_offset(input,4,0,1,2,10000,SIZE_MAX,output,4)==-1);
    assert(nexa_rope_offset(input,4,2,0,2,10000,0,output,4)==-1);
    assert(nexa_rope_offset(input,4,2,1,2,NAN,1,output,4)==-1);
    assert(nexa_rope_offset(input,4,2,1,2,0,1,output,4)==-1);
    assert(nexa_rope_offset(NULL,4,2,1,2,10000,1,output,4)==-1);
    assert(nexa_rope_offset(input,4,2,1,2,10000,1,NULL,4)==-1);
    assert(nexa_rope_offset(input,3,2,1,2,10000,1,output,4)==-2);
    assert(nexa_rope_offset(input,4,2,1,2,10000,1,output,3)==-2);
    assert(nexa_rope_offset(input,4,2,1,2,10000,1,input,4)==-1);
    assert(nexa_rope_offset(input,2,1,1,2,10000,1,input+1,2)==-1);
    input[3]=NAN;
    assert(nexa_rope_offset(input,4,2,1,2,10000,1,output,4)==-4);
    /* Excess input capacity is not part of the current chunk. */
    assert(nexa_rope_offset(input,4,1,1,2,10000,1,output,4)==0);
    float huge[2]={FLT_MAX,FLT_MAX};
    assert(nexa_rope_offset(huge,2,1,1,2,10000,1,output,2)==-5);
}

static void test_cached_attention(void) {
    /* Two committed rows + two current rows; unused cache capacity is dirty. */
    float q[8]={0}, k[10]={0,0,0,0,0,0,0,0,NAN,NAN};
    float v[10]={2,10,4,14,6,18,8,22,NAN,NAN};
    float out[9]={0,0,0,0,0,0,0,0,987}, scratch[5]={0,0,0,0,654};
    float before_k[10], before_v[10];
    memcpy(before_k,k,sizeof(k)); memcpy(before_v,v,sizeof(v));
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,10,2,2,4,2,1,scratch,4,out,8)==0);
    float expected[8]={4,4,14,14,5,5,16,16};
    for(size_t i=0;i<8;i++) assert(out[i]==expected[i]);
    assert(out[8]==987 && scratch[4]==654);
    assert(memcmp(k,before_k,sizeof(k))==0 && memcmp(v,before_v,sizeof(v))==0);
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,10,2,2,4,2,1,scratch,3,out,8)==-2);
    assert(nexa_causal_gqa_attention_cached(q,7,k,10,v,10,2,2,4,2,1,scratch,4,out,8)==-2);
    assert(nexa_causal_gqa_attention_cached(q,8,k,7,v,10,2,2,4,2,1,scratch,4,out,8)==-2);
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,7,2,2,4,2,1,scratch,4,out,8)==-2);
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,10,2,2,4,2,1,scratch,4,out,7)==-2);
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,10,SIZE_MAX,1,4,2,1,scratch,4,out,8)==-3);
    assert(nexa_causal_gqa_attention_cached(q,8,k,SIZE_MAX,v,SIZE_MAX,SIZE_MAX/4,1,1,1,1,scratch,SIZE_MAX,out,8)==-3);
    assert(nexa_causal_gqa_attention_cached(q,SIZE_MAX,k,10,v,10,0,1,SIZE_MAX,1,2,scratch,4,out,SIZE_MAX)==-3);
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,10,2,0,4,2,1,scratch,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,10,2,2,0,2,1,scratch,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,10,2,2,4,0,1,scratch,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,10,2,2,4,2,0,scratch,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,10,2,2,3,2,1,scratch,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_cached(NULL,8,k,10,v,10,2,2,4,2,1,scratch,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_cached(q,8,NULL,10,v,10,2,2,4,2,1,scratch,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,NULL,10,2,2,4,2,1,scratch,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,10,2,2,4,2,1,NULL,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,10,2,2,4,2,1,scratch,4,NULL,8)==-1);
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,10,2,2,4,2,1,scratch,4,q,8)==-1);
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,10,2,2,4,2,1,scratch,4,k,8)==-1);
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,10,2,2,4,2,1,scratch,4,v,8)==-1);
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,10,2,2,4,2,1,q,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,10,2,2,4,2,1,k,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,10,2,2,4,2,1,v,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,10,2,2,4,2,1,out+1,4,out,8)==-1);
    q[7]=NAN;
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,10,2,2,4,2,1,scratch,4,out,8)==-4);
    q[7]=0; k[0]=NAN; /* A corrupt committed cache prefix cannot be ignored. */
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,10,2,2,4,2,1,scratch,4,out,8)==-4);
    k[0]=0; v[7]=INFINITY;
    assert(nexa_causal_gqa_attention_cached(q,8,k,10,v,10,2,2,4,2,1,scratch,4,out,8)==-4);
    float largeq[1]={FLT_MAX}, largek[3]={FLT_MAX,-FLT_MAX,NAN}, value[3]={3,-7,NAN};
    assert(nexa_causal_gqa_attention_cached(largeq,1,largek,3,value,3,1,1,1,1,1,scratch,2,out,1)==0);
    assert(out[0]==3); /* Stable score subtraction despite float32 score overflow. */
}

int main(void) {
    test_rope_offset(); test_cached_attention();
    puts("Incremental KV kernels passed sanitizer regressions");
    return 0;
}
'''


class KVKernelRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("clang") or shutil.which("cc")
        if not cls.compiler:
            raise unittest.SkipTest("C compiler unavailable")

    def test_c_sanitizers_capacities_dirty_suffix_and_no_heap(self):
        with tempfile.TemporaryDirectory(prefix="nexa-kv-native-") as directory:
            directory = Path(directory)
            source = directory / "regressions.c"
            source.write_text(C_REGRESSIONS)
            binary = directory / ("kernels.exe" if os.name == "nt" else "kernels")
            command = [self.compiler, "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                       "-I", str(SOURCE), str(SOURCE / "q4.c"), str(SOURCE / "transformer.c"),
                       str(source), "-o", str(binary)]
            command.extend(f"-D{name}=nexa_kv_forbidden_{name}" for name in ("malloc", "calloc", "realloc", "free"))
            if os.name != "nt":
                command.extend(["-lm", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"])
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            env = dict(os.environ)
            env["ASAN_OPTIONS"] = "detect_leaks=0" if sys.platform == "darwin" else "detect_leaks=1"
            env["UBSAN_OPTIONS"] = "halt_on_error=1"
            result = subprocess.run([str(binary)], capture_output=True, text=True, env=env, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cached_chunks_match_full_mha_mqa_gqa_and_scalar_oracle(self):
        with tempfile.TemporaryDirectory(prefix="nexa-kv-reference-") as directory:
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
