"""Paged F32 attention: physical indirection, causal prefixes and C boundaries."""
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
from test_kv_kernels_regressions import bind as bind_cached
from test_transformer_kernels_regressions import attention_oracle, equal, f32, floats


def bind(library):
    lib = bind_cached(library)
    fp, sz = ctypes.POINTER(ctypes.c_float), ctypes.c_size_t
    table = ctypes.POINTER(fp)
    lib.nexa_causal_gqa_attention_paged.argtypes = [fp, sz, table, sz, table, sz, sz, sz,
                                                  sz, sz, sz, sz, sz, fp, sz, fp, sz]
    lib.nexa_causal_gqa_attention_paged.restype = ctypes.c_int
    return lib


def ctypes_probe(library):
    lib = bind(library)
    rng = random.Random(401)
    fp = ctypes.POINTER(ctypes.c_float)
    sequence = 7
    for query_heads, kv_heads, dim in ((4, 2, 4), (4, 1, 6), (3, 3, 2)):
        qwidth, kvwidth = query_heads * dim, kv_heads * dim
        query = [f32(rng.uniform(-2, 2)) for _ in range(sequence * qwidth)]
        key = [f32(rng.uniform(-2, 2)) for _ in range(sequence * kvwidth)]
        value = [f32(rng.uniform(-5, 5)) for _ in range(sequence * kvwidth)]
        reference = attention_oracle(query, key, value, sequence, query_heads, kv_heads, dim)
        full = floats([0] * len(query))
        assert lib.nexa_causal_gqa_attention(floats(query), len(query), floats(key), len(key), floats(value), len(value),
                                             sequence, query_heads, kv_heads, dim, floats([0] * sequence), sequence,
                                             full, len(query)) == 0
        equal(full, reference, "full oracle")
        for page_tokens in (1, 2, 3):
            page_floats = page_tokens * kvwidth + 3  # Uniform excess capacity must be ignored.
            pages = (sequence + page_tokens - 1) // page_tokens
            for chunk_size in (1, 2, 3):
                # Separate allocations with guard lanes; logical table order is a
                # permutation of physical pages and contains extra NULL entries.
                key_owners = [floats([987] + [float("nan")] * page_floats + [654]) for _ in range(pages)]
                value_owners = [floats([987] + [float("nan")] * page_floats + [654]) for _ in range(pages)]
                permutation = list(reversed(range(pages)))
                key_table, value_table = (fp * (pages + 1))(), (fp * (pages + 1))()
                collected = []
                for past in range(0, sequence, chunk_size):
                    current = min(chunk_size, sequence - past)
                    end = past + current
                    for position in range(past, end):
                        logical, lane = divmod(position, page_tokens)
                        owner_index = permutation[logical]
                        key_owner, value_owner = key_owners[owner_index], value_owners[owner_index]
                        key_table[logical] = ctypes.cast(ctypes.byref(key_owner, 4), fp)
                        value_table[logical] = ctypes.cast(ctypes.byref(value_owner, 4), fp)
                        start = 1 + lane * kvwidth
                        key_owner[start:start + kvwidth] = key[position * kvwidth:(position + 1) * kvwidth]
                        value_owner[start:start + kvwidth] = value[position * kvwidth:(position + 1) * kvwidth]
                    before = [bytes(owner) for owner in key_owners + value_owners]
                    table_before = bytes(key_table), bytes(value_table)
                    current_query = floats(query[past * qwidth:end * qwidth])
                    out, scratch = floats([0] * len(current_query) + [123]), floats([float("nan")] * end + [321])
                    status = lib.nexa_causal_gqa_attention_paged(
                        current_query, len(current_query), key_table, len(key_table), value_table, len(value_table),
                        page_tokens, page_floats, past, current, query_heads, kv_heads, dim,
                        scratch, end, out, len(current_query))
                    assert status == 0, (status, page_tokens, chunk_size, past)
                    equal(out[:-1], reference[past * qwidth:end * qwidth], "paged oracle")
                    cached = floats([0] * len(current_query))
                    assert lib.nexa_causal_gqa_attention_cached(
                        current_query, len(current_query), floats(key), len(key), floats(value), len(value),
                        past, current, query_heads, kv_heads, dim, scratch, end, cached, len(cached)) == 0
                    assert out[:-1] == list(cached), "paged arithmetic differs from contiguous cache"
                    assert [bytes(owner) for owner in key_owners + value_owners] == before
                    assert (bytes(key_table), bytes(value_table)) == table_before
                    assert out[-1] == 123 and scratch[-1] == 321
                    collected.extend(out[:-1])
                assert collected == list(full), "paging/chunking changed full attention"
                assert all(owner[0] == 987 and owner[-1] == 654 for owner in key_owners + value_owners)
    print("Paged attention matches contiguous/full/oracle for MHA/MQA/GQA, pages 1/2/3 and chunks 1/2/3")


C_REGRESSIONS = r'''
#include "transformer.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void test_paged_boundaries(void) {
    float q[8]={0}, out[9]={0,0,0,0,0,0,0,0,987}, scratch[5]={0,0,0,0,654};
    float kfirst[8]={0,0,0,0,0,0,NAN,NAN}, klast[8]={0,0,NAN,NAN,NAN,NAN,NAN,NAN};
    float vfirst[8]={2,10,4,14,6,18,NAN,NAN}, vlast[8]={8,22,NAN,NAN,NAN,NAN,NAN,NAN};
    const float *keys[3]={kfirst,klast,NULL}, *values[3]={vfirst,vlast,NULL};
    float before_kfirst[8],before_klast[8],before_vfirst[8],before_vlast[8];
    memcpy(before_kfirst,kfirst,sizeof(kfirst)); memcpy(before_klast,klast,sizeof(klast));
    memcpy(before_vfirst,vfirst,sizeof(vfirst)); memcpy(before_vlast,vlast,sizeof(vlast));
    #define VALID nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,8,2,2,4,2,1,scratch,4,out,8)
    assert(VALID==0);
    float expected[8]={4,4,14,14,5,5,16,16};
    for(size_t i=0;i<8;i++) assert(out[i]==expected[i]);
    assert(out[8]==987 && scratch[4]==654);
    assert(memcmp(kfirst,before_kfirst,sizeof(kfirst))==0 && memcmp(klast,before_klast,sizeof(klast))==0);
    assert(memcmp(vfirst,before_vfirst,sizeof(vfirst))==0 && memcmp(vlast,before_vlast,sizeof(vlast))==0);
    vlast[0]=1e6f; vlast[1]=-1e6f;
    assert(VALID==0 && out[0]==4 && out[2]==14); /* Future page cannot affect earlier query. */
    vlast[0]=8; vlast[1]=22;

    assert(nexa_causal_gqa_attention_paged(q,8,keys,1,values,3,3,8,2,2,4,2,1,scratch,4,out,8)==-2);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,1,3,8,2,2,4,2,1,scratch,4,out,8)==-2);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,5,2,2,4,2,1,scratch,4,out,8)==-2);
    assert(nexa_causal_gqa_attention_paged(q,7,keys,3,values,3,3,8,2,2,4,2,1,scratch,4,out,8)==-2);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,8,2,2,4,2,1,scratch,3,out,8)==-2);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,8,2,2,4,2,1,scratch,4,out,7)==-2);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,0,8,2,2,4,2,1,scratch,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,8,2,0,4,2,1,scratch,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,8,2,2,3,2,1,scratch,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,8,2,2,4,0,1,scratch,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,8,2,2,4,2,0,scratch,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,8,SIZE_MAX,1,4,2,1,scratch,4,out,8)==-3);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,SIZE_MAX,8,2,2,4,2,1,scratch,4,out,8)==-3);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,SIZE_MAX/4+1,SIZE_MAX,0,1,1,1,1,scratch,4,out,8)==-3);
    assert(nexa_causal_gqa_attention_paged(q,SIZE_MAX,keys,3,values,3,1,1,0,1,SIZE_MAX,1,2,scratch,4,out,SIZE_MAX)==-3);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,SIZE_MAX,values,SIZE_MAX,1,1,SIZE_MAX/4,1,1,1,1,scratch,SIZE_MAX,out,8)==-3);

    assert(nexa_causal_gqa_attention_paged(NULL,8,keys,3,values,3,3,8,2,2,4,2,1,scratch,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_paged(q,8,NULL,3,values,3,3,8,2,2,4,2,1,scratch,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,NULL,3,3,8,2,2,4,2,1,scratch,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,8,2,2,4,2,1,NULL,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,8,2,2,4,2,1,scratch,4,NULL,8)==-1);
    keys[1]=NULL; assert(VALID==-1); keys[1]=klast;
    values[0]=NULL; assert(VALID==-1); values[0]=vfirst;
    keys[1]=(const float *)(uintptr_t)(UINTPTR_MAX-1); assert(VALID==-1); keys[1]=klast;

    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,8,2,2,4,2,1,scratch,4,q,8)==-1);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,8,2,2,4,2,1,scratch,4,kfirst,8)==-1);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,8,2,2,4,2,1,scratch,4,vlast,8)==-1);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,8,2,2,4,2,1,q,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,8,2,2,4,2,1,kfirst,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,8,2,2,4,2,1,vlast,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,8,2,2,4,2,1,out+1,4,out,8)==-1);
    /* Pointer tables are read inputs too; writing through their storage is invalid. */
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,8,2,2,4,2,1,scratch,4,(float*)keys,8)==-1);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,8,2,2,4,2,1,scratch,4,(float*)values,8)==-1);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,8,2,2,4,2,1,(float*)keys,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_paged(q,8,keys,3,values,3,3,8,2,2,4,2,1,(float*)values,4,out,8)==-1);
    assert(nexa_causal_gqa_attention_paged(q,8,(const float *const *)(uintptr_t)(UINTPTR_MAX-1),3,values,3,3,8,2,2,4,2,1,scratch,4,out,8)==-1);

    q[7]=NAN; assert(VALID==-4); q[7]=0;
    kfirst[0]=NAN; assert(VALID==-4); kfirst[0]=0;
    klast[1]=INFINITY; assert(VALID==-4); klast[1]=0;
    vfirst[0]=NAN; assert(VALID==-4); vfirst[0]=2;
    vlast[1]=INFINITY; assert(VALID==-4); vlast[1]=22;
    assert(VALID==0); /* Remaining suffix and unused pointer entry remain invalid but unread. */
    #undef VALID
}

static void test_extreme_scores(void) {
    float q[1]={FLT_MAX}, k0[1]={FLT_MAX}, k1[1]={-FLT_MAX}, v0[1]={3}, v1[1]={-7};
    const float *keys[2]={k0,k1}, *values[2]={v0,v1};
    float scratch[2], out[1];
    assert(nexa_causal_gqa_attention_paged(q,1,keys,2,values,2,1,1,1,1,1,1,1,scratch,2,out,1)==0);
    assert(out[0]==3);
}

int main(void) {
    test_paged_boundaries(); test_extreme_scores();
    puts("Paged KV kernel sanitizer regressions passed");
    return 0;
}
'''


class PagedKVKernelRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("clang") or shutil.which("cc")
        if not cls.compiler:
            raise unittest.SkipTest("C compiler unavailable")

    def test_c_sanitizers_tables_pages_and_no_heap(self):
        with tempfile.TemporaryDirectory(prefix="nexa-paged-kv-native-") as directory:
            directory = Path(directory)
            source = directory / "regressions.c"
            source.write_text(C_REGRESSIONS)
            binary = directory / ("kernels.exe" if os.name == "nt" else "kernels")
            command = [self.compiler, "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                       "-I", str(SOURCE), str(SOURCE / "q4.c"), str(SOURCE / "transformer.c"),
                       str(source), "-o", str(binary)]
            command.extend(f"-D{name}=nexa_paged_kv_forbidden_{name}" for name in ("malloc", "calloc", "realloc", "free"))
            if os.name != "nt":
                command.extend(["-lm", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"])
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            env = dict(os.environ)
            env["ASAN_OPTIONS"] = "detect_leaks=0" if sys.platform == "darwin" else "detect_leaks=1"
            env["UBSAN_OPTIONS"] = "halt_on_error=1"
            result = subprocess.run([str(binary)], capture_output=True, text=True, env=env, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_noncontiguous_pages_match_cached_full_and_scalar_reference(self):
        with tempfile.TemporaryDirectory(prefix="nexa-paged-kv-reference-") as directory:
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
