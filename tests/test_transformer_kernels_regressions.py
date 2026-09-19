"""Allocation-free CPU Transformer kernels: scalar oracle and native sanitizers."""
import ctypes
import math
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime/nexapack"


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def floats(values):
    return (ctypes.c_float * len(values))(*values)


def bind(library):
    lib = ctypes.CDLL(str(library))
    fp, bp, sz = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
    lib.nexa_rmsnorm.argtypes = [fp, sz, fp, sz, sz, sz, ctypes.c_double, fp, sz]
    lib.nexa_rope.argtypes = [fp, sz, sz, sz, sz, ctypes.c_double, fp, sz]
    lib.nexa_swiglu.argtypes = [fp, sz, fp, sz, sz, fp, sz]
    lib.nexa_add.argtypes = [fp, sz, fp, sz, sz, fp, sz]
    lib.nexa_causal_gqa_attention.argtypes = [fp, sz, fp, sz, fp, sz, sz, sz, sz, sz, fp, sz, fp, sz]
    lib.nexa_q4_decode_row.argtypes = [bp, sz, sz, sz, fp, sz]
    for name in ("nexa_rmsnorm", "nexa_rope", "nexa_swiglu", "nexa_add", "nexa_causal_gqa_attention", "nexa_q4_decode_row"):
        getattr(lib, name).restype = ctypes.c_int
    return lib


def equal(actual, expected, label):
    assert len(actual) == len(expected), label
    for index, (left, right) in enumerate(zip(actual, expected)):
        assert math.isfinite(left) and math.isclose(left, right, abs_tol=3e-6, rel_tol=3e-6), (
            label, index, left, right)


def attention_oracle(q, k, v, sequence, query_heads, kv_heads, dim):
    output = []
    repeats = query_heads // kv_heads
    for position in range(sequence):
        for head in range(query_heads):
            qoffset = (position * query_heads + head) * dim
            offset = (head // repeats) * dim
            scores = [sum(q[qoffset + lane] * k[past * kv_heads * dim + offset + lane]
                          for lane in range(dim)) / math.sqrt(dim) for past in range(position + 1)]
            maximum = max(scores)
            weights = [f32(math.exp(score - maximum)) for score in scores]
            denominator = sum(weights)
            for lane in range(dim):
                output.append(f32(sum(weight * v[past * kv_heads * dim + offset + lane]
                                      for past, weight in enumerate(weights)) / denominator))
    return output


def ctypes_probe(library):
    """Isolated process: a native memory error cannot terminate unittest itself."""
    sys.path.insert(0, str(ROOT))
    from runtime.nexapack.format import decode_q4_row, quantize_q4_row

    lib = bind(library)
    rng = random.Random(512)
    for sequence, hidden in ((1, 1), (3, 5), (4, 16)):
        count = sequence * hidden
        x = [f32(rng.uniform(-5, 5)) for _ in range(count)]
        weight = [f32(rng.uniform(-2, 2)) for _ in range(hidden)]
        output = floats([-111.0] * (count + 1))
        assert lib.nexa_rmsnorm(floats(x), count, floats(weight), hidden, sequence, hidden, 1e-5, output, count) == 0
        expected = []
        for position in range(sequence):
            values = x[position * hidden:(position + 1) * hidden]
            factor = 1 / math.sqrt(sum(value * value for value in values) / hidden + 1e-5)
            expected.extend(f32((value * factor) * w) for value, w in zip(values, weight))
        equal(output[:count], expected, "rmsnorm")
        assert output[count] == -111
        up = [f32(rng.uniform(-2, 2)) for _ in range(count)]
        assert lib.nexa_swiglu(floats(x), count, floats(up), count, count, output, count) == 0
        equal(output[:count], [f32((a / (1 + math.exp(-a))) * b) for a, b in zip(x, up)], "swiglu")
        assert lib.nexa_add(floats(x), count, floats(up), count, count, output, count) == 0
        equal(output[:count], [f32(a + b) for a, b in zip(x, up)], "add")
        assert output[count] == -111

    for sequence, heads, dim, theta in ((1, 1, 2, 10000), (5, 3, 6, 10000), (3, 2, 8, 2), (2, 1, 4, 0.5)):
        count = sequence * heads * dim
        x = [f32(rng.uniform(-3, 3)) for _ in range(count)]
        output = floats([-222.0] * (count + 1))
        assert lib.nexa_rope(floats(x), count, sequence, heads, dim, theta, output, count) == 0
        expected = [0.0] * count
        for position in range(sequence):
            for head in range(heads):
                offset = (position * heads + head) * dim
                for lane in range(dim // 2):
                    angle = position * theta ** (-2 * lane / dim)
                    c, s = math.cos(angle), math.sin(angle)
                    a, b = x[offset + lane], x[offset + lane + dim // 2]
                    expected[offset + lane] = f32(a * c - b * s)
                    expected[offset + lane + dim // 2] = f32(b * c + a * s)
        equal(output[:count], expected, "rope")
        assert output[count] == -222
        assert list(output[:heads * dim]) == x[:heads * dim], "position zero must be identity"

    for sequence, query_heads, kv_heads, dim in ((1, 1, 1, 1), (2, 4, 1, 2), (5, 4, 2, 3), (3, 2, 2, 8)):
        qcount, kvcount = sequence * query_heads * dim, sequence * kv_heads * dim
        q = [f32(rng.uniform(-2, 2)) for _ in range(qcount)]
        k = [f32(rng.uniform(-2, 2)) for _ in range(kvcount)]
        v = [f32(rng.uniform(-4, 4)) for _ in range(kvcount)]
        scratch, output = floats([-333.0] * (sequence + 1)), floats([-444.0] * (qcount + 1))
        assert lib.nexa_causal_gqa_attention(floats(q), qcount, floats(k), kvcount, floats(v), kvcount,
                                             sequence, query_heads, kv_heads, dim, scratch, sequence, output, qcount) == 0
        equal(output[:qcount], attention_oracle(q, k, v, sequence, query_heads, kv_heads, dim), "attention")
        assert scratch[sequence] == -333 and output[qcount] == -444
        before = list(output[:query_heads * dim])
        # Causality: arbitrary future keys/values cannot affect position zero.
        k[kv_heads * dim:] = [f32(1000 + index) for index in range(kvcount - kv_heads * dim)]
        v[kv_heads * dim:] = [f32(-1000 - index) for index in range(kvcount - kv_heads * dim)]
        assert lib.nexa_causal_gqa_attention(floats(q), qcount, floats(k), kvcount, floats(v), kvcount,
                                             sequence, query_heads, kv_heads, dim, scratch, sequence, output, qcount) == 0
        assert list(output[:query_heads * dim]) == before

    # Scores around 1e76 overflow float32; max-shifting in double remains valid.
    maximum = f32(3.4028234663852886e38)
    q, k, v = floats([maximum, maximum]), floats([maximum, -maximum]), floats([3, -7])
    scratch, output = floats([0, 0]), floats([0, 0])
    assert lib.nexa_causal_gqa_attention(q, 2, k, 2, v, 2, 2, 1, 1, 1, scratch, 2, output, 2) == 0
    assert list(output) == [3, 3]
    assert lib.nexa_swiglu(floats([-maximum, -1000, 1000]), 3, floats([maximum, 4, 0.25]), 3,
                            3, (extreme := floats([0] * 3)), 3) == 0
    equal(extreme, [0, 0, 250], "stable silu")

    for cols, group_size in ((1, 1), (5, 3), (7, 8), (33, 32), (8, 1)):
        values = [f32(rng.uniform(-7, 7)) for _ in range(cols)]
        packed = quantize_q4_row(values, group_size)
        native = (ctypes.c_uint8 * len(packed)).from_buffer_copy(packed)
        output = floats([-555] * (cols + 1))
        assert lib.nexa_q4_decode_row(native, len(packed), cols, group_size, output, cols) == 0
        equal(output[:cols], [f32(v) for v in decode_q4_row(packed, cols, group_size)], "embedding row")
        assert output[cols] == -555
        assert lib.nexa_q4_decode_row(native, len(packed) - 1, cols, group_size, output, cols) == -2
        assert lib.nexa_q4_decode_row(native, len(packed), cols, group_size, output, cols - 1) == -2
    print("Transformer kernels match independent scalar formulas, causal GQA and packed embeddings")


C_REGRESSIONS = r'''
#include "transformer.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void test_boundaries(void) {
    float x[4] = {1, 2, 3, 4}, w[4] = {1, 1, 1, 1}, out[5] = {0,0,0,0,123};
    assert(nexa_rmsnorm(x,4,w,2,2,2,1e-5,out,4) == 0);
    assert(fabs(out[0] - 1/sqrt(2.5+1e-5)) < 1e-6);
    assert(out[4] == 123);
    assert(nexa_rmsnorm(x,4,w,1,2,2,1e-5,out,4) == -2);
    assert(nexa_rmsnorm(x,3,w,2,2,2,1e-5,out,4) == -2);
    assert(nexa_rmsnorm(x,4,w,2,2,2,1e-5,out,3) == -2);
    assert(nexa_rmsnorm(x,4,w,2,0,2,1e-5,out,4) == -1);
    assert(nexa_rmsnorm(x,4,w,2,2,2,NAN,out,4) == -1);
    assert(nexa_rmsnorm(x,4,w,2,2,2,0,out,4) == -1);
    assert(nexa_rmsnorm(x,SIZE_MAX,w,2,SIZE_MAX,2,1e-5,out,4) == -3);
    assert(nexa_rmsnorm(x,4,w,2,2,2,1e-5,x,4) == -1);
    assert(nexa_rmsnorm(x,4,w,2,2,2,1e-5,w,4) == -1);
    assert(nexa_rmsnorm(NULL,4,w,2,2,2,1e-5,out,4) == -1);
    w[1] = NAN;
    assert(nexa_rmsnorm(x,4,w,2,2,2,1e-5,out,4) == -4);
    w[1] = 1;
    x[3] = INFINITY;
    assert(nexa_rmsnorm(x,4,w,2,2,2,1e-5,out,4) == -4);
    x[3] = 4;

    assert(nexa_rope(x,4,2,1,2,10000,out,4) == 0);
    assert(out[0] == 1 && out[1] == 2);
    assert(fabs(out[2] - (3*cos(1)-4*sin(1))) < 1e-6);
    assert(nexa_rope(x,4,2,1,1,10000,out,4) == -1);
    assert(nexa_rope(x,4,2,1,2,-1,out,4) == -1);
    assert(nexa_rope(x,4,2,1,2,INFINITY,out,4) == -1);
    assert(nexa_rope(x,4,2,0,2,10000,out,4) == -1);
    assert(nexa_rope(x,SIZE_MAX,2,SIZE_MAX,2,10000,out,4) == -3);
    assert(nexa_rope(x,3,2,1,2,10000,out,4) == -2);
    assert(nexa_rope(x,4,2,1,2,10000,out,3) == -2);
    assert(nexa_rope(x,4,2,1,2,10000,x,4) == -1);
    x[0] = NAN;
    assert(nexa_rope(x,4,2,1,2,10000,out,4) == -4);
    x[0] = 1;

    assert(nexa_add(x,4,w,4,4,out,4) == 0 && out[3] == 5);
    assert(nexa_swiglu(x,4,w,4,4,out,4) == 0);
    assert(nexa_add(x,3,w,4,4,out,4) == -2);
    assert(nexa_swiglu(x,4,w,3,4,out,4) == -2);
    assert(nexa_swiglu(x,4,w,4,4,out,3) == -2);
    assert(nexa_add(x,4,w,4,0,out,4) == -1);
    assert(nexa_swiglu(x,4,w,4,0,out,4) == -1);
    assert(nexa_add(x,SIZE_MAX,w,SIZE_MAX,SIZE_MAX,out,SIZE_MAX) == -3);
    assert(nexa_swiglu(x,SIZE_MAX,w,SIZE_MAX,SIZE_MAX,out,SIZE_MAX) == -3);
    assert(nexa_add(x,4,w,4,4,x,4) == -1);
    assert(nexa_swiglu(x,4,w,4,4,w,4) == -1);
    assert(nexa_add(x,3,w,3,3,x+1,3) == -1);
    x[2] = NAN;
    assert(nexa_add(x,4,w,4,4,out,4) == -4);
    assert(nexa_swiglu(x,4,w,4,4,out,4) == -4);
    assert(nexa_add(NULL,4,w,4,4,out,4) == -1);
    assert(nexa_swiglu(x,4,NULL,4,4,out,4) == -1);
    float largest[2] = {FLT_MAX, FLT_MAX}, zeros[2] = {0,0};
    assert(nexa_rmsnorm(largest,2,w,2,1,2,1e-5,out,2) == 0);
    assert(fabs(out[0] - 1) < 1e-6);
    assert(nexa_rmsnorm(zeros,2,w,2,1,2,1e-5,out,2) == 0 && out[0] == 0);
    assert(nexa_add(largest,2,largest,2,2,out,2) == -5);
    assert(nexa_swiglu(largest,2,largest,2,2,out,2) == -5);
    float rotate[4] = {1,2,FLT_MAX,FLT_MAX};
    assert(nexa_rope(rotate,4,2,1,2,10000,out,4) == -5);
    float concentrated[2] = {0,1};
    assert(nexa_rmsnorm(concentrated,2,largest,2,1,2,1e-5,out,2) == -5);
}

static void test_attention(void) {
    /* Uniform scores: heads 0,1 use KV 0, heads 2,3 use KV 1. */
    float q[8] = {0}, k[4] = {0}, v[4] = {2,10,4,14};
    float out[9] = {0,0,0,0,0,0,0,0,123}, scratch[3] = {0,0,456};
    assert(nexa_causal_gqa_attention(q,8,k,4,v,4,2,4,2,1,scratch,2,out,8) == 0);
    float expected[8] = {2,2,10,10,3,3,12,12};
    for (size_t i=0;i<8;i++) assert(out[i] == expected[i]);
    assert(scratch[2] == 456 && out[8] == 123);
    assert(nexa_causal_gqa_attention(q,8,k,4,v,4,2,3,2,1,scratch,2,out,8) == -1);
    assert(nexa_causal_gqa_attention(q,8,k,4,v,4,0,4,2,1,scratch,2,out,8) == -1);
    assert(nexa_causal_gqa_attention(q,8,k,4,v,4,2,4,0,1,scratch,2,out,8) == -1);
    assert(nexa_causal_gqa_attention(q,8,k,4,v,4,2,4,2,0,scratch,2,out,8) == -1);
    assert(nexa_causal_gqa_attention(q,SIZE_MAX,k,4,v,4,SIZE_MAX,4,2,1,scratch,2,out,8) == -3);
    assert(nexa_causal_gqa_attention(q,7,k,4,v,4,2,4,2,1,scratch,2,out,8) == -2);
    assert(nexa_causal_gqa_attention(q,8,k,3,v,4,2,4,2,1,scratch,2,out,8) == -2);
    assert(nexa_causal_gqa_attention(q,8,k,4,v,3,2,4,2,1,scratch,2,out,8) == -2);
    assert(nexa_causal_gqa_attention(q,8,k,4,v,4,2,4,2,1,scratch,1,out,8) == -2);
    assert(nexa_causal_gqa_attention(q,8,k,4,v,4,2,4,2,1,scratch,2,out,7) == -2);
    assert(nexa_causal_gqa_attention(q,8,k,4,v,4,2,4,2,1,scratch,2,q,8) == -1);
    assert(nexa_causal_gqa_attention(q,8,k,4,v,4,2,4,2,1,q,2,out,8) == -1);
    assert(nexa_causal_gqa_attention(q,8,k,4,v,4,2,4,2,1,k,2,out,8) == -1);
    assert(nexa_causal_gqa_attention(q,8,k,4,v,4,2,4,2,1,v,2,out,8) == -1);
    assert(nexa_causal_gqa_attention(q,8,k,4,v,4,2,4,2,1,out+1,2,out,8) == -1);
    assert(nexa_causal_gqa_attention(q,8,k,4,v,4,2,4,2,1,NULL,2,out,8) == -1);
    q[7] = INFINITY;
    assert(nexa_causal_gqa_attention(q,8,k,4,v,4,2,4,2,1,scratch,2,out,8) == -4);
    q[7] = 0; k[3] = NAN;
    assert(nexa_causal_gqa_attention(q,8,k,4,v,4,2,4,2,1,scratch,2,out,8) == -4);
    k[3] = 0; v[3] = NAN;
    assert(nexa_causal_gqa_attention(q,8,k,4,v,4,2,4,2,1,scratch,2,out,8) == -4);
    /* No raw float32 scores: these dot products exceed FLT_MAX. */
    float largeq[2] = {FLT_MAX,FLT_MAX}, largek[2] = {FLT_MAX,-FLT_MAX}, value[2] = {3,-7};
    assert(nexa_causal_gqa_attention(largeq,2,largek,2,value,2,2,1,1,1,scratch,2,out,2) == 0);
    assert(out[0] == 3 && out[1] == 3);
}

static void test_packed_embedding(void) {
    /* scale1, values7,1,-1,2,-2 and the unused high nibble zero. */
    uint8_t packed[7] = {0,0,128,63,0x17,0x2f,0x0e};
    float out[6] = {0,0,0,0,0,321};
    assert(nexa_q4_decode_row(packed,7,5,5,out,5) == 0);
    float expected[5] = {7,1,-1,2,-2};
    for(size_t i=0;i<5;i++) assert(out[i] == expected[i]);
    assert(out[5] == 321);
    assert(nexa_q4_decode_row(packed,6,5,5,out,5) == -2);
    assert(nexa_q4_decode_row(packed,7,5,5,out,4) == -2);
    assert(nexa_q4_decode_row(packed,7,0,5,out,5) == -1);
    assert(nexa_q4_decode_row(packed,7,5,0,out,5) == -1);
    assert(nexa_q4_decode_row(packed,SIZE_MAX,SIZE_MAX,1,out,SIZE_MAX) == -3);
    assert(nexa_q4_decode_row(NULL,7,5,5,out,5) == -1);
    assert(nexa_q4_decode_row(packed,7,5,5,NULL,5) == -1);
    assert(nexa_q4_decode_row(packed,7,5,5,(float*)packed,5) == -1);
    packed[4] = 0x18;
    assert(nexa_q4_decode_row(packed,7,5,5,out,5) == -4);
    packed[4] = 0x17; packed[6] = 0x1e;
    assert(nexa_q4_decode_row(packed,7,5,5,out,5) == -4);
    packed[6] = 0x0e; packed[3] = 0xbf;
    assert(nexa_q4_decode_row(packed,7,5,5,out,5) == -4);
    packed[3] = 0x7f; packed[2] = 0xc0;
    assert(nexa_q4_decode_row(packed,7,5,5,out,5) == -4);
    /* Finite scale whose product cannot be represented as output float32. */
    uint8_t overflow[5] = {0xff,0xff,0x7f,0x7f,7};
    assert(nexa_q4_decode_row(overflow,5,1,1,out,1) == -5);
}

int main(void) {
    test_boundaries(); test_attention(); test_packed_embedding();
    puts("Transformer kernel sanitizer regressions passed");
    return 0;
}
'''


class TransformerKernelRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("clang") or shutil.which("cc")
        if not cls.compiler:
            raise unittest.SkipTest("C compiler unavailable")

    def test_c_sanitizers_and_no_heap_dependencies(self):
        with tempfile.TemporaryDirectory(prefix="nexa-transformer-native-") as directory:
            directory = Path(directory)
            source = directory / "regressions.c"
            source.write_text(C_REGRESSIONS)
            binary = directory / ("kernels.exe" if os.name == "nt" else "kernels")
            command = [self.compiler, "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                       "-I", str(SOURCE), str(SOURCE / "q4.c"), str(SOURCE / "transformer.c"),
                       str(source), "-o", str(binary)]
            command.extend(f"-D{name}=nexa_transformer_forbidden_{name}" for name in ("malloc", "calloc", "realloc", "free"))
            if os.name != "nt":
                command.extend(["-lm", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"])
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            env = dict(os.environ)
            env["ASAN_OPTIONS"] = "detect_leaks=0" if sys.platform == "darwin" else "detect_leaks=1"
            env["UBSAN_OPTIONS"] = "halt_on_error=1"
            result = subprocess.run([str(binary)], capture_output=True, text=True, env=env, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_scalar_formulas_causal_gqa_and_embedding_reference(self):
        with tempfile.TemporaryDirectory(prefix="nexa-transformer-reference-") as directory:
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
