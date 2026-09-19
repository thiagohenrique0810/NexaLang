"""Direct packed-Q3 KV attention without a decoded cache or head-sized vector."""
import ctypes
import math
import struct
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

# Deliberately independent from runtime codecs and the end-to-end Q3 oracle.
# Packing uses a Python integer bitstream rather than the C byte operations.
def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def floats(values):
    return (ctypes.c_float * len(values))(*values)


def equal(actual, expected, label):
    assert len(actual) == len(expected), label
    for index, (left, right) in enumerate(zip(actual, expected)):
        assert math.isclose(left, right, rel_tol=2e-6, abs_tol=2e-6), (label, index, left, right)


def reference_pack(values, rows, cols, group):
    output, decoded_rows = bytearray(), []
    code_bytes = (3 * group + 7) // 8
    for row in range(rows):
        decoded = []
        for start in range(0, cols, group):
            part = [f32(v) for v in values[row * cols + start:row * cols + min(start + group, cols)]]
            maximum = max(map(abs, part))
            scale = f32(maximum / 3)
            if maximum and not scale:
                raise ValueError("scale underflow")
            codes = []
            for value in part:
                ratio = value / scale if scale else 0
                magnitude = math.floor(abs(ratio) + 0.5)
                codes.append(max(-3, min(3, -magnitude if ratio < 0 else magnitude)))
            bits = sum((code & 7) << (3 * lane) for lane, code in enumerate(codes))
            output += struct.pack("<f", scale) + bits.to_bytes(code_bytes, "little")
            # Multiplication stays double: no artificial F32 dequantization.
            decoded.extend(scale * code for code in codes)
        decoded_rows.append(decoded)
    return bytes(output), decoded_rows


def attention_oracle(q, k, v, sequence, qheads, kvheads, dim):
    output = []
    for position in range(sequence):
        for head in range(qheads):
            kh = head // (qheads // kvheads)
            scores = []
            for past in range(position + 1):
                dot = 0.0
                for lane in range(dim):
                    dot += q[(position * qheads + head) * dim + lane] * k[(past * kvheads + kh) * dim + lane]
                scores.append(dot * (1.0 / math.sqrt(dim)))
            maximum = max(scores)
            weights = [f32(math.exp(score - maximum)) for score in scores]
            denominator = sum(weights)
            for lane in range(dim):
                weighted = 0.0
                for past, weight in enumerate(weights):
                    weighted += weight * v[(past * kvheads + kh) * dim + lane]
                output.append(f32(weighted / denominator))
    return output


def bind(library):
    lib = ctypes.CDLL(str(library))
    fp, bp, sz = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
    table = ctypes.POINTER(bp)
    lib.nexa_q3_quantize.argtypes = [fp, sz, sz, sz, sz, bp, sz]
    lib.nexa_q3_quantize.restype = ctypes.c_int
    lib.nexa_q3_row_size.argtypes = [sz, sz]
    lib.nexa_q3_row_size.restype = sz
    lib.nexa_causal_gqa_attention_paged_q3.argtypes = [fp, sz, table, sz, table, sz,
                                                     sz, sz, sz, sz, sz, sz, sz, sz, fp, sz, fp, sz]
    lib.nexa_causal_gqa_attention_paged_q3.restype = ctypes.c_int
    return lib


def check_encoder(lib):
    rng = random.Random(403)
    for group in range(1, 10):
        for cols in range(1, 2 * group + 2):
            values = [f32(rng.uniform(-1e3, 1e3)) for _ in range(cols)] + [0.0] * cols
            expected, _ = reference_pack(values, 2, cols, group)
            row_bytes = ((cols + group - 1) // group) * (4 + (3 * group + 7) // 8)
            assert lib.nexa_q3_row_size(cols, group) == row_bytes
            owner = (ctypes.c_uint8 * (len(expected) + 2))(*([91] * (len(expected) + 2)))
            packed = ctypes.cast(ctypes.byref(owner, 1), ctypes.POINTER(ctypes.c_uint8))
            assert lib.nexa_q3_quantize(floats(values), len(values), 2, cols, group, packed, len(expected)) == 0
            assert bytes(owner)[1:-1] == expected, (cols, group)
            assert owner[0] == 91 and owner[-1] == 91
    # The first group fixes scale=1, exercises every valid code plus both half ties.
    values = [3, -3, 2, -2, 1, -1, 0, 0.5, -0.5]
    expected, decoded = reference_pack(values, 1, 9, 9)
    assert decoded == [[3, -3, 2, -2, 1, -1, 0, 1, -1]]
    packed = (ctypes.c_uint8 * len(expected))()
    assert lib.nexa_q3_quantize(floats(values), 9, 1, 9, 9, packed, len(expected)) == 0
    assert bytes(packed) == expected
    # Subnormal scale survives if representable; input values are finite F32.
    for values in ([3 * 2**-149], [f32(3.4028234663852886e38), 0.0]):
        expected, _ = reference_pack(values, 1, len(values), 2)
        packed = (ctypes.c_uint8 * len(expected))()
        assert lib.nexa_q3_quantize(floats(values), len(values), 1, len(values), 2, packed, len(expected)) == 0
        assert bytes(packed) == expected


def ctypes_probe(library):
    lib = bind(library)
    rng = random.Random(402)
    bp = ctypes.POINTER(ctypes.c_uint8)
    sequence = 7
    check_encoder(lib)
    # All bit offsets, byte crossings, odd tails, G>D, and unaligned row scales.
    cases = [(qh, kh, dim, group) for group in range(1, 10)
             for qh, kh, dim in ((4, 1, 6), (4, 2, 5), (2, 2, 7), (1, 1, 1))]
    for query_heads, kv_heads, dim, group in cases:
        qwidth, kvwidth = query_heads * dim, kv_heads * dim
        query = [f32(rng.uniform(-2, 2)) for _ in range(sequence * qwidth)]
        key = [f32(rng.uniform(-4, 4)) for _ in range(sequence * kvwidth)]
        value = [f32(rng.uniform(-5, 5)) for _ in range(sequence * kvwidth)]
        expected_key, decoded_key = reference_pack(key, sequence * kv_heads, dim, group)
        expected_value, decoded_value = reference_pack(value, sequence * kv_heads, dim, group)
        flat_key = [value for row in decoded_key for value in row]
        flat_value = [value for row in decoded_value for value in row]
        reference = attention_oracle(query, flat_key, flat_value, sequence, query_heads, kv_heads, dim)
        row_bytes = lib.nexa_q3_row_size(dim, group)
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
                        assert lib.nexa_q3_quantize(floats(part), len(part), current * kv_heads, dim, group,
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
                    status = lib.nexa_causal_gqa_attention_paged_q3(q, len(q), table_k, len(table_k), table_v, len(table_v),
                        page_tokens, page_bytes, group, past, current, query_heads, kv_heads, dim,
                        scratch, end, output, len(q))
                    assert status == 0, (status, page_tokens, chunk_size, past, dim, group)
                    equal(output[:-1], reference[past * qwidth:end * qwidth], "Q3 KV scalar reference")
                    assert [bytes(owner) for owner in owners_k + owners_v] == before
                    assert (bytes(table_k), bytes(table_v)) == before_tables
                    assert output[-1] == 123 and scratch[-1] == 321
                    collected.extend(output[:-1])
                if full_native is None:
                    full_native = collected
                assert collected == full_native, "paging/chunking changed native packed-Q3 arithmetic"
                assert all(owner[0] == 0xA5 and owner[-1] == 0x5A for owner in owners_k + owners_v)
    print("Packed Q3 KV matches independent codec/attention oracle across heads, odd groups, tails, pages and chunks")


C_REGRESSIONS = r'''
#include "transformer.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>


static void test_quantizer(void) {
    float values[40]; uint8_t packed[256];
    for (size_t i=0;i<40;i++) values[i]=(float)((int)(i%7)-3);
    for (size_t group=1;group<=9;group++) {
        for (size_t cols=1;cols<=19;cols++) {
            size_t bytes=2*nexa_q3_row_size(cols,group);
            assert(bytes>0 && bytes<sizeof(packed));
            memset(packed,0xa5,sizeof(packed));
            assert(nexa_q3_quantize(values,2*cols,2,cols,group,packed,bytes)==0);
            assert(packed[bytes]==0xa5);
            /* Exercise decoder byte boundaries under ASan for every G/column tail. */
            const uint8_t *tables[1]={packed}; float q[40]={0},scratch[2],out[40];
            assert(nexa_causal_gqa_attention_paged_q3(q,2*cols,tables,1,tables,1,
                2,bytes,group,0,2,1,1,cols,scratch,2,out,2*cols)==0);
        }
    }
    assert(nexa_q3_row_size(0,1)==0 && nexa_q3_row_size(1,0)==0);
    assert(nexa_q3_row_size(1,SIZE_MAX/3+1)==0);
    assert(nexa_q3_row_size(SIZE_MAX,1)==0);
    assert(nexa_q3_quantize(NULL,1,1,1,1,packed,5)==-1);
    assert(nexa_q3_quantize(values,1,1,1,1,NULL,5)==-1);
    assert(nexa_q3_quantize(values,1,0,1,1,packed,5)==-1);
    assert(nexa_q3_quantize(values,1,1,0,1,packed,5)==-1);
    assert(nexa_q3_quantize(values,1,1,1,0,packed,5)==-1);
    assert(nexa_q3_quantize(values,0,1,1,1,packed,5)==-2);
    assert(nexa_q3_quantize(values,1,1,1,1,packed,4)==-2);
    assert(nexa_q3_quantize(values,1,1,1,SIZE_MAX/3+1,packed,SIZE_MAX)==-3);
    assert(nexa_q3_quantize(values,SIZE_MAX,SIZE_MAX,1,1,packed,SIZE_MAX)==-3);
    assert(nexa_q3_quantize(values,SIZE_MAX,1,SIZE_MAX,1,packed,SIZE_MAX)==-3);
    assert(nexa_q3_quantize(values,1,1,1,1,(uint8_t*)values,5)==-1);
    assert(nexa_q3_quantize((const float*)(uintptr_t)(UINTPTR_MAX-1),1,1,1,1,packed,5)==-1);
    assert(nexa_q3_quantize(values,1,1,1,1,(uint8_t*)(uintptr_t)(UINTPTR_MAX-1),5)==-1);
    values[0]=NAN; assert(nexa_q3_quantize(values,1,1,1,1,packed,5)==-4);
    values[0]=INFINITY; assert(nexa_q3_quantize(values,1,1,1,1,packed,5)==-4);
    values[0]=ldexpf(1,-149); assert(nexa_q3_quantize(values,1,1,1,1,packed,5)==-5);
    values[0]=3*ldexpf(1,-149); assert(nexa_q3_quantize(values,1,1,1,1,packed,5)==0);
    assert(packed[0]==1 && packed[1]==0 && packed[2]==0 && packed[3]==0 && packed[4]==3);
    values[0]=FLT_MAX; assert(nexa_q3_quantize(values,1,1,1,1,packed,5)==0);
    /* Existing Q4 stays available and retains its exact low-nibble codec. */
    float oldvalues[5]={7,1,-1,2,-2}; uint8_t oldpacked[7];
    const uint8_t expected[7]={0,0,128,63,0x17,0x2f,0x0e};
    assert(nexa_q4_quantize(oldvalues,5,1,5,5,oldpacked,7)==0);
    assert(memcmp(oldpacked,expected,7)==0);
}

static void test_codec_and_boundaries(void) {
    /* D5/G3: two six-byte groups per head; P2:24 bytes per page. */
    float q[20]={0}, keys_f32[15]={0};
    float values_f32[15]={3,0,-3,1,-1, 6,0,-6,2,-2, 9,0,-9,3,-3};
    uint8_t k[48],v[48]; memset(k,255,sizeof(k)); memset(v,255,sizeof(v));
    assert(nexa_q3_row_size(5,3)==12);
    assert(nexa_q3_quantize(keys_f32,15,3,5,3,k,36)==0);
    assert(nexa_q3_quantize(values_f32,15,3,5,3,v,36)==0);
    const uint8_t *kt[3]={k,k+24,NULL}, *vt[3]={v,v+24,NULL};
    float out[21]={0}, scratch[4]={0}; out[20]=987; scratch[3]=654;
    #define VALID nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,out,20)
    assert(VALID==0);
    assert(out[0]==4.5f && out[5]==4.5f && out[10]==6 && out[15]==6);
    assert(out[20]==987 && scratch[3]==654);
    uint8_t before_k[48],before_v[48]; memcpy(before_k,k,48); memcpy(before_v,v,48);
    assert(VALID==0 && memcmp(before_k,k,48)==0 && memcmp(before_v,v,48)==0);

    /* Reserved code, negative/nonfinite scale, zero-scale code, odd padding. */
    k[4]=4; assert(VALID==-4); k[4]=0;
    k[2]=128;k[3]=191; assert(VALID==-4); k[2]=0;k[3]=0;
    k[2]=192;k[3]=127; assert(VALID==-4); k[2]=0;k[3]=0;
    k[2]=128;k[3]=127; assert(VALID==-4); k[2]=0;k[3]=0;
    k[4]=1; assert(VALID==-4); k[4]=0;
    k[2]=128;k[3]=63;k[5]=2; assert(VALID==-4); k[2]=0;k[3]=0;k[5]=0;
    k[8]=128;k[9]=63;k[10]=64; assert(VALID==-4); k[8]=0;k[9]=0;k[10]=0;
    v[26]=192;v[27]=127; assert(VALID==-4); memcpy(v,before_v,48);
    q[19]=NAN; assert(VALID==-4); q[19]=0;
    assert(VALID==0); /* Unused fourth token is all 0xff and is never inspected. */

    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,1,vt,3,2,24,3,1,2,2,1,5,scratch,3,out,20)==-2);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,1,2,24,3,1,2,2,1,5,scratch,3,out,20)==-2);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,23,3,1,2,2,1,5,scratch,3,out,20)==-2);
    assert(nexa_causal_gqa_attention_paged_q3(q,19,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,out,20)==-2);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,2,out,20)==-2);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,out,19)==-2);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,0,24,3,1,2,2,1,5,scratch,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,0,1,2,2,1,5,scratch,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,3,1,0,2,1,5,scratch,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,3,1,2,3,2,5,scratch,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,3,1,2,2,0,5,scratch,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,3,1,2,2,1,0,scratch,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,3,SIZE_MAX,1,2,1,5,scratch,3,out,20)==-3);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,SIZE_MAX,SIZE_MAX/3+1,1,2,2,1,5,scratch,3,out,20)==-3);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,SIZE_MAX,SIZE_MAX,3,1,2,2,1,5,scratch,3,out,20)==-3);
    assert(nexa_causal_gqa_attention_paged_q3(q,SIZE_MAX,kt,3,vt,3,1,SIZE_MAX,1,0,1,1,1,SIZE_MAX,scratch,3,out,SIZE_MAX)==-3);
    assert(nexa_causal_gqa_attention_paged_q3(q,SIZE_MAX,kt,3,vt,3,1,SIZE_MAX,1,0,1,SIZE_MAX,1,2,scratch,3,out,SIZE_MAX)==-3);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,SIZE_MAX,vt,SIZE_MAX,1,5,1,SIZE_MAX/4,1,1,1,1,scratch,SIZE_MAX,out,20)==-3);

    assert(nexa_causal_gqa_attention_paged_q3(NULL,20,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,NULL,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,NULL,3,2,24,3,1,2,2,1,5,scratch,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,NULL,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,NULL,20)==-1);
    kt[1]=NULL; assert(VALID==-1); kt[1]=k+24;
    vt[0]=NULL; assert(VALID==-1); vt[0]=v;
    kt[1]=(const uint8_t *)(uintptr_t)(UINTPTR_MAX-1); assert(VALID==-1); kt[1]=k+24;
    assert(nexa_causal_gqa_attention_paged_q3(q,20,(const uint8_t*const*)(uintptr_t)(UINTPTR_MAX-1),3,vt,3,
        2,24,3,1,2,2,1,5,scratch,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3((const float*)(uintptr_t)(UINTPTR_MAX-1),20,kt,3,vt,3,
        2,24,3,1,2,2,1,5,scratch,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,q,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,(float*)k,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,(float*)v,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,q,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,(float*)k,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,(float*)v,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,out+1,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,(float*)kt,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,scratch,3,(float*)vt,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,(float*)kt,3,out,20)==-1);
    assert(nexa_causal_gqa_attention_paged_q3(q,20,kt,3,vt,3,2,24,3,1,2,2,1,5,(float*)vt,3,out,20)==-1);
    #undef VALID
}

static void test_reserved_codes_and_padding(void) {
    /* Positive scales isolate reserved/padding checks from zero-scale checks.
     * Every lane position exercises all eight possible bit offsets. */
    for (size_t group=1;group<=9;group++) {
        size_t cols=group+1, group_bytes=4+(3*group+7)/8;
        size_t bytes=nexa_q3_row_size(cols,group);
        uint8_t key[16]={0},value[16]={0};
        key[2]=128;key[3]=63;key[group_bytes+2]=128;key[group_bytes+3]=63;
        const uint8_t *kt[1]={key},*vt[1]={value};
        float q[10]={0},scratch[1],out[10];
        #define CHECK nexa_causal_gqa_attention_paged_q3(q,cols,kt,1,vt,1,1,bytes,group,0,1,1,1,cols,scratch,1,out,cols)
        assert(CHECK==0);
        for (size_t lane=0;lane<group;lane++) {
            /* Reserved -4 sets the code's most significant bit. */
            size_t bit=3*lane+2;
            key[4+bit/8]=(uint8_t)(1u<<(bit%8));
            assert(CHECK==-4); key[4+bit/8]=0;
        }
        if (group>1) {
            /* Second group has only one lane. Nonzero code in lane1 is padding. */
            key[group_bytes+4]=8;
            assert(CHECK==-4); key[group_bytes+4]=0;
        }
        unsigned int tail=(unsigned int)((3*group)%8);
        if (tail) {
            key[group_bytes-1]=(uint8_t)(1u<<tail);
            assert(CHECK==-4); key[group_bytes-1]=0;
        }
        assert(CHECK==0);
        #undef CHECK
    }
}

static void test_double_decode_and_numeric_range(void) {
    /* Uniform attention: 0.1f*3 and -0.3f cancel only if prematurely rounded. */
    uint8_t zero[10]={0};
    uint8_t values[10]={0xcd,0xcc,0xcc,0x3d,3, 0x9a,0x99,0x99,0x3e,7};
    const uint8_t *kt[1]={zero},*vt[1]={values};
    float q[1]={0},scratch[2],out[1];
    assert(nexa_causal_gqa_attention_paged_q3(q,1,kt,1,vt,1,2,10,1,1,1,1,1,1,scratch,2,out,1)==0);
    assert(out[0]==-ldexpf(1,-28));
    /* Finite scale with decoded value 3*FLT_MAX: only the final output overflows. */
    uint8_t huge[5]={0xff,0xff,0x7f,0x7f,3}; vt[0]=huge;
    assert(nexa_causal_gqa_attention_paged_q3(q,1,kt,1,vt,1,1,5,1,0,1,1,1,1,scratch,1,out,1)==-5);
    /* Scores larger than FLT_MAX remain stable because dots/max are double. */
    uint8_t largekeys[10]={0xff,0xff,0x7f,0x7f,3, 0xff,0xff,0x7f,0x7f,5};
    uint8_t finitevalues[10]={0,0,128,63,3, 0,0,128,63,5};
    kt[0]=largekeys;vt[0]=finitevalues;q[0]=FLT_MAX;
    assert(nexa_causal_gqa_attention_paged_q3(q,1,kt,1,vt,1,2,10,1,1,1,1,1,1,scratch,2,out,1)==0);
    assert(out[0]==3);
}

int main(void) {
    test_quantizer(); test_codec_and_boundaries(); test_double_decode_and_numeric_range();
    test_reserved_codes_and_padding();
    puts("Packed Q3 KV sanitizer regressions passed");
    return 0;
}
'''


class Q3KVKernelRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("clang") or shutil.which("cc")
        if not cls.compiler:
            raise unittest.SkipTest("C compiler unavailable")

    def test_c_sanitizers_codec_boundaries_and_no_heap(self):
        with tempfile.TemporaryDirectory(prefix="nexa-q3-kv-native-") as directory:
            directory = Path(directory)
            source = directory / "regressions.c"
            source.write_text(C_REGRESSIONS)
            binary = directory / ("kernels.exe" if os.name == "nt" else "kernels")
            command = [self.compiler, "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                       "-I", str(SOURCE), str(SOURCE / "q4.c"), str(SOURCE / "transformer.c"),
                       str(source), "-o", str(binary)]
            command.extend(f"-D{name}=nexa_q3_kv_forbidden_{name}" for name in ("malloc", "calloc", "realloc", "free"))
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
        with tempfile.TemporaryDirectory(prefix="nexa-q3-kv-reference-") as directory:
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
