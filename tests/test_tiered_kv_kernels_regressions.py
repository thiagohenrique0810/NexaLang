"""Mixed F32/Q4/Q3 attention and bounded one-head transcoding, independent oracle."""
import ctypes
import itertools
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime.build_runtime import build_runtime


def f32(value):
    return struct.unpack('=f', struct.pack('=f', value))[0]


def pack(values, codec, group):
    """Integer-bitstream oracle; never calls production codecs."""
    values = list(map(f32, values))
    if codec == 0:
        return struct.pack('=' + 'f' * len(values), *values), values
    result, decoded = bytearray(), []
    maximum_code = (1 << (codec - 1)) - 1
    for start in range(0, len(values), group):
        part = values[start:start + group]
        scale = f32(max(map(abs, part)) / maximum_code)
        codes = []
        for value in part:
            ratio = value / scale if scale else 0.0
            magnitude = math.floor(abs(ratio) + 0.5)
            codes.append(max(-maximum_code, min(maximum_code, -magnitude if ratio < 0 else magnitude)))
        packed = sum((code & ((1 << codec) - 1)) << (codec * i) for i, code in enumerate(codes))
        result += struct.pack('<f', scale) + packed.to_bytes((group * codec + 7) // 8, 'little')
        decoded.extend(scale * code for code in codes)
    return bytes(result), decoded


def oracle(query, keys, values, past, sequence, heads, kv_heads, dim):
    result = []
    for pos in range(sequence):
        for head in range(heads):
            kh = head // (heads // kv_heads)
            q = query[(pos * heads + head) * dim:(pos * heads + head + 1) * dim]
            scores = [sum(a * b for a, b in zip(q, keys[t * kv_heads + kh])) / math.sqrt(dim)
                      for t in range(past + pos + 1)]
            maximum = max(scores)
            weights = [f32(math.exp(score - maximum)) for score in scores]
            for lane in range(dim):
                result.append(f32(sum(w * values[t * kv_heads + kh][lane]
                                      for t, w in enumerate(weights)) / sum(weights)))
    return result


class TieredKVNumericsRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='nexa-tiered-numerics-')
        cls.addClassCleanup(cls.directory.cleanup)
        cls.lib = ctypes.CDLL(str(build_runtime('nexa_transformer', cls.directory.name)))
        fp, bp, sp, dp, sz = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double), ctypes.c_size_t
        cls.lib.nexa_causal_gqa_attention_paged_mixed.argtypes = [fp, sz, ctypes.POINTER(bp), sz, ctypes.POINTER(bp), sz,
            bp, sz, sp, sz, sz, sz, sz, sz, sz, sz, sz, fp, sz, fp, sz]
        cls.lib.nexa_causal_gqa_attention_paged_mixed.restype = ctypes.c_int
        cls.lib.nexa_kv_reencode_rows.argtypes = [bp, sz, ctypes.c_int, ctypes.c_int, sz, sz, sz, bp, sz, fp, sz, dp, sz]
        cls.lib.nexa_kv_reencode_rows.restype = ctypes.c_int

    def exercise_attention(self, layouts):
        bp = ctypes.POINTER(ctypes.c_uint8)
        for dim, group, heads, kv_heads, page_tokens in layouts:
            tokens = 3 * page_tokens - 1 if page_tokens > 1 else 3
            source = [[f32(((i * 19 + lane * 11 + bank * 7) % 53 - 26) / 8)
                       for lane in range(dim)] for bank in (0, 1) for i in range(tokens * kv_heads)]
            for order in itertools.permutations((0, 3, 4)):
                codecs = (ctypes.c_uint8 * 4)(*order, 99)
                capacities = (ctypes.c_size_t * 4)()
                owners, tables, decoded = [], [], []
                for bank in range(2):
                    pointers, rows = [], []
                    for page, codec in enumerate(order):
                        first, end = page * page_tokens, min(tokens, (page + 1) * page_tokens)
                        encoded = [pack(source[bank * tokens * kv_heads + token * kv_heads + head], codec, group)
                                   for token in range(first, end) for head in range(kv_heads)]
                        data = b''.join(record for record, _ in encoded)
                        rows.extend(row for _, row in encoded)
                        capacity = page_tokens * kv_heads * len(encoded[0][0])
                        capacities[page] = capacity
                        owner = (ctypes.c_uint8 * (capacity + 2))(*([203] * (capacity + 2)))
                        ctypes.memmove(ctypes.byref(owner, 1), data, len(data))
                        owners.append(owner)
                        pointers.append(ctypes.cast(ctypes.byref(owner, 1), bp))
                    pointers.append(ctypes.cast(ctypes.c_void_p(9), bp))
                    tables.append((bp * 4)(*pointers))
                    decoded.append(rows)
                original = [bytes(owner) for owner in owners]
                for past, count in ((0, tokens), (tokens - 1, 1), (0, 1), (1, tokens - 1)):
                    q = [f32(((i * 13 + 5) % 37 - 18) / 9) for i in range(count * heads * dim)]
                    query = (ctypes.c_float * len(q))(*q)
                    scores = (ctypes.c_float * (past + count + 1))(*([77.0] * (past + count + 1)))
                    out = (ctypes.c_float * (len(q) + 1))(*([77.0] * (len(q) + 1)))
                    status = self.lib.nexa_causal_gqa_attention_paged_mixed(query, len(q), tables[0], 4, tables[1], 4,
                        codecs, 4, capacities, 4, page_tokens, group, past, count, heads, kv_heads, dim,
                        scores, len(scores), out, len(out))
                    self.assertEqual(status, 0)
                    expected = oracle(q, *decoded, past, count, heads, kv_heads, dim)
                    for actual, wanted in zip(out, expected):
                        self.assertTrue(math.isclose(actual, wanted, abs_tol=3e-6, rel_tol=3e-6), (dim, group, order, actual, wanted))
                    self.assertEqual((scores[-1], out[-1]), (77.0, 77.0))
                    self.assertEqual([bytes(owner) for owner in owners], original)

    def test_mixed_page_orders_gqa_mqa_mha_and_causal_chunks(self):
        self.exercise_attention(((7, 3, 4, 2, 2), (8, 8, 4, 1, 3), (3, 5, 3, 3, 1)))

    def test_mixed_single_lane_and_large_odd_group(self):
        self.exercise_attention(((1, 1, 2, 1, 2), (17, 9, 4, 2, 2)))

    def test_all_transcode_pairs_identity_and_exact_delta_statistics(self):
        bp = ctypes.POINTER(ctypes.c_uint8)
        for dim, group in ((1, 1), (7, 3), (8, 9), (64, 32)):
            rows = [[f32(((row * 17 + lane * 7) % 29 - 14) / 6) for lane in range(dim)] for row in range(3)]
            rows.append([0.0] * dim)
            for source_codec, target_codec in itertools.product((0, 3, 4), repeat=2):
                encoded = [pack(row, source_codec, group) for row in rows]
                input_data = b''.join(data for data, _ in encoded)
                expected_rows = [pack([f32(value) for value in decoded], target_codec, group)
                                 for _, decoded in encoded] if source_codec != target_codec else encoded
                expected = b''.join(data for data, _ in expected_rows)
                input_owner = (ctypes.c_uint8 * (len(input_data) + 2))(*([31] * (len(input_data) + 2)))
                ctypes.memmove(ctypes.byref(input_owner, 1), input_data, len(input_data))
                output_owner = (ctypes.c_uint8 * (len(expected) + 2))(*([41] * (len(expected) + 2)))
                scratch = (ctypes.c_float * (dim + 1))(*([77.0] * (dim + 1)))
                stats = (ctypes.c_double * 4)(-1, -1, -1, 77)
                status = self.lib.nexa_kv_reencode_rows(ctypes.cast(ctypes.byref(input_owner, 1), bp), len(input_data),
                    source_codec, target_codec, len(rows), dim, group,
                    ctypes.cast(ctypes.byref(output_owner, 1), bp), len(expected), scratch, len(scratch), stats, len(stats))
                self.assertEqual(status, 0)
                self.assertEqual(bytes(output_owner)[1:-1], expected, (dim, group, source_codec, target_codec))
                errors = [a - b for (_, before), (_, after) in zip(encoded, expected_rows) for a, b in zip(before, after)]
                self.assertEqual(stats[0], max(map(abs, errors)))
                self.assertTrue(math.isclose(stats[1], sum(error * error for error in errors), rel_tol=2e-14, abs_tol=1e-28))
                self.assertEqual(stats[2], dim * len(rows))
                self.assertEqual((input_owner[0], input_owner[-1], output_owner[0], output_owner[-1], scratch[-1], stats[-1]),
                                 (31, 31, 41, 41, 77, 77))
                self.assertEqual(bytes(input_owner)[1:-1], input_data)


HARNESS = r'''
#include "transformer.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
/* All codec/kernel translation units redirect allocator symbols here. */
void *nexa_forbidden_malloc(size_t n) { (void)n; abort(); }
void *nexa_forbidden_calloc(size_t n, size_t s) { (void)n; (void)s; abort(); }
void *nexa_forbidden_realloc(void *p, size_t n) { (void)p; (void)n; abort(); }
void nexa_forbidden_free(void *p) { (void)p; abort(); }

typedef struct {
    const float *q;
    const uint8_t *const *kt, *const *vt;
    const uint8_t *codecs;
    const size_t *bytes;
    size_t qc, kc, vc, cc, bc, pt, group, past, seq, heads, kvheads, dim;
    float *scores, *out;
    size_t sc, oc;
} attention_args;
static int attend(attention_args a) {
    return nexa_causal_gqa_attention_paged_mixed(a.q,a.qc,a.kt,a.kc,a.vt,a.vc,a.codecs,a.cc,a.bytes,a.bc,
        a.pt,a.group,a.past,a.seq,a.heads,a.kvheads,a.dim,a.scores,a.sc,a.out,a.oc);
}
static void attention_errors(int mode) {
    float source[4]={1,-2,3,-4}, query[8]={1,-2,3,-4,5,-6,7,-8}, output[8], scores[3];
    uint8_t k[16], v[16]; memset(k,203,sizeof k); memset(v,203,sizeof v);
    assert(nexa_q3_quantize(source,2,1,2,3,k,6)==0);
    memcpy(v,k,6);
    const uint8_t *kt[3]={(const uint8_t*)source,k,(const uint8_t*)9};
    const uint8_t *vt[3]={(const uint8_t*)source,v,(const uint8_t*)9};
    uint8_t codecs[3]={0,3,255}; size_t capacities[3]={16,12,0};
    attention_args base={query,kt,vt,codecs,capacities,8,3,3,3,3,2,3,1,2,2,1,2,scores,output,3,8}, a;
    assert(attend(base)==0);
#define BAD(field,value,want) do {a=base;a.field=value;assert(attend(a)==want);} while(0)
    if(mode==0) {
        BAD(qc,7,-2); BAD(kc,1,-2); BAD(vc,1,-2); BAD(cc,1,-2); BAD(bc,1,-2);
        BAD(sc,2,-2); BAD(oc,7,-2);
        BAD(q,NULL,-1); BAD(kt,NULL,-1); BAD(vt,NULL,-1); BAD(codecs,NULL,-1); BAD(bytes,NULL,-1);
        BAD(scores,NULL,-1); BAD(out,NULL,-1); BAD(pt,0,-1); BAD(group,0,-1); BAD(seq,0,-1);
        BAD(heads,0,-1); BAD(kvheads,0,-1); BAD(dim,0,-1); BAD(kvheads,3,-1);
        BAD(past,SIZE_MAX,-3); BAD(pt,SIZE_MAX,-3); BAD(heads,SIZE_MAX,-3);
        BAD(dim,SIZE_MAX,-3); BAD(group,SIZE_MAX,-3);
        capacities[1]=11; assert(attend(base)==-2); capacities[1]=12;
        codecs[1]=1; assert(attend(base)==-1); codecs[1]=3;
        kt[1]=NULL; assert(attend(base)==-1); kt[1]=k;
        vt[1]=NULL; assert(attend(base)==-1); vt[1]=v;
    } else if(mode==1) {
        BAD(out,(float*)query,-1); BAD(scores,(float*)query,-1); BAD(out,scores,-1);
        BAD(out,(float*)kt,-1); BAD(scores,(float*)kt,-1); BAD(out,(float*)vt,-1); BAD(scores,(float*)vt,-1);
        BAD(out,(float*)codecs,-1); BAD(scores,(float*)codecs,-1);
        BAD(out,(float*)capacities,-1); BAD(scores,(float*)capacities,-1);
        BAD(out,(float*)source,-1); BAD(scores,(float*)source,-1); BAD(out,(float*)k,-1); BAD(scores,(float*)v,-1);
        BAD(q,(float*)((uint8_t*)query+1),-1); BAD(scores,(float*)((uint8_t*)scores+1),-1);
        BAD(out,(float*)((uint8_t*)output+1),-1);
        BAD(kt,(const uint8_t*const*)((uint8_t*)kt+1),-1);
        BAD(bytes,(const size_t*)((uint8_t*)capacities+1),-1);
        BAD(q,(const float*)(uintptr_t)(UINTPTR_MAX-3),-1);
        BAD(kt,(const uint8_t*const*)(uintptr_t)(UINTPTR_MAX-7),-1);
        BAD(bytes,(const size_t*)(uintptr_t)(UINTPTR_MAX-7),-1);
        BAD(codecs,(const uint8_t*)(uintptr_t)UINTPTR_MAX,-1);
        kt[1]=(const uint8_t*)(uintptr_t)(UINTPTR_MAX-2); assert(attend(base)==-1); kt[1]=k;
    } else {
        for(int i=0;i<8;i++) output[i]=91;
        v[4]=(uint8_t)((v[4]&~7u)|4u); assert(attend(base)==-4); /* reserved */
        for(int i=0;i<8;i++) assert(output[i]==91);
        memcpy(v,k,6); v[5]|=128; assert(attend(base)==-4); /* unused high bits */
        memcpy(v,k,6); v[4]|=64; assert(attend(base)==-4); /* missing lane */
        memcpy(v,k,6); memset(v,0,4); assert(attend(base)==-4); /* zero scale with code */
        memcpy(v,k,6); v[3]|=128; assert(attend(base)==-4); /* negative scale */
        memcpy(v,k,6); source[0]=NAN; assert(attend(base)==-4); source[0]=1;
        query[7]=INFINITY; assert(attend(base)==-4); query[7]=-8;
        assert(attend(base)==0); /* dirty unused bytes/metadata stay ignored */
        codecs[1]=4; capacities[1]=12;
        assert(nexa_q4_quantize(source,2,1,2,3,k,6)==0); memcpy(v,k,6);
        assert(attend(base)==0); v[4]=(uint8_t)((v[4]&~15u)|8u); assert(attend(base)==-4);
        memcpy(v,k,6); v[5]=1; assert(attend(base)==-4);
    }
#undef BAD
}

typedef struct {
    const uint8_t *in; uint8_t *out;
    size_t ib, ob, rows, dim, group;
    int source, target;
    float *scratch; size_t sc;
    double *stats; size_t stc;
} recode_args;
static int recode(recode_args a) {
    return nexa_kv_reencode_rows(a.in,a.ib,a.source,a.target,a.rows,a.dim,a.group,a.out,a.ob,
                                 a.scratch,a.sc,a.stats,a.stc);
}
static void recode_errors(int mode) {
    float input[4]={1,-2,3,-4}, scratch[2]; uint8_t packed[12]; double stats[3]={91,91,91};
    recode_args base={(const uint8_t*)input,packed,sizeof input,sizeof packed,2,2,3,0,3,scratch,2,stats,3},a;
    assert(recode(base)==0);
#define BAD(field,value,want) do {a=base;a.field=value;assert(recode(a)==want);} while(0)
    if(mode==0) {
        BAD(in,NULL,-1); BAD(out,NULL,-1); BAD(scratch,NULL,-1); BAD(stats,NULL,-1);
        BAD(ib,15,-2); BAD(ob,11,-2); BAD(sc,1,-2); BAD(stc,2,-2);
        BAD(rows,0,-1); BAD(dim,0,-1); BAD(group,0,-1); BAD(source,2,-1); BAD(target,1,-1);
        BAD(rows,SIZE_MAX,-3); BAD(dim,SIZE_MAX,-3); BAD(group,SIZE_MAX,-3);
    } else if(mode==1) {
        BAD(out,(uint8_t*)input,-1); BAD(scratch,input,-1); BAD(stats,(double*)input,-1);
        BAD(out,(uint8_t*)scratch,-1); BAD(out,(uint8_t*)stats,-1); BAD(scratch,(float*)stats,-1);
        BAD(scratch,(float*)((uint8_t*)scratch+1),-1); BAD(stats,(double*)((uint8_t*)stats+1),-1);
        BAD(in,(const uint8_t*)(uintptr_t)(UINTPTR_MAX-2),-1);
        BAD(out,(uint8_t*)(uintptr_t)(UINTPTR_MAX-2),-1);
    } else {
        input[0]=NAN; assert(recode(base)==-4); input[0]=1;
        /* Source overflow in F32 bridge, even though packed double is finite. */
        uint8_t big[5]={255,255,127,127,7}; /* FLT_MAX * 7 */
        a=base; a.in=big; a.ib=5; a.source=4; a.dim=1; a.rows=1; a.group=1;
        stats[0]=stats[1]=stats[2]=91;
        assert(recode(a)==-5); assert(stats[0]==91 && stats[1]==91 && stats[2]==91);
        /* Identity copies the validated payload without requiring an F32 bridge. */
        a.target=4; assert(recode(a)==0); assert(!memcmp(big,packed,5));
        assert(stats[0]==0 && stats[1]==0 && stats[2]==1);
        /* Smallest F32 cannot provide a nonzero Q3 scale. */
        input[0]=0x1p-149f; input[1]=0;
        a=base; a.rows=1; stats[0]=stats[1]=stats[2]=91;
        assert(recode(a)==-5); assert(stats[0]==91 && stats[1]==91 && stats[2]==91);
        /* Invalid source fails before writing target or statistics. */
        a=base; a.in=big; a.ib=5; a.source=4; a.dim=1; a.rows=1; a.group=1;
        big[4]=8; memset(packed,73,sizeof packed); assert(recode(a)==-4);
        for(size_t i=0;i<sizeof packed;i++) assert(packed[i]==73);
    }
#undef BAD
}
static void numeric(void) {
    uint8_t packed[5]={255,255,127,127,7};
    const uint8_t *pages[]={packed}; uint8_t codecs[]={4}; size_t bytes[]={5};
    float q[1]={FLT_MAX}, scores[1], out[1];
    attention_args a={q,pages,pages,codecs,bytes,1,1,1,1,1,1,1,0,1,1,1,1,scores,out,1,1};
    assert(attend(a)==-5); /* Attention cannot return FLT_MAX*7 in F32. */
    q[0]=0; assert(attend(a)==-5);
    packed[4]=1; assert(attend(a)==0 && out[0]==FLT_MAX);
}
int main(int argc,char **argv) {
    assert(argc==2);
    if(!strcmp(argv[1],"attention_capacity")) attention_errors(0);
    else if(!strcmp(argv[1],"attention_alias")) attention_errors(1);
    else if(!strcmp(argv[1],"attention_data")) attention_errors(2);
    else if(!strcmp(argv[1],"recode_capacity")) recode_errors(0);
    else if(!strcmp(argv[1],"recode_alias")) recode_errors(1);
    else if(!strcmp(argv[1],"recode_numeric")) recode_errors(2);
    else if(!strcmp(argv[1],"numeric")) numeric();
    else return 2;
    return 0;
}
'''


class TieredKVNativeSafetyRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('clang') or shutil.which('cc')
        if not compiler:
            raise unittest.SkipTest('C compiler unavailable')
        cls.directory = tempfile.TemporaryDirectory(prefix='nexa-tiered-safety-')
        cls.addClassCleanup(cls.directory.cleanup)
        source = Path(cls.directory.name) / 'safety.c'
        source.write_text(HARNESS)
        cls.binary = Path(cls.directory.name) / ('safety.exe' if os.name == 'nt' else 'safety')
        directory = ROOT / 'runtime/nexapack'
        command = [compiler, '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror', '-I', str(directory),
                   str(source), str(directory / 'transformer.c'), str(directory / 'q4.c'), '-o', str(cls.binary),
                   '-Dmalloc=nexa_forbidden_malloc', '-Dcalloc=nexa_forbidden_calloc',
                   '-Drealloc=nexa_forbidden_realloc', '-Dfree=nexa_forbidden_free']
        if os.name != 'nt':
            command += ['-lm', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.environment = dict(os.environ)
        cls.environment['ASAN_OPTIONS'] = 'detect_leaks=0' if sys.platform == 'darwin' else 'detect_leaks=1'
        cls.environment['UBSAN_OPTIONS'] = 'halt_on_error=1'

    def run_mode(self, mode):
        result = subprocess.run([str(self.binary), mode], env=self.environment, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_attention_capacity_dimensions_overflow_and_no_heap(self):
        self.run_mode('attention_capacity')

    def test_attention_alias_alignment_and_wrapped_pointer_rejection(self):
        self.run_mode('attention_alias')

    def test_attention_corruption_and_invisible_suffix_contract(self):
        self.run_mode('attention_data')

    def test_reencode_capacity_dimensions_overflow_and_no_heap(self):
        self.run_mode('recode_capacity')

    def test_reencode_alias_alignment_and_wrapped_pointer_rejection(self):
        self.run_mode('recode_alias')

    def test_reencode_numeric_underflow_overflow_and_atomic_statistics(self):
        self.run_mode('recode_numeric')

    def test_attention_output_numeric_range(self):
        self.run_mode('numeric')


if __name__ == '__main__':
    unittest.main()
