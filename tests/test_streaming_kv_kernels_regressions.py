"""Two-pass single-page attention: exact resident parity and hostile ABI inputs."""
import ctypes
import itertools
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
from test_tiered_kv_kernels_regressions import f32, oracle, pack


class StreamingKVNumericsRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='nexa-streaming-numerics-')
        cls.addClassCleanup(cls.directory.cleanup)
        cls.lib = ctypes.CDLL(str(build_runtime('nexa_transformer', cls.directory.name)))
        fp, bp, dp, sz = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_double), ctypes.c_size_t
        cls.lib.nexa_causal_gqa_attention_page.argtypes = [fp, sz, bp, sz, bp, sz, ctypes.c_int,
            sz, sz, sz, sz, sz, sz, sz, sz, ctypes.c_int, dp, sz, dp, sz]
        cls.lib.nexa_causal_gqa_attention_page.restype = ctypes.c_int
        cls.lib.nexa_causal_gqa_attention_finish.argtypes = [dp, sz, dp, sz, sz, sz, sz, fp, sz]
        cls.lib.nexa_causal_gqa_attention_finish.restype = ctypes.c_int
        cls.lib.nexa_causal_gqa_attention_paged_mixed.argtypes = [fp, sz, ctypes.POINTER(bp), sz, ctypes.POINTER(bp), sz,
            bp, sz, ctypes.POINTER(sz), sz, sz, sz, sz, sz, sz, sz, sz, fp, sz, fp, sz]
        cls.lib.nexa_causal_gqa_attention_paged_mixed.restype = ctypes.c_int

    def exercise(self, layouts, orders):
        bp = ctypes.POINTER(ctypes.c_uint8)
        for dim, group, heads, kv_heads, page_tokens in layouts:
            tokens = 3 * page_tokens - 1 if page_tokens > 1 else 3
            source = [[f32(((i * 19 + lane * 11 + bank * 7) % 53 - 26) / 8)
                       for lane in range(dim)] for bank in (0, 1) for i in range(tokens * kv_heads)]
            for order in orders:
                codecs = (ctypes.c_uint8 * 3)(*order)
                capacities = (ctypes.c_size_t * 3)()
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
                    tables.append((bp * 3)(*pointers))
                    decoded.append(rows)
                original = [bytes(owner) for owner in owners]
                for past, count in ((0, tokens), (tokens - 1, 1), (0, 1), (1, tokens - 1)):
                    q = [f32(((i * 13 + 5) % 37 - 18) / 9) for i in range(count * heads * dim)]
                    query = (ctypes.c_float * len(q))(*q)
                    scores = (ctypes.c_float * (past + count))()
                    resident = (ctypes.c_float * len(q))()
                    out = (ctypes.c_float * (len(q) + 1))(*([77.0] * (len(q) + 1)))
                    maxima = (ctypes.c_double * (count * heads + 1))(*([-math.inf] * (count * heads)), 77.0)
                    sums = (ctypes.c_double * (count * heads * (dim + 1) + 1))()
                    sums[-1] = 77.0
                    status = self.lib.nexa_causal_gqa_attention_paged_mixed(query, len(q), tables[0], 3, tables[1], 3,
                        codecs, 3, capacities, 3, page_tokens, group, past, count, heads, kv_heads, dim,
                        scores, len(scores), resident, len(resident))
                    self.assertEqual(status, 0)
                    for phase in (0, 1):
                        before_maxima, before_sums = bytes(maxima), bytes(sums)
                        for start in range(0, past + count, page_tokens):
                            page = start // page_tokens
                            valid = min(page_tokens, past + count - start)
                            status = self.lib.nexa_causal_gqa_attention_page(query, len(q), tables[0][page], capacities[page],
                                tables[1][page], capacities[page], order[page], group, start, valid, past, count, heads,
                                kv_heads, dim, phase, maxima, len(maxima), sums, len(sums))
                            self.assertEqual(status, 0, (dim, group, order, phase, past, count, start))
                        if phase == 0:
                            self.assertEqual(bytes(sums), before_sums)
                        else:
                            self.assertEqual(bytes(maxima), before_maxima)
                    state = bytes(maxima), bytes(sums)
                    status = self.lib.nexa_causal_gqa_attention_finish(maxima, len(maxima), sums, len(sums),
                                                                                    count, heads, dim, out, len(out))
                    self.assertEqual(status, 0)
                    self.assertEqual(list(out)[:-1], list(resident), (dim, group, order, past, count))
                    expected = oracle(q, *decoded, past, count, heads, kv_heads, dim)
                    for actual, wanted in zip(out, expected):
                        self.assertTrue(math.isclose(actual, wanted, abs_tol=3e-6, rel_tol=3e-6))
                    self.assertEqual((maxima[-1], sums[-1], out[-1]), (77.0,) * 3)
                    self.assertEqual((bytes(maxima), bytes(sums)), state)
                    self.assertEqual([bytes(owner) for owner in owners], original)

    def test_mixed_pages_are_bit_identical_to_resident_gqa_mqa_mha(self):
        self.exercise(((7, 3, 4, 2, 2), (8, 8, 4, 1, 3), (3, 5, 3, 3, 1)),
                      tuple(itertools.permutations((0, 3, 4))))

    def test_tiny_heads_and_odd_groups_causal_partial_pages(self):
        self.exercise(((1, 1, 2, 1, 2), (17, 9, 4, 2, 2)), ((0, 4, 3), (4, 3, 0)))

    def test_homogeneous_codecs_and_larger_head_are_bit_identical(self):
        self.exercise(((64, 32, 4, 2, 2),), ((0, 0, 0), (4, 4, 4), (3, 3, 3)))


HARNESS = r'''
#include "transformer.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
void *nexa_forbidden_malloc(size_t n) { (void)n; abort(); }
void *nexa_forbidden_calloc(size_t n, size_t s) { (void)n; (void)s; abort(); }
void *nexa_forbidden_realloc(void *p, size_t n) { (void)p; (void)n; abort(); }
void nexa_forbidden_free(void *p) { (void)p; abort(); }

typedef struct {
    const float *q; const uint8_t *k,*v;
    size_t qc,kb,vb,group,start,tokens,past,seq,heads,kvheads,dim;
    int codec,phase; double *maxima,*sums; size_t mc,sc;
} args;
static int run(args a) {
    return nexa_causal_gqa_attention_page(a.q,a.qc,a.k,a.kb,a.v,a.vb,a.codec,a.group,a.start,a.tokens,
        a.past,a.seq,a.heads,a.kvheads,a.dim,a.phase,a.maxima,a.mc,a.sums,a.sc);
}
static void page_errors(int mode) {
    float input[4]={1,-2,3,-4}, query[8]={1,-2,3,-4,5,-6,7,-8};
    uint8_t k[24],v[24]; memset(k,203,sizeof k);memset(v,203,sizeof v);
    assert(nexa_q3_quantize(input,4,2,2,3,k,12)==0); memcpy(v,k,12);
    double maxima[4]={-INFINITY,-INFINITY,-INFINITY,-INFINITY}, sums[12]={0};
    args base={query,k,v,8,24,24,3,1,2,1,2,2,1,2,3,0,maxima,sums,4,12},a;
    assert(run(base)==0);
#define BAD(field,value,want) do {a=base;a.field=value;assert(run(a)==want);} while(0)
    if(mode==0) {
        BAD(q,NULL,-1);BAD(k,NULL,-1);BAD(v,NULL,-1);BAD(maxima,NULL,-1);BAD(sums,NULL,-1);
        BAD(qc,7,-2);BAD(kb,11,-2);BAD(vb,11,-2);BAD(mc,3,-2);BAD(sc,11,-2);
        BAD(group,0,-1);BAD(tokens,0,-1);BAD(seq,0,-1);BAD(heads,0,-1);BAD(kvheads,0,-1);BAD(dim,0,-1);
        BAD(kvheads,3,-1);BAD(codec,2,-1);BAD(phase,-1,-1);BAD(phase,2,-1);BAD(start,2,-1);
        BAD(start,SIZE_MAX,-3);BAD(tokens,SIZE_MAX,-3);BAD(past,SIZE_MAX,-3);
        BAD(dim,SIZE_MAX,-3);BAD(heads,SIZE_MAX,-3);BAD(group,SIZE_MAX,-3);
    } else if(mode==1) {
        BAD(maxima,(double*)query,-1);BAD(sums,(double*)query,-1);BAD(maxima,sums,-1);
        BAD(maxima,(double*)k,-1);BAD(sums,(double*)k,-1);BAD(maxima,(double*)v,-1);BAD(sums,(double*)v,-1);
        BAD(q,(float*)((uint8_t*)query+1),-1);BAD(maxima,(double*)((uint8_t*)maxima+1),-1);
        BAD(sums,(double*)((uint8_t*)sums+1),-1);
        BAD(q,(const float*)(uintptr_t)(UINTPTR_MAX-3),-1);
        BAD(k,(const uint8_t*)(uintptr_t)(UINTPTR_MAX-2),-1);
        BAD(v,(const uint8_t*)(uintptr_t)(UINTPTR_MAX-2),-1);
        BAD(maxima,(double*)(uintptr_t)(UINTPTR_MAX-7),-1);
        BAD(sums,(double*)(uintptr_t)(UINTPTR_MAX-7),-1);
    } else if(mode==2) {
        double original_max[4],original_sums[12];memcpy(original_max,maxima,sizeof maxima);memcpy(original_sums,sums,sizeof sums);
        v[10]=(uint8_t)((v[10]&~7u)|4u); assert(run(base)==-4);
        assert(!memcmp(maxima,original_max,sizeof maxima));assert(!memcmp(sums,original_sums,sizeof sums));
        memcpy(v,k,12);v[11]|=128;assert(run(base)==-4); /* unused high bits */
        memcpy(v,k,12);v[10]|=64;assert(run(base)==-4); /* missing lane */
        memcpy(v,k,12);memset(v+6,0,4);assert(run(base)==-4); /* zero scale codes */
        memcpy(v,k,12);v[9]|=128;assert(run(base)==-4); /* negative scale */
        memcpy(v,k,12);query[7]=NAN;assert(run(base)==-4);query[7]=-8;
        a=base;a.phase=1;v[10]=4;assert(run(a)==-4);memcpy(v,k,12);
        /* Trailing invalid bytes and extra capacities remain untouched. */
        assert(run(base)==0);assert(run(a)==0);
        a=base;a.codec=4;assert(nexa_q4_quantize(input,4,2,2,3,k,12)==0);memcpy(v,k,12);
        assert(run(a)==0);v[10]=(uint8_t)((v[10]&~15u)|8u);assert(run(a)==-4);
        memcpy(v,k,12);v[11]=1;assert(run(a)==-4);
        a=base;a.codec=0;a.k=(const uint8_t*)input;a.v=(const uint8_t*)input;a.kb=a.vb=sizeof input;
        input[3]=INFINITY;assert(run(a)==-4);
    } else {
        maxima[0]=INFINITY;assert(run(base)==-4);maxima[0]=NAN;assert(run(base)==-4);
        maxima[0]=-INFINITY;assert(run(base)==0);
        sums[0]=-1;assert(run(base)==-4);sums[0]=0;
        sums[11]=NAN;assert(run(base)==-4);sums[11]=0;
        a=base;a.phase=1;maxima[0]=-INFINITY;assert(run(a)==-4);
        maxima[0]=-DBL_MAX;assert(run(a)==-4); /* incorrect maximum pass */
        for(int i=0;i<4;i++) maxima[i]=-INFINITY;
        assert(run(base)==0);assert(run(a)==0);
    }
#undef BAD
}

typedef struct {const double *m,*s;size_t mc,sc,seq,heads,dim;float *o;size_t oc;} final_args;
static int finish(final_args a) {
    return nexa_causal_gqa_attention_finish(a.m,a.mc,a.s,a.sc,a.seq,a.heads,a.dim,a.o,a.oc);
}
static void finish_errors(int mode) {
    double maxima[4]={0,0,0,0}, sums[12]={1,1,-2,1,3,-4,1,5,-6,1,7,-8};float out[8];
    final_args base={maxima,sums,4,12,2,2,2,out,8},a;
    assert(finish(base)==0);assert(out[0]==1&&out[7]==-8);
#define BAD(field,value,want) do {a=base;a.field=value;assert(finish(a)==want);} while(0)
    if(mode==0) {
        BAD(m,NULL,-1);BAD(s,NULL,-1);BAD(o,NULL,-1);BAD(mc,3,-2);BAD(sc,11,-2);BAD(oc,7,-2);
        BAD(seq,0,-1);BAD(heads,0,-1);BAD(dim,0,-1);BAD(seq,SIZE_MAX,-3);BAD(dim,SIZE_MAX,-3);
        BAD(o,(float*)maxima,-1);BAD(o,(float*)sums,-1);BAD(m,sums,-1);
        BAD(m,(const double*)((uint8_t*)maxima+1),-1);BAD(s,(const double*)((uint8_t*)sums+1),-1);
        BAD(o,(float*)((uint8_t*)out+1),-1);BAD(m,(const double*)(uintptr_t)(UINTPTR_MAX-7),-1);
        BAD(s,(const double*)(uintptr_t)(UINTPTR_MAX-7),-1);BAD(o,(float*)(uintptr_t)(UINTPTR_MAX-3),-1);
    } else {
        for(int i=0;i<8;i++) out[i]=91;
        maxima[3]=-INFINITY;assert(finish(base)==-4);maxima[3]=0;
        sums[9]=0;assert(finish(base)==-4);sums[9]=-1;assert(finish(base)==-4);sums[9]=1;
        sums[11]=NAN;assert(finish(base)==-4);sums[11]=INFINITY;assert(finish(base)==-4);sums[11]=-8;
        for(int i=0;i<8;i++) assert(out[i]==91);
        sums[11]=2.0*(double)FLT_MAX;assert(finish(base)==-5);
    }
#undef BAD
}
static void extreme(void) {
    uint8_t packed[5]={255,255,127,127,7};float query[1]={FLT_MAX},out[1];
    double maxima[1]={-INFINITY},sums[2]={0,0};
    args a={query,packed,packed,1,5,5,1,0,1,0,1,1,1,1,4,0,maxima,sums,1,2};
    assert(run(a)==0);assert(isfinite(maxima[0])&&maxima[0]>FLT_MAX);
    a.phase=1;assert(run(a)==0);assert(sums[0]==1);
    final_args f={maxima,sums,1,2,1,1,1,out,1};assert(finish(f)==-5);
    packed[4]=1;maxima[0]=-INFINITY;sums[0]=sums[1]=0;a.phase=0;
    assert(run(a)==0);a.phase=1;assert(run(a)==0);assert(finish(f)==0&&out[0]==FLT_MAX);
}
int main(int argc,char **argv) {
    assert(argc==2);
    if(!strcmp(argv[1],"capacity"))page_errors(0);
    else if(!strcmp(argv[1],"alias"))page_errors(1);
    else if(!strcmp(argv[1],"corruption"))page_errors(2);
    else if(!strcmp(argv[1],"state"))page_errors(3);
    else if(!strcmp(argv[1],"finish_capacity"))finish_errors(0);
    else if(!strcmp(argv[1],"finish_state"))finish_errors(1);
    else if(!strcmp(argv[1],"numeric"))extreme();
    else return 2;
    return 0;
}
'''


class StreamingKVNativeSafetyRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('clang') or shutil.which('cc')
        if not compiler:
            raise unittest.SkipTest('C compiler unavailable')
        cls.directory = tempfile.TemporaryDirectory(prefix='nexa-streaming-safety-')
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

    def test_page_capacity_dimensions_overflow_and_no_heap(self):
        self.run_mode('capacity')

    def test_page_alias_alignment_and_wrapped_pointer_rejection(self):
        self.run_mode('alias')

    def test_page_corruption_validation_both_phases_and_dirty_suffix(self):
        self.run_mode('corruption')

    def test_accumulator_state_and_incomplete_maxima_rejection(self):
        self.run_mode('state')

    def test_finish_capacity_alignment_alias_and_overflow(self):
        self.run_mode('finish_capacity')

    def test_finish_invalid_accumulators_and_float_overflow(self):
        self.run_mode('finish_state')

    def test_large_double_scores_and_float_output_range(self):
        self.run_mode('numeric')


if __name__ == '__main__':
    unittest.main()
