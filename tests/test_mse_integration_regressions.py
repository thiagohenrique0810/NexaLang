"""MSE-only TurboQuant integration through the supported language and Python APIs."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FFI = '''
extern "C" {
    fn tq_create(dim: i32, bits: i32, seed: i32) -> *u8;
    fn tq_destroy(ctx: *u8);
    fn tq_context_memory_bytes(ctx: *u8) -> u64;
    fn tq_prod_qjl_packed_size(ctx: *u8, n: i32) -> u64;
    fn tq_quantize_prod(ctx: *u8, input: *f32, idx: *u8, qjl: *u8, gamma: *f32, n: i32) -> i32;
    fn tq_dequantize_prod(ctx: *u8, idx: *u8, qjl: *u8, gamma: *f32, output: *f32, n: i32) -> i32;
    fn malloc(size: i32) -> *u8;
    fn free(ptr: *u8);
}
'''


class MSEIntegrationRegressions(unittest.TestCase):
    def setUp(self):
        if not shutil.which("clang"):
            self.skipTest("clang required for native integration")
        self.temp = tempfile.TemporaryDirectory(prefix="nexa-mse-integration-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def run_language(self, source, *, failure=None):
        program = self.directory / "main.nxl"
        program.write_text(source, encoding="utf-8")
        for mode in ([], ["--jit"]):
            with self.subTest(mode=mode or ["native"]):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "nx.py"), "run", str(program), *mode],
                    cwd=self.directory, text=True, capture_output=True, timeout=60,
                )
                output = result.stdout + result.stderr
                if failure is None:
                    self.assertEqual(result.returncode, 0, output)
                    self.assertIn("MSE_INTEGRATION_OK", output)
                else:
                    self.assertNotEqual(result.returncode, 0, output)
                    self.assertIn(failure, output)

    def test_builtin_mse_and_explicit_ffi_preserve_legacy_prod(self):
        self.run_language(FFI + '''
fn main() -> i32 {
    let mse = compress::create_mse(8, 3);
    let seeded = compress::create_mse(8, 3, 42);
    let legacy = compress::create(8, 3, 42);
    let ffi_legacy = tq_create(8, 3, 42);
    assert!(cast::<i64>(mse) != 0, "MSE context");
    assert!(tq_context_memory_bytes(mse) < tq_context_memory_bytes(legacy), "linear state");
    assert!(tq_prod_qjl_packed_size(mse, 1) == cast::<u64>(0), "no Prod in MSE");
    assert!(tq_prod_qjl_packed_size(legacy, 1) == cast::<u64>(1), "builtin Prod preserved");
    assert!(tq_prod_qjl_packed_size(ffi_legacy, 1) == cast::<u64>(1), "FFI Prod preserved");
    let input = cast::<*f32>(malloc(32));
    let output = cast::<*f32>(malloc(32));
    let indices = malloc(16);
    let other = malloc(16);
    for i in 0..8 { input[i] = cast::<f32>(i - 4) * 0.1; }
    compress::quantize(mse, input, indices, 1);
    compress::quantize(legacy, input, other, 1);
    for i in 0..16 { assert!(indices[i] == other[i], "unpacked indices compatibility"); }
    compress::quantize(seeded, input, other, 1);
    for i in 0..16 { assert!(indices[i] == other[i], "default seed compatibility"); }
    compress::dequantize(mse, indices, output, 1);
    assert!(compress::mse(mse, input, 1) == compress::mse(legacy, input, 1), "MSE compatibility");
    let qjl = malloc(1);
    let gamma = cast::<*f32>(malloc(4));
    indices[0] = 99; qjl[0] = 87; gamma[0] = 9.0; output[0] = 7.0;
    assert!(tq_quantize_prod(mse, input, indices, qjl, gamma, 1) == -2, "Prod mode status");
    assert!(indices[0] == 99 and qjl[0] == 87 and gamma[0] == 9.0, "no output writes");
    assert!(tq_dequantize_prod(mse, indices, qjl, gamma, output, 1) == -2, "Prod decode mode status");
    assert!(output[0] == 7.0, "no decoded writes");
    assert!(tq_quantize_prod(legacy, input, indices, qjl, gamma, 1) == 0, "legacy Prod encode");
    assert!(tq_dequantize_prod(ffi_legacy, indices, qjl, gamma, output, 1) == 0, "legacy Prod decode");
    free(indices); free(other); free(qjl); free(cast::<*u8>(gamma));
    free(cast::<*u8>(input)); free(cast::<*u8>(output));
    compress::destroy(mse); compress::destroy(seeded); compress::destroy(legacy);
    tq_destroy(ffi_legacy);
    print("MSE_INTEGRATION_OK"); return 0;
}
''')

    def test_quantizer_mse_constructors_preserve_packed_seed_contract(self):
        self.run_language('use std::compress::Quantizer;\n' + FFI + '''
fn main() -> i32 {
    let q = Quantizer::new_mse(8, 3);
    let seeded = Quantizer::with_seed_mse(8, 3, 91);
    let legacy = Quantizer::new(8, 3);
    let old_seeded = Quantizer::with_seed(8, 3, 91);
    let input = cast::<*f32>(malloc(64));
    let output = cast::<*f32>(malloc(64));
    let packed = malloc(q.compressed_size(2));
    let reference = malloc(legacy.compressed_size(2));
    for i in 0..8 { input[i] = cast::<f32>(i - 4) * 1.5; input[i + 8] = 0.0; }
    q.quantize(input, packed, 2); legacy.quantize(input, reference, 2);
    for i in 0..q.compressed_size(2) { assert!(packed[i] == reference[i], "TQ01 compatibility"); }
    q.dequantize(packed, output, 2);
    for i in 8..16 { assert!(output[i] == 0.0, "zero norm preserved"); }
    seeded.quantize(input, packed, 2); old_seeded.quantize(input, reference, 2);
    for i in 0..seeded.compressed_size(2) { assert!(packed[i] == reference[i], "custom seed compatibility"); }
    old_seeded.dequantize(packed, output, 2);
    assert!(tq_prod_qjl_packed_size(q.handle, 1) == cast::<u64>(0), "MSE constructor");
    assert!(tq_prod_qjl_packed_size(seeded.handle, 1) == cast::<u64>(0), "seeded MSE constructor");
    let prod = old_seeded.quantize_prod_alloc(input, 2);
    old_seeded.dequantize_prod(prod.idx, prod.qjl, prod.gamma, output, 2);
    prod.drop();
    free(packed); free(reference); free(cast::<*u8>(input)); free(cast::<*u8>(output));
    print("MSE_INTEGRATION_OK"); return 0;
}
''')

    def test_kv_cache_uses_mse_and_preserves_roundtrip_and_capacity(self):
        self.run_language('''use std::kv_cache_quant::QuantizedKVCache;
use std::compress::Quantizer;
''' + FFI + '''
fn main() -> i32 {
    let mut cache = QuantizedKVCache::new(8, 3, 2);
    let old = Quantizer::new(8, 3);
    assert!(tq_prod_qjl_packed_size(cache.qk, 1) == cast::<u64>(0), "K is MSE");
    assert!(tq_prod_qjl_packed_size(cache.qv, 1) == cast::<u64>(0), "V is MSE");
    let input = cast::<*f32>(malloc(32));
    let output = cast::<*f32>(malloc(32));
    let reference = cast::<*f32>(malloc(32));
    let packed = malloc(old.compressed_size(1));
    for i in 0..8 { input[i] = cast::<f32>(i - 3) * 2.0; }
    old.quantize(input, packed, 1); old.dequantize(packed, reference, 1);
    assert!(cache.push(input, input) == 1, "first token");
    assert!(cache.push(input, input) == 1, "second token");
    assert!(cache.push(input, input) == 0, "capacity enforced");
    assert!(cache.len() == 2, "length stable after failure");
    assert!(cache.bytes_per_entry() == old.compressed_size(1), "packed layout unchanged");
    cache.get_k(0, output);
    for i in 0..8 { assert!(output[i] == reference[i], "key decode compatibility"); }
    cache.get_v(1, output);
    for i in 0..8 { assert!(output[i] == reference[i], "value decode compatibility"); }
    free(packed); free(cast::<*u8>(input)); free(cast::<*u8>(output)); free(cast::<*u8>(reference));
    print("MSE_INTEGRATION_OK"); return 0;
}
''')

    def test_mse_buffer_and_quick_helpers_keep_legacy_buffer_prod(self):
        self.run_language('''use std::compress::CompressedBuffer;
use std::compress::quick_compress;
use std::compress::quick_decompress;
''' + FFI + '''
fn main() -> i32 {
    let mut buffer = CompressedBuffer::new_mse(8, 3, 2);
    let mut legacy = CompressedBuffer::new(8, 3, 2);
    assert!(tq_prod_qjl_packed_size(buffer.quant.handle, 1) == cast::<u64>(0), "MSE buffer");
    assert!(tq_prod_qjl_packed_size(legacy.quant.handle, 1) == cast::<u64>(1), "legacy exposed Prod preserved");
    let input = cast::<*f32>(malloc(32));
    let output = cast::<*f32>(malloc(32));
    for i in 0..8 { input[i] = cast::<f32>(i - 3) * 1.5; }
    buffer.push(input); legacy.push(input);
    buffer.get(0, output);
    let packed = quick_compress(input, 8, 1, 3);
    let decoded = quick_decompress(packed, 8, 1, 3);
    for i in 0..8 { assert!(output[i] == decoded[i], "quick decode compatibility"); }
    for i in 0..buffer.quant.compressed_size(1) {
        assert!(packed[i] == buffer.data[i] and packed[i] == legacy.data[i], "packed bytes preserved");
    }
    assert!(buffer.len() == 1, "buffer length");
    free(packed); free(cast::<*u8>(decoded)); free(cast::<*u8>(input)); free(cast::<*u8>(output));
    print("MSE_INTEGRATION_OK"); return 0;
}
''')

    def test_mse_stdlib_prod_misuse_fails_deterministically(self):
        self.run_language('use std::compress::Quantizer;\n' + FFI + '''
fn main() -> i32 {
    let q = Quantizer::new_mse(8, 3);
    let input = cast::<*f32>(malloc(32));
    let idx = malloc(2); let qjl = malloc(1); let gamma = cast::<*f32>(malloc(4));
    for i in 0..8 { input[i] = 0.0; }
    q.quantize_prod(input, idx, qjl, gamma, 1);
    return 0;
}
''', failure="TurboQuantProd operation failed")

    def test_python_prompt_wrapper_uses_mse_without_model_dependencies(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy required only for the optional prompt wrapper")
        from runtime.build_runtime import build_runtime
        from tools.chat_tinyllama_turboquant import TurboQuant
        library = build_runtime("turboquant", self.directory)
        quantizer = TurboQuant(library, dim=8, bits=3, seed=91)
        self.addCleanup(quantizer.close)
        lib = quantizer.lib
        lib.tq_prod_qjl_packed_size.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.tq_prod_qjl_packed_size.restype = ctypes.c_size_t
        self.assertEqual(lib.tq_prod_qjl_packed_size(quantizer.ctx, 1), 0)
        values = np.arange(16, dtype=np.float32).reshape(2, 8) - np.float32(5)
        decoded, error, size = quantizer.roundtrip(values[:, ::-1])
        lib.tq_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.tq_create.restype = ctypes.c_void_p
        legacy = lib.tq_create(8, 3, 91)
        self.assertTrue(legacy)
        try:
            fp = ctypes.POINTER(ctypes.c_float)
            contiguous = np.ascontiguousarray(values[:, ::-1])
            packed = (ctypes.c_uint8 * size)()
            reference = np.zeros_like(contiguous)
            self.assertEqual(lib.tq_quantize_packed(legacy, contiguous.ctypes.data_as(fp), packed, 2), 0)
            self.assertEqual(lib.tq_dequantize_packed(legacy, packed, reference.ctypes.data_as(fp), 2), 0)
            np.testing.assert_array_equal(decoded, reference)
            self.assertEqual(size, 22)
            self.assertEqual(error, float(np.mean((contiguous - reference) ** 2)))
        finally:
            lib.tq_destroy(legacy)
        quantizer.close()
        quantizer.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            quantizer.roundtrip(values)


if __name__ == "__main__":
    unittest.main()
