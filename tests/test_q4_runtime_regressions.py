"""Q4 physical-format interop and allocation-free CPU kernel regressions."""
from pathlib import Path
import ctypes
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime" / "nexapack"


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def reference_pack(weights, rows, cols, group_size):
    """Independent format oracle, using only Python's standard library."""
    result = bytearray()
    reconstructed = []
    for row in range(rows):
        row_values = []
        for start in range(0, cols, group_size):
            values = weights[row * cols + start:row * cols + min(start + group_size, cols)]
            maximum = max(abs(value) for value in values)
            scale = f32(maximum / 7)
            if maximum and not scale:
                raise ValueError("nonzero scale is not representable")
            codes = []
            for value in values:
                normalized = value / scale if scale else 0
                rounded = math.floor(normalized + 0.5) if normalized >= 0 else math.ceil(normalized - 0.5)
                quantized = max(-7, min(7, rounded))
                codes.append(quantized % 16)
                row_values.append(quantized * scale)
            codes.extend([0] * (group_size - len(codes)))
            if len(codes) % 2:
                codes.append(0)
            result.extend(struct.pack("<f", scale))
            result.extend(codes[i] | codes[i + 1] << 4 for i in range(0, len(codes), 2))
        reconstructed.append(row_values)
    return bytes(result), reconstructed


def ctypes_probe(library):
    """Run in a separate process so a broken native kernel cannot kill unittest."""
    sys.path.insert(0, str(ROOT))
    from runtime.nexapack.format import quantize_q4_row, decode_q4_row

    lib = ctypes.CDLL(str(library))
    fp = ctypes.POINTER(ctypes.c_float)
    bp = ctypes.POINTER(ctypes.c_uint8)
    size = ctypes.c_size_t
    lib.nexa_q4_row_size.argtypes = [size, size]
    lib.nexa_q4_row_size.restype = size
    lib.nexa_q4_size.argtypes = [size, size, size]
    lib.nexa_q4_size.restype = size
    lib.nexa_q4_quantize.argtypes = [fp, size, size, size, size, bp, size]
    lib.nexa_q4_quantize.restype = ctypes.c_int
    lib.nexa_q4_matmul.argtypes = [fp, size, size, bp, size, size, size, size, fp, size]
    lib.nexa_q4_matmul.restype = ctypes.c_int
    rng = random.Random(512)
    shapes = [(1, 1, 1), (1, 1, 3), (2, 17, 8), (5, 11, 7), (7, 37, 32), (3, 9, 1), (2, 5, 128)]
    for rows, cols, group_size in shapes:
        weights = [f32(rng.uniform(-11, 11)) for _ in range(rows * cols)]
        expected, decoded = reference_pack(weights, rows, cols, group_size)
        codec_rows = [quantize_q4_row(weights[row * cols:(row + 1) * cols], group_size)
                      for row in range(rows)]
        assert b"".join(codec_rows) == expected
        for row in range(rows):
            assert decode_q4_row(codec_rows[row], cols, group_size) == decoded[row]
        packed_size = lib.nexa_q4_size(rows, cols, group_size)
        assert packed_size == len(expected), (rows, cols, group_size, packed_size, len(expected))
        packed = (ctypes.c_uint8 * (packed_size + 8))(*([0xA5] * (packed_size + 8)))
        weight_array = (ctypes.c_float * len(weights))(*weights)
        assert lib.nexa_q4_quantize(weight_array, len(weights), rows, cols, group_size, packed, packed_size) == 0
        assert bytes(packed[:packed_size]) == expected, (rows, cols, group_size)
        assert bytes(packed[packed_size:]) == b"\xa5" * 8
        for batch in (1, 3):
            values = [f32(rng.uniform(-2, 2)) for _ in range(batch * cols)]
            inputs = (ctypes.c_float * len(values))(*values)
            output = (ctypes.c_float * (batch * rows + 1))(*([-999] * (batch * rows + 1)))
            status = lib.nexa_q4_matmul(inputs, len(values), batch, packed, packed_size,
                                        rows, cols, group_size, output, batch * rows)
            assert status == 0, status
            for item in range(batch):
                for row in range(rows):
                    expected_value = f32(sum(values[item * cols + k] * decoded[row][k] for k in range(cols)))
                    assert math.isclose(output[item * rows + row], expected_value, rel_tol=2e-6, abs_tol=2e-6), (
                        rows, cols, group_size, item, row, output[item * rows + row], expected_value)
            assert output[batch * rows] == -999
            assert lib.nexa_q4_matmul(inputs, len(values) - 1, batch, packed, packed_size,
                                      rows, cols, group_size, output, batch * rows) == -2
            assert lib.nexa_q4_matmul(inputs, len(values), batch, packed, packed_size - 1,
                                      rows, cols, group_size, output, batch * rows) == -2
            assert lib.nexa_q4_matmul(inputs, len(values), batch, packed, packed_size,
                                      rows, cols, group_size, output, batch * rows - 1) == -2
    # A row offset must be sufficient to consume one row without repacking it.
    weights = [f32(value) for value in (7, 0, -7, 7, -7, 1, 7, 0, -7, 1)]
    raw, _ = reference_pack(weights, 2, 5, 3)
    packed = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
    second_row = ctypes.cast(ctypes.byref(packed, lib.nexa_q4_row_size(5, 3)), bp)
    inputs = (ctypes.c_float * 5)(1, 2, 3, 4, 5)
    output = (ctypes.c_float * 1)()
    assert lib.nexa_q4_matmul(inputs, 5, 1, second_row, len(raw) // 2, 1, 5, 3, output, 1) == 0
    assert output[0] == -8
    print("Q4 ctypes interop passed")


class Q4RuntimeRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("clang") or shutil.which("cc")
        if not cls.compiler:
            raise unittest.SkipTest("C compiler unavailable")

    def test_c_sanitizers_and_no_heap_dependencies(self):
        with tempfile.TemporaryDirectory(prefix="nexa-q4-native-") as temporary:
            binary = Path(temporary) / ("q4_tests.exe" if os.name == "nt" else "q4_tests")
            command = [self.compiler, "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                       str(SOURCE / "q4.c"), str(SOURCE / "q4_regressions.c"), "-o", str(binary)]
            # Any new heap call from these translation units would fail to link.
            command.extend(f"-D{name}=nexa_q4_forbidden_{name}" for name in ("malloc", "calloc", "realloc", "free"))
            if os.name != "nt":
                command.extend(["-lm", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"])
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            env = dict(os.environ)
            env["ASAN_OPTIONS"] = "detect_leaks=0" if sys.platform == "darwin" else "detect_leaks=1"
            env["UBSAN_OPTIONS"] = "halt_on_error=1"
            result = subprocess.run([str(binary)], capture_output=True, text=True, env=env, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_ctypes_format_and_matmul_reference(self):
        with tempfile.TemporaryDirectory(prefix="nexa-q4-interop-") as temporary:
            extension = "dll" if os.name == "nt" else "dylib" if sys.platform == "darwin" else "so"
            library = Path(temporary) / f"q4.{extension}"
            command = [self.compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror"]
            if os.name != "nt":
                command.append("-fPIC")
            command.extend(["-dynamiclib" if sys.platform == "darwin" else "-shared",
                            str(SOURCE / "q4.c"), "-o", str(library)])
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
