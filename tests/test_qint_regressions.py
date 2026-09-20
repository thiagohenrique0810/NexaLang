"""qint<N>/PackedVector<N>: three-way byte identity and the compiler's gates.

The central claim is narrow and checkable: for every width the language
exposes, what `qpack::pack` writes is the same bytes `runtime/nexapack/format.py`
writes and the same bytes an oracle written from the published prose writes.
Two of those agreeing would only show that one copied the other, so all three
have to agree, over group sizes whose tails land differently.
"""
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import qint_reference
from runtime.nexapack import format as nexapack_format

WIDTHS = (2, 3, 4, 8)
GROUP_SIZES = (1, 7, 8, 16, 17, 32, 33)
VALUE_COUNTS = (1, 7, 32, 33, 64)
MODES = (0, 1, 2)
FORMAT_WRITER = {
    2: nexapack_format.quantize_q2_row,
    3: nexapack_format.quantize_q3_row,
    4: nexapack_format.quantize_q4_row,
    8: nexapack_format.quantize_q8_row,
}
FORMAT_READER = {
    2: nexapack_format.decode_q2_row,
    3: nexapack_format.decode_q3_row,
    4: nexapack_format.decode_q4_row,
    8: nexapack_format.decode_q8_row,
}

# Mirrors `sample` in the emitted program. Every value is a small integer times
# a negative power of two, so both sides hold exactly the same F32 and the
# comparison cannot be blurred by a rounding difference in the fixture itself.
def sample(index, mode):
    if mode == 1:
        return 0.0
    if mode == 2:
        if (index // 8) % 2 == 0:
            return 0.0
        return float((index * 13) % 17 - 8) * 0.125
    return float((index * 37) % 71 - 35) * 0.0625


EMITTER = '''extern "C" {{
    fn malloc(size: i32) -> *u8;
    fn free(ptr: *u8);
}}

fn group_size(index: i32) -> i32 {{
    if (index == 0) {{ return 1; }}
    if (index == 1) {{ return 7; }}
    if (index == 2) {{ return 8; }}
    if (index == 3) {{ return 16; }}
    if (index == 4) {{ return 17; }}
    if (index == 5) {{ return 32; }}
    return 33;
}}

fn value_count(index: i32) -> i32 {{
    if (index == 0) {{ return 1; }}
    if (index == 1) {{ return 7; }}
    if (index == 2) {{ return 32; }}
    if (index == 3) {{ return 33; }}
    return 64;
}}

fn sample(index: i32, mode: i32) -> f32 {{
    if (mode == 1) {{ return 0.0; }}
    if (mode == 2) {{
        if ((index / 8) % 2 == 0) {{ return 0.0; }}
        return cast::<f32>((index * 13) % 17 - 8) * 0.125;
    }}
    return cast::<f32>((index * 37) % 71 - 35) * 0.0625;
}}

fn main() -> i32 {{
    let values = cast::<*f32>(malloc(64 * 4));
    let back = cast::<*f32>(malloc(64 * 4));
    for g in 0..7 {{
        for c in 0..5 {{
            for m in 0..3 {{
                let group = group_size(g);
                let count = value_count(c);
                for i in 0..count {{ values[i] = sample(i, m); }}
                let bytes = cast::<i32>(qpack::size::<{width}>(count, group));
                let packed = cast::<PackedVector<{width}>>(malloc(bytes));
                let status = qpack::pack(packed, values, count, group);
                let decoded = qpack::unpack(packed, back, count, group);
                print("CASE");
                print({width}); print(group); print(count); print(m);
                print(status); print(decoded); print(bytes);
                print(cast::<i32>(qpack::groups::<{width}>(count, group)));
                let raw = cast::<*u8>(packed);
                for b in 0..bytes {{ print(cast::<i32>(raw[b])); }}
                print("CODES");
                for i in 0..count {{ print(cast::<i32>(qpack::code(packed, count, group, i))); }}
                free(cast::<*u8>(packed));
            }}
        }}
    }}
    free(cast::<*u8>(values));
    free(cast::<*u8>(back));
    print("QINT_EMIT_OK");
    return 0;
}}
'''


def parse_cases(output):
    """Read the emitted stream back: a header, the raw bytes, then the codes."""
    lines = output.splitlines()
    start = lines.index("CASE")
    tokens = lines[start:lines.index("QINT_EMIT_OK")]
    cases, position = [], 0
    while position < len(tokens):
        assert tokens[position] == "CASE", tokens[position]
        header = [int(value) for value in tokens[position + 1:position + 9]]
        width, group, count, mode, status, decoded, size, groups = header
        position += 9
        payload = bytes(int(value) for value in tokens[position:position + size])
        position += size
        assert tokens[position] == "CODES", tokens[position]
        position += 1
        codes = [int(value) for value in tokens[position:position + count]]
        position += count
        cases.append({"width": width, "group_size": group, "count": count, "mode": mode,
                      "status": status, "decode_status": decoded, "groups": groups,
                      "packed": payload, "codes": codes})
    return cases


class QintLanguageIdentity(unittest.TestCase):
    """The language, format.py and the independent oracle emit the same bytes."""

    emitted = {}

    @classmethod
    def setUpClass(cls):
        if not shutil.which("clang"):
            raise unittest.SkipTest("clang required to build the packed-level kernels")
        cls.temp = tempfile.TemporaryDirectory(prefix="nexa-qint-")
        directory = Path(cls.temp.name)
        for width in WIDTHS:
            program = directory / f"pack{width}.nxl"
            program.write_text(EMITTER.format(width=width), encoding="utf-8")
            for mode in ("native", "jit"):
                arguments = ["--jit"] if mode == "jit" else []
                result = subprocess.run(
                    [sys.executable, str(ROOT / "nx.py"), "run", str(program), *arguments],
                    cwd=directory, text=True, capture_output=True, timeout=300)
                output = result.stdout + result.stderr
                assert result.returncode == 0, output
                cls.emitted[(width, mode)] = parse_cases(output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_language_bytes_match_format_py_and_the_independent_oracle(self):
        seen = 0
        for width in WIDTHS:
            for case in self.emitted[(width, "native")]:
                with self.subTest(width=width, group=case["group_size"],
                                  count=case["count"], mode=case["mode"]):
                    values = [sample(index, case["mode"]) for index in range(case["count"])]
                    self.assertEqual(case["status"], 0)
                    runtime = FORMAT_WRITER[width](values, case["group_size"])
                    oracle = qint_reference.pack(values, width, case["group_size"])
                    self.assertEqual(case["packed"], runtime)
                    self.assertEqual(case["packed"], oracle)
                    seen += 1
        self.assertEqual(seen, len(WIDTHS) * len(GROUP_SIZES) * len(VALUE_COUNTS) * len(MODES))

    def test_jit_and_native_agree_byte_for_byte(self):
        for width in WIDTHS:
            native, jit = self.emitted[(width, "native")], self.emitted[(width, "jit")]
            self.assertEqual(len(native), len(jit))
            for left, right in zip(native, jit):
                with self.subTest(width=width, group=left["group_size"], count=left["count"]):
                    self.assertEqual(left, right)

    def test_packed_size_and_group_count_follow_the_layout(self):
        for width in WIDTHS:
            for case in self.emitted[(width, "native")]:
                with self.subTest(width=width, group=case["group_size"], count=case["count"]):
                    self.assertEqual(case["groups"],
                                     qint_reference.group_count(width, case["count"],
                                                                case["group_size"]))
                    self.assertEqual(len(case["packed"]),
                                     qint_reference.packed_size(width, case["count"],
                                                                case["group_size"]))

    def test_qint_codes_read_back_signed_and_match_the_oracle(self):
        negatives = 0
        for width in WIDTHS:
            for case in self.emitted[(width, "native")]:
                with self.subTest(width=width, group=case["group_size"],
                                  count=case["count"], mode=case["mode"]):
                    expected = qint_reference.codes(case["packed"], width, case["count"],
                                                    case["group_size"])
                    self.assertEqual(case["codes"], expected)
                    limit = qint_reference.levels(width)
                    for code in case["codes"]:
                        self.assertTrue(-limit <= code <= limit, code)
                    negatives += sum(1 for code in case["codes"] if code < 0)
        self.assertGreater(negatives, 0, "the fixture never exercised a negative code")

    def test_unpack_reproduces_the_format_py_decode(self):
        for width in WIDTHS:
            for case in self.emitted[(width, "native")]:
                with self.subTest(width=width, group=case["group_size"], count=case["count"]):
                    self.assertEqual(case["decode_status"], 0)
                    runtime = FORMAT_READER[width](case["packed"], case["count"],
                                                   case["group_size"])
                    oracle = qint_reference.unpack(case["packed"], width, case["count"],
                                                   case["group_size"])
                    self.assertEqual(runtime, oracle)

    def test_bits_per_value_measured_from_the_emitted_buffer(self):
        """The published cost table, read off the bytes the language wrote."""
        measured = {}
        for width in WIDTHS:
            for case in self.emitted[(width, "native")]:
                key = (width, case["group_size"], case["count"])
                bits = 8.0 * len(case["packed"]) / case["count"]
                if key in measured:  # the value pattern must not move the cost
                    self.assertAlmostEqual(measured[key], bits, places=12)
                measured[key] = bits
        self.assertEqual(len(measured), len(WIDTHS) * len(GROUP_SIZES) * len(VALUE_COUNTS))

        # Whole groups only: this is the steady-state cost the docs publish.
        full = {}
        for (width, group_size, count), bits in measured.items():
            if count % group_size:
                continue
            record_bits = 8.0 * (4 + -(-width * group_size // 8))
            self.assertAlmostEqual(bits, record_bits / group_size, places=12)
            full.setdefault((width, group_size), bits)

        # The four-byte scale is the whole overhead, and it is shared by the
        # group: exactly 32/group_size bits per value wherever the codes
        # themselves fill whole bytes. Group 32 pays one bit, group 8 pays four.
        for (width, group_size), bits in full.items():
            if (width * group_size) % 8 == 0:
                self.assertAlmostEqual(bits - width, 32.0 / group_size, places=12)
        for width in WIDTHS:
            self.assertAlmostEqual(full[(width, 32)], width + 1.0, places=12)
            self.assertAlmostEqual(full[(width, 8)], width + 4.0, places=12)

        # Spot values, spelled out, so a wrong layout formula cannot agree with
        # itself on both sides of the comparison above.
        self.assertAlmostEqual(full[(4, 32)], 5.0, places=12)
        self.assertAlmostEqual(full[(2, 32)], 3.0, places=12)
        self.assertAlmostEqual(full[(8, 8)], 12.0, places=12)
        self.assertAlmostEqual(full[(3, 16)], 5.0, places=12)
        self.assertAlmostEqual(full[(8, 1)], 40.0, places=12)
        # A tail is not free: 33 values at group 32 buy a second whole record.
        self.assertAlmostEqual(measured[(4, 32, 33)], 8.0 * 40 / 33, places=12)
        self.assertAlmostEqual(measured[(2, 33, 64)], 8.0 * 26 / 64, places=12)


class QintCompilerGates(unittest.TestCase):
    """Widths, code literals and extern ABIs the compiler must refuse by name."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nexa-qint-gate-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def compile(self, source):
        program = self.directory / "main.nxl"
        program.write_text(source, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(ROOT / "bootstrap/main.py"), str(program),
             "--target", "native", "--emit", "ll", "--out", str(self.directory / "out.ll")],
            cwd=self.directory, text=True, capture_output=True, timeout=120)

    def assert_refused(self, source, code, fragment):
        result = self.compile(source)
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertIn(code, output)
        self.assertIn(fragment, output)

    def assert_accepted(self, source):
        result = self.compile(source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_widths_without_a_codec_are_refused(self):
        for width in (0, 1, 5, 6, 16, 32):
            with self.subTest(width=width):
                self.assert_refused(
                    'fn main() -> i32 { let x: qint<%d> = 0; return 0; }' % width,
                    "E0008", "has no packed codec")
                self.assert_refused(
                    'fn main() -> i32 { let x: PackedVector<%d> = '
                    'cast::<PackedVector<%d>>(0); return 0; }' % (width, width),
                    "E0008", "has no packed codec")

    def test_supported_widths_are_accepted(self):
        for width in WIDTHS:
            with self.subTest(width=width):
                self.assert_accepted(
                    'fn main() -> i32 { let x: qint<%d> = 0; return cast::<i32>(x); }' % width)

    def test_code_literals_outside_the_codec_range_are_refused(self):
        for width in WIDTHS:
            limit = qint_reference.levels(width)
            with self.subTest(width=width):
                self.assert_accepted('fn main() -> i32 { let x: qint<%d> = %d; '
                                     'return cast::<i32>(x); }' % (width, -limit))
                self.assert_accepted('fn main() -> i32 { let x: qint<%d> = %d; '
                                     'return cast::<i32>(x); }' % (width, limit))
                # -2**(N-1) is the reserved code, so it is rejected like any
                # other value the storage contract says cannot be written.
                self.assert_refused('fn main() -> i32 { let x: qint<%d> = %d; return 0; }'
                                    % (width, -limit - 1), "E0008", "is reserved")
                self.assert_refused('fn main() -> i32 { let x: qint<%d> = %d; return 0; }'
                                    % (width, limit + 1), "E0008", "is not a qint")

    def test_qpack_requires_a_packed_vector_and_an_explicit_width(self):
        self.assert_refused(
            'extern "C" { fn malloc(size: i32) -> *u8; }\n'
            'fn main() -> i32 { let p = malloc(64); '
            'return qpack::pack(p, cast::<*f32>(malloc(16)), 4, 4); }',
            "E0002", "expects a PackedVector<N>")
        self.assert_refused(
            'fn main() -> i32 { return cast::<i32>(qpack::size(4, 4)); }',
            "E0008", "needs an explicit width")
        self.assert_refused(
            'fn main() -> i32 { return cast::<i32>(qpack::size::<5>(4, 4)); }',
            "E0008", "has no packed codec")
        self.assert_refused(
            'extern "C" { fn malloc(size: i32) -> *u8; }\n'
            'fn main() -> i32 { let p = cast::<PackedVector<4>>(malloc(64)); '
            'return qpack::squeeze(p, 4, 4, 0); }',
            "E0004", "Unknown intrinsic")

    def test_extern_abi_the_backend_cannot_emit_is_refused(self):
        """This compiled and exited 0 before the ABI was checked anywhere."""
        for abi in ('Fortran-77', 'stdcall', 'Rust', 'c', ''):
            with self.subTest(abi=abi):
                self.assert_refused(
                    'extern "%s" { fn nope(x: i32) -> i32; }\n'
                    'fn main() -> i32 { return nope(1); }' % abi,
                    "E0009", "Unsupported extern ABI")

    def test_extern_c_still_compiles(self):
        self.assert_accepted('extern "C" { fn abs(x: i32) -> i32; }\n'
                             'fn main() -> i32 { return abs(-1) - 1; }')


if __name__ == "__main__":
    unittest.main()
