"""Behavioral regressions for ownership, numeric conversion and control flow.

Run with: python3 -m unittest discover -s tests -p test_compiler_regressions.py
All binaries and generated IR are isolated in a temporary directory.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bootstrap"))

from lexer import Lexer
from n_parser import Parser
from semantic import SemanticAnalyzer
from codegen import CodeGen
from errors import CompilerError
from llvmlite import binding as llvm


RESOURCE = """
struct Resource { value: i32 }
impl Resource { fn drop(self) { print(self.value); } }
"""


def compile_source(source):
    ast = Parser(Lexer(source).tokenize()).parse()
    SemanticAnalyzer().analyze(ast)
    ir = CodeGen().generate(ast)
    llvm.parse_assembly(ir).verify()
    return ir


class CompilerRegressions(unittest.TestCase):
    def run_source(self, source, expected, status=0, optimization="-O0"):
        ir = compile_source(source)
        clang = shutil.which("clang")
        if not clang:
            self.skipTest("clang is required for native behavioral tests")
        with tempfile.TemporaryDirectory(prefix="nexa-regression-") as directory:
            ll = Path(directory) / "test.ll"
            executable = Path(directory) / ("test.exe" if os.name == 'nt' else "test")
            ll.write_text(ir)
            linked = subprocess.run([clang, str(ll), optimization, "-o", str(executable)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(linked.returncode, 0, linked.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, status, result.stderr)
            self.assertEqual(result.stdout, expected)

    def test_signed_division_and_remainder_match_constant_and_variable(self):
        source = """fn main() { let x = -7; let divisor = 2;
            print(x / 2); print(x / divisor); print(x % 2); print(x % divisor); }"""
        for optimization in ("-O0", "-O3"):
            with self.subTest(optimization=optimization):
                self.run_source(source, "-3\n-3\n-1\n-1\n", optimization=optimization)

    def test_mir_constant_folding_truncates_signed_division(self):
        from mir import MIROptimizer, MIRConst, MIRType
        optimizer = MIROptimizer()
        left, right = MIRConst(-7, MIRType('i32')), MIRConst(2, MIRType('i32'))
        self.assertEqual(optimizer._eval_binop('div', left, right).value, -3)
        self.assertEqual(optimizer._eval_binop('rem', left, right).value, -1)

    def test_numeric_assignment_converts_value(self):
        self.run_source("""fn main() {
            let x: f32 = 0.0; x = 1; print(x);
            let y: i64 = -7; print(y); y = -9; print(y);
            let z: i32 = 0; z = 9.75; print(z);
        }""", "1.000000\n-7\n-9\n9\n")

    def test_field_and_pointer_assignment_do_not_overwrite_neighbors(self):
        self.run_source("""struct Bytes { a: u8, b: u8 }
            fn main() {
                let pair = Bytes(0, 99); pair.a = 257;
                print(pair.a); print(pair.b);
                let array = [0, 0]; array[1] = 2.75; print(array[1]);
                let x: f32 = 0.0; let p = &x; *p = 3; print(x);
            }""", "1\n99\n2\n3.000000\n")

    def test_function_and_return_numeric_conversions(self):
        self.run_source("""fn f(x: f32) -> i64 { return x; }
            fn main() { print(f(12)); print(1 + 2.5); }""", "12\n3.500000\n")

    def test_if_shadow_keeps_outer_binding(self):
        self.run_source("""fn main() { let x = 1;
            if (true) { let x = 2; print(x); }
            else { let x = 3; print(x); }
            print(x); }""", "2\n1\n")

    def test_boolean_operators_short_circuit(self):
        self.run_source("""fn side() -> bool { print(99); return true; }
            fn main() { let a = false and side(); let b = true or side();
                print(a); print(b); }""", "0\n1\n")

    def test_move_has_exactly_one_drop(self):
        self.run_source(RESOURCE + """fn main() -> i32 {
            let a = Resource(7); let b = a; return 0; }""", "7\n")

    def test_implicit_return_drops_in_reverse_order(self):
        self.run_source(RESOURCE + """fn main() {
            let a = Resource(1); let b = Resource(2); }""", "2\n1\n")

    def test_return_transfers_ownership(self):
        self.run_source(RESOURCE + """fn make() -> Resource {
            let a = Resource(7); return a; }
            fn main() { let b = make(); print(1); }""", "1\n7\n")

    def test_call_transfers_ownership(self):
        self.run_source(RESOURCE + """fn take(a: Resource) { print(1); }
            fn main() { let a = Resource(7); take(a); print(2); }""", "1\n7\n2\n")

    def test_explicit_drop_is_not_repeated(self):
        self.run_source(RESOURCE + """fn main() {
            let a = Resource(7); a.drop(); print(1); }""", "7\n1\n")

    def test_reassignment_drops_old_owner(self):
        self.run_source(RESOURCE + """fn main() {
            let a = Resource(1); a = Resource(2); print(3); }""", "1\n3\n2\n")

    def test_owned_fields_are_transferred_and_remaining_fields_dropped(self):
        self.run_source(RESOURCE + """struct Pair { a: Resource, b: Resource }
            fn main() { let p = Pair(Resource(1), Resource(2));
                let a = p.a; p.a = Resource(3); print(p.b.value); }
            """, "2\n1\n2\n3\n")

    def test_reading_moved_field_is_rejected(self):
        with self.assertRaisesRegex(CompilerError, "moved field"):
            compile_source(RESOURCE + """struct Box { item: Resource }
                fn main() { let b = Box(Resource(1)); let a = b.item; print(b.item.value); }
                """)

    def test_mutating_method_updates_struct_field(self):
        self.run_source("""struct Counter { value: i32 }
            impl Counter { fn add(&mut self) { self.value = self.value + 1; } }
            struct Outer { counter: Counter }
            fn main() { let outer = Outer(Counter(1)); outer.counter.add(); print(outer.counter.value); }
            """, "2\n")

    def test_loop_cannot_consume_same_outer_owner_twice(self):
        with self.assertRaisesRegex(CompilerError, "next loop iteration"):
            compile_source(RESOURCE + """fn take(a: Resource) { }
                fn main() { let a = Resource(1); for i in 0..2 { take(a); } }
                """)

    def test_conditional_move_drops_only_live_owner(self):
        self.run_source(RESOURCE + """fn take(a: Resource) { }
            fn choose(flag: bool) {
                let a = Resource(7); if (flag) { take(a); }
            }
            fn main() { choose(true); choose(false); }""", "7\n7\n")

    def test_both_branches_can_consume_same_owner(self):
        self.run_source(RESOURCE + """fn take(a: Resource) { }
            fn main() { let a = Resource(7);
                if (true) { take(a); } else { take(a); }
            }""", "7\n")

    def test_break_continue_and_nested_return_drop_once(self):
        self.run_source(RESOURCE + """fn stop() -> i32 {
                while (true) { let a = Resource(8); return 0; }
                return 1;
            }
            fn main() {
                for i in 0..3 { let a = Resource(i); if (i == 1) { continue; }
                    if (i == 2) { break; }
                }
                stop();
            }""", "0\n1\n2\n8\n")

    def test_indirect_local_reference_cannot_escape(self):
        with self.assertRaisesRegex(CompilerError, "Lifetime Error"):
            compile_source("fn escape() -> i32* { let x = 1; let p = &x; return p; }")

    def test_inner_reference_cannot_be_assigned_to_outer_scope(self):
        with self.assertRaisesRegex(CompilerError, "Lifetime Error"):
            compile_source("""fn main() { let x = 1; let p = &x;
                { let inner = 2; p = &inner; } }""")

    def test_pointer_parameter_can_be_returned(self):
        self.run_source("""fn identity(p: i32*) -> i32* { return p; }
            fn main() { let x = 7; let p = identity(&x); print(*p); }""", "7\n")

    def test_scalar_copy_does_not_inherit_pointer_lifetime(self):
        self.run_source("""struct Value { number: i32 }
            fn make() -> Value { let number = 7; let p = &number; return Value(*p); }
            fn main() { let value = make(); print(value.number); }
            """, "7\n")

    def test_named_function_callback_stored_in_field(self):
        self.run_source("""struct Callback { call: fn(i32)->i32 }
            fn twice(value: i32) -> i32 { return value * 2; }
            fn main() { let cb = Callback(twice); print((cb.call)(21)); }
            """, "42\n")

    def test_relative_nested_module_call(self):
        from modules import resolve_modules
        with tempfile.TemporaryDirectory(prefix="nexa-modules-") as directory:
            path = Path(directory)
            (path / 'a.nxl').write_text('mod b; pub fn value()->i32 { return b::value(); }')
            (path / 'b.nxl').write_text('pub fn value()->i32 { return 42; }')
            ast = resolve_modules(Parser(Lexer('mod a; fn main(){ print(a::value()); }').tokenize()).parse(), path)
            SemanticAnalyzer().analyze(ast)
            llvm.parse_assembly(CodeGen().generate(ast)).verify()

    def test_captured_closure_runs(self):
        self.run_source((ROOT / "examples/closure_test.nxl").read_text(), "50\n")

    def test_captured_closure_cannot_escape_stack(self):
        with self.assertRaisesRegex(CompilerError, "Lifetime Error"):
            compile_source("""fn make() -> fn(i32)->i32 {
                let factor = 2; let f = |x: i32| x * factor; return f; }
            """)

    def test_slices_work(self):
        self.run_source((ROOT / "examples/slices.nxl").read_text(), "4\n30\n")

    def test_slice_cannot_outlive_local_array(self):
        with self.assertRaisesRegex(CompilerError, "Lifetime Error"):
            compile_source("""fn escape() -> []i32 {
                let a = [1, 2]; let s = slice_from_array(&a); return s; }
            """)

    def test_generic_vec_methods_are_specialized(self):
        self.run_source((ROOT / "examples/vec_test.nxl").read_text(), "3\n1\n")

    def test_distinct_generic_types_are_not_interchangeable(self):
        with self.assertRaisesRegex(CompilerError, "Type Error"):
            compile_source("""struct Box<T> { value: T }
                fn main() { let b: Box<i32> = Box::<f32>(1.0); }""")

    def test_async_main_awaits_completed_state(self):
        self.run_source((ROOT / "examples/async_test.nxl").read_text(),
                        "Main starting...\nInside async fn...\nAwaited value:\n42\n")

    def test_async_state_reserves_full_result_layout(self):
        self.run_source("""struct Big { a: i64, b: i64, c: i64, d: i64 }
            async fn make() -> Big { return Big(1, 2, 3, 4); }
            async fn main() { let task = make(); let value = await task;
                print(value.a); print(value.d); }""", "1\n4\n")

    def test_await_consumes_task(self):
        with self.assertRaisesRegex(CompilerError, "moved variable"):
            compile_source("""async fn make() -> i32 { return 1; }
                async fn main() { let t = make(); let a = await t; let b = await t; }
            """)

    def test_unawaited_task_releases_owned_result(self):
        self.run_source(RESOURCE + """async fn make() -> Resource { return Resource(7); }
            fn main() { let task = make(); print(1); }""", "1\n7\n")

    def test_unsafe_gpu_quantization_reports_error(self):
        with self.assertRaisesRegex(ValueError, "does not preserve"):
            CodeGen(quantize_gpu=3)

    def test_assert_reports_failure(self):
        self.run_source('fn main() { assert(false, "expected failure"); }',
                        "ASSERTION FAILED at unknown:1: expected failure\n", status=1)


if __name__ == "__main__":
    unittest.main()
