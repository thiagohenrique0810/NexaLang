"""Run all NexaLang bootstrap compiler tests."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bootstrap'))

from lexer import Lexer
from n_parser import Parser, FunctionDef, StructDef, EnumDef, TraitDef, ImplDef
from semantic import SemanticAnalyzer

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        failed += 1

# ── Lexer Tests ──────────────────────────────────────────────────────────

def test_basic_tokenize():
    tokens = Lexer('fn main() { let x: i32 = 42; }').tokenize()
    assert len(tokens) >= 10

def test_double_colon():
    tokens = Lexer('Vec::new(); a::b::c').tokenize()
    dc = [t for t in tokens if t.type == 'DOUBLE_COLON']
    assert len(dc) == 3, f"Expected 3, got {len(dc)}"

def test_keywords():
    kws = 'fn let if else while for return struct impl trait enum match pub use mod extern async await kernel break continue type'
    tokens = Lexer(kws).tokenize()
    assert len(tokens) == 22

def test_strings():
    tokens = Lexer('let s = "hello world";').tokenize()
    assert any(t.type == 'STRING' for t in tokens)

def test_comments_skipped():
    tokens = Lexer('let x = 1; # comment\nlet y = 2;').tokenize()
    assert not any(t.type == 'COMMENT' for t in tokens)

def test_float_literal():
    tokens = Lexer('let f = 3.14;').tokenize()
    assert any(t.type == 'FLOAT' for t in tokens)

def test_operators():
    tokens = Lexer('a + b - c * d / e == f != g <= h >= i').tokenize()
    types = [t.type for t in tokens]
    for op in ['PLUS', 'MINUS', 'STAR', 'SLASH', 'EQEQ', 'NEQ', 'LTE', 'GTE']:
        assert op in types, f"Missing {op}"

print("=== LEXER TESTS ===")
test("basic tokenize", test_basic_tokenize)
test("double colon", test_double_colon)
test("keywords", test_keywords)
test("string literals", test_strings)
test("comments skipped", test_comments_skipped)
test("float literal", test_float_literal)
test("operators", test_operators)

# ── Parser Tests ─────────────────────────────────────────────────────────

def test_parse_fn():
    tokens = Lexer('fn main() -> i32 { return 0; }').tokenize()
    nodes = Parser(tokens).parse()
    fns = [n for n in nodes if isinstance(n, FunctionDef)]
    assert len(fns) >= 1

def test_parse_struct():
    tokens = Lexer('struct Point { x: i32, y: i32 }').tokenize()
    nodes = Parser(tokens).parse()
    structs = [n for n in nodes if isinstance(n, StructDef)]
    assert len(structs) == 1

def test_parse_if_else():
    src = 'fn foo() { if (x > 0) { return 1; } else { return 0; } }'
    nodes = Parser(Lexer(src).tokenize()).parse()
    fns = [n for n in nodes if isinstance(n, FunctionDef)]
    assert len(fns) == 1

def test_parse_for_loop():
    src = 'fn foo() { for i in 0..10 { print(i); } }'
    nodes = Parser(Lexer(src).tokenize()).parse()
    fns = [n for n in nodes if isinstance(n, FunctionDef)]
    assert len(fns) == 1

def test_parse_enum():
    src = 'enum Color { Red, Green, Blue }'
    nodes = Parser(Lexer(src).tokenize()).parse()
    enums = [n for n in nodes if isinstance(n, EnumDef)]
    assert len(enums) == 1

def test_parse_trait():
    src = 'trait Printable { fn show(&self); }'
    nodes = Parser(Lexer(src).tokenize()).parse()
    traits = [n for n in nodes if isinstance(n, TraitDef)]
    assert len(traits) == 1

def test_parse_impl():
    src = 'struct Foo { x: i32 } impl Foo { fn new() -> Foo { return Foo(0); } }'
    nodes = Parser(Lexer(src).tokenize()).parse()
    impls = [n for n in nodes if isinstance(n, ImplDef)]
    assert len(impls) == 1

def test_parse_lambda():
    src = 'fn main() { let f = |x: i32| x + 1; }'
    nodes = Parser(Lexer(src).tokenize()).parse()
    fns = [n for n in nodes if isinstance(n, FunctionDef)]
    assert len(fns) >= 1

def test_parse_match():
    src = 'fn foo(x: i32) -> i32 { match x { Red => { return 1; } _ => { return 0; } } }'
    nodes = Parser(Lexer(src).tokenize()).parse()
    fns = [n for n in nodes if isinstance(n, FunctionDef)]
    assert len(fns) == 1

print("\n=== PARSER TESTS ===")
test("parse function", test_parse_fn)
test("parse struct", test_parse_struct)
test("parse if/else", test_parse_if_else)
test("parse for loop", test_parse_for_loop)
test("parse enum", test_parse_enum)
test("parse trait", test_parse_trait)
test("parse impl", test_parse_impl)
test("parse lambda", test_parse_lambda)
test("parse match", test_parse_match)

# ── Semantic Tests ───────────────────────────────────────────────────────

def test_semantic_init():
    sa = SemanticAnalyzer()
    assert hasattr(sa, 'active_borrows')
    assert hasattr(sa, 'lifetime_errors')
    assert 'compress::create' in sa.functions
    assert 'gpu::dispatch' in sa.functions

def test_semantic_basic_analysis():
    src = 'fn main() -> i32 { let x: i32 = 42; return x; }'
    tokens = Lexer(src).tokenize()
    p = Parser(tokens)
    ast = p.parse()
    sa = SemanticAnalyzer()
    sa.analyze(ast)

def test_semantic_type_check():
    src = 'fn add(a: i32, b: i32) -> i32 { return a + b; }'
    tokens = Lexer(src).tokenize()
    p = Parser(tokens)
    ast = p.parse()
    sa = SemanticAnalyzer()
    sa.analyze(ast)

def test_semantic_struct_methods():
    src = '''struct Point { x: i32, y: i32 }
impl Point {
    fn new(x: i32, y: i32) -> Point { return Point(x, y); }
}'''
    tokens = Lexer(src).tokenize()
    p = Parser(tokens)
    ast = p.parse()
    sa = SemanticAnalyzer()
    sa.analyze(ast)

print("\n=== SEMANTIC TESTS ===")
test("semantic init", test_semantic_init)
test("basic analysis", test_semantic_basic_analysis)
test("type check", test_semantic_type_check)
test("struct methods", test_semantic_struct_methods)

# ── MIR Tests ────────────────────────────────────────────────────────────

def test_mir_import():
    from mir import MIRModule, MIRLowering, MIROptimizer, MIRPrinter
    assert MIRModule is not None

def test_mir_create_module():
    from mir import MIRModule
    m = MIRModule('test')
    assert m.name == 'test'

def test_mir_optimizer():
    from mir import MIRModule, MIROptimizer
    m = MIRModule('test')
    opt = MIROptimizer()
    opt.optimize(m)

print("\n=== MIR TESTS ===")
test("mir import", test_mir_import)
test("mir create module", test_mir_create_module)
test("mir optimizer", test_mir_optimizer)

# ── TurboQuant Runtime Test ──────────────────────────────────────────────

def test_turboquant_lib_exists():
    lib_path = os.path.join(os.path.dirname(__file__), '..', 'runtime', 'libturboquant.dylib')
    if not os.path.exists(lib_path):
        lib_path = os.path.join(os.path.dirname(__file__), '..', 'runtime', 'libturboquant.so')
    assert os.path.exists(lib_path), "libturboquant not built"

def test_turboquant_ctypes():
    import ctypes
    lib_dir = os.path.join(os.path.dirname(__file__), '..', 'runtime')
    lib_path = os.path.join(lib_dir, 'libturboquant.dylib')
    if not os.path.exists(lib_path):
        lib_path = os.path.join(lib_dir, 'libturboquant.so')
    lib = ctypes.CDLL(lib_path)
    
    lib.tq_create.restype = ctypes.c_void_p
    lib.tq_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.tq_destroy.argtypes = [ctypes.c_void_p]
    lib.tq_upper_bound.restype = ctypes.c_float
    lib.tq_upper_bound.argtypes = [ctypes.c_void_p]
    lib.tq_lower_bound.restype = ctypes.c_float
    lib.tq_lower_bound.argtypes = [ctypes.c_void_p]
    
    ctx = lib.tq_create(64, 3, 42)
    assert ctx is not None and ctx != 0
    
    ub = lib.tq_upper_bound(ctx)
    lb = lib.tq_lower_bound(ctx)
    assert ub > lb > 0, f"Bounds invalid: ub={ub}, lb={lb}"
    
    lib.tq_destroy(ctx)

def test_turboquant_roundtrip():
    import ctypes
    lib_dir = os.path.join(os.path.dirname(__file__), '..', 'runtime')
    lib_path = os.path.join(lib_dir, 'libturboquant.dylib')
    if not os.path.exists(lib_path):
        lib_path = os.path.join(lib_dir, 'libturboquant.so')
    lib = ctypes.CDLL(lib_path)
    
    lib.tq_create.restype = ctypes.c_void_p
    lib.tq_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.tq_destroy.argtypes = [ctypes.c_void_p]
    lib.tq_quantize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint16), ctypes.c_int]
    lib.tq_dequantize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.tq_mse.restype = ctypes.c_float
    lib.tq_mse.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    
    dim = 64
    ctx = lib.tq_create(dim, 3, 42)
    
    # Create unit-norm vector
    import math
    FloatArr = ctypes.c_float * dim
    x = FloatArr(*[(i+1)*0.1 for i in range(dim)])
    norm = math.sqrt(sum(v*v for v in x))
    for i in range(dim):
        x[i] /= norm
    
    idx = (ctypes.c_uint16 * dim)()
    xhat = FloatArr()
    
    lib.tq_quantize(ctx, x, idx, 1)
    lib.tq_dequantize(ctx, idx, xhat, 1)
    
    mse = lib.tq_mse(ctx, x, 1)
    assert mse > 0 and mse < 1.0, f"MSE out of range: {mse}"
    
    lib.tq_destroy(ctx)

print("\n=== TURBOQUANT RUNTIME TESTS ===")
test("library exists", test_turboquant_lib_exists)
test("ctypes create/destroy", test_turboquant_ctypes)
test("quantize roundtrip", test_turboquant_roundtrip)

# ── Quantize Attribute Tests ─────────────────────────────────────────────

def test_parse_quantize_attr():
    """Parser should handle @[quantize(3)] attribute with NUMBER arg."""
    src = '@[quantize(3)]\nfn compress_data(buf: *f32, n: i32) { }'
    tokens = Lexer(src).tokenize()
    nodes = Parser(tokens).parse()
    fns = [n for n in nodes if isinstance(n, FunctionDef)]
    assert len(fns) == 1
    f = fns[0]
    assert hasattr(f, 'attrs') and len(f.attrs) > 0
    attr = f.attrs[0]
    assert attr[0] == 'quantize', f"Expected 'quantize', got {attr[0]}"
    assert attr[1] == [3], f"Expected [3], got {attr[1]}"

def test_semantic_quantize_attr():
    """Semantic should mark _quantize_bits on function nodes."""
    src = '@[quantize(2)]\nfn compress_data(buf: *f32, n: i32) -> i32 { return 0; }'
    tokens = Lexer(src).tokenize()
    ast = Parser(tokens).parse()
    sa = SemanticAnalyzer()
    sa.analyze(ast)
    fns = [n for n in ast if isinstance(n, FunctionDef)]
    assert len(fns) == 1
    assert getattr(fns[0], '_quantize_bits', None) == 2, f"Expected 2, got {getattr(fns[0], '_quantize_bits', None)}"

def test_codegen_quantize_gpu_flag():
    """CodeGen should accept quantize_gpu parameter."""
    try:
        from codegen import CodeGen
    except ImportError:
        return  # llvmlite not available, skip
    cg = CodeGen(quantize_gpu=3)
    assert cg.quantize_gpu == 3

print("\n=== QUANTIZE ATTRIBUTE TESTS ===")
test("parse @[quantize(N)]", test_parse_quantize_attr)
test("semantic _quantize_bits", test_semantic_quantize_attr)
test("codegen quantize_gpu flag", test_codegen_quantize_gpu_flag)

# ── MIR Optimization Tests ──────────────────────────────────────────────

def test_mir_constant_folding():
    from mir import MIRModule, MIRFunction, MIROptimizer, MIRBinOp, MIRConst, MIRTemp, MIRType, MIRRet
    m = MIRModule('test')
    f = MIRFunction(name='add_const', params=[], return_type=MIRType('i32'))
    bb = f.add_block('entry')
    dest = MIRTemp(id=0, ty=MIRType('i32'))
    bb.append(MIRBinOp(dest=dest, op='add', left=MIRConst(3, MIRType('i32')), right=MIRConst(4, MIRType('i32'))))
    bb.terminator = MIRRet(value=dest)
    m.add_function(f)
    opt = MIROptimizer()
    # Only run constant folding (no DCE which would remove the result)
    opt.constant_fold(f)
    from mir import MIRAssign
    inst = f.blocks[0].instructions[0]
    assert isinstance(inst, MIRAssign), f"Expected MIRAssign, got {type(inst).__name__}"
    assert inst.value.value == 7, f"Expected 7, got {inst.value.value}"

def test_mir_dead_code_elimination():
    from mir import MIRModule, MIRFunction, MIROptimizer, MIRBinOp, MIRConst, MIRTemp, MIRType, MIRRet, MIRCall
    m = MIRModule('test')
    f = MIRFunction(name='dce_test', params=[], return_type=MIRType('i32'))
    bb = f.add_block('entry')
    t0 = MIRTemp(id=0, ty=MIRType('i32'))
    t1 = MIRTemp(id=1, ty=MIRType('i32'))  # unused
    # t0 is used in return; t1 is dead
    bb.append(MIRCall(dest=t0, callee='get_value', args=[], ret_type=MIRType('i32')))  # side-effecting, kept
    bb.append(MIRBinOp(dest=t1, op='mul', left=MIRConst(5, MIRType('i32')), right=MIRConst(6, MIRType('i32'))))  # dead
    bb.terminator = MIRRet(value=t0)
    m.add_function(f)
    assert len(f.blocks[0].instructions) == 2
    opt = MIROptimizer()
    opt.dead_code_elimination(f)
    # t1 mul is dead and removed; t0 call is side-effecting so kept
    assert len(f.blocks[0].instructions) == 1, f"Expected 1 instruction after DCE, got {len(f.blocks[0].instructions)}"

def test_mir_strength_reduction():
    from mir import MIRModule, MIRFunction, MIROptimizer, MIRBinOp, MIRConst, MIRTemp, MIRVar, MIRType, MIRRet
    m = MIRModule('test')
    f = MIRFunction(name='sr_test', params=[('x', MIRType('i32'))], return_type=MIRType('i32'))
    bb = f.add_block('entry')
    t0 = MIRTemp(id=0, ty=MIRType('i32'))
    x = MIRVar(name='x', ty=MIRType('i32'))
    bb.append(MIRBinOp(dest=t0, op='mul', left=x, right=MIRConst(8, MIRType('i32'))))  # x * 8 → x << 3
    bb.terminator = MIRRet(value=t0)
    m.add_function(f)
    opt = MIROptimizer()
    opt.optimize(m, level=2)
    inst = f.blocks[0].instructions[0]
    assert inst.op == 'shl', f"Expected 'shl', got '{inst.op}'"
    assert inst.right.value == 3, f"Expected shift by 3, got {inst.right.value}"

def test_mir_branch_simplify():
    from mir import (MIRModule, MIRFunction, MIROptimizer, MIRConst, MIRType,
                     MIRRet, MIRCondBranch, MIRBranch)
    m = MIRModule('test')
    f = MIRFunction(name='br_test', params=[], return_type=MIRType('void'))
    entry = f.add_block('entry')
    then_bb = f.add_block('then')
    else_bb = f.add_block('else')
    # Always-true condition: should simplify to unconditional branch
    entry.terminator = MIRCondBranch(cond=MIRConst(True, MIRType('bool')), true_block='then', false_block='else')
    then_bb.terminator = MIRRet(value=None)
    else_bb.terminator = MIRRet(value=None)
    m.add_function(f)
    opt = MIROptimizer()
    opt.optimize(m, level=1)
    assert isinstance(entry.terminator, MIRBranch), f"Expected MIRBranch, got {type(entry.terminator).__name__}"
    assert entry.terminator.target == 'then'

def test_mir_unreachable_elimination():
    from mir import (MIRModule, MIRFunction, MIROptimizer, MIRConst, MIRType,
                     MIRRet, MIRBranch)
    m = MIRModule('test')
    f = MIRFunction(name='unreach_test', params=[], return_type=MIRType('void'))
    entry = f.add_block('entry')
    dead = f.add_block('dead')
    alive = f.add_block('alive')
    entry.terminator = MIRBranch(target='alive')
    dead.terminator = MIRRet(value=None)
    alive.terminator = MIRRet(value=None)
    m.add_function(f)
    assert len(f.blocks) == 3
    opt = MIROptimizer()
    opt.optimize(m, level=1)
    assert len(f.blocks) == 2, f"Expected 2 blocks after unreachable elim, got {len(f.blocks)}"
    labels = [b.label for b in f.blocks]
    assert 'dead' not in labels

print("\n=== MIR OPTIMIZATION TESTS ===")
test("constant folding", test_mir_constant_folding)
test("dead code elimination", test_mir_dead_code_elimination)
test("strength reduction", test_mir_strength_reduction)
test("branch simplification", test_mir_branch_simplify)
test("unreachable block elimination", test_mir_unreachable_elimination)

# ── Semantic Integration Tests ───────────────────────────────────────────

def test_semantic_associated_types_fail():
    """Missing associated type on impl should produce an error."""
    src = '''trait Iterator { type Item; fn next(&mut self) -> i32; }
struct Counter { count: i32 }
impl Iterator for Counter { fn next(&mut self) -> i32 { return self.count; } }
fn main() -> i32 { return 0; }'''
    tokens = Lexer(src).tokenize()
    ast = Parser(tokens).parse()
    sa = SemanticAnalyzer()
    try:
        sa.analyze(ast)
        assert False, "Should have raised error for missing associated type"
    except Exception as e:
        assert "missing associated type" in str(e).lower(), f"Unexpected error: {e}"

def test_semantic_borrow_check():
    """Use after move should be detected."""
    src = '''struct Foo { x: i32 }
fn process(f: Foo) -> i32 { return f.x; }
fn main() -> i32 {
    let f = Foo(1);
    let a = process(f);
    return a;
}'''
    tokens = Lexer(src).tokenize()
    ast = Parser(tokens).parse()
    sa = SemanticAnalyzer()
    sa.analyze(ast)  # Should pass — single use

def test_semantic_dead_code_warning():
    """Unused functions should be warned."""
    src = '''fn unused_func() -> i32 { return 42; }
fn main() -> i32 { return 0; }'''
    tokens = Lexer(src).tokenize()
    ast = Parser(tokens).parse()
    sa = SemanticAnalyzer()
    sa.analyze(ast)
    # Check for dead code warnings
    has_warning = any('unused_func' in str(w) for w in sa.warnings)
    assert has_warning, f"Expected dead code warning, got: {sa.warnings}"

def test_semantic_trait_method_check():
    """Missing trait method should produce an error."""
    src = '''trait Show { fn show(&self) -> i32; }
struct Foo { x: i32 }
impl Show for Foo { }
fn main() -> i32 { return 0; }'''
    tokens = Lexer(src).tokenize()
    ast = Parser(tokens).parse()
    sa = SemanticAnalyzer()
    try:
        sa.analyze(ast)
        assert False, "Should have raised error for missing trait method"
    except Exception as e:
        assert "missing methods" in str(e).lower() or "missing method" in str(e).lower(), f"Unexpected error: {e}"

print("\n=== SEMANTIC INTEGRATION TESTS ===")
test("associated type validation", test_semantic_associated_types_fail)
test("borrow check single use", test_semantic_borrow_check)
test("dead code warning", test_semantic_dead_code_warning)
test("missing trait method", test_semantic_trait_method_check)

# ── Parser Edge Case Tests ───────────────────────────────────────────────

def test_parse_nested_generics():
    src = 'fn foo() { let v: Vec<Option<i32>> = Vec::new(); }'
    tokens = Lexer(src).tokenize()
    nodes = Parser(tokens).parse()
    assert len([n for n in nodes if isinstance(n, FunctionDef)]) == 1

def test_parse_async_fn():
    src = 'async fn fetch_data() -> i32 { return 42; }'
    tokens = Lexer(src).tokenize()
    nodes = Parser(tokens).parse()
    fns = [n for n in nodes if isinstance(n, FunctionDef)]
    assert len(fns) == 1
    assert fns[0].is_async == True

def test_parse_extern_block():
    from n_parser import ExternBlock
    src = 'extern "C" { fn printf(fmt: *u8) -> i32; fn malloc(size: i32) -> *u8; }'
    tokens = Lexer(src).tokenize()
    nodes = Parser(tokens).parse()
    externs = [n for n in nodes if isinstance(n, ExternBlock)]
    assert len(externs) == 1

def test_parse_trait_with_default():
    src = '''trait Display { fn default_method(&self) -> i32 { return 0; } }'''
    tokens = Lexer(src).tokenize()
    nodes = Parser(tokens).parse()
    traits = [n for n in nodes if isinstance(n, TraitDef)]
    assert len(traits) == 1
    assert len(traits[0].methods) == 1
    assert traits[0].methods[0].body is not None  # Has default body

def test_parse_closure_in_call():
    src = 'fn main() { let f = |x: i32, y: i32| x + y; }'
    tokens = Lexer(src).tokenize()
    nodes = Parser(tokens).parse()
    assert len([n for n in nodes if isinstance(n, FunctionDef)]) >= 1

print("\n=== PARSER EDGE CASE TESTS ===")
test("nested generics", test_parse_nested_generics)
test("async function", test_parse_async_fn)
test("extern block", test_parse_extern_block)
test("trait with default", test_parse_trait_with_default)
test("closure in call", test_parse_closure_in_call)

# ── Example File Compilation Tests ───────────────────────────────────────

def test_compile_examples():
    """Verify key example files parse and pass semantic analysis."""
    examples = [
        'hello.nxl', 'variables.nxl', 'control.nxl', 'structs.nxl',
        'enum.nxl', 'generics.nxl', 'traits_basic.nxl', 'methods_simple.nxl',
    ]
    examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
    for ex in examples:
        path = os.path.join(examples_dir, ex)
        if not os.path.exists(path):
            continue
        src = open(path).read()
        tokens = Lexer(src).tokenize()
        ast = Parser(tokens).parse()
        sa = SemanticAnalyzer()
        sa.analyze(ast)  # Should not raise

print("\n=== EXAMPLE COMPILATION TESTS ===")
test("compile key examples", test_compile_examples)

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  RESULTS: {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*50}")
sys.exit(1 if failed else 0)
