"""
NexaLang MIR (Mid-level Intermediate Representation)

Sits between the AST and LLVM IR codegen. Provides:
- Flat basic-block structure (CFG) instead of nested AST trees
- Explicit temporaries (SSA-like) for all intermediate values
- Optimization passes: constant folding, dead code elimination, copy propagation
- Type-annotated instructions for easier analysis

The MIR is optional — the pipeline can still go AST → LLVM directly.
When enabled: AST → MIR → (optimize) → LLVM IR
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any


# ── MIR Types ──────────────────────────────────────────────────────────────────

@dataclass
class MIRType:
    """Base type representation in MIR."""
    name: str  # e.g. "i32", "bool", "MyStruct", "Vec<i32>"

    def __eq__(self, other):
        return isinstance(other, MIRType) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name

    def is_numeric(self):
        return self.name in ('i32', 'i64', 'u8', 'u64', 'f32', 'f64')

    def is_integer(self):
        return self.name in ('i32', 'i64', 'u8', 'u64', 'bool')

    def is_float(self):
        return self.name in ('f32', 'f64')

    def is_pointer(self):
        return self.name.endswith('*') or self.name.startswith('&')

    def is_void(self):
        return self.name == 'void'


# ── MIR Values ─────────────────────────────────────────────────────────────────

@dataclass
class MIRValue:
    """A value in MIR — could be a temporary, constant, or variable reference."""
    pass

@dataclass
class MIRTemp(MIRValue):
    """SSA temporary: %t0, %t1, ..."""
    id: int
    ty: MIRType

    def __repr__(self):
        return f"%t{self.id}"

@dataclass
class MIRConst(MIRValue):
    """Compile-time constant."""
    value: Any  # int, float, str, bool
    ty: MIRType

    def __repr__(self):
        return f"const({self.value}: {self.ty})"

@dataclass
class MIRVar(MIRValue):
    """Named variable reference."""
    name: str
    ty: MIRType

    def __repr__(self):
        return f"${self.name}"

@dataclass
class MIRGlobal(MIRValue):
    """Global symbol (function name, global var)."""
    name: str
    ty: MIRType

    def __repr__(self):
        return f"@{self.name}"


# ── MIR Instructions ──────────────────────────────────────────────────────────

@dataclass
class MIRInst:
    """Base class for all MIR instructions."""
    line: int = 0
    column: int = 0

@dataclass
class MIRAssign(MIRInst):
    """dest = value"""
    dest: MIRTemp = None
    value: MIRValue = None

@dataclass
class MIRBinOp(MIRInst):
    """dest = left op right"""
    dest: MIRTemp = None
    op: str = ""  # "add", "sub", "mul", "div", "rem", "eq", "ne", "lt", "gt", "le", "ge", "and", "or"
    left: MIRValue = None
    right: MIRValue = None

@dataclass
class MIRUnaryOp(MIRInst):
    """dest = op operand"""
    dest: MIRTemp = None
    op: str = ""  # "neg", "not", "deref", "addr_of"
    operand: MIRValue = None

@dataclass
class MIRCall(MIRInst):
    """dest = callee(args...)"""
    dest: Optional[MIRTemp] = None  # None for void calls
    callee: str = ""
    args: List[MIRValue] = field(default_factory=list)
    ret_type: MIRType = field(default_factory=lambda: MIRType("void"))

@dataclass
class MIRMethodCall(MIRInst):
    """dest = receiver.method(args...)"""
    dest: Optional[MIRTemp] = None
    receiver: MIRValue = None
    method: str = ""
    struct_type: str = ""
    args: List[MIRValue] = field(default_factory=list)
    ret_type: MIRType = field(default_factory=lambda: MIRType("void"))

@dataclass
class MIRAlloca(MIRInst):
    """dest = alloca(ty)"""
    dest: MIRTemp = None
    alloc_type: MIRType = None

@dataclass
class MIRStore(MIRInst):
    """*ptr = value"""
    ptr: MIRValue = None
    value: MIRValue = None

@dataclass
class MIRLoad(MIRInst):
    """dest = *ptr"""
    dest: MIRTemp = None
    ptr: MIRValue = None

@dataclass
class MIRGetField(MIRInst):
    """dest = base.field"""
    dest: MIRTemp = None
    base: MIRValue = None
    field_name: str = ""
    struct_type: str = ""

@dataclass
class MIRSetField(MIRInst):
    """base.field = value"""
    base: MIRValue = None
    field_name: str = ""
    struct_type: str = ""
    value: MIRValue = None

@dataclass
class MIRIndex(MIRInst):
    """dest = base[index]"""
    dest: MIRTemp = None
    base: MIRValue = None
    index: MIRValue = None

@dataclass
class MIRSetIndex(MIRInst):
    """base[index] = value"""
    base: MIRValue = None
    index: MIRValue = None
    value: MIRValue = None

@dataclass
class MIRCast(MIRInst):
    """dest = cast<target_type>(value)"""
    dest: MIRTemp = None
    value: MIRValue = None
    target_type: MIRType = None

@dataclass
class MIRConstructStruct(MIRInst):
    """dest = StructName(field_values...)"""
    dest: MIRTemp = None
    struct_name: str = ""
    fields: List[MIRValue] = field(default_factory=list)

@dataclass
class MIRConstructEnum(MIRInst):
    """dest = EnumName::Variant(payload...)"""
    dest: MIRTemp = None
    enum_name: str = ""
    variant: str = ""
    payload: List[MIRValue] = field(default_factory=list)

@dataclass
class MIRReturn(MIRInst):
    """return value"""
    value: Optional[MIRValue] = None

@dataclass
class MIRDrop(MIRInst):
    """Explicit drop call for RAII."""
    var: MIRVar = None
    type_name: str = ""


# ── MIR Terminators (end a basic block) ───────────────────────────────────────

@dataclass
class MIRTerminator:
    """Base for block terminators."""
    pass

@dataclass
class MIRBranch(MIRTerminator):
    """Unconditional jump."""
    target: str = ""  # block label

@dataclass
class MIRCondBranch(MIRTerminator):
    """Conditional jump."""
    cond: MIRValue = None
    true_block: str = ""
    false_block: str = ""

@dataclass
class MIRRet(MIRTerminator):
    """Return from function."""
    value: Optional[MIRValue] = None

@dataclass
class MIRMatch(MIRTerminator):
    """Multi-way branch for match expressions."""
    value: MIRValue = None
    arms: List[Tuple[Any, str]] = field(default_factory=list)  # [(pattern, block_label), ...]
    default_block: Optional[str] = None


# ── MIR Basic Block ───────────────────────────────────────────────────────────

@dataclass
class MIRBasicBlock:
    """A linear sequence of instructions ending with a terminator."""
    label: str
    instructions: List[MIRInst] = field(default_factory=list)
    terminator: Optional[MIRTerminator] = None

    def append(self, inst: MIRInst):
        self.instructions.append(inst)

    def is_terminated(self):
        return self.terminator is not None


# ── MIR Function ──────────────────────────────────────────────────────────────

@dataclass
class MIRFunction:
    """A function in MIR — a list of basic blocks forming a CFG."""
    name: str
    params: List[Tuple[str, MIRType]]  # [(name, type), ...]
    return_type: MIRType
    blocks: List[MIRBasicBlock] = field(default_factory=list)
    is_kernel: bool = False
    is_async: bool = False
    is_extern: bool = False

    def entry_block(self) -> Optional[MIRBasicBlock]:
        return self.blocks[0] if self.blocks else None

    def add_block(self, label: str) -> MIRBasicBlock:
        bb = MIRBasicBlock(label=label)
        self.blocks.append(bb)
        return bb

    def block_map(self) -> Dict[str, MIRBasicBlock]:
        return {b.label: b for b in self.blocks}


# ── MIR Module ────────────────────────────────────────────────────────────────

@dataclass
class MIRModule:
    """Top-level MIR container — parallel to a compilation unit."""
    name: str = "nexalang_module"
    functions: List[MIRFunction] = field(default_factory=list)
    structs: Dict[str, List[Tuple[str, MIRType]]] = field(default_factory=dict)  # name -> [(field, type)]
    enums: Dict[str, Dict[str, List[MIRType]]] = field(default_factory=dict)  # name -> {variant: [payloads]}
    globals: Dict[str, MIRType] = field(default_factory=dict)

    def add_function(self, func: MIRFunction):
        self.functions.append(func)

    def find_function(self, name: str) -> Optional[MIRFunction]:
        for f in self.functions:
            if f.name == name:
                return f
        return None


# ── AST → MIR Lowering ───────────────────────────────────────────────────────

class MIRLowering:
    """Lowers a semantically-analyzed AST into MIR."""

    def __init__(self):
        self.module = MIRModule()
        self.temp_counter = 0
        self.block_counter = 0
        self.current_func: Optional[MIRFunction] = None
        self.current_block: Optional[MIRBasicBlock] = None
        self.var_types: Dict[str, MIRType] = {}
        self.loop_stack = []  # [(continue_label, break_label)]

    def new_temp(self, ty: MIRType) -> MIRTemp:
        t = MIRTemp(id=self.temp_counter, ty=ty)
        self.temp_counter += 1
        return t

    def new_block(self, hint: str = "bb") -> MIRBasicBlock:
        label = f"{hint}_{self.block_counter}"
        self.block_counter += 1
        bb = self.current_func.add_block(label)
        return bb

    def emit(self, inst: MIRInst):
        if self.current_block and not self.current_block.is_terminated():
            self.current_block.append(inst)

    def terminate(self, term: MIRTerminator):
        if self.current_block and not self.current_block.is_terminated():
            self.current_block.terminator = term

    def switch_to(self, block: MIRBasicBlock):
        self.current_block = block

    def lower(self, ast, struct_info=None, enum_info=None):
        """Lower a full AST (list of top-level nodes) into MIR."""
        from n_parser import FunctionDef, StructDef, EnumDef, ImplDef, ExternBlock

        # Pass 1: Collect types
        if struct_info:
            for name, fields in struct_info.items():
                self.module.structs[name] = [(f, MIRType(t)) for f, t in fields.items()]
        if enum_info:
            for name, variants in enum_info.items():
                self.module.enums[name] = {v: [MIRType(p) for p in payloads] for v, payloads in variants.items()}

        # Pass 2: Functions
        for node in ast:
            if isinstance(node, FunctionDef):
                self._lower_function(node)
            elif isinstance(node, ImplDef):
                for method in node.methods:
                    self._lower_function(method)

        return self.module

    def _lower_function(self, node):
        from n_parser import FunctionDef
        if getattr(node, 'generics', None):
            return  # Skip uninstantiated generics

        params = [(p[0], MIRType(p[1])) for p in node.params]
        ret_type = MIRType(node.return_type)

        func = MIRFunction(
            name=node.name,
            params=params,
            return_type=ret_type,
            is_kernel=getattr(node, 'is_kernel', False),
            is_async=getattr(node, 'is_async', False),
        )
        self.current_func = func
        self.module.add_function(func)

        # Register params
        for pname, pty in params:
            self.var_types[pname] = pty

        # Entry block
        entry = func.add_block("entry")
        self.switch_to(entry)

        # Lower body
        if node.body:
            for stmt in node.body:
                self._lower_stmt(stmt)

        # Ensure terminator
        if not self.current_block.is_terminated():
            if ret_type.is_void():
                self.terminate(MIRRet(value=None))
            else:
                self.terminate(MIRRet(value=MIRConst(0, ret_type)))

        self.current_func = None

    def _lower_stmt(self, node):
        from n_parser import (VarDecl, Assignment, ReturnStmt, IfStmt, WhileStmt,
                              ForStmt, BreakStmt, ContinueStmt, CallExpr,
                              MethodCall, BlockStmt, MatchExpr, RegionStmt)

        if isinstance(node, VarDecl):
            self._lower_var_decl(node)
        elif isinstance(node, Assignment):
            self._lower_assignment(node)
        elif isinstance(node, ReturnStmt):
            self._lower_return(node)
        elif isinstance(node, IfStmt):
            self._lower_if(node)
        elif isinstance(node, WhileStmt):
            self._lower_while(node)
        elif isinstance(node, ForStmt):
            self._lower_for(node)
        elif isinstance(node, BreakStmt):
            self._lower_break(node)
        elif isinstance(node, ContinueStmt):
            self._lower_continue(node)
        elif isinstance(node, BlockStmt):
            for s in node.stmts:
                self._lower_stmt(s)
        elif isinstance(node, (CallExpr, MethodCall)):
            self._lower_expr(node)
        elif isinstance(node, MatchExpr):
            self._lower_match(node)
        else:
            # Expression statement
            self._lower_expr(node)

    def _lower_var_decl(self, node):
        ty = MIRType(node.type_name) if node.type_name else MIRType("i32")
        self.var_types[node.name] = ty

        if node.initializer:
            val = self._lower_expr(node.initializer)
            dest = MIRVar(name=node.name, ty=ty)
            self.emit(MIRStore(ptr=dest, value=val))
        else:
            self.emit(MIRAlloca(dest=MIRTemp(id=self.temp_counter, ty=ty), alloc_type=ty))

    def _lower_assignment(self, node):
        from n_parser import VariableExpr, MemberAccess, IndexAccess, UnaryExpr
        val = self._lower_expr(node.value)

        if isinstance(node.target, VariableExpr):
            dest = MIRVar(name=node.target.name, ty=val.ty if isinstance(val, (MIRTemp, MIRConst, MIRVar)) else MIRType("i32"))
            self.emit(MIRStore(ptr=dest, value=val))
        elif isinstance(node.target, MemberAccess):
            base = self._lower_expr(node.target.object)
            self.emit(MIRSetField(
                base=base,
                field_name=node.target.member,
                struct_type=getattr(node.target, 'struct_type', ''),
                value=val
            ))
        elif isinstance(node.target, IndexAccess):
            base = self._lower_expr(node.target.object)
            index = self._lower_expr(node.target.index)
            self.emit(MIRSetIndex(base=base, index=index, value=val))

    def _lower_return(self, node):
        if node.value:
            val = self._lower_expr(node.value)
            self.terminate(MIRRet(value=val))
        else:
            self.terminate(MIRRet(value=None))

    def _lower_if(self, node):
        cond = self._lower_expr(node.condition)

        then_bb = self.new_block("then")
        else_bb = self.new_block("else")
        merge_bb = self.new_block("if_end")

        self.terminate(MIRCondBranch(cond=cond, true_block=then_bb.label, false_block=else_bb.label))

        # Then branch
        self.switch_to(then_bb)
        for s in node.then_branch:
            self._lower_stmt(s)
        if not self.current_block.is_terminated():
            self.terminate(MIRBranch(target=merge_bb.label))

        # Else branch
        self.switch_to(else_bb)
        if node.else_branch:
            for s in node.else_branch:
                self._lower_stmt(s)
        if not self.current_block.is_terminated():
            self.terminate(MIRBranch(target=merge_bb.label))

        self.switch_to(merge_bb)

    def _lower_while(self, node):
        cond_bb = self.new_block("while_cond")
        body_bb = self.new_block("while_body")
        end_bb = self.new_block("while_end")

        self.terminate(MIRBranch(target=cond_bb.label))

        self.switch_to(cond_bb)
        cond = self._lower_expr(node.condition)
        self.terminate(MIRCondBranch(cond=cond, true_block=body_bb.label, false_block=end_bb.label))

        self.loop_stack.append((cond_bb.label, end_bb.label))
        self.switch_to(body_bb)
        for s in node.body:
            self._lower_stmt(s)
        if not self.current_block.is_terminated():
            self.terminate(MIRBranch(target=cond_bb.label))
        self.loop_stack.pop()

        self.switch_to(end_bb)

    def _lower_for(self, node):
        if hasattr(node, 'is_iterator') and node.is_iterator:
            # Iterator-based for loop
            iter_val = self._lower_expr(node.start_expr)
            cond_bb = self.new_block("for_iter_cond")
            body_bb = self.new_block("for_iter_body")
            end_bb = self.new_block("for_iter_end")

            self.terminate(MIRBranch(target=cond_bb.label))
            self.switch_to(cond_bb)

            # Call next() — simplified
            next_result = self.new_temp(MIRType(getattr(node, 'option_type', 'Option')))
            self.emit(MIRMethodCall(
                dest=next_result, receiver=iter_val, method="next",
                struct_type=getattr(node, 'iterator_type', ''), ret_type=next_result.ty
            ))
            # Check if Some
            is_some = self.new_temp(MIRType("bool"))
            self.emit(MIRCall(dest=is_some, callee="__mir_option_is_some", args=[next_result], ret_type=MIRType("bool")))
            self.terminate(MIRCondBranch(cond=is_some, true_block=body_bb.label, false_block=end_bb.label))

            self.loop_stack.append((cond_bb.label, end_bb.label))
            self.switch_to(body_bb)
            # Bind loop variable
            self.var_types[node.var_name] = MIRType(node.item_type)
            for s in node.body:
                self._lower_stmt(s)
            if not self.current_block.is_terminated():
                self.terminate(MIRBranch(target=cond_bb.label))
            self.loop_stack.pop()
            self.switch_to(end_bb)
        else:
            # Range-based for
            start = self._lower_expr(node.start_expr)
            end = self._lower_expr(node.end_expr)
            self.var_types[node.var_name] = MIRType("i32")

            init_bb = self.current_block
            cond_bb = self.new_block("for_cond")
            body_bb = self.new_block("for_body")
            inc_bb = self.new_block("for_inc")
            end_bb = self.new_block("for_end")

            # Store start
            loop_var = MIRVar(name=node.var_name, ty=MIRType("i32"))
            self.emit(MIRStore(ptr=loop_var, value=start))
            self.terminate(MIRBranch(target=cond_bb.label))

            # Condition
            self.switch_to(cond_bb)
            cmp_op = "le" if getattr(node, 'inclusive', False) else "lt"
            cond_temp = self.new_temp(MIRType("bool"))
            self.emit(MIRBinOp(dest=cond_temp, op=cmp_op, left=loop_var, right=end))
            self.terminate(MIRCondBranch(cond=cond_temp, true_block=body_bb.label, false_block=end_bb.label))

            # Body
            self.loop_stack.append((inc_bb.label, end_bb.label))
            self.switch_to(body_bb)
            for s in node.body:
                self._lower_stmt(s)
            if not self.current_block.is_terminated():
                self.terminate(MIRBranch(target=inc_bb.label))
            self.loop_stack.pop()

            # Increment
            self.switch_to(inc_bb)
            inc_temp = self.new_temp(MIRType("i32"))
            self.emit(MIRBinOp(dest=inc_temp, op="add", left=loop_var, right=MIRConst(1, MIRType("i32"))))
            self.emit(MIRStore(ptr=loop_var, value=inc_temp))
            self.terminate(MIRBranch(target=cond_bb.label))

            self.switch_to(end_bb)

    def _lower_break(self, node):
        if self.loop_stack:
            _, break_label = self.loop_stack[-1]
            self.terminate(MIRBranch(target=break_label))

    def _lower_continue(self, node):
        if self.loop_stack:
            continue_label, _ = self.loop_stack[-1]
            self.terminate(MIRBranch(target=continue_label))

    def _lower_match(self, node):
        val = self._lower_expr(node.value)
        merge_bb = self.new_block("match_end")

        for arm in node.cases:
            arm_bb = self.new_block("match_arm")
            next_bb = self.new_block("match_next")

            # Simplified: compare tag for enum patterns
            cond = self.new_temp(MIRType("bool"))
            pattern_val = MIRConst(arm.pattern if isinstance(arm.pattern, int) else 0, MIRType("i32"))
            self.emit(MIRBinOp(dest=cond, op="eq", left=val, right=pattern_val))
            self.terminate(MIRCondBranch(cond=cond, true_block=arm_bb.label, false_block=next_bb.label))

            self.switch_to(arm_bb)
            if isinstance(arm.body, list):
                for s in arm.body:
                    self._lower_stmt(s)
            else:
                self._lower_stmt(arm.body)
            if not self.current_block.is_terminated():
                self.terminate(MIRBranch(target=merge_bb.label))

            self.switch_to(next_bb)

        # Default: jump to merge
        if not self.current_block.is_terminated():
            self.terminate(MIRBranch(target=merge_bb.label))

        self.switch_to(merge_bb)

    def _lower_expr(self, node) -> MIRValue:
        from n_parser import (IntegerLiteral, FloatLiteral, StringLiteral,
                              BooleanLiteral, CharLiteral, VariableExpr,
                              BinaryExpr, UnaryExpr, CallExpr, MethodCall,
                              MemberAccess, IndexAccess, ArrayLiteral,
                              MatchExpr, LambdaExpr, MacroCallExpr, IfStmt)

        if isinstance(node, IntegerLiteral):
            return MIRConst(node.value, MIRType("i32"))
        elif isinstance(node, FloatLiteral):
            return MIRConst(node.value, MIRType("f32"))
        elif isinstance(node, StringLiteral):
            return MIRConst(node.value, MIRType("string"))
        elif isinstance(node, BooleanLiteral):
            return MIRConst(node.value, MIRType("bool"))
        elif isinstance(node, CharLiteral):
            return MIRConst(node.value, MIRType("char"))
        elif isinstance(node, VariableExpr):
            ty = self.var_types.get(node.name, MIRType("i32"))
            return MIRVar(name=node.name, ty=ty)
        elif isinstance(node, BinaryExpr):
            return self._lower_binop(node)
        elif isinstance(node, UnaryExpr):
            return self._lower_unary(node)
        elif isinstance(node, CallExpr):
            return self._lower_call(node)
        elif isinstance(node, MethodCall):
            return self._lower_method_call(node)
        elif isinstance(node, MemberAccess):
            return self._lower_member(node)
        elif isinstance(node, IndexAccess):
            return self._lower_index(node)
        elif isinstance(node, MacroCallExpr):
            if hasattr(node, 'expanded'):
                return self._lower_expr(node.expanded)
            return MIRConst(0, MIRType("i32"))
        elif isinstance(node, IfStmt):
            # If-expression (returns value)
            self._lower_if(node)
            return MIRConst(0, MIRType("void"))
        elif isinstance(node, list):
            # Statement list (e.g. block)
            for s in node:
                self._lower_stmt(s)
            return MIRConst(0, MIRType("void"))
        else:
            return MIRConst(0, MIRType("void"))

    def _lower_binop(self, node) -> MIRValue:
        left = self._lower_expr(node.left)
        right = self._lower_expr(node.right)

        op_map = {
            'PLUS': 'add', 'MINUS': 'sub', 'STAR': 'mul', 'SLASH': 'div',
            'PERCENT': 'rem', 'EQEQ': 'eq', 'NEQ': 'ne',
            'LT': 'lt', 'GT': 'gt', 'LTE': 'le', 'GTE': 'ge',
            'AND': 'and', 'OR': 'or',
        }
        mir_op = op_map.get(node.op, node.op)

        # Determine result type
        if mir_op in ('eq', 'ne', 'lt', 'gt', 'le', 'ge', 'and', 'or'):
            res_ty = MIRType("bool")
        else:
            res_ty = left.ty if isinstance(left, (MIRTemp, MIRVar, MIRConst)) else MIRType("i32")

        dest = self.new_temp(res_ty)
        self.emit(MIRBinOp(dest=dest, op=mir_op, left=left, right=right))
        return dest

    def _lower_unary(self, node) -> MIRValue:
        operand = self._lower_expr(node.operand)
        op_map = {'!': 'not', '-': 'neg', '&': 'addr_of', '*': 'deref'}
        mir_op = op_map.get(node.op, node.op)

        if mir_op == 'addr_of':
            res_ty = MIRType(f"&{operand.ty.name}" if isinstance(operand, (MIRVar, MIRTemp)) else "i32")
        elif mir_op == 'deref':
            inner_name = operand.ty.name
            if inner_name.startswith('&'):
                res_ty = MIRType(inner_name[1:])
            elif inner_name.endswith('*'):
                res_ty = MIRType(inner_name[:-1])
            else:
                res_ty = MIRType("i32")
        elif mir_op == 'not':
            res_ty = MIRType("bool")
        else:
            res_ty = operand.ty if isinstance(operand, (MIRTemp, MIRVar, MIRConst)) else MIRType("i32")

        dest = self.new_temp(res_ty)
        self.emit(MIRUnaryOp(dest=dest, op=mir_op, operand=operand))
        return dest

    def _lower_call(self, node) -> MIRValue:
        callee = node.callee if isinstance(node.callee, str) else node.callee.name
        args = [self._lower_expr(a) for a in node.args]

        ret_ty = MIRType(getattr(node, 'type_name', 'void') or 'void')
        dest = self.new_temp(ret_ty) if not ret_ty.is_void() else None

        self.emit(MIRCall(dest=dest, callee=callee, args=args, ret_type=ret_ty))
        return dest if dest else MIRConst(0, MIRType("void"))

    def _lower_method_call(self, node) -> MIRValue:
        receiver = self._lower_expr(node.receiver)
        args = [self._lower_expr(a) for a in node.args]

        ret_ty = MIRType(getattr(node, 'type_name', 'void') or 'void')
        dest = self.new_temp(ret_ty) if not ret_ty.is_void() else None

        self.emit(MIRMethodCall(
            dest=dest, receiver=receiver,
            method=getattr(node, 'method_name', node.method),
            struct_type=getattr(node, 'struct_type', ''),
            args=args, ret_type=ret_ty
        ))
        return dest if dest else MIRConst(0, MIRType("void"))

    def _lower_member(self, node) -> MIRValue:
        base = self._lower_expr(node.object)
        struct_type = getattr(node, 'struct_type', '')
        # Infer result type from struct fields
        field_ty = MIRType("i32")  # Default; ideally looked up from struct_info
        if struct_type in self.module.structs:
            for fname, ftype in self.module.structs[struct_type]:
                if fname == node.member:
                    field_ty = ftype
                    break

        dest = self.new_temp(field_ty)
        self.emit(MIRGetField(dest=dest, base=base, field_name=node.member, struct_type=struct_type))
        return dest

    def _lower_index(self, node) -> MIRValue:
        base = self._lower_expr(node.object)
        index = self._lower_expr(node.index)
        dest = self.new_temp(MIRType("i32"))  # Element type would be looked up
        self.emit(MIRIndex(dest=dest, base=base, index=index))
        return dest


# ── Optimization Passes ──────────────────────────────────────────────────────

class MIROptimizer:
    """Runs optimization passes on MIR."""

    def __init__(self):
        self.stats = {'constant_folded': 0, 'dead_eliminated': 0, 'copies_propagated': 0}

    def optimize(self, module: MIRModule, level: int = 1) -> MIRModule:
        """Run optimization passes. level: 0=none, 1=basic, 2=aggressive."""
        if level == 0:
            return module

        for func in module.functions:
            if func.is_extern:
                continue
            self.constant_fold(func)
            self.copy_propagation(func)
            self.dead_code_elimination(func)
            if level >= 2:
                self.constant_fold(func)  # Second pass after propagation
                self.dead_code_elimination(func)

        return module

    def constant_fold(self, func: MIRFunction):
        """Evaluate constant expressions at compile time."""
        for block in func.blocks:
            new_insts = []
            for inst in block.instructions:
                if isinstance(inst, MIRBinOp) and isinstance(inst.left, MIRConst) and isinstance(inst.right, MIRConst):
                    result = self._eval_binop(inst.op, inst.left, inst.right)
                    if result is not None:
                        new_insts.append(MIRAssign(dest=inst.dest, value=result))
                        self.stats['constant_folded'] += 1
                        continue
                elif isinstance(inst, MIRUnaryOp) and isinstance(inst.operand, MIRConst):
                    result = self._eval_unary(inst.op, inst.operand)
                    if result is not None:
                        new_insts.append(MIRAssign(dest=inst.dest, value=result))
                        self.stats['constant_folded'] += 1
                        continue
                new_insts.append(inst)
            block.instructions = new_insts

    def copy_propagation(self, func: MIRFunction):
        """Replace uses of copies with original values."""
        copies = {}  # MIRTemp.id -> MIRValue (the source)

        for block in func.blocks:
            for inst in block.instructions:
                if isinstance(inst, MIRAssign) and isinstance(inst.dest, MIRTemp):
                    copies[inst.dest.id] = inst.value

        if not copies:
            return

        for block in func.blocks:
            for inst in block.instructions:
                self._substitute_copies(inst, copies)
            # Also substitute in terminator
            if block.terminator:
                self._substitute_copies_terminator(block.terminator, copies)

        # Remove pure copy instructions that are no longer needed
        used_temps = self._collect_used_temps(func)
        for block in func.blocks:
            block.instructions = [
                inst for inst in block.instructions
                if not (isinstance(inst, MIRAssign) and isinstance(inst.dest, MIRTemp) and inst.dest.id not in used_temps)
            ]
            self.stats['copies_propagated'] += 1

    def dead_code_elimination(self, func: MIRFunction):
        """Remove instructions whose results are never used."""
        used_temps = self._collect_used_temps(func)

        for block in func.blocks:
            new_insts = []
            for inst in block.instructions:
                # Keep side-effecting instructions (calls, stores, drops)
                if isinstance(inst, (MIRCall, MIRMethodCall, MIRStore, MIRSetField, MIRSetIndex, MIRDrop)):
                    new_insts.append(inst)
                    continue

                # Check if the instruction's dest is used
                dest = getattr(inst, 'dest', None)
                if dest and isinstance(dest, MIRTemp) and dest.id not in used_temps:
                    self.stats['dead_eliminated'] += 1
                    continue

                new_insts.append(inst)
            block.instructions = new_insts

    def _eval_binop(self, op: str, left: MIRConst, right: MIRConst) -> Optional[MIRConst]:
        """Try to evaluate a binary operation on constants."""
        lv, rv = left.value, right.value
        if not isinstance(lv, (int, float)) or not isinstance(rv, (int, float)):
            return None

        try:
            ops = {
                'add': lambda: lv + rv,
                'sub': lambda: lv - rv,
                'mul': lambda: lv * rv,
                'div': lambda: lv // rv if isinstance(lv, int) and rv != 0 else (lv / rv if rv != 0 else None),
                'rem': lambda: lv % rv if rv != 0 else None,
                'eq': lambda: lv == rv,
                'ne': lambda: lv != rv,
                'lt': lambda: lv < rv,
                'gt': lambda: lv > rv,
                'le': lambda: lv <= rv,
                'ge': lambda: lv >= rv,
            }
            if op not in ops:
                return None
            result = ops[op]()
            if result is None:
                return None

            if isinstance(result, bool):
                return MIRConst(result, MIRType("bool"))
            return MIRConst(result, left.ty)
        except (ZeroDivisionError, OverflowError):
            return None

    def _eval_unary(self, op: str, operand: MIRConst) -> Optional[MIRConst]:
        """Try to evaluate a unary operation on a constant."""
        v = operand.value
        if op == 'neg' and isinstance(v, (int, float)):
            return MIRConst(-v, operand.ty)
        if op == 'not' and isinstance(v, (bool, int)):
            return MIRConst(not v, MIRType("bool"))
        return None

    def _collect_used_temps(self, func: MIRFunction) -> set:
        """Collect all MIRTemp IDs that are used (read) somewhere."""
        used = set()

        def _scan_value(v):
            if isinstance(v, MIRTemp):
                used.add(v.id)

        for block in func.blocks:
            for inst in block.instructions:
                for attr_name in ('value', 'left', 'right', 'operand', 'ptr', 'base', 'index', 'receiver', 'cond'):
                    val = getattr(inst, attr_name, None)
                    if val:
                        _scan_value(val)
                args = getattr(inst, 'args', None)
                if args:
                    for a in args:
                        _scan_value(a)
                fields = getattr(inst, 'fields', None)
                if fields:
                    for f in fields:
                        _scan_value(f)
                payload = getattr(inst, 'payload', None)
                if payload:
                    for p in payload:
                        _scan_value(p)

            # Terminator
            if block.terminator:
                if isinstance(block.terminator, MIRCondBranch):
                    _scan_value(block.terminator.cond)
                elif isinstance(block.terminator, MIRRet):
                    if block.terminator.value:
                        _scan_value(block.terminator.value)
                elif isinstance(block.terminator, MIRMatch):
                    _scan_value(block.terminator.value)

        return used

    def _substitute_copies(self, inst, copies):
        """Replace MIRTemp references with their copy source."""
        for attr_name in ('value', 'left', 'right', 'operand', 'ptr', 'base', 'index', 'receiver'):
            val = getattr(inst, attr_name, None)
            if isinstance(val, MIRTemp) and val.id in copies:
                setattr(inst, attr_name, copies[val.id])
        args = getattr(inst, 'args', None)
        if args:
            for i, a in enumerate(args):
                if isinstance(a, MIRTemp) and a.id in copies:
                    args[i] = copies[a.id]

    def _substitute_copies_terminator(self, term, copies):
        if isinstance(term, MIRCondBranch):
            if isinstance(term.cond, MIRTemp) and term.cond.id in copies:
                term.cond = copies[term.cond.id]
        elif isinstance(term, MIRRet):
            if isinstance(term.value, MIRTemp) and term.value.id in copies:
                term.value = copies[term.value.id]


# ── MIR Pretty Printer ──────────────────────────────────────────────────────

class MIRPrinter:
    """Pretty-prints MIR for debugging."""

    def print_module(self, module: MIRModule) -> str:
        lines = [f"// MIR Module: {module.name}\n"]

        for name, fields in module.structs.items():
            field_str = ", ".join(f"{f}: {t}" for f, t in fields)
            lines.append(f"struct {name} {{ {field_str} }}")

        for name, variants in module.enums.items():
            var_strs = []
            for v, payloads in variants.items():
                if payloads:
                    var_strs.append(f"{v}({', '.join(str(p) for p in payloads)})")
                else:
                    var_strs.append(v)
            lines.append(f"enum {name} {{ {', '.join(var_strs)} }}")

        lines.append("")

        for func in module.functions:
            lines.append(self.print_function(func))

        return "\n".join(lines)

    def print_function(self, func: MIRFunction) -> str:
        params = ", ".join(f"{n}: {t}" for n, t in func.params)
        header = f"fn {func.name}({params}) -> {func.return_type}"
        if func.is_kernel:
            header = f"kernel {header}"
        if func.is_async:
            header = f"async {header}"

        lines = [f"{header} {{"]
        for block in func.blocks:
            lines.append(f"  {block.label}:")
            for inst in block.instructions:
                lines.append(f"    {self._format_inst(inst)}")
            if block.terminator:
                lines.append(f"    {self._format_terminator(block.terminator)}")
        lines.append("}")
        return "\n".join(lines)

    def _format_inst(self, inst) -> str:
        if isinstance(inst, MIRAssign):
            return f"{inst.dest} = {inst.value}"
        elif isinstance(inst, MIRBinOp):
            return f"{inst.dest} = {inst.op} {inst.left}, {inst.right}"
        elif isinstance(inst, MIRUnaryOp):
            return f"{inst.dest} = {inst.op} {inst.operand}"
        elif isinstance(inst, MIRCall):
            args = ", ".join(str(a) for a in inst.args)
            dest_str = f"{inst.dest} = " if inst.dest else ""
            return f"{dest_str}call {inst.callee}({args})"
        elif isinstance(inst, MIRMethodCall):
            args = ", ".join(str(a) for a in inst.args)
            dest_str = f"{inst.dest} = " if inst.dest else ""
            return f"{dest_str}{inst.receiver}.{inst.method}({args})"
        elif isinstance(inst, MIRAlloca):
            return f"{inst.dest} = alloca {inst.alloc_type}"
        elif isinstance(inst, MIRStore):
            return f"store {inst.value} -> {inst.ptr}"
        elif isinstance(inst, MIRLoad):
            return f"{inst.dest} = load {inst.ptr}"
        elif isinstance(inst, MIRGetField):
            return f"{inst.dest} = {inst.base}.{inst.field_name}"
        elif isinstance(inst, MIRSetField):
            return f"{inst.base}.{inst.field_name} = {inst.value}"
        elif isinstance(inst, MIRIndex):
            return f"{inst.dest} = {inst.base}[{inst.index}]"
        elif isinstance(inst, MIRSetIndex):
            return f"{inst.base}[{inst.index}] = {inst.value}"
        elif isinstance(inst, MIRCast):
            return f"{inst.dest} = cast<{inst.target_type}>({inst.value})"
        elif isinstance(inst, MIRConstructStruct):
            fields = ", ".join(str(f) for f in inst.fields)
            return f"{inst.dest} = {inst.struct_name}({fields})"
        elif isinstance(inst, MIRConstructEnum):
            payload = ", ".join(str(p) for p in inst.payload)
            return f"{inst.dest} = {inst.enum_name}::{inst.variant}({payload})"
        elif isinstance(inst, MIRDrop):
            return f"drop {inst.var} : {inst.type_name}"
        return f"<unknown: {type(inst).__name__}>"

    def _format_terminator(self, term) -> str:
        if isinstance(term, MIRBranch):
            return f"br {term.target}"
        elif isinstance(term, MIRCondBranch):
            return f"br {term.cond}, {term.true_block}, {term.false_block}"
        elif isinstance(term, MIRRet):
            return f"ret {term.value}" if term.value else "ret void"
        elif isinstance(term, MIRMatch):
            arms = ", ".join(f"{p} => {l}" for p, l in term.arms)
            return f"match {term.value} {{ {arms} }}"
        return f"<unknown term: {type(term).__name__}>"
