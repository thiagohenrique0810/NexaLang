from llvmlite import ir
from parser import StructDef, EnumDef, FunctionDef, VariableExpr, UnaryExpr

class CodeGen:
    def __init__(self):
        self.module = ir.Module(name="nexalang_module")
        self.builder = None
        self.printf = None
        self.exit_func = None
        self.malloc = None
        self.free = None
        self._declare_printf()
        self._declare_exit()
        self._declare_malloc_free()

    def _declare_malloc_free(self):
        # void* malloc(i32)
        # void free(void*)
        
        # malloc
        void_ptr = ir.IntType(8).as_pointer()
        malloc_ty = ir.FunctionType(void_ptr, [ir.IntType(32)])
        self.malloc = ir.Function(self.module, malloc_ty, name="malloc")
        
        # free
        free_ty = ir.FunctionType(ir.VoidType(), [void_ptr])
        self.free = ir.Function(self.module, free_ty, name="free")
        
        # realloc
        realloc_ty = ir.FunctionType(void_ptr, [void_ptr, ir.IntType(32)])
        self.realloc = ir.Function(self.module, realloc_ty, name="realloc")
        self.struct_types = {} # name -> ir.LiteralStructType
        self.struct_fields = {} # name -> {field_name: index}
        self.enum_types = {} # name -> {variant: tag_id}
        self.enum_payloads = {} # name -> {variant: payload_type}
        self.enum_definitions = {} # name -> (ir_struct_type, payload_size)
        self.scopes = []
        self._declare_memcpy()

    def _declare_memcpy(self):
        # Declare llvm.memcpy.p0i8.p0i8.i32
        # void @llvm.memcpy.p0i8.p0i8.i32(i8* <dest>, i8* <src>, i32 <len>, i1 <isvolatile>)
        void_ptr = ir.IntType(8).as_pointer()
        bool_ty = ir.IntType(1)
        i32_ty = ir.IntType(32)
        
        fnty = ir.FunctionType(ir.VoidType(), [void_ptr, void_ptr, i32_ty, bool_ty])
        self.memcpy = ir.Function(self.module, fnty, name="llvm.memcpy.p0i8.p0i8.i32")

    def _declare_printf(self):
        voidptr_ty = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        self.printf = ir.Function(self.module, printf_ty, name="printf")

    def _declare_exit(self):
        exit_ty = ir.FunctionType(ir.VoidType(), [ir.IntType(32)])
        self.exit_func = ir.Function(self.module, exit_ty, name="exit")

    def get_llvm_type(self, type_name):
        if type_name == 'i32':
            return ir.IntType(32)
        elif type_name == 'i64':
             return ir.IntType(64)
        elif type_name == 'u8':
             return ir.IntType(8)
        elif type_name == 'void':
             return ir.VoidType()
        elif type_name == 'string':
             return ir.IntType(8).as_pointer()
        elif type_name.endswith('*'):
             inner_type = type_name[:-1]
             # Handle *void (void*)
             if inner_type == 'void':
                  return ir.IntType(8).as_pointer()
             return self.get_llvm_type(inner_type).as_pointer()
        elif type_name in self.struct_types:
             return self.struct_types[type_name]
        # Enum types? Pointers usually.
        # Fallback
        raise Exception(f"CodeGen: Unknown type '{type_name}'")

    def visit_StructDef(self, node):
        if node.generics:
            return # Skip generic definition
            
        # Create LLVM Struct Type
        field_types = []
        for _, type_name in node.fields:
             field_types.append(self.get_llvm_type(type_name))
        
        struct_ty = ir.LiteralStructType(field_types)
        self.struct_types[node.name] = struct_ty
        
        # Map field names to indices
        self.struct_fields[node.name] = {name: i for i, (name, _) in enumerate(node.fields)}

    def visit_EnumDef(self, node):
        if node.generics: return
        
        # Representation: { i32 tag, [MaxPayloadSize x i8] data }
        # 1. Determine max payload size
        max_size = 0
        variant_tags = {}
        variant_payload_types = {}
        
        # For this phase, assume only i32 payloads for simplicity of size calc
        # i32 = 4 bytes
        
        for i, (vname, payloads) in enumerate(node.variants):
            variant_tags[vname] = i
            if payloads:
                # Assume single payload for now
                ptype = payloads[0]
                if ptype == 'i32':
                    max_size = max(max_size, 4)
                    variant_payload_types[vname] = ir.IntType(32)
                elif ptype == 'string':
                    max_size = max(max_size, 8) # Pointer size
                    variant_payload_types[vname] = ir.IntType(8).as_pointer()
                elif ptype in self.struct_types:
                    # Struct payload - assume pointer or by value?
                    # For simplicity, treat as by-value, but size might be large.
                    # Or maybe we store structs as pointers in enums?
                    # Let's say we support stored structs.
                    # Need size of struct.
                    # self.struct_types[ptype] is ir.LiteralStructType.
                    # We can't easily get size in bytes from llvmlite without target data layout.
                    # Hack: assume large defaults or skip for now?
                    # Let's Skip arbitrary structs for now to avoid complexity, or assume 64 bytes?
                    pass
                elif ptype.endswith('*') or ptype.startswith('['):
                     # Pointers/Arrays
                     max_size = max(max_size, 8)
                     # Need actual type
                     pass
                else: 
                     # Try generic fallback
                     max_size = max(max_size, 8)
                     variant_payload_types[vname] = ir.IntType(32) # Placeholder? Dangerous
            else:
                 variant_payload_types[vname] = None
        
        # Create LLVM type
        # Tag (i32) + Payload (Array of i8)
        padding_ty = ir.ArrayType(ir.IntType(8), max_size)
        enum_ty = ir.LiteralStructType([ir.IntType(32), padding_ty])
        
        self.enum_types[node.name] = variant_tags
        self.enum_payloads[node.name] = variant_payload_types
        self.enum_definitions[node.name] = (enum_ty, max_size)

    def generate(self, ast):
        # Pass 1: Types (Structs, Enums)
        for node in ast:
            if isinstance(node, (StructDef, EnumDef)):
                self.visit(node)
        
        # Pass 2: Function Headers
        for node in ast:
            if isinstance(node, FunctionDef):
                self._declare_function(node)
                
        # Pass 3: Bodies
        for node in ast:
            if not isinstance(node, (StructDef, EnumDef)):
                self.visit(node)
                
        return str(self.module)

    def visit(self, node):
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f"No visit_{type(node).__name__} method")
        
    def visit_UnaryExpr(self, node):
        if node.op == '&':
             # Address Of: We need the address of the operand.
             # visit(operand) usually returns a value (load).
             # We need a method to get address.
             # HACK: If operand is VariableExpr, look up alloca.
             if isinstance(node.operand, VariableExpr):
                  for scope in reversed(self.scopes):
                       if node.operand.name in scope:
                            return scope[node.operand.name]
                  raise Exception(f"Undefined variable for address of: {node.operand.name}")
             else:
                  raise Exception("Address of (&) only supported for variables currently")
                  
        elif node.op == '*':
             # Dereference
             ptr_val = self.visit(node.operand)
             return self.builder.load(ptr_val)
        else:
            raise Exception(f"Unsupported unary operator: {node.op}")

    def visit_MemberAccess(self, node):
        # ... existing implementation ...
        struct_val = self.visit(node.object)
        if not hasattr(node, 'struct_type'):
             raise Exception("CodeGen Error: MemberAccess node missing 'struct_type' annotation from Semantic phase.")
        struct_name = node.struct_type
        field_index = self.struct_fields[struct_name][node.member]
        return self.builder.extract_value(struct_val, field_index)

    def visit_ArrayLiteral(self, node):
        # Create array value
        # Assume i32 elements for now since we parsed [i32:N] or inferred
        # We need to know the type!
        # SemanticAnalyzer verified types are consistent.
        # We can inspect the first element's LLVM type?
        val0 = self.visit(node.elements[0])
        elem_ty = val0.type
        size = len(node.elements)
        array_ty = ir.ArrayType(elem_ty, size)
        
        array_val = ir.Constant(array_ty, ir.Undefined)
        array_val = self.builder.insert_value(array_val, val0, 0)
        
        for i in range(1, size):
            val = self.visit(node.elements[i])
            array_val = self.builder.insert_value(array_val, val, i)
            
        return array_val

    def visit_IndexAccess(self, node):
        index_val = self.visit(node.index)
        
        # Optimization: If object is a Variable, use GEP on pointer
        # We need to check if node.object is VariableExpr
        # We also need 'VariableExpr' class available here? Or inspect type.
        # AST classes are not imported. We can check class name.
        if type(node.object).__name__ == 'VariableExpr':
             ptr = self.scopes[-1].get(node.object.name)
             # If not in local scope? (Global?) Assumed local for now.
             if ptr:
                 # GEP: ptr, 0, index
                 zero = ir.Constant(ir.IntType(32), 0)
                 ptr = self.builder.gep(ptr, [zero, index_val])
                 return self.builder.load(ptr)
        
        # Fallback: Evaluate object to value (loads it)
        array_val = self.visit(node.object)
        
        # If index_val is Constant, we can use extract_value
        if isinstance(index_val, ir.Constant):
             # extract_value index must be python int
             idx = index_val.constant
             return self.builder.extract_value(array_val, idx)
             
        # Runtime index on SSA value: Spill to stack
        temp_ptr = self.builder.alloca(array_val.type)
        self.builder.store(array_val, temp_ptr)
        zero = ir.Constant(ir.IntType(32), 0)
        ptr = self.builder.gep(temp_ptr, [zero, index_val])
        return self.builder.load(ptr)

        func = ir.Function(self.module, func_ty, name=node.name)
        
        if node.is_kernel:
             func.calling_convention = 'spir_kernel'
        
        block = func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)
        
        # Scope dictionary: variable_name -> pointer
        self.scopes = [{}]
        
        for stmt in node.body:
            self.visit(stmt)
            
        self.builder.ret_void()

    def emit_scope_drops(self, scope):
        # Iterate in reverse order of declaration (LIFO)
        # Scope is dict, assuming insertion order preserved (Python 3.7+)
        for var_name, (ptr, type_name) in reversed(list(scope.items())):
            # Look for destructor: {TypeName}_drop(T) or {TypeName}_drop(&T)
            # Current convention: {TypeName}_drop(T) (Pass by value, consumes)
            drop_func_name = f"{type_name}_drop"
            if drop_func_name in self.module.globals:
                 drop_func = self.module.get_global(drop_func_name)
                 # Load value
                 val = self.builder.load(ptr)
                 # Call drop
                 self.builder.call(drop_func, [val])

    def visit_Block(self, node):
        self.scopes.append({})
        for stmt in node.statements:
             self.visit(stmt)
        
        # Auto-drop variables in this scope
        self.emit_scope_drops(self.scopes[-1])
        self.scopes.pop()

    def visit_VarDecl(self, node):
        # Determine type
        try:
             llvm_type = self.get_llvm_type(node.type_name)
        except Exception:
             # Handle array type [T:N] manually if not in get_llvm_type yet
             if node.type_name.startswith('['):
                 content = node.type_name[1:-1]
                 elem_type_str, size_str = content.split(':')
                 elem_ty = self.get_llvm_type(elem_type_str)
                 size = int(size_str)
                 llvm_type = ir.ArrayType(elem_ty, size)
             else:
                 raise

        # Evaluate initializer
        init_val = self.visit(node.initializer)
        
        # Auto-cast for string (array ptr -> i8*)
        if node.type_name == "string" and init_val.type != llvm_type:
             init_val = self.builder.bitcast(init_val, llvm_type)
        
        # Alloca
        ptr = self.builder.alloca(llvm_type, name=node.name)
        self.builder.store(init_val, ptr)
        
        self.builder.store(init_val, ptr)
        
        # Store in scope: (Pointer, TypeName)
        self.scopes[-1][node.name] = (ptr, node.type_name)

    def visit_Assignment(self, node):
        val = self.visit(node.value)
        
        if isinstance(node.target, VariableExpr):
             # Find stack allocation
             ptr = None
             for scope in reversed(self.scopes):
                   if node.target.name in scope:
                        ptr, _ = scope[node.target.name]
                        break
             if not ptr: raise Exception(f"Undefined var {node.target.name}")
             self.builder.store(val, ptr)
             
        elif isinstance(node.target, UnaryExpr):
             if node.target.op == '*':
                  # *ptr = val
                  # Evaluate 'ptr' (the operand of *)
                  ptr_val = self.visit(node.target.operand)
                  self.builder.store(val, ptr_val)
             else:
                  raise Exception("Invalid assignment target")

        elif isinstance(node.target, MemberAccess):
             # Assign to struct field: struct.field = val
             # We need address of the struct.
             struct_ptr = None
             
             # Case 1: struct is a variable
             if isinstance(node.target.object, VariableExpr):
                  var_name = node.target.object.name
                  for scope in reversed(self.scopes):
                        if var_name in scope:
                             struct_ptr, _ = scope[var_name]
                             break
                  if not struct_ptr: raise Exception(f"Undefined var {var_name}")
                  
             # Case 2: struct is dereferenced pointer (*ptr).field
             elif isinstance(node.target.object, UnaryExpr) and node.target.object.op == '*':
                  # ptr expression
                  struct_ptr = self.visit(node.target.object.operand)
             else:
                  raise Exception("Assignment to field requires variable or pointer dereference")
             
             # Get field index
             # Semantic analyzer should have annotated struct type? 
             # Or we look it up.
             # MemberAccess node usually has .struct_type if we annotated it in Semantic
             if not hasattr(node.target, 'struct_type'):
                  # Fallback: We need to know the type to find the index.
                  # It's tricky to get if not annotated.
                  # Assuming Semantic annotated it.
                  # If not, we might fail.
                  # Let's hope Semantic visit_MemberAccess does it.
                  # CHECK Semantic logic later.
                  # For now, raise or try?
                  raise Exception("CodeGen: Missing struct_type on MemberAccess target. Semantic analysis incomplete.")
             
             struct_name = node.target.struct_type
             field_index = self.struct_fields[struct_name][node.target.member]
             
             # GEP to field
             zero = ir.Constant(ir.IntType(32), 0)
             idx = ir.Constant(ir.IntType(32), field_index)
             field_ptr = self.builder.gep(struct_ptr, [zero, idx])
             
             self.builder.store(val, field_ptr)

    def visit_BinaryExpr(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        # Mapping token types to operations
        if node.op == 'PLUS':
            return self.builder.add(left, right, name="addtmp")
        elif node.op == 'MINUS':
            return self.builder.sub(left, right, name="subtmp")
        elif node.op == 'STAR':
            return self.builder.mul(left, right, name="multmp")
        elif node.op == 'SLASH':
            return self.builder.sdiv(left, right, name="divtmp")
        elif node.op == 'EQEQ':
            return self.builder.icmp_signed('==', left, right, name="eqtmp")
        elif node.op == 'NEQ':
            return self.builder.icmp_signed('!=', left, right, name="neqtmp")
        elif node.op == 'LT':
            return self.builder.icmp_signed('<', left, right, name="lttmp")
        elif node.op == 'GT':
            return self.builder.icmp_signed('>', left, right, name="gttmp")
        elif node.op == 'LTE':
            return self.builder.icmp_signed('<=', left, right, name="ltetmp")
        elif node.op == 'GTE':
            return self.builder.icmp_signed('>=', left, right, name="gtetmp")
        else:
            raise Exception(f"Unknown operator: {node.op}")

    def visit_IfStmt(self, node):
        cond_val = self.visit(node.condition)
        
        # Ensure condition is a boolean (i1)
        # If it's not (e.g. i32), compare it to 0
        if cond_val.type != ir.IntType(1):
             cond_val = self.builder.icmp_signed('!=', cond_val, ir.Constant(cond_val.type, 0), name="ifcond")
        
        then_bb = self.builder.append_basic_block(name="then")
        else_bb = self.builder.append_basic_block(name="else")
        merge_bb = self.builder.append_basic_block(name="ifcont")
        
        self.builder.cbranch(cond_val, then_bb, else_bb)
        
        # Generate 'then' block
        self.builder.position_at_end(then_bb)
        for stmt in node.then_branch:
            self.visit(stmt)
        if not self.builder.block.is_terminated:
            self.builder.branch(merge_bb)
            
        # Generate 'else' block
        self.builder.position_at_end(else_bb)
        if node.else_branch:
            for stmt in node.else_branch:
                self.visit(stmt)
        if not self.builder.block.is_terminated:
            self.builder.branch(merge_bb)
            
        # Continue
        self.builder.position_at_end(merge_bb)

    def visit_WhileStmt(self, node):
        cond_bb = self.builder.append_basic_block(name="whilecond")
        loop_bb = self.builder.append_basic_block(name="whileloop")
        end_bb = self.builder.append_basic_block(name="whileend")
        
        # Jump to condition
        self.builder.branch(cond_bb)
        
        # Condition Block
        self.builder.position_at_end(cond_bb)
        cond_val = self.visit(node.condition)
        if cond_val.type != ir.IntType(1):
             cond_val = self.builder.icmp_signed('!=', cond_val, ir.Constant(cond_val.type, 0), name="loopcond")
        self.builder.cbranch(cond_val, loop_bb, end_bb)
        
        # Loop Block
        self.builder.position_at_end(loop_bb)
        for stmt in node.body:
            self.visit(stmt)
        if not self.builder.block.is_terminated:
            self.builder.branch(cond_bb) # Loop back to condition
            
        # End Block
        self.builder.position_at_end(end_bb)

    def visit_IntegerLiteral(self, node):
        # Infer i32 for simplicity or u8/i64 if context?
        # For now return i32 constant
        return ir.Constant(ir.IntType(32), node.value)

    def visit_BooleanLiteral(self, node):
        val = 1 if node.value else 0
        return ir.Constant(ir.IntType(1), val)

    def visit_VariableExpr(self, node):
        # Lookup in scopes (LIFO)
        for scope in reversed(self.scopes):
            if node.name in scope:
                ptr, _ = scope[node.name]
                return self.builder.load(ptr, name=node.name)
        raise Exception(f"Ref to undefined variable: {node.name}")

    def visit_CallExpr(self, node):
        if node.callee == "print":
            # Generate printf
            # Check arg type
            val = self.visit(node.args[0])
            
            voidptr_ty = ir.IntType(8).as_pointer()
            
            if val.type == ir.IntType(32):
                fmt_str = self.visit_StringLiteral(None, name="fmt_d", value_override="%d\n\0")
            else:
                 # Assume string
                 fmt_str = self.visit_StringLiteral(None, name="fmt_s", value_override="%s\n\0")
            
            fmt_arg = self.builder.bitcast(fmt_str, voidptr_ty)
            
            if val.type == ir.IntType(32):
                self.builder.call(self.printf, [fmt_arg, val])
            else:
                val_arg = self.builder.bitcast(val, voidptr_ty)
                self.builder.call(self.printf, [fmt_arg, val_arg])
        
        elif node.callee == 'panic':
             # Print message
             msg_val = self.visit(node.args[0]) # Assuming string literal
             
             # Hardcoded "PANIC: " prefix
             prefix = self.visit_StringLiteral(None, name="panic_prefix", value_override="PANIC: ")
             
             voidptr_ty = ir.IntType(8).as_pointer()
             fmt_prefix = self.builder.bitcast(prefix, voidptr_ty)
             
             # Print "PANIC: "
             # We need a format string for pure string printing? Or just printf(str)
             # printf(str) works if str contains no %
             # Let's verify visit_StringLiteral returns a [N x i8]*
             
             # Reuse helper for printing string:
             fmt_str = self.visit_StringLiteral(None, name="fmt_s", value_override="%s\n\0")
             fmt_arg = self.builder.bitcast(fmt_str, voidptr_ty)
             
             prefix_arg = self.builder.bitcast(prefix, voidptr_ty)
             self.builder.call(self.printf, [fmt_arg, prefix_arg])
             
             msg_arg = self.builder.bitcast(msg_val, voidptr_ty)
             self.builder.call(self.printf, [fmt_arg, msg_arg])
             
             # Call exit(1)
             one = ir.Constant(ir.IntType(32), 1)
             self.builder.call(self.exit_func, [one])
             
             # Unreachable instruction to terminate block
             self.builder.unreachable()
        
        elif node.callee == 'malloc':
             # Call malloc
             size_val = self.visit(node.args[0])
             return self.builder.call(self.malloc, [size_val])
             
        elif node.callee == 'free':
             ptr_val = self.visit(node.args[0])
             # Cast to void* (i8*) if needed?
             # self.free expects i8*.
             if ptr_val.type != ir.IntType(8).as_pointer():
                  ptr_val = self.builder.bitcast(ptr_val, ir.IntType(8).as_pointer())
             return self.builder.call(self.free, [ptr_val])
             
        elif node.callee == 'realloc':
             ptr_val = self.visit(node.args[0])
             size_val = self.visit(node.args[1])
             
             if ptr_val.type != ir.IntType(8).as_pointer():
                  ptr_val = self.builder.bitcast(ptr_val, ir.IntType(8).as_pointer())
                  
             return self.builder.call(self.realloc, [ptr_val, size_val])

        elif node.callee == 'assert':
             cond_val = self.visit(node.args[0])
             msg_val = self.visit(node.args[1]) # Should be string
             
             # Compare cond != 0
             zero = ir.Constant(ir.IntType(32), 0)
             cmp = self.builder.icmp_signed('!=', cond_val, zero, name="assert_cond")
             
             # Create blocks
             fail_bb = self.builder.append_basic_block(name="assert_fail")
             cont_bb = self.builder.append_basic_block(name="assert_cont")
             
             self.builder.cbranch(cmp, cont_bb, fail_bb)
             
             # Fail Block
             self.builder.position_at_end(fail_bb)
             
             # Reuse panic logic? Or just manually do it to avoid recursion issues if helper
             # Let's manually print assertion failed
             
             # Print "ASSERTION FAILED: "
             voidptr_ty = ir.IntType(8).as_pointer()
             fails_str = self.visit_StringLiteral(None, name="assert_prefix", value_override="ASSERTION FAILED: ")
             fmt_str = self.visit_StringLiteral(None, name="fmt_s", value_override="%s\n\0")
             fmt_arg = self.builder.bitcast(fmt_str, voidptr_ty)
             fails_arg = self.builder.bitcast(fails_str, voidptr_ty)
             
             self.builder.call(self.printf, [fmt_arg, fails_arg])
             
             msg_arg = self.builder.bitcast(msg_val, voidptr_ty)
             self.builder.call(self.printf, [fmt_arg, msg_arg])
             
             one = ir.Constant(ir.IntType(32), 1)
             self.builder.call(self.exit_func, [one])
             self.builder.unreachable()
             
             # Continue Block
             self.builder.position_at_end(cont_bb)
             return ir.Constant(ir.IntType(32), 0) # void check returns 0

        # Check for Enum Variant Instantiation (Enum::Variant)
        elif '::' in node.callee:
            lhs, rhs = node.callee.split('::')
            if lhs in self.enum_definitions:
                 enum_ty, max_size = self.enum_definitions[lhs]
                 tag = self.enum_types[lhs][rhs]
                 
                 # Create struct { tag, padding }
                 enum_val = ir.Constant(enum_ty, ir.Undefined)
                 
                 # Set Tag
                 tag_val = ir.Constant(ir.IntType(32), tag)
                 enum_val = self.builder.insert_value(enum_val, tag_val, 0)
                 
                 # Set Payload
                 if node.args:
                     payload_val = self.visit(node.args[0])
                     # We need to bitcast payload_val to [Max x i8]* or similar?
                     # No, insert_value expects matching type.
                     # But our struct has [Max x i8].
                     # We cannot insert i32 into [4 x i8] directly with insert_value?
                     # LLVM structs are rigid. 
                     # Approach: Store enum to stack, bitcast pointer to payload, store payload.
                     
                     # Allocate temp enum
                     enum_ptr = self.builder.alloca(enum_ty)
                     self.builder.store(enum_val, enum_ptr)
                     
                     # Get pointer to data field (index 1)
                     # data_ptr points to [Max x i8]
                     zero = ir.Constant(ir.IntType(32), 0)
                     one = ir.Constant(ir.IntType(32), 1)
                     data_ptr = self.builder.gep(enum_ptr, [zero, one])
                     
                     # Cast data_ptr to payload type pointer (e.g. i32*)
                     payload_ptr = self.builder.bitcast(data_ptr, payload_val.type.as_pointer())
                     
                     self.builder.store(payload_val, payload_ptr)
                     
                     # Load back
                     return self.builder.load(enum_ptr)
                 else:
                     return enum_val

        elif node.callee in self.struct_types:
            # Struct Instantiation
            struct_ty = self.struct_types[node.callee]
            struct_val = ir.Constant(struct_ty, ir.Undefined)
            
            for i, arg in enumerate(node.args):
                arg_val = self.visit(arg)
                struct_val = self.builder.insert_value(struct_val, arg_val, i)
            return struct_val

             
        elif node.callee.startswith('cast<'):
             val = self.visit(node.args[0])
             target_type_name = node.callee[5:].rstrip('>')
             
             # Resolve target LLVM type
             target_ty = None
             if target_type_name == 'i32':
                 target_ty = ir.IntType(32)
             elif target_type_name == 'u8':
                 target_ty = ir.IntType(8)
             elif target_type_name.endswith('*'): 
                  # Recursion needed for T**?
                  # Just pointer cast
                  base = target_type_name[:-1]
                  if base == 'i32': target_ty = ir.IntType(32).as_pointer()
                  elif base == 'u8': target_ty = ir.IntType(8).as_pointer()
                  else: target_ty = ir.IntType(8).as_pointer() # Default to i8* for unknown base
             else:
                 # Fallback for other types, or raise error
                 raise Exception(f"Unsupported cast target type: {target_type_name}")
             
             
             src_ty = val.type
             # Integer casting
             if isinstance(src_ty, ir.IntType) and isinstance(target_ty, ir.IntType):
                 if src_ty.width > target_ty.width:
                     return self.builder.trunc(val, target_ty)
                 elif src_ty.width < target_ty.width:
                     return self.builder.zext(val, target_ty)
                 else:
                     return val
             
             # Pointer casting
             if isinstance(src_ty, ir.PointerType) and isinstance(target_ty, ir.PointerType):
                 return self.builder.bitcast(val, target_ty)
                 
             # Ptr <-> Int
             if isinstance(src_ty, ir.PointerType) and isinstance(target_ty, ir.IntType):
                 return self.builder.ptrtoint(val, target_ty)
             if isinstance(src_ty, ir.IntType) and isinstance(target_ty, ir.PointerType):
                 return self.builder.inttoptr(val, target_ty)
                 
             # Fallback
             return self.builder.bitcast(val, target_ty)
             
        elif node.callee.startswith('sizeof<'):
             type_name = node.callee[7:].rstrip('>')
             llvm_ty = ir.IntType(32) # Default
             if type_name == 'i32': llvm_ty = ir.IntType(32)
             elif type_name == 'u8': llvm_ty = ir.IntType(8)
             
             null_ptr = ir.Constant(llvm_ty.as_pointer(), None)
             gep = self.builder.gep(null_ptr, [ir.Constant(ir.IntType(32), 1)])
             return self.builder.ptrtoint(gep, ir.IntType(32))
             
        elif node.callee.startswith('ptr_offset<'):
             ptr_val = self.visit(node.args[0])
             idx_val = self.visit(node.args[1])
             return self.builder.gep(ptr_val, [idx_val])
             
        elif node.callee == 'memcpy':
             dest = self.visit(node.args[0])
             src = self.visit(node.args[1])
             size = self.visit(node.args[2])
             
             # Cast to i8* if needed
             void_ptr = ir.IntType(8).as_pointer()
             if dest.type != void_ptr: dest = self.builder.bitcast(dest, void_ptr)
             if src.type != void_ptr: src = self.builder.bitcast(src, void_ptr)
             
             # Volatile = false
             is_volatile = ir.Constant(ir.IntType(1), 0)
             
             return self.builder.call(self.memcpy, [dest, src, size, is_volatile])
        
        elif node.callee == "gpu::global_id":
             return ir.Constant(ir.IntType(32), 0)
        else:
             # Try to find function in module
             if node.callee in self.module.globals:
                 callee_func = self.module.globals[node.callee]
                 processed_args = [self.visit(arg) for arg in node.args]
                 return self.builder.call(callee_func, processed_args)
             else:
                 raise Exception(f"Unknown function call: {node.callee}")

    def visit_StringLiteral(self, node, name="str", value_override=None):
        # Create a global constant string
        if value_override:
            value = value_override
            if not value.endswith('\0'): 
                value += '\0'
        else:
            if isinstance(node, ir.Constant):
                # Fallback if internal code passes constants (which I removed, but just in case)
                # But really execution shouldn't reach here if callers are fixed.
                # Let's just create a unique name based on content if possible
                value = node.constant.decode('utf-8')
            else:
                value = node.value + '\0'
            
        # Optimization: Deduplicate strings?
        # For now, just generate unique name if not provided or collision risk
        # But instructions verify unique name usage
        
        # Use simple hashing or counter?
        # Helper: self.module.get_unique_name(name) is standard usage
        
        c_str_val = ir.Constant(ir.ArrayType(ir.IntType(8), len(value)), bytearray(value.encode("utf8")))
        
        unique_name = self.module.get_unique_name(name)
        global_var = ir.GlobalVariable(self.module, c_str_val.type, name=unique_name)
        global_var.linkage = 'internal'
        global_var.global_constant = True
        global_var.initializer = c_str_val
        # Return i8* pointer to start
        return self.builder.bitcast(global_var, ir.IntType(8).as_pointer())

    def visit_MatchExpr(self, node):
        # 1. Evaluate value
        val = self.visit(node.value) # Expect {tag, data} (value or pointer?)
        
        # Allocate val to stack to easily access payload address later
        val_ptr = self.builder.alloca(val.type)
        self.builder.store(val, val_ptr)
        
        # Extract Tag
        tag_val = self.builder.extract_value(val, 0)
        
        # Create blocks
        merge_bb = self.builder.append_basic_block("match_merge")
        
        # Switch
        switch = self.builder.switch(tag_val, merge_bb)
        
        enum_name = node.enum_name
        
        # Cases
        for case in node.cases:
            tag_id = self.enum_types[enum_name][case.variant_name]
            case_bb = self.builder.append_basic_block(f"case_{case.variant_name}")
            switch.add_case(ir.Constant(ir.IntType(32), tag_id), case_bb)
            
            self.builder.position_at_end(case_bb)
            
            # Bound Variable
            if case.var_name:
                # Get Payload Address
                zero = ir.Constant(ir.IntType(32), 0)
                one = ir.Constant(ir.IntType(32), 1)
                data_ptr = self.builder.gep(val_ptr, [zero, one])
                
                # Bitcast to payload type
                payload_ty = self.enum_payloads[enum_name][case.variant_name]
                if payload_ty:
                    cast_ptr = self.builder.bitcast(data_ptr, payload_ty.as_pointer())
                    
                    # Register in scope
                    self.scopes[-1][case.var_name] = cast_ptr 
                
            # Execute body
            self.visit(case.body)
            
            # Branch to merge
            if not self.builder.block.is_terminated:
                self.builder.branch(merge_bb)
                
        self.builder.position_at_end(merge_bb)
        return ir.Constant(ir.IntType(32), 0) # Void

    def visit_ReturnStmt(self, node):
        # Evaluate return value FIRST (before drops)
        ret_val = None
        if node.value:
            ret_val = self.visit(node.value)
            
        # Unwind scopes: Drop everything in current function scopes (LIFO)
        # Note: self.scopes[0] is Global scope? No, self.scopes tracks local scopes.
        # But we don't want to drop globals here?
        # Actually in _declare_function we might set up scopes.
        # Let's assume self.scopes contains all local scopes.
        for scope in reversed(self.scopes):
             self.emit_scope_drops(scope)
        
        if ret_val:
            self.builder.ret(ret_val)
        else:
            self.builder.ret_void()

    def _declare_function(self, node):
        if node.generics: return

        # Determine function type
        arg_types = []
        for _, ptype in node.params:
             # Handle arrays manually for now or add to get_llvm_type
             if ptype.startswith('['):
                  content = ptype[1:-1]
                  elem_str, size_str = content.split(':')
                  elem_ty = self.get_llvm_type(elem_str)
                  arg_types.append(ir.ArrayType(elem_ty, int(size_str)))
             elif ptype in self.enum_definitions: # Enums need special handling? 
                  # get_llvm_type might handle if added? 
                  # For now fallback or manual
                  ety, _ = self.enum_definitions[ptype]
                  arg_types.append(ety)
             else:
                  arg_types.append(self.get_llvm_type(ptype))

        ret_type = ir.VoidType()
        if node.return_type != 'void':
             ret_type = self.get_llvm_type(node.return_type)
             
        func_ty = ir.FunctionType(ret_type, arg_types)
        
        # Check if exists
        try:
            func = self.module.get_global(node.name)
        except KeyError:
            func = ir.Function(self.module, func_ty, name=node.name)
            
        if node.is_kernel:
             func.calling_convention = 'spir_kernel'
        
        return func

    def visit_FunctionDef(self, node):
        if node.generics: return
        
        # Function already declared in Pass 2
        func = self.module.get_global(node.name)
        
        entry_block = func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(entry_block)
        
        # Create new scope
        self.scopes.append({})
        
        # Register arguments in scope
        for i, (pname, ptype) in enumerate(node.params):
             arg_val = func.args[i]
             arg_val.name = pname
             alloca = self.builder.alloca(self.get_llvm_type(ptype), name=pname)
             self.builder.store(arg_val, alloca)
             
             # Store in scope: (Pointer, TypeName)
             self.scopes[-1][pname] = (alloca, ptype)
        
        # Process body
        for stmt in node.body:
            self.visit(stmt)
            
        # Add return void/undef if missing
        if not self.builder.block.is_terminated:
            if isinstance(func.function_type.return_type, ir.VoidType):
                self.builder.ret_void()
            elif isinstance(func.function_type.return_type, ir.IntType):
                self.builder.ret(ir.Constant(func.function_type.return_type, 0))
            elif isinstance(func.function_type.return_type, ir.PointerType):
                 self.builder.ret(ir.Constant(func.function_type.return_type, None)) # Null ptr
            else:
                self.builder.unreachable()
            
        self.scopes.pop()
