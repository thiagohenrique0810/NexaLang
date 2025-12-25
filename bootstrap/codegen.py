from llvmlite import ir

class CodeGen:
    def __init__(self):
        self.module = ir.Module(name="nexalang_module")
        self.builder = None
        self.printf = None
        self._declare_printf()
        self.struct_types = {} # name -> ir.LiteralStructType
        self.struct_fields = {} # name -> {field_name: index}

    def _declare_printf(self):
        voidptr_ty = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        self.printf = ir.Function(self.module, printf_ty, name="printf")

    def visit_StructDef(self, node):
        # Create LLVM Struct Type
        field_types = []
        field_indices = {}
        for idx, (name, type_name) in enumerate(node.fields):
            if type_name == 'i32':
                field_types.append(ir.IntType(32))
            else:
                 raise Exception(f"Unsupported field type: {type_name}")
            field_indices[name] = idx
            
        struct_ty = ir.LiteralStructType(field_types)
        self.struct_types[node.name] = struct_ty
        self.struct_fields[node.name] = field_indices

    def generate(self, ast):
        for node in ast:
            self.visit(node)
        return str(self.module)

    def visit(self, node):
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f"No visit_{type(node).__name__} method")
        
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

    def visit_FunctionDef(self, node):
        func_ty = ir.FunctionType(ir.VoidType(), [])
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

    def visit_VarDecl(self, node):
        # Determine type
        # Determine type
        if node.type_name == "i32":
            llvm_type = ir.IntType(32)
        elif node.type_name in self.struct_types:
            llvm_type = self.struct_types[node.type_name]
        elif node.type_name.startswith('['):
            # Parse [T:N]
            content = node.type_name[1:-1]
            elem_type_str, size_str = content.split(':')
            size = int(size_str)
            if elem_type_str == 'i32':
                elem_ty = ir.IntType(32)
            else:
                 raise Exception(f"Unsupported array element type: {elem_type_str}")
            llvm_type = ir.ArrayType(elem_ty, size)
        else:
            raise Exception(f"Unknown type: {node.type_name}")

        # Evaluate initializer
        init_val = self.visit(node.initializer)
        
        # Alloca
        ptr = self.builder.alloca(llvm_type, name=node.name)
        self.builder.store(init_val, ptr)
        
        # Store in scope
        self.scopes[-1][node.name] = ptr

    def visit_Assignment(self, node):
        # Lookup in scopes (LIFO) to get the pointer
        ptr = None
        for scope in reversed(self.scopes):
            if node.name in scope:
                ptr = scope[node.name]
                break
        
        if not ptr:
            raise Exception(f"Undefined variable in assignment: {node.name}")
            
        # Evaluate value
        val = self.visit(node.value)
        
        # Store new value
        self.builder.store(val, ptr)

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
        return ir.Constant(ir.IntType(32), node.value)

    def visit_VariableExpr(self, node):
        # Lookup in scopes (LIFO)
        for scope in reversed(self.scopes):
            if node.name in scope:
                return self.builder.load(scope[node.name], name=node.name)
        raise Exception(f"Undefined variable: {node.name}")

    def visit_CallExpr(self, node):
        if node.callee == "print":
            arg = self.visit(node.args[0])
            
            # Check type of arg to choose format string
            if isinstance(arg.type, ir.IntType):
                fmt_str = self.visit_StringLiteral(ir.Constant(ir.ArrayType(ir.IntType(8), 4), bytearray("%d\n\0", "utf8")), name="fmt_d")
            else:
                fmt_str = self.visit_StringLiteral(ir.Constant(ir.ArrayType(ir.IntType(8), 4), bytearray("%s\n\0", "utf8")), name="fmt_s")
                
            voidptr_ty = ir.IntType(8).as_pointer()
            
            fmt_arg = self.builder.bitcast(fmt_str, voidptr_ty)
            
            if isinstance(arg.type, ir.IntType):
                self.builder.call(self.printf, [fmt_arg, arg])
            else:
                val_arg = self.builder.bitcast(arg, voidptr_ty)
                self.builder.call(self.printf, [fmt_arg, val_arg])
        elif node.callee in self.struct_types:
             # Struct Instantiation
             struct_ty = self.struct_types[node.callee]
             # Create a stack allocation for the struct (or just a constant value?)
             # To make it mutable/addressable, better to alloca, store fields, and load.
             # Wait, generate() returns a Value.
             # Let's create an undefined struct value and insert values.
             
             struct_val = ir.Constant(struct_ty, ir.Undefined)
             for i, arg in enumerate(node.args):
                 val = self.visit(arg)
                 struct_val = self.builder.insert_value(struct_val, val, i)
                 
             return struct_val
             
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

    def visit_StringLiteral(self, node, name="str"):
        # Create a global constant string
        if isinstance(node, ir.Constant):
             # Internal use for formatting string
             c_str_val = node
        else:
            val = node.value + "\0" # Null terminator
            c_str_val = ir.Constant(ir.ArrayType(ir.IntType(8), len(val)), bytearray(val.encode("utf8")))
        
        global_var = ir.GlobalVariable(self.module, c_str_val.type, name=self.module.get_unique_name(name))
        global_var.linkage = 'internal'
        global_var.global_constant = True
        global_var.initializer = c_str_val
        return global_var
