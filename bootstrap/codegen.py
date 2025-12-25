from llvmlite import ir

class CodeGen:
    def __init__(self):
        self.module = ir.Module(name="nexalang_module")
        self.builder = None
        self.printf = None
        self.exit_func = None
        self._declare_printf()
        self._declare_exit()
        self.struct_types = {} # name -> ir.LiteralStructType
        self.struct_fields = {} # name -> {field_name: index}
        self.enum_types = {} # name -> {variant: tag_id}
        self.enum_payloads = {} # name -> {variant: payload_type}
        self.enum_definitions = {} # name -> (ir_struct_type, payload_size)
        self.scopes = []

    def _declare_printf(self):
        voidptr_ty = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        self.printf = ir.Function(self.module, printf_ty, name="printf")

    def _declare_exit(self):
        exit_ty = ir.FunctionType(ir.VoidType(), [ir.IntType(32)])
        self.exit_func = ir.Function(self.module, exit_ty, name="exit")

    def visit_StructDef(self, node):
        # Create LLVM Struct Type
        field_types = []
        for _, type_name in node.fields:
             if type_name == 'i32':
                 field_types.append(ir.IntType(32))
             else:
                 raise Exception(f"Unsupported struct field type: {type_name}")
        
        struct_ty = ir.LiteralStructType(field_types)
        self.struct_types[node.name] = struct_ty
        
        # Map field names to indices
        self.struct_fields[node.name] = {name: i for i, (name, _) in enumerate(node.fields)}

    def visit_EnumDef(self, node):
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
                if payloads[0] == 'i32':
                    max_size = max(max_size, 4)
                    variant_payload_types[vname] = ir.IntType(32)
                else:
                    # Support Structs later?
                    pass
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

    def visit_MatchExpr(self, node):
        # 1. Evaluate value
        val = self.visit(node.value) # Expect {tag, data} (value or pointer?)
        # If val is loaded (Value), we need to extract tag.
        
        # If val is pointer (optimization), we load tag.
        # Assuming val is loaded Value for now.
        
        # Allocate val to stack to easily access payload address later
        val_ptr = self.builder.alloca(val.type)
        self.builder.store(val, val_ptr)
        
        # Extract Tag
        tag_val = self.builder.extract_value(val, 0)
        
        # Create blocks
        merge_bb = self.builder.append_basic_block("match_merge")
        
        # Switch
        switch = self.builder.switch(tag_val, merge_bb)
        
        # Determine Enum Name from val type? 
        # Semantic Analysis node could have annotation?
        # Or look up by traversing all enum definitions? Slow.
        # But we need to know Mapping {Variant -> Tag} which is in self.enum_types[EnumName].
        # We need EnumName. 
        # AST Node `match val` knows `val` type from Semantic.
        # But CodeGen generic `visit` doesn't pass type info.
        # We need to rely on `SemanticAnalyzer` annotating the AST node with `enum_type`.
        # NOTE: I missed annotating `node.enum_type` in semantic.py!
        # I SHOULD FIX SEMANTIC.PY FIRST.
        # BUT I AM IN CODEGEN STEP.
        # For now, let's assume I will fix Semantic.py or infer it?
        # I can't infer easily.
        # WAIT! `node.value` is AST node. Semantic visit_MatchExpr calculated `val_type`.
        # I should have attached it to `node`.
        # I will patch `SemanticAnalyzer` in next step if I forgot.
        # Let's write CodeGen assuming `node.enum_name` exists.
        
        enum_name = node.enum_name # Expecting this!
        
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
                # Need payload type!
                payload_ty = self.enum_payloads[enum_name][case.variant_name]
                if payload_ty:
                    cast_ptr = self.builder.bitcast(data_ptr, payload_ty.as_pointer())
                    payload_val = self.builder.load(cast_ptr)
                    
                    # Register in scope
                    self.scopes[-1][case.var_name] = cast_ptr # Pointer for variable? 
                    # Wrapper uses alloca for VarDecl. Here we have a pointer.
                    # Standard `visit_VariableExpr` loads if it finds a pointer.
                    # So we map name -> pointer.
                
            # Execute body
            self.visit(case.body)
            
            # Branch to merge
            if not self.builder.block.is_terminated:
                self.builder.branch(merge_bb)
                
        self.builder.position_at_end(merge_bb)
        return ir.Constant(ir.IntType(32), 0) # Void
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
        elif node.type_name in self.enum_definitions:
            llvm_type, _ = self.enum_definitions[node.type_name]
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
        return global_var

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

    def visit_FunctionDef(self, node):
        func_ty = ir.FunctionType(ir.VoidType(), [])
        func = ir.Function(self.module, func_ty, name=node.name)
        
        if node.is_kernel:
             func.calling_convention = 'spir_kernel'
             # Add metadata for OpenCL/Vulkan later if needed
             
        entry_block = func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(entry_block)
        
        # Create new scope
        self.scopes.append({})
        
        # Process body
        for stmt in node.body:
            self.visit(stmt)
            
        # Add return void if missing
        if not self.builder.block.is_terminated:
            self.builder.ret_void()
            
        self.scopes.pop()
