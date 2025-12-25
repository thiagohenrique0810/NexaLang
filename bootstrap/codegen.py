from llvmlite import ir

class CodeGen:
    def __init__(self):
        self.module = ir.Module(name="nexalang_module")
        self.builder = None
        self.printf = None
        self._declare_printf()

    def _declare_printf(self):
        voidptr_ty = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        self.printf = ir.Function(self.module, printf_ty, name="printf")

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

    def visit_FunctionDef(self, node):
        func_ty = ir.FunctionType(ir.VoidType(), [])
        func = ir.Function(self.module, func_ty, name=node.name)
        block = func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)
        
        for stmt in node.body:
            self.visit(stmt)
            
        self.builder.ret_void()

    def visit_CallExpr(self, node):
        if node.callee == "print":
            # Handle print specially using printf
            arg = self.visit(node.args[0])
            fmt_str = self.visit_StringLiteral(ir.Constant(ir.ArrayType(ir.IntType(8), 4), bytearray("%s\n\0", "utf8")), name="fmt")
            
            # Create global constant for string if not already
            # Ideally StringLiteral visit should return a pointer to the global string
            
            # Simple hack for print(string_literal)
            voidptr_ty = ir.IntType(8).as_pointer()
            
            # Bitcast to i8*
            fmt_arg = self.builder.bitcast(fmt_str, voidptr_ty)
            val_arg = self.builder.bitcast(arg, voidptr_ty)
            
            self.builder.call(self.printf, [fmt_arg, val_arg])
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
