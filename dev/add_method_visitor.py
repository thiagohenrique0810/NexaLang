lines = open('bootstrap/codegen.py', 'r', encoding='utf-8').readlines()

# Find visit_ArrayLiteral (line after visit_MemberAccess)
for i, line in enumerate(lines):
    if 'def visit_ArrayLiteral(self' in line:
        # Insert visit_MethodCall before it
        method_code = '''    def visit_MethodCall(self, node):
        # Desugar obj.method(args) -> Type_method(obj, args)
        # 1. Visit receiver
        receiver_val = self.visit(node.receiver)
        
        # 2. Get struct type from semantic analysis
        struct_type = node.struct_type
        
        # 3. Build mangled function name
        func_name = f"{struct_type}_{node.method_name}"
        
        if func_name not in self.module.globals:
            raise Exception(f"CodeGen Error: Method '{func_name}' not found in module")
        
        func = self.module.globals[func_name]
        
        # 4. Prepare arguments: [receiver, ...args]
        args = [receiver_val] + [self.visit(arg) for arg in node.args]
        
        # 5. Call the mangled function
        return self.builder.call(func, args)

'''
        lines.insert(i, method_code)
        break

with open('bootstrap/codegen.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Added visit_MethodCall to codegen.py")
