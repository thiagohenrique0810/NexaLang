lines = open('bootstrap/codegen.py', 'r', encoding='utf-8').readlines()

import_found = False
for line in lines:
    if 'from parser import VariableExpr' in line:
        import_found = True
        break

# The script I ran before added an import inside visit_MethodCall. 
# I'll replace the whole visit_MethodCall to be clean.

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if 'def visit_MethodCall(self, node):' in line:
        start_idx = i
    if start_idx != -1 and 'def visit_ArrayLiteral(self, node):' in line:
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    new_method = '''    def visit_MethodCall(self, node):
        # Desugar obj.method(args) -> Type_method(obj, args)
        receiver_val = self.visit(node.receiver)
        struct_type = node.struct_type
        func_name = f"{struct_type}_{node.method_name}"
        
        if func_name not in self.module.globals:
            raise Exception(f"CodeGen Error: Method '{func_name}' not found in module")
        
        func = self.module.globals[func_name]
        expected_param_type = func.function_type.args[0]
        
        receiver_arg = None
        if isinstance(expected_param_type, ir.PointerType) and not isinstance(receiver_val.type, ir.PointerType):
            # Method expects pointer but we have value.
            from parser import VariableExpr
            if isinstance(node.receiver, VariableExpr):
                # Look up variable's address in all scopes
                for scope in reversed(self.scopes):
                    if node.receiver.name in scope:
                        entry = scope[node.receiver.name]
                        if isinstance(entry, tuple) and (len(entry) == 2 or (len(entry) == 3 and entry[2] != "value")):
                            receiver_arg = entry[0] # The alloca
                        break
            
            if receiver_arg is None:
                # Fallback: create temporary
                temp = self.builder.alloca(receiver_val.type, name="method_self_tmp")
                self.builder.store(receiver_val, temp)
                receiver_arg = temp
        else:
            receiver_arg = receiver_val
        
        args = [receiver_arg] + [self.visit(arg) for arg in node.args]
        return self.builder.call(func, args)

'''
    del lines[start_idx:end_idx]
    lines.insert(start_idx, new_method)
    
    with open('bootstrap/codegen.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("Successfully replaced visit_MethodCall")
else:
    print(f"Failed to find indices: start={start_idx}, end={end_idx}")
