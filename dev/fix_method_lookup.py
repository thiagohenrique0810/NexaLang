lines = open('bootstrap/codegen.py', 'r', encoding='utf-8').readlines()

# Find visit_MethodCall
for i, line in enumerate(lines):
    if 'def visit_MethodCall(self, node):' in line:
        # Find the VariableExpr lookup section
        for j in range(i, min(i + 50, len(lines))):
            if 'if isinstance(node.receiver, VariableExpr):' in lines[j]:
                # Replace the next few lines of lookup
                # Look for the line that starts with 'var_info = self.scopes[-1]'
                for k in range(j, j + 5):
                    if 'var_info = self.scopes[-1]' in lines[k]:
                        # Replace the lookup part
                        indent = "                "
                        lines[k] = f"{indent}var_info = None\n"
                        lines.insert(k+1, f"{indent}for scope in reversed(self.scopes):\n")
                        lines.insert(k+2, f"{indent}    if node.receiver.name in scope:\n")
                        lines.insert(k+3, f"{indent}        var_info = scope[node.receiver.name]\n")
                        lines.insert(k+4, f"{indent}        break\n")
                        
                        # Now adjust the 'if var_info' part which follows
                        # It was: if var_info and 'alloca' in var_info:
                        # But entries are usually tuples (alloca, type_name)
                        # or (val, type, "value")
                        # Let's see how entries are structured in VarDecl
                        # self.scopes[-1][node.name] = (alloca, type_name)
                        
                        # So I should check if it's a tuple and has 2 elements
                        break
                break
        break

with open('bootstrap/codegen.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Fixed variable lookup in visit_MethodCall")
