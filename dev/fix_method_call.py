lines = open('bootstrap/codegen.py', 'r', encoding='utf-8').readlines()

# Find visit_MethodCall
for i, line in enumerate(lines):
    if 'def visit_MethodCall(self, node):' in line:
        # Find the section that handles pointer conversion (around line 573-580)
        for j in range(i, min(i + 50, len(lines))):
            if 'Method expects pointer but we have value' in lines[j]:
                # Replace this section
                # Find the start and end of the if block
                start_idx = j - 1  # Line with: if isinstance(expected_param_type...
                # Find the else clause
                end_idx = j
                for k in range(j, min(j + 15, len(lines))):
                    if lines[k].strip().startswith('else:'):
                        end_idx = k
                        break
                
                # New code
                new_code = '''        if isinstance(expected_param_type, ir.PointerType) and not isinstance(receiver_val.type, ir.PointerType):
            # Method expects pointer but we have value
            # For &mut self, we need the actual variable address, not a temporary!
            from parser import VariableExpr
            if isinstance(node.receiver, VariableExpr):
                # Look up variable's stack location
                var_info = self.scopes[-1].get(node.receiver.name)
                if var_info and 'alloca' in var_info:
                    receiver_arg = var_info['alloca']
                else:
                    # Fallback: create temporary
                    temp = self.builder.alloca(receiver_val.type, name="method_self_tmp")
                    self.builder.store(receiver_val, temp)
                    receiver_arg = temp
            else:
                # Receiver is expression - need temporary
                temp = self.builder.alloca(receiver_val.type, name="method_self_tmp")
                self.builder.store(receiver_val, temp)
                receiver_arg = temp
'''
                # Remove old lines
                del lines[start_idx:end_idx]
                # Insert new code
                lines.insert(start_idx, new_code)
                break
        break

with open('bootstrap/codegen.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Updated visit_MethodCall")
