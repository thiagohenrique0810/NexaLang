import os

filepath = 'bootstrap/codegen.py'
with open(filepath, 'r') as f:
    lines = f.readlines()

# Find line with "processed_args = [self.visit(arg) for arg in node.args]"
for i, line in enumerate(lines):
    if 'processed_args = [self.visit(arg) for arg in node.args]' in line:
        # Insert debug prints after this line
        indent = '                  '
        debug_lines = [
            indent + 'print(f"\\nDEBUG CALL {node.callee}:")\n',
            indent + 'print(f"  Expected param types: {[str(p) for p in callee_func.function_type.args]}")\n',
            indent + 'print(f"  Actual arg types: {[str(a.type) for a in processed_args]}")\n',
        ]
        lines = lines[:i+1] + debug_lines + lines[i+1:]
        break

with open(filepath, 'w') as f:
    f.writelines(lines)

print("Added debug to codegen.py")
