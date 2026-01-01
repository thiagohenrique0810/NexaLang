import re

def patch_all_stores(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    in_relevant_method = False
    
    # regex to match self.builder.store(val, ptr)
    store_pattern = re.compile(r'^(\s*)self\.builder\.store\(([^,]+),\s*([^)]+)\)')

    for line in lines:
        if 'def visit_Assignment(self, node):' in line or 'def visit_VarDecl(self, node):' in line:
            in_relevant_method = True
            new_lines.append(line)
            continue
            
        if in_relevant_method:
            if line.startswith('    def ') or line.startswith('class '):
                in_relevant_method = False
                new_lines.append(line)
                continue
            
            match = store_pattern.match(line)
            if match:
                indent = match.group(1)
                val_expr = match.group(2)
                ptr_expr = match.group(3)
                
                # Special cases: if it's already patched, skip
                if 'if ' in lines[len(new_lines)-1] and 'bitcast' in lines[len(new_lines)-1]:
                    new_lines.append(line)
                else:
                    # Insert bitcast before store
                    new_lines.append(f'{indent}# Auto-bitcast for type erasure\n')
                    new_lines.append(f'{indent}if {val_expr}.type != {ptr_expr}.type.pointee:\n')
                    new_lines.append(f'{indent}    {ptr_expr} = self.builder.bitcast({ptr_expr}, {val_expr}.type.as_pointer())\n')
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

patch_all_stores('bootstrap/codegen.py')
print("Patched all stores in visit_Assignment and visit_VarDecl with bitcast hacks")
