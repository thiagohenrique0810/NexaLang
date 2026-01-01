import re

def patch_codegen():
    content = open('bootstrap/codegen.py', 'r', encoding='utf-8').read()
    
    # 1. Add _current_generics to __init__
    if 'self._current_generics = []' not in content:
        content = content.replace('self.struct_fields = {}', 'self.struct_fields = {}\n        self._current_generics = []')

    # 2. Update get_llvm_type to handle generic params
    # We'll insert it before the last fallback
    new_generic_check = '''
        # Handle generic parameters as placeholders (e.g. 'T')
        if type_name in self._current_generics:
             return ir.IntType(8) # Placeholder
'''
    if 'if type_name in self._current_generics:' not in content:
        # Find where to insert. Before the 'if "<" in type_name:' block we added.
        content = content.replace('# Handle generic types via erasure', new_generic_check + '        # Handle generic types via erasure')

    # 3. Reset _current_generics in visit_StructDef
    # We already added the start. Now we need the end.
    # Actually, let's just use a try-finally style if possible, or just reset at end of visit_StructDef.
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'def visit_StructDef(self, node):' in line:
            # Find end of method
            for j in range(i+1, len(lines)):
                if (lines[j].startswith('    def ') or lines[j].startswith('class ')) and lines[j].strip():
                    # Insert reset before this line
                    lines.insert(j-1, '        self._current_generics = []')
                    break
            break
    
    content = '\n'.join(lines)

    # 4. Fix the bug in get_llvm_type we saw earlier
    content = content.replace('return self.struct_types[type_name]', 'pass # Handled by fallback')
    # Actually, let's just replace the whole get_llvm_type logic to be clean.
    
    with open('bootstrap/codegen.py', 'w', encoding='utf-8') as f:
        f.write(content)

patch_codegen()
print("Patched codegen for generic support")
