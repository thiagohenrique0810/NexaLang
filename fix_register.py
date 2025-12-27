lines = open('bootstrap/semantic.py', 'r', encoding='utf-8').readlines()

# Find register_impl_methods
for i, line in enumerate(lines):
    if 'def register_impl_methods' in line:
        # Found it at line i
        # Need to:
        # 1. Add struct_methods init after struct_name line
        # 2. Add &mut Self handling
        # 3. Add method to struct_methods dict before mangling
        
        # Find struct_name line
        struct_name_idx = i + 1
        
        # Insert struct_methods init
        lines.insert(struct_name_idx + 1, '        \n')
        lines.insert(struct_name_idx + 2, '        # Initialize methods dict for this struct if needed\n')
        lines.insert(struct_name_idx + 3, '        if struct_name not in self.struct_methods:\n')
        lines.insert(struct_name_idx + 4, '            self.struct_methods[struct_name] = {}\n')
        lines.insert(struct_name_idx + 5, '        \n')
        
        # Find the elif ptype == '&Self': line and add &mut Self after it
        for j in range(i, i + 30):
            if "elif ptype == '&Self':" in lines[j]:
                # Insert after the method.params[0] = ... line
                insert_idx = j + 2
                lines.insert(insert_idx, "                       elif ptype == '\u0026mut Self':\n")
                lines.insert(insert_idx + 1, f'                           method.params[0] = (\"self\", f\"{{struct_name}}*\")\n')
                break
        
        # Find "# Mangle name" and add struct_methods registration before it
        for j in range(i, i + 35):
            if '# Mangle name' in lines[j]:
                lines.insert(j, '             \n')
                lines.insert(j + 1, '             # Register method in struct_methods for resolution\n')
                lines.insert(j + 2, '             self.struct_methods[struct_name][method.name] = method\n')
                lines.insert(j + 3, '             \n')
                break
        
        break

with open('bootstrap/semantic.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Updated register_impl_methods")
