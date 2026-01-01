lines = open('bootstrap/codegen.py', 'r', encoding='utf-8').readlines()

for i, line in enumerate(lines):
    if 'def get_llvm_type(self, type_name):' in line:
        # Find the end of basic type checks
        for j in range(i, i + 50):
            if 'elif type_name in self.struct_types:' in lines[j]:
                # Found it
                # Replace that elif and the following lines
                lines[j] = '        elif type_name in self.struct_types:\n            return self.struct_types[type_name]\n        \n        # Handle generic types via erasure\n        if "<" in type_name:\n            base = type_name.split("<")[0]\n            if base in self.struct_types:\n                return self.struct_types[base]\n            if base in self.enum_definitions:\n                enum_ty, _ = self.enum_definitions[base]\n                return enum_ty\n'
                break
        break

with open('bootstrap/codegen.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print("Updated get_llvm_type")
