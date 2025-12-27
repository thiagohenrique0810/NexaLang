
import os

filepath = 'bootstrap/codegen.py'
with open(filepath, 'r') as f:
    content = f.read()

target = 'elif isinstance(node.target, UnaryExpr):'
payload = """        elif isinstance(node.target, MemberAccess):
             struct_val = self.visit(node.target.object)
             if not hasattr(node.target, 'struct_type'): raise Exception('Missing struct_type')
             struct_name = node.target.struct_type
             field_idx = self.struct_fields[struct_name][node.target.member]
             if isinstance(struct_val.type, ir.PointerType):
                 ptr = self.builder.gep(struct_val, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), field_idx)])
                 self.builder.store(val, ptr)
             else:
                 raise Exception('Cannot assign to member of Value-type struct')

"""

if payload.strip() not in content:
    idx = content.find(target)
    if idx != -1:
        new_content = content[:idx] + payload + content[idx:]
        with open(filepath, 'w') as f:
            f.write(new_content)
        print("Patched codegen.py")
    else:
        print("Target not found")
else:
    print("Already patched")
