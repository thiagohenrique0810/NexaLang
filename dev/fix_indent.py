
src_path = 'bootstrap/codegen.py'
with open(src_path, 'r') as f:
    lines = f.readlines()

# Verify content matches expectation
if 'elif isinstance(node.target, MemberAccess):' in lines[758]:
    # Delete bad lines (approx 11 lines)
    # The bad block ends at raise Exception
    end_idx = 758
    found = False
    for i in range(758, 780):
        if 'elif isinstance(node.target, UnaryExpr):' in lines[i]:
            end_idx = i
            found = True
            break
            
    if found:
        # Replace [758:end_idx] with good code
        good_code = [
            "        elif isinstance(node.target, MemberAccess):\n",
            "             struct_val = self.visit(node.target.object)\n",
            "             if not hasattr(node.target, 'struct_type'): raise Exception('Missing struct_type')\n",
            "             struct_name = node.target.struct_type\n",
            "             field_idx = self.struct_fields[struct_name][node.target.member]\n",
            "             if isinstance(struct_val.type, ir.PointerType):\n",
            "                 ptr = self.builder.gep(struct_val, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), field_idx)])\n",
            "                 self.builder.store(val, ptr)\n",
            "             else:\n",
            "                 raise Exception('Cannot assign to member of Value-type struct')\n",
            "\n"
        ]
        
        new_lines = lines[:758] + good_code + lines[end_idx:]
        
        # Also fix UnaryExpr indentation if needed (it seemed broken in view)
        # Check the line after insertion
        unary_idx = 758 + len(good_code)
        if len(new_lines) > unary_idx:
            line = new_lines[unary_idx]
            if 'elif isinstance(node.target, UnaryExpr):' in line:
                if not line.startswith('        '):
                    new_lines[unary_idx] = '        ' + line.lstrip()
        
        with open(src_path, 'w') as f:
            f.writelines(new_lines)
        print("Fixed indentation")
    else:
        print("Couldn't find end of block")
else:
    print("Start line mismatch: " + lines[758])
