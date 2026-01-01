lines = open('selfhost/stage5_full.nxl', 'r', encoding='utf-8').readlines()

# Find line with "consume(p, 5); # )" - linha 553
target_idx = None
for i, line in enumerate(lines):
    if i >= 552 and i <= 554 and 'consume(p, 5); # )' in line:
        target_idx = i
        break

if target_idx:
    # Insert new code after line 553, before line 554 (consume { )
    new_code = '''    
    # Check for optional return type: -> type
    let tok_ptr_ret: *Token = ptr_offset::<Token>((*p).tokens, (*p).pos);
    let s_ret: i32 = (*tok_ptr_ret).start;
    let c_ret: u8 = ptr_offset::<u8>((*p).src, s_ret)[0];
    # '-' is 45, check if we have '->'
    if (cast::<i32>(c_ret) == 45) {
        consume(p, 5); # ->
        consume(p, 1); # type name (i32, etc.)
    }
    
'''
    lines.insert(target_idx + 1, new_code)
    
    with open('selfhost/stage5_full.nxl', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"Added return type parsing at line {target_idx + 1}")
else:
    print("Target line not found")
