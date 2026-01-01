def fix_indent(filepath):
    lines = open(filepath, 'r', encoding='utf-8').readlines()
    new_lines = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            new_lines.append('\n')
            continue
        indent = len(line) - len(stripped)
        # Normalize: round to nearest multiple of 4
        # BUT only if it looks like it was intended to be 4-space based.
        # If it's 3, it should be 4. If it's 7, it should be 8.
        # If it's 11, it should be 12.
        # We'll use (indent + 1) // 4 * 4 for small errors.
        if indent > 0:
            new_indent = ((indent + 1) // 4) * 4
            new_lines.append(' ' * new_indent + stripped)
        else:
            new_lines.append(line)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

fix_indent('bootstrap/codegen.py')
print("Normalized indentation to 4-space units")
