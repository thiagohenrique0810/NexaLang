import re

def fix_indentation(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # We want to find common methods and ensure they use 4-space increments.
    # The file seems to have a mix. We'll specifically fix the ones we touched.
    
    new_lines = []
    in_method = None
    method_indent = 0
    
    for line in lines:
        # Check if line is a method definition
        match = re.match(r'^(\s*)def\s+(\w+)\(', line)
        if match:
            in_method = match.group(2)
            method_indent = len(match.group(1))
            new_lines.append(line)
            continue
            
        if in_method:
            stripped = line.lstrip()
            if not stripped: # Empty line
                new_lines.append('\n')
                continue
                
            current_indent = len(line) - len(stripped)
            if current_indent <= method_indent and stripped:
                # We exited the method
                in_method = None
                new_lines.append(line)
            else:
                # We are in the method. 
                # If it's one of the methods we care about (visit_MethodCall, etc.)
                # we'll normalize it to multiples of 4.
                if in_method in ['visit_MethodCall', 'visit_MemberAccess', 'visit_VarDecl']:
                    # Re-calculate indent. 
                    # If it was 13 (8+5), make it 12 (8+4).
                    # Actually, let's just use a simple heuristic: 
                    # If it's > 8, it's a nested block.
                    # Base body should be method_indent + 4.
                    # Nested should be method_indent + 8, etc.
                    
                    # For simplicity, let's just replace all leading whitespace 
                    # with a normalized version of itself.
                    # e.g. round(current_indent / 4) * 4
                    normalized_indent = round(current_indent / 4) * 4
                    new_lines.append(' ' * normalized_indent + stripped)
                else:
                    new_lines.append(line)
        else:
            new_lines.append(line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

fix_indentation('bootstrap/codegen.py')
print("Fixed indentation for visit_MethodCall, visit_MemberAccess, and visit_VarDecl")
