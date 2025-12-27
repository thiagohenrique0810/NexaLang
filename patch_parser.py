
import os

path = 'bootstrap/parser.py'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Target block from parse_while (indentation 8 spaces)
target = """        while self.peek().type != 'RBRACE':
            body.append(self.parse_statement())
        self.consume('RBRACE')"""

replacement = """        while self.peek().type != 'RBRACE':
            body.append(self.parse_statement())
            if self.peek().type == 'SEMICOLON':
                 self.consume('SEMICOLON')
        self.consume('RBRACE')"""

if target in content:
    new_content = content.replace(target, replacement)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Successfully patched parse_while")
else:
    print("Target not found in content!")
    # Debug: print surrounding lines
    start_idx = content.find("def parse_while")
    if start_idx != -1:
        print("Found parse_while at:", start_idx)
        print("Content snippet:")
        print(content[start_idx:start_idx+300])
    
