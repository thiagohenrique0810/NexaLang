import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'bootstrap'))

from semantic import SemanticAnalyzer
from n_parser import ForStmt

s = SemanticAnalyzer()
print(f"Has visit_ForStmt: {hasattr(s, 'visit_ForStmt')}")

node = ForStmt('i', None, None, None)
print(f"Node Type: {type(node)}")
print(f"Node Type Name: {type(node).__name__}")

# Check generic visit
try:
    s.visit(node)
except Exception as e:
    print(f"Visit Error: {e}")
