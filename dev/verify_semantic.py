import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'bootstrap'))

import semantic
print(f"Verified semantic file: {semantic.__file__}")

from semantic import SemanticAnalyzer
from n_parser import ForStmt

s = SemanticAnalyzer()
print(f"Has visit_ForStmt: {hasattr(s, 'visit_ForStmt')}")

node = ForStmt('i', None, None, [], False)
print(f"Node Type: {type(node)}")

# Mock generic visit to print missing
s.generic_visit = lambda n: print(f"Missing visit_{type(n).__name__}")

# Basic visit check (will fail arguments checks inside visit_ForStmt but should dispatch)
try:
    s.visit(node)
except Exception as e:
    print(f"Visit Result (Exception expected): {e}")
