    def substitute_generics(self, node, mapping):
        # Recursive substitution of types in AST nodes
        if isinstance(node, list):
             for x in node: self.substitute_generics(x, mapping)
             return
             
        if isinstance(node, VarDecl):
             if node.type_name:
                 node.type_name = self.apply_submap(node.type_name, mapping)
             if node.initializer:
                 self.substitute_generics(node.initializer, mapping)
        
        elif isinstance(node, CallExpr):
             # Handle TurboFish or generic calls
             if '<' in node.callee:
                  # ID<T> -> ID<i32>
                  node.callee = self.apply_submap(node.callee, mapping)
             if node.callee.startswith('cast<') or node.callee.startswith('sizeof<') or node.callee.startswith('ptr_offset<'):
                  node.callee = self.apply_submap(node.callee, mapping)
             
             for arg in node.args:
                  self.substitute_generics(arg, mapping)
                  
        elif isinstance(node, FunctionDef):
             # Should not happen inside body usually, but nested func?
             pass
        
        elif isinstance(node, IfStmt):
             self.substitute_generics(node.condition, mapping)
             self.substitute_generics(node.then_branch, mapping)
             if node.else_branch: self.substitute_generics(node.else_branch, mapping)
             
        elif isinstance(node, WhileStmt):
             self.substitute_generics(node.condition, mapping)
             self.substitute_generics(node.body, mapping)
             
        elif isinstance(node, ReturnStmt):
             if node.value: self.substitute_generics(node.value, mapping)
             
        elif isinstance(node, Assignment):
             self.substitute_generics(node.target, mapping)
             self.substitute_generics(node.value, mapping)
             
        elif isinstance(node, BinaryExpr):
             self.substitute_generics(node.left, mapping)
             self.substitute_generics(node.right, mapping)
             
        elif isinstance(node, UnaryExpr):
             self.substitute_generics(node.operand, mapping)
             
        elif isinstance(node, MatchExpr):
             self.substitute_generics(node.value, mapping)
             for case in node.cases:
                 self.substitute_generics(case.body, mapping)
                 
        # ... other nodes ...
        
    def apply_submap(self, type_str, mapping):
        # Handle T -> i32
        # Handle *T -> *i32 or i32* (using nexalang syntax)
        # Handle Vec<T> -> Vec<i32>
        
        # Simple recursion/split
        if not type_str: return type_str
        
        # Pointer
        if type_str.endswith('*'):
             inner = type_str[:-1]
             return self.apply_submap(inner, mapping) + '*'
             
        # Generic: Name<Args>
        if '<' in type_str and type_str.endswith('>'):
             base = type_str[:type_str.find('<')]
             inside = type_str[type_str.find('<')+1:-1]
             args = [x.strip() for x in inside.split(',')]
             new_args = [self.apply_submap(a, mapping) for a in args]
             return f"{base}<{','.join(new_args)}>"
             
        # Base type
        if type_str in mapping:
             return mapping[type_str]
             
        return type_str
