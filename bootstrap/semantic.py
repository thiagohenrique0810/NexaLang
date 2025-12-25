class SemanticAnalyzer:
    def __init__(self):
        self.scopes = [{}] # List of dictionaries {name: {'type': type, 'moved': bool}}
        self.current_function = None

    def analyze(self, ast):
        for node in ast:
            self.visit(node)

    def visit(self, node):
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f"No visit_{type(node).__name__} method in SemanticAnalyzer")

    def enter_scope(self):
        self.scopes.append({})

    def exit_scope(self):
        self.scopes.pop()

    def declare(self, name, type_name):
        if name in self.scopes[-1]:
            raise Exception(f"Semantic Error: Variable '{name}' already declared in this scope.")
        self.scopes[-1][name] = {'type': type_name, 'moved': False}

    def lookup(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def visit_FunctionDef(self, node):
        self.current_function = node.name
        self.enter_scope()
        for stmt in node.body:
            self.visit(stmt)
        self.exit_scope()
        self.current_function = None

    def visit_VarDecl(self, node):
        # 1. Analyze initializer
        init_type = self.visit(node.initializer)
        
        # 2. Check type mismatch
        if init_type != node.type_name:
             raise Exception(f"Type Error: Cannot assign type '{init_type}' to variable '{node.name}' of type '{node.type_name}'")
        
        # 3. Declare variable
        self.declare(node.name, node.type_name)

    def visit_Assignment(self, node):
        # 1. Check if variable exists
        var_info = self.lookup(node.name)
        if not var_info:
            raise Exception(f"Semantic Error: Assignment to undefined variable '{node.name}'")
        
        if var_info['moved']:
             raise Exception(f"Ownership Error: Cannot assign to moved variable '{node.name}'")
            
        # 2. Analyze value
        val_type = self.visit(node.value)
        
        # 3. Check type mismatch
        if var_info['type'] != val_type:
            raise Exception(f"Type Error: Cannot assign type '{val_type}' to variable '{node.name}' of type '{var_info['type']}'")

    def visit_BinaryExpr(self, node):
        left_type = self.visit(node.left)
        right_type = self.visit(node.right)
        
        if left_type != right_type:
             raise Exception(f"Type Error: Operands must be same type. Got '{left_type}' and '{right_type}'")
             
        # For now, everything is i32
        if left_type != 'i32':
             raise Exception(f"Type Error: Arithmetic only supported for i32. Got '{left_type}'")
             
        return 'i32' # Result type

    def visit_IntegerLiteral(self, node):
        return 'i32'

    def visit_StringLiteral(self, node):
        return 'string'

    def visit_VariableExpr(self, node):
        var_info = self.lookup(node.name)
        if not var_info:
             raise Exception(f"Semantic Error: Undefined variable '{node.name}'")
             
        if var_info['moved']:
            raise Exception(f"Ownership Error: Use of moved variable '{node.name}'")
            
        return var_info['type']

    def visit_CallExpr(self, node):
        if node.callee == 'print':
            # print allows any type for now
            for arg in node.args:
                self.visit(arg)
            return 'void'
        else:
             raise Exception(f"Semantic Error: Unknown function '{node.callee}'")

    def visit_IfStmt(self, node):
        cond_type = self.visit(node.condition)
        # In C-like, i32 is valid bool, but let's be strict if we had bool type.
        # For now, we allow i32 as condition.
        
        self.enter_scope()
        for stmt in node.then_branch:
            self.visit(stmt)
        self.exit_scope()
        
        if node.else_branch:
            self.enter_scope()
            for stmt in node.else_branch:
                self.visit(stmt)
            self.exit_scope()

    def visit_WhileStmt(self, node):
        self.visit(node.condition)
        self.enter_scope()
        for stmt in node.body:
            self.visit(stmt)
        self.exit_scope()

    def visit_ReturnStmt(self, node):
        self.visit(node.value)
        # Should check against function return type
