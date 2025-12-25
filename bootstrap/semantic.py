from parser import FunctionDef, StructDef

class SemanticAnalyzer:
    def __init__(self):
        self.scopes = [{}] # List of dictionaries {name: {'type': type, 'moved': bool}}
        self.current_function = None

    def analyze(self, ast):
        # 1. Collect function and struct names
        self.functions = set(['print', 'gpu::global_id', 'panic', 'assert'])
        self.structs = {} # name -> {field: type}
        
        for node in ast:
            if isinstance(node, FunctionDef):
                self.functions.add(node.name)
            elif isinstance(node, StructDef):
                fields = {name: type_name for name, type_name in node.fields}
                self.structs[node.name] = fields

        # 2. Analyze bodies
        for node in ast:
            self.visit(node)

    def visit(self, node):
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def visit_StructDef(self, node):
        pass # Already collected

    def visit_MemberAccess(self, node):
        # 1. Evaluate object
        obj_type = self.visit(node.object)
        
        # 2. Check if object is a struct
        if obj_type not in self.structs:
             raise Exception(f"Type Error: access member '{node.member}' on non-struct type '{obj_type}'")
        
        # 3. Check if member exists
        fields = self.structs[obj_type]
        if node.member not in fields:
             raise Exception(f"Type Error: Struct '{obj_type}' has no member '{node.member}'")
        
        # Annotate node for CodeGen
        node.struct_type = obj_type
             
        return fields[node.member]

    def visit_ArrayLiteral(self, node):
        if not node.elements:
             raise Exception("Semantic Error: Empty array literals not supported (cannot infer type)")
        
        first_type = self.visit(node.elements[0])
        for el in node.elements[1:]:
            type_ = self.visit(el)
            if type_ != first_type:
                 raise Exception(f"Type Error: Array elements must be same type. Expected '{first_type}', got '{type_}'")
                 
        size = len(node.elements)
        return f"[{first_type}:{size}]"

    def visit_IndexAccess(self, node):
        # 1. Analyze object
        obj_type = self.visit(node.object)
        
        # 2. Verify it is an array
        if not (obj_type.startswith('[') and obj_type.endswith(']')):
             raise Exception(f"Type Error: Indexing non-array type '{obj_type}'")
        
        # Parse type string [T:N]
        content = obj_type[1:-1]
        elem_type, size_str = content.split(':')
        
        # 3. Analyze index
        index_type = self.visit(node.index)
        if index_type != 'i32':
             raise Exception(f"Type Error: Array index must be i32, got '{index_type}'")
             
        return elem_type

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
        
        # NOTE: Assigning to a moved variable is valid (re-initialization).
        # We just need to ensure the variable is considered 'alive' (not moved) after this.
            
        # 2. Analyze value
        val_type = self.visit(node.value)
        
        # 3. Check type mismatch
        if var_info['type'] != val_type:
            raise Exception(f"Type Error: Cannot assign type '{val_type}' to variable '{node.name}' of type '{var_info['type']}'")
            
        # Revive variable
        var_info['moved'] = False

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
        if node.callee in self.functions:
            if node.callee == 'print':
                for arg in node.args:
                    self.visit(arg)
                return 'void'
            elif node.callee == 'panic':
                if len(node.args) != 1:
                    raise Exception("Semantic Error: panic() expects 1 argument (message)")
                self.visit(node.args[0])
                return 'void'
            elif node.callee == 'assert':
                if len(node.args) != 2:
                    raise Exception("Semantic Error: assert() expects 2 arguments (condition, message)")
                cond_type = self.visit(node.args[0])
                # We don't have explicit bool type in semantic yet? visit_BinaryExpr returns 'i32'.
                # Let's assume i32 for now (0=false, !0=true)
                self.visit(node.args[1])
                return 'void'
            elif node.callee == 'gpu::global_id':
                return 'i32'
            else:
                 # User function call
                 for arg in node.args:
                     self.visit(arg)
                 return 'void'
        elif node.callee in self.structs:
            # Struct Instantiation
            fields = self.structs[node.callee]
            if len(node.args) != len(fields):
                raise Exception(f"Semantic Error: Struct '{node.callee}' expects {len(fields)} arguments, got {len(node.args)}")
            
            for arg in node.args:
                self.visit(arg)
            return node.callee
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
