class ASTNode:
    pass

class Assignment(ASTNode):
    def __init__(self, name, value):
        self.name = name
        self.value = value

class FunctionDef(ASTNode):
    def __init__(self, name, body, is_kernel=False):
        self.name = name
        self.body = body
        self.is_kernel = is_kernel

class CallExpr(ASTNode):
    def __init__(self, callee, args):
        self.callee = callee
        self.args = args

class StringLiteral(ASTNode):
    def __init__(self, value):
        self.value = value

class VarDecl(ASTNode):
    def __init__(self, name, type_name, initializer):
        self.name = name
        self.type_name = type_name
        self.initializer = initializer

class ReturnStmt(ASTNode):
    def __init__(self, value):
        self.value = value

class IfStmt(ASTNode):
    def __init__(self, condition, then_branch, else_branch=None):
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch

class WhileStmt(ASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class BinaryExpr(ASTNode):
    def __init__(self, left, op, right):
        self.left = left
        self.right = right
        self.op = op
    # ... (rest of classes)

# ... inside Parser class ...

    def parse_statement(self):
        token = self.peek()
        if token.type == 'LET':
            return self.parse_var_decl()
        elif token.type == 'RETURN':
            return self.parse_return()
        elif token.type == 'IF':
            return self.parse_if()
        elif token.type == 'WHILE':
            return self.parse_while()
        elif token.type == 'IDENTIFIER':
            # Could be assignment or expression statement
            # For now, simplistic check
            return self.parse_expression_stmt()
        else:
            raise Exception(f"Unexpected token in statement: {token}")

    # ... (other parse methods) ...

    def parse_while(self):
        self.consume('WHILE')
        self.consume('LPAREN')
        cond = self.parse_expression()
        self.consume('RPAREN')
        
        self.consume('LBRACE')
        body = []
        while self.peek().type != 'RBRACE':
            body.append(self.parse_statement())
        self.consume('RBRACE')
        
        return WhileStmt(cond, body)

class IntegerLiteral(ASTNode):
    def __init__(self, value):
        self.value = value

class VariableExpr(ASTNode):
    def __init__(self, name):
        self.name = name

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset=0):
        if self.pos + offset < len(self.tokens):
            return self.tokens[self.pos + offset]
        return Token('EOF', '')

    def consume(self, type):
        if self.pos < len(self.tokens) and self.tokens[self.pos].type == type:
            self.pos += 1
            return self.tokens[self.pos - 1]
        raise Exception(f"Expected token type {type}, found {self.tokens[self.pos].type if self.pos < len(self.tokens) else 'EOF'}")

    def parse(self):
        functions = []
        while self.pos < len(self.tokens):
            if self.peek().type == 'KERNEL':
                functions.append(self.parse_function(is_kernel=True))
            elif self.peek().type == 'FN':
                functions.append(self.parse_function(is_kernel=False))
            else:
                raise Exception(f"Unexpected token at top level: {self.tokens[self.pos]}")
        return functions

    def parse_function(self, is_kernel=False):
        if is_kernel:
            self.consume('KERNEL')
        self.consume('FN')
        name = self.consume('IDENTIFIER').value
        self.consume('LPAREN')
        self.consume('RPAREN')
        self.consume('LBRACE')
        body = []
        while self.pos < len(self.tokens) and self.tokens[self.pos].type != 'RBRACE':
            body.append(self.parse_statement())
        self.consume('RBRACE')
        return FunctionDef(name, body, is_kernel=is_kernel)


    def parse_statement(self):
        token = self.peek()
        if token.type == 'LET':
            return self.parse_var_decl()
        elif token.type == 'RETURN':
            return self.parse_return()
        elif token.type == 'IF':
            return self.parse_if()
        elif token.type == 'WHILE':
            return self.parse_while()
        elif token.type == 'IDENTIFIER':
            # Check for assignment: ID = expr
            if self.peek(1).type == 'EQ':
                return self.parse_assignment()
            return self.parse_expression_stmt()
        else:
            raise Exception(f"Unexpected token in statement: {token}")

    def parse_assignment(self):
        name = self.consume('IDENTIFIER').value
        self.consume('EQ')
        value = self.parse_expression()
        return Assignment(name, value)

    def parse_var_decl(self):
        self.consume('LET')
        name = self.consume('IDENTIFIER').value
        self.consume('COLON')
        type_name = self.consume('IDENTIFIER').value # e.g. i32
        self.consume('EQ')
        initializer = self.parse_expression()
        return VarDecl(name, type_name, initializer)

    def parse_return(self):
        self.consume('RETURN')
        value = self.parse_expression()
        return ReturnStmt(value)

    def parse_if(self):
        self.consume('IF')
        self.consume('LPAREN')
        cond = self.parse_expression()
        self.consume('RPAREN')
        
        self.consume('LBRACE')
        then_branch = []
        while self.peek().type != 'RBRACE':
            then_branch.append(self.parse_statement())
        self.consume('RBRACE')
        
        else_branch = None
        if self.peek().type == 'ELSE':
            self.consume('ELSE')
            self.consume('LBRACE')
            else_branch = []
            while self.peek().type != 'RBRACE':
                else_branch.append(self.parse_statement())
            self.consume('RBRACE')
            
        return IfStmt(cond, then_branch, else_branch)

    def parse_while(self):
        self.consume('WHILE')
        self.consume('LPAREN')
        cond = self.parse_expression()
        self.consume('RPAREN')
        
        self.consume('LBRACE')
        body = []
        while self.peek().type != 'RBRACE':
            body.append(self.parse_statement())
        self.consume('RBRACE')
        
        return WhileStmt(cond, body)

    def parse_expression_stmt(self):
        expr = self.parse_expression()
        # Expect semicolon? For now optional/not implemented
        return expr

    def parse_expression(self):
        return self.parse_binary_expr(0)

    def parse_binary_expr(self, min_prec):
        left = self.parse_primary()
        
        while True:
            token = self.peek()
            prec = self.get_precedence(token.type)
            
            if prec < min_prec:
                break
                
            op = self.consume(token.type).type
            right = self.parse_binary_expr(prec + 1)
            left = BinaryExpr(left, op, right)
            
        return left

    def get_precedence(self, op_type):
        if op_type in ('PLUS', 'MINUS'): return 1
        if op_type in ('STAR', 'SLASH'): return 2
        if op_type == 'EQEQ': return 0
        return -1

    def parse_primary(self):
        token = self.peek()
        if token.type == 'NUMBER':
            self.consume('NUMBER')
            return IntegerLiteral(int(token.value))
        elif token.type == 'STRING':
            self.consume('STRING')
            return StringLiteral(token.value)
        elif token.type == 'IDENTIFIER':
            # Check for namespace: ID :: ID
            if self.peek(1).type == 'DOUBLE_COLON':
                 # Namespace call or variable
                 lhs = self.consume('IDENTIFIER').value
                 self.consume('DOUBLE_COLON')
                 rhs = self.consume('IDENTIFIER').value
                 full_name = f"{lhs}::{rhs}"
                 
                 if self.peek().type == 'LPAREN':
                     return self.parse_call_explicit(full_name)
                 else:
                     return VariableExpr(full_name)
                     
            if self.peek(1).type == 'LPAREN':
                return self.parse_call()
            else:
                self.consume('IDENTIFIER')
                return VariableExpr(token.value)
        elif token.type == 'LPAREN':
            self.consume('LPAREN')
            expr = self.parse_expression()
            self.consume('RPAREN')
            return expr
        else:
            raise Exception(f"Unexpected token in primary: {token}")

    def parse_call(self):
        name = self.consume('IDENTIFIER').value
        return self.parse_call_explicit(name)

    def parse_call_explicit(self, name):
        self.consume('LPAREN')
        args = []
        if self.peek().type != 'RPAREN':
            while True:
                args.append(self.parse_expression())
                if self.peek().type == 'RPAREN':
                    break
                # Comma parsing could go here
        self.consume('RPAREN')
        return CallExpr(name, args)
