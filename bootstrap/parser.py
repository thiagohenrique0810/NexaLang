class ASTNode:
    pass

class Assignment(ASTNode):
    def __init__(self, target, value):
        self.target = target # Can be name(str) or ASTNode (LValue)
        self.value = value

class ArrayLiteral(ASTNode):
    def __init__(self, elements):
        self.elements = elements

class IndexAccess(ASTNode):
    def __init__(self, object, index):
        self.object = object
        self.index = index
    
    def __repr__(self):
        return f"{self.object}[{self.index}]"

class FunctionDef(ASTNode):
    def __init__(self, name, params, return_type, body, is_kernel=False, generics=None):
        self.name = name
        self.params = params
        self.return_type = return_type
        self.body = body
        self.is_kernel = is_kernel
        self.generics = generics or []

class StructDef(ASTNode):
    def __init__(self, name, fields, generics=None):
        self.name = name
        self.fields = fields # list of (name, type) tuples
        self.generics = generics or []

class EnumDef(ASTNode):
    def __init__(self, name, variants, generics=None):
        self.name = name
        self.variants = variants # list of (name, payload_types) tuples. payload_types is list of strings
        self.generics = generics or []

class ImplDef(ASTNode):
    def __init__(self, struct_name, methods):
        self.struct_name = struct_name
        self.methods = methods

class MatchExpr(ASTNode):
    def __init__(self, value, cases):
        self.value = value
        self.cases = cases # list of CaseArm

class CaseArm(ASTNode):
    def __init__(self, variant_name, var_name, body):
        self.variant_name = variant_name # "Ok"
        self.var_name = var_name         # "val" (bound variable) or None
        self.body = body

class MemberAccess(ASTNode):
    def __init__(self, object, member):
        self.object = object
        self.member = member
    
    def __repr__(self):
        return f"{self.object}.{self.member}"

class CallExpr(ASTNode):
    def __init__(self, callee, args):
        self.callee = callee
        self.args = args

class StringLiteral(ASTNode):
    def __init__(self, value):
        self.value = value

class CharLiteral(ASTNode):
    def __init__(self, value: int):
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

class UnaryExpr(ASTNode):
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand
    # ... (rest of classes)

# ... inside Parser class ...

    def parse_statement(self):
        token = self.peek()
        if token.type == 'LBRACE':
            self.consume('LBRACE')
            body = []
            while self.peek().type != 'RBRACE':
                body.append(self.parse_statement())
                if self.peek().type == 'SEMICOLON':
                    self.consume('SEMICOLON')
            self.consume('RBRACE')
            # Return a Block? Or list of Stmts?
            # Since parse_statement returns single ASTNode, we need a Block node.
            # But for bootstrap simplicity, if caller expects list, this is tricky.
            # However, Block IS a statement usually.
            # Let's define BlockStmt.
            return BlockStmt(body)
        elif token.type == 'LET':
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
            if self.peek().type == 'SEMICOLON':
                 self.consume('SEMICOLON')
        self.consume('RBRACE')
        
        return WhileStmt(cond, body)

class IntegerLiteral(ASTNode):
    def __init__(self, value):
        self.value = value

class BooleanLiteral(ASTNode):
    def __init__(self, value):
        self.value = value

class FloatLiteral(ASTNode):
    def __init__(self, value):
        self.value = value

class VariableExpr(ASTNode):
    def __init__(self, name):
        self.name = name

class BlockStmt(ASTNode):
    def __init__(self, stmts):
        self.stmts = stmts

class RegionStmt(ASTNode):
    def __init__(self, name, body):
        self.name = name
        self.body = body

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
        nodes = []
        while self.pos < len(self.tokens):
            if self.peek().type == 'KERNEL':
                nodes.append(self.parse_function(is_kernel=True))
            elif self.peek().type == 'FN':
                nodes.append(self.parse_function(is_kernel=False))
            elif self.peek().type == 'STRUCT':
                nodes.append(self.parse_struct())
            elif self.peek().type == 'ENUM':
                nodes.append(self.parse_enum())
            elif self.peek().type == 'IMPL':
                nodes.append(self.parse_impl())
            else:
                raise Exception(f"Unexpected token at top level: {self.tokens[self.pos]}")
        return nodes

    def parse_impl(self):
        self.consume('IMPL')
        struct_name = self.consume('IDENTIFIER').value
        self.consume('LBRACE')
        methods = []
        while self.peek().type != 'RBRACE':
            methods.append(self.parse_function(is_kernel=False))
        self.consume('RBRACE')
        return ImplDef(struct_name, methods)

    def parse_struct(self):
        self.consume('STRUCT')
        name = self.consume('IDENTIFIER').value
        
        generics = []
        if self.peek().type == 'LT':
            self.consume('LT')
            while self.peek().type != 'GT':
                generics.append(self.consume('IDENTIFIER').value)
                if self.peek().type == 'COMMA':
                    self.consume('COMMA')
            self.consume('GT')
            
        self.consume('LBRACE')
        fields = []
        while self.peek().type != 'RBRACE':
            field_name = self.consume('IDENTIFIER').value
            self.consume('COLON')
            field_type = self.parse_type() # Use parse_type instead of raw ID to support Generic Members
            fields.append((field_name, field_type))
            if self.peek().type == 'COMMA':
                self.consume('COMMA')
        self.consume('RBRACE')
        return StructDef(name, fields, generics)

    def parse_enum(self):
        self.consume('ENUM')
        name = self.consume('IDENTIFIER').value
        
        generics = []
        if self.peek().type == 'LT':
            self.consume('LT')
            while self.peek().type != 'GT':
                generics.append(self.consume('IDENTIFIER').value)
                if self.peek().type == 'COMMA':
                    self.consume('COMMA')
            self.consume('GT')
            
        self.consume('LBRACE')
        variants = []
        while self.peek().type != 'RBRACE':
            variant_name = self.consume('IDENTIFIER').value
            payloads = []
            if self.peek().type == 'LPAREN':
                self.consume('LPAREN')
                while self.peek().type != 'RPAREN':
                    payloads.append(self.consume('IDENTIFIER').value) # Type name
                    if self.peek().type == 'COMMA':
                        self.consume('COMMA')
                self.consume('RPAREN')
            variants.append((variant_name, payloads))
            if self.peek().type == 'COMMA':
                self.consume('COMMA')
        self.consume('RBRACE')
        return EnumDef(name, variants, generics)

    def parse_function(self, is_kernel=False):
        if is_kernel:
            self.consume('KERNEL')
        self.consume('FN')
        name = self.consume('IDENTIFIER').value
        
        generics = []
        if self.peek().type == 'LT':
            self.consume('LT')
            while self.peek().type != 'GT':
                generics.append(self.consume('IDENTIFIER').value)
                if self.peek().type == 'COMMA':
                   self.consume('COMMA')
            self.consume('GT')
            
        self.consume('LPAREN')
        params = []
        while self.peek().type != 'RPAREN':
            # Handle self
            if self.peek().type == 'SELF':
                 self.consume('SELF')
                 params.append(('self', 'Self'))
            elif self.peek().type == 'AMPERSAND' and self.peek(1).type == 'SELF':
                 self.consume('AMPERSAND')
                 self.consume('SELF')
                 params.append(('self', '&Self'))
            else:
                 param_name = self.consume('IDENTIFIER').value
                 self.consume('COLON')
                 param_type = self.parse_type()
                 params.append((param_name, param_type))
            
            if self.peek().type == 'COMMA':
                 self.consume('COMMA')
        self.consume('RPAREN')
        
        return_type = 'void'
        if self.peek().type == 'THIN_ARROW':
            self.consume('THIN_ARROW')
            return_type = self.parse_type()
            
        self.consume('LBRACE')
        body = []
        while self.peek().type != 'RBRACE':
            body.append(self.parse_statement())
            if self.peek().type == 'SEMICOLON':
                self.consume('SEMICOLON')
        self.consume('RBRACE')
        return FunctionDef(name, params, return_type, body, is_kernel, generics)


    def parse_statement(self):
        token = self.peek()
        print(f"DEBUG: parse_statement peek: {token}. type='{token.type}' == LET? {token.type == 'LET'}")
        if token.type == 'LET':
            # print(f"DEBUG: LET check passed")
            return self.parse_var_decl()
        print(f"DEBUG: Checking others. type='{token.type}'")
        elif token.type == 'RETURN':
            return self.parse_return()
        elif token.type == 'IF':
            return self.parse_if()
        elif token.type == 'WHILE':
            return self.parse_while()
        elif token.type == 'MATCH':
            return self.parse_match()
        elif token.type == 'REGION':
            return self.parse_region()
        elif token.type == 'LBRACE':
            self.consume('LBRACE')
            stmts = []
            while self.peek().type != 'RBRACE':
                stmts.append(self.parse_statement())
                if self.peek().type == 'SEMICOLON':
                     self.consume('SEMICOLON')
            self.consume('RBRACE')
            return BlockStmt(stmts)
        else:
            # General Expression or Assignment
            # Try parsing as expression
            # If followed by =, it's an assignment.
            # Otherwise it's an expression statement.
            # Expression statement
            # expr = self.parse_expression() # Line 330 (approx)
            expr = self.parse_expression()
            
            if self.peek().type == 'EQ':
                 self.consume('EQ')
                 value = self.parse_expression()
                 return Assignment(expr, value)
                 
            return expr

    def parse_var_decl(self):
        self.consume('LET')
        name = self.consume('IDENTIFIER').value
        
        type_name = None
        if self.peek().type == 'COLON':
            self.consume('COLON')
            type_name = self.parse_type()
            
        self.consume('EQ')
        initializer = self.parse_expression()
        return VarDecl(name, type_name, initializer)

    def parse_type(self):
        if self.peek().type == 'IDENTIFIER':
            name = self.consume('IDENTIFIER').value
            
            # Check for Generic Arguments <T, U>
            if self.peek().type == 'LT':
                self.consume('LT')
                args = []
                while self.peek().type != 'GT':
                    args.append(self.parse_type())
                    if self.peek().type == 'COMMA':
                        self.consume('COMMA')
                self.consume('GT')
                name = f"{name}<{','.join(args)}>"
            
            # Support Postfix '*' (e.g. i32*)
            while self.peek().type == 'STAR':
                self.consume('STAR')
                name = f"{name}*"
            
            return name
        elif self.peek().type == 'STAR':
            self.consume('STAR')
            inner = self.parse_type()
            return f"{inner}*" # Use postfix * for internal string representation
        elif self.peek().type == 'LBRACKET':
            # Slice Type: []T   (lowered to Slice<T>)
            # Array Type: [T:N]
            self.consume('LBRACKET')
            if self.peek().type == 'RBRACKET':
                self.consume('RBRACKET')
                elem_type = self.parse_type()
                return f"Slice<{elem_type}>"
            elem_type = self.consume('IDENTIFIER').value
            self.consume('COLON') # We use semicolon usually [T;N] but Lexer has COLON?
            # Lexer doesn't have SEMICOLON token yet! 
            # NexaLang syntax used [i32; 3]? Or [i32: 3]?
            # Let's use COLON for now as it exists, or add SEMICOLON.
            # Using SEMICOLON is standard for Rust-like.
            # I will use COLON since SEMICOLON token is missing, or I add SEMICOLON token.
            # Let's check Lexer. It does NOT have SEMICOLON.
            # I'll use COLON for array type [i32: 3] or update Lexer.
            # Updating Lexer is better for correct syntax. But for speed, let's stick to what we have?
            # Or just parse SEMICOLON as a specific char if needed?
            # Let's assume [i32: 3] for this bootstrap to save a turn, or use already existing tokens.
            # Wait, `examples/arrays.nxl` in plan used `[i32; 3]`.
            # I should add SEMICOLON to Lexer.
            # For now, to proceed in this step, I'll use COLON [i32: 3] or fix it after.
            # Let's add SEMICOLON to Lexer quickly? No, I am in Parser edit.
            # I will use COLON for now: [i32: 3].
            
            # Wait, I can just match on character if token is unknown? No, lexer produces tokens.
            # If I encounter ';', Lexer throws exception!
            # So I MUST update Lexer if I want ';'.
            # I'll use COLON [i32: 3] for now.
            
            size = self.consume('NUMBER').value
            self.consume('RBRACKET')
            return f"[{elem_type}:{size}]"
        else:
             raise Exception(f"Expected type, found {self.peek()}")
             
    def parse_return(self):
        self.consume('RETURN')
        value = self.parse_expression()
        return ReturnStmt(value)
        
    def parse_match(self):
        # match expr { Variant(var) => stmt, ... }
        self.consume('MATCH')
        value = self.parse_expression()
        self.consume('LBRACE')
        cases = []
        while self.peek().type != 'RBRACE':
            variant_name = self.consume('IDENTIFIER').value
            var_name = None
            if self.peek().type == 'LPAREN':
                self.consume('LPAREN')
                var_name = self.consume('IDENTIFIER').value
                self.consume('RPAREN')
            
            self.consume('FAT_ARROW')
            # For now, body is a single statement or expression?
            # Let's say it's a statement for now to allow return/print.
            # If we want expression-based match, we need blocks.
            # Let's call parse_statement().
            # Wait, if we use braces it might be block?
            # Basic version: Single statement.
            body = self.parse_statement()
            cases.append(CaseArm(variant_name, var_name, body))
            
            # Optional comma?
            if self.peek().type == 'COMMA':
                self.consume('COMMA')
        self.consume('RBRACE')
        return MatchExpr(value, cases)

    def parse_if(self):
        self.consume('IF')
        self.consume('LPAREN')
        cond = self.parse_expression()
        self.consume('RPAREN')
        
        self.consume('LBRACE')
        then_branch = []
        while self.peek().type != 'RBRACE':
            then_branch.append(self.parse_statement())
            if self.peek().type == 'SEMICOLON':
                 self.consume('SEMICOLON')
        self.consume('RBRACE')
        
        else_branch = None
        if self.peek().type == 'ELSE':
            self.consume('ELSE')
            self.consume('LBRACE')
            else_branch = []
            while self.peek().type != 'RBRACE':
                else_branch.append(self.parse_statement())
                if self.peek().type == 'SEMICOLON':
                     self.consume('SEMICOLON')
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
            if self.peek().type == 'SEMICOLON':
                 self.consume('SEMICOLON')
        self.consume('RBRACE')
        
        return WhileStmt(cond, body)

    def parse_region(self):
        self.consume('REGION')
        name = self.consume('IDENTIFIER').value
        self.consume('LBRACE')
        body = []
        while self.peek().type != 'RBRACE':
            body.append(self.parse_statement())
            if self.peek().type == 'SEMICOLON':
                 self.consume('SEMICOLON')
        self.consume('RBRACE')
        return RegionStmt(name, body)

    def parse_expression_stmt(self):
        expr = self.parse_expression()
        # Expect semicolon? For now optional/not implemented
        return expr

    def parse_expression(self):
        return self.parse_binary_expr(0)

    def parse_unary(self):
        token = self.peek()
        if token.type == 'STAR':
            self.consume('STAR')
            operand = self.parse_unary()
            return UnaryExpr('*', operand)
        elif token.type == 'AMPERSAND':
            self.consume('AMPERSAND')
            if self.peek().type == 'MUT':
                self.consume('MUT')
                operand = self.parse_unary()
                return UnaryExpr('&mut', operand)
            else:
                operand = self.parse_unary()
                return UnaryExpr('&', operand)
        else:
            return self.parse_primary()

    def parse_binary_expr(self, min_prec):
        left = self.parse_unary()
        
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
        if op_type in ('EQEQ', 'NEQ', 'LT', 'GT', 'LTE', 'GTE'): return 0
        return -1

    def parse_primary(self):
        token = self.peek()
        expr = None
        
        if token.type == 'NUMBER':
            self.consume('NUMBER')
            expr = IntegerLiteral(int(token.value))
        elif token.type == 'CHAR':
            self.consume('CHAR')
            expr = CharLiteral(int(token.value))
        elif token.type == 'FLOAT':
            self.consume('FLOAT')
            expr = FloatLiteral(float(token.value))
        elif token.type == 'STRING':
            self.consume('STRING')
            expr = StringLiteral(token.value)
        elif token.type == 'TRUE':
            self.consume('TRUE')
            expr = BooleanLiteral(True)
        elif token.type == 'FALSE':
            self.consume('FALSE')
            expr = BooleanLiteral(False)
        elif token.type == 'IDENTIFIER':
            # Check for namespace or TurboFish: ID :: ID or ID :: <Args>
            if self.peek(1).type == 'DOUBLE_COLON':
                 lhs = self.consume('IDENTIFIER').value
                 self.consume('DOUBLE_COLON')
                 
                 full_name = lhs
                 if self.peek().type == 'LT':
                     # Turbo fish: ID :: <T, U>
                     self.consume('LT')
                     args = []
                     while self.peek().type != 'GT':
                         args.append(self.parse_type())
                         if self.peek().type == 'COMMA': self.consume('COMMA')
                     self.consume('GT')
                     full_name = f"{lhs}<{','.join(args)}>"
                     
                     # Check for Variant chaining: :: Variant
                     if self.peek().type == 'DOUBLE_COLON':
                         self.consume('DOUBLE_COLON')
                         rhs = self.consume('IDENTIFIER').value
                         full_name = f"{full_name}::{rhs}"
                         
                     # Check for Call
                     if self.peek().type == 'LPAREN':
                         expr = self.parse_call_explicit(full_name)
                     else:
                         expr = VariableExpr(full_name)
                 else:
                     # Namespace: ID :: ID
                     rhs = self.consume('IDENTIFIER').value
                     # Check if it's a function call (Enum variant constructor)
                     if self.peek().type == 'LPAREN':
                         # We pass the full name "Enum::Variant" as callee
                         full_name = f"{lhs}::{rhs}"
                         expr = self.parse_call_explicit(full_name)
                     else:
                         # Enum variant without args (if supported) or variable
                         full_name = f"{lhs}::{rhs}"
                         expr = VariableExpr(full_name)
            
            elif self.peek(1).type == 'LPAREN':
                 expr = self.parse_call()
            else:
                 expr = VariableExpr(token.value)
                 self.consume('IDENTIFIER')

        elif token.type == 'LPAREN':
            self.consume('LPAREN')
            expr = self.parse_expression()
            self.consume('RPAREN')
            
        elif token.type == 'LBRACKET':
            # Array Literal: [1, 2, 3]
            self.consume('LBRACKET')
            elements = []
            if self.peek().type != 'RBRACKET':
                while True:
                    elements.append(self.parse_expression())
                    if self.peek().type == 'RBRACKET':
                        break
                    self.consume('COMMA')
            self.consume('RBRACKET')
            expr = ArrayLiteral(elements)
        else:
            # Print context
            start = max(0, self.pos - 10)
            end = min(len(self.tokens), self.pos + 10)
            context = self.tokens[start:end]
            raise Exception(f"Unexpected token in primary: {token}. Type: '{token.type}'. Context: {context}")
            
        # Postfix Handlers (Member Access, Index Access, Call)
        while True:
            if self.peek().type == 'DOT':
                self.consume('DOT')
                member = self.consume('IDENTIFIER').value
                expr = MemberAccess(expr, member)
            elif self.peek().type == 'LBRACKET':
                self.consume('LBRACKET')
                index = self.parse_expression()
                self.consume('RBRACKET')
                expr = IndexAccess(expr, index)
            elif self.peek().type == 'LPAREN':
                args = self.parse_call_arguments()
                expr = CallExpr(expr, args)
            else:
                break
                
        return expr

    def parse_call(self):
        name = self.consume('IDENTIFIER').value
        return CallExpr(name, self.parse_call_arguments())

    def parse_call_explicit(self, name):
        # Used by DoubleColon path
        return CallExpr(name, self.parse_call_arguments())

    def parse_call_arguments(self):
        self.consume('LPAREN')
        args = []
        if self.peek().type != 'RPAREN':
            while True:
                args.append(self.parse_expression())
                if self.peek().type == 'RPAREN':
                    break
                self.consume('COMMA')
        self.consume('RPAREN')
        return args
