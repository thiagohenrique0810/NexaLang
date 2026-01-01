class ASTNode:
    def __init__(self):
        self.line = 0
        self.column = 0

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
    def __init__(self, name, params, return_type, body, is_kernel=False, generics=None, is_pub=False):
        self.name = name
        self.params = params
        self.return_type = return_type
        self.body = body
        self.is_kernel = is_kernel
        self.generics = generics or []
        self.is_pub = is_pub
        self.module = ""
        self.used = False

class StructDef(ASTNode):
    def __init__(self, name, fields, generics=None, is_pub=False):
        self.name = name
        self.fields = fields # list of (name, type) tuples
        self.generics = generics or []
        self.is_pub = is_pub
        self.module = ""

class EnumDef(ASTNode):
    def __init__(self, name, variants, generics=None, is_pub=False):
        self.name = name
        self.variants = variants # list of (name, payload_types) tuples. payload_types is list of strings
        self.generics = generics or []
        self.is_pub = is_pub
        self.module = ""

class TraitDef(ASTNode):
    def __init__(self, name, methods, generics=None, is_pub=False, associated_types=None):
        self.name = name
        self.methods = methods # list of FunctionDef (likely without body)
        self.generics = generics or []
        self.is_pub = is_pub
        self.associated_types = associated_types or [] # list of names
        self.module = ""

class ImplDef(ASTNode):
    def __init__(self, struct_name, methods, generics=None, trait_name=None, associated_types=None):
        self.struct_name = struct_name
        self.methods = methods
        self.generics = generics or []
        self.trait_name = trait_name
        self.associated_types = associated_types or {} # dict: name -> type

class MatchExpr(ASTNode):
    def __init__(self, value, cases):
        self.value = value
        self.cases = cases # list of CaseArm

class CaseArm(ASTNode):
    def __init__(self, variant_name, var_names, body):
        self.variant_name = variant_name # "Ok"
        self.var_names = var_names       # list of bound variables
        self.body = body

class MemberAccess(ASTNode):
    def __init__(self, object, member):
        self.object = object
        self.member = member
    
    def __repr__(self):
        return f"{self.object}.{self.member}"

class MethodCall(ASTNode):
    def __init__(self, receiver, method_name, args):
        self.receiver = receiver
        self.method_name = method_name
        self.args = args
    
    def __repr__(self):
        return f"{self.receiver}.{self.method_name}(...)"


class CallExpr(ASTNode):
    def __init__(self, callee, args):
        self.callee = callee
        self.args = args

class LambdaExpr(ASTNode):
    def __init__(self, params, return_type, body):
        self.params = params # [(name, type), ...]
        self.return_type = return_type
        self.body = body # list of statements
        self.captures = {} # {name: type} - filled during semantic analysis

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

class ForStmt(ASTNode):
    def __init__(self, var_name, start_expr, end_expr, body, inclusive=False, is_iterator=False):
        self.var_name = var_name
        self.start_expr = start_expr
        self.end_expr = end_expr
        self.body = body
        self.inclusive = inclusive
        self.is_iterator = is_iterator

class BinaryExpr(ASTNode):
    def __init__(self, left, op, right):
        self.left = left
        self.right = right
        self.op = op

class UnaryExpr(ASTNode):
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

class UseStmt(ASTNode):
    def __init__(self, path, is_pub=False, is_glob=False):
        self.path = path # list of strings
        self.is_pub = is_pub
        self.is_glob = is_glob
        
    def __repr__(self):
        return f"use {'::'.join(self.path)}"
    # ... (rest of classes)

# ... inside Parser class ...

    def parse_statement(self):
        token = self.peek()
        # No debug print
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
        elif token.type == 'MATCH':
            return self.parse_match()
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

class BreakStmt(ASTNode):
    pass

class ContinueStmt(ASTNode):
    pass

class ModDecl(ASTNode):
    def __init__(self, name, body=None, is_pub=False):
        self.name = name
        self.body = body # list of statements if it's a block mod
        self.is_pub = is_pub

class TypeAlias(ASTNode):
    def __init__(self, alias, original_type, is_pub=False):
        self.alias = alias
        self.original_type = original_type
        self.is_pub = is_pub

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
            # print(f"DEBUG CONSUME: {type}")
            self.pos += 1
            return self.tokens[self.pos - 1]
        
        token = self.tokens[self.pos] if self.pos < len(self.tokens) else "EOF"
        prev = self.tokens[self.pos-1] if self.pos > 0 else "START"
        nxt = self.tokens[self.pos+1] if self.pos+1 < len(self.tokens) else "EOF"
        raise Exception(f"Expected token type {type}, found {token} (Prev: {prev}, Next: {nxt})")



# ...

    def parse(self):
        nodes = []
        while self.pos < len(self.tokens):
            is_pub = False
            if self.peek().type == 'PUB':
                self.consume('PUB')
                is_pub = True

            if self.peek().type == 'KERNEL':
                nodes.append(self.parse_function(is_kernel=True, is_pub=is_pub))
            elif self.peek().type == 'FN':
                nodes.append(self.parse_function(is_kernel=False, is_pub=is_pub))
            elif self.peek().type == 'STRUCT':
                nodes.append(self.parse_struct(is_pub=is_pub))
            elif self.peek().type == 'ENUM':
                nodes.append(self.parse_enum(is_pub=is_pub))
            elif self.peek().type == 'IMPL':
                if is_pub: raise Exception("impl blocks cannot be declared public")
                nodes.append(self.parse_impl())
            elif self.peek().type == 'MOD':
                nodes.append(self.parse_mod(is_pub=is_pub))
            elif self.peek().type == 'TRAIT':
                nodes.append(self.parse_trait(is_pub=is_pub))
            elif self.peek().type == 'USE':
                nodes.append(self.parse_use(is_pub=is_pub))
            elif self.peek().type == 'TYPE':
                nodes.append(self.parse_type_alias(is_pub=is_pub))
            else:
                raise Exception(f"Unexpected token at top level: {self.tokens[self.pos]}")
        return nodes

    def parse_mod(self, is_pub=False):
        self.consume('MOD')
        name = self.consume('IDENTIFIER').value
        
        if self.peek().type == 'LBRACE':
            self.consume('LBRACE')
            body = []
            while self.peek().type != 'RBRACE':
                # Similar to parse() but within a block
                # We need to support top-level items within mod blocks
                is_nested_pub = False
                if self.peek().type == 'PUB':
                    self.consume('PUB')
                    is_nested_pub = True
                
                t = self.peek().type
                if t == 'FN': body.append(self.parse_function(is_pub=is_nested_pub))
                elif t == 'STRUCT': body.append(self.parse_struct(is_pub=is_nested_pub))
                elif t == 'ENUM': body.append(self.parse_enum(is_pub=is_nested_pub))
                elif t == 'IMPL': body.append(self.parse_impl())
                elif t == 'MOD': body.append(self.parse_mod(is_pub=is_nested_pub))
                elif t == 'TRAIT': body.append(self.parse_trait(is_pub=is_nested_pub))
                elif t == 'USE': body.append(self.parse_use(is_pub=is_nested_pub))
                elif t == 'TYPE': body.append(self.parse_type_alias(is_pub=is_nested_pub))
                else: raise Exception(f"Unexpected token in nested mod: {self.peek()}")
                
                # Semicolons are optional/required depending on the item, 
                # but parse_* methods usually consume what they need.
                # However, parse_use does NOT consume semicolon in some versions?
                # Actually most parse_* in this codebase handle their own terminators.
            self.consume('RBRACE')
            return ModDecl(name, body=body, is_pub=is_pub)
        else:
            self.consume('SEMICOLON')
            return ModDecl(name, body=None, is_pub=is_pub)

    def parse_use(self, is_pub=False):
        self.consume('USE')
        path = []
        is_glob = False
        
        if self.peek().type == 'STAR':
            self.consume('STAR')
            is_glob = True
        else:
            path.append(self.consume('IDENTIFIER').value)
            while self.peek().type == 'DOUBLE_COLON':
                self.consume('DOUBLE_COLON')
                if self.peek().type == 'STAR':
                    self.consume('STAR')
                    is_glob = True
                    break
                path.append(self.consume('IDENTIFIER').value)
        
        self.consume('SEMICOLON')
        return UseStmt(path, is_pub=is_pub, is_glob=is_glob)

    def parse_type_alias(self, is_pub=False):
        self.consume('TYPE')
        alias = self.consume('IDENTIFIER').value
        self.consume('EQ')
        original_type = self.parse_type()
        if self.peek().type == 'SEMICOLON':
            self.consume('SEMICOLON')
        return TypeAlias(alias, original_type, is_pub=is_pub)



    def parse_impl(self):
        token = self.consume('IMPL')
        
        generics = []
        if self.peek().type == 'LT':
            self.consume('LT')
            while self.peek().type != 'GT':
                g_name = self.consume('IDENTIFIER').value
                bound = None
                if self.peek().type == 'COLON':
                     self.consume('COLON')
                     bound = self.consume('IDENTIFIER').value
                generics.append((g_name, bound))
                if self.peek().type == 'COMMA': self.consume('COMMA')
            self.consume('GT')

        # Check for 'impl Trait for Type'
        # We read first identifier. It could be Trait or Type.
        first_id = self.consume('IDENTIFIER').value
        
        trait_name = None
        struct_name = None
        
        if self.peek().type == 'FOR':
             self.consume('FOR')
             # first_id was Trait
             trait_name = first_id
             # Next is Type (struct)
             struct_name = self.consume('IDENTIFIER').value
        else:
             # Regular impl Type
             struct_name = first_id
        
        # Optional struct generics: Struct<T>
        if self.peek().type == 'LT':
            self.consume('LT')
            while self.peek().type != 'GT':
                self.consume('IDENTIFIER').value
                if self.peek().type == 'COMMA': self.consume('COMMA')
            self.consume('GT')

        self.consume('LBRACE')
        methods = []
        associated_types = {}
        while self.peek().type != 'RBRACE':
            is_pub = False
            token = self.peek()
            if token.type == 'TYPE': # type Item = i32;
                 self.consume('TYPE')
                 assoc_name = self.consume('IDENTIFIER').value
                 self.consume('EQ')
                 target_type = self.parse_type()
                 self.consume('SEMICOLON')
                 associated_types[assoc_name] = target_type
                 continue
            
            if self.peek().type == 'PUB':
                 self.consume('PUB')
                 is_pub = True
            methods.append(self.parse_function(is_kernel=False, is_pub=is_pub))
        self.consume('RBRACE')
        node = ImplDef(struct_name, methods, generics, trait_name=trait_name, associated_types=associated_types)
        node.line = token.line
        node.column = token.column
        return node

    def parse_trait(self, is_pub=False):
        self.consume('TRAIT')
        name = self.consume('IDENTIFIER').value
        
        generics = []
        if self.peek().type == 'LT':
            self.consume('LT')
            while self.peek().type != 'GT':
                g_name = self.consume('IDENTIFIER').value
                bound = None
                if self.peek().type == 'COLON':
                     self.consume('COLON')
                     bound = self.consume('IDENTIFIER').value
                generics.append((g_name, bound))
                if self.peek().type == 'COMMA': self.consume('COMMA')
            self.consume('GT')
            
        self.consume('LBRACE')
        methods = []
        associated_types = []
        while self.peek().type != 'RBRACE':
             if self.peek().type == 'TYPE': # type Item;
                 self.consume('TYPE')
                 assoc_name = self.consume('IDENTIFIER').value
                 self.consume('SEMICOLON')
                 associated_types.append(assoc_name)
                 continue
                 
             # Trait methods might not have bodies.
             # Call parse_function with allow_empty_body=True
             # But parse_function signature update needed.
             methods.append(self.parse_function(allow_empty_body=True))
        self.consume('RBRACE')
        return TraitDef(name, methods, generics, is_pub=is_pub, associated_types=associated_types)

    def parse_struct(self, is_pub=False):
        self.consume('STRUCT')
        name = self.consume('IDENTIFIER').value
        
        generics = []
        if self.peek().type == 'LT':
            self.consume('LT')
            while self.peek().type != 'GT':
                is_const = False
                if self.peek().type == 'CONST':
                     self.consume('CONST')
                     is_const = True
                
                g_name = self.consume('IDENTIFIER').value
                bound = None
                if self.peek().type == 'COLON':
                     self.consume('COLON')
                     bound = self.consume('IDENTIFIER').value
                generics.append((g_name, bound, is_const))
                if self.peek().type == 'COMMA':
                    self.consume('COMMA')
            self.consume('GT')
            
        self.consume('LBRACE')
        fields = []
        while self.peek().type != 'RBRACE':
            if self.peek().type == 'PUB':
                 self.consume('PUB')
            field_name = self.consume('IDENTIFIER').value
            self.consume('COLON')
            field_type = self.parse_type()
            fields.append((field_name, field_type))
            if self.peek().type == 'COMMA':
                self.consume('COMMA')
        self.consume('RBRACE')
        start_token = self.tokens[self.pos-1] # AFTER consume
        node = StructDef(name, fields, generics, is_pub=is_pub)
        node.line = start_token.line
        node.column = start_token.column
        return node

    def parse_enum(self, is_pub=False):
        self.consume('ENUM')
        name = self.consume('IDENTIFIER').value
        
        generics = []
        if self.peek().type == 'LT':
            self.consume('LT')
            while self.peek().type != 'GT':
                is_const = False
                if self.peek().type == 'CONST':
                     self.consume('CONST')
                     is_const = True

                g_name = self.consume('IDENTIFIER').value
                bound = None
                if self.peek().type == 'COLON':
                     self.consume('COLON')
                     bound = self.consume('IDENTIFIER').value
                generics.append((g_name, bound, is_const))
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
                    payloads.append(self.parse_type()) # Type name
                    if self.peek().type == 'COMMA':
                        self.consume('COMMA')
                self.consume('RPAREN')
            variants.append((variant_name, payloads))
            if self.peek().type == 'COMMA':
                self.consume('COMMA')
        self.consume('RBRACE')
        return EnumDef(name, variants, generics, is_pub=is_pub)

    def parse_function(self, is_kernel=False, is_pub=False, allow_empty_body=False):
        start_token = self.peek()
        if is_kernel:
            self.consume('KERNEL')
        self.consume('FN')
        name = self.consume('IDENTIFIER').value
        
        generics = []
        if self.peek().type == 'LT':
            self.consume('LT')
            while self.peek().type != 'GT':
                is_const = False
                if self.peek().type == 'CONST':
                     self.consume('CONST')
                     is_const = True
                
                g_name = self.consume('IDENTIFIER').value
                bound = None
                if self.peek().type == 'COLON':
                     self.consume('COLON')
                     bound = self.consume('IDENTIFIER').value
                generics.append((g_name, bound, is_const))
                if self.peek().type == 'COMMA':
                   self.consume('COMMA')
            self.consume('GT')
            
        self.consume('LPAREN')
        params = []
        while self.peek().type != 'RPAREN':
            # Handle self parameters
            if self.peek().type == 'SELF':
                 self.consume('SELF')
                 params.append(('self', 'Self'))
            elif self.peek().type == 'AMPERSAND':
                 # Could be &self or &mut self
                 self.consume('AMPERSAND')
                 if self.peek().type == 'MUT':
                     self.consume('MUT')
                     self.consume('SELF')
                     params.append(('self', '&mut Self'))
                 elif self.peek().type == 'SELF':
                     self.consume('SELF')
                     params.append(('self', '&Self'))
                 else:
                     # Regular reference parameter &name: Type
                     param_name = self.consume('IDENTIFIER').value
                     self.consume('COLON')
                     param_type = '&' + self.parse_type()
                     params.append((param_name, param_type))
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
            
        if self.peek().type == 'SEMICOLON' and allow_empty_body:
            self.consume('SEMICOLON')
            body = None
        else:
            self.consume('LBRACE')
            body = []
            while self.peek().type != 'RBRACE':
                body.append(self.parse_statement())
                if self.peek().type == 'SEMICOLON':
                    self.consume('SEMICOLON')
            self.consume('RBRACE')
        node = FunctionDef(name, params, return_type, body, is_kernel, generics, is_pub=is_pub)
        node.line = start_token.line
        node.column = start_token.column
        return node



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
        elif token.type == 'FOR':
            return self.parse_for()
        elif token.type == 'MATCH':
            return self.parse_match()
        elif token.type == 'BREAK':
            self.consume('BREAK')
            if self.peek().type == 'SEMICOLON': self.consume('SEMICOLON')
            return BreakStmt()
        elif token.type == 'CONTINUE':
            self.consume('CONTINUE')
            if self.peek().type == 'SEMICOLON': self.consume('SEMICOLON')
            return ContinueStmt()
        elif token.type == 'REGION':
            return self.parse_region()
        elif token.type == 'USE':
            return self.parse_use()
        elif token.type == 'TYPE':
            return self.parse_type_alias()
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
        start_token = self.consume('LET')
        
        # Check for optional 'mut'
        is_mut = False
        if self.peek().type == 'MUT':
            self.consume('MUT')
            is_mut = True
        
        name = self.consume('IDENTIFIER').value
        
        type_name = None
        if self.peek().type == 'COLON':
            self.consume('COLON')
            type_name = self.parse_type()
            
        self.consume('EQ')
        initializer = self.parse_expression()
        
        # Create VarDecl with mutability flag
        var_decl = VarDecl(name, type_name, initializer)
        var_decl.is_mut = is_mut
        var_decl.line = start_token.line
        var_decl.column = start_token.column
        return var_decl

    def parse_type(self):
        if self.peek().type == 'IDENTIFIER':
            name = self.consume('IDENTIFIER').value
            
            # Support Namespaced Types (mod::Struct)
            while self.peek().type == 'DOUBLE_COLON':
                self.consume('DOUBLE_COLON')
                part = self.consume('IDENTIFIER').value
                name = f"{name}_{part}"   
            
            # Check for Generic Arguments <T, U>
            if self.peek().type == 'LT':
                self.consume('LT')
                args = []
                while self.peek().type != 'GT':
                    if self.peek().type == 'NUMBER':
                         args.append(str(self.consume('NUMBER').value))
                    else:
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
        elif self.peek().type == 'AMPERSAND':
            self.consume('AMPERSAND')
            is_mut = ""
            if self.peek().type == 'MUT':
                self.consume('MUT')
                is_mut = "mut "
            inner = self.parse_type()
            return f"&{is_mut}{inner}"
        elif self.peek().type == 'FN':
            self.consume('FN')
            self.consume('LPAREN')
            params = []
            while self.peek().type != 'RPAREN':
                params.append(self.parse_type())
                if self.peek().type == 'COMMA':
                    self.consume('COMMA')
            self.consume('RPAREN')
            
            ret_type = 'void'
            if self.peek().type == 'THIN_ARROW':
                self.consume('THIN_ARROW')
                ret_type = self.parse_type()
            
            return f"fn({','.join(params)})->{ret_type}"
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
        start_token = self.consume('RETURN')
        value = None
        if self.peek().type != 'SEMICOLON':
            value = self.parse_expression()
        node = ReturnStmt(value)
        node.line = start_token.line
        node.column = start_token.column
        return node
        
    def parse_match(self):
        # match expr { Variant(var) => stmt, ... }
        start_token = self.consume('MATCH')
        value = self.parse_expression()
        self.consume('LBRACE')
        cases = []
        while self.peek().type != 'RBRACE':
            variant_name = self.consume('IDENTIFIER').value
            var_names = []
            if self.peek().type == 'LPAREN':
                self.consume('LPAREN')
                while self.peek().type != 'RPAREN':
                    var_names.append(self.consume('IDENTIFIER').value)
                    if self.peek().type == 'COMMA':
                        self.consume('COMMA')
                self.consume('RPAREN')
            
            self.consume('FAT_ARROW')
            # For now, body is a single statement or expression?
            # Let's say it's a statement for now to allow return/print.
            # If we want expression-based match, we need blocks.
            # Let's call parse_statement().
            # Wait, if we use braces it might be block?
            # Basic version: Single statement.
            body = self.parse_statement()
            cases.append(CaseArm(variant_name, var_names, body))
            
            # Optional comma?
            if self.peek().type == 'COMMA':
                self.consume('COMMA')
        self.consume('RBRACE')
        node = MatchExpr(value, cases)
        node.line = start_token.line
        node.column = start_token.column
        return node

    def parse_if(self):
        start_token = self.consume('IF')
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
            
        node = IfStmt(cond, then_branch, else_branch)
        node.line = start_token.line
        node.column = start_token.column
        return node

    def parse_for(self):
        start_token = self.consume('FOR')
        var_name = self.consume('IDENTIFIER').value
        self.consume('IN')
        
        expr1 = self.parse_expression()
        
        is_range = False
        inclusive = False
        end_expr = None
        
        if self.peek().type == 'DOT_DOT':
            self.consume('DOT_DOT')
            is_range = True
            end_expr = self.parse_expression()
        elif self.peek().type == 'DOT_DOT_EQ':
            self.consume('DOT_DOT_EQ')
            is_range = True
            inclusive = True
            end_expr = self.parse_expression()
            
        self.consume('LBRACE')
        body = []
        while self.peek().type != 'RBRACE':
            body.append(self.parse_statement())
            if self.peek().type == 'SEMICOLON':
                 self.consume('SEMICOLON')
        self.consume('RBRACE')
        
        node = ForStmt(var_name, expr1, end_expr, body, inclusive, is_iterator=not is_range)
        node.line = start_token.line
        node.column = start_token.column
        return node

    def parse_while(self):
        start_token = self.consume('WHILE')
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
        
        node = WhileStmt(cond, body)
        node.line = start_token.line
        node.column = start_token.column
        return node

    def parse_region(self):
        start_token = self.consume('REGION')
        name = self.consume('IDENTIFIER').value
        self.consume('LBRACE')
        body = []
        while self.peek().type != 'RBRACE':
            body.append(self.parse_statement())
            if self.peek().type == 'SEMICOLON':
                 self.consume('SEMICOLON')
        self.consume('RBRACE')
        node = RegionStmt(name, body)
        node.line = start_token.line
        node.column = start_token.column
        return node

    def parse_expression_stmt(self):
        expr = self.parse_expression()
        if self.peek().type == 'EQ':
            self.consume('EQ')
            rhs = self.parse_expression()
            if self.peek().type == 'SEMICOLON':
                self.consume('SEMICOLON')
            return Assignment(expr, rhs)
        
        if self.peek().type == 'SEMICOLON':
            self.consume('SEMICOLON')
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
        if op_type in ('STAR', 'SLASH', 'PERCENT'): return 2
        if op_type in ('EQEQ', 'NEQ', 'LT', 'GT', 'LTE', 'GTE', 'AND', 'OR'): return 0
        return -1

    def parse_primary(self):
        start_token = self.peek()
        token = start_token
        # No debug print
        expr = None
        
        if start_token.type == 'SELF':
            self.consume('SELF')
            expr = VariableExpr('self')
        elif token.type == 'NUMBER':
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
            peek1 = self.peek(1).type
            # 1. Struct Instantiation: Name { ... }
            if peek1 == 'LBRACE':
                name = self.consume('IDENTIFIER').value
                self.consume('LBRACE')
                args = []
                while self.peek().type != 'RBRACE':
                    self.consume('IDENTIFIER') # field name
                    self.consume('COLON')
                    args.append(self.parse_expression())
                    if self.peek().type == 'COMMA': self.consume('COMMA')
                self.consume('RBRACE')
                expr = CallExpr(name, args)
            
            # 2. Generic Instantiation or Templated Variable (cast<T>)
            elif peek1 == 'LT' and token.value in ('cast', 'sizeof', 'ptr_offset', 'slice_from_array', 'ptr_to_int', 'int_to_ptr'):
                name = self.consume('IDENTIFIER').value
                self.consume('LT')
                types = []
                while self.peek().type != 'GT':
                    if self.peek().type == 'NUMBER':
                         types.append(str(self.consume('NUMBER').value))
                    else:
                         types.append(self.parse_type())
                    if self.peek().type == 'COMMA': self.consume('COMMA')
                self.consume('GT')
                full_name = f"{name}<{','.join(types)}>"
                
                if self.peek().type == 'LBRACE':
                    # Struct Instantiation: Name<T> { ... }
                    self.consume('LBRACE')
                    args = []
                    while self.peek().type != 'RBRACE':
                        self.consume('IDENTIFIER')
                        self.consume('COLON')
                        args.append(self.parse_expression())
                        if self.peek().type == 'COMMA': self.consume('COMMA')
                    self.consume('RBRACE')
                    expr = CallExpr(full_name, args)
                else:
                    expr = VariableExpr(full_name)
            
            # 3. Namespaces or TurboFish: ID :: ...
            elif peek1 == 'DOUBLE_COLON':
                full_name = self.consume('IDENTIFIER').value
                while self.peek().type == 'DOUBLE_COLON':
                    self.consume('DOUBLE_COLON')
                    if self.peek().type == 'LT':
                        # Turbo fish: ... :: <T, U>
                        self.consume('LT')
                        types = []
                        while self.peek().type != 'GT':
                            types.append(self.parse_type())
                            if self.peek().type == 'COMMA': self.consume('COMMA')
                        self.consume('GT')
                        full_name = f"{full_name}<{','.join(types)}>"
                    else:
                        rhs = self.consume('IDENTIFIER').value
                        full_name = f"{full_name}::{rhs}"
                
                
                # Check for Struct Instantiation: Mod::Struct { ... }
                if self.peek().type == 'LBRACE':
                    self.consume('LBRACE')
                    args = []
                    while self.peek().type != 'RBRACE':
                        # self.consume('IDENTIFIER') # field name (not consumed in call args logic usually? Wait.)
                        # Logic in lines 752: self.consume('IDENTIFIER'); consume('COLON'); parse_expr()
                        # call args usually list of exprs.
                        # But struct instantiation logic uses field names?
                        # My CallExpr supports named args? 
                        # Review line 757: CallExpr(name, args). args is list of exprs.
                        # Wait, StructDef stores fields. 
                        # Compiler needs to know field order.
                        # If I preserve fields?
                        # bootstrap `visit_CallExpr` for struct (line 945 semantic.py) iterates `args` and visits them.
                        # It assumes `args` are Expressions.
                        # If I drop keys, I assume order matches?
                        # Line 750 (existing Struct Instantiation)
                        # self.consume('IDENTIFIER'); self.consume('COLON'); args.append(self.parse_expression())
                        # It drops keys.
                        # So I should do the same.
                        
                        self.consume('IDENTIFIER')
                        self.consume('COLON')
                        args.append(self.parse_expression())
                        if self.peek().type == 'COMMA': self.consume('COMMA')
                    self.consume('RBRACE')
                    expr = CallExpr(full_name, args)
                else:
                    expr = VariableExpr(full_name) # Postfix handles '(' or '.'
            
            # 4. Standard Case: ID
            else:
                expr = VariableExpr(token.value)
                self.consume('IDENTIFIER')

        elif token.type == 'STRUCT':
            # Struct Instantiation: struct Name { fields }
            self.consume('STRUCT')
            name = self.consume('IDENTIFIER').value
            self.consume('LBRACE')
            args = []
            while self.peek().type != 'RBRACE':
                self.consume('IDENTIFIER') # field name
                self.consume('COLON')
                args.append(self.parse_expression())
                if self.peek().type == 'COMMA': self.consume('COMMA')
            self.consume('RBRACE')
            # Treat as CallExpr with struct name for codegen
            expr = CallExpr(name, args)

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
        elif token.type == 'STAR':
            return self.parse_unary()
        elif token.type == 'PIPE':
            expr = self.parse_lambda()
        else:
            # Print context
            start = max(0, self.pos - 10)
            end = min(len(self.tokens), self.pos + 10)
            context = self.tokens[start:end]
            raise Exception(f"Unexpected token in primary: {token}. Type: '{token.type}'. Prev: {self.tokens[self.pos-1] if self.pos > 0 else 'START'} Next: {self.tokens[self.pos+1] if self.pos+1 < len(self.tokens) else 'EOF'}")
            
        # Postfix Handlers (Member Access, Index Access, Call)
        while True:
            if self.peek().type == 'DOT':
                self.consume('DOT')
                member = self.consume('IDENTIFIER').value
                # Check if this is a method call (followed by '(')
                if self.peek().type == 'LPAREN':
                    args = self.parse_call_arguments()
                    expr = MethodCall(expr, member, args)
                else:
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
                
        if expr:
            expr.line = start_token.line
            expr.column = start_token.column
            
        return expr

    def parse_lambda(self):
        start_token = self.peek()
        self.consume('PIPE')
        params = []
        if self.peek().type != 'PIPE':
            while True:
                pname = self.consume('IDENTIFIER').value
                ptype = None
                if self.peek().type == 'COLON':
                    self.consume('COLON')
                    ptype = self.parse_type()
                params.append((pname, ptype))
                if self.peek().type == 'COMMA':
                    self.consume('COMMA')
                    if self.peek().type == 'PIPE': break
                else:
                    break
        self.consume('PIPE')
        
        ret_type = None
        if self.peek().type == 'THIN_ARROW':
            self.consume('THIN_ARROW')
            ret_type = self.parse_type()
            
        if self.peek().type == 'LBRACE':
            self.consume('LBRACE')
            body = []
            while self.peek().type != 'RBRACE':
                body.append(self.parse_statement())
                if self.peek().type == 'SEMICOLON': self.consume('SEMICOLON')
            self.consume('RBRACE')
        else:
            # Single expression body: |x| x + 1
            body = [ReturnStmt(self.parse_expression())]
            
        node = LambdaExpr(params, ret_type, body)
        node.line = start_token.line
        node.column = start_token.column
        return node

    def parse_call(self):
        name = self.consume('IDENTIFIER').value
        return CallExpr(name, self.parse_call_arguments())

    def parse_call_explicit(self, name):
        # Used by DoubleColon path
        return CallExpr(name, self.parse_call_arguments())

    def parse_call_arguments(self):
        # No debug print
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
