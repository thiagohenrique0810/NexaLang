class ASTNode:
    pass

class FunctionDef(ASTNode):
    def __init__(self, name, body):
        self.name = name
        self.body = body

class CallExpr(ASTNode):
    def __init__(self, callee, args):
        self.callee = callee
        self.args = args

class StringLiteral(ASTNode):
    def __init__(self, value):
        self.value = value

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def consume(self, type):
        if self.pos < len(self.tokens) and self.tokens[self.pos].type == type:
            self.pos += 1
            return self.tokens[self.pos - 1]
        raise Exception(f"Expected token type {type}, found {self.tokens[self.pos].type if self.pos < len(self.tokens) else 'EOF'}")

    def parse(self):
        functions = []
        while self.pos < len(self.tokens):
            functions.append(self.parse_function())
        return functions

    def parse_function(self):
        self.consume('FN')
        name = self.consume('IDENTIFIER').value
        self.consume('LPAREN')
        self.consume('RPAREN')
        self.consume('LBRACE')
        body = []
        while self.pos < len(self.tokens) and self.tokens[self.pos].type != 'RBRACE':
            body.append(self.parse_statement())
        self.consume('RBRACE')
        return FunctionDef(name, body)

    def parse_statement(self):
        token = self.tokens[self.pos]
        if token.type == 'IDENTIFIER':
            return self.parse_call()
        raise Exception(f"Unexpected token in statement: {token}")

    def parse_call(self):
        name = self.consume('IDENTIFIER').value
        self.consume('LPAREN')
        args = []
        if self.tokens[self.pos].type != 'RPAREN':
            args.append(self.parse_expression())
        self.consume('RPAREN')
        return CallExpr(name, args)

    def parse_expression(self):
        token = self.tokens[self.pos]
        if token.type == 'STRING':
            self.consume('STRING')
            return StringLiteral(token.value)
        raise Exception(f"Unexpected token in expression: {token}")
