import re

class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {self.value})"

class Lexer:
    def __init__(self, source):
        self.source = source
        self.pos = 0
        self.length = len(source)
    
    def tokenize(self):
        tokens = []
        while self.pos < self.length:
            char = self.source[self.pos]
            
            if char.isspace():
                self.pos += 1
                continue
            
            if char == '(':
                tokens.append(Token('LPAREN', '('))
                self.pos += 1
            elif char == ')':
                tokens.append(Token('RPAREN', ')'))
                self.pos += 1
            elif char == '{':
                tokens.append(Token('LBRACE', '{'))
                self.pos += 1
            elif char == '}':
                tokens.append(Token('RBRACE', '}'))
                self.pos += 1
            elif char == '"':
                self.pos += 1
                start = self.pos
                while self.pos < self.length and self.source[self.pos] != '"':
                    self.pos += 1
                value = self.source[start:self.pos]
                tokens.append(Token('STRING', value))
                self.pos += 1
            elif char.isalpha():
                start = self.pos
                while self.pos < self.length and (self.source[self.pos].isalnum() or self.source[self.pos] == '_'):
                    self.pos += 1
                value = self.source[start:self.pos]
                if value == 'fn':
                    tokens.append(Token('FN', value))
                else:
                    tokens.append(Token('IDENTIFIER', value))
            else:
                raise Exception(f"Unexpected character: {char}")
                
        return tokens
