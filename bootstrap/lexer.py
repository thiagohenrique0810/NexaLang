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
            
            # Punctuation
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
            elif char == ':':
                tokens.append(Token('COLON', ':'))
                self.pos += 1
            
            # Operators
            elif char == '=':
                if self.pos + 1 < self.length and self.source[self.pos+1] == '=':
                   tokens.append(Token('EQEQ', '=='))
                   self.pos += 2
                else:
                   tokens.append(Token('EQ', '='))
                   self.pos += 1
            elif char == '+':
                tokens.append(Token('PLUS', '+'))
                self.pos += 1
            elif char == '-':
                tokens.append(Token('MINUS', '-'))
                self.pos += 1
            elif char == '*':
                tokens.append(Token('STAR', '*'))
                self.pos += 1
            elif char == '/':
                tokens.append(Token('SLASH', '/'))
                self.pos += 1

            # String Literals
            elif char == '"':
                self.pos += 1
                start = self.pos
                while self.pos < self.length and self.source[self.pos] != '"':
                    self.pos += 1
                value = self.source[start:self.pos]
                tokens.append(Token('STRING', value))
                self.pos += 1
            
            # Numbers (Integers for now)
            elif char.isdigit():
                start = self.pos
                while self.pos < self.length and self.source[self.pos].isdigit():
                    self.pos += 1
                value = self.source[start:self.pos]
                tokens.append(Token('NUMBER', value))

            # Identifiers and Keywords
            elif char.isalpha() or char == '_':
                start = self.pos
                while self.pos < self.length and (self.source[self.pos].isalnum() or self.source[self.pos] == '_'):
                    self.pos += 1
                value = self.source[start:self.pos]
                if value == 'fn':
                    tokens.append(Token('FN', value))
                elif value == 'let':
                    tokens.append(Token('LET', value))
                elif value == 'return':
                    tokens.append(Token('RETURN', value))
                elif value == 'if':
                    tokens.append(Token('IF', value))
                elif value == 'else':
                    tokens.append(Token('ELSE', value))
                elif value == 'while':
                    tokens.append(Token('WHILE', value))
                else:
                    tokens.append(Token('IDENTIFIER', value))
            else:
                raise Exception(f"Unexpected character: {char} at pos {self.pos}")
                
        return tokens
