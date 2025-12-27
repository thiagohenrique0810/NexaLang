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
            
            # Skip whitespace
            if char.isspace():
                self.pos += 1
                continue
            # Skip comments
            elif char == '#':
                while self.pos < self.length and self.source[self.pos] != '\n':
                    self.pos += 1
                # If the comment ends with a newline, consume it too
                if self.pos < self.length and self.source[self.pos] == '\n':
                    self.pos += 1
                continue
            
            # Punctuation
            if char == ':' and self.pos + 1 < self.length and self.source[self.pos+1] == ':':
                tokens.append(Token('DOUBLE_COLON', '::'))
                self.pos += 2
                continue

            if char == '(':
                tokens.append(Token('LPAREN', '('))
                self.pos += 1
            elif char == ')':
                tokens.append(Token('RPAREN', ')'))
                self.pos += 1
            elif char == '[':
                tokens.append(Token('LBRACKET', '['))
                self.pos += 1
            elif char == ']':
                tokens.append(Token('RBRACKET', ']'))
                self.pos += 1
            elif char == '{':
                tokens.append(Token('LBRACE', '{'))
                self.pos += 1
            elif char == '}':
                tokens.append(Token('RBRACE', '}'))
                self.pos += 1
            elif char == '.':
                tokens.append(Token('DOT', '.'))
                self.pos += 1
            elif char == ':':
                tokens.append(Token('COLON', ':'))
                self.pos += 1
            elif char == ',':
                tokens.append(Token('COMMA', ','))
                self.pos += 1
            elif char == ';':
                tokens.append(Token('SEMICOLON', ';'))
                self.pos += 1
            
            # Operators
            elif char == '=':
                if self.pos + 1 < self.length and self.source[self.pos+1] == '=':
                   tokens.append(Token('EQEQ', '=='))
                   self.pos += 2
                elif self.pos + 1 < self.length and self.source[self.pos+1] == '>':
                   tokens.append(Token('FAT_ARROW', '=>'))
                   self.pos += 2
                else:
                   tokens.append(Token('EQ', '='))
                   self.pos += 1
            elif char == '<':
                tokens.append(Token('LT', '<'))
                self.pos += 1
            elif char == '>':
                tokens.append(Token('GT', '>'))
                self.pos += 1
            elif char == '+':
                tokens.append(Token('PLUS', '+'))
                self.pos += 1
            elif char == '-':
                if self.pos + 1 < self.length and self.source[self.pos+1] == '>':
                    tokens.append(Token('THIN_ARROW', '->'))
                    self.pos += 2
                else:
                    tokens.append(Token('MINUS', '-'))
                    self.pos += 1
            elif char == '*':
                tokens.append(Token('STAR', '*'))
                self.pos += 1
            elif char == '/':
                tokens.append(Token('SLASH', '/'))
                self.pos += 1
            elif char == '&':
                tokens.append(Token('AMPERSAND', '&'))
                self.pos += 1
            elif char == '!':
                if self.pos + 1 < self.length and self.source[self.pos+1] == '=':
                   tokens.append(Token('NEQ', '!='))
                   self.pos += 2
                else:
                   tokens.append(Token('BANG', '!'))
                   self.pos += 1

            # String Literals
            elif char == '"':
                self.pos += 1
                start = self.pos
                while self.pos < self.length:
                    if self.source[self.pos] == '"':
                        break
                    if self.source[self.pos] == '\\' and self.pos + 1 < self.length:
                        self.pos += 2 # Skip escape sequence
                    else:
                        self.pos += 1
                value = self.source[start:self.pos]
                tokens.append(Token('STRING', value))
                self.pos += 1

            # Char Literals: 'a', '\n', '\'', '\\', '\x41'
            elif char == "'":
                self.pos += 1  # consume opening '
                if self.pos >= self.length:
                    raise Exception("Unterminated char literal")

                c = self.source[self.pos]
                if c == '\\':
                    self.pos += 1
                    if self.pos >= self.length:
                        raise Exception("Unterminated char escape")
                    esc = self.source[self.pos]
                    self.pos += 1
                    if esc == 'n':
                        codepoint = ord('\n')
                    elif esc == 't':
                        codepoint = ord('\t')
                    elif esc == 'r':
                        codepoint = ord('\r')
                    elif esc == '0':
                        codepoint = 0
                    elif esc == "'":
                        codepoint = ord("'")
                    elif esc == '"':
                        codepoint = ord('"')
                    elif esc == '\\':
                        codepoint = ord('\\')
                    elif esc == 'x':
                        # \xNN
                        if self.pos + 1 >= self.length:
                            raise Exception("Invalid \\x escape in char literal")
                        h1 = self.source[self.pos]
                        h2 = self.source[self.pos + 1]
                        if not re.match(r"[0-9a-fA-F]", h1) or not re.match(r"[0-9a-fA-F]", h2):
                            raise Exception("Invalid \\x escape in char literal")
                        codepoint = int(h1 + h2, 16)
                        self.pos += 2
                    else:
                        raise Exception(f"Unknown char escape: \\{esc}")
                else:
                    codepoint = ord(c)
                    self.pos += 1

                if self.pos >= self.length or self.source[self.pos] != "'":
                    raise Exception("Unterminated char literal (missing closing ')")
                self.pos += 1  # consume closing '
                tokens.append(Token('CHAR', str(codepoint)))
            
            # Numbers: integer or float (e.g. 123, 3.14)
            elif char.isdigit():
                start = self.pos
                while self.pos < self.length and self.source[self.pos].isdigit():
                    self.pos += 1

                is_float = False
                # Float: digits '.' digits
                if (
                    self.pos + 1 < self.length
                    and self.source[self.pos] == '.'
                    and self.source[self.pos + 1].isdigit()
                ):
                    is_float = True
                    self.pos += 1  # consume '.'
                    while self.pos < self.length and self.source[self.pos].isdigit():
                        self.pos += 1

                value = self.source[start:self.pos]
                tokens.append(Token('FLOAT' if is_float else 'NUMBER', value))

            # Identifiers and Keywords
            elif char.isalpha() or char == '_':
                start = self.pos
                while self.pos < self.length and (self.source[self.pos].isalnum() or self.source[self.pos] == '_'):
                    self.pos += 1
                value = self.source[start:self.pos]
                if value == 'fn':
                    tokens.append(Token('FN', value))
                elif value == 'kernel':
                     tokens.append(Token('KERNEL', value))
                elif value == 'struct':
                     tokens.append(Token('STRUCT', value))
                elif value == 'enum':
                     tokens.append(Token('ENUM', value))
                elif value == 'region':
                     tokens.append(Token('REGION', value))
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
                elif value == 'match':
                    tokens.append(Token('MATCH', value))
                elif value == 'true':
                    tokens.append(Token('TRUE', value))
                elif value == 'false':
                    tokens.append(Token('FALSE', value))
                elif value == 'mut':
                    tokens.append(Token('MUT', value))
                elif value == 'impl':
                    tokens.append(Token('IMPL', value))
                elif value == 'self':
                    tokens.append(Token('SELF', value))
                elif value == 'or':
                    tokens.append(Token('OR', value))
                elif value == 'and':
                    tokens.append(Token('AND', value))
                else:
                    tokens.append(Token('IDENTIFIER', value))
            else:
                raise Exception(f"Unexpected character: {char} at pos {self.pos}")
                
        return tokens
