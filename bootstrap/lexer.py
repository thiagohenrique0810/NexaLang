import re

class Token:
    def __init__(self, type, value, line=1, column=1):
        self.type = type
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"Token({self.type}, {self.value}, {self.line}:{self.column})"

class Lexer:
    def __init__(self, source):
        self.source = source
        self.pos = 0
        self.length = len(source)
        self.line = 1
        self.column = 1
    
    def advance(self, n=1):
        for _ in range(n):
            if self.pos < self.length:
                if self.source[self.pos] == '\n':
                    self.line += 1
                    self.column = 1
                else:
                    self.column += 1
                self.pos += 1

    def tokenize(self):
        tokens = []
        while self.pos < self.length:
            char = self.source[self.pos]
            start_line = self.line
            start_col = self.column
            
            # Skip whitespace
            if char.isspace():
                self.advance()
                continue
            # Skip comments
            elif char == '#':
                while self.pos < self.length and self.source[self.pos] != '\n':
                    self.advance()
                # If the comment ends with a newline, consume it too
                if self.pos < self.length and self.source[self.pos] == '\n':
                    self.advance()
                continue
            
            # Multi-line comments: /* ... */
            elif char == '/' and self.pos + 1 < self.length and self.source[self.pos+1] == '*':
                self.advance(2)
                while self.pos + 1 < self.length:
                    if self.source[self.pos] == '*' and self.source[self.pos+1] == '/':
                        self.advance(2)
                        break
                    self.advance()
                continue
            
            # Punctuation
            if char == ':' and self.pos + 1 < self.length and self.source[self.pos+1] == ':':
                tokens.append(Token('DOUBLE_COLON', '::', start_line, start_col))
                self.advance(2)
                continue

            if char == '(':
                tokens.append(Token('LPAREN', '(', start_line, start_col))
                self.advance()
            elif char == ')':
                tokens.append(Token('RPAREN', ')', start_line, start_col))
                self.advance()
            elif char == '[':
                tokens.append(Token('LBRACKET', '[', start_line, start_col))
                self.advance()
            elif char == ']':
                tokens.append(Token('RBRACKET', ']', start_line, start_col))
                self.advance()
            elif char == '{':
                tokens.append(Token('LBRACE', '{', start_line, start_col))
                self.advance()
            elif char == '}':
                tokens.append(Token('RBRACE', '}', start_line, start_col))
                self.advance()
            elif char == '.':
                if self.pos + 1 < self.length and self.source[self.pos+1] == '.':
                    if self.pos + 2 < self.length and self.source[self.pos+2] == '.':
                        tokens.append(Token('ELLIPSIS', '...', start_line, start_col))
                        self.advance(3)
                    elif self.pos + 2 < self.length and self.source[self.pos+2] == '=':
                        tokens.append(Token('DOT_DOT_EQ', '..=', start_line, start_col))
                        self.advance(3)
                    else:
                        tokens.append(Token('DOT_DOT', '..', start_line, start_col))
                        self.advance(2)
                else:
                    tokens.append(Token('DOT', '.', start_line, start_col))
                    self.advance()
            elif char == ':':
                if self.pos + 1 < self.length and self.source[self.pos+1] == ':':
                    tokens.append(Token('DOUBLE_COLON', '::', start_line, start_col))
                    self.advance(2)
                else:
                    tokens.append(Token('COLON', ':', start_line, start_col))
                    self.advance()
            elif char == ',':
                tokens.append(Token('COMMA', ',', start_line, start_col))
                self.advance()
            elif char == ';':
                tokens.append(Token('SEMICOLON', ';', start_line, start_col))
                self.advance()
            
            # Operators
            elif char == '=':
                if self.pos + 1 < self.length and self.source[self.pos+1] == '=':
                   tokens.append(Token('EQEQ', '==', start_line, start_col))
                   self.advance(2)
                elif self.pos + 1 < self.length and self.source[self.pos+1] == '>':
                   tokens.append(Token('FAT_ARROW', '=>', start_line, start_col))
                   self.advance(2)
                else:
                   tokens.append(Token('EQ', '=', start_line, start_col))
                   self.advance()
            elif char == '<':
                if self.pos + 1 < self.length and self.source[self.pos+1] == '=':
                   tokens.append(Token('LTE', '<=', start_line, start_col))
                   self.advance(2)
                else:
                   tokens.append(Token('LT', '<', start_line, start_col))
                   self.advance()
            elif char == '>':
                if self.pos + 1 < self.length and self.source[self.pos+1] == '=':
                   tokens.append(Token('GTE', '>=', start_line, start_col))
                   self.advance(2)
                else:
                   tokens.append(Token('GT', '>', start_line, start_col))
                   self.advance()
            elif char == '+':
                tokens.append(Token('PLUS', '+', start_line, start_col))
                self.advance()
            elif char == '-':
                if self.pos + 1 < self.length and self.source[self.pos+1] == '>':
                    tokens.append(Token('THIN_ARROW', '->', start_line, start_col))
                    self.advance(2)
                else:
                    tokens.append(Token('MINUS', '-', start_line, start_col))
                    self.advance()
            elif char == '*':
                tokens.append(Token('STAR', '*', start_line, start_col))
                self.advance()
            elif char == '/':
                tokens.append(Token('SLASH', '/', start_line, start_col))
                self.advance()
            elif char == '%':
                tokens.append(Token('PERCENT', '%', start_line, start_col))
                self.advance()
            elif char == '&':
                tokens.append(Token('AMPERSAND', '&', start_line, start_col))
                self.advance()
            elif char == '!':
                if self.pos + 1 < self.length and self.source[self.pos+1] == '=':
                   tokens.append(Token('NEQ', '!=', start_line, start_col))
                   self.advance(2)
                else:
                   tokens.append(Token('NOT', '!', start_line, start_col))
                   self.advance()
            elif char == '|':
                tokens.append(Token('PIPE', '|', start_line, start_col))
                self.advance()

            # String Literals
            elif char == '"':
                self.advance()
                value = ""
                while self.pos < self.length:
                    if self.source[self.pos] == '"':
                        break
                    if self.source[self.pos] == '\\' and self.pos + 1 < self.length:
                        esc = self.source[self.pos + 1]
                        if esc == 'n':
                            value += '\n'
                        elif esc == 't':
                            value += '\t'
                        elif esc == '\\':
                            value += '\\'
                        elif esc == '"':
                            value += '"'
                        else:
                            value += '\\' + esc
                        self.advance(2)
                    else:
                        value += self.source[self.pos]
                        self.advance()
                tokens.append(Token('STRING', value, start_line, start_col))
                self.advance()

            # Char Literals or Labels ('label)
            elif char == "'":
                self.advance()  # consume opening '
                if self.pos < self.length and (self.source[self.pos].isalpha() or self.source[self.pos] == '_'):
                    # Could be a label like 'label or a char literal like 'a'
                    start_ident = self.pos
                    while self.pos < self.length and (self.source[self.pos].isalnum() or self.source[self.pos] == '_'):
                        self.advance()
                    
                    ident_value = self.source[start_ident:self.pos]
                    
                    # If followed by another ', it's a char literal (if single char)
                    if self.pos < self.length and self.source[self.pos] == "'":
                        if len(ident_value) == 1:
                            self.advance() # consume closing '
                            tokens.append(Token('CHAR', str(ord(ident_value)), start_line, start_col))
                            continue
                    
                    # Otherwise, it's a LABEL
                    tokens.append(Token('LABEL', ident_value, start_line, start_col))
                    continue

                if self.pos >= self.length:
                    raise Exception(f"Unterminated char literal at {start_line}:{start_col}")

                c = self.source[self.pos]
                if c == '\\':
                    self.advance()
                    if self.pos >= self.length:
                        raise Exception(f"Unterminated char escape at {start_line}:{start_col}")
                    esc = self.source[self.pos]
                    self.advance()
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
                            raise Exception(f"Invalid \\x escape in char literal at {start_line}:{start_col}")
                        h1 = self.source[self.pos]
                        h2 = self.source[self.pos + 1]
                        if not re.match(r"[0-9a-fA-F]", h1) or not re.match(r"[0-9a-fA-F]", h2):
                            raise Exception(f"Invalid \\x escape in char literal at {start_line}:{start_col}")
                        codepoint = int(h1 + h2, 16)
                        self.advance(2)
                    else:
                        raise Exception(f"Unknown char escape: \\{esc} at {start_line}:{start_col}")
                else:
                    codepoint = ord(c)
                    self.advance()

                if self.pos >= self.length or self.source[self.pos] != "'":
                    raise Exception(f"Unterminated char literal (missing closing ') at {start_line}:{start_col}")
                self.advance()  # consume closing '
                tokens.append(Token('CHAR', str(codepoint), start_line, start_col))
            
            # Numbers: integer or float (e.g. 123, 3.14)
            elif char.isdigit():
                start = self.pos
                while self.pos < self.length and self.source[self.pos].isdigit():
                    self.advance()

                is_float = False
                # Float: digits '.' digits
                if (
                    self.pos + 1 < self.length
                    and self.source[self.pos] == '.'
                    and self.source[self.pos + 1].isdigit()
                ):
                    is_float = True
                    self.advance()  # consume '.'
                    while self.pos < self.length and self.source[self.pos].isdigit():
                        self.advance()

                value = self.source[start:self.pos]
                tokens.append(Token('FLOAT' if is_float else 'NUMBER', value, start_line, start_col))

            # Identifiers and Keywords
            elif char.isalpha() or char == '_':
                start = self.pos
                while self.pos < self.length and (self.source[self.pos].isalnum() or self.source[self.pos] == '_'):
                    self.advance()
                value = self.source[start:self.pos]
                
                # Keywords map
                keywords = {
                    'fn': 'FN', 'kernel': 'KERNEL', 'struct': 'STRUCT', 'enum': 'ENUM',
                    'region': 'REGION', 'let': 'LET', 'return': 'RETURN', 'if': 'IF',
                    'else': 'ELSE', 'while': 'WHILE', 'match': 'MATCH', 'true': 'TRUE',
                    'false': 'FALSE', 'mut': 'MUT', 'impl': 'IMPL', 'self': 'SELF',
                    'or': 'OR', 'and': 'AND', 'mod': 'MOD', 'pub': 'PUB', 'for': 'FOR',
                    'in': 'IN',                     'break': 'BREAK', 'continue': 'CONTINUE', 'trait': 'TRAIT',
                    'use': 'USE', 'type': 'TYPE', 'extern': 'EXTERN', 'async': 'ASYNC',
                    'await': 'AWAIT'
                }

                type_name = keywords.get(value, 'IDENTIFIER')
                tokens.append(Token(type_name, value, start_line, start_col))

            else:
                raise Exception(f"Unexpected character: {char} at {start_line}:{start_col}")
                
        return tokens
