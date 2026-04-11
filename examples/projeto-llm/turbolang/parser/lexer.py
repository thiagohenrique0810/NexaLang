"""TurboLang Lexer — tokenizes .tl source files."""

from turbolang.grammar.tokens import Token, TokenType, KEYWORDS


class LexerError(Exception):
    def __init__(self, message: str, line: int, col: int):
        self.line = line
        self.col = col
        super().__init__(f"Lexer error at L{line}:{col}: {message}")


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1

    def _peek(self) -> str | None:
        if self.pos < len(self.source):
            return self.source[self.pos]
        return None

    def _advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _skip_whitespace_and_comments(self):
        while self.pos < len(self.source):
            ch = self.source[self.pos]
            if ch in (' ', '\t', '\r'):
                self._advance()
            elif ch == '#':
                # Line comment
                while self.pos < len(self.source) and self.source[self.pos] != '\n':
                    self._advance()
            elif ch == '/' and self.pos + 1 < len(self.source) and self.source[self.pos + 1] == '/':
                while self.pos < len(self.source) and self.source[self.pos] != '\n':
                    self._advance()
            else:
                break

    def _read_string(self) -> str:
        quote = self._advance()  # consume opening quote
        result = []
        while self.pos < len(self.source):
            ch = self._advance()
            if ch == '\\':
                if self.pos < len(self.source):
                    esc = self._advance()
                    escape_map = {'n': '\n', 't': '\t', '\\': '\\', '"': '"', "'": "'"}
                    result.append(escape_map.get(esc, esc))
                else:
                    raise LexerError("Unterminated escape sequence", self.line, self.col)
            elif ch == quote:
                return ''.join(result)
            else:
                result.append(ch)
        raise LexerError("Unterminated string literal", self.line, self.col)

    def _read_number(self) -> Token:
        start_line, start_col = self.line, self.col
        chars = []
        has_dot = False
        while self.pos < len(self.source):
            ch = self.source[self.pos]
            if ch == '.' and not has_dot:
                has_dot = True
                chars.append(self._advance())
            elif ch.isdigit():
                chars.append(self._advance())
            else:
                break
        num_str = ''.join(chars)
        if has_dot:
            return Token(TokenType.FLOAT, float(num_str), start_line, start_col)
        return Token(TokenType.INTEGER, int(num_str), start_line, start_col)

    def _read_identifier(self) -> Token:
        start_line, start_col = self.line, self.col
        chars = []
        while self.pos < len(self.source):
            ch = self.source[self.pos]
            if ch.isalnum() or ch == '_':
                chars.append(self._advance())
            else:
                break
        word = ''.join(chars)
        if word in KEYWORDS:
            tt = KEYWORDS[word]
            val = word
            if tt == TokenType.BOOLEAN:
                val = word == 'true'
            return Token(tt, val, start_line, start_col)
        return Token(TokenType.IDENTIFIER, word, start_line, start_col)

    def tokenize(self) -> list[Token]:
        tokens = []
        while self.pos < len(self.source):
            self._skip_whitespace_and_comments()
            if self.pos >= len(self.source):
                break

            ch = self.source[self.pos]
            line, col = self.line, self.col

            if ch == '\n':
                self._advance()
                tokens.append(Token(TokenType.NEWLINE, '\\n', line, col))
            elif ch == '{':
                self._advance()
                tokens.append(Token(TokenType.LBRACE, '{', line, col))
            elif ch == '}':
                self._advance()
                tokens.append(Token(TokenType.RBRACE, '}', line, col))
            elif ch == '=':
                self._advance()
                tokens.append(Token(TokenType.EQUALS, '=', line, col))
            elif ch in ('"', "'"):
                val = self._read_string()
                tokens.append(Token(TokenType.STRING, val, line, col))
            elif ch.isdigit():
                tokens.append(self._read_number())
            elif ch.isalpha() or ch == '_':
                tokens.append(self._read_identifier())
            else:
                raise LexerError(f"Unexpected character: {ch!r}", line, col)

        tokens.append(Token(TokenType.EOF, None, self.line, self.col))
        return tokens
