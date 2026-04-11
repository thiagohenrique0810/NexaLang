"""TurboLang Parser — builds AST from token stream."""

from turbolang.grammar.tokens import Token, TokenType
from turbolang.ast.nodes import Program, ModelBlock, Directive


class ParseError(Exception):
    def __init__(self, message: str, token: Token):
        self.token = token
        super().__init__(f"Parse error at L{token.line}:{token.col}: {message}")


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def _current(self) -> Token:
        return self.tokens[self.pos]

    def _peek(self) -> Token:
        if self.pos + 1 < len(self.tokens):
            return self.tokens[self.pos + 1]
        return self.tokens[-1]

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, tt: TokenType) -> Token:
        tok = self._current()
        if tok.type != tt:
            raise ParseError(f"Expected {tt.name}, got {tok.type.name} ({tok.value!r})", tok)
        return self._advance()

    def _skip_newlines(self):
        while self._current().type == TokenType.NEWLINE:
            self._advance()

    def _is_value_token(self, tok: Token) -> bool:
        return tok.type in (
            TokenType.STRING, TokenType.INTEGER, TokenType.FLOAT,
            TokenType.BOOLEAN, TokenType.IDENTIFIER,
        )

    def _is_directive_keyword(self, tok: Token) -> bool:
        return tok.type in (
            TokenType.WEIGHTS, TokenType.KV_CACHE, TokenType.ATTENTION,
            TokenType.DECODE, TokenType.SCHEDULER, TokenType.TARGET,
        )

    def parse(self) -> Program:
        program = Program()
        self._skip_newlines()
        while self._current().type != TokenType.EOF:
            program.models.append(self._parse_model())
            self._skip_newlines()
        return program

    def _parse_model(self) -> ModelBlock:
        tok = self._expect(TokenType.MODEL)
        name_tok = self._expect(TokenType.STRING)
        self._skip_newlines()
        self._expect(TokenType.LBRACE)
        self._skip_newlines()

        model = ModelBlock(line=tok.line, col=tok.col, name=name_tok.value)

        while self._current().type != TokenType.RBRACE:
            if self._current().type == TokenType.EOF:
                raise ParseError("Unterminated model block — missing '}'", self._current())
            model.directives.append(self._parse_directive())
            self._skip_newlines()

        self._expect(TokenType.RBRACE)
        return model

    def _parse_directive(self) -> Directive:
        tok = self._current()
        if not self._is_directive_keyword(tok):
            raise ParseError(
                f"Expected directive keyword (weights, kv_cache, attention, decode, scheduler, target), "
                f"got {tok.type.name} ({tok.value!r})", tok
            )
        self._advance()
        keyword = tok.value
        directive = Directive(line=tok.line, col=tok.col, keyword=keyword)

        # Parse params until newline, '}', EOF, or next directive keyword
        while self._current().type not in (TokenType.NEWLINE, TokenType.RBRACE, TokenType.EOF):
            if self._is_directive_keyword(self._current()):
                break
            param_tok = self._current()

            # Check for named param: IDENTIFIER '=' value
            if param_tok.type == TokenType.IDENTIFIER and self._peek().type == TokenType.EQUALS:
                name = self._advance().value  # IDENTIFIER
                self._advance()               # '='
                val_tok = self._current()
                if not self._is_value_token(val_tok):
                    raise ParseError(f"Expected value after '=', got {val_tok.type.name}", val_tok)
                self._advance()
                directive.named[name] = val_tok.value
            elif self._is_value_token(param_tok):
                self._advance()
                directive.positional.append(param_tok.value)
            else:
                raise ParseError(f"Unexpected token in directive: {param_tok.type.name}", param_tok)

        return directive
