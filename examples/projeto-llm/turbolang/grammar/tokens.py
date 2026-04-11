"""Token definitions for TurboLang DSL."""

from enum import Enum, auto


class TokenType(Enum):
    # Literals
    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    IDENTIFIER = auto()

    # Delimiters
    LBRACE = auto()
    RBRACE = auto()
    EQUALS = auto()

    # Keywords
    MODEL = auto()
    WEIGHTS = auto()
    KV_CACHE = auto()
    ATTENTION = auto()
    DECODE = auto()
    SCHEDULER = auto()
    TARGET = auto()

    # End
    EOF = auto()
    NEWLINE = auto()


# Keywords mapping
KEYWORDS = {
    'model': TokenType.MODEL,
    'weights': TokenType.WEIGHTS,
    'kv_cache': TokenType.KV_CACHE,
    'attention': TokenType.ATTENTION,
    'decode': TokenType.DECODE,
    'scheduler': TokenType.SCHEDULER,
    'target': TokenType.TARGET,
    'true': TokenType.BOOLEAN,
    'false': TokenType.BOOLEAN,
}


class Token:
    __slots__ = ('type', 'value', 'line', 'col')

    def __init__(self, type: TokenType, value, line: int = 0, col: int = 0):
        self.type = type
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, L{self.line}:{self.col})"
