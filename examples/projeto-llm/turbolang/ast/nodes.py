"""TurboLang AST node definitions."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ASTNode:
    line: int = 0
    col: int = 0


@dataclass
class NamedParam(ASTNode):
    name: str = ""
    value: Any = None


@dataclass
class Directive(ASTNode):
    keyword: str = ""
    positional: list[Any] = field(default_factory=list)
    named: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelBlock(ASTNode):
    name: str = ""
    directives: list[Directive] = field(default_factory=list)

    def get_directive(self, keyword: str) -> Directive | None:
        for d in self.directives:
            if d.keyword == keyword:
                return d
        return None


@dataclass
class Program(ASTNode):
    models: list[ModelBlock] = field(default_factory=list)
