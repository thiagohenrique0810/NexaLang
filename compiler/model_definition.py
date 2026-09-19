"""Lower the declarative model DSL to ModelConfig, independently of nxc.

The existing Nexa lexer supplies tokens and source locations. This parser has
its own strict grammar; it does not pretend model declarations are executable
Nexa language code. Hash and block comments follow the shared lexer syntax.
"""
from bootstrap.lexer import Lexer

from .model_config import ModelConfig


class ModelDefinitionError(ValueError):
    pass


_TOP_REQUIRED = {"vocab_size", "hidden_size", "layers", "attention", "ffn",
                 "norm", "context", "tied_embeddings"}
_TOP_OPTIONAL = {"rope_theta", "rms_norm_eps"}
_SECTIONS = {"attention": {"query_heads", "kv_heads", "head_dim", "position"},
             "ffn": {"hidden_size", "activation"}}


def _check_comments(text):
    # The shared lexer accepts a block comment at EOF without reporting its
    # missing terminator. Reject that case here without changing the bootstrap.
    position = 0
    while position < len(text):
        if text[position] == "#":
            end = text.find("\n", position)
            position = len(text) if end < 0 else end + 1
        elif text.startswith("/*", position):
            end = text.find("*/", position + 2)
            if end < 0:
                line = text.count("\n", 0, position) + 1
                raise ModelDefinitionError(f"unterminated block comment at line {line}")
            position = end + 2
        else:
            position += 1


class _Parser:
    def __init__(self, text):
        if not isinstance(text, str):
            raise ModelDefinitionError("model definition must be source text")
        _check_comments(text)
        try:
            self.tokens = Lexer(text).tokenize()
        except Exception as error:
            raise ModelDefinitionError(str(error)) from error
        self.position = 0

    def _error(self, message):
        if self.position < len(self.tokens):
            token = self.tokens[self.position]
            return ModelDefinitionError(f"{message} at {token.line}:{token.column}")
        return ModelDefinitionError(f"{message} at end of model definition")

    def _at(self, kind):
        return self.position < len(self.tokens) and self.tokens[self.position].type == kind

    def _take(self, kind, value=None):
        if not self._at(kind) or (value is not None and self.tokens[self.position].value != value):
            raise self._error(f"expected {value or kind}")
        token = self.tokens[self.position]
        self.position += 1
        return token.value

    def _scalar(self):
        if self._at("NUMBER"):
            return int(self._take("NUMBER"))
        if self._at("FLOAT"):
            return float(self._take("FLOAT"))
        if self._at("TRUE"):
            self._take("TRUE")
            return True
        if self._at("FALSE"):
            self._take("FALSE")
            return False
        if self._at("IDENTIFIER"):
            return self._take("IDENTIFIER")
        raise self._error("expected an integer, decimal, boolean or architecture identifier")

    def _block(self, required, optional=(), allow_sections=False):
        self._take("LBRACE")
        result = {}
        while not self._at("RBRACE"):
            name = self._take("IDENTIFIER")
            if name not in required and name not in optional:
                raise self._error(f"unknown model field {name}")
            if name in result:
                raise self._error(f"duplicate model field {name}")
            if allow_sections and name in _SECTIONS:
                result[name] = self._block(_SECTIONS[name])
            else:
                self._take("COLON")
                result[name] = self._scalar()
                self._take("SEMICOLON")
        self._take("RBRACE")
        missing = set(required) - set(result)
        if missing:
            raise self._error(f"missing model fields: {', '.join(sorted(missing))}")
        return result

    def parse(self):
        result = {}
        while self.position < len(self.tokens):
            self._take("IDENTIFIER", "model")
            name = self._take("IDENTIFIER")
            if name in result:
                raise self._error(f"duplicate model name {name}")
            body = self._block(_TOP_REQUIRED, _TOP_OPTIONAL, allow_sections=True)
            attention, ffn = body["attention"], body["ffn"]
            if body["norm"] != "rmsnorm":
                raise self._error("only norm: rmsnorm is supported")
            if attention["position"] != "rope":
                raise self._error("only attention position: rope is supported")
            if ffn["activation"] != "swiglu":
                raise self._error("only ffn activation: swiglu is supported")
            try:
                config = ModelConfig(
                    name=name, vocab_size=body["vocab_size"], hidden_size=body["hidden_size"],
                    intermediate_size=ffn["hidden_size"], num_hidden_layers=body["layers"],
                    num_attention_heads=attention["query_heads"], num_key_value_heads=attention["kv_heads"],
                    max_position_embeddings=body["context"], tie_word_embeddings=body["tied_embeddings"],
                    rope_theta=body.get("rope_theta", 10000.0), rms_norm_eps=body.get("rms_norm_eps", 1e-5))
                if type(attention["head_dim"]) is not int or attention["head_dim"] != config.head_dim:
                    raise ValueError("attention head_dim must equal hidden_size / query_heads")
            except ValueError as error:
                raise self._error(f"{name}: {error}") from error
            result[name] = config
        if not result:
            raise self._error("expected at least one model declaration")
        return result


def compile_model_definition(text):
    """Compile one or more named model declarations to validated configs."""
    return _Parser(text).parse()
