"""Tests for TurboLang parser — lexer, parser, AST, validator."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from turbolang.parser.lexer import Lexer, LexerError
from turbolang.parser.parser import Parser, ParseError
from turbolang.ast.nodes import Program, ModelBlock, Directive
from turbolang.validator.validator import Validator, ValidationError


# ── Lexer tests ──────────────────────────────────────────────

class TestLexer:
    def test_basic_tokenize(self):
        src = 'model "gpt2" { weights full fp16 }'
        tokens = Lexer(src).tokenize()
        assert tokens[0].type.name == "MODEL"
        assert tokens[1].value == "gpt2"
        assert tokens[2].type.name == "LBRACE"
        assert tokens[3].type.name == "WEIGHTS"

    def test_named_params(self):
        src = 'model "test" { kv_cache turboquant bits=3 }'
        tokens = Lexer(src).tokenize()
        values = [t.value for t in tokens]
        assert "bits" in values
        assert 3 in values

    def test_boolean_literal(self):
        src = 'model "t" { attention flash streaming=true }'
        tokens = Lexer(src).tokenize()
        bool_tokens = [t for t in tokens if t.type.name == "BOOLEAN"]
        assert len(bool_tokens) == 1
        assert bool_tokens[0].value is True

    def test_comments(self):
        src = '# This is a comment\nmodel "test" { }'
        tokens = Lexer(src).tokenize()
        assert any(t.type.name == "MODEL" for t in tokens)

    def test_unterminated_string(self):
        with pytest.raises(LexerError):
            Lexer('model "unterminated').tokenize()

    def test_multiline(self):
        src = '''model "llama" {
  weights quantized int4
  kv_cache turboquant bits=3
  attention flash
}'''
        tokens = Lexer(src).tokenize()
        assert any(t.value == "llama" for t in tokens)


# ── Parser tests ─────────────────────────────────────────────

class TestParser:
    def test_basic_parse(self):
        src = 'model "gpt2" { weights full fp16 }'
        tokens = Lexer(src).tokenize()
        program = Parser(tokens).parse()
        assert len(program.models) == 1
        assert program.models[0].name == "gpt2"

    def test_directives(self):
        src = '''model "test" {
  weights quantized int8
  kv_cache turboquant bits=4
  attention flash streaming=true
  decode autoregressive
  scheduler continuous_batching max_batch=32
  target gpu "cuda"
}'''
        tokens = Lexer(src).tokenize()
        program = Parser(tokens).parse()
        model = program.models[0]
        assert len(model.directives) == 6

        weights = model.get_directive("weights")
        assert weights.positional == ["quantized", "int8"]

        kv = model.get_directive("kv_cache")
        assert kv.positional == ["turboquant"]
        assert kv.named["bits"] == 4

        attn = model.get_directive("attention")
        assert attn.named["streaming"] is True

    def test_multiple_models(self):
        src = '''
model "model-a" { weights full fp32 }
model "model-b" { weights quantized int4 }
'''
        tokens = Lexer(src).tokenize()
        program = Parser(tokens).parse()
        assert len(program.models) == 2
        assert program.models[0].name == "model-a"
        assert program.models[1].name == "model-b"

    def test_missing_brace(self):
        src = 'model "broken" { weights full'
        tokens = Lexer(src).tokenize()
        with pytest.raises(ParseError):
            Parser(tokens).parse()

    def test_empty_model(self):
        src = 'model "empty" { }'
        tokens = Lexer(src).tokenize()
        program = Parser(tokens).parse()
        assert len(program.models[0].directives) == 0


# ── Validator tests ──────────────────────────────────────────

class TestValidator:
    def test_valid_model(self):
        src = '''model "gpt2" {
  weights quantized int8
  kv_cache standard
  attention flash
  decode autoregressive
  scheduler static max_batch=1
  target auto
}'''
        tokens = Lexer(src).tokenize()
        program = Parser(tokens).parse()
        warnings = Validator().validate(program)
        # Should pass without errors

    def test_empty_program(self):
        program = Program()
        with pytest.raises(ValidationError):
            Validator().validate(program)

    def test_duplicate_model_name(self):
        src = '''
model "same" { weights full fp16 }
model "same" { weights full fp32 }
'''
        tokens = Lexer(src).tokenize()
        program = Parser(tokens).parse()
        with pytest.raises(ValidationError, match="Duplicate model name"):
            Validator().validate(program)

    def test_invalid_strategy(self):
        src = 'model "test" { weights invalid_strategy fp16 }'
        tokens = Lexer(src).tokenize()
        program = Parser(tokens).parse()
        with pytest.raises(ValidationError, match="Invalid value"):
            Validator().validate(program)

    def test_invalid_named_param_type(self):
        src = 'model "test" { kv_cache standard bits=hello }'
        tokens = Lexer(src).tokenize()
        program = Parser(tokens).parse()
        with pytest.raises(ValidationError, match="expects integer"):
            Validator().validate(program)

    def test_warnings_for_empty_model(self):
        src = 'model "empty" { }'
        tokens = Lexer(src).tokenize()
        program = Parser(tokens).parse()
        warnings = Validator().validate(program)
        assert len(warnings) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
