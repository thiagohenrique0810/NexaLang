"""Tests for TurboIR — IR generation and execution planning."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from turbolang.parser.lexer import Lexer
from turbolang.parser.parser import Parser
from turboir.ir.nodes import IRGenerator, QuantType, CacheStrategy, DecodeMode
from turboir.planner.planner import ExecutionPlanner


class TestIRGenerator:
    def _parse_and_generate(self, src: str):
        tokens = Lexer(src).tokenize()
        ast = Parser(tokens).parse()
        return IRGenerator().generate(ast)

    def test_basic_ir(self):
        src = 'model "gpt2" { weights full fp16 }'
        ir = self._parse_and_generate(src)
        assert len(ir.models) == 1
        assert ir.models[0].name == "gpt2"
        assert ir.models[0].weights.quant_type == QuantType.FP16

    def test_quantized_weights(self):
        src = 'model "test" { weights quantized int4 }'
        ir = self._parse_and_generate(src)
        assert ir.models[0].weights.quant_type == QuantType.INT4
        assert ir.models[0].weights.strategy == "quantized"

    def test_turboquant_cache(self):
        src = 'model "test" { kv_cache turboquant bits=3 }'
        ir = self._parse_and_generate(src)
        assert ir.models[0].cache.strategy == CacheStrategy.TURBOQUANT
        assert ir.models[0].cache.bits == 3

    def test_speculative_decode(self):
        src = 'model "test" { decode speculative drafter="tiny" k=4 }'
        ir = self._parse_and_generate(src)
        assert ir.models[0].decode.mode == DecodeMode.SPECULATIVE
        assert ir.models[0].decode.drafter == "tiny"
        assert ir.models[0].decode.k == 4

    def test_full_config(self):
        src = '''model "llama3-8b" {
  weights quantized int4
  kv_cache turboquant bits=3
  attention flash streaming=true
  decode speculative drafter="tiny-draft"
  scheduler continuous_batching max_batch=32 max_ctx=8192
  target gpu "cuda"
}'''
        ir = self._parse_and_generate(src)
        m = ir.models[0]
        assert m.name == "llama3-8b"
        assert m.weights.quant_type == QuantType.INT4
        assert m.cache.strategy == CacheStrategy.TURBOQUANT
        assert m.cache.bits == 3
        assert m.attention.streaming is True
        assert m.decode.mode == DecodeMode.SPECULATIVE
        assert m.scheduler.max_batch == 32
        assert m.scheduler.max_ctx == 8192
        assert m.target.backend == "cuda"

    def test_to_dict(self):
        src = 'model "gpt2" { weights quantized int8 }'
        ir = self._parse_and_generate(src)
        d = ir.models[0].to_dict()
        assert d['name'] == 'gpt2'
        assert d['weights']['quant_type'] == 'int8'


class TestExecutionPlanner:
    def _plan(self, src: str):
        tokens = Lexer(src).tokenize()
        ast = Parser(tokens).parse()
        ir = IRGenerator().generate(ast)
        return ExecutionPlanner().plan(ir)

    def test_basic_plan(self):
        src = 'model "gpt2" { weights full fp16 target cpu }'
        plans = self._plan(src)
        assert len(plans) == 1
        plan = plans[0]
        assert plan.model_name == "gpt2"

        step_names = [s.name for s in plan.steps]
        assert "load_model" in step_names
        assert "init_kv_cache" in step_names
        assert "ready" in step_names

    def test_quantized_plan_has_quant_step(self):
        src = 'model "gpt2" { weights quantized int8 target cpu }'
        plans = self._plan(src)
        step_names = [s.name for s in plans[0].steps]
        assert "quantize_weights" in step_names

    def test_no_quant_step_for_full(self):
        src = 'model "gpt2" { weights full fp32 target cpu }'
        plans = self._plan(src)
        step_names = [s.name for s in plans[0].steps]
        assert "quantize_weights" not in step_names

    def test_continuous_batching_has_scheduler_step(self):
        src = 'model "gpt2" { scheduler continuous_batching max_batch=16 target cpu }'
        plans = self._plan(src)
        step_names = [s.name for s in plans[0].steps]
        assert "init_scheduler" in step_names

    def test_speculative_has_init_step(self):
        src = 'model "gpt2" { decode speculative drafter="tiny" target cpu }'
        plans = self._plan(src)
        step_names = [s.name for s in plans[0].steps]
        assert "init_speculative" in step_names

    def test_step_ordering(self):
        src = '''model "gpt2" {
  weights quantized int8
  scheduler continuous_batching max_batch=4
  decode speculative drafter="tiny"
  target cpu
}'''
        plans = self._plan(src)
        steps = plans[0].steps
        orders = {s.name: s.order for s in steps}
        assert orders["load_model"] < orders["quantize_weights"]
        assert orders["quantize_weights"] < orders["init_kv_cache"]
        assert orders["init_kv_cache"] < orders["ready"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
