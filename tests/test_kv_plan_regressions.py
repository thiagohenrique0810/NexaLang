"""Explicit cache state transitions, persistent storage and copy-aware liveness."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compiler.kv_plan import KVCachePlan, KVStepPlan, make_kv_cache_plan, make_step_plan, plan_step
from compiler.model_config import ModelConfig
from compiler.model_lowering import lower_model
from compiler.planner.memory import MemoryBudgetError, MemoryPlanner


def tiny(**changes):
    return ModelConfig(**{"name": "TinyKV", "vocab_size": 16, "hidden_size": 8,
                          "intermediate_size": 12, "num_hidden_layers": 2,
                          "num_attention_heads": 2, "num_key_value_heads": 1,
                          "max_position_embeddings": 16, **changes})


class KVCacheLayoutRegressions(unittest.TestCase):
    def test_both_banks_have_aligned_nonoverlapping_layer_buffers(self):
        cache = make_kv_cache_plan(tiny(), 3)
        self.assertEqual((cache.kv_width, cache.banks), (4, 2))
        self.assertEqual((cache.bytes_per_bank, cache.cache_bytes, cache.arena_bytes,
                          cache.allocation_bytes), (256, 384, 512, 575))
        buffers = list(cache.buffers.values())
        self.assertEqual(len(buffers), 8)
        for index, buffer in enumerate(buffers):
            self.assertEqual(buffer.offset, index * 64)
            self.assertEqual((buffer.shape, buffer.size_bytes), ((3, 4), 48))
            self.assertLessEqual(buffer.offset + buffer.size_bytes, cache.arena_bytes)
            self.assertIs(cache.buffer(buffer.bank, buffer.layer, buffer.kind), buffer)
        requests = cache.persistent_requests(40)
        self.assertEqual(sum(r.size_bytes for r in requests), cache.arena_bytes)
        self.assertTrue(all((r.start, r.end, r.alignment) == (0, 40, 64) for r in requests))

    def test_gqa_reserves_only_kv_heads_and_all_layers(self):
        gqa = make_kv_cache_plan(tiny(), 16)
        mha = make_kv_cache_plan(tiny(num_key_value_heads=2), 16)
        self.assertEqual(mha.cache_bytes, 2 * gqa.cache_bytes)
        self.assertEqual(gqa.cache_bytes, 2 * 2 * 2 * 16 * 4 * 4)
        self.assertEqual(make_kv_cache_plan(tiny(num_hidden_layers=1), 16).cache_bytes,
                         gqa.cache_bytes // 2)

    def test_invalid_capacity_dimensions_and_indices_fail_before_expansion(self):
        for capacity in (0, -1, True, 1.5, 17):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                make_kv_cache_plan(tiny(), capacity)
        with self.assertRaisesRegex(ValueError, "physical tensors"):
            make_kv_cache_plan(tiny(num_hidden_layers=10 ** 15), 1)
        with self.assertRaisesRegex(ValueError, "byte range"):
            make_kv_cache_plan(tiny(max_position_embeddings=10 ** 30), 10 ** 30)
        cache = make_kv_cache_plan(tiny(), 3)
        for arguments in ((True, 0, "key"), (2, 0, "key"), (0, -1, "key"),
                          (0, True, "key"), (0, 2, "key"), (0, 0, "keys")):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                cache.buffer(*arguments)

    def test_serialization_rejects_layout_tampering_and_boolean_numbers(self):
        cache = make_kv_cache_plan(tiny(), 3)
        self.assertEqual(KVCachePlan.from_json(cache.to_json()).to_dict(), cache.to_dict())
        mutations = [lambda d: d.update(schema_version=True), lambda d: d.update(banks=1),
                     lambda d: d.update(allocation_bytes=d["arena_bytes"]),
                     lambda d: d["buffers"]["bank0.layers.0.key"].update(offset=64),
                     lambda d: d["buffers"]["bank0.layers.0.key"].update(size_bytes=64),
                     lambda d: d["buffers"]["bank0.layers.0.key"].update(bank=False),
                     lambda d: d["buffers"].pop("bank1.layers.1.value"),
                     lambda d: d.update(unrecognized=1)]
        for mutate in mutations:
            data = deepcopy(cache.to_dict())
            mutate(data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                KVCachePlan.from_dict(data)
        with self.assertRaises(ValueError):
            KVCachePlan.from_json('{"capacity": 3, "capacity": 4}')


class KVExecutionPlanRegressions(unittest.TestCase):
    def test_prefill_replaces_inactive_bank_and_decode_appends_at_committed_position(self):
        config = tiny()
        cache = make_kv_cache_plan(config, 8)
        for active_bank in (0, 1):
            prefill = plan_step(lower_model(config, 3), cache, 7, mode="prefill", active_bank=active_bank)
            self.assertEqual((prefill.position_offset, prefill.new_length, prefill.target_bank),
                             (0, 3, 1 - active_bank))
            decode = plan_step(lower_model(config, 2), cache, 3, active_bank=active_bank)
            self.assertEqual((decode.position_offset, decode.new_length, decode.target_bank),
                             (3, 5, active_bank))
            for binding in decode.bindings.values():
                self.assertEqual(binding["write_bytes"], 2 * 4 * 4)
                self.assertEqual(binding["key_write_offset"] - binding["key_offset"], 3 * 4 * 4)
                self.assertEqual(binding["value_write_offset"] - binding["value_offset"], 3 * 4 * 4)
                self.assertEqual((binding["past_length"], binding["new_length"], binding["query_length"]),
                                 (3, 5, 2))
        alias = make_step_plan(cache, lower_model(config, 2), mode="append", past_length=3)
        self.assertEqual(alias.to_dict(), plan_step(lower_model(config, 2), cache, 3).to_dict())

    def test_steps_put_copy_before_attention_and_commit_after_logits(self):
        config = tiny(num_hidden_layers=3)
        step = plan_step(lower_model(config, 1), make_kv_cache_plan(config, 8), 3)
        scheduled = [action.op_name for action in step.steps if action.kind not in ("CacheWrite", "Commit")]
        self.assertEqual(scheduled, [op.name for op in step.graph.ops])
        self.assertEqual(step.steps[-2].op_name, "logits")
        self.assertEqual(step.steps[-1].kind, "Commit")
        self.assertEqual(dict(step.steps[-1].bindings), {"target_bank": 0, "new_length": 4})
        count = 0
        for index, action in enumerate(step.steps):
            if action.kind == "RoPE":
                self.assertEqual(dict(action.bindings), {"position_offset": 3})
            if action.kind == "CachedAttention":
                copy = step.steps[index - 1]
                self.assertEqual(copy.kind, "CacheWrite")
                self.assertEqual((copy.op_name, dict(copy.bindings)), (action.op_name, dict(action.bindings)))
                self.assertEqual(action.bindings["layer"], count)
                self.assertEqual(action.bindings["key_input"], f"layers.{count}.k_rope")
                self.assertEqual(action.bindings["value_input"], f"layers.{count}.v")
                count += 1
        self.assertEqual(count, 3)

    def test_capacity_boundary_and_invalid_state(self):
        config = tiny()
        cache = make_kv_cache_plan(config, 4)
        graph = lower_model(config, 1)
        self.assertEqual(plan_step(graph, cache, 3).new_length, 4)
        self.assertEqual(plan_step(lower_model(config, 4), cache, 0, mode="prefill").new_length, 4)
        for parameters in ({"past_length": 4}, {"past_length": 0}, {"past_length": -1},
                           {"past_length": True}, {"past_length": 1.5}, {"past_length": 1, "active_bank": True},
                           {"past_length": 1, "active_bank": 2}, {"past_length": 1, "mode": "append"}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                plan_step(graph, cache, **parameters)
        with self.assertRaises(ValueError):
            plan_step(lower_model(config, 5), cache, 0, mode="prefill")

    def test_valid_but_noncanonical_graphs_cannot_bind_the_wrong_layer_state(self):
        config = tiny()
        cache = make_kv_cache_plan(config, 8)
        graph = lower_model(config, 1)
        changed_ops = tuple(replace(op, attributes={**op.attributes, "theta": 5000})
                            if op.kind.value == "RoPE" else op for op in graph.ops)
        altered_inputs = tuple(replace(op, inputs=(op.inputs[0], "layers.0.k_rope", "layers.0.v"))
                               if op.name == "layers.1.attention" else op for op in graph.ops)
        for changed in (replace(graph, ops=changed_ops), replace(graph, ops=altered_inputs),
                        lower_model(tiny(rms_norm_eps=0.01), 1), lower_model(tiny(num_hidden_layers=1), 1)):
            with self.subTest(graph=changed.name), self.assertRaises(ValueError):
                plan_step(changed, cache, 1)

    def test_step_json_validates_schedule_copies_commit_and_lifetimes(self):
        config = tiny()
        step = plan_step(lower_model(config, 1), make_kv_cache_plan(config, 8), 3)
        self.assertEqual(KVStepPlan.from_json(step.to_json()).to_dict(), step.to_dict())
        copy_index = next(i for i, action in enumerate(step.steps) if action.kind == "CacheWrite")
        mutations = [lambda d: d.update(new_length=5), lambda d: d.update(active_bank=False),
                     lambda d: d["steps"].pop(), lambda d: d["steps"].reverse(),
                     lambda d: d["steps"][copy_index]["bindings"].update(key_write_offset=0),
                     lambda d: d["steps"][copy_index]["bindings"].update(value_input="layers.0.k_rope"),
                     lambda d: d["activation_requests"][0].update(end=1),
                     lambda d: d["steps"][-1]["bindings"].update(target_bank=True)]
        for mutate in mutations:
            data = deepcopy(step.to_dict())
            mutate(data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                KVStepPlan.from_dict(data)

    def test_plans_are_immutable_and_exported_dictionaries_are_detached(self):
        config = tiny()
        step = plan_step(lower_model(config, 1), make_kv_cache_plan(config, 8), 3)
        with self.assertRaises(FrozenInstanceError):
            step.target_bank = 1
        with self.assertRaises(TypeError):
            step.cache_plan.buffers["other"] = step.cache_plan.buffer(0, 0, "key")
        with self.assertRaises(TypeError):
            step.bindings["layers.0.attention"]["key_offset"] = 123
        serialized = step.to_dict()
        serialized["steps"][-1]["bindings"]["new_length"] = 7
        self.assertEqual(step.steps[-1].bindings["new_length"], 4)


class KVStorageLifetimeRegressions(unittest.TestCase):
    def test_copy_releases_local_kv_but_query_and_result_cannot_overlap(self):
        config = tiny()
        step = plan_step(lower_model(config, 2), make_kv_cache_plan(config, 8), 3)
        requests = {request.name: request for request in step.activation_requests}
        self.assertEqual(set(requests), {t.name for t in step.graph.tensors} - set(step.graph.constants))
        for event, action in enumerate(step.steps, 1):
            if action.kind != "CachedAttention":
                continue
            layer = action.bindings["layer"]
            self.assertEqual(requests[f"layers.{layer}.k_rope"].end, event)
            self.assertEqual(requests[f"layers.{layer}.v"].end, event)
            self.assertEqual(requests[f"layers.{layer}.q_rope"].end, event + 1)
            self.assertEqual(requests[f"layers.{layer}.attention"].start, event)
        self.assertEqual(requests["logits"].end, len(step.steps) + 2)

    def test_actual_arena_reuse_preserves_each_operand_and_cached_copy(self):
        config = tiny(num_hidden_layers=3)
        step = plan_step(lower_model(config, 2), make_kv_cache_plan(config, 8), 3)
        plan = MemoryPlanner().plan(step.activation_requests, {"host": 1 << 20})
        arena = bytearray(plan.peak_bytes["host"])
        cache = bytearray(step.cache_plan.arena_bytes)
        expected, operations = {}, {op.name: op for op in step.graph.ops}

        def span(name):
            a = plan.allocations[name]
            return slice(a.offset, a.offset + a.size_bytes)

        def write(name, tag):
            expected[name] = bytes([tag]) * plan.allocations[name].size_bytes
            arena[span(name)] = expected[name]

        write("tokens", 1)
        for event, action in enumerate(step.steps, 2):
            if action.kind == "Commit":
                continue
            op = operations[action.op_name]
            if action.kind == "CacheWrite":
                for kind in ("key", "value"):
                    name, offset = action.bindings[kind + "_input"], action.bindings[kind + "_write_offset"]
                    self.assertEqual(arena[span(name)], expected[name])
                    cache[offset:offset + action.bindings["write_bytes"]] = arena[span(name)]
                continue
            inputs = op.inputs[:1] if action.kind == "CachedAttention" else op.inputs
            for name in inputs:
                if name not in step.graph.constants:
                    self.assertEqual(arena[span(name)], expected[name], f"{action.op_name} lost {name}")
            for name in op.outputs:
                write(name, event)
            if action.kind == "CachedAttention":
                for kind in ("key", "value"):
                    offset = action.bindings[kind + "_write_offset"]
                    self.assertEqual(cache[offset:offset + action.bindings["write_bytes"]],
                                     expected[action.bindings[kind + "_input"]])
        self.assertEqual(arena[span("logits")], expected["logits"])
        self.assertLess(plan.peak_bytes["host"], sum(r.size_bytes for r in step.activation_requests))

    def test_failed_partial_writes_preserve_committed_prefix_and_prefill_active_bank(self):
        config = tiny()
        cache_plan = make_kv_cache_plan(config, 8)
        for mode in ("prefill", "decode"):
            step = plan_step(lower_model(config, 2), cache_plan, 3, mode=mode, active_bank=1)
            arena = bytearray([99]) * cache_plan.arena_bytes
            action = next(s for s in step.steps if s.kind == "CacheWrite")
            for kind in ("key", "value"):
                start, size = action.bindings[kind + "_write_offset"], action.bindings["write_bytes"]
                arena[start:start + size] = bytes([7]) * size
            # Abort before layer 1: only uncommitted storage may have changed.
            for layer in range(config.num_hidden_layers):
                for kind in ("key", "value"):
                    allocation = cache_plan.buffer(1, layer, kind)
                    retained = allocation.size_bytes if mode == "prefill" else 3 * cache_plan.kv_width * 4
                    self.assertEqual(arena[allocation.offset:allocation.offset + retained], bytes([99]) * retained)

    def test_budget_includes_both_cache_banks_before_an_arena_is_created(self):
        config = tiny()
        step = plan_step(lower_model(config, 1), make_kv_cache_plan(config, 8), 3)
        planner = MemoryPlanner()
        workspace = planner.plan(step.activation_requests, {"host": 1 << 20})
        required = workspace.peak_bytes["host"] + step.cache_plan.allocation_bytes
        complete = planner.plan(step.activation_requests, {"host": required},
                                {"host": step.cache_plan.allocation_bytes})
        self.assertEqual(complete.peak_bytes, workspace.peak_bytes)
        with self.assertRaises(MemoryBudgetError):
            planner.plan(step.activation_requests, {"host": required - 1},
                         {"host": step.cache_plan.allocation_bytes})


if __name__ == "__main__":
    unittest.main()
