"""Explicit serial execution plans for transactional, double-bank F32 KV.

Cache offsets are relative to one 64-byte-aligned arena base. Both banks are
reserved for the entire session. Prefill writes the inactive bank; decode writes
only the uncommitted suffix of the active bank. The final Commit action records
the proposed bank/length transition, which the executor applies only after logits
and its result/report have been produced successfully. No old cache is copied.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from types import MappingProxyType

from .model_config import ModelConfig
from .model_ir import ModelGraph, _integer, _keys, _load_json, _name
from .model_lowering import lower_model
from .planner.memory import MemoryRequest


ALIGNMENT = 64
SCHEMA_VERSION = 1


def _aligned(size):
    return (size + ALIGNMENT - 1) & -ALIGNMENT


def _bank(value):
    _integer(value, "bank")
    if value not in (0, 1):
        raise ValueError("KV bank must be 0 or 1")
    return value


def _strict_equal(actual, expected):
    """JSON equality must not equate booleans with integers or ignore fields."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(_strict_equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_strict_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


@dataclass(frozen=True)
class KVBuffer:
    name: str
    bank: int
    layer: int
    kind: str
    shape: tuple[int, int]
    offset: int
    size_bytes: int

    def __post_init__(self):
        _bank(self.bank)
        _integer(self.layer, "layer")
        if self.kind not in ("key", "value"):
            raise ValueError("KV buffer kind must be key or value")
        if self.name != f"bank{self.bank}.layers.{self.layer}.{self.kind}":
            raise ValueError("KV buffer name does not match its bank/layer/kind")
        if not isinstance(self.shape, tuple) or len(self.shape) != 2:
            raise ValueError("KV shape must be (capacity, kv_width)")
        for dimension in self.shape:
            _integer(dimension, "KV dimension", 1)
        _integer(self.offset, "KV offset")
        _integer(self.size_bytes, "KV size_bytes", 1)
        if self.offset % ALIGNMENT or self.size_bytes != self.shape[0] * self.shape[1] * 4:
            raise ValueError("KV buffer size or alignment is invalid")

    def to_dict(self):
        return {"name": self.name, "bank": self.bank, "layer": self.layer, "kind": self.kind,
                "shape": list(self.shape), "offset": self.offset, "size_bytes": self.size_bytes,
                "dtype": "f32", "alignment": ALIGNMENT}


@dataclass(frozen=True)
class KVCachePlan:
    config: ModelConfig
    capacity: int
    buffers: Mapping[str, KVBuffer] = field(init=False)
    bytes_per_bank: int = field(init=False)
    cache_bytes: int = field(init=False)
    arena_bytes: int = field(init=False)

    def __post_init__(self):
        if not isinstance(self.config, ModelConfig):
            raise ValueError("KV config must be a ModelConfig")
        _integer(self.capacity, "KV capacity", 1)
        if self.capacity > self.config.max_position_embeddings:
            raise ValueError("KV capacity exceeds the configured context")
        # Enforce the existing physical tensor/layer bound before expansion.
        self.config.required_tensor_shapes()
        width = self.config.num_key_value_heads * self.config.head_dim
        size = self.capacity * width * 4
        stride = _aligned(size)
        bank_bytes = self.config.num_hidden_layers * 2 * stride
        arena_bytes = 2 * bank_bytes
        if arena_bytes + ALIGNMENT - 1 > (1 << 63) - 1:
            raise ValueError("KV allocation exceeds the supported byte range")
        buffers = {}
        for bank in range(2):
            for layer in range(self.config.num_hidden_layers):
                for index, kind in enumerate(("key", "value")):
                    name = f"bank{bank}.layers.{layer}.{kind}"
                    offset = bank * bank_bytes + (2 * layer + index) * stride
                    buffers[name] = KVBuffer(name, bank, layer, kind, (self.capacity, width), offset, size)
        object.__setattr__(self, "buffers", MappingProxyType(buffers))
        object.__setattr__(self, "bytes_per_bank", bank_bytes)
        object.__setattr__(self, "cache_bytes", 4 * self.config.num_hidden_layers * size)
        object.__setattr__(self, "arena_bytes", arena_bytes)

    @property
    def allocation_bytes(self):
        """Maximum backing allocation, including one aligned-base adjustment."""
        return self.arena_bytes + ALIGNMENT - 1

    @property
    def bank_bytes(self):
        return self.bytes_per_bank

    @property
    def banks(self):
        return 2

    @property
    def kv_width(self):
        return self.config.num_key_value_heads * self.config.head_dim

    def buffer(self, bank, layer, kind):
        _bank(bank)
        _integer(layer, "layer")
        if layer >= self.config.num_hidden_layers or kind not in ("key", "value"):
            raise ValueError("Unknown KV layer or buffer kind")
        return self.buffers[f"bank{bank}.layers.{layer}.{kind}"]

    def persistent_requests(self, end):
        """Account for both banks across a step when using a shared arena.

        The dedicated-cache executor instead reserves allocation_bytes outside
        its activation arena; callers must not count both alternatives together.
        These requests are accounting sizes, not replacements for cache offsets.
        """
        _integer(end, "persistent lifetime end", 1)
        return tuple(MemoryRequest(f"__kv_bank{bank}", self.bytes_per_bank, 0, end, ALIGNMENT)
                     for bank in range(2))

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, "config": self.config.to_dict(),
                "capacity": self.capacity, "banks": self.banks, "dtype": "f32",
                "alignment": ALIGNMENT, "kv_width": self.kv_width,
                "bytes_per_bank": self.bytes_per_bank, "cache_bytes": self.cache_bytes,
                "arena_bytes": self.arena_bytes, "allocation_bytes": self.allocation_bytes,
                "buffers": {name: buffer.to_dict() for name, buffer in self.buffers.items()}}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {"schema_version", "config", "capacity", "banks", "dtype", "alignment",
                     "kv_width", "bytes_per_bank", "cache_bytes", "arena_bytes", "allocation_bytes",
                     "buffers"}, "KVCachePlan")
        plan = cls(ModelConfig.from_dict(data["config"]), data["capacity"])
        if not _strict_equal(data, plan.to_dict()):
            raise ValueError("KVCachePlan metadata, offsets, sizes or version differ from the canonical layout")
        return plan

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(_load_json(text))


_CACHE_BINDINGS = {"layer", "bank", "key_input", "value_input", "key_offset", "value_offset",
                   "key_write_offset", "value_write_offset", "write_bytes", "past_length",
                   "new_length", "query_length", "kv_width"}


@dataclass(frozen=True)
class KVAction:
    kind: str
    op_name: str | None
    bindings: Mapping = field(default_factory=dict)

    def __post_init__(self):
        fields = {"Compute": set(), "RoPE": {"position_offset"},
                  "CacheWrite": _CACHE_BINDINGS, "CachedAttention": _CACHE_BINDINGS,
                  "Commit": {"target_bank", "new_length"}}
        if self.kind not in fields or not isinstance(self.bindings, Mapping) or set(self.bindings) != fields[self.kind]:
            raise ValueError("Invalid KV action kind or bindings")
        if self.kind == "Commit":
            if self.op_name is not None:
                raise ValueError("Commit must not reference a graph operation")
        else:
            _name(self.op_name, "operation name")
        bindings = dict(self.bindings)
        for name, value in bindings.items():
            if name in ("key_input", "value_input"):
                _name(value, name)
            else:
                _integer(value, name, 1 if name in ("write_bytes", "new_length", "query_length", "kv_width") else 0)
        if "bank" in bindings:
            _bank(bindings["bank"])
        if "target_bank" in bindings:
            _bank(bindings["target_bank"])
        object.__setattr__(self, "bindings", MappingProxyType(bindings))

    def to_dict(self):
        return {"kind": self.kind, "op_name": self.op_name, "bindings": dict(self.bindings)}


def _step_lifetimes(graph, steps):
    constants = set(graph.constants)
    operations = {op.name: op for op in graph.ops}
    starts = {name: 0 for name in graph.inputs}
    ends = {name: 1 for name in graph.inputs}
    for event, step in enumerate(steps, 1):
        if step.kind == "Commit":
            continue
        op = operations[step.op_name]
        if step.kind == "CacheWrite":
            inputs = (step.bindings["key_input"], step.bindings["value_input"])
            outputs = ()
        else:
            inputs = op.inputs[:1] if step.kind == "CachedAttention" else op.inputs
            outputs = op.outputs
        for name in inputs:
            if name not in constants:
                ends[name] = max(ends[name], event + 1)
        for name in outputs:
            starts[name] = event
            ends[name] = event + 1
    for name in graph.outputs:
        if name not in constants:
            ends[name] = len(steps) + 2
    return tuple(MemoryRequest(t.name, t.storage_nbytes, starts[t.name], ends[t.name],
                               t.alignment, t.tier.value)
                 for t in graph.tensors if t.name not in constants)


@dataclass(frozen=True)
class KVStepPlan:
    graph: ModelGraph
    cache_plan: KVCachePlan
    past_length: int
    mode: str = "decode"
    active_bank: int = 0
    target_bank: int = field(init=False)
    position_offset: int = field(init=False)
    new_length: int = field(init=False)
    steps: tuple[KVAction, ...] = field(init=False)
    activation_requests: tuple[MemoryRequest, ...] = field(init=False)

    def __post_init__(self):
        if not isinstance(self.graph, ModelGraph) or not isinstance(self.cache_plan, KVCachePlan):
            raise ValueError("KV step requires a ModelGraph and KVCachePlan")
        if self.mode not in ("prefill", "decode"):
            raise ValueError("KV mode must be prefill or decode")
        _bank(self.active_bank)
        _integer(self.past_length, "past_length")
        if self.past_length > self.cache_plan.capacity or self.mode == "decode" and self.past_length == 0:
            raise ValueError("past_length is outside the valid committed cache range")
        self.graph.validate()
        tensors = {tensor.name: tensor for tensor in self.graph.tensors}
        if self.graph.inputs != ("tokens",) or len(tensors["tokens"].shape) != 1:
            raise ValueError("KV steps require the canonical token input")
        count = tensors["tokens"].shape[0]
        position = 0 if self.mode == "prefill" else self.past_length
        total = position + count
        if total > self.cache_plan.capacity:
            raise ValueError("KV step exceeds cache capacity")
        expected = lower_model(self.cache_plan.config, count,
                               weight_storage={name: tensors[name] for name in self.graph.constants})
        if self.graph.to_dict() != expected.to_dict():
            raise ValueError("KV steps require the canonical lower_model graph and matching config")
        target = 1 - self.active_bank if self.mode == "prefill" else self.active_bank
        width = self.cache_plan.kv_width
        layers = {f"layers.{layer}.attention": layer for layer in range(self.cache_plan.config.num_hidden_layers)}
        steps = []
        for op in self.graph.ops:
            if op.kind.value == "RoPE":
                steps.append(KVAction("RoPE", op.name, {"position_offset": position}))
            elif op.kind.value == "CausalAttention":
                layer = layers[op.name]
                key, value = (self.cache_plan.buffer(target, layer, kind) for kind in ("key", "value"))
                binding = {"layer": layer, "bank": target, "key_input": op.inputs[1], "value_input": op.inputs[2],
                           "key_offset": key.offset, "value_offset": value.offset,
                           "key_write_offset": key.offset + position * width * 4,
                           "value_write_offset": value.offset + position * width * 4,
                           "write_bytes": count * width * 4, "past_length": position,
                           "new_length": total, "query_length": count, "kv_width": width}
                steps.append(KVAction("CacheWrite", op.name, binding))
                steps.append(KVAction("CachedAttention", op.name, binding))
            else:
                steps.append(KVAction("Compute", op.name))
        steps.append(KVAction("Commit", None, {"target_bank": target, "new_length": total}))
        object.__setattr__(self, "target_bank", target)
        object.__setattr__(self, "position_offset", position)
        object.__setattr__(self, "new_length", total)
        object.__setattr__(self, "steps", tuple(steps))
        object.__setattr__(self, "activation_requests", _step_lifetimes(self.graph, steps))

    @property
    def bindings(self):
        return MappingProxyType({step.op_name: step.bindings for step in self.steps if step.kind == "CachedAttention"})

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, "graph": self.graph.to_dict(),
                "cache_plan": self.cache_plan.to_dict(), "past_length": self.past_length,
                "mode": self.mode, "active_bank": self.active_bank, "target_bank": self.target_bank,
                "position_offset": self.position_offset, "new_length": self.new_length,
                "steps": [step.to_dict() for step in self.steps],
                "activation_requests": [request.to_dict() for request in self.activation_requests]}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {"schema_version", "graph", "cache_plan", "past_length", "mode", "active_bank",
                     "target_bank", "position_offset", "new_length", "steps", "activation_requests"}, "KVStepPlan")
        plan = cls(ModelGraph.from_dict(data["graph"]), KVCachePlan.from_dict(data["cache_plan"]),
                   data["past_length"], data["mode"], data["active_bank"])
        if not _strict_equal(data, plan.to_dict()):
            raise ValueError("KV step actions, bindings, lifetimes or version differ from the validated execution plan")
        return plan

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(_load_json(text))


def make_kv_cache_plan(config: ModelConfig, capacity: int) -> KVCachePlan:
    return KVCachePlan(config, capacity)


def plan_step(graph: ModelGraph, cache_plan: KVCachePlan, past_length: int,
              *, mode="decode", active_bank=0) -> KVStepPlan:
    return KVStepPlan(graph, cache_plan, past_length, mode, active_bank)


def make_step_plan(cache_plan: KVCachePlan, graph: ModelGraph, *, mode, past_length, active_bank=0) -> KVStepPlan:
    """Cache-first spelling; append is accepted as an alias of decode."""
    return plan_step(graph, cache_plan, past_length, mode="decode" if mode == "append" else mode,
                     active_bank=active_bank)
