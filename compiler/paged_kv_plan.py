"""Explicit F32/Q4/Q3/TQ KV layouts and transactional serial execution schedules.

One physical page contains every layer's key/value buffers. Page indices address
the session's logical token order, never a physical pointer. The executor owns
the page table and allocations: prefill stages fresh pages while the committed
pages remain readable; decode only writes the uncommitted suffix. Commit is the
last action and must be published after output/report creation succeeds.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from types import MappingProxyType

from .kv_plan import ALIGNMENT, _aligned, _step_lifetimes, _strict_equal
from .model_config import ModelConfig
from .model_ir import ModelGraph, _integer, _keys, _load_json, _name
from .model_lowering import lower_model
from .planner.memory import MemoryRequest
from runtime.nexapack.tq import (
    TQ_CODEC_ID, TQ_CODEC_VERSION, TQ_TRANSFORM_ID,
    tq_row_bytes, validate_tq_codebook, validate_tq_parameters,
)


SCHEMA_VERSION = 1
MAX_PLAN_SEGMENTS = 65536
MAX_ALLOCATION_BYTES = (1 << 63) - 1
MAX_Q4_GROUP_SIZE = 1 << 20
MAX_Q3_GROUP_SIZE = MAX_Q4_GROUP_SIZE


def _packed_head_row_bytes(head_dim, group_size, codec):
    _integer(head_dim, "head_dim", 1)
    _integer(group_size, "group_size", 1)
    if group_size > MAX_Q4_GROUP_SIZE:
        raise ValueError(f"Packed KV group_size exceeds {MAX_Q4_GROUP_SIZE}")
    # Q8 stores whole bytes; the bit-packed codecs share the rounded formula.
    payload = group_size if codec == "q8" else ({"q4": 4, "q3": 3}[codec] * group_size + 7) // 8
    return ((head_dim + group_size - 1) // group_size) * (4 + payload)


_PACKED_LAYOUT_FIELDS = {"codec", "codec_id", "codec_version", "layout", "logical_dtype", "group_size",
                         "head_dim", "head_row_bytes", "token_bytes"}
_TQ_LAYOUT_FIELDS = (_PACKED_LAYOUT_FIELDS - {"group_size"}) | {
    "bits", "seed", "transform_id", "codebook_f32le"}


@dataclass(frozen=True)
class PagedKVCachePlan:
    config: ModelConfig
    capacity: int
    page_tokens: int
    codec: str = "f32"
    group_size: int | None = None
    bits: int | None = None
    seed: int | None = None
    codebook_f32le: str | None = None

    def __post_init__(self):
        if not isinstance(self.config, ModelConfig):
            raise ValueError("Paged KV config must be a ModelConfig")
        _integer(self.capacity, "KV capacity", 1)
        _integer(self.page_tokens, "page_tokens", 1)
        if self.codec not in ("f32", "q4", "q3", "q8", "tq"):
            raise ValueError("Paged KV codec must be f32, q4, q3, q8 or tq")
        if self.codec != "tq" and any(value is not None for value in
                                      (self.bits, self.seed, self.codebook_f32le)):
            raise ValueError("bits, seed and codebook_f32le require the TQ KV codec")
        if self.codec == "tq":
            if self.group_size is not None:
                raise ValueError("TQ KV does not accept group_size")
            bits = 3 if self.bits is None else self.bits
            seed = 42 if self.seed is None else self.seed
            validate_tq_parameters(self.config.head_dim, bits, seed)
            if self.codebook_f32le is not None:
                validate_tq_codebook(self.codebook_f32le, bits)
            object.__setattr__(self, "bits", bits)
            object.__setattr__(self, "seed", seed)
        elif self.codec == "f32":
            if self.group_size is not None:
                raise ValueError("F32 KV does not accept group_size")
        else:
            group_size = 32 if self.group_size is None else self.group_size
            _packed_head_row_bytes(self.config.head_dim, group_size, self.codec)
            object.__setattr__(self, "group_size", group_size)
        if self.capacity > self.config.max_position_embeddings:
            raise ValueError("KV capacity exceeds the configured context")
        self.config.required_tensor_shapes()
        if self.max_pages * self.page_allocation_bytes > MAX_ALLOCATION_BYTES:
            raise ValueError("Paged KV capacity exceeds the supported allocation byte range")

    @property
    def kv_width(self):
        return self.config.num_key_value_heads * self.config.head_dim

    @property
    def head_row_bytes(self):
        if self.codec == "f32":
            return self.config.head_dim * 4
        if self.codec == "tq":
            return tq_row_bytes(self.config.head_dim, self.bits)
        return _packed_head_row_bytes(self.config.head_dim, self.group_size, self.codec)

    @property
    def token_bytes(self):
        return self.config.num_key_value_heads * self.head_row_bytes

    @property
    def max_pages(self):
        return (self.capacity + self.page_tokens - 1) // self.page_tokens

    @property
    def page_payload_bytes(self):
        return 2 * self.config.num_hidden_layers * self.page_tokens * self.token_bytes

    @property
    def buffer_stride(self):
        return _aligned(self.page_tokens * self.token_bytes)

    @property
    def page_extent_bytes(self):
        return 2 * self.config.num_hidden_layers * self.buffer_stride

    @property
    def page_allocation_bytes(self):
        """Backing allocation per page, including its aligned-base adjustment."""
        return self.page_extent_bytes + ALIGNMENT - 1

    def page_count(self, length):
        _integer(length, "KV length")
        if length > self.capacity:
            raise ValueError("KV length exceeds cache capacity")
        return (length + self.page_tokens - 1) // self.page_tokens

    def buffer_offset(self, layer, kind):
        _integer(layer, "layer")
        if layer >= self.config.num_hidden_layers or kind not in ("key", "value"):
            raise ValueError("Unknown KV layer or buffer kind")
        return (2 * layer + (kind == "value")) * self.buffer_stride

    def head_offset(self, layer, kind, head):
        """Head start in token zero; subsequent tokens are token_bytes apart.

        Packed rows contain groups or TQ02 vectors without extra head padding.
        Their F32 fields may be unaligned; readers use the byte-oriented codec.
        """
        _integer(head, "KV head")
        if head >= self.config.num_key_value_heads:
            raise ValueError("KV head exceeds num_key_value_heads")
        return self.buffer_offset(layer, kind) + head * self.head_row_bytes

    def reservation_pages(self, max_chunk_length):
        _integer(max_chunk_length, "max_chunk_length", 1)
        if max_chunk_length > self.capacity:
            raise ValueError("Chunk capacity exceeds cache capacity")
        # A decode may begin partway through a page. Admit the largest span
        # allowed by both chunk and context capacities, not just fresh prefill.
        max_start_offset = min(self.page_tokens - 1, self.capacity - max_chunk_length)
        max_segments = (max_chunk_length + max_start_offset + self.page_tokens - 1) // self.page_tokens
        if max_segments > MAX_PLAN_SEGMENTS:
            raise ValueError(f"Chunk capacity exceeds the {MAX_PLAN_SEGMENTS} segment metadata limit")
        return self.max_pages + self.page_count(max_chunk_length)

    def allocation_limit_bytes(self, max_chunk_length):
        size = self.reservation_pages(max_chunk_length) * self.page_allocation_bytes
        if size > MAX_ALLOCATION_BYTES:
            raise ValueError("Paged KV reservation exceeds the supported allocation byte range")
        return size

    def to_dict(self):
        buffers = {}
        for layer in range(self.config.num_hidden_layers):
            for kind in ("key", "value"):
                buffers[f"layers.{layer}.{kind}"] = {
                    "layer": layer, "kind": kind, "shape": [self.page_tokens, self.kv_width],
                    "offset": self.buffer_offset(layer, kind),
                    "size_bytes": self.page_tokens * self.token_bytes}
        result = {"schema_version": SCHEMA_VERSION, "config": self.config.to_dict(),
                "capacity": self.capacity, "page_tokens": self.page_tokens, "dtype": self.codec,
                "alignment": ALIGNMENT, "kv_width": self.kv_width, "max_pages": self.max_pages,
                "page_payload_bytes": self.page_payload_bytes, "buffer_stride": self.buffer_stride,
                "page_extent_bytes": self.page_extent_bytes,
                "page_allocation_bytes": self.page_allocation_bytes, "buffers": buffers}
        if self.codec == "tq":
            result.update({"codec": self.codec, "codec_id": TQ_CODEC_ID,
                           "codec_version": TQ_CODEC_VERSION, "transform_id": TQ_TRANSFORM_ID,
                           "layout": "token_head_tq02", "logical_dtype": "f32",
                           "bits": self.bits, "seed": self.seed,
                           "codebook_f32le": self.codebook_f32le,
                           "head_dim": self.config.head_dim,
                           "head_row_bytes": self.head_row_bytes, "token_bytes": self.token_bytes})
        elif self.codec != "f32":
            result.update({"codec": self.codec, "codec_id": f"{self.codec.upper()}_GROUPED", "codec_version": 1,
                           "layout": "token_head_grouped", "logical_dtype": "f32",
                           "group_size": self.group_size, "head_dim": self.config.head_dim,
                           "head_row_bytes": self.head_row_bytes, "token_bytes": self.token_bytes})
        return result

    @classmethod
    def from_dict(cls, data):
        fields = {"schema_version", "config", "capacity", "page_tokens", "dtype", "alignment",
                  "kv_width", "max_pages", "page_payload_bytes", "buffer_stride",
                  "page_extent_bytes", "page_allocation_bytes", "buffers"}
        packed = isinstance(data, dict) and "codec" in data
        codec_fields = _TQ_LAYOUT_FIELDS if packed and data["codec"] == "tq" else _PACKED_LAYOUT_FIELDS
        _keys(data, fields | codec_fields if packed else fields, "PagedKVCachePlan")
        plan = cls(ModelConfig.from_dict(data["config"]), data["capacity"], data["page_tokens"],
                   data.get("codec", "f32"), data.get("group_size"),
                   data.get("bits"), data.get("seed"), data.get("codebook_f32le"))
        if not _strict_equal(data, plan.to_dict()):
            raise ValueError("Paged KV layout, byte counts or version differ from the canonical plan")
        return plan

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(_load_json(text))


_SEGMENT_FIELDS = ("page_index", "page_token_offset", "source_token_offset", "token_count")


@dataclass(frozen=True)
class PagedKVSegment(Mapping):
    """Immutable token-copy descriptor shared by the per-layer actions."""
    page_index: int
    page_token_offset: int
    source_token_offset: int
    token_count: int

    def __post_init__(self):
        for name in _SEGMENT_FIELDS:
            _integer(getattr(self, name), name, 1 if name == "token_count" else 0)

    def __getitem__(self, name):
        if name not in _SEGMENT_FIELDS:
            raise KeyError(name)
        return getattr(self, name)

    def __iter__(self):
        return iter(_SEGMENT_FIELDS)

    def __len__(self):
        return len(_SEGMENT_FIELDS)


_CACHE_BINDINGS = {"layer", "key_input", "value_input", "key_offset", "value_offset",
                   "past_length", "new_length", "query_length", "kv_width", "page_tokens",
                   "page_count", "segments"}
_PACKED_BINDINGS = {"codec", "group_size", "head_dim", "head_row_bytes", "token_bytes"}
_TQ_BINDINGS = (_PACKED_BINDINGS - {"group_size"}) | {"bits", "seed"}


@dataclass(frozen=True)
class PagedKVAction:
    kind: str
    op_name: str | None
    bindings: Mapping = field(default_factory=dict)

    def __post_init__(self):
        fields = {"Compute": set(), "RoPE": {"position_offset"}, "CacheWrite": _CACHE_BINDINGS,
                  "CachedAttention": _CACHE_BINDINGS, "Commit": {"new_length", "page_count"}}
        if (not isinstance(self.kind, str) or self.kind not in fields
                or not isinstance(self.bindings, Mapping)):
            raise ValueError("Invalid paged KV action kind or bindings")
        accepted = (fields[self.kind],)
        if self.kind in ("CacheWrite", "CachedAttention"):
            accepted += (_CACHE_BINDINGS | _PACKED_BINDINGS, _CACHE_BINDINGS | _TQ_BINDINGS)
        if set(self.bindings) not in accepted:
            raise ValueError("Invalid paged KV action kind or bindings")
        if self.kind == "Commit":
            if self.op_name is not None:
                raise ValueError("Commit must not reference a graph operation")
        else:
            _name(self.op_name, "operation name")
        bindings = dict(self.bindings)
        for name, value in bindings.items():
            if name in ("key_input", "value_input"):
                _name(value, name)
            elif name not in ("segments", "codec", "seed"):
                _integer(value, name, 1 if name in ("new_length", "query_length", "kv_width",
                                                  "page_tokens", "page_count", "group_size",
                                                  "head_dim", "head_row_bytes", "token_bytes", "bits") else 0)
        if "codec" in bindings:
            head_dim = bindings["head_dim"]
            if bindings["codec"] == "tq" and set(bindings) == _CACHE_BINDINGS | _TQ_BINDINGS:
                validate_tq_parameters(head_dim, bindings["bits"], bindings["seed"])
                row_bytes = tq_row_bytes(head_dim, bindings["bits"])
            elif bindings["codec"] in ("q4", "q3", "q8") and set(bindings) == _CACHE_BINDINGS | _PACKED_BINDINGS:
                row_bytes = _packed_head_row_bytes(head_dim, bindings["group_size"], bindings["codec"])
            else:
                raise ValueError("Packed KV bindings require consistent q4, q3, q8 or tq codec parameters")
            if (head_dim % 2 or bindings["kv_width"] % head_dim
                    or bindings["head_row_bytes"] != row_bytes
                    or bindings["token_bytes"] != bindings["kv_width"] // head_dim * row_bytes):
                raise ValueError("Packed KV bindings have inconsistent head geometry or storage bytes")
        if "segments" in bindings:
            values = bindings["segments"]
            if not isinstance(values, (list, tuple)) or not 0 < len(values) <= MAX_PLAN_SEGMENTS:
                raise ValueError("Invalid paged KV segment count")
            segments = []
            for value in values:
                if not isinstance(value, Mapping) or set(value) != set(_SEGMENT_FIELDS):
                    raise ValueError("Invalid paged KV segment fields")
                segments.append(value if isinstance(value, PagedKVSegment) else PagedKVSegment(**value))
            position, source = bindings["past_length"], 0
            page_tokens, length = bindings["page_tokens"], bindings["query_length"]
            if (bindings["new_length"] != position + length
                    or bindings["page_count"] != (position + length + page_tokens - 1) // page_tokens
                    or bindings["key_offset"] % ALIGNMENT or bindings["value_offset"] % ALIGNMENT):
                raise ValueError("Paged KV bindings have inconsistent positions or offsets")
            for segment in segments:
                page_index, offset = divmod(position + source, page_tokens)
                if (segment.page_index != page_index or segment.page_token_offset != offset
                        or segment.source_token_offset != source
                        or segment.token_count != min(page_tokens - offset, length - source)):
                    raise ValueError("Paged KV segments must cover the chunk in page order without gaps")
                source += segment.token_count
            if source != length:
                raise ValueError("Paged KV segments do not cover the whole chunk")
            bindings["segments"] = tuple(segments)
        object.__setattr__(self, "bindings", MappingProxyType(bindings))

    def to_dict(self):
        bindings = dict(self.bindings)
        if "segments" in bindings:
            bindings["segments"] = [dict(segment) for segment in bindings["segments"]]
        return {"kind": self.kind, "op_name": self.op_name, "bindings": bindings}


@dataclass(frozen=True)
class PagedKVStepPlan:
    graph: ModelGraph
    cache_plan: PagedKVCachePlan
    past_length: int
    mode: str = "decode"
    position_offset: int = field(init=False)
    new_length: int = field(init=False)
    page_count: int = field(init=False)
    new_pages: int = field(init=False)
    resident_pages_peak: int = field(init=False)
    steps: tuple[PagedKVAction, ...] = field(init=False)
    activation_requests: tuple[MemoryRequest, ...] = field(init=False)

    def __post_init__(self):
        if not isinstance(self.graph, ModelGraph) or not isinstance(self.cache_plan, PagedKVCachePlan):
            raise ValueError("Paged KV step requires a ModelGraph and PagedKVCachePlan")
        if self.mode not in ("prefill", "decode"):
            raise ValueError("Paged KV mode must be prefill or decode")
        old_pages = self.cache_plan.page_count(self.past_length)
        if self.mode == "decode" and self.past_length == 0:
            raise ValueError("Decode requires a committed KV prefix")
        self.graph.validate()
        tensors = {tensor.name: tensor for tensor in self.graph.tensors}
        if self.graph.inputs != ("tokens",) or len(tensors["tokens"].shape) != 1:
            raise ValueError("Paged KV steps require the canonical token input")
        count = tensors["tokens"].shape[0]
        position = 0 if self.mode == "prefill" else self.past_length
        total = position + count
        page_count = self.cache_plan.page_count(total)
        segment_count = ((position % self.cache_plan.page_tokens) + count + self.cache_plan.page_tokens - 1) // self.cache_plan.page_tokens
        if segment_count > MAX_PLAN_SEGMENTS:
            raise ValueError(f"Paged KV step exceeds the {MAX_PLAN_SEGMENTS} segment metadata limit")
        expected = lower_model(self.cache_plan.config, count,
                               weight_storage={name: tensors[name] for name in self.graph.constants})
        if self.graph.to_dict() != expected.to_dict():
            raise ValueError("Paged KV steps require the canonical lower_model graph and matching config")
        segments = []
        source = 0
        while source < count:
            page_index, offset = divmod(position + source, self.cache_plan.page_tokens)
            copied = min(self.cache_plan.page_tokens - offset, count - source)
            segments.append(PagedKVSegment(page_index, offset, source, copied))
            source += copied
        segments = tuple(segments)
        layers = {f"layers.{layer}.attention": layer for layer in range(self.cache_plan.config.num_hidden_layers)}
        steps = []
        for op in self.graph.ops:
            if op.kind.value == "RoPE":
                steps.append(PagedKVAction("RoPE", op.name, {"position_offset": position}))
            elif op.kind.value == "CausalAttention":
                layer = layers[op.name]
                binding = {"layer": layer, "key_input": op.inputs[1], "value_input": op.inputs[2],
                           "key_offset": self.cache_plan.buffer_offset(layer, "key"),
                           "value_offset": self.cache_plan.buffer_offset(layer, "value"),
                           "past_length": position, "new_length": total, "query_length": count,
                           "kv_width": self.cache_plan.kv_width, "page_tokens": self.cache_plan.page_tokens,
                           "page_count": page_count, "segments": segments}
                if self.cache_plan.codec == "tq":
                    binding.update({"codec": "tq", "bits": self.cache_plan.bits,
                                    "seed": self.cache_plan.seed,
                                    "head_dim": self.cache_plan.config.head_dim,
                                    "head_row_bytes": self.cache_plan.head_row_bytes,
                                    "token_bytes": self.cache_plan.token_bytes})
                elif self.cache_plan.codec != "f32":
                    binding.update({"codec": self.cache_plan.codec, "group_size": self.cache_plan.group_size,
                                    "head_dim": self.cache_plan.config.head_dim,
                                    "head_row_bytes": self.cache_plan.head_row_bytes,
                                    "token_bytes": self.cache_plan.token_bytes})
                steps.append(PagedKVAction("CacheWrite", op.name, binding))
                steps.append(PagedKVAction("CachedAttention", op.name, binding))
            else:
                steps.append(PagedKVAction("Compute", op.name))
        steps.append(PagedKVAction("Commit", None, {"new_length": total, "page_count": page_count}))
        object.__setattr__(self, "position_offset", position)
        object.__setattr__(self, "new_length", total)
        object.__setattr__(self, "page_count", page_count)
        object.__setattr__(self, "new_pages", page_count if self.mode == "prefill" else page_count - old_pages)
        object.__setattr__(self, "resident_pages_peak", old_pages + page_count if self.mode == "prefill" else page_count)
        object.__setattr__(self, "steps", tuple(steps))
        object.__setattr__(self, "activation_requests", _step_lifetimes(self.graph, steps))

    @property
    def bindings(self):
        return MappingProxyType({step.op_name: step.bindings for step in self.steps if step.kind == "CachedAttention"})

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, "graph": self.graph.to_dict(),
                "cache_plan": self.cache_plan.to_dict(), "past_length": self.past_length, "mode": self.mode,
                "position_offset": self.position_offset, "new_length": self.new_length,
                "page_count": self.page_count, "new_pages": self.new_pages,
                "resident_pages_peak": self.resident_pages_peak,
                "steps": [step.to_dict() for step in self.steps],
                "activation_requests": [request.to_dict() for request in self.activation_requests]}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {"schema_version", "graph", "cache_plan", "past_length", "mode", "position_offset",
                     "new_length", "page_count", "new_pages", "resident_pages_peak", "steps",
                     "activation_requests"}, "PagedKVStepPlan")
        plan = cls(ModelGraph.from_dict(data["graph"]), PagedKVCachePlan.from_dict(data["cache_plan"]),
                   data["past_length"], data["mode"])
        if not _strict_equal(data, plan.to_dict()):
            raise ValueError("Paged KV actions, bindings, lifetimes or version differ from the validated plan")
        return plan

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(_load_json(text))


def make_paged_kv_cache_plan(config: ModelConfig, capacity: int, page_tokens: int,
                           *, codec="f32", group_size=None, bits=None, seed=None,
                           codebook_f32le=None) -> PagedKVCachePlan:
    """Plan page storage; TQ's None codebook is an uninitialized session.

    Runtime admission may use this provisional geometry without generating a
    codebook or loading C. Before allocating pages or executing actions, the
    session replaces it with a plan containing its exact F32LE codebook.
    """
    return PagedKVCachePlan(config, capacity, page_tokens, codec, group_size,
                            bits, seed, codebook_f32le)


def plan_paged_step(graph: ModelGraph, cache_plan: PagedKVCachePlan, past_length: int,
                    *, mode="decode") -> PagedKVStepPlan:
    return PagedKVStepPlan(graph, cache_plan, past_length, mode)
