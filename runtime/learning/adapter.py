"""Low-rank adapter contract applied over a frozen, packed base model (PL.03a).

The base weights are never dequantized, expanded or rewritten: an adapter adds
a delta to a projection the base kernel already produced. There is no training,
no versioned persistence and no change to the bundle manifest here; a payload is
a plain little-endian float32 file that a caller produced by some other means.

Fixed semantics, per adapted projection and per position ``x``::

    y = W_base x + (alpha / rank) * B (A x)

with ``A[rank, in_features]`` and ``B[out_features, rank]``, both row-major.
The payload holds A first, then B, so it is exactly
``rank * (in_features + out_features) * 4`` bytes and nothing else.

Canonical accumulation order, which is part of the contract because floating
point addition is not associative:

1. adapters are applied in their declared ``order``, each one complete before
   the next; the delta of adapter *k* is visible to adapter *k+1*;
2. ``t = A x`` is reduced in increasing input coordinate, in double, and
   rounded once per element, by the same dense kernel the base path uses;
3. the scale multiplies ``t`` -- the rank-sized intermediate -- not the output,
   so one adapter adds ``length * rank`` roundings, not ``length * out``;
4. ``B t`` is reduced in increasing rank coordinate, and the sum with the base
   output is taken in increasing flat index ``position * out_features + feature``.

A null adapter is therefore bit-exact: ``B`` all zero, or ``alpha`` zero, makes
every delta element ``+0.0`` (a double accumulator that starts at ``0.0`` never
produces ``-0.0``), and ``value + 0.0`` returns ``value`` unchanged for every
finite float the base path can produce.
"""
from dataclasses import dataclass, fields
from functools import lru_cache
import math
import os
import struct

from compiler.model_config import ModelConfig
from compiler.model_ir import OpKind, _integer, _keys, _name
from compiler.model_lowering import lower_model
from runtime.nexapack.format import READ_CHUNK_BYTES


SCHEMA_VERSION = 1
SEMANTICS = "y = W_base x + (alpha / rank) * B (A x)"
ACCUMULATION_ORDER = ("declared adapter order; A x in increasing input coordinate; "
                      "scale applied to the rank intermediate; B t in increasing rank "
                      "coordinate; base sum in increasing position*out_features+feature")
PAYLOAD_LAYOUT = "row-major float32 little-endian: A[rank, in_features] then B[out_features, rank]"

MAX_ADAPTER_RANK = 64
MAX_ADAPTERS = 64
MAX_ADAPTER_PAYLOAD_BYTES = 16 << 20
SUPPORTED_PRECISIONS = ("f32",)

_F32 = struct.Struct("<f")


class AdapterError(ValueError):
    """An adapter set refused before any payload byte was consumed."""


def _finite_float(value, label):
    if type(value) not in (int, float):
        raise AdapterError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise AdapterError(f"{label} must be a finite number")
    return value


@lru_cache(maxsize=8)
def matmul_weight_targets(config: ModelConfig):
    """Physical weights a V1 adapter may target, derived from the real graph.

    Derived rather than listed: the lowering decides which constant each op
    consumes, and with tied embeddings the logits MatMul consumes the embedding
    tensor itself. Both the embedding table and the output head are excluded in
    V1 -- an adapter there would change token identity, not a projection, and
    the tied case would silently adapt two different ops at once.
    """
    if not isinstance(config, ModelConfig):
        raise AdapterError("config must be a ModelConfig")
    graph = lower_model(config, 1)
    embedding = {op.inputs[1] for op in graph.ops if op.kind is OpKind.EMBEDDING}
    matmul = {op.inputs[1] for op in graph.ops if op.kind is OpKind.MATMUL}
    head = config.tensor_aliases().get("lm_head.weight", "lm_head.weight")
    return frozenset(matmul - embedding - {head})


@dataclass(frozen=True)
class AdapterSpec:
    """One declared adapter; shapes are declared, not inferred from the payload.

    Declaring ``in_features``/``out_features`` is what lets the whole set be
    refused before a single payload byte is read: the declaration is compared
    against the base tensor as the bundle stores it.
    """
    id: str
    target: str
    rank: int
    alpha: float
    in_features: int
    out_features: int
    payload_path: str
    order: int
    precision: str = "f32"

    def __post_init__(self):
        _name(self.id, "adapter id")
        _name(self.target, "adapter target tensor")
        if not isinstance(self.payload_path, str) or not self.payload_path:
            raise AdapterError("adapter payload_path must be a nonempty string")
        if type(self.rank) is not int or not 0 < self.rank <= MAX_ADAPTER_RANK:
            raise AdapterError(f"adapter rank must be an integer from 1 to {MAX_ADAPTER_RANK}")
        object.__setattr__(self, "alpha", _finite_float(self.alpha, "adapter alpha"))
        _integer(self.in_features, "in_features", 1)
        _integer(self.out_features, "out_features", 1)
        _integer(self.order, "adapter order", 0)
        if self.precision not in SUPPORTED_PRECISIONS:
            raise AdapterError("V1 adapters are float32 only")
        if self.payload_bytes > MAX_ADAPTER_PAYLOAD_BYTES:
            raise AdapterError(f"adapter payload exceeds {MAX_ADAPTER_PAYLOAD_BYTES} bytes")

    @property
    def payload_bytes(self):
        return self.rank * (self.in_features + self.out_features) * _F32.size

    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {field.name for field in fields(cls)}, "AdapterSpec")
        return cls(**data)


class AdapterSet:
    """An ordered set of adapters, with at most one adapter per base tensor.

    One adapter per target in V1: composing two deltas on the same projection
    needs a declared composition rule, and inventing one here would be a rule
    no test could check against anything outside this file.
    """
    def __init__(self, adapters=()):
        specs = tuple(adapter if isinstance(adapter, AdapterSpec) else AdapterSpec.from_dict(dict(adapter))
                      for adapter in adapters)
        if len(specs) > MAX_ADAPTERS:
            raise AdapterError(f"an adapter set holds at most {MAX_ADAPTERS} adapters")
        for label, values in (("id", [spec.id for spec in specs]),
                              ("target", [spec.target for spec in specs]),
                              ("order", [spec.order for spec in specs])):
            if len(set(values)) != len(values):
                raise AdapterError(f"adapter {label} values must be unique within a set")
        self._specs = tuple(sorted(specs, key=lambda spec: spec.order))

    def __len__(self):
        return len(self._specs)

    def __iter__(self):
        return iter(self._specs)

    @property
    def specs(self):
        return self._specs

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, "semantics": SEMANTICS,
                "accumulation_order": ACCUMULATION_ORDER,
                "payload_layout": PAYLOAD_LAYOUT,
                "adapters": [spec.to_dict() for spec in self._specs]}

    def bind(self, bundle):
        """Validate every adapter against a bundle before reading any payload.

        Raises on the first problem; a set that binds has a target, a shape, a
        role and a payload size that all agree with the packed base model.
        """
        summary = bundle.inspect()
        shapes = {item["name"]: tuple(item["shape"]) for item in summary["tensors"]}
        config = bundle.config
        allowed = matmul_weight_targets(config)
        aliases = config.tensor_aliases()
        bound = []
        for spec in self._specs:
            if spec.target in aliases:
                raise AdapterError(f"adapter {spec.id} targets alias {spec.target}; "
                                   "name the physical tensor")
            if spec.target not in shapes:
                raise AdapterError(f"adapter {spec.id} targets unknown tensor {spec.target}")
            if spec.target not in allowed:
                raise AdapterError(f"adapter {spec.id} targets {spec.target}, which is not a "
                                   "MatMul projection weight; V1 refuses embedding and lm_head")
            shape = shapes[spec.target]
            if len(shape) != 2 or shape != (spec.out_features, spec.in_features):
                raise AdapterError(f"adapter {spec.id} declares A[{spec.rank}, {spec.in_features}] "
                                   f"and B[{spec.out_features}, {spec.rank}], but {spec.target} "
                                   f"stores shape {list(shape)}")
            if spec.rank > min(shape):
                raise AdapterError(f"adapter {spec.id} rank {spec.rank} exceeds the smaller "
                                   f"dimension of {spec.target}")
            try:
                size = os.stat(spec.payload_path).st_size
            except OSError as error:
                raise AdapterError(f"adapter {spec.id} payload is unreadable: {error}") from error
            if size != spec.payload_bytes:
                raise AdapterError(f"adapter {spec.id} payload has {size} bytes; the declared "
                                   f"shapes need exactly {spec.payload_bytes}")
            bound.append(BoundAdapter(spec, shape[0], shape[1]))
        return BoundAdapterSet(self, tuple(bound))


@dataclass(frozen=True)
class BoundAdapter:
    spec: AdapterSpec
    rows: int
    cols: int

    @property
    def payload_bytes(self):
        return self.spec.payload_bytes

    @property
    def a_bytes(self):
        return self.spec.rank * self.cols * _F32.size

    def read_payload_into(self, destination):
        """Stream the payload straight into a caller-owned buffer of exact size.

        No intermediate copy: the destination is the executor's arena slot, so
        the payload never doubles the resident bytes it is accounted for.
        """
        view = memoryview(destination).cast("B")
        if len(view) != self.payload_bytes:
            raise AdapterError(f"adapter {self.spec.id} needs a {self.payload_bytes}-byte buffer")
        with open(self.spec.payload_path, "rb", buffering=0) as stream:
            if os.fstat(stream.fileno()).st_size != self.payload_bytes:
                raise AdapterError(f"adapter {self.spec.id} payload changed size")
            consumed = 0
            while consumed < self.payload_bytes:
                chunk = min(READ_CHUNK_BYTES, self.payload_bytes - consumed)
                read = stream.readinto(view[consumed:consumed + chunk])
                if not read:
                    raise AdapterError(f"adapter {self.spec.id} payload is truncated")
                consumed += read
            if stream.read(1):
                raise AdapterError(f"adapter {self.spec.id} payload grew during the read")
        for value in view.cast("f"):
            if not math.isfinite(value):
                raise AdapterError(f"adapter {self.spec.id} payload holds a nonfinite float32")
        return consumed


class BoundAdapterSet:
    """A validated adapter set, indexed by the base tensor each one adapts."""
    def __init__(self, declared, bound):
        self._declared = declared
        self._bound = bound
        self._by_target = {}
        for adapter in bound:
            self._by_target.setdefault(adapter.spec.target, []).append(adapter)
        self._by_target = {name: tuple(items) for name, items in self._by_target.items()}

    def __len__(self):
        return len(self._bound)

    def __iter__(self):
        return iter(self._bound)

    def for_target(self, name):
        return self._by_target.get(name, ())

    @property
    def max_rank(self):
        return max((adapter.spec.rank for adapter in self._bound), default=0)

    @property
    def max_payload_bytes(self):
        return max((adapter.payload_bytes for adapter in self._bound), default=0)

    @property
    def max_target_rows(self):
        return max((adapter.rows for adapter in self._bound), default=0)

    @property
    def payload_bytes_per_pass(self):
        """Bytes one full graph evaluation reads, with every adapter applied once."""
        return sum(adapter.payload_bytes for adapter in self._bound)

    def to_dict(self):
        result = self._declared.to_dict()
        result["bound_targets"] = [{"target": adapter.spec.target, "id": adapter.spec.id,
                                    "rows": adapter.rows, "cols": adapter.cols,
                                    "rank": adapter.spec.rank,
                                    "payload_bytes": adapter.payload_bytes}
                                   for adapter in self._bound]
        result["payload_bytes_per_pass"] = self.payload_bytes_per_pass
        return result


def write_adapter_payload(path, a_rows, b_rows):
    """Reference writer for the payload layout; the only producer in the tree.

    Kept beside the reader so the layout has exactly one definition: a second
    writer somewhere else would be free to disagree with it.
    """
    a_rows = [list(row) for row in a_rows]
    b_rows = [list(row) for row in b_rows]
    if not a_rows or not b_rows:
        raise AdapterError("an adapter payload needs both A and B")
    rank, cols = len(a_rows), len(a_rows[0])
    rows = len(b_rows)
    if any(len(row) != cols for row in a_rows):
        raise AdapterError("A must be rectangular")
    if any(len(row) != rank for row in b_rows):
        raise AdapterError("B must have exactly rank columns")
    payload = bytearray()
    for row in a_rows + b_rows:
        for value in row:
            value = _finite_float(value, "adapter coefficient")
            packed = _F32.pack(value)
            if not math.isfinite(_F32.unpack(packed)[0]):
                raise AdapterError("adapter coefficient does not fit a finite float32")
            payload += packed
    expected = rank * (cols + rows) * _F32.size
    if len(payload) != expected:
        raise AdapterError("adapter payload size does not match its declared shapes")
    with open(path, "wb") as stream:
        stream.write(payload)
    return {"path": str(path), "rank": rank, "in_features": cols, "out_features": rows,
            "payload_bytes": expected, "layout": PAYLOAD_LAYOUT}
