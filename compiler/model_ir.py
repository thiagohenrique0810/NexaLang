"""Validated tensor graphs, independent of the language bootstrap compiler.

The original rank-two MatMul contract remains supported. Transformer operators
describe stateless, single-sequence prefill with dense F32 activations. RoPE uses
positions 0..S-1 and HF half-rotation, attention is causal GQA with scale
1/sqrt(head_dim), and SwiGLU means silu(gate) * up. Packed storage sizes include
metadata and must be supplied by the codec. Validation does not select a backend.
"""
from dataclasses import dataclass, field
from enum import Enum
import json
from math import isfinite, prod
from types import MappingProxyType
from typing import Mapping


SCHEMA_VERSION = 1


class DType(str, Enum):
    BOOL = "bool"
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"
    F64 = "f64"
    I8 = "i8"
    U8 = "u8"
    I16 = "i16"
    U16 = "u16"
    I32 = "i32"
    U32 = "u32"
    I64 = "i64"
    U64 = "u64"
    Q2 = "q2"
    Q3 = "q3"
    Q4 = "q4"


class MemoryTier(str, Enum):
    HOST = "host"
    DEVICE = "device"
    DISK = "disk"


class OpKind(str, Enum):
    MATMUL = "MatMul"
    EMBEDDING = "Embedding"
    RMSNORM = "RMSNorm"
    ROPE = "RoPE"
    CAUSAL_ATTENTION = "CausalAttention"
    SWIGLU = "SwiGLU"
    ADD = "Add"


_PACKED_BITS = {DType.Q2: 2, DType.Q3: 3, DType.Q4: 4}
_ITEM_BYTES = {
    DType.BOOL: 1, DType.F16: 2, DType.BF16: 2, DType.F32: 4,
    DType.F64: 8, DType.I8: 1, DType.U8: 1, DType.I16: 2,
    DType.U16: 2, DType.I32: 4, DType.U32: 4, DType.I64: 8,
    DType.U64: 8,
}


def _name(value, label):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty string without surrounding whitespace")
    return value


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _alignment(value):
    _integer(value, "alignment", 1)
    if value & (value - 1):
        raise ValueError("alignment must be a power of two")
    return value


def _positive_real(value, label):
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite positive number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be a finite positive number") from error
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return result


def _keys(data, required, label):
    if not isinstance(data, dict) or set(data) != set(required):
        raise ValueError(f"{label} must contain exactly these keys: {', '.join(sorted(required))}")


def _sequence(value, label):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list or tuple")
    return tuple(value)


def _names(value, label):
    values = _sequence(value, label)
    for item in values:
        _name(item, label)
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicates")
    return values


def _load_json(text):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    return json.loads(text, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)


@dataclass(frozen=True)
class TensorDesc:
    name: str
    shape: tuple[int, ...]
    logical_dtype: DType = DType.F32
    storage_dtype: DType = DType.F32
    storage_nbytes: int | None = None
    alignment: int = 64
    tier: MemoryTier = MemoryTier.HOST

    def __post_init__(self):
        _name(self.name, "tensor name")
        shape = _sequence(self.shape, "shape")
        if not shape:
            raise ValueError("shape must have at least one positive dimension")
        for dim in shape:
            _integer(dim, "shape dimension", 1)
        object.__setattr__(self, "shape", shape)
        logical, storage = DType(self.logical_dtype), DType(self.storage_dtype)
        if logical in _PACKED_BITS:
            raise ValueError("logical_dtype must describe unpacked values")
        object.__setattr__(self, "logical_dtype", logical)
        object.__setattr__(self, "storage_dtype", storage)
        object.__setattr__(self, "tier", MemoryTier(self.tier))
        _alignment(self.alignment)
        if storage in _PACKED_BITS:
            _integer(self.storage_nbytes, "packed storage_nbytes", 1)
            minimum = (self.numel * _PACKED_BITS[storage] + 7) // 8
            if self.storage_nbytes < minimum:
                raise ValueError(f"packed storage_nbytes is smaller than its {minimum}-byte payload")
        else:
            expected = self.numel * _ITEM_BYTES[storage]
            if self.storage_nbytes is not None:
                _integer(self.storage_nbytes, "storage_nbytes", 1)
                if self.storage_nbytes != expected:
                    raise ValueError(f"dense storage_nbytes must equal {expected}")
            object.__setattr__(self, "storage_nbytes", expected)

    @property
    def numel(self):
        return prod(self.shape)

    def to_dict(self):
        return {"name": self.name, "shape": list(self.shape),
                "logical_dtype": self.logical_dtype.value,
                "storage_dtype": self.storage_dtype.value,
                "storage_nbytes": self.storage_nbytes,
                "alignment": self.alignment, "tier": self.tier.value}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {"name", "shape", "logical_dtype", "storage_dtype",
                     "storage_nbytes", "alignment", "tier"}, "TensorDesc")
        return cls(**data)


@dataclass(frozen=True)
class ModelOp:
    name: str
    kind: OpKind
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attributes: Mapping = field(default_factory=dict)

    def __post_init__(self):
        _name(self.name, "operation name")
        object.__setattr__(self, "kind", OpKind(self.kind))
        # Repeated inputs are valid, e.g. MatMul(x, x).
        inputs = _sequence(self.inputs, "operation inputs")
        for item in inputs:
            _name(item, "operation input")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", _names(self.outputs, "operation outputs"))
        contracts = {
            OpKind.MATMUL: (2, set(), {"transpose_b"}),
            OpKind.EMBEDDING: (2, set(), set()),
            OpKind.RMSNORM: (2, {"epsilon"}, set()),
            OpKind.ROPE: (1, {"num_heads", "head_dim", "theta"}, set()),
            OpKind.CAUSAL_ATTENTION: (3, {"num_heads", "num_key_value_heads", "head_dim"}, set()),
            OpKind.SWIGLU: (2, set(), set()),
            OpKind.ADD: (2, set(), set()),
        }
        arity, required, optional = contracts[self.kind]
        if (not isinstance(self.attributes, Mapping) or not required.issubset(self.attributes)
                or set(self.attributes) - required - optional):
            raise ValueError(f"{self.kind.value} has missing or unsupported attributes")
        attributes = dict(self.attributes)
        if "transpose_b" in attributes and type(attributes["transpose_b"]) is not bool:
            raise ValueError("MatMul transpose_b must be a boolean")
        for name in ("num_heads", "num_key_value_heads", "head_dim"):
            if name in attributes:
                _integer(attributes[name], name, 1)
        for name in ("epsilon", "theta"):
            if name in attributes:
                attributes[name] = _positive_real(attributes[name], name)
        if self.kind == OpKind.ROPE and attributes["head_dim"] % 2:
            raise ValueError("RoPE head_dim must be even for half-rotation")
        if (self.kind == OpKind.CAUSAL_ATTENTION
                and attributes["num_heads"] % attributes["num_key_value_heads"]):
            raise ValueError("CausalAttention query heads must be divisible by KV heads")
        object.__setattr__(self, "attributes", MappingProxyType(attributes))
        if len(self.inputs) != arity or len(self.outputs) != 1:
            raise ValueError(f"{self.kind.value} requires exactly {arity} inputs and one output")

    def to_dict(self):
        return {"name": self.name, "kind": self.kind.value,
                "inputs": list(self.inputs), "outputs": list(self.outputs),
                "attributes": dict(self.attributes)}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {"name", "kind", "inputs", "outputs", "attributes"}, "ModelOp")
        return cls(**data)


def _dense_f32(tensors, operation):
    if any(t.logical_dtype != DType.F32 or t.storage_dtype != DType.F32 for t in tensors):
        raise ValueError(f"{operation}: activations and norm weights must be dense F32")


def _rank(tensors, rank, operation):
    if any(len(t.shape) != rank for t in tensors):
        raise ValueError(f"{operation}: expected rank-{rank} tensors")


def _validate_operation(op, inputs, result):
    label = f"{op.name} ({op.kind.value})"
    if op.kind == OpKind.MATMUL:
        left, right = inputs
        _rank((*inputs, result), 2, label)
        inner, columns = ((right.shape[1], right.shape[0])
                          if op.attributes.get("transpose_b", False) else right.shape)
        if left.shape[1] != inner or result.shape != (left.shape[0], columns):
            raise ValueError(f"{label}: incompatible MatMul shapes")
        if len({left.logical_dtype, right.logical_dtype, result.logical_dtype}) != 1:
            raise ValueError(f"{label}: MatMul logical dtypes must match")
        if left.logical_dtype == DType.BOOL:
            raise ValueError(f"{label}: boolean MatMul is not supported")
    elif op.kind == OpKind.EMBEDDING:
        tokens, weight = inputs
        _rank((tokens,), 1, label)
        _rank((weight, result), 2, label)
        if tokens.logical_dtype != DType.U32 or tokens.storage_dtype != DType.U32:
            raise ValueError(f"{label}: token IDs must be dense U32")
        if weight.logical_dtype != DType.F32 or weight.storage_dtype not in (DType.F32, DType.Q4):
            raise ValueError(f"{label}: embedding weights must have F32 logical values and F32/Q4 storage")
        _dense_f32((result,), label)
        if result.shape != (tokens.shape[0], weight.shape[1]):
            raise ValueError(f"{label}: incompatible Embedding shapes")
    elif op.kind == OpKind.RMSNORM:
        source, weight = inputs
        _rank((source, result), 2, label)
        _rank((weight,), 1, label)
        _dense_f32((*inputs, result), label)
        if result.shape != source.shape or weight.shape != (source.shape[1],):
            raise ValueError(f"{label}: incompatible RMSNorm shapes")
    elif op.kind == OpKind.ROPE:
        source, = inputs
        _rank((source, result), 2, label)
        _dense_f32((source, result), label)
        if (result.shape != source.shape
                or source.shape[1] != op.attributes["num_heads"] * op.attributes["head_dim"]):
            raise ValueError(f"{label}: incompatible RoPE shapes or head layout")
    elif op.kind == OpKind.CAUSAL_ATTENTION:
        query, key, value = inputs
        _rank((*inputs, result), 2, label)
        _dense_f32((*inputs, result), label)
        width = op.attributes["num_heads"] * op.attributes["head_dim"]
        kv_width = op.attributes["num_key_value_heads"] * op.attributes["head_dim"]
        if (query.shape[1] != width or key.shape != (query.shape[0], kv_width)
                or value.shape != key.shape or result.shape != query.shape):
            raise ValueError(f"{label}: incompatible CausalAttention shapes or head layout")
    elif op.kind in (OpKind.ADD, OpKind.SWIGLU):
        _rank((*inputs, result), 2, label)
        _dense_f32((*inputs, result), label)
        if any(t.shape != result.shape for t in inputs):
            raise ValueError(f"{label}: input and output shapes must match without broadcasting")


@dataclass(frozen=True)
class ModelGraph:
    name: str
    tensors: tuple[TensorDesc, ...]
    ops: tuple[ModelOp, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    constants: tuple[str, ...] = ()

    def __post_init__(self):
        _name(self.name, "graph name")
        for field_name in ("tensors", "ops"):
            object.__setattr__(self, field_name, _sequence(getattr(self, field_name), field_name))
        for field_name in ("inputs", "outputs", "constants"):
            object.__setattr__(self, field_name, _names(getattr(self, field_name), field_name))
        self.validate()

    def validate(self):
        if any(not isinstance(t, TensorDesc) for t in self.tensors):
            raise ValueError("tensors must contain TensorDesc objects")
        if any(not isinstance(op, ModelOp) for op in self.ops):
            raise ValueError("ops must contain ModelOp objects")
        tensors = {tensor.name: tensor for tensor in self.tensors}
        if len(tensors) != len(self.tensors):
            raise ValueError("tensor names must be unique")
        if len({op.name for op in self.ops}) != len(self.ops):
            raise ValueError("operation names must be unique")
        if not self.outputs:
            raise ValueError("graph must declare an output")
        available = set(self.inputs) | set(self.constants)
        if set(self.inputs) & set(self.constants):
            raise ValueError("graph inputs and constants must be disjoint")
        if not available.issubset(tensors) or not set(self.outputs).issubset(tensors):
            raise ValueError("graph inputs, outputs and constants must reference declared tensors")
        for op in self.ops:
            if not set(op.inputs).issubset(available):
                raise ValueError(f"{op.name}: input is undefined or produced after its use")
            if not set(op.outputs).issubset(tensors):
                raise ValueError(f"{op.name}: output tensor is not declared")
            if set(op.outputs) & available:
                raise ValueError(f"{op.name}: tensor has multiple definitions or aliases an input")
            _validate_operation(op, tuple(tensors[name] for name in op.inputs), tensors[op.outputs[0]])
            available.update(op.outputs)
        if not set(self.outputs).issubset(available):
            raise ValueError("graph output has no producer")
        if available != set(tensors):
            raise ValueError("every tensor must be an input, constant or operation output")
        return self

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, "name": self.name,
                "tensors": [t.to_dict() for t in self.tensors],
                "ops": [op.to_dict() for op in self.ops], "inputs": list(self.inputs),
                "outputs": list(self.outputs), "constants": list(self.constants)}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {"schema_version", "name", "tensors", "ops", "inputs",
                     "outputs", "constants"}, "ModelGraph")
        if type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported ModelGraph schema_version")
        tensors = _sequence(data["tensors"], "tensors")
        ops = _sequence(data["ops"], "ops")
        return cls(name=data["name"], tensors=tuple(TensorDesc.from_dict(t) for t in tensors),
                   ops=tuple(ModelOp.from_dict(op) for op in ops), inputs=data["inputs"],
                   outputs=data["outputs"], constants=data["constants"])

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(_load_json(text))
