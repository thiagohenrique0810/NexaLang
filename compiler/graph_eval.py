"""Operator-level ModelGraph oracle in pure Python, without kernels or PyTorch.

This evaluator exists to be *wrong differently* from the runtime. It reads the
same validated ModelGraph the native executor consumes and computes every
operator from the contract in model_ir, so a disagreement points at one of the
two implementations instead of at a shared helper. tests/transformer_reference.py
is a Torch oracle wired to the Llama structure; this one knows only operators,
which is what a rewrite verifier and a streaming driver can actually use.

Arithmetic contract: accumulation happens in Python floats (IEEE double) and
every produced element is rounded to float32, because every activation the IR
declares is a dense F32 tensor. Two deliberate departures from the C kernels:
attention keeps the softmax exponentials in double where the native scratch
stores them as float32, and the sigmoid is written from the stable form. Both
keep the oracle independent; the residual error they cause is measured by the
regression suite instead of being defined away.

Values are immutable: a rank-2 tensor is a tuple of row tuples, a rank-1 token
tensor is a tuple of ints. Limits keep this from being mistaken for a runner.
"""
from __future__ import annotations

import math
import random
import struct

from .model_ir import DType, ModelGraph, ModelOp, OpKind


# A tiny-graph diagnostic. Real models run through runtime/nexapack, which
# streams packed weights instead of materializing every tensor as Python floats.
MAX_EVAL_ELEMENTS = 4_000_000

_PACK_F32 = struct.Struct("<f")


def f32(value):
    """Round one Python float to the nearest float32, as a tensor store would."""
    return _PACK_F32.unpack(_PACK_F32.pack(value))[0]


def tensor_bytes(value):
    """Serialize a tensor value to little-endian float32 bytes, row by row.

    Byte equality of this encoding is the strongest statement available about
    two evaluations: it admits no tolerance and no NaN-equals-NaN accident.
    """
    rows = value if _is_rank_two(value) else (value,)
    payload = bytearray()
    for row in rows:
        for item in row:
            payload += _PACK_F32.pack(item)
    return bytes(payload)


def _is_rank_two(value):
    return bool(value) and isinstance(value[0], (tuple, list))


def _rows(value, label):
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"{label} must be a nonempty sequence")
    if not _is_rank_two(value):
        raise ValueError(f"{label} must be a rank-2 value")
    width = len(value[0])
    if not width or any(len(row) != width for row in value):
        raise ValueError(f"{label} must be rectangular with a nonempty width")
    return len(value), width


def matmul(left, right, *, transpose_b=False):
    """Rank-2 matmul with double accumulation and float32 results.

    The reduction runs over ascending k so the summation order is defined by
    the contract rather than by whichever container happens to be iterated.
    """
    rows, inner = _rows(left, "matmul left")
    right_rows, right_cols = _rows(right, "matmul right")
    if transpose_b:
        if right_cols != inner:
            raise ValueError("matmul right operand width does not match the reduction")
        columns = right_rows
    else:
        if right_rows != inner:
            raise ValueError("matmul right operand height does not match the reduction")
        columns = right_cols
    result = []
    for row in left:
        output = []
        for column in range(columns):
            total = 0.0
            if transpose_b:
                other = right[column]
                for index in range(inner):
                    total += row[index] * other[index]
            else:
                for index in range(inner):
                    total += row[index] * right[index][column]
            output.append(f32(total))
        result.append(tuple(output))
    return tuple(result)


def _embedding(tokens, weight):
    if not isinstance(tokens, (tuple, list)) or not tokens:
        raise ValueError("Embedding tokens must be a nonempty sequence")
    rows, _ = _rows(weight, "Embedding weight")
    result = []
    for token in tokens:
        if type(token) is not int or isinstance(token, bool) or not 0 <= token < rows:
            raise ValueError("Embedding token is outside the weight row range")
        result.append(tuple(f32(value) for value in weight[token]))
    return tuple(result)


def _rmsnorm(source, weight, epsilon):
    _, width = _rows(source, "RMSNorm input")
    if len(weight) != width:
        raise ValueError("RMSNorm weight width does not match the row width")
    result = []
    for row in source:
        total = 0.0
        for value in row:
            total += value * value
        factor = 1.0 / math.sqrt(total / width + epsilon)
        result.append(tuple(f32((value * factor) * scale) for value, scale in zip(row, weight)))
    return tuple(result)


def _rope(source, heads, head_dim, theta, row_offset):
    rows, width = _rows(source, "RoPE input")
    if width != heads * head_dim:
        raise ValueError("RoPE head layout does not fill the row width")
    half = head_dim // 2
    frequencies = [theta ** (-2.0 * lane / head_dim) for lane in range(half)]
    result = []
    for index, row in enumerate(source):
        position = row_offset + index
        output = list(row)
        for lane, frequency in enumerate(frequencies):
            angle = position * frequency
            cosine, sine = math.cos(angle), math.sin(angle)
            for head in range(heads):
                offset = head * head_dim + lane
                first, second = row[offset], row[offset + half]
                output[offset] = f32(first * cosine - second * sine)
                output[offset + half] = f32(second * cosine + first * sine)
        result.append(tuple(output))
    return tuple(result)


def _causal_attention(query, key, value, heads, kv_heads, head_dim):
    rows, width = _rows(query, "CausalAttention query")
    key_rows, kv_width = _rows(key, "CausalAttention key")
    if _rows(value, "CausalAttention value") != (key_rows, kv_width):
        raise ValueError("CausalAttention key and value shapes differ")
    if width != heads * head_dim or kv_width != kv_heads * head_dim:
        raise ValueError("CausalAttention head layout does not fill the row width")
    if key_rows < rows:
        raise ValueError("CausalAttention needs one key row per query row")
    scale = 1.0 / math.sqrt(head_dim)
    repeats = heads // kv_heads
    result = []
    for position in range(rows):
        output = [0.0] * width
        for head in range(heads):
            start = head * head_dim
            query_row = query[position][start:start + head_dim]
            kv_offset = (head // repeats) * head_dim
            scores = []
            for past in range(position + 1):
                key_row = key[past]
                dot = 0.0
                for lane in range(head_dim):
                    dot += query_row[lane] * key_row[kv_offset + lane]
                scores.append(dot * scale)
            maximum = max(scores)
            weights = [math.exp(score - maximum) for score in scores]
            denominator = 0.0
            for weight in weights:
                denominator += weight
            for lane in range(head_dim):
                total = 0.0
                for past, weight in enumerate(weights):
                    total += weight * value[past][kv_offset + lane]
                output[start + lane] = f32(total / denominator)
        result.append(tuple(output))
    return tuple(result)


def _elementwise(left, right, combine, label):
    shape = _rows(left, f"{label} left")
    if _rows(right, f"{label} right") != shape:
        raise ValueError(f"{label} operands must have the same shape")
    return tuple(tuple(f32(combine(a, b)) for a, b in zip(first, second))
                 for first, second in zip(left, right))


def _silu_times(gate, up):
    # The stable sigmoid halves avoid overflowing exp for large |gate|.
    if gate >= 0.0:
        sigmoid = 1.0 / (1.0 + math.exp(-gate))
    else:
        exponential = math.exp(gate)
        sigmoid = exponential / (1.0 + exponential)
    return (gate * sigmoid) * up


def evaluate_op(op, inputs, *, row_offset=0):
    """Evaluate one ModelOp over already-materialized input values.

    row_offset is the absolute index of the first supplied row. It exists for
    RoPE, which is independent of every other row and still not independent of
    *which* row it is: a tile driver that forgets the offset silently rotates
    every tile as if it started the sequence.
    """
    if not isinstance(op, ModelOp):
        raise ValueError("op must be a ModelOp")
    if type(row_offset) is not int or isinstance(row_offset, bool) or row_offset < 0:
        raise ValueError("row_offset must be a nonnegative integer")
    inputs = tuple(inputs)
    if len(inputs) != len(op.inputs):
        raise ValueError(f"{op.name}: expected {len(op.inputs)} input values")
    attributes = op.attributes
    if op.kind == OpKind.MATMUL:
        return matmul(inputs[0], inputs[1], transpose_b=attributes.get("transpose_b", False))
    if op.kind == OpKind.EMBEDDING:
        return _embedding(inputs[0], inputs[1])
    if op.kind == OpKind.RMSNORM:
        return _rmsnorm(inputs[0], inputs[1], attributes["epsilon"])
    if op.kind == OpKind.ROPE:
        return _rope(inputs[0], attributes["num_heads"], attributes["head_dim"],
                     attributes["theta"], row_offset)
    if op.kind == OpKind.CAUSAL_ATTENTION:
        return _causal_attention(inputs[0], inputs[1], inputs[2], attributes["num_heads"],
                                 attributes["num_key_value_heads"], attributes["head_dim"])
    if op.kind == OpKind.SWIGLU:
        return _elementwise(inputs[0], inputs[1], _silu_times, "SwiGLU")
    if op.kind == OpKind.ADD:
        return _elementwise(inputs[0], inputs[1], lambda a, b: a + b, "Add")
    raise ValueError(f"{op.name}: unsupported operator {op.kind.value}")


def _check_binding(tensor, value):
    if tensor.logical_dtype == DType.U32:
        if not isinstance(value, (tuple, list)) or len(value) != tensor.shape[0]:
            raise ValueError(f"{tensor.name}: expected {tensor.shape[0]} integer IDs")
        for item in value:
            if type(item) is not int or isinstance(item, bool) or item < 0:
                raise ValueError(f"{tensor.name}: token IDs must be nonnegative integers")
        return tuple(value)
    # A rank-1 weight binds as a flat sequence; treat it as a single row.
    value = value if len(tensor.shape) == 2 else (value,)
    rows, width = _rows(value, tensor.name)
    expected = tensor.shape if len(tensor.shape) == 2 else (1, tensor.shape[0])
    if (rows, width) != expected:
        raise ValueError(f"{tensor.name}: expected shape {expected}, got {(rows, width)}")
    for row in value:
        for item in row:
            if type(item) not in (int, float) or not math.isfinite(item):
                raise ValueError(f"{tensor.name}: values must be finite numbers")
    return tuple(tuple(float(item) for item in row) for row in value)


def _binding_value(tensor, bindings):
    if tensor.name not in bindings:
        raise ValueError(f"missing binding for {tensor.name}")
    value = _check_binding(tensor, bindings[tensor.name])
    # Rank-1 weights bind as one row so callers can pass a flat list; operators
    # that read them (RMSNorm) want the flat form back.
    return value[0] if len(tensor.shape) == 1 and tensor.logical_dtype != DType.U32 else value


def evaluate_graph(graph, bindings):
    """Evaluate every tensor of a validated graph from its inputs and constants.

    Returns one dict with the bound values and every operation result. Keeping
    all of them is what lets a caller compare intermediates, which is the whole
    point of an oracle; the element limit keeps that affordable.
    """
    if not isinstance(graph, ModelGraph):
        raise ValueError("graph must be a ModelGraph")
    graph.validate()
    total = sum(tensor.numel for tensor in graph.tensors)
    if total > MAX_EVAL_ELEMENTS:
        raise ValueError(f"graph evaluation is limited to {MAX_EVAL_ELEMENTS} tensor elements")
    declared = {tensor.name: tensor for tensor in graph.tensors}
    unexpected = set(bindings) - set(graph.inputs) - set(graph.constants)
    if unexpected:
        raise ValueError(f"bindings name tensors that are not inputs or constants: "
                         f"{', '.join(sorted(unexpected))}")
    values = {name: _binding_value(declared[name], bindings)
              for name in (*graph.inputs, *graph.constants)}
    for op in graph.ops:
        result = evaluate_op(op, tuple(values[name] for name in op.inputs))
        expected = declared[op.outputs[0]].shape
        if (len(result), len(result[0])) != expected:
            raise ValueError(f"{op.name}: produced {(len(result), len(result[0]))}, "
                             f"graph declares {expected}")
        values[op.outputs[0]] = result
    return values


def random_bindings(graph, *, seed=0):
    """Deterministic pseudo-random inputs and constants for a validated graph.

    Norm vectors stay near one and matrices stay small so a deep graph neither
    saturates nor collapses; token IDs respect the embedding row count, which
    the graph itself reveals.
    """
    if not isinstance(graph, ModelGraph):
        raise ValueError("graph must be a ModelGraph")
    graph.validate()
    declared = {tensor.name: tensor for tensor in graph.tensors}
    limits = {}
    for op in graph.ops:
        if op.kind == OpKind.EMBEDDING:
            limits[op.inputs[0]] = declared[op.inputs[1]].shape[0]
    rng = random.Random(seed)
    bindings = {}
    for name in sorted({*graph.inputs, *graph.constants}):
        tensor = declared[name]
        if tensor.logical_dtype == DType.U32:
            bound = limits.get(name)
            if bound is None:
                raise ValueError(f"{name}: cannot bound integer IDs without an Embedding consumer")
            bindings[name] = tuple(rng.randrange(bound) for _ in range(tensor.shape[0]))
        elif len(tensor.shape) == 1:
            bindings[name] = [f32(rng.uniform(0.85, 1.15)) for _ in range(tensor.shape[0])]
        else:
            bindings[name] = [[f32(rng.uniform(-0.45, 0.45)) for _ in range(tensor.shape[1])]
                              for _ in range(tensor.shape[0])]
    return bindings
