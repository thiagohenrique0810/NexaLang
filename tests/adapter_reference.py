"""Pure-Python float64 Llama oracle with its own Q4 decoder, for PL.03a.

Deliberately dependency-free and deliberately *not* sharing code with the
runtime: no PyTorch, and no import of ``decode_q4_row`` or any other kernel or
codec helper from ``runtime``. An oracle that decodes with the same decoder and
reduces in the same function proves only that one call site equals itself; this
one re-derives the Q4_GROUPED layout from the format description and re-derives
every equation from the C kernels' documented arithmetic.

Boundaries deliberately match the runtime's: activations round to float32 at
each operator boundary, reductions accumulate in double, and decoded Q4 weights
keep their exact ``scale * code`` products inside a matmul while an embedding
lookup rounds the decoded row to float32.

The adapter delta follows the contract in ``runtime/learning/adapter.py``:
``y = W_base x + (alpha / rank) B (A x)``, with the scale applied to the rank
intermediate and the base sum taken last.

Only Q4_GROUPED matrices are supported; a bundle in any other codec is refused
rather than silently approximated.
"""
from __future__ import annotations

import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_REFERENCE_PARAMETERS = 2_000_000
MAX_REFERENCE_SEQUENCE = 64

_F32 = struct.Struct('<f')


def f32(value):
    """Round a Python float to the nearest float32, as a store would."""
    result = _F32.unpack(_F32.pack(value))[0]
    if not math.isfinite(result):
        raise ArithmeticError('reference value left the float32 range')
    return result


def q4_row_bytes(cols, group_size):
    if type(cols) is not int or cols <= 0:
        raise ValueError('Q4 columns must be a positive integer')
    if type(group_size) is not int or group_size <= 0:
        raise ValueError('Q4 group size must be a positive integer')
    return ((cols + group_size - 1) // group_size) * (4 + (group_size + 1) // 2)


def decode_q4_row_local(data, cols, group_size):
    """Decode one Q4_GROUPED row into exact scale*code products.

    Layout, restated from the container description rather than imported: each
    group stores a little-endian float32 scale followed by four-bit two's
    complement codes, two per byte, low nibble first, padded to an even count.
    Code -8 is reserved; padding and zero-scale groups must hold zero codes.
    """
    payload = memoryview(data).cast('B')
    expected = q4_row_bytes(cols, group_size)
    if len(payload) != expected:
        raise ValueError(f'Q4 row needs {expected} bytes, got {len(payload)}')
    group_bytes = 4 + (group_size + 1) // 2
    values = []
    for offset in range(0, expected, group_bytes):
        scale = _F32.unpack_from(payload, offset)[0]
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('Q4 scale must be finite and nonnegative')
        count = min(group_size, cols - len(values))
        for index in range(((group_size + 1) // 2) * 2):
            nibble = payload[offset + 4 + index // 2]
            code = (nibble >> 4) if index % 2 else (nibble & 0xF)
            signed = code if code < 8 else code - 16
            if signed == -8:
                raise ValueError('Q4 code -8 is reserved')
            if signed and (index >= count or not scale):
                raise ValueError('Q4 padding and zero-scale codes must be zero')
            if index < count:
                values.append(scale * signed)
    if len(values) != cols:
        raise ValueError('Q4 row decoded to the wrong width')
    return values


def load_bundle_weights(bundle_path):
    """Read a small Q4 bundle into exact decoded float64 reference tensors."""
    from runtime.nexapack.bundle import ModelBundleReader
    with ModelBundleReader(bundle_path) as bundle:
        config = bundle.config
        if config.parameter_count() > MAX_REFERENCE_PARAMETERS:
            raise ValueError(f'the reference is limited to {MAX_REFERENCE_PARAMETERS} parameters')
        codecs = {item['name']: item['codec'] for item in bundle.inspect()['tensors']}
        weights = {}
        for name, shape in config.required_tensor_shapes().items():
            if len(shape) == 1:
                weights[name] = [float(value) for value in bundle.read_f32(name)]
                continue
            if codecs[name] != 'Q4_GROUPED':
                raise ValueError(f'the reference only decodes Q4_GROUPED matrices: {name}')
            rows = []
            with bundle.open_packed(name) as reader:
                for row in range(reader.rows):
                    rows.append(decode_q4_row_local(reader.read_rows(row, 1),
                                                    reader.cols, reader.group_size))
            if len(rows) != shape[0] or any(len(row) != shape[1] for row in rows):
                raise ValueError(f'decoded matrix does not match its declared shape: {name}')
            weights[name] = rows
        for alias, target in config.tensor_aliases().items():
            weights[alias] = weights[target]
    return config, weights


def read_adapter_payload(path, rank, in_features, out_features):
    """Read the payload layout back without importing the runtime reader."""
    data = Path(path).read_bytes()
    expected = rank * (in_features + out_features) * 4
    if len(data) != expected:
        raise ValueError(f'adapter payload needs {expected} bytes, got {len(data)}')
    values = list(struct.unpack(f'<{rank * (in_features + out_features)}f', data))
    if any(not math.isfinite(value) for value in values):
        raise ValueError('adapter payload holds a nonfinite float32')
    a_rows = [values[i * in_features:(i + 1) * in_features] for i in range(rank)]
    base = rank * in_features
    b_rows = [values[base + i * rank:base + (i + 1) * rank] for i in range(out_features)]
    return a_rows, b_rows


def _matmul(x, weight):
    """out[p][j] = f32(sum_i x[p][i] * weight[j][i]) accumulated in double."""
    result = []
    for row in x:
        line = []
        for record in weight:
            total = 0.0
            for left, right in zip(row, record):
                total += left * right
            line.append(f32(total))
        result.append(line)
    return result


def _rmsnorm(x, weight, epsilon):
    result = []
    for row in x:
        total = 0.0
        for value in row:
            total += value * value
        factor = 1.0 / math.sqrt(total / len(row) + epsilon)
        result.append([f32((value * factor) * scale) for value, scale in zip(row, weight)])
    return result


def _rope(x, heads, head_dim, theta, offset):
    half = head_dim // 2
    result = [list(row) for row in x]
    for lane in range(half):
        frequency = math.pow(theta, -2.0 * lane / head_dim)
        for position, row in enumerate(x):
            angle = (offset + position) * frequency
            cosine, sine = math.cos(angle), math.sin(angle)
            for head in range(heads):
                index = head * head_dim + lane
                first, second = row[index], row[index + half]
                result[position][index] = f32(first * cosine - second * sine)
                result[position][index + half] = f32(second * cosine + first * sine)
    return result


def _attention(query, key, value, heads, kv_heads, head_dim):
    scale = 1.0 / math.sqrt(head_dim)
    repeats = heads // kv_heads
    result = [[0.0] * (heads * head_dim) for _ in query]
    for position in range(len(query)):
        for head in range(heads):
            start = head * head_dim
            q = query[position][start:start + head_dim]
            kv_offset = (head // repeats) * head_dim
            scores = []
            for past in range(position + 1):
                dot = 0.0
                row = key[past]
                for lane in range(head_dim):
                    dot += q[lane] * row[kv_offset + lane]
                scores.append(dot * scale)
            maximum = max(scores)
            # The native scratch holds the unnormalized exponentials as float32;
            # the denominator and the weighted sum stay in double.
            exponentials = [f32(math.exp(score - maximum)) for score in scores]
            denominator = 0.0
            for item in exponentials:
                denominator += item
            for lane in range(head_dim):
                total = 0.0
                for past in range(position + 1):
                    total += exponentials[past] * value[past][kv_offset + lane]
                result[position][start + lane] = f32(total / denominator)
    return result


def _swiglu(gate, up):
    result = []
    for gate_row, up_row in zip(gate, up):
        line = []
        for value, other in zip(gate_row, up_row):
            exponential = math.exp(-value if value >= 0.0 else value)
            sigmoid = 1.0 / (1.0 + exponential) if value >= 0.0 else exponential / (1.0 + exponential)
            line.append(f32((value * sigmoid) * other))
        result.append(line)
    return result


def _add(left, right):
    return [[f32(a + b) for a, b in zip(first, second)] for first, second in zip(left, right)]


def _apply_adapter(base, x, rank, alpha, a_rows, b_rows):
    """y = base + (alpha/rank) B (A x), in the contract's rounding order."""
    low = _matmul(x, a_rows)
    scale = alpha / rank
    low = [[f32(value * scale) for value in row] for row in low]
    delta = _matmul(low, b_rows)
    return _add(base, delta)


class LocalLlamaReference:
    """Functional float64 Llama prefill with optional low-rank adapters.

    ``adapters`` maps a physical weight name to an ordered list of
    ``(rank, alpha, A, B)``. Adapters compose in list order; each is complete
    before the next, matching the executor's declared order.
    """
    def __init__(self, config, weights, *, adapters=None):
        self.config = config
        self.weights = weights
        self.adapters = {} if adapters is None else dict(adapters)
        for name in self.adapters:
            if name not in weights:
                raise ValueError(f'adapter targets an unknown weight: {name}')

    def _project(self, x, name):
        base = _matmul(x, self.weights[name])
        for rank, alpha, a_rows, b_rows in self.adapters.get(name, ()):
            base = _apply_adapter(base, x, rank, alpha, a_rows, b_rows)
        return base

    def prefill(self, token_ids):
        config = self.config
        tokens = tuple(token_ids)
        if not tokens or len(tokens) > min(config.max_position_embeddings, MAX_REFERENCE_SEQUENCE):
            raise ValueError('reference prefill needs a nonempty in-range sequence')
        if any(type(token) is not int or not 0 <= token < config.vocab_size for token in tokens):
            raise ValueError('every token must be a vocabulary integer')
        heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
        width = config.head_dim
        table = self.weights['model.embed_tokens.weight']
        # The embedding kernel writes a decoded row as float32; a matmul keeps
        # the same decoded values in double. Both readings come from one table.
        x = [[f32(value) for value in table[token]] for token in tokens]
        for layer in range(config.num_hidden_layers):
            prefix = f'model.layers.{layer}.'
            normalized = _rmsnorm(x, self.weights[prefix + 'input_layernorm.weight'],
                                  config.rms_norm_eps)
            query = _rope(self._project(normalized, prefix + 'self_attn.q_proj.weight'),
                          heads, width, config.rope_theta, 0)
            key = _rope(self._project(normalized, prefix + 'self_attn.k_proj.weight'),
                        kv_heads, width, config.rope_theta, 0)
            value = self._project(normalized, prefix + 'self_attn.v_proj.weight')
            attended = _attention(query, key, value, heads, kv_heads, width)
            x = _add(x, self._project(attended, prefix + 'self_attn.o_proj.weight'))
            normalized = _rmsnorm(x, self.weights[prefix + 'post_attention_layernorm.weight'],
                                  config.rms_norm_eps)
            gated = _swiglu(self._project(normalized, prefix + 'mlp.gate_proj.weight'),
                            self._project(normalized, prefix + 'mlp.up_proj.weight'))
            x = _add(x, self._project(gated, prefix + 'mlp.down_proj.weight'))
        normalized = _rmsnorm(x, self.weights['model.norm.weight'], config.rms_norm_eps)
        head = config.tensor_aliases().get('lm_head.weight', 'lm_head.weight')
        return _matmul(normalized, self.weights[head])


def compare_logits(actual, expected, *, atol=1e-5, rtol=1e-4):
    """Report the measured deviation; the caller decides what it means."""
    if len(actual) != len(expected) or not actual:
        raise ValueError('logit shapes differ or are empty')
    worst_abs = worst_rel = 0.0
    for first, second in zip(actual, expected):
        if len(first) != len(second):
            raise ValueError('logit row widths differ')
        for left, right in zip(first, second):
            if not (math.isfinite(left) and math.isfinite(right)):
                raise ValueError('logits must be finite')
            error = abs(left - right)
            worst_abs = max(worst_abs, error)
            worst_rel = max(worst_rel, error / max(abs(right), 1e-12))
    allowed_failures = [
        (left, right) for first, second in zip(actual, expected)
        for left, right in zip(first, second) if abs(left - right) > atol + rtol * abs(right)]
    return {'passed': not allowed_failures, 'max_abs_error': worst_abs,
            'max_rel_error': worst_rel, 'atol': atol, 'rtol': rtol,
            'failing_elements': len(allowed_failures)}
