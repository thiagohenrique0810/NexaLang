"""Small independent Python Q3_GROUPED v1 codec for numerical test oracles.

This implementation does not import or call the native codec. Rows are encoded
headwise: little-endian F32 scale followed by G three-bit signed codes packed
from the least significant bit. The -4 code, nonzero padding and nonfinite
scales are invalid. Decoding retains exact double scale*code products.
"""
import math
import struct

MAX_GROUP_SIZE = 1 << 20


def _dimensions(cols, group_size):
    if type(cols) is not int or cols <= 0:
        raise ValueError('Q3 columns must be a positive integer')
    if type(group_size) is not int or not 0 < group_size <= MAX_GROUP_SIZE:
        raise ValueError(f'Q3 group size must be an integer from 1 to {MAX_GROUP_SIZE}')


def row_size(cols, group_size):
    _dimensions(cols, group_size)
    return ((cols + group_size - 1) // group_size) * (4 + (3 * group_size + 7) // 8)


def _f32(value):
    try:
        value = struct.unpack('<f', struct.pack('<f', float(value)))[0]
    except (TypeError, ValueError, OverflowError, struct.error) as error:
        raise ValueError('Q3 input must fit a finite F32') from error
    if not math.isfinite(value):
        raise ValueError('Q3 input must fit a finite F32')
    return value


def quantize_q3_row(values, group_size):
    """Encode one row after rounding input coordinates to F32."""
    values = [_f32(value) for value in values]
    _dimensions(len(values), group_size)
    payload_bytes = (3 * group_size + 7) // 8
    result = bytearray()
    for start in range(0, len(values), group_size):
        group = values[start:start + group_size]
        max_abs = max(abs(value) for value in group)
        scale = _f32(max_abs / 3.0)
        if max_abs and not scale:
            raise ArithmeticError('Q3 nonzero scale underflowed F32')
        codes = 0
        for index, value in enumerate(group):
            ratio = value / scale if scale else 0.0
            magnitude = math.floor(abs(ratio) + 0.5)
            quantized = max(-3, min(3, -magnitude if ratio < 0 else magnitude))
            codes |= (quantized & 7) << (3 * index)
        result += struct.pack('<f', scale)
        result += codes.to_bytes(payload_bytes, 'little')
    return bytes(result)


def decode_q3_row(data, cols, group_size):
    """Validate a complete row and return exact scale*code Python floats."""
    expected = row_size(cols, group_size)
    try:
        data = memoryview(data).cast('B')
    except (TypeError, ValueError) as error:
        raise ValueError('Q3 data must be a contiguous byte buffer') from error
    if len(data) != expected:
        raise ValueError(f'Q3 row requires {expected} bytes, got {len(data)}')
    payload_bytes = (3 * group_size + 7) // 8
    result = []
    for offset in range(0, expected, 4 + payload_bytes):
        scale = struct.unpack_from('<f', data, offset)[0]
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('Q3 scale must be finite and nonnegative')
        packed = int.from_bytes(data[offset + 4:offset + 4 + payload_bytes], 'little')
        if packed >> (3 * group_size):
            raise ValueError('Q3 unused high bits must be zero')
        count = min(group_size, cols - len(result))
        for index in range(group_size):
            code = (packed >> (3 * index)) & 7
            quantized = code if code < 4 else code - 8
            if quantized == -4:
                raise ValueError('Q3 code -4 is reserved')
            if quantized and (index >= count or not scale):
                raise ValueError('Q3 padding and zero-scale codes must be zero')
            if index < count:
                result.append(scale * quantized)
    return result
