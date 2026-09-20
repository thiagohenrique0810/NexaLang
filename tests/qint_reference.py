"""Independent Python oracle for the qint<N>/PackedVector<N> storage ABI.

Written from the prose of `docs/NEXAPACK_V1.md` and `docs/NEXALM_CODECS_PESOS.md`,
never from `runtime/nexapack/format.py` or the native kernels: a test whose
oracle is the implementation only proves the implementation is itself.

The four widths the language exposes are one layout read four ways. A packed
vector is a sequence of group records. Each record is a little-endian IEEE
binary32 scale followed by `group_size` two's-complement codes of `bits` bits
each, packed from the least significant bit of the record payload -- which is
what "two Q4 values per byte, the first in the low nibble" means once it is
written for an arbitrary width. Codes run over [-L, L] with
L = 2**(bits-1) - 1, which is 1 for Q2, 3 for Q3, 7 for Q4 and 127 for Q8, and
-2**(bits-1) stays reserved. The scale is float32(max(abs(group)) / L); an
all-zero group stores a zero scale and a zero payload. The last group encodes
only the values that exist: its remaining code slots, and the spare high bits
of its last payload byte, are zero.
"""
import math
import struct

WIDTHS = (2, 3, 4, 8)
MAX_GROUP_SIZE = 1 << 20


def levels(bits):
    """Largest magnitude a code may carry at this width."""
    _width(bits)
    return (1 << (bits - 1)) - 1


def _width(bits):
    if type(bits) is not int or bits not in WIDTHS:
        raise ValueError(f'qint width must be one of {WIDTHS}, got {bits!r}')
    return bits


def _dimensions(bits, count, group_size):
    _width(bits)
    if type(count) is not int or count <= 0:
        raise ValueError('qint value count must be a positive integer')
    if type(group_size) is not int or not 0 < group_size <= MAX_GROUP_SIZE:
        raise ValueError(f'qint group size must be an integer from 1 to {MAX_GROUP_SIZE}')


def group_bytes(bits, group_size):
    """Scale plus the bits of one full group, rounded up to whole bytes."""
    _dimensions(bits, 1, group_size)
    return 4 + (bits * group_size + 7) // 8


def group_count(bits, count, group_size):
    _dimensions(bits, count, group_size)
    return (count + group_size - 1) // group_size


def packed_size(bits, count, group_size):
    return group_count(bits, count, group_size) * group_bytes(bits, group_size)


def _f32(value):
    try:
        result = struct.unpack('<f', struct.pack('<f', float(value)))[0]
    except (TypeError, ValueError, OverflowError, struct.error) as error:
        raise ValueError('qint input must fit a finite F32') from error
    if not math.isfinite(result):
        raise ValueError('qint input must fit a finite F32')
    return result


def pack(values, bits, group_size):
    """Encode a whole vector after rounding every input to F32."""
    values = [_f32(value) for value in values]
    _dimensions(bits, len(values), group_size)
    limit = levels(bits)
    payload_bytes = group_bytes(bits, group_size) - 4
    mask = (1 << bits) - 1
    result = bytearray()
    for start in range(0, len(values), group_size):
        group = values[start:start + group_size]
        largest = max(abs(value) for value in group)
        scale = _f32(largest / limit)
        if largest and not scale:
            raise ArithmeticError(f'qint<{bits}> nonzero scale underflowed F32')
        codes = 0
        for index, value in enumerate(group):
            ratio = value / scale if scale else 0.0
            magnitude = min(limit, math.floor(abs(ratio) + 0.5))
            code = -magnitude if ratio < 0 else magnitude
            codes |= (code & mask) << (bits * index)
        result += struct.pack('<f', scale)
        result += codes.to_bytes(payload_bytes, 'little')
    return bytes(result)


def unpack(data, bits, count, group_size):
    """Validate a whole packed vector and return exact scale*code products."""
    expected = packed_size(bits, count, group_size)
    try:
        data = memoryview(data).cast('B')
    except (TypeError, ValueError) as error:
        raise ValueError('qint data must be a contiguous byte buffer') from error
    if len(data) != expected:
        raise ValueError(f'qint vector requires {expected} bytes, got {len(data)}')
    record_bytes = group_bytes(bits, group_size)
    payload_bytes = record_bytes - 4
    reserved = -(1 << (bits - 1))
    mask = (1 << bits) - 1
    result = []
    for offset in range(0, expected, record_bytes):
        scale = struct.unpack_from('<f', data, offset)[0]
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('qint scale must be finite and nonnegative')
        packed = int.from_bytes(data[offset + 4:offset + record_bytes], 'little')
        if packed >> (bits * group_size):
            raise ValueError('qint unused high bits must be zero')
        present = min(group_size, count - len(result))
        for index in range(group_size):
            raw = (packed >> (bits * index)) & mask
            code = raw if raw < (1 << (bits - 1)) else raw - (1 << bits)
            if code == reserved:
                raise ValueError(f'qint<{bits}> code {reserved} is reserved')
            if code and (index >= present or not scale):
                raise ValueError('qint padding and zero-scale codes must be zero')
            if index < present:
                result.append(scale * code)
    return result


def codes(data, bits, count, group_size):
    """The stored level codes themselves, which is what a qint<N> holds."""
    unpack(data, bits, count, group_size)
    record_bytes = group_bytes(bits, group_size)
    mask = (1 << bits) - 1
    result = []
    for offset in range(0, len(data), record_bytes):
        packed = int.from_bytes(bytes(memoryview(data).cast('B')[offset + 4:offset + record_bytes]), 'little')
        for index in range(group_size):
            if len(result) == count:
                return result
            raw = (packed >> (bits * index)) & mask
            result.append(raw if raw < (1 << (bits - 1)) else raw - (1 << bits))
    return result


def bits_per_value(bits, count, group_size):
    """Measured cost: every byte the layout really writes, per stored value."""
    return 8.0 * packed_size(bits, count, group_size) / count
