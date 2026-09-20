"""NexaPack v1: bounded metadata and independently verified packed row blocks.

The 64-byte little-endian prefix is ``<8sHHIQQ32s``: magic, format version,
flags, JSON byte length, payload offset, total file size, and SHA-256 of JSON.
The JSON index precedes zero padding to a 4096-byte boundary. Payload blocks
are contiguous. Each index entry carries row bounds, absolute byte bounds,
and SHA-256. Checksums detect corruption; they do not authenticate a publisher.

Q4 groups contain a little-endian float32 scale followed by ceil(group_size/2)
bytes. The low nibble comes first, and signed values use two's complement in
[-7, 7]. Both missing coordinates and an unused high nibble are zero.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from dataclasses import dataclass
from typing import Iterable

MAGIC = b'NEXAPACK'
FORMAT_VERSION = 1
CODEC_ID = 'Q4_GROUPED'
CODEC_VERSION = 1
Q8_CODEC_ID = 'Q8_GROUPED'
Q8_CODEC_VERSION = 1
Q3_CODEC_ID = 'Q3_GROUPED'
Q3_CODEC_VERSION = 1
Q2_CODEC_ID = 'Q2_GROUPED'
Q2_CODEC_VERSION = 1
# Storage dtype per matrix codec; the container layout is otherwise identical.
_GROUPED_CODECS = {CODEC_ID: 'q4', Q8_CODEC_ID: 'q8', Q3_CODEC_ID: 'q3', Q2_CODEC_ID: 'q2'}
# Codes per group: the scale is max|v| / levels, and -(levels + 1) is reserved.
_CODEC_LEVELS = {CODEC_ID: 7, Q8_CODEC_ID: 127, Q3_CODEC_ID: 3, Q2_CODEC_ID: 1}
TQ_CODEC_ID = 'TQ_MSE_SRHT'
TQ_CODEC_VERSION = 1
TQ_TRANSFORM_ID = 'SRHT_XOSHIRO256SS_V1'
HEADER = struct.Struct('<8sHHIQQ32s')
ALIGNMENT = 4096
MAX_METADATA_BYTES = 1024 * 1024
MAX_BLOCKS = 8192
MAX_GROUP_SIZE = 1024 * 1024
MAX_ROW_BYTES = 64 * 1024 * 1024
MAX_READ_BYTES = 64 * 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024
MAX_INTEGER = (1 << 63) - 1
_FLOAT32 = struct.Struct('<f')
_SHA256 = re.compile(r'[0-9a-f]{64}')
_METADATA_KEYS = {
    'format', 'format_version', 'shape', 'logical_dtype', 'storage_dtype',
    'codec_id', 'codec_version', 'group_size', 'endianness', 'checksum',
    'row_bytes', 'block_rows', 'blocks',
}
_TQ_METADATA_KEYS = (_METADATA_KEYS - {'group_size'}) | {
    'bits', 'seed', 'transform_id', 'codebook_f32le',
}
_BLOCK_KEYS = {'start_row', 'row_count', 'offset', 'size', 'sha256'}


class NexaPackError(ValueError):
    """Invalid matrix, unsupported format, corrupted data, or exceeded limit."""


def _integer(value, name, *, minimum=1, maximum=MAX_INTEGER):
    if type(value) is not int or not minimum <= value <= maximum:
        raise NexaPackError(f'{name} must be an integer in [{minimum}, {maximum}]')
    return value


def _group_bytes(group_size, codec=CODEC_ID):
    _integer(group_size, 'group_size', maximum=MAX_GROUP_SIZE)
    if codec == Q8_CODEC_ID:
        # One signed byte per value, after the shared float32 scale.
        return 4 + group_size
    if codec == Q3_CODEC_ID:
        # Three bits per value, packed from the least significant bit, with
        # the same layout the paged KV cache already stores.
        return 4 + (3 * group_size + 7) // 8
    if codec == Q2_CODEC_ID:
        # Two bits per value: ternary codes -1, 0 and 1, with -2 reserved.
        return 4 + (2 * group_size + 7) // 8
    if codec != CODEC_ID:
        raise NexaPackError(f'Unsupported grouped codec: {codec}')
    return 4 + (group_size + 1) // 2


def _row_bytes(cols, group_size, codec=CODEC_ID):
    _integer(cols, 'cols')
    group_bytes = _group_bytes(group_size, codec)
    result = ((cols + group_size - 1) // group_size) * group_bytes
    if result > MAX_ROW_BYTES:
        raise NexaPackError(f'Encoded row exceeds {MAX_ROW_BYTES} bytes')
    return result


def _float32(value):
    try:
        converted = float(value)
        if not math.isfinite(converted):
            raise NexaPackError('Q4 input must contain finite float32 values')
        converted = _FLOAT32.unpack(_FLOAT32.pack(converted))[0]
    except (TypeError, ValueError, OverflowError, struct.error) as error:
        raise NexaPackError('Q4 input must contain finite float32 values') from error
    if not math.isfinite(converted):
        raise NexaPackError('Q4 input is outside the float32 range')
    return converted


def _encode_group(values, group_size):
    maximum = max(abs(value) for value in values)
    scale = _FLOAT32.unpack(_FLOAT32.pack(maximum / 7.0))[0]
    if maximum and not scale:
        raise NexaPackError('Q4 scale underflows float32; rescale the input')
    output = bytearray(_group_bytes(group_size))
    _FLOAT32.pack_into(output, 0, scale)
    if scale:
        for index, value in enumerate(values):
            quotient = value / scale
            magnitude = math.floor(abs(quotient) + 0.5)
            quantized = min(7, magnitude) * (-1 if quotient < 0 else 1)
            output[4 + index // 2] |= (quantized & 0xF) << (4 * (index % 2))
    return bytes(output)


def _encode_group_q8(values, group_size):
    """Symmetric int8 group: scale = max|v| / 127, code -128 stays reserved."""
    maximum = max(abs(value) for value in values)
    scale = _FLOAT32.unpack(_FLOAT32.pack(maximum / 127.0))[0]
    if maximum and not scale:
        raise NexaPackError('Q8 scale underflows float32; rescale the input')
    output = bytearray(_group_bytes(group_size, Q8_CODEC_ID))
    _FLOAT32.pack_into(output, 0, scale)
    if scale:
        for index, value in enumerate(values):
            quotient = value / scale
            magnitude = math.floor(abs(quotient) + 0.5)
            quantized = min(127, magnitude) * (-1 if quotient < 0 else 1)
            output[4 + index] = quantized & 0xFF
    return bytes(output)


def _encode_packed_bits(values, group_size, codec, bits):
    """Shared encoder for the bit-packed grouped codecs (Q3 and Q2)."""
    levels = _CODEC_LEVELS[codec]
    maximum = max(abs(value) for value in values)
    scale = _FLOAT32.unpack(_FLOAT32.pack(maximum / levels))[0]
    if maximum and not scale:
        raise NexaPackError(f'{codec} scale underflows float32; rescale the input')
    payload_bytes = _group_bytes(group_size, codec) - 4
    mask = (1 << bits) - 1
    codes = 0
    if scale:
        for index, value in enumerate(values):
            quotient = value / scale
            magnitude = math.floor(abs(quotient) + 0.5)
            quantized = max(-levels, min(levels, -magnitude if quotient < 0 else magnitude))
            codes |= (quantized & mask) << (bits * index)
    return _FLOAT32.pack(scale) + codes.to_bytes(payload_bytes, 'little')


def _iter_decoded_packed_bits(data, cols, group_size, codec, bits):
    size = _row_bytes(cols, group_size, codec)
    try:
        payload = memoryview(data).cast('B')
    except (TypeError, ValueError) as error:
        raise NexaPackError(f'{codec} row must be a contiguous byte buffer') from error
    if len(payload) != size:
        raise NexaPackError(f'{codec} row has {len(payload)} bytes; expected {size}')
    group_bytes = _group_bytes(group_size, codec)
    payload_bytes, reserved = group_bytes - 4, -(1 << (bits - 1))
    mask, decoded = (1 << bits) - 1, 0
    for offset in range(0, size, group_bytes):
        scale = _FLOAT32.unpack_from(payload, offset)[0]
        if not math.isfinite(scale) or scale < 0:
            raise NexaPackError(f'{codec} scale must be finite and nonnegative')
        packed = int.from_bytes(bytes(payload[offset + 4:offset + 4 + payload_bytes]), 'little')
        if packed >> (bits * group_size):
            raise NexaPackError(f'{codec} padding bits must be zero')
        count = min(group_size, cols - decoded)
        for index in range(group_size):
            code = (packed >> (bits * index)) & mask
            value = code if code < (1 << (bits - 1)) else code - (1 << bits)
            if value == reserved:
                raise NexaPackError(f'{codec} code {reserved} is reserved and invalid')
            if (index >= count or not scale) and value:
                raise NexaPackError(f'{codec} padding and zero-scale groups must contain zero codes')
            if index < count:
                yield scale * value
        decoded += count


def _encode_group_q2(values, group_size):
    """Q2_GROUPED v1: ternary codes -1, 0 and 1, scale = max|v|, -2 reserved."""
    return _encode_packed_bits(values, group_size, Q2_CODEC_ID, 2)


def validate_q2_row(data, cols: int, group_size: int) -> None:
    for _ in _iter_decoded_packed_bits(data, cols, group_size, Q2_CODEC_ID, 2):
        pass


def decode_q2_row(data: bytes, cols: int, group_size: int) -> list[float]:
    """Decode one Q2 row for reference and calibration, never for execution."""
    return list(_iter_decoded_packed_bits(data, cols, group_size, Q2_CODEC_ID, 2))


def quantize_q2_row(values: Iterable[float], group_size: int) -> bytes:
    return b''.join(_iter_grouped(values, group_size, None, Q2_CODEC_ID))


def write_q2_matrix(path, rows: int, cols: int, group_size: int,
                    row_source: Iterable[Iterable[float]], *, block_rows: int = 64) -> None:
    """Atomically write a row-major ternary Q2 matrix."""
    write_grouped_matrix(path, rows, cols, group_size, row_source,
                         block_rows=block_rows, codec=Q2_CODEC_ID)


def _encode_group_q3(values, group_size):
    """Q3_GROUPED v1: scale = max|v| / 3, signed codes in [-3, 3], -4 invalid."""
    maximum = max(abs(value) for value in values)
    scale = _FLOAT32.unpack(_FLOAT32.pack(maximum / 3.0))[0]
    if maximum and not scale:
        raise NexaPackError('Q3 scale underflows float32; rescale the input')
    payload_bytes = _group_bytes(group_size, Q3_CODEC_ID) - 4
    codes = 0
    if scale:
        for index, value in enumerate(values):
            quotient = value / scale
            magnitude = math.floor(abs(quotient) + 0.5)
            quantized = max(-3, min(3, -magnitude if quotient < 0 else magnitude))
            codes |= (quantized & 7) << (3 * index)
    return _FLOAT32.pack(scale) + codes.to_bytes(payload_bytes, 'little')


def _iter_decoded_q3_row(data, cols, group_size):
    size = _row_bytes(cols, group_size, Q3_CODEC_ID)
    try:
        payload = memoryview(data).cast('B')
    except (TypeError, ValueError) as error:
        raise NexaPackError('Q3 row must be a contiguous byte buffer') from error
    if len(payload) != size:
        raise NexaPackError(f'Q3 row has {len(payload)} bytes; expected {size}')
    group_bytes = _group_bytes(group_size, Q3_CODEC_ID)
    payload_bytes = group_bytes - 4
    decoded = 0
    for offset in range(0, size, group_bytes):
        scale = _FLOAT32.unpack_from(payload, offset)[0]
        if not math.isfinite(scale) or scale < 0:
            raise NexaPackError('Q3 scale must be finite and nonnegative')
        packed = int.from_bytes(bytes(payload[offset + 4:offset + 4 + payload_bytes]), 'little')
        if packed >> (3 * group_size):
            raise NexaPackError('Q3 padding bits must be zero')
        count = min(group_size, cols - decoded)
        for index in range(group_size):
            code = (packed >> (3 * index)) & 7
            value = code if code < 4 else code - 8
            if value == -4:
                raise NexaPackError('Q3 code -4 is reserved and invalid')
            if (index >= count or not scale) and value:
                raise NexaPackError('Q3 padding and zero-scale groups must contain zero codes')
            if index < count:
                yield scale * value
        decoded += count


def validate_q3_row(data, cols: int, group_size: int) -> None:
    for _ in _iter_decoded_q3_row(data, cols, group_size):
        pass


def decode_q3_row(data: bytes, cols: int, group_size: int) -> list[float]:
    """Decode one Q3 row for reference and calibration, never for execution."""
    return list(_iter_decoded_q3_row(data, cols, group_size))


def quantize_q3_row(values: Iterable[float], group_size: int) -> bytes:
    return b''.join(_iter_grouped(values, group_size, None, Q3_CODEC_ID))


def write_q3_matrix(path, rows: int, cols: int, group_size: int,
                    row_source: Iterable[Iterable[float]], *, block_rows: int = 64) -> None:
    """Atomically write a row-major Q3 matrix; three bits per coordinate."""
    write_grouped_matrix(path, rows, cols, group_size, row_source,
                         block_rows=block_rows, codec=Q3_CODEC_ID)


def _iter_decoded_q8_row(data, cols, group_size):
    size = _row_bytes(cols, group_size, Q8_CODEC_ID)
    try:
        payload = memoryview(data).cast('B')
    except (TypeError, ValueError) as error:
        raise NexaPackError('Q8 row must be a contiguous byte buffer') from error
    if len(payload) != size:
        raise NexaPackError(f'Q8 row has {len(payload)} bytes; expected {size}')
    group_bytes = _group_bytes(group_size, Q8_CODEC_ID)
    decoded = 0
    for offset in range(0, size, group_bytes):
        scale = _FLOAT32.unpack_from(payload, offset)[0]
        if not math.isfinite(scale) or scale < 0:
            raise NexaPackError('Q8 scale must be finite and nonnegative')
        count = min(group_size, cols - decoded)
        for index in range(group_size):
            code = payload[offset + 4 + index]
            value = code if code < 128 else code - 256
            if value == -128:
                raise NexaPackError('Q8 code -128 is reserved and invalid')
            if (index >= count or not scale) and value:
                raise NexaPackError('Q8 padding and zero-scale groups must contain zero codes')
            if index < count:
                yield scale * value
        decoded += count


def validate_q8_row(data, cols: int, group_size: int) -> None:
    for _ in _iter_decoded_q8_row(data, cols, group_size):
        pass


def decode_q8_row(data: bytes, cols: int, group_size: int) -> list[float]:
    """Decode one Q8 row for reference and calibration, never for execution."""
    return list(_iter_decoded_q8_row(data, cols, group_size))


def quantize_q8_row(values: Iterable[float], group_size: int) -> bytes:
    return b''.join(_iter_grouped(values, group_size, None, Q8_CODEC_ID))


def _iter_grouped(values, group_size, cols=None, codec=CODEC_ID):
    group_bytes = _group_bytes(group_size, codec)
    encode = {Q8_CODEC_ID: _encode_group_q8, Q3_CODEC_ID: _encode_group_q3,
              Q2_CODEC_ID: _encode_group_q2}.get(codec, _encode_group)
    if cols is not None:
        _row_bytes(cols, group_size, codec)
    try:
        iterator = iter(values)
    except TypeError as error:
        raise NexaPackError('Each matrix row must be iterable') from error
    seen = 0
    encoded_size = 0
    while cols is None or seen < cols:
        count = group_size if cols is None else min(group_size, cols - seen)
        group = [_float32(value) for value in itertools.islice(iterator, count)]
        if not group:
            if cols is not None and seen != cols:
                raise NexaPackError(f'Row has {seen} values; expected {cols}')
            break
        if cols is not None and len(group) != count:
            raise NexaPackError(f'Row has {seen + len(group)} values; expected {cols}')
        seen += len(group)
        encoded_size += group_bytes
        if encoded_size > MAX_ROW_BYTES:
            raise NexaPackError(f'Encoded row exceeds {MAX_ROW_BYTES} bytes')
        yield encode(group, group_size)
        if len(group) != count:
            break
    if not seen:
        raise NexaPackError('A matrix row must contain at least one value')
    if cols is not None:
        sentinel = object()
        if next(iterator, sentinel) is not sentinel:
            raise NexaPackError(f'Row contains more than {cols} values')


def quantize_q4_row(values: Iterable[float], group_size: int) -> bytes:
    """Encode one nonempty row; inputs are first rounded to finite float32.

    Scale is float32(max(abs(values))/7). Division uses that stored scale,
    rounding is half away from zero, and zero groups have zero scale/payload.
    A nonzero group whose scale rounds to zero is rejected.
    """
    return b''.join(_iter_grouped(values, group_size))


def _iter_decoded_q4_row(data, cols, group_size):
    """Shared codec checks, yielding one decoded value at a time."""
    size = _row_bytes(cols, group_size)
    try:
        payload = memoryview(data).cast('B')
    except (TypeError, ValueError) as error:
        raise NexaPackError('Q4 row must be a contiguous byte buffer') from error
    if len(payload) != size:
        raise NexaPackError(f'Q4 row has {len(payload)} bytes; expected {size}')
    group_bytes = _group_bytes(group_size)
    decoded = 0
    for offset in range(0, size, group_bytes):
        scale = _FLOAT32.unpack_from(payload, offset)[0]
        if not math.isfinite(scale) or scale < 0:
            raise NexaPackError('Q4 scale must be finite and nonnegative')
        count = min(group_size, cols - decoded)
        for index in range(((group_size + 1) // 2) * 2):
            code = (payload[offset + 4 + index // 2] >> (4 * (index % 2))) & 0xF
            value = code if code < 8 else code - 16
            if value == -8:
                raise NexaPackError('Q4 code -8 is reserved and invalid')
            if (index >= count or not scale) and value:
                raise NexaPackError('Q4 padding and zero-scale groups must contain zero codes')
            if index < count:
                yield scale * value
        decoded += count


def validate_q4_row(data, cols: int, group_size: int) -> None:
    """Validate scales, signed codes and padding with constant auxiliary space.

    The packed input is borrowed through a memoryview; no decoded row or copy
    of the packed row is retained. Callers perform container checksums separately.
    """
    for _ in _iter_decoded_q4_row(data, cols, group_size):
        pass


def decode_q4_row(data: bytes, cols: int, group_size: int) -> list[float]:
    """Decode one row for reference/testing; validate signed codes and padding.

    Products scale * code are returned as Python floats without an additional
    float32 rounding, matching the C kernel's double-precision accumulation.
    This helper is not used by the streaming reader or fused compute kernel.
    """
    return list(_iter_decoded_q4_row(data, cols, group_size))


def _json_bytes(metadata):
    encoded = json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(encoded) > MAX_METADATA_BYTES:
        raise NexaPackError(f'Metadata exceeds {MAX_METADATA_BYTES} bytes; increase block_rows')
    return encoded


def _align(value):
    return ((value + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT


def _new_metadata(rows, cols, group_size, block_rows, codec=CODEC_ID):
    _integer(rows, 'rows')
    _integer(block_rows, 'block_rows')
    if codec not in _GROUPED_CODECS:
        raise NexaPackError(f'Unsupported grouped codec: {codec}')
    row_bytes = _row_bytes(cols, group_size, codec)
    block_count = (rows + block_rows - 1) // block_rows
    if block_count > MAX_BLOCKS:
        raise NexaPackError(f'More than {MAX_BLOCKS} blocks; increase block_rows')
    blocks = [
        {'start_row': start, 'row_count': min(block_rows, rows - start),
         'offset': 0, 'size': min(block_rows, rows - start) * row_bytes, 'sha256': '0' * 64}
        for start in range(0, rows, block_rows)
    ]
    metadata = {
        'format': 'NexaPack', 'format_version': FORMAT_VERSION, 'shape': [rows, cols],
        'logical_dtype': 'f32', 'storage_dtype': _GROUPED_CODECS[codec], 'codec_id': codec,
        'codec_version': CODEC_VERSION, 'group_size': group_size,
        'endianness': 'little', 'checksum': 'sha256', 'row_bytes': row_bytes,
        'block_rows': block_rows, 'blocks': blocks,
    }
    # The checksum strings keep their width. Resolve header alignment before
    # consuming the source, so the writer only needs one streaming pass.
    payload_offset = _align(HEADER.size + len(_json_bytes(metadata)))
    while True:
        offset = payload_offset
        for block in blocks:
            block['offset'] = offset
            offset += block['size']
        if offset > MAX_INTEGER:
            raise NexaPackError('Encoded file size exceeds the v1 integer limit')
        wanted_offset = _align(HEADER.size + len(_json_bytes(metadata)))
        if wanted_offset == payload_offset:
            return metadata, payload_offset, offset
        payload_offset = wanted_offset


def write_q8_matrix(path, rows: int, cols: int, group_size: int,
                    row_source: Iterable[Iterable[float]], *, block_rows: int = 64) -> None:
    """Atomically write a row-major Q8 matrix; same container, wider codes."""
    write_grouped_matrix(path, rows, cols, group_size, row_source,
                         block_rows=block_rows, codec=Q8_CODEC_ID)


def write_q4_matrix(path, rows: int, cols: int, group_size: int,
                    row_source: Iterable[Iterable[float]], *, block_rows: int = 64) -> None:
    """Atomically write a row-major Q4 matrix without materializing the matrix.

    The source must yield exactly rows rows, each containing exactly cols
    float32-convertible values. Only one group and the bounded index are held
    internally. Existing output survives validation, source, and write errors.
    """
    write_grouped_matrix(path, rows, cols, group_size, row_source, block_rows=block_rows)


def write_grouped_matrix(path, rows: int, cols: int, group_size: int,
                         row_source: Iterable[Iterable[float]], *,
                         block_rows: int = 64, codec: str = CODEC_ID) -> None:
    """Shared streaming writer for the grouped codecs; one group at a time."""
    metadata, payload_offset, total_size = _new_metadata(rows, cols, group_size, block_rows, codec)
    try:
        source = iter(row_source)
    except TypeError as error:
        raise NexaPackError('row_source must be iterable') from error
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.nexapack-', dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'w+b') as stream:
            stream.seek(payload_offset)
            for block in metadata['blocks']:
                digest = hashlib.sha256()
                for row_index in range(block['start_row'], block['start_row'] + block['row_count']):
                    try:
                        values = next(source)
                    except StopIteration as error:
                        raise NexaPackError(f'Matrix ended at row {row_index}; expected {rows}') from error
                    for group in _iter_grouped(values, group_size, cols, codec):
                        stream.write(group)
                        digest.update(group)
                block['sha256'] = digest.hexdigest()
            sentinel = object()
            if next(source, sentinel) is not sentinel:
                raise NexaPackError(f'Matrix contains more than {rows} rows')
            encoded = _json_bytes(metadata)
            if _align(HEADER.size + len(encoded)) != payload_offset or stream.tell() != total_size:
                raise NexaPackError('Internal NexaPack layout mismatch')
            stream.seek(0)
            stream.write(HEADER.pack(MAGIC, FORMAT_VERSION, 0, len(encoded), payload_offset,
                                     total_size, hashlib.sha256(encoded).digest()))
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _new_tq_metadata(rows, cols, bits, seed, codebook_f32le, block_rows):
    # These helpers validate only Python data; inspection/migration must never
    # compile or load a native library. The actual quantizer is opened later.
    from .tq import tq_row_bytes, validate_tq_codebook, validate_tq_parameters
    try:
        validate_tq_parameters(cols, bits, seed)
        if codebook_f32le is not None:
            validate_tq_codebook(codebook_f32le, bits)
    except ValueError as error:
        raise NexaPackError(str(error)) from error
    _integer(rows, 'rows')
    _integer(block_rows, 'block_rows')
    row_bytes = tq_row_bytes(cols, bits)
    if codebook_f32le is None:
        # A generated codebook has this exact serialized width. Resolve index
        # limits and offsets before allocating a context or touching the source.
        codebook_f32le = '0' * (8 * (1 << bits))
    block_count = (rows + block_rows - 1) // block_rows
    if block_count > MAX_BLOCKS:
        raise NexaPackError(f'More than {MAX_BLOCKS} blocks; increase block_rows')
    blocks = [
        {'start_row': start, 'row_count': min(block_rows, rows - start),
         'offset': 0, 'size': min(block_rows, rows - start) * row_bytes, 'sha256': '0' * 64}
        for start in range(0, rows, block_rows)
    ]
    metadata = {
        'format': 'NexaPack', 'format_version': FORMAT_VERSION, 'shape': [rows, cols],
        'logical_dtype': 'f32', 'storage_dtype': 'tq_mse', 'codec_id': TQ_CODEC_ID,
        'codec_version': TQ_CODEC_VERSION, 'bits': bits, 'seed': seed,
        'transform_id': TQ_TRANSFORM_ID, 'codebook_f32le': codebook_f32le,
        'endianness': 'little', 'checksum': 'sha256', 'row_bytes': row_bytes,
        'block_rows': block_rows, 'blocks': blocks,
    }
    payload_offset = _align(HEADER.size + len(_json_bytes(metadata)))
    while True:
        offset = payload_offset
        for block in blocks:
            block['offset'] = offset
            offset += block['size']
        if offset > MAX_INTEGER:
            raise NexaPackError('Encoded file size exceeds the v1 integer limit')
        wanted_offset = _align(HEADER.size + len(_json_bytes(metadata)))
        if wanted_offset == payload_offset:
            return metadata, payload_offset, offset
        payload_offset = wanted_offset


def _write_tq_rows(path, layout, row_source, encode_row):
    """Publish validated rows atomically with one borrowed/encoded row live."""
    metadata, payload_offset, total_size = layout
    rows = metadata['shape'][0]
    try:
        source = iter(row_source)
    except TypeError as error:
        raise NexaPackError('row_source must be iterable') from error
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.nexapack-', dir=destination.parent)
    values = record = view = None
    try:
        # Unbuffered output prevents an additional implicit row/staging buffer.
        with os.fdopen(descriptor, 'w+b', buffering=0) as stream:
            stream.seek(payload_offset)
            for block in metadata['blocks']:
                digest = hashlib.sha256()
                for row_index in range(block['start_row'], block['start_row'] + block['row_count']):
                    try:
                        values = next(source)
                    except StopIteration as error:
                        raise NexaPackError(f'Matrix ended at row {row_index}; expected {rows}') from error
                    try:
                        record = encode_row(values)
                    except ValueError as error:
                        raise NexaPackError(str(error)) from error
                    view = memoryview(record).cast('B')
                    written = 0
                    while written < len(view):
                        count = stream.write(view[written:])
                        if not count:
                            raise OSError('Incomplete NexaPack row write')
                        written += count
                    digest.update(view)
                    view.release()
                    # Drop this row before encoding the next, including when
                    # encode_row itself needs both native and returned buffers.
                    values = record = view = None
                block['sha256'] = digest.hexdigest()
            sentinel = object()
            if next(source, sentinel) is not sentinel:
                raise NexaPackError(f'Matrix contains more than {rows} rows')
            encoded = _json_bytes(metadata)
            if _align(HEADER.size + len(encoded)) != payload_offset or stream.tell() != total_size:
                raise NexaPackError('Internal NexaPack layout mismatch')
            stream.seek(0)
            prefix = HEADER.pack(MAGIC, FORMAT_VERSION, 0, len(encoded), payload_offset,
                                 total_size, hashlib.sha256(encoded).digest())
            # Prefix/JSON are bounded metadata, outside the managed payload
            # budget. FileIO can theoretically short-write either of them.
            for payload in (prefix, encoded):
                offset = 0
                while offset < len(payload):
                    count = stream.write(memoryview(payload)[offset:])
                    if not count:
                        raise OSError('Incomplete NexaPack header write')
                    offset += count
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if view is not None:
            view.release()
        values = record = view = source = None
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_tq_matrix(path, rows: int, cols: int, bits: int, seed: int,
                    row_source: Iterable[Iterable[float]], *, block_rows: int = 64,
                    memory_budget=None, codebook_f32le=None) -> dict:
    """Write portable TQ02 rows, retaining one vector rather than a matrix.

    Codebook centroids are persisted exactly as little-endian float32 bytes;
    the seed identifies the SRHT signs, not platform-dependent Lloyd-Max output.
    Schema/layout and codec budget admission precede source consumption. The
    returned memory report covers the codec's managed native buffers, excluding
    Python metadata, caller-owned rows and file-system cache (not process RSS).
    """
    from .tq import TQCodec
    layout = _new_tq_metadata(rows, cols, bits, seed, codebook_f32le, block_rows)
    with TQCodec(cols, bits, seed, codebook_f32le=codebook_f32le,
                 memory_budget=memory_budget) as codec:
        layout[0]['codebook_f32le'] = codec.codebook_f32le
        report = codec.memory_report()
        _write_tq_rows(path, layout, row_source, codec.encode_row)
    return report


def write_tq_records(path, rows: int, cols: int, bits: int, seed: int,
                     codebook_f32le: str, row_source, *, block_rows: int = 64) -> None:
    """Store existing TQ02 rows without native code or re-quantization.

    This is also the publication stage for an explicitly endian-converted
    legacy TQ01 stream. Rows are borrowed and validated with constant scratch;
    the caller is responsible for associating them with the correct codebook.
    """
    from .tq import validate_tq_codebook, validate_tq_row
    try:
        validate_tq_codebook(codebook_f32le, bits)
    except ValueError as error:
        raise NexaPackError(str(error)) from error
    layout = _new_tq_metadata(rows, cols, bits, seed, codebook_f32le, block_rows)

    def validated_record(record):
        validate_tq_row(record, cols, bits)
        return record

    _write_tq_rows(path, layout, row_source, validated_record)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise NexaPackError(f'Duplicate metadata key: {key}')
        result[key] = value
    return result


def _reject_json_constant(value):
    raise NexaPackError(f'Invalid JSON constant: {value}')


@dataclass(frozen=True)
class _Block:
    start_row: int
    row_count: int
    offset: int
    size: int
    sha256: str


class NexaPackReader:
    """Read only requested row blocks, verifying SHA-256 before returning data.

    Opening reads at most the bounded metadata area; no payload is loaded or
    mapped eagerly. Each read_rows call is capped at 64 MiB of returned bytes.
    Use repeated requests for larger ranges. Checksums for intersecting blocks
    are streamed in 64 KiB chunks, including when only part of a block is asked
    for. No external paths or object deserialization exist in the format.
    """
    def __init__(self, path):
        # Unbuffered reads keep payload scratch accounting explicit: one fixed
        # READ_CHUNK_BYTES bytearray, without a hidden BufferedReader buffer.
        self._stream = open(path, 'rb', buffering=0)
        self._payload_bytes_read = 0
        try:
            self._load_metadata()
        except BaseException:
            self._stream.close()
            raise

    def _load_metadata(self):
        size = os.fstat(self._stream.fileno()).st_size
        prefix = self._stream.read(HEADER.size)
        if len(prefix) != HEADER.size:
            raise NexaPackError('Truncated NexaPack header')
        magic, version, flags, length, payload_offset, total_size, expected_digest = HEADER.unpack(prefix)
        if magic != MAGIC or version != FORMAT_VERSION or flags:
            raise NexaPackError('Unsupported NexaPack magic, version, or flags')
        if not 0 < length <= MAX_METADATA_BYTES:
            raise NexaPackError('Invalid or oversized metadata length')
        if payload_offset != _align(HEADER.size + length):
            raise NexaPackError('Invalid NexaPack payload offset')
        if total_size != size or not payload_offset < total_size <= MAX_INTEGER:
            raise NexaPackError('Truncated file or inconsistent file size')
        encoded = self._stream.read(length)
        if len(encoded) != length or hashlib.sha256(encoded).digest() != expected_digest:
            raise NexaPackError('Metadata checksum mismatch')
        try:
            metadata = json.loads(encoded.decode('utf-8'), object_pairs_hook=_unique_object,
                                  parse_constant=_reject_json_constant)
        except (UnicodeError, ValueError, RecursionError) as error:
            raise NexaPackError(f'Invalid NexaPack JSON: {error}') from error
        if not isinstance(metadata, dict):
            raise NexaPackError('Unexpected NexaPack metadata fields')
        codec = metadata.get('codec_id')
        if codec in _GROUPED_CODECS:
            keys, storage_dtype = _METADATA_KEYS, _GROUPED_CODECS[codec]
        elif codec == TQ_CODEC_ID:
            keys, storage_dtype = _TQ_METADATA_KEYS, 'tq_mse'
        else:
            raise NexaPackError('Unsupported metadata codec_id')
        if set(metadata) != keys:
            raise NexaPackError('Unexpected NexaPack metadata fields')
        expected = {'format': 'NexaPack', 'format_version': FORMAT_VERSION,
                    'logical_dtype': 'f32', 'storage_dtype': storage_dtype, 'codec_id': codec,
                    'codec_version': CODEC_VERSION if codec in _GROUPED_CODECS else TQ_CODEC_VERSION,
                    'endianness': 'little', 'checksum': 'sha256'}
        if codec == TQ_CODEC_ID:
            expected['transform_id'] = TQ_TRANSFORM_ID
        for name, value in expected.items():
            if type(metadata[name]) is not type(value) or metadata[name] != value:
                raise NexaPackError(f'Unsupported metadata {name}')
        shape = metadata['shape']
        if not isinstance(shape, list) or len(shape) != 2:
            raise NexaPackError('shape must contain [rows, cols]')
        self._rows = _integer(shape[0], 'rows')
        self._cols = _integer(shape[1], 'cols')
        self._codec_id = codec
        self._codec_version = metadata['codec_version']
        self._group_size = self._bits = self._seed = self._codebook_f32le = None
        if codec in _GROUPED_CODECS:
            self._group_size = _integer(metadata['group_size'], 'group_size', maximum=MAX_GROUP_SIZE)
            self._row_bytes = _row_bytes(self._cols, self._group_size, codec)
        else:
            from .tq import tq_row_bytes, validate_tq_codebook, validate_tq_parameters
            try:
                validate_tq_parameters(self._cols, metadata['bits'], metadata['seed'])
                validate_tq_codebook(metadata['codebook_f32le'], metadata['bits'])
            except ValueError as error:
                raise NexaPackError(str(error)) from error
            self._bits = metadata['bits']
            self._seed = metadata['seed']
            self._codebook_f32le = metadata['codebook_f32le']
            self._row_bytes = tq_row_bytes(self._cols, self._bits)
        if _integer(metadata['row_bytes'], 'row_bytes') != self._row_bytes:
            raise NexaPackError('row_bytes does not match the matrix shape and codec')
        self._block_rows = _integer(metadata['block_rows'], 'block_rows')
        blocks = metadata['blocks']
        expected_count = (self._rows + self._block_rows - 1) // self._block_rows
        if not isinstance(blocks, list) or not 0 < len(blocks) <= MAX_BLOCKS or len(blocks) != expected_count:
            raise NexaPackError('Invalid block index length')
        next_row, next_offset = 0, payload_offset
        validated = []
        for block in blocks:
            if not isinstance(block, dict) or set(block) != _BLOCK_KEYS:
                raise NexaPackError('Unexpected block index fields')
            start = _integer(block['start_row'], 'start_row', minimum=0)
            count = _integer(block['row_count'], 'row_count')
            offset = _integer(block['offset'], 'offset')
            block_size = _integer(block['size'], 'size')
            checksum = block['sha256']
            if not isinstance(checksum, str) or not _SHA256.fullmatch(checksum):
                raise NexaPackError('Invalid block checksum')
            if start != next_row or count != min(self._block_rows, self._rows - next_row):
                raise NexaPackError('Overlapping, missing, or out-of-order matrix rows')
            if offset != next_offset or block_size != count * self._row_bytes or offset + block_size > total_size:
                raise NexaPackError('Overlapping, truncated, or invalid block byte range')
            validated.append(_Block(start, count, offset, block_size, checksum))
            next_row += count
            next_offset += block_size
        if next_row != self._rows or next_offset != total_size:
            raise NexaPackError('Block index does not cover the complete payload')
        padding = self._stream.read(payload_offset - HEADER.size - length)
        if any(padding):
            raise NexaPackError('Nonzero metadata padding')
        self._metadata = metadata
        self._blocks = tuple(validated)

    @property
    def metadata(self):
        # Return a detached copy; callers cannot change the validated I/O plan.
        return json.loads(json.dumps(self._metadata))

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def group_size(self):
        return self._group_size

    @property
    def codec_id(self):
        return self._codec_id

    @property
    def codec_version(self):
        return self._codec_version

    @property
    def bits(self):
        return self._bits

    @property
    def seed(self):
        return self._seed

    @property
    def codebook_f32le(self):
        return self._codebook_f32le

    def validate_row(self, data) -> None:
        """Validate codec structure without native code or decoded row storage.

        This complements container checksums. Like the original Q4 reader,
        read_rows returns packed bytes; callers select when to validate values.
        """
        if self.codec_id == CODEC_ID:
            validate_q4_row(data, self.cols, self.group_size)
        elif self.codec_id == Q8_CODEC_ID:
            validate_q8_row(data, self.cols, self.group_size)
        elif self.codec_id == Q3_CODEC_ID:
            validate_q3_row(data, self.cols, self.group_size)
        elif self.codec_id == Q2_CODEC_ID:
            validate_q2_row(data, self.cols, self.group_size)
        else:
            from .tq import validate_tq_row
            try:
                validate_tq_row(data, self.cols, self.bits)
            except ValueError as error:
                raise NexaPackError(str(error)) from error

    @property
    def row_bytes(self):
        return self._row_bytes

    @property
    def payload_bytes_read(self):
        """Actual payload bytes read, including checksum reads outside returned rows."""
        return self._payload_bytes_read

    def _request_size(self, start, count):
        if self._stream.closed:
            raise NexaPackError('NexaPack reader is closed')
        _integer(start, 'start', minimum=0)
        _integer(count, 'count', minimum=0)
        if start > self.rows or count > self.rows - start:
            raise NexaPackError('Requested rows are outside the matrix')
        requested_size = count * self.row_bytes
        if requested_size > MAX_READ_BYTES:
            raise NexaPackError(f'Row request exceeds {MAX_READ_BYTES} bytes; use smaller batches')
        return requested_size

    def read_rows(self, start: int, count: int) -> bytes:
        """Convenience allocating API; use read_rows_into for a caller-owned arena."""
        output = bytearray(self._request_size(start, count))
        self.read_rows_into(start, count, output)
        return bytes(output)

    def read_rows_into(self, start: int, count: int, destination) -> int:
        """Fill an exact-size writable contiguous byte buffer and return byte count.

        Only one fixed READ_CHUNK_BYTES scratch buffer is allocated. Destination
        contents are unspecified after an exception and must be discarded.
        """
        requested_size = self._request_size(start, count)
        try:
            output = memoryview(destination).cast('B')
        except (TypeError, ValueError) as error:
            raise NexaPackError('Destination must be a contiguous writable byte buffer') from error
        scratch = scratch_view = chunk = None
        try:
            if output.readonly or len(output) != requested_size:
                raise NexaPackError(f'Destination must be writable and exactly {requested_size} bytes')
            if not count:
                return 0
            scratch = bytearray(READ_CHUNK_BYTES)
            scratch_view = memoryview(scratch)
            written = 0
            end = start + count
            for block in self._blocks[start // self._block_rows:(end - 1) // self._block_rows + 1]:
                digest = hashlib.sha256()
                low = max(start - block.start_row, 0) * self.row_bytes
                high = min(end - block.start_row, block.row_count) * self.row_bytes
                self._stream.seek(block.offset)
                consumed = 0
                while consumed < block.size:
                    chunk = scratch_view[:min(READ_CHUNK_BYTES, block.size - consumed)]
                    received = self._stream.readinto(chunk)
                    if not received:
                        raise NexaPackError('Payload truncated while reading a block')
                    self._payload_bytes_read += received
                    chunk = chunk[:received]
                    digest.update(chunk)
                    left = max(low - consumed, 0)
                    right = min(high - consumed, received)
                    if left < right:
                        size = right - left
                        output[written:written + size] = chunk[left:right]
                        written += size
                    consumed += received
                if digest.hexdigest() != block.sha256:
                    raise NexaPackError(f'Block checksum mismatch at row {block.start_row}')
            if written != requested_size:
                raise NexaPackError('Incomplete row request')
            return written
        finally:
            # An exception kept by a caller retains this frame. Release every
            # view and owner now so retries cannot accumulate 64 KiB scratches.
            if chunk is not None:
                chunk.release()
            if scratch_view is not None:
                scratch_view.release()
            output.release()
            scratch = scratch_view = chunk = output = None

    def close(self):
        self._stream.close()

    def __enter__(self):
        if self._stream.closed:
            raise NexaPackError('NexaPack reader is closed')
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
