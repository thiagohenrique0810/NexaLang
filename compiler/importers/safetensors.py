"""Bounded local Safetensors reader; no pickle, remote code, or dependencies.

Format reference: https://github.com/safetensors/safetensors/blob/main/README.md#format
Only the F32, F16 and BF16 little-endian row-major dtypes are supported here.
The header/index limit is intentionally stricter than the upstream 100 MB cap.
Tensor NaNs/infinities are legal Safetensors and are passed through; importers
and quantizers decide whether their destination representation permits them.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct

MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_TENSORS = 100000
MAX_SHARDS = 4096
MAX_RANK = 32
MAX_INTEGER = (1 << 63) - 1
READ_CHUNK_BYTES = 65536
_WIDTHS = {'F32': 4, 'F16': 2, 'BF16': 2}


class SafeTensorError(ValueError):
    """Invalid, unsupported, changed, or unsafe local checkpoint."""


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SafeTensorError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _constant(value):
    raise SafeTensorError(f'Invalid JSON constant: {value}')


def _json(data, label):
    try:
        return json.loads(data.decode('utf-8'), object_pairs_hook=_object,
                          parse_constant=_constant)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise SafeTensorError(f'Invalid {label}: {error}') from error


def read_json(path, *, max_bytes=MAX_HEADER_BYTES):
    """Read bounded strict local JSON, rejecting duplicate keys and NaN."""
    with open(path, 'rb') as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise SafeTensorError(f'JSON exceeds {max_bytes} bytes: {Path(path).name}')
    return _json(data, Path(path).name)


def safe_child(directory, relative):
    """Resolve a local relative path, rejecting traversal and escaping symlinks."""
    if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative:
        raise SafeTensorError('Checkpoint filenames must be relative POSIX paths')
    if any(part in ('', '.', '..') for part in relative.split('/')):
        raise SafeTensorError(f'Unsafe checkpoint filename: {relative}')
    root = Path(directory).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise SafeTensorError(f'Checkpoint path escapes its directory or is not a file: {relative}')
    return path


def _integer(value, name):
    if type(value) is not int or not 0 <= value <= MAX_INTEGER:
        raise SafeTensorError(f'{name} must be a nonnegative bounded integer')
    return value


def _signature(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def file_sha256(path):
    """Hash a complete local file in fixed chunks, detecting concurrent changes."""
    digest = hashlib.sha256()
    scratch = bytearray(READ_CHUNK_BYTES)
    view = memoryview(scratch)
    with open(path, 'rb', buffering=0) as stream:
        before = _signature(os.fstat(stream.fileno()))
        while received := stream.readinto(scratch):
            digest.update(view[:received])
        if _signature(os.fstat(stream.fileno())) != before or _signature(Path(path).stat()) != before:
            raise SafeTensorError(f'File changed while hashing: {Path(path).name}')
    return digest.hexdigest()


@dataclass(frozen=True)
class _Tensor:
    dtype: str
    shape: tuple
    begin: int
    end: int


class SafeTensorReader:
    """Validate a file header eagerly; decode tensor values only on demand.

    iter_rows yields lazy row iterators, including one row for a vector/scalar.
    Consume rows before closing the reader. Higher ranks flatten their leading
    dimensions; no tensor or complete row is materialized by this reader.
    """
    def __init__(self, path):
        self.path = Path(path)
        self._stream = open(path, 'rb', buffering=0)
        self.payload_bytes_read = 0
        try:
            self._load()
        except BaseException:
            self._stream.close()
            raise

    def _load(self):
        stat = os.fstat(self._stream.fileno())
        self._signature = _signature(stat)
        prefix = self._stream.read(8)
        if len(prefix) != 8:
            raise SafeTensorError('Truncated Safetensors length prefix')
        length, = struct.unpack('<Q', prefix)
        if not 2 <= length <= MAX_HEADER_BYTES or length + 8 > stat.st_size:
            raise SafeTensorError('Invalid, oversized, or truncated Safetensors header')
        data = self._stream.read(length)
        if len(data) != length or not data.startswith(b'{'):
            raise SafeTensorError('Safetensors JSON header must start with {')
        header = _json(data, 'Safetensors header')
        if not isinstance(header, dict) or len(header) > MAX_TENSORS + 1:
            raise SafeTensorError('Invalid or oversized tensor index')
        metadata = header.pop('__metadata__', {})
        if not isinstance(metadata, dict) or any(not isinstance(value, str) for value in metadata.values()):
            raise SafeTensorError('__metadata__ must be a string-to-string map')
        self._metadata = metadata
        self._data_offset = 8 + length
        tensors = {}
        for name, entry in header.items():
            if not isinstance(entry, dict) or set(entry) != {'dtype', 'shape', 'data_offsets'}:
                raise SafeTensorError(f'Invalid tensor descriptor: {name}')
            dtype, shape, offsets = entry['dtype'], entry['shape'], entry['data_offsets']
            if not isinstance(dtype, str) or dtype not in _WIDTHS:
                raise SafeTensorError(f'Unsupported tensor dtype for {name}: {dtype!r}')
            if not isinstance(shape, list) or len(shape) > MAX_RANK:
                raise SafeTensorError(f'Invalid tensor shape: {name}')
            for size in shape:
                _integer(size, 'dimension')
            elements = math.prod(shape)
            if elements > MAX_INTEGER:
                raise SafeTensorError(f'Tensor shape overflows the supported range: {name}')
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise SafeTensorError(f'Invalid tensor offsets: {name}')
            begin, end = (_integer(offset, 'data offset') for offset in offsets)
            if end < begin or end - begin != elements * _WIDTHS[dtype]:
                raise SafeTensorError(f'Tensor byte size does not match shape and dtype: {name}')
            tensors[name] = _Tensor(dtype, tuple(shape), begin, end)
        if len(tensors) > MAX_TENSORS:
            raise SafeTensorError('Too many tensors')
        cursor = 0
        for tensor in sorted(tensors.values(), key=lambda tensor: (tensor.begin, tensor.end)):
            if tensor.begin != cursor:
                raise SafeTensorError('Tensor data contains overlapping ranges or gaps')
            cursor = tensor.end
        if self._data_offset + cursor != stat.st_size:
            raise SafeTensorError('Tensor payload is truncated or contains trailing data')
        self._tensors = tensors

    @property
    def tensors(self):
        return {name: {'dtype': tensor.dtype, 'shape': tensor.shape,
                       'data_offsets': (tensor.begin, tensor.end)}
                for name, tensor in self._tensors.items()}

    @property
    def metadata(self):
        return dict(self._metadata)

    def _unchanged(self):
        if self._stream.closed:
            raise SafeTensorError('Safetensors reader is closed')
        if _signature(os.fstat(self._stream.fileno())) != self._signature:
            raise SafeTensorError('Safetensors file changed after its header was validated')

    def _values(self, offset, elements, dtype):
        width = _WIDTHS[dtype]
        scratch = bytearray(READ_CHUNK_BYTES)
        view = memoryview(scratch)
        self._unchanged()
        remaining = elements * width
        while remaining:
            size = min(remaining, READ_CHUNK_BYTES)
            self._stream.seek(offset)
            received = 0
            while received < size:
                count = self._stream.readinto(view[received:size])
                if not count:
                    raise SafeTensorError('Truncated tensor payload')
                received += count
                self.payload_bytes_read += count
            code = {'F32': '<f', 'F16': '<e', 'BF16': '<H'}[dtype]
            for value, in struct.iter_unpack(code, view[:size]):
                yield struct.unpack('<f', struct.pack('<I', value << 16))[0] if dtype == 'BF16' else value
            offset += size
            remaining -= size
        self._unchanged()

    def iter_rows(self, name):
        self._unchanged()
        if name not in self._tensors:
            raise SafeTensorError(f'Unknown tensor: {name}')
        tensor = self._tensors[name]
        cols = tensor.shape[-1] if tensor.shape else 1
        rows = math.prod(tensor.shape[:-1]) if tensor.shape else 1
        for row in range(rows):
            offset = self._data_offset + tensor.begin + row * cols * _WIDTHS[tensor.dtype]
            yield self._values(offset, cols, tensor.dtype)
        self._unchanged()

    def close(self):
        self._stream.close()

    def __enter__(self):
        self._unchanged()
        return self

    def __exit__(self, *_):
        self.close()


class SafeTensorCheckpoint:
    """Validated local single-file or indexed checkpoint; at most one open shard.

    No weight payload is read at construction. File hashes are computed only
    when provenance(hash_files=True) is requested and identify entire files,
    never pretend to be tensor checksums.
    """
    def __init__(self, directory):
        self.directory = Path(directory).resolve(strict=True)
        if not self.directory.is_dir():
            raise SafeTensorError('Checkpoint source must be a directory')
        index_name = 'model.safetensors.index.json'
        single_name = 'model.safetensors'
        has_index = (self.directory / index_name).exists()
        has_single = (self.directory / single_name).exists()
        if has_index == has_single:
            raise SafeTensorError('Expected exactly one model.safetensors or model.safetensors.index.json')
        self._paths, self._signatures, self._tensors = {}, {}, {}
        self._index_path = None
        if has_single:
            assignments = None
            metadata = {}
            filenames = [single_name]
        else:
            self._index_path = safe_child(self.directory, index_name)
            self._index_signature = _signature(self._index_path.stat())
            index = read_json(self._index_path)
            if not isinstance(index, dict) or set(index) - {'metadata', 'weight_map'} or 'weight_map' not in index:
                raise SafeTensorError('Invalid checkpoint index fields')
            metadata = index.get('metadata', {})
            if not isinstance(metadata, dict):
                raise SafeTensorError('Checkpoint index metadata must be an object')
            assignments = index['weight_map']
            if not isinstance(assignments, dict) or not 0 < len(assignments) <= MAX_TENSORS:
                raise SafeTensorError('Invalid checkpoint weight_map')
            if any(not isinstance(value, str) or not value.endswith('.safetensors') for value in assignments.values()):
                raise SafeTensorError('Every shard filename must end in .safetensors')
            filenames = sorted(set(assignments.values()))
            if len(filenames) > MAX_SHARDS:
                raise SafeTensorError('Too many checkpoint shards')
        for filename in filenames:
            path = safe_child(self.directory, filename)
            if path in self._paths.values():
                raise SafeTensorError('Several shard names resolve to the same file')
            with SafeTensorReader(path) as reader:
                self._paths[filename] = path
                self._signatures[filename] = reader._signature
                for name, descriptor in reader.tensors.items():
                    if name in self._tensors or assignments is not None and assignments.get(name) != filename:
                        raise SafeTensorError(f'Duplicate tensor or incorrect shard assignment: {name}')
                    self._tensors[name] = descriptor
                    self._tensors[name]['filename'] = filename
                    if len(self._tensors) > MAX_TENSORS:
                        raise SafeTensorError('Too many checkpoint tensors')
        if assignments is not None and set(assignments) != set(self._tensors):
            raise SafeTensorError('Checkpoint index references tensors missing from its shards')
        if 'total_size' in metadata:
            expected = _integer(metadata['total_size'], 'index total_size')
            actual = sum(descriptor['data_offsets'][1] - descriptor['data_offsets'][0]
                         for descriptor in self._tensors.values())
            if expected != actual:
                raise SafeTensorError('Checkpoint index total_size does not match tensor payloads')
        if not self._tensors:
            raise SafeTensorError('Checkpoint contains no tensors')

    @property
    def tensor_shapes(self):
        return {name: descriptor['shape'] for name, descriptor in self._tensors.items()}

    @property
    def tensor_dtypes(self):
        return {name: descriptor['dtype'] for name, descriptor in self._tensors.items()}

    def iter_rows(self, name):
        if name not in self._tensors:
            raise SafeTensorError(f'Unknown tensor: {name}')
        filename = self._tensors[name]['filename']
        with SafeTensorReader(self._paths[filename]) as reader:
            if reader._signature != self._signatures[filename]:
                raise SafeTensorError('Checkpoint shard changed after validation')
            yield from reader.iter_rows(name)

    def provenance(self, *, hash_files=True):
        files = []
        paths = dict(self._paths)
        signatures = dict(self._signatures)
        if self._index_path is not None:
            paths[self._index_path.name] = self._index_path
            signatures[self._index_path.name] = self._index_signature
        for name, path in sorted(paths.items()):
            if _signature(path.stat()) != signatures[name]:
                raise SafeTensorError('Checkpoint file changed after validation')
            entry = {'path': name, 'size_bytes': signatures[name][2]}
            if hash_files:
                entry['sha256'] = file_sha256(path)
                if _signature(path.stat()) != signatures[name]:
                    raise SafeTensorError('Checkpoint file changed while computing provenance')
            files.append(entry)
        return {'format': 'safetensors', 'hash_scope': 'complete_files' if hash_files else None,
                'files': files}
