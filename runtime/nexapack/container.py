"""Single-file `.nxb` container: a validated bundle directory, byte for byte.

The 64-byte little-endian prefix reuses the NexaPack `HEADER` layout
(`<8sHHIQQ32s`): magic `NEXABNDL`, format version, flags, JSON index length,
payload offset, exact physical size and the SHA-256 of the index bytes. The
JSON index precedes zero padding to a 4096-byte boundary, exactly like a
`.nxp`.

Every bundle file becomes one section. Sections are ordered and each one owns a
4096-aligned slot: a section starts where the previous slot ended, so the index
covers the whole payload with neither a hole nor an overlap, and the padding
inside a slot is verified to be zero. Alignment is not decoration: a nested
`.nxp` starts on a page boundary, so its own payload keeps the page alignment
it had as a standalone file, and a future reader may map it without copying.

Section `kind` is the versioned hook. V1 accepts `manifest`, `tensor` and
`asset` only, which is exactly what a bundle directory contains today. The
executable-package payloads M1.10b names -- plan, kernels, variants, fallback --
are deliberately absent (see `docs/NEXALM_PACOTE_NXB.md`), and an unknown kind
is refused rather than skipped, so a V1 reader can never half-understand a file
written by a later version.

Section checksums detect corruption; they do not authenticate a publisher.
"""
from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile

from .format import ALIGNMENT, HEADER, MAX_INTEGER, NexaPackError, READ_CHUNK_BYTES

MAGIC = b'NEXABNDL'
FORMAT = 'NexaBundleContainer'
FORMAT_VERSION = 1
SUFFIX = '.nxb'
MANIFEST_NAME = 'manifest.json'
MAX_INDEX_BYTES = 1024 * 1024
MAX_SECTIONS = 8192
# A sanity ceiling per section; the exact file size still has to fit MAX_INTEGER.
MAX_SECTION_BYTES = 1 << 40
# Kind to the path shape it is allowed to carry. A kind that could name any
# path would be a label, not a check: this pins the two to each other.
SECTION_PREFIX = {'manifest': None, 'tensor': 'tensors/', 'asset': 'assets/'}
_INDEX_KEYS = {'format', 'format_version', 'alignment', 'sections'}
_SECTION_KEYS = {'kind', 'path', 'offset', 'bytes', 'sha256'}
_SHA256 = re.compile(r'[0-9a-f]{64}')
_PLACEHOLDER_SHA = '0' * 64


class NexaContainerError(NexaPackError):
    """Invalid `.nxb` prefix, index, section layout, padding or section payload."""


def _align(value):
    return ((value + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT


def _integer(value, name, *, minimum=0, maximum=MAX_INTEGER):
    if type(value) is not int or not minimum <= value <= maximum:
        raise NexaContainerError(f'{name} must be an integer in [{minimum}, {maximum}]')
    return value


def relative_parts(value):
    """Split a confined POSIX relative path, or refuse it.

    This is the single rule for both the bundle manifest and the container
    index: a path that one of them accepted and the other did not would let a
    packed bundle differ from the directory it claims to reproduce.
    """
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value or ':' in value:
        raise NexaContainerError('Invalid bundle file path')
    parts = value.split('/')
    if any(part in ('', '.', '..') for part in parts) or PurePosixPath(value).is_absolute():
        raise NexaContainerError('Bundle file paths must be confined relative paths')
    return parts


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise NexaContainerError(f'Duplicate container index key: {key}')
        result[key] = value
    return result


def _reject_constant(value):
    raise NexaContainerError(f'Invalid JSON constant: {value}')


def _index_bytes(index):
    encoded = json.dumps(index, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    if len(encoded) > MAX_INDEX_BYTES:
        raise NexaContainerError(f'Container index exceeds {MAX_INDEX_BYTES} bytes')
    return encoded


def publish_directory(source, destination):
    """Atomic publication that never replaces even an empty existing directory."""
    if os.name == 'nt':
        # MoveFile semantics used by os.rename reject an existing destination.
        os.rename(source, destination)
        return
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == 'darwin':
        function = library.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        arguments = (os.fsencode(source), os.fsencode(destination), 0x4)  # RENAME_EXCL
    elif sys.platform.startswith('linux') and hasattr(library, 'renameat2'):
        function = library.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        arguments = (-100, os.fsencode(source), -100, os.fsencode(destination), 1)  # RENAME_NOREPLACE
    else:
        raise NexaContainerError('Atomic exclusive directory publication is unavailable on this platform')
    function.restype = ctypes.c_int
    if function(*arguments):
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise FileExistsError(number, 'Destination already exists', str(destination))
        raise OSError(number, os.strerror(number), str(destination))


class SectionStream:
    """A read-only view of one section, addressed from its own zero.

    The window is the unit of verification. Offsets are section-relative and
    every read is clamped to the section, so a caller that seeks or reads past
    the end gets short data instead of the neighbouring section's payload --
    the failure mode that would otherwise still satisfy the neighbour's own
    checksum, because both offsets slide together.
    """
    def __init__(self, stream, offset, size):
        self._stream = stream
        self._offset = offset
        self._size = size
        self._position = 0
        stream.seek(offset)

    @property
    def size(self):
        return self._size

    def seek(self, position, whence=os.SEEK_SET):
        if whence != os.SEEK_SET:
            raise NexaContainerError('Section streams seek only from their own start')
        if type(position) is not int or not 0 <= position <= self._size:
            raise NexaContainerError('Seek outside the section window')
        self._position = position
        self._stream.seek(self._offset + position)
        return position

    def tell(self):
        return self._position

    def _remaining(self, count):
        return min(self._size - self._position, count)

    def read(self, count=-1):
        wanted = self._size - self._position if count is None or count < 0 else self._remaining(count)
        if wanted <= 0:
            return b''
        data = self._stream.read(wanted)
        self._position += len(data)
        return data

    def readinto(self, buffer):
        view = memoryview(buffer).cast('B')
        try:
            wanted = self._remaining(len(view))
            if wanted <= 0:
                return 0
            received = self._stream.readinto(view[:wanted])
            self._position += received or 0
            return received
        finally:
            view.release()

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class NexaContainerReader:
    """Validated `.nxb` index; section payloads stay lazy, like a `.nxp` block.

    Opening reads the bounded prefix, the JSON index and every alignment gap.
    No section payload is read or mapped: the bundle reader verifies a tensor
    when it reads it, and `verify` exists for an explicit whole-file pass.
    """
    def __init__(self, path):
        self._path = Path(path)
        if self._path.is_symlink() or not self._path.is_file():
            raise NexaContainerError('Container must be a regular file without a symlink')
        self._stream = open(self._path, 'rb', buffering=0)
        try:
            self._load_index()
        except BaseException:
            self._stream.close()
            raise

    def _load_index(self):
        size = os.fstat(self._stream.fileno()).st_size
        prefix = self._stream.read(HEADER.size)
        if len(prefix) != HEADER.size:
            raise NexaContainerError('Truncated container header')
        magic, version, flags, length, payload_offset, total_size, expected = HEADER.unpack(prefix)
        if magic != MAGIC or version != FORMAT_VERSION or flags:
            raise NexaContainerError('Unsupported container magic, version, or flags')
        if not 0 < length <= MAX_INDEX_BYTES:
            raise NexaContainerError('Invalid or oversized container index length')
        if payload_offset != _align(HEADER.size + length):
            raise NexaContainerError('Invalid container payload offset')
        if total_size != size or not payload_offset < total_size <= MAX_INTEGER:
            raise NexaContainerError('Truncated container or inconsistent file size')
        encoded = self._stream.read(length)
        if len(encoded) != length or hashlib.sha256(encoded).digest() != expected:
            raise NexaContainerError('Container index checksum mismatch')
        try:
            index = json.loads(encoded.decode('utf-8'), object_pairs_hook=_unique_pairs,
                               parse_constant=_reject_constant)
        except (UnicodeError, ValueError, RecursionError) as error:
            raise NexaContainerError(f'Invalid container JSON: {error}') from error
        if not isinstance(index, dict) or set(index) != _INDEX_KEYS:
            raise NexaContainerError('Unexpected container index fields')
        if (index['format'] != FORMAT or type(index['format_version']) is not int
                or index['format_version'] != FORMAT_VERSION
                or type(index['alignment']) is not int or index['alignment'] != ALIGNMENT):
            raise NexaContainerError('Unsupported container format, version, or alignment')
        sections = index['sections']
        if not isinstance(sections, list) or not 0 < len(sections) <= MAX_SECTIONS:
            raise NexaContainerError('Invalid container section count')
        validated, used, gaps = [], set(), []
        cursor = payload_offset
        for position, section in enumerate(sections):
            if not isinstance(section, dict) or set(section) != _SECTION_KEYS:
                raise NexaContainerError('Unexpected container section fields')
            kind = section['kind']
            if kind not in SECTION_PREFIX:
                raise NexaContainerError(f'Unsupported container section kind: {kind!r}')
            path = section['path']
            relative_parts(path)
            prefix_rule = SECTION_PREFIX[kind]
            if prefix_rule is None:
                if path != MANIFEST_NAME or position:
                    raise NexaContainerError('The manifest section must come first and be manifest.json')
            elif not path.startswith(prefix_rule):
                raise NexaContainerError(f'Section kind {kind} cannot carry path {path}')
            if path.casefold() in used:
                raise NexaContainerError(f'Duplicate container section path: {path}')
            used.add(path.casefold())
            offset = _integer(section['offset'], 'section offset')
            length_bytes = _integer(section['bytes'], 'section bytes', maximum=MAX_SECTION_BYTES)
            checksum = section['sha256']
            if not isinstance(checksum, str) or not _SHA256.fullmatch(checksum):
                raise NexaContainerError(f'Invalid container section checksum: {path}')
            if offset != cursor:
                raise NexaContainerError(f'Overlapping, missing, or out-of-order section: {path}')
            end = offset + length_bytes
            cursor = _align(end)
            if cursor > MAX_INTEGER:
                raise NexaContainerError('Container layout exceeds the v1 integer limit')
            if end < cursor:
                gaps.append((end, cursor - end))
            validated.append({'kind': kind, 'path': path, 'offset': offset,
                              'bytes': length_bytes, 'sha256': checksum,
                              'padding_bytes': cursor - end})
        if validated[0]['kind'] != 'manifest':
            raise NexaContainerError('Container index has no manifest section')
        if cursor != total_size:
            raise NexaContainerError('Container index does not cover the complete payload')
        head = self._stream.read(payload_offset - HEADER.size - length)
        if any(head):
            raise NexaContainerError('Nonzero container index padding')
        # Alignment padding carries no checksum of its own, so it is pinned to
        # zero here; otherwise a slot's tail would be a place to hide bytes
        # that nothing in the file ever verifies.
        for start, count in gaps:
            self._stream.seek(start)
            consumed = 0
            while consumed < count:
                chunk = self._stream.read(min(READ_CHUNK_BYTES, count - consumed))
                if not chunk:
                    raise NexaContainerError('Truncated container alignment padding')
                if any(chunk):
                    raise NexaContainerError('Nonzero container alignment padding')
                consumed += len(chunk)
        self._index = index
        self._sections = tuple(validated)
        self._by_path = {section['path']: section for section in self._sections}
        self._payload_offset = payload_offset
        self._file_bytes = total_size
        self._index_length = length

    @property
    def path(self):
        return self._path

    @property
    def sections(self):
        return tuple(dict(section) for section in self._sections)

    @property
    def file_bytes(self):
        return self._file_bytes

    @property
    def payload_offset(self):
        return self._payload_offset

    def section(self, path):
        try:
            return dict(self._by_path[path])
        except KeyError as error:
            raise NexaContainerError(f'Unknown container section: {path}') from error

    def open_section(self, path, expected_bytes=None):
        """Open an independent handle windowed to one section.

        A fresh descriptor per read keeps the reader stateless between calls,
        the same contract the directory backend has.
        """
        section = self.section(path)
        if expected_bytes is not None and section['bytes'] != expected_bytes:
            raise NexaContainerError(f'Section size mismatch: {path}')
        stream = open(self._path, 'rb', buffering=0)
        try:
            if os.fstat(stream.fileno()).st_size != self._file_bytes:
                raise NexaContainerError('Container changed size while open')
            return SectionStream(stream, section['offset'], section['bytes'])
        except BaseException:
            stream.close()
            raise

    def window(self, path, expected_bytes=None):
        """Absolute (offset, bytes) of a section, for a reader that opens the file itself."""
        section = self.section(path)
        if expected_bytes is not None and section['bytes'] != expected_bytes:
            raise NexaContainerError(f'Section size mismatch: {path}')
        return section['offset'], section['bytes']

    def read_section_bytes(self, path):
        """Whole section into memory, checksum verified. Small sections only."""
        section = self.section(path)
        if section['bytes'] > MAX_INDEX_BYTES:
            raise NexaContainerError(f'Section is too large to read at once: {path}')
        with self.open_section(path) as stream:
            data = stream.read(section['bytes'])
        if len(data) != section['bytes']:
            raise NexaContainerError(f'Truncated container section: {path}')
        if hashlib.sha256(data).hexdigest() != section['sha256']:
            raise NexaContainerError(f'Section checksum mismatch: {path}')
        return data

    def verify(self):
        """Stream every section and compare its checksum; returns bytes read."""
        total = 0
        for section in self._sections:
            digest = hashlib.sha256()
            with self.open_section(section['path']) as stream:
                consumed = 0
                while consumed < section['bytes']:
                    chunk = stream.read(min(READ_CHUNK_BYTES, section['bytes'] - consumed))
                    if not chunk:
                        raise NexaContainerError(f'Truncated container section: {section["path"]}')
                    digest.update(chunk)
                    consumed += len(chunk)
            total += consumed
            if digest.hexdigest() != section['sha256']:
                raise NexaContainerError(f'Section checksum mismatch: {section["path"]}')
        return total

    def measurements(self):
        """Measured container cost; nothing here is estimated."""
        payload = sum(section['bytes'] for section in self._sections)
        padding = sum(section['padding_bytes'] for section in self._sections)
        return {'format': FORMAT, 'format_version': FORMAT_VERSION,
                'alignment': ALIGNMENT, 'sections': len(self._sections),
                'file_bytes': self._file_bytes,
                'header_bytes': HEADER.size, 'index_bytes': self._index_length,
                'index_padding_bytes': self._payload_offset - HEADER.size - self._index_length,
                'section_bytes': payload,
                'alignment_padding_bytes': padding,
                'container_overhead_bytes': self._file_bytes - payload,
                'section_padding': [{'path': section['path'], 'bytes': section['bytes'],
                                     'padding_bytes': section['padding_bytes']}
                                    for section in self._sections]}

    def close(self):
        self._stream.close()

    def __enter__(self):
        if self._stream.closed:
            raise NexaContainerError('Container reader is closed')
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def _bundle_sections(manifest):
    """Section order: manifest first, then tensors and assets sorted by path."""
    tensors = sorted({entry['path'] for entry in manifest['tensors'].values()})
    assets = sorted(entry['path'] for entry in manifest['assets'].values())
    return ([('manifest', MANIFEST_NAME)] + [('tensor', path) for path in tensors]
            + [('asset', path) for path in assets])


def _layout(records):
    """Resolve index length and section offsets together, like `.nxp` blocks.

    Checksum strings keep their width, but offsets do not, so the index length
    and the payload offset are a fixed point rather than a single computation.
    """
    sections = [{'kind': kind, 'path': path, 'offset': 0, 'bytes': size,
                 'sha256': _PLACEHOLDER_SHA} for kind, path, size in records]
    index = {'format': FORMAT, 'format_version': FORMAT_VERSION,
             'alignment': ALIGNMENT, 'sections': sections}
    payload_offset = _align(HEADER.size + len(_index_bytes(index)))
    for _ in range(8):
        cursor = payload_offset
        for section in sections:
            section['offset'] = cursor
            cursor = _align(cursor + section['bytes'])
        if cursor > MAX_INTEGER:
            raise NexaContainerError('Container layout exceeds the v1 integer limit')
        wanted = _align(HEADER.size + len(_index_bytes(index)))
        if wanted == payload_offset:
            return index, payload_offset, cursor
        payload_offset = wanted
    raise NexaContainerError('Container index length did not converge')


def pack_bundle(directory, destination) -> dict:
    """Write one `.nxb` holding a validated bundle directory, byte for byte.

    The directory is opened with the reader the runtime uses, so an invalid
    bundle is refused before anything is written. Publication follows the same
    contract as `write_model_bundle`: temporary file, fsync, then `os.replace`.
    """
    # Deferred: bundle.py imports this module for its container backend.
    from .bundle import ModelBundleReader, _confined_file

    directory = Path(directory)
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(f'Container destination already exists: {destination}')
    with ModelBundleReader(directory) as model:
        manifest = model.manifest
    root = directory.resolve()
    records, sources = [], []
    for kind, relative in _bundle_sections(manifest):
        path = _confined_file(root, relative)
        records.append((kind, relative, path.stat().st_size))
        sources.append(path)
    if len(records) > MAX_SECTIONS:
        raise NexaContainerError(f'A container holds at most {MAX_SECTIONS} sections')
    index, payload_offset, total_size = _layout(records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.nexa-container-', dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'w+b') as stream:
            for section, source in zip(index['sections'], sources):
                stream.seek(section['offset'])
                digest = hashlib.sha256()
                consumed = 0
                with source.open('rb', buffering=0) as incoming:
                    if os.fstat(incoming.fileno()).st_size != section['bytes']:
                        raise NexaContainerError(f'Bundle file size changed: {section["path"]}')
                    while consumed < section['bytes']:
                        chunk = incoming.read(min(READ_CHUNK_BYTES, section['bytes'] - consumed))
                        if not chunk:
                            raise NexaContainerError(f'Bundle file was truncated while packing: {section["path"]}')
                        stream.write(chunk)
                        digest.update(chunk)
                        consumed += len(chunk)
                    if incoming.read(1):
                        raise NexaContainerError(f'Bundle file grew while packing: {section["path"]}')
                section['sha256'] = digest.hexdigest()
            encoded = _index_bytes(index)
            if _align(HEADER.size + len(encoded)) != payload_offset:
                raise NexaContainerError('Internal container layout mismatch')
            # Alignment gaps are file holes, which read as zeros; the tail past
            # the last section has to be materialized so the physical size is
            # the one the header promises.
            os.ftruncate(stream.fileno(), total_size)
            stream.seek(0)
            stream.write(HEADER.pack(MAGIC, FORMAT_VERSION, 0, len(encoded), payload_offset,
                                     total_size, hashlib.sha256(encoded).digest()))
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
            if os.fstat(stream.fileno()).st_size != total_size:
                raise NexaContainerError('Internal container size mismatch')
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    with NexaContainerReader(destination) as container:
        return container.measurements()


def unpack_bundle(source, destination) -> dict:
    """Rebuild the bundle directory from a `.nxb`, verifying every section."""
    # Deferred for the same reason as pack_bundle.
    from .bundle import ModelBundleReader

    source = Path(source)
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(f'Bundle destination already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.nexa-unpack-', dir=destination.parent))
    try:
        with NexaContainerReader(source) as container:
            written = 0
            for section in container.sections:
                parts = relative_parts(section['path'])
                target = staging
                for part in parts[:-1]:
                    target = target / part
                target.mkdir(parents=True, exist_ok=True)
                target = target / parts[-1]
                digest = hashlib.sha256()
                with container.open_section(section['path']) as incoming, target.open('xb') as outgoing:
                    consumed = 0
                    while consumed < section['bytes']:
                        chunk = incoming.read(min(READ_CHUNK_BYTES, section['bytes'] - consumed))
                        if not chunk:
                            raise NexaContainerError(f'Truncated container section: {section["path"]}')
                        outgoing.write(chunk)
                        digest.update(chunk)
                        consumed += len(chunk)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
                if digest.hexdigest() != section['sha256']:
                    raise NexaContainerError(f'Section checksum mismatch: {section["path"]}')
                written += consumed
            measurements = container.measurements()
        # The same preflight write_model_bundle runs: publish only what the
        # consumer's own reader already accepted.
        with ModelBundleReader(staging):
            pass
        publish_directory(staging, destination)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
    measurements['unpacked_bytes'] = written
    measurements['unpacked_files'] = len(measurements['section_padding'])
    return measurements


def is_container(path) -> bool:
    """True when the file starts with the container magic; no index is parsed."""
    path = Path(path)
    try:
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
            return False
        with path.open('rb', buffering=0) as stream:
            return stream.read(len(MAGIC)) == MAGIC
    except OSError:
        return False
