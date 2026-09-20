"""Versioned multi-tensor bundles built from unchanged NexaPack v1 matrices.

The manifest describes physical files and tied-weight aliases. Opening validates
every Q4 header/index, but weight payload checksums remain lazy. RAW_F32 vectors
are read explicitly; small tokenizer assets are checked in chunks. No asset is
executed.

The same bundle reads from a directory or from a single `.nxb` container. A
backend supplies the manifest bytes and one windowed stream per payload; every
check above it -- sizes, checksums, codecs, confinement -- is the same code in
both cases, because a check that ran on only one of the two would be the thing
that lets a container diverge from the directory it claims to reproduce.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import tempfile

from compiler.model_config import ModelConfig
from .container import (NexaContainerError, NexaContainerReader,
                        publish_directory as _publish_directory, relative_parts)
from .format import (NexaPackError, NexaPackReader, READ_CHUNK_BYTES,
                     write_grouped_matrix)

# Matrix codecs a bundle may store, with the NexaPack id each one publishes.
PACKED_CODECS = {"q2": "Q2_GROUPED", "q3": "Q3_GROUPED",
                 "q4": "Q4_GROUPED", "q8": "Q8_GROUPED"}
# Dense matrices carry no scale: the stored width is the whole contract.
DENSE_CODECS = {"f16": "RAW_F16_MATRIX", "f32": "RAW_F32_MATRIX"}
DENSE_WIDTH = {"RAW_F16_MATRIX": 2, "RAW_F32_MATRIX": 4}
MATRIX_CODECS = (*PACKED_CODECS, *DENSE_CODECS)
CODEC_BITS = {"Q2_GROUPED": 2, "Q3_GROUPED": 3, "Q4_GROUPED": 4, "Q8_GROUPED": 8,
              "RAW_F16_MATRIX": 16, "RAW_F32_MATRIX": 32, "RAW_F32": 32}
_F16 = struct.Struct("<e")

FORMAT = "NexaModelBundle"
FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_PROVENANCE_BYTES = 64 * 1024
MAX_TENSORS = 4096
MAX_RAW_BYTES = 16 * 1024 * 1024
# A dense F32 matrix is a reference and calibration format, not a shipping one:
# it costs eight times its Q4 form, so only the block bound protects a read.
MAX_RAW_MATRIX_BYTES = 1 << 34
MAX_MATRIX_BLOCKS = 1 << 16
MAX_ASSETS = 16
MAX_ASSET_BYTES = 16 * 1024 * 1024
MAX_ASSETS_TOTAL_BYTES = 32 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ASSET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_F32 = struct.Struct("<f")
_MISSING = object()
_RESERVED_NAMES = {"con", "prn", "aux", "nul"} | {f"{kind}{n}" for kind in ("com", "lpt") for n in range(1, 10)}


class ModelBundleError(NexaPackError):
    """Malformed, inconsistent, corrupted, or unsupported model bundle."""


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ModelBundleError(f"Unexpected {label} fields")


def _integer(value, label, *, minimum=1, maximum=(1 << 63) - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ModelBundleError(f"Invalid {label}")
    return value


def _sha(value, label):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ModelBundleError(f"Invalid {label} checksum")
    return value


def _asset_name(value):
    return (isinstance(value, str) and _ASSET_NAME.fullmatch(value)
            and not value.endswith(".") and value.split(".", 1)[0].casefold() not in _RESERVED_NAMES)


def _config_shapes(config):
    # Guard before expanding layer names, including when parsing untrusted JSON.
    if 9 * config.num_hidden_layers + 3 > MAX_TENSORS:
        raise ModelBundleError("Model layer count exceeds the tensor index limit")
    if config.hidden_size * _F32.size > MAX_RAW_BYTES:
        raise ModelBundleError("Model norm vectors exceed the RAW_F32 size limit")
    return config.required_tensor_shapes()


def _json_tree(value, depth=0):
    if depth > 12:
        raise ModelBundleError("JSON nesting limit exceeded")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _json_tree(item, depth + 1)
        return
    if isinstance(value, dict) and all(type(key) is str for key in value):
        for item in value.values():
            _json_tree(item, depth + 1)
        return
    raise ModelBundleError("Expected finite JSON values and string object keys")


def _json_bytes(value, limit=MAX_MANIFEST_BYTES):
    _json_tree(value)
    try:
        result = json.dumps(value, sort_keys=True, separators=(",", ":"),
                            allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise ModelBundleError("Invalid JSON metadata") from error
    if len(result) > limit:
        raise ModelBundleError(f"JSON metadata exceeds {limit} bytes")
    return result


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ModelBundleError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ModelBundleError(f"Invalid JSON constant: {value}")


def _provenance(value):
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ModelBundleError("Provenance must be a JSON object")
    # Round-trip to detach caller-owned mutable values before streaming begins.
    return json.loads(_json_bytes(value, MAX_PROVENANCE_BYTES))


def _relative_path(value):
    # One rule, shared with the container index, translated into this module's
    # error type: a path the manifest accepts and the index refuses (or the
    # reverse) would make a packed bundle unequal to its directory.
    try:
        return relative_parts(value)
    except NexaContainerError as error:
        raise ModelBundleError(str(error)) from error


def _confined_file(directory, value):
    parts = _relative_path(value)
    path = directory
    for index, part in enumerate(parts):
        path = path / part
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ModelBundleError(f"Symlinks are not allowed in bundles: {value}")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise ModelBundleError(f"Bundle path component is not a directory: {value}")
        if index == len(parts) - 1 and not stat.S_ISREG(info.st_mode):
            raise ModelBundleError(f"Bundle payload must be a regular file: {value}")
    if not path.resolve().is_relative_to(directory):
        raise ModelBundleError(f"Bundle path escapes its directory: {value}")
    return path


class _FileStream:
    """One whole file, carrying the size the manifest will be checked against.

    Unlike a container section, reads are not clamped: a file that grew past
    its declared size still hands back the extra byte, which is exactly how
    the readers below notice that it grew.
    """
    def __init__(self, stream, size):
        self._stream = stream
        self.size = size

    def read(self, count=-1):
        return self._stream.read(count)

    def readinto(self, buffer):
        return self._stream.readinto(buffer)

    def seek(self, position, whence=os.SEEK_SET):
        return self._stream.seek(position, whence)

    def tell(self):
        return self._stream.tell()

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class _DirectoryBackend:
    """Bundle payloads as separate files under a directory."""
    kind = "directory"

    def __init__(self, directory):
        self.root = directory

    def size(self, relative):
        return _confined_file(self.root, relative).stat().st_size

    def describe(self, relative):
        # Identity and size from one path walk: hard links and the same file
        # named twice collapse to one inode.
        info = _confined_file(self.root, relative).stat()
        return (info.st_dev, info.st_ino), info.st_size

    def open(self, relative):
        path = _confined_file(self.root, relative)
        stream = path.open("rb", buffering=0)
        try:
            return _FileStream(stream, os.fstat(stream.fileno()).st_size)
        except BaseException:
            stream.close()
            raise

    def packed(self, relative):
        path = _confined_file(self.root, relative)
        # Positional call: a standalone matrix owns its whole file, and tests
        # that substitute the reader class rely on this exact signature.
        return path.stat().st_size, lambda: NexaPackReader(path)

    def close(self):
        pass


class _ContainerBackend:
    """Bundle payloads as windowed sections of one `.nxb` file."""
    kind = "container"

    def __init__(self, path):
        self.container = NexaContainerReader(path)

    def size(self, relative):
        return self.container.section(relative)["bytes"]

    def describe(self, relative):
        # Two sections cannot start at the same offset: the index refuses
        # overlaps, so the offset is the section's identity.
        section = self.container.section(relative)
        return ("section", section["offset"]), section["bytes"]

    def open(self, relative):
        return self.container.open_section(relative)

    def packed(self, relative):
        offset, size = self.container.window(relative)
        return size, lambda: NexaPackReader(self.container.path,
                                            window_offset=offset, window_bytes=size)

    def close(self):
        self.container.close()


def _open_backend(source):
    """Pick the backend from what the path is, never from its spelling."""
    path = Path(source)
    if path.is_symlink():
        raise ModelBundleError("Bundle root must be a directory without a symlink")
    if path.is_dir():
        return _DirectoryBackend(path.resolve())
    if not path.is_file():
        raise ModelBundleError("Bundle root must be a directory or a container file")
    try:
        return _ContainerBackend(path)
    except NexaContainerError as error:
        raise ModelBundleError(str(error)) from error


def _hash_file(stream, label, expected_size, maximum, *, finite_f32=False):
    _integer(expected_size, "file_bytes", minimum=0, maximum=maximum)
    digest = hashlib.sha256()
    if stream.size != expected_size:
        raise ModelBundleError(f"File size mismatch: {label}")
    consumed = 0
    while consumed < expected_size:
        chunk = stream.read(min(READ_CHUNK_BYTES, expected_size - consumed))
        if not chunk:
            raise ModelBundleError(f"Truncated file: {label}")
        if finite_f32:
            if len(chunk) % _F32.size:
                raise ModelBundleError(f"Truncated float32 vector: {label}")
            if any(not math.isfinite(item[0]) for item in struct.iter_unpack("<f", chunk)):
                raise ModelBundleError(f"Nonfinite float32 vector: {label}")
        digest.update(chunk)
        consumed += len(chunk)
    if stream.read(1):
        raise ModelBundleError(f"File grew during read: {label}")
    return digest.hexdigest()


def _check_dense_blocks(name, entry, shape):
    """Validate that the blocks tile every row exactly once, in order."""
    blocks = entry["blocks"]
    rows, cols = shape
    width = DENSE_WIDTH[entry["codec"]]
    if (not isinstance(blocks, list) or not 0 < len(blocks) <= MAX_MATRIX_BLOCKS
            or entry["file_bytes"] != rows * cols * width):
        raise ModelBundleError(f"Invalid RAW_F32 matrix blocks or size: {name}")
    _integer(entry["file_bytes"], "RAW_F32 matrix size", minimum=0, maximum=MAX_RAW_MATRIX_BYTES)
    covered = 0
    for block in blocks:
        _keys(block, {"start_row", "row_count", "sha256"}, "matrix block")
        _integer(block["start_row"], "start_row", minimum=0)
        _integer(block["row_count"], "row_count")
        _sha(block["sha256"], name)
        if block["start_row"] != covered or covered + block["row_count"] > rows:
            raise ModelBundleError(f"RAW_F32 matrix blocks must tile every row once: {name}")
        covered += block["row_count"]
    if covered != rows:
        raise ModelBundleError(f"RAW_F32 matrix blocks do not cover every row: {name}")
    return tuple({"start_row": block["start_row"], "row_count": block["row_count"],
                  "sha256": block["sha256"], "bytes": block["row_count"] * cols * width}
                 for block in blocks)


def _metadata_sha(reader):
    # Canonical JSON of the validated NexaPack metadata, not the weight payload.
    return hashlib.sha256(_json_bytes(reader.metadata)).hexdigest()


def _check_q4(backend, relative, entry):
    label = PurePosixPath(relative).name
    size, open_reader = backend.packed(relative)
    if size != entry["file_bytes"]:
        raise ModelBundleError(f"Packed file size mismatch: {label}")
    reader = open_reader()
    try:
        if reader.codec_id != entry["codec"] or reader.codec_version != 1:
            raise ModelBundleError(f'Matrix codec differs from its manifest entry: {label}')
        if [reader.rows, reader.cols] != entry["shape"]:
            raise ModelBundleError(f"Q4 shape mismatch: {label}")
        if _metadata_sha(reader) != entry["metadata_sha256"]:
            raise ModelBundleError(f"Q4 metadata checksum mismatch: {label}")
        return reader
    except BaseException:
        reader.close()
        raise


def _write_raw_matrix(path, rows, cols, source, block_rows, width=4):
    """Write a row-major F32 matrix in verified blocks, one row at a time.

    Each block carries its own checksum, so a tile read verifies exactly the
    bytes it consumes instead of trusting a whole-file digest.
    """
    _integer(block_rows, "block_rows", minimum=1)
    encoder = _F32 if width == 4 else _F16
    total = rows * cols * width
    _integer(total, "RAW_F32 matrix size", maximum=MAX_RAW_MATRIX_BYTES)
    if (rows + block_rows - 1) // block_rows > MAX_MATRIX_BLOCKS:
        raise ModelBundleError("RAW_F32 matrix exceeds the supported block count")
    blocks, produced = [], iter(source)
    with path.open("xb") as stream:
        for start in range(0, rows, block_rows):
            count = min(block_rows, rows - start)
            digest = hashlib.sha256()
            for offset in range(count):
                try:
                    values = next(produced)
                except StopIteration as error:
                    raise ModelBundleError(f"Matrix ended at row {start + offset}; expected {rows}") from error
                encoded = bytearray()
                consumed = 0
                for value in values:
                    if consumed == cols:
                        raise ModelBundleError(f"Matrix row {start + offset} exceeds {cols} values")
                    try:
                        encoded += encoder.pack(float(value))
                    except (ValueError, TypeError, OverflowError, struct.error) as error:
                        raise ModelBundleError("Matrix values must fit the stored float width") from error
                    consumed += 1
                if consumed != cols:
                    raise ModelBundleError(f"Matrix row {start + offset} has {consumed} of {cols} values")
                for value, in encoder.iter_unpack(bytes(encoded)):
                    # float16 overflows to inf well inside the float32 range.
                    if not math.isfinite(value):
                        raise ModelBundleError("Matrix values must be finite in the stored width")
                stream.write(encoded)
                digest.update(encoded)
            blocks.append({"start_row": start, "row_count": count, "sha256": digest.hexdigest()})
        if next(produced, _MISSING) is not _MISSING:
            raise ModelBundleError(f"Matrix contains more than {rows} rows")
        stream.flush()
        os.fsync(stream.fileno())
    return blocks, total


def _write_raw(path, count, source):
    _integer(count * _F32.size, "RAW_F32 size", maximum=MAX_RAW_BYTES)
    try:
        rows = iter(source)
        values = iter(next(rows))
    except (TypeError, StopIteration) as error:
        raise ModelBundleError("Vector source must yield exactly one row") from error
    digest = hashlib.sha256()
    with path.open("xb") as stream:
        buffer = bytearray()
        for index in range(count):
            value = next(values, _MISSING)
            if value is _MISSING:
                raise ModelBundleError(f"Vector ended at {index}; expected {count} values")
            try:
                value = float(value)
                encoded = _F32.pack(value)
            except (ValueError, TypeError, OverflowError, struct.error) as error:
                raise ModelBundleError("Vector values must fit finite float32") from error
            if not math.isfinite(value):
                raise ModelBundleError("Vector values must be finite")
            buffer.extend(encoded)
            if len(buffer) == READ_CHUNK_BYTES:
                stream.write(buffer)
                digest.update(buffer)
                buffer.clear()
        if next(values, _MISSING) is not _MISSING or next(rows, _MISSING) is not _MISSING:
            raise ModelBundleError("Vector source contains extra values or rows")
        stream.write(buffer)
        digest.update(buffer)
        stream.flush()
        os.fsync(stream.fileno())
    return digest.hexdigest()


def _copy_asset(source, destination):
    source = Path(source)
    if not stat.S_ISREG(source.lstat().st_mode):
        raise ModelBundleError("Tokenizer sources must be regular files, without symlinks")
    size = source.stat().st_size
    _integer(size, "tokenizer asset size", minimum=0, maximum=MAX_ASSET_BYTES)
    digest = hashlib.sha256()
    with source.open("rb", buffering=0) as incoming, destination.open("xb") as outgoing:
        consumed = 0
        while True:
            chunk = incoming.read(READ_CHUNK_BYTES)
            if not chunk:
                break
            consumed += len(chunk)
            if consumed > size:
                raise ModelBundleError("Tokenizer source grew while copying")
            outgoing.write(chunk)
            digest.update(chunk)
        if consumed != size:
            raise ModelBundleError("Tokenizer source was truncated while copying")
        outgoing.flush()
        os.fsync(outgoing.fileno())
    return size, digest.hexdigest()


def write_model_bundle(destination, config: ModelConfig, tensor_sources: Mapping,
                       *, group_size=32, block_rows=64, tokenizer_files=None,
                       provenance=None, asset_checksums=None, tensor_codecs=None) -> None:
    """Stream sources into a new bundle and publish the directory atomically.

    Sources contain exactly physical required tensor names. Each callable yields
    rows; a rank-one vector yields one row containing all of its values. Tied
    aliases have no source or additional payload. Existing destinations survive.
    Optional asset_checksums pin copied assets to previously inspected bytes.
    """
    if not isinstance(config, ModelConfig):
        raise ModelBundleError("config must be a validated ModelConfig")
    shapes = _config_shapes(config)
    if not isinstance(tensor_sources, Mapping) or set(tensor_sources) != set(shapes):
        raise ModelBundleError("Tensor sources must exactly match required physical tensors")
    if not 0 < len(shapes) <= MAX_TENSORS or any(not callable(value) for value in tensor_sources.values()):
        raise ModelBundleError("Invalid tensor source count or callable")
    codecs = {} if tensor_codecs is None else dict(tensor_codecs)
    if not set(codecs) <= set(shapes) or any(value not in MATRIX_CODECS for value in codecs.values()):
        raise ModelBundleError(f"Tensor codecs must name declared tensors and be one of {MATRIX_CODECS}")
    if any(len(shapes[name]) == 1 and codec != "f32" for name, codec in codecs.items()):
        raise ModelBundleError("Rank-one vectors are always RAW_F32")
    assets = {} if tokenizer_files is None else tokenizer_files
    if not isinstance(assets, Mapping) or len(assets) > MAX_ASSETS:
        raise ModelBundleError("Invalid tokenizer asset mapping")
    names = set()
    for name in assets:
        if not _asset_name(name) or name.casefold() in names:
            raise ModelBundleError("Tokenizer asset names must be unique portable basenames")
        names.add(name.casefold())
    if asset_checksums is not None:
        if not isinstance(asset_checksums, Mapping) or set(asset_checksums) != set(assets):
            raise ModelBundleError("Asset checksums must exactly match tokenizer asset names")
        asset_checksums = {name: _sha(value, name) for name, value in asset_checksums.items()}
    provenance = _provenance(provenance)
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(f"Bundle destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".nexa-model-", dir=destination.parent))
    try:
        (staging / "tensors").mkdir()
        (staging / "assets").mkdir()
        tensors = {}
        for index, (name, shape) in enumerate(sorted(shapes.items())):
            source = tensor_sources[name]()
            if len(shape) == 2 and codecs.get(name, "q4") in DENSE_CODECS:
                choice = codecs[name]
                relative = f"tensors/{index:04d}.{choice}"
                width = 4 if choice == "f32" else 2
                blocks, total = _write_raw_matrix(staging / relative, shape[0], shape[1],
                                                  source, block_rows, width)
                tensors[name] = {"shape": list(shape), "codec": DENSE_CODECS[choice], "codec_version": 1,
                                 "path": relative, "file_bytes": total, "blocks": blocks}
            elif len(shape) == 2:
                relative = f"tensors/{index:04d}.nxp"
                path = staging / relative
                codec = PACKED_CODECS[codecs.get(name, "q4")]
                write_grouped_matrix(path, shape[0], shape[1], group_size, source,
                                     block_rows=block_rows, codec=codec)
                with NexaPackReader(path) as reader:
                    tensors[name] = {"shape": list(shape), "codec": codec, "codec_version": 1,
                                     "path": relative, "file_bytes": path.stat().st_size,
                                     "metadata_sha256": _metadata_sha(reader)}
            elif len(shape) == 1:
                relative = f"tensors/{index:04d}.f32"
                checksum = _write_raw(staging / relative, shape[0], source)
                tensors[name] = {"shape": list(shape), "codec": "RAW_F32", "codec_version": 1,
                                 "path": relative, "file_bytes": shape[0] * _F32.size, "sha256": checksum}
            else:
                raise ModelBundleError("Only rank-one vectors and rank-two matrices are supported")
        asset_entries = {}
        total = 0
        for name, source in sorted(assets.items()):
            relative = f"assets/{name}"
            size, digest = _copy_asset(source, staging / relative)
            if asset_checksums is not None and digest != asset_checksums[name]:
                raise ModelBundleError(f"Tokenizer source checksum changed: {name}")
            total += size
            if total > MAX_ASSETS_TOTAL_BYTES:
                raise ModelBundleError("Tokenizer assets exceed total size limit")
            asset_entries[name] = {"path": relative, "file_bytes": size, "sha256": digest}
        manifest = {"format": FORMAT, "format_version": FORMAT_VERSION,
                    "architecture": config.architecture, "config": config.to_dict(),
                    "tensors": tensors, "aliases": config.tensor_aliases(), "assets": asset_entries,
                    "provenance": {"schema_version": 1, "data": provenance}}
        with (staging / MANIFEST_NAME).open("xb") as stream:
            stream.write(_json_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        # Run the same index/header checks the consumer will apply, before publish.
        with ModelBundleReader(staging):
            pass
        _publish_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


class ModelBundleReader:
    """Validated bundle index with lazy Q4 payload reads and bounded raw vectors.

    open_q4 returns an independent reader owned by the caller; closing this bundle
    does not close readers already returned. No payload handles are retained by
    preflight. The returned manifest is detached from the validated state.

    The source is either a bundle directory or a single `.nxb` container. The
    public API is the same for both; only the backend below differs.
    """
    def __init__(self, source):
        self._closed = False
        self._q4_summaries = {}
        self._matrix_blocks = {}
        self._backend = _open_backend(source)
        try:
            manifest = self._load_manifest()
            _json_tree(manifest)
            self._validate(manifest)
        except BaseException:
            self._backend.close()
            raise
        self._manifest = manifest

    def _load_manifest(self):
        with self._backend.open(MANIFEST_NAME) as stream:
            if stream.size > MAX_MANIFEST_BYTES:
                raise ModelBundleError("Bundle manifest exceeds metadata limit")
            encoded = stream.read(MAX_MANIFEST_BYTES + 1)
        if len(encoded) > MAX_MANIFEST_BYTES:
            raise ModelBundleError("Bundle manifest exceeds metadata limit")
        try:
            return json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_pairs,
                              parse_constant=_reject_constant)
        except (ValueError, UnicodeError, RecursionError) as error:
            raise ModelBundleError(f"Invalid bundle JSON: {error}") from error

    def _validate(self, manifest):
        _keys(manifest, {"format", "format_version", "architecture", "config", "tensors", "aliases",
                         "assets", "provenance"}, "bundle manifest")
        if manifest["format"] != FORMAT or type(manifest["format_version"]) is not int or manifest["format_version"] != 1:
            raise ModelBundleError("Unsupported model bundle format/version")
        try:
            self._config = ModelConfig.from_dict(manifest["config"])
        except (ValueError, TypeError, KeyError) as error:
            raise ModelBundleError(f"Invalid model config: {error}") from error
        if manifest["architecture"] != self._config.architecture:
            raise ModelBundleError("Bundle architecture differs from config")
        shapes = _config_shapes(self._config)
        tensors, aliases = manifest["tensors"], manifest["aliases"]
        if not isinstance(tensors, dict) or not 0 < len(tensors) <= MAX_TENSORS or set(tensors) != set(shapes):
            raise ModelBundleError("Manifest must describe exactly the required physical tensors")
        if not isinstance(aliases, dict) or aliases != self._config.tensor_aliases():
            raise ModelBundleError("Aliases must exactly match the validated tied-weight config")
        for name, target in aliases.items():
            if name in tensors or target not in tensors:
                raise ModelBundleError("Aliases must resolve directly to physical tensors")
        used_paths = {MANIFEST_NAME.casefold()}
        used_files = set()

        def payload(entry):
            relative = entry["path"]
            _relative_path(relative)
            if relative.casefold() in used_paths:
                raise ModelBundleError("Duplicate bundle payload path")
            used_paths.add(relative.casefold())
            identity, size = self._backend.describe(relative)
            if identity in used_files:
                raise ModelBundleError("Duplicate bundle payload file")
            used_files.add(identity)
            _integer(entry["file_bytes"], "file_bytes", minimum=0)
            if size != entry["file_bytes"]:
                raise ModelBundleError(f"File size mismatch: {relative}")
            return relative

        for name, shape in shapes.items():
            entry = tensors[name]
            dense = (len(shape) == 2 and isinstance(entry, dict)
                     and entry.get("codec") in DENSE_CODECS.values())
            checksum_key = "blocks" if dense else ("metadata_sha256" if len(shape) == 2 else "sha256")
            _keys(entry, {"shape", "codec", "codec_version", "path", "file_bytes", checksum_key}, "tensor entry")
            if (not isinstance(entry["shape"], list) or any(type(v) is not int for v in entry["shape"])
                    or entry["shape"] != list(shape)):
                raise ModelBundleError(f"Tensor shape mismatch: {name}")
            allowed = (set(DENSE_CODECS.values()) if dense else set(PACKED_CODECS.values())) if len(shape) == 2 else {"RAW_F32"}
            if entry["codec"] not in allowed or type(entry["codec_version"]) is not int or entry["codec_version"] != 1:
                raise ModelBundleError(f"Unsupported tensor codec: {name}")
            if not dense:
                _sha(entry[checksum_key], name)
            relative = payload(entry)
            if dense:
                self._matrix_blocks[name] = _check_dense_blocks(name, entry, shape)
            elif len(shape) == 2:
                with _check_q4(self._backend, relative, entry) as reader:
                    self._q4_summaries[name] = {"group_size": reader.group_size,
                                                "packed_payload_bytes": reader.rows * reader.row_bytes}
            else:
                if entry["file_bytes"] != shape[0] * _F32.size:
                    raise ModelBundleError(f"Vector size mismatch: {name}")
                _integer(entry["file_bytes"], "RAW_F32 size", maximum=MAX_RAW_BYTES)
        assets = manifest["assets"]
        if not isinstance(assets, dict) or len(assets) > MAX_ASSETS:
            raise ModelBundleError("Invalid tokenizer assets")
        total = 0
        folded_names = set()
        for name, entry in assets.items():
            if not _asset_name(name) or name.casefold() in folded_names:
                raise ModelBundleError("Invalid or duplicate tokenizer asset name")
            folded_names.add(name.casefold())
            _keys(entry, {"path", "file_bytes", "sha256"}, "asset entry")
            _sha(entry["sha256"], name)
            relative = payload(entry)
            total += entry["file_bytes"]
            if total > MAX_ASSETS_TOTAL_BYTES:
                raise ModelBundleError("Tokenizer assets exceed total size limit")
            with self._backend.open(relative) as stream:
                digest = _hash_file(stream, PurePosixPath(relative).name,
                                    entry["file_bytes"], MAX_ASSET_BYTES)
            if digest != entry["sha256"]:
                raise ModelBundleError(f"Tokenizer checksum mismatch: {name}")
        provenance = manifest["provenance"]
        _keys(provenance, {"schema_version", "data"}, "provenance")
        if type(provenance["schema_version"]) is not int or provenance["schema_version"] != 1:
            raise ModelBundleError("Unsupported provenance schema")
        if not isinstance(provenance["data"], dict):
            raise ModelBundleError("Provenance data must be a JSON object")
        _provenance(provenance["data"])

    @property
    def config(self):
        return self._config

    @property
    def manifest(self):
        return json.loads(_json_bytes(self._manifest))

    @property
    def tensor_names(self):
        return tuple(sorted(set(self._manifest["tensors"]) | set(self._manifest["aliases"])))

    def _entry(self, name):
        if self._closed:
            raise ModelBundleError("Model bundle reader is closed")
        if not isinstance(name, str):
            raise ModelBundleError("Tensor name must be a string")
        name = self._manifest["aliases"].get(name, name)
        try:
            return self._manifest["tensors"][name]
        except KeyError as error:
            raise ModelBundleError(f"Unknown tensor: {name}") from error

    def open_packed(self, name):
        """Open any grouped-codec matrix; the reader knows its own codec."""
        entry = self._entry(name)
        if entry["codec"] not in PACKED_CODECS.values():
            raise ModelBundleError(f"Tensor is not a packed matrix: {name}")
        return _check_q4(self._backend, entry["path"], entry)

    def open_q4(self, name):
        entry = self._entry(name)
        if entry["codec"] != "Q4_GROUPED":
            raise ModelBundleError(f"Tensor is not Q4: {name}")
        return _check_q4(self._backend, entry["path"], entry)

    def read_f32(self, name):
        entry = self._entry(name)
        if entry["codec"] != "RAW_F32":
            raise ModelBundleError(f"Tensor is not RAW_F32: {name}")
        size = entry["file_bytes"]
        _integer(size, "RAW_F32 size", maximum=MAX_RAW_BYTES)
        digest = hashlib.sha256()
        output = []
        with self._backend.open(entry["path"]) as stream:
            if stream.size != size:
                raise ModelBundleError(f"Vector size mismatch: {name}")
            consumed = 0
            while consumed < size:
                chunk = stream.read(min(READ_CHUNK_BYTES, size - consumed))
                if not chunk or len(chunk) % _F32.size:
                    raise ModelBundleError(f"Truncated float32 vector: {name}")
                digest.update(chunk)
                for value, in struct.iter_unpack("<f", chunk):
                    if not math.isfinite(value):
                        raise ModelBundleError(f"Nonfinite float32 vector: {name}")
                    output.append(value)
                consumed += len(chunk)
            if stream.read(1):
                raise ModelBundleError(f"Vector grew during read: {name}")
        if digest.hexdigest() != entry["sha256"]:
            raise ModelBundleError(f"Vector checksum mismatch: {name}")
        return output

    def matrix_blocks(self, name):
        """Read-verified tiles of a dense F32 matrix, in row order."""
        entry = self._entry(name)
        if entry["codec"] not in DENSE_CODECS.values():
            raise ModelBundleError(f"Tensor is not a dense matrix: {name}")
        return self._matrix_blocks[name]

    def read_matrix_block_into(self, name, block_index, destination):
        """Read one whole block into an exact buffer, verifying its checksum.

        Blocks are the unit of verification: a partial read could not check
        the bytes it consumed, so the writer's block size is also the reader's.
        """
        blocks = self.matrix_blocks(name)
        if type(block_index) is not int or not 0 <= block_index < len(blocks):
            raise ModelBundleError(f"Block index outside the matrix: {name}")
        block = blocks[block_index]
        try:
            view = memoryview(destination).cast("B")
        except (TypeError, ValueError) as error:
            raise ModelBundleError("Destination must be a contiguous byte buffer") from error
        if view.readonly or len(view) != block["bytes"]:
            raise ModelBundleError("Destination must be writable and match the block byte size")
        entry = self._entry(name)
        cols = entry["shape"][1]
        offset = block["start_row"] * cols * DENSE_WIDTH[entry["codec"]]
        digest = hashlib.sha256()
        with self._backend.open(entry["path"]) as stream:
            if stream.size != entry["file_bytes"]:
                raise ModelBundleError(f"Matrix size mismatch: {name}")
            # Relative to the tensor, which may be a section of a container.
            stream.seek(offset)
            consumed = 0
            while consumed < block["bytes"]:
                end = min(consumed + READ_CHUNK_BYTES, block["bytes"])
                filled = consumed
                while filled < end:
                    count = stream.readinto(view[filled:end])
                    if not count:
                        raise ModelBundleError(f"Truncated float32 matrix: {name}")
                    filled += count
                chunk = view[consumed:end]
                digest.update(chunk)
                for value, in (_F32 if DENSE_WIDTH[entry["codec"]] == 4 else _F16).iter_unpack(bytes(chunk)):
                    if not math.isfinite(value):
                        raise ModelBundleError(f"Nonfinite float matrix: {name}")
                consumed = end
        if digest.hexdigest() != block["sha256"]:
            raise ModelBundleError(f"Matrix block checksum mismatch: {name}")
        return block["bytes"]

    def read_f32_into(self, name, destination):
        """Read a RAW_F32 vector into an exact writable byte buffer, without a list.

        Bytes retain the persisted little-endian representation. The caller must
        discard the destination after any error. No weight-sized staging copy is
        allocated; native callers on other endiannesses must convert explicitly.
        """
        entry = self._entry(name)
        if entry["codec"] != "RAW_F32":
            raise ModelBundleError(f"Tensor is not RAW_F32: {name}")
        try:
            view = memoryview(destination).cast("B")
        except (TypeError, ValueError) as error:
            raise ModelBundleError("Destination must be a contiguous byte buffer") from error
        size = entry["file_bytes"]
        if view.readonly or len(view) != size:
            raise ModelBundleError("Destination must be writable and match the vector byte size")
        digest = hashlib.sha256()
        with self._backend.open(entry["path"]) as stream:
            if stream.size != size:
                raise ModelBundleError(f"Vector size mismatch: {name}")
            consumed = 0
            while consumed < size:
                end = min(consumed + READ_CHUNK_BYTES, size)
                filled = consumed
                while filled < end:
                    count = stream.readinto(view[filled:end])
                    if not count:
                        raise ModelBundleError(f"Truncated float32 vector: {name}")
                    filled += count
                chunk = view[consumed:end]
                digest.update(chunk)
                for value, in struct.iter_unpack("<f", chunk):
                    if not math.isfinite(value):
                        raise ModelBundleError(f"Nonfinite float32 vector: {name}")
                consumed = end
            if stream.read(1):
                raise ModelBundleError(f"Vector grew during read: {name}")
        if digest.hexdigest() != entry["sha256"]:
            raise ModelBundleError(f"Vector checksum mismatch: {name}")
        return size

    @property
    def source_kind(self):
        """Either "directory" or "container"; nothing else in the API changes with it."""
        return self._backend.kind

    @property
    def container(self):
        """The validated container index, or None when reading a directory."""
        return getattr(self._backend, "container", None)

    def _storage(self):
        """Measured file-count and byte cost of this bundle's storage form.

        `nxb_vs_directory_bytes` is signed on purpose. M6.02c measured that the
        NexaPack container charges a fixed overhead per packed tensor; nesting
        the same `.nxp` files verbatim keeps every one of those, adds a 4096
        byte slot boundary per section, and can therefore come out larger than
        the directory. The number says which way it went, it does not promise
        a direction.
        """
        tensors, assets = self._manifest["tensors"], self._manifest["assets"]
        manifest_bytes = self._backend.size(MANIFEST_NAME)
        declared = manifest_bytes + sum(entry["file_bytes"] for entry
                                        in (*tensors.values(), *assets.values()))
        files = len(tensors) + len(assets) + 1
        result = {"source_kind": self._backend.kind, "bundle_file_bytes": declared,
                  "directory_files": files, "manifest_file_bytes": manifest_bytes}
        container = self.container
        if container is None:
            result.update({"container": None, "stored_files": files,
                           "container_overhead_bytes": None, "nxb_vs_directory_bytes": None})
            return result
        measurements = container.measurements()
        result.update({"container": measurements, "stored_files": 1,
                       "container_overhead_bytes": measurements["container_overhead_bytes"],
                       "nxb_vs_directory_bytes": measurements["file_bytes"] - result["bundle_file_bytes"]})
        return result

    def inspect(self):
        tensors = self._manifest["tensors"]
        summaries = []
        for name, entry in sorted(tensors.items()):
            logical_bytes = math.prod(entry["shape"]) * _F32.size
            q4 = entry["codec"] in PACKED_CODECS.values()
            payload_bytes = self._q4_summaries[name]["packed_payload_bytes"] if q4 else entry["file_bytes"]
            summaries.append({"name": name, "shape": list(entry["shape"]), "codec": entry["codec"],
                              "storage_bits": CODEC_BITS[entry["codec"]],
                              "group_size": self._q4_summaries[name]["group_size"] if q4 else None,
                              "packed_payload_bytes": payload_bytes, "physical_file_bytes": entry["file_bytes"],
                              "logical_f32_bytes": logical_bytes, "compression_vs_f32": logical_bytes / payload_bytes})
        return {"format": FORMAT, "format_version": FORMAT_VERSION,
                "architecture": self.config.architecture, "config": self.config.to_dict(),
                "physical_tensors": len(tensors), "logical_tensors": len(self.tensor_names),
                "tensor_names": list(self.tensor_names), "aliases": dict(self._manifest["aliases"]),
                "tensors": summaries,
                "storage": self._storage(),
                "tensor_file_bytes": sum(entry["file_bytes"] for entry in tensors.values()),
                "packed_payload_bytes": sum(entry["packed_payload_bytes"] for entry in summaries),
                "logical_f32_bytes": sum(entry["logical_f32_bytes"] for entry in summaries),
                "compression_basis": "logical_f32_bytes / packed_payload_bytes; excludes file headers and aliases",
                "tokenizer_assets": sorted(self._manifest["assets"]),
                "validation": {"q4_headers": True, "q4_payload": "lazy block checksums",
                               "raw_vectors": "lazy size, checksum and finite checks in read_f32",
                               "tokenizer_assets": True}}

    def close(self):
        self._closed = True
        self._backend.close()

    def __enter__(self):
        if self._closed:
            raise ModelBundleError("Model bundle reader is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
