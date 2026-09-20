"""Versioned multi-tensor bundles built from unchanged NexaPack v1 matrices.

The manifest describes physical files and tied-weight aliases. Opening validates
every Q4 header/index, but weight payload checksums remain lazy. RAW_F32 vectors
are read explicitly; small tokenizer assets are checked in chunks. No asset is
executed.
"""
from __future__ import annotations

from collections.abc import Mapping
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import sys
import tempfile

from compiler.model_config import ModelConfig
from .format import (NexaPackError, NexaPackReader, READ_CHUNK_BYTES,
                     write_grouped_matrix)

# Matrix codecs a bundle may store, with the NexaPack id each one publishes.
PACKED_CODECS = {"q4": "Q4_GROUPED", "q8": "Q8_GROUPED"}
MATRIX_CODECS = (*PACKED_CODECS, "f32")
CODEC_BITS = {"Q4_GROUPED": 4, "Q8_GROUPED": 8, "RAW_F32_MATRIX": 32, "RAW_F32": 32}

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
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value or ":" in value:
        raise ModelBundleError("Invalid bundle file path")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts) or PurePosixPath(value).is_absolute():
        raise ModelBundleError("Bundle file paths must be confined relative paths")
    return parts


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


def _hash_file(path, expected_size, maximum, *, finite_f32=False):
    _integer(expected_size, "file_bytes", minimum=0, maximum=maximum)
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        if os.fstat(stream.fileno()).st_size != expected_size:
            raise ModelBundleError(f"File size mismatch: {path.name}")
        consumed = 0
        while consumed < expected_size:
            chunk = stream.read(min(READ_CHUNK_BYTES, expected_size - consumed))
            if not chunk:
                raise ModelBundleError(f"Truncated file: {path.name}")
            if finite_f32:
                if len(chunk) % _F32.size:
                    raise ModelBundleError(f"Truncated float32 vector: {path.name}")
                if any(not math.isfinite(item[0]) for item in struct.iter_unpack("<f", chunk)):
                    raise ModelBundleError(f"Nonfinite float32 vector: {path.name}")
            digest.update(chunk)
            consumed += len(chunk)
        if stream.read(1):
            raise ModelBundleError(f"File grew during read: {path.name}")
    return digest.hexdigest()


def _check_dense_blocks(name, entry, shape):
    """Validate that the blocks tile every row exactly once, in order."""
    blocks = entry["blocks"]
    rows, cols = shape
    if (not isinstance(blocks, list) or not 0 < len(blocks) <= MAX_MATRIX_BLOCKS
            or entry["file_bytes"] != rows * cols * _F32.size):
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
                  "sha256": block["sha256"], "bytes": block["row_count"] * cols * _F32.size}
                 for block in blocks)


def _metadata_sha(reader):
    # Canonical JSON of the validated NexaPack metadata, not the weight payload.
    return hashlib.sha256(_json_bytes(reader.metadata)).hexdigest()


def _check_q4(path, entry):
    if path.stat().st_size != entry["file_bytes"]:
        raise ModelBundleError(f"Packed file size mismatch: {path.name}")
    reader = NexaPackReader(path)
    try:
        if reader.codec_id != entry["codec"] or reader.codec_version != 1:
            raise ModelBundleError(f'Matrix codec differs from its manifest entry: {path.name}')
        if [reader.rows, reader.cols] != entry["shape"]:
            raise ModelBundleError(f"Q4 shape mismatch: {path.name}")
        if _metadata_sha(reader) != entry["metadata_sha256"]:
            raise ModelBundleError(f"Q4 metadata checksum mismatch: {path.name}")
        return reader
    except BaseException:
        reader.close()
        raise


def _write_raw_matrix(path, rows, cols, source, block_rows):
    """Write a row-major F32 matrix in verified blocks, one row at a time.

    Each block carries its own checksum, so a tile read verifies exactly the
    bytes it consumes instead of trusting a whole-file digest.
    """
    _integer(block_rows, "block_rows", minimum=1)
    total = rows * cols * _F32.size
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
                        encoded += _F32.pack(float(value))
                    except (ValueError, TypeError, OverflowError, struct.error) as error:
                        raise ModelBundleError("Matrix values must fit finite float32") from error
                    consumed += 1
                if consumed != cols:
                    raise ModelBundleError(f"Matrix row {start + offset} has {consumed} of {cols} values")
                for value, in struct.iter_unpack("<f", encoded):
                    if not math.isfinite(value):
                        raise ModelBundleError("Matrix values must be finite")
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


def _publish_directory(source, destination):
    """Atomic publication that never replaces even an empty existing directory."""
    if os.name == "nt":
        # MoveFile semantics used by os.rename reject an existing destination.
        os.rename(source, destination)
        return
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = library.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        arguments = (os.fsencode(source), os.fsencode(destination), 0x4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        function = library.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        arguments = (-100, os.fsencode(source), -100, os.fsencode(destination), 1)  # RENAME_NOREPLACE
    else:
        raise ModelBundleError("Atomic exclusive directory publication is unavailable on this platform")
    function.restype = ctypes.c_int
    if function(*arguments):
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise FileExistsError(number, "Bundle destination already exists", str(destination))
        raise OSError(number, os.strerror(number), str(destination))


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
            if len(shape) == 2 and codecs.get(name, "q4") == "f32":
                relative = f"tensors/{index:04d}.f32"
                blocks, total = _write_raw_matrix(staging / relative, shape[0], shape[1], source, block_rows)
                tensors[name] = {"shape": list(shape), "codec": "RAW_F32_MATRIX", "codec_version": 1,
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
    """
    def __init__(self, directory):
        original = Path(directory)
        if original.is_symlink() or not original.is_dir():
            raise ModelBundleError("Bundle root must be a directory without a symlink")
        self._directory = original.resolve()
        self._closed = False
        self._q4_summaries = {}
        self._matrix_blocks = {}
        path = _confined_file(self._directory, MANIFEST_NAME)
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ModelBundleError("Bundle manifest exceeds metadata limit")
        with path.open("rb") as stream:
            encoded = stream.read(MAX_MANIFEST_BYTES + 1)
        if len(encoded) > MAX_MANIFEST_BYTES:
            raise ModelBundleError("Bundle manifest exceeds metadata limit")
        try:
            manifest = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_pairs,
                                  parse_constant=_reject_constant)
        except (ValueError, UnicodeError, RecursionError) as error:
            raise ModelBundleError(f"Invalid bundle JSON: {error}") from error
        _json_tree(manifest)
        self._validate(manifest)
        self._manifest = manifest

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
            path = _confined_file(self._directory, relative)
            info = path.stat()
            identity = info.st_dev, info.st_ino
            if identity in used_files:
                raise ModelBundleError("Duplicate bundle payload file")
            used_files.add(identity)
            _integer(entry["file_bytes"], "file_bytes", minimum=0)
            if info.st_size != entry["file_bytes"]:
                raise ModelBundleError(f"File size mismatch: {relative}")
            return path

        for name, shape in shapes.items():
            entry = tensors[name]
            dense = len(shape) == 2 and isinstance(entry, dict) and entry.get("codec") == "RAW_F32_MATRIX"
            checksum_key = "blocks" if dense else ("metadata_sha256" if len(shape) == 2 else "sha256")
            _keys(entry, {"shape", "codec", "codec_version", "path", "file_bytes", checksum_key}, "tensor entry")
            if (not isinstance(entry["shape"], list) or any(type(v) is not int for v in entry["shape"])
                    or entry["shape"] != list(shape)):
                raise ModelBundleError(f"Tensor shape mismatch: {name}")
            allowed = ({"RAW_F32_MATRIX"} if dense else set(PACKED_CODECS.values())) if len(shape) == 2 else {"RAW_F32"}
            if entry["codec"] not in allowed or type(entry["codec_version"]) is not int or entry["codec_version"] != 1:
                raise ModelBundleError(f"Unsupported tensor codec: {name}")
            if not dense:
                _sha(entry[checksum_key], name)
            path = payload(entry)
            if dense:
                self._matrix_blocks[name] = _check_dense_blocks(name, entry, shape)
            elif len(shape) == 2:
                with _check_q4(path, entry) as reader:
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
            path = payload(entry)
            total += entry["file_bytes"]
            if total > MAX_ASSETS_TOTAL_BYTES:
                raise ModelBundleError("Tokenizer assets exceed total size limit")
            if _hash_file(path, entry["file_bytes"], MAX_ASSET_BYTES) != entry["sha256"]:
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
        path = _confined_file(self._directory, entry["path"])
        return _check_q4(path, entry)

    def open_q4(self, name):
        entry = self._entry(name)
        if entry["codec"] != "Q4_GROUPED":
            raise ModelBundleError(f"Tensor is not Q4: {name}")
        path = _confined_file(self._directory, entry["path"])
        return _check_q4(path, entry)

    def read_f32(self, name):
        entry = self._entry(name)
        if entry["codec"] != "RAW_F32":
            raise ModelBundleError(f"Tensor is not RAW_F32: {name}")
        path = _confined_file(self._directory, entry["path"])
        size = entry["file_bytes"]
        _integer(size, "RAW_F32 size", maximum=MAX_RAW_BYTES)
        digest = hashlib.sha256()
        output = []
        with path.open("rb", buffering=0) as stream:
            if os.fstat(stream.fileno()).st_size != size:
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
        if entry["codec"] != "RAW_F32_MATRIX":
            raise ModelBundleError(f"Tensor is not a dense F32 matrix: {name}")
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
        path = _confined_file(self._directory, entry["path"])
        cols = entry["shape"][1]
        offset = block["start_row"] * cols * _F32.size
        digest = hashlib.sha256()
        with path.open("rb", buffering=0) as stream:
            if os.fstat(stream.fileno()).st_size != entry["file_bytes"]:
                raise ModelBundleError(f"Matrix size mismatch: {name}")
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
                for value, in struct.iter_unpack("<f", chunk):
                    if not math.isfinite(value):
                        raise ModelBundleError(f"Nonfinite float32 matrix: {name}")
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
        path = _confined_file(self._directory, entry["path"])
        digest = hashlib.sha256()
        with path.open("rb", buffering=0) as stream:
            if os.fstat(stream.fileno()).st_size != size:
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

    def __enter__(self):
        if self._closed:
            raise ModelBundleError("Model bundle reader is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
