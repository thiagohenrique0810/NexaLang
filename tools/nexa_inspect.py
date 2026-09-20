#!/usr/bin/env python3
"""Inspect a NexaPack matrix, a bundle directory or a `.nxb`; verification is explicit.

A container reports what it costs: the bytes it adds over the sum of the
bundle's files, the alignment padding of every section, and how many files the
same model occupies in each form.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.nexapack.bundle import DENSE_CODECS, ModelBundleReader
from runtime.nexapack.container import is_container
from runtime.nexapack.format import MIXED_CODEC_ID, NexaPackReader


def verify_matrix(reader):
    row_count = min(reader.rows, max(1, (1024 * 1024) // reader.row_bytes))
    scratch = bytearray(row_count * reader.row_bytes)
    view = memoryview(scratch)
    tile = record = None
    try:
        for start in range(0, reader.rows, row_count):
            count = min(row_count, reader.rows - start)
            tile = view[:count * reader.row_bytes]
            reader.read_rows_into(start, count, tile)
            tile.release()
            tile = None
            for row in range(count):
                offset = row * reader.row_bytes
                record = view[offset:offset + reader.row_bytes]
                try:
                    reader.validate_row(record)
                finally:
                    record.release()
                    record = None
        return reader.payload_bytes_read
    finally:
        if tile is not None:
            tile.release()
        view.release()
        scratch = view = tile = record = None


def verify_mixed_matrix(reader):
    """Validate a heterogeneous matrix one block at a time, under its codec.

    Blocks are the unit here because they are the unit that owns a codec and a
    checksum; a tile that spanned two of them would have no single row width.
    """
    for block in reader.blocks:
        width = block["row_bytes"]
        tile_rows = min(block["row_count"], max(1, (1024 * 1024) // width))
        scratch = bytearray(tile_rows * width)
        view = memoryview(scratch)
        tile = record = None
        try:
            end = block["start_row"] + block["row_count"]
            for start in range(block["start_row"], end, tile_rows):
                count = min(tile_rows, end - start)
                tile = view[:count * width]
                reader.read_rows_into(start, count, tile)
                tile.release()
                tile = None
                for row in range(count):
                    record = view[row * width:(row + 1) * width]
                    try:
                        reader.validate_row(record, codec_id=block["codec_id"])
                    finally:
                        record.release()
                        record = None
        finally:
            if tile is not None:
                tile.release()
            if record is not None:
                record.release()
            view.release()
            scratch = view = tile = record = None
    return reader.payload_bytes_read


def mixed_composition(reader):
    """How many blocks, rows and payload bytes each codec holds."""
    composition = {}
    for block in reader.blocks:
        entry = composition.setdefault(block["codec_id"],
                                       {"blocks": 0, "rows": 0, "payload_bytes": 0})
        entry["blocks"] += 1
        entry["rows"] += block["row_count"]
        entry["payload_bytes"] += block["size"]
    return composition


def inspect_artifact(path, *, verify=False):
    path = Path(path)
    # A container is recognized by its magic, not by its name: an inspection
    # that trusted the suffix would report on a file it never opened.
    if path.is_dir() or is_container(path):
        with ModelBundleReader(path) as bundle:
            result = bundle.inspect()
            verified = []
            q4_read_bytes = dense_read_bytes = 0
            codecs = {item["name"]: item["codec"] for item in result["tensors"]}
            if verify:
                for name, shape in bundle.config.required_tensor_shapes().items():
                    if codecs[name] in DENSE_CODECS.values():
                        # Verify each stored block, which is the unit the
                        # executor reads and the unit a checksum covers.
                        for index, block in enumerate(bundle.matrix_blocks(name)):
                            dense_read_bytes += bundle.read_matrix_block_into(
                                name, index, bytearray(block["bytes"]))
                    elif len(shape) == 2:
                        with bundle.open_packed(name) as reader:
                            q4_read_bytes += verify_matrix(reader)
                    else:
                        bundle.read_f32(name)
                    verified.append(name)
            container_read_bytes = None
            if verify and bundle.container is not None:
                # Section checksums cover the container's own copy of each
                # file, including the bytes no tensor read would touch.
                container_read_bytes = bundle.container.verify()
            result["validation"].update({"payloads_verified": verify, "verified_tensors": verified,
                                         "checksums_verified": verify, "q4_codec_validated": verify,
                                         "q4_payload_bytes_read": q4_read_bytes,
                                         "dense_payload_bytes_read": dense_read_bytes,
                                         "container_bytes_read": container_read_bytes,
                                         "model_quality_measured": False})
            return result
    with NexaPackReader(path) as reader:
        if reader.codec_id == MIXED_CODEC_ID:
            read = verify_mixed_matrix(reader) if verify else 0
            return {"format": "NexaPack", "format_version": 1,
                    "shape": [reader.rows, reader.cols], "group_size": reader.group_size,
                    "codec": reader.codec_id, "codec_version": reader.codec_version,
                    "packed_payload_bytes": sum(block["size"] for block in reader.blocks),
                    "file_bytes": path.stat().st_size,
                    "composition": mixed_composition(reader),
                    "blocks": [{key: block[key] for key in
                                ("start_row", "row_count", "codec_id", "row_bytes", "size")}
                               for block in reader.blocks],
                    "metadata": reader.metadata,
                    "validation": {"payloads_verified": verify, "payload_bytes_read": read,
                                   "checksums_verified": verify, "codec_validated": verify,
                                   "model_quality_measured": False}}
        result = {"format": "NexaPack", "format_version": 1,
                  "shape": [reader.rows, reader.cols], "group_size": reader.group_size,
                  "packed_payload_bytes": reader.rows * reader.row_bytes,
                  "file_bytes": path.stat().st_size, "metadata": reader.metadata}
        read = verify_matrix(reader) if verify else 0
        result["validation"] = {"payloads_verified": verify, "q4_payload_bytes_read": read,
                                "checksums_verified": verify, "q4_codec_validated": verify,
                                "model_quality_measured": False}
        if reader.codec_id != 'Q4_GROUPED':
            result.pop('group_size')
            result.update({'codec': reader.codec_id, 'codec_version': reader.codec_version,
                           'bits': reader.bits, 'seed': reader.seed})
            result['validation'] = {'payloads_verified': verify, 'payload_bytes_read': read,
                                    'checksums_verified': verify, 'tq_codec_validated': verify,
                                    'model_quality_measured': False}
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--verify", action="store_true", help="Verify payload checksums and codec values")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(inspect_artifact(args.input, verify=args.verify), indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, ArithmeticError, MemoryError) as exc:
        print(f"Nexa inspection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
