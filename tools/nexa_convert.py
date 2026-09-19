#!/usr/bin/env python3
"""Convert float32 matrices, legacy TQ01 vectors or a local Llama checkpoint to NexaPack."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.nexapack.format import NexaPackReader, write_q4_matrix, READ_CHUNK_BYTES
from compiler.planner.memory import parse_memory_size

TQ_DEFAULT_BUDGET = '96KiB'
MAX_CODEBOOK_FILE_BYTES = 8192


def _different_paths(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Input and output must be different files')
    if destination.exists() and source.samefile(destination):
        raise ValueError('Input and output refer to the same file')
    return source, destination


def _load_tq_codebook(path, dim, bits, seed):
    from runtime.nexapack.format import _unique_object, _reject_json_constant
    from runtime.nexapack.tq import validate_tq_parameters, validate_tq_codebook
    with Path(path).open('rb') as stream:
        encoded = stream.read(MAX_CODEBOOK_FILE_BYTES + 1)
    if len(encoded) > MAX_CODEBOOK_FILE_BYTES:
        raise ValueError('TQ codebook file exceeds 8192 bytes')
    try:
        data = json.loads(encoded.decode('utf-8'), object_pairs_hook=_unique_object,
                          parse_constant=_reject_json_constant)
    except RecursionError as error:
        raise ValueError('TQ codebook JSON nesting is too deep') from error
    fields = {'dim', 'bits', 'seed', 'transform_id', 'codebook_f32le'}
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError('Unexpected TQ codebook file fields')
    validate_tq_parameters(data['dim'], data['bits'], data['seed'])
    if (data['dim'], data['bits'], data['seed']) != (dim, bits, seed):
        raise ValueError('TQ codebook dimensions/bits/seed differ from conversion parameters')
    if data['transform_id'] != 'SRHT_XOSHIRO256SS_V1':
        raise ValueError('Unsupported TQ codebook transform')
    validate_tq_codebook(data['codebook_f32le'], bits)
    return data['codebook_f32le']


def _read_f32_rows(stream, rows, cols):
    """Feed the writer with bounded chunks, including when one row is large."""
    scratch = bytearray(min(READ_CHUNK_BYTES, cols * 4))
    view = memoryview(scratch)

    def values():
        chunk = unpacker = None
        try:
            remaining = cols * 4
            while remaining:
                size = min(remaining, len(view))
                filled = 0
                while filled < size:
                    received = stream.readinto(view[filled:size])
                    if not received:
                        raise ValueError("Input matrix was truncated during conversion")
                    filled += received
                chunk = view[:size]
                unpacker = struct.iter_unpack("<f", chunk)
                for value in unpacker:
                    yield value[0]
                unpacker = None
                chunk.release()
                chunk = None
                remaining -= size
        finally:
            unpacker = None
            if chunk is not None:
                chunk.release()
            chunk = None

    current = None
    try:
        for _ in range(rows):
            current = values()
            yield current
            current.close()
            current = None
        if stream.read(1):
            raise ValueError("Input matrix grew during conversion")
    finally:
        if current is not None:
            current.close()
        view.release()
        current = scratch = view = None


def convert_matrix(source, destination, rows, cols, group_size=32, block_rows=64, *,
                   codec='q4', bits=None, seed=None, memory_budget=None, codebook_f32le=None):
    for value, name in [(rows, "rows"), (cols, "cols"), (group_size, "group_size"), (block_rows, "block_rows")]:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if codec not in ('q4', 'tq'):
        raise ValueError('Unsupported matrix codec')
    if codec == 'q4' and any(value is not None for value in (bits, seed, memory_budget, codebook_f32le)):
        raise ValueError('bits, seed, memory_budget and codebook apply only to TQ conversion')
    memory = None
    if codec == 'tq':
        from runtime.nexapack.tq import validate_tq_parameters
        bits = 3 if bits is None else bits
        seed = 42 if seed is None else seed
        validate_tq_parameters(cols, bits, seed)
        budget = parse_memory_size(TQ_DEFAULT_BUDGET if memory_budget is None else memory_budget)
        input_scratch = min(READ_CHUNK_BYTES, cols * 4)
        if budget <= input_scratch:
            raise ValueError('TQ memory budget does not fit the input read scratch')
    source, destination = _different_paths(source, destination)
    row_size = cols * 4
    with source.open("rb", buffering=0) as stream:
        import os
        actual = os.fstat(stream.fileno()).st_size
        if actual != rows * row_size:
            raise ValueError(f"Expected {rows * row_size} input bytes, found {actual}")

        source_rows = _read_f32_rows(stream, rows, cols)
        try:
            if codec == 'q4':
                write_q4_matrix(destination, rows, cols, group_size,
                                source_rows, block_rows=block_rows)
            else:
                from runtime.nexapack.format import write_tq_matrix
                memory = write_tq_matrix(destination, rows, cols, bits, seed,
                                         source_rows, block_rows=block_rows,
                                         memory_budget=budget - input_scratch,
                                         codebook_f32le=codebook_f32le)
                memory = dict(memory)
                memory.update({'budget_bytes': budget, 'input_read_scratch_bytes': input_scratch,
                               'scope': 'managed codec and input read buffers; excludes Python metadata, '
                                        'object/allocator overhead, consumer lists, libraries and OS cache; not RSS/VRAM',
                               'managed_buffers_peak_bound_bytes':
                                   memory['managed_buffers_peak_bound_bytes'] + input_scratch})
        finally:
            source_rows.close()
    with NexaPackReader(destination) as pack:
        result = {"output": str(destination.resolve()), "shape": [rows, cols],
                  "source_bytes": actual, "packed_payload_bytes": rows * pack.row_bytes,
                  "file_bytes": destination.stat().st_size, "codec": pack.codec_id}
        if codec == 'tq':
            result.update({'codec_version': 1, 'bits': bits, 'seed': seed, 'memory': memory,
                           'source_format': 'f32le', 'quality_or_perplexity_measured': False})
        return result


def convert_tq01(source, destination, rows, cols, bits, seed, codebook_f32le, *,
                 source_endianness, block_rows=64, memory_budget=TQ_DEFAULT_BUDGET):
    """Migrate explicit legacy metadata without reconstructing or requantizing vectors."""
    from runtime.nexapack.format import write_tq_records
    from runtime.nexapack.tq import (validate_tq_parameters, validate_tq_codebook,
                                   tq_row_bytes, migrate_tq01_row)
    validate_tq_parameters(cols, bits, seed)
    validate_tq_codebook(codebook_f32le, bits)
    for value, name in ((rows, 'rows'), (block_rows, 'block_rows')):
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if source_endianness not in ('little', 'big'):
        raise ValueError('Legacy TQ01 requires explicit little or big source endianness')
    stride = tq_row_bytes(cols, bits)
    budget = parse_memory_size(memory_budget)
    peak = stride * 3 + 8  # source, canonicalization, return copy and norm byte temporaries
    if peak > budget:
        raise ValueError(f'TQ migration requires {peak} managed buffer bytes; budget is {budget}')
    source, destination = _different_paths(source, destination)
    with source.open('rb', buffering=0) as stream:
        import os
        actual = os.fstat(stream.fileno()).st_size
        if actual != rows * stride:
            raise ValueError(f'Expected {rows * stride} legacy input bytes, found {actual}')

        def records():
            row = bytearray(stride)
            view = memoryview(row)
            try:
                for _ in range(rows):
                    filled = 0
                    while filled < stride:
                        received = stream.readinto(view[filled:])
                        if not received:
                            raise ValueError('Legacy TQ01 input was truncated during conversion')
                        filled += received
                    yield migrate_tq01_row(view, cols, bits, source_endianness=source_endianness)
                if stream.read(1):
                    raise ValueError('Legacy TQ01 input grew during conversion')
            finally:
                view.release()
                row = view = None

        source_rows = records()
        try:
            write_tq_records(destination, rows, cols, bits, seed, codebook_f32le,
                             source_rows, block_rows=block_rows)
        finally:
            source_rows.close()
    return {'output': str(destination.resolve()), 'shape': [rows, cols], 'source_bytes': actual,
            'packed_payload_bytes': rows * stride, 'file_bytes': destination.stat().st_size,
            'codec': 'TQ_MSE_SRHT', 'codec_version': 1, 'bits': bits, 'seed': seed,
            'source_format': 'TQ01', 'source_endianness': source_endianness,
            'requantized': False, 'quality_or_perplexity_measured': False,
            'memory': {'budget_bytes': budget, 'managed_buffers_peak_bound_bytes': peak,
                       'input_row_bytes': stride, 'norm_staging_bytes': 8,
                       'context_bytes': 0, 'quantize_scratch_bytes': 0,
                       'scope': 'source row and canonical record buffers; excludes Python metadata, '
                                'allocator overhead, libraries and OS cache; not RSS/VRAM'}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="Row-major float32 little-endian matrix")
    parser.add_argument("--checkpoint", type=Path, help="Local Llama Safetensors/config/tokenizer directory")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--cols", type=int)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--block-rows", type=int, default=64)
    parser.add_argument('--codec', choices=('q4', 'tq'), default='q4')
    parser.add_argument('--bits', type=int, help='TQ bits per coordinate (default: 3)')
    parser.add_argument('--seed', type=int, help='TQ transform seed, signed int32 (default: 42)')
    parser.add_argument('--memory-budget', help='TQ managed conversion buffers (default: 96KiB)')
    parser.add_argument('--codebook', type=Path, help='Explicit source TQ codebook JSON')
    parser.add_argument('--legacy-tq01', action='store_true', help='Input contains raw legacy TQ01 records')
    parser.add_argument('--source-endianness', choices=('little', 'big'), help='Required for legacy TQ01 migration')
    args = parser.parse_args(argv)
    if (args.input is None) == (args.checkpoint is None):
        parser.error("Supply either a float32 input or --checkpoint")
    if args.input is not None and (args.rows is None or args.cols is None):
        parser.error("Float32 input requires --rows and --cols")
    if args.checkpoint is not None and (args.rows is not None or args.cols is not None):
        parser.error("--checkpoint gets shapes from the architecture; do not pass --rows/--cols")
    tq_options = any(value is not None for value in
                     (args.bits, args.seed, args.memory_budget, args.codebook, args.source_endianness)) or args.legacy_tq01
    if args.codec == 'q4' and tq_options:
        parser.error('TQ options require --codec tq')
    if args.codec == 'tq' and (args.checkpoint is not None or args.group_size is not None):
        parser.error('TQ conversion supports matrices only and does not accept --checkpoint/--group-size')
    if args.legacy_tq01 and (args.source_endianness is None or args.codebook is None):
        parser.error('--legacy-tq01 requires --source-endianness and --codebook from the source')
    if args.source_endianness is not None and not args.legacy_tq01:
        parser.error('--source-endianness requires --legacy-tq01')
    try:
        bits = 3 if args.bits is None else args.bits
        seed = 42 if args.seed is None else args.seed
        codebook = _load_tq_codebook(args.codebook, args.cols, bits, seed) if args.codebook else None
        group_size = 32 if args.group_size is None else args.group_size
        if args.checkpoint is not None:
            from compiler.importers.llama import import_llama_checkpoint
            result = import_llama_checkpoint(args.checkpoint, args.out,
                                             group_size=group_size, block_rows=args.block_rows)
        elif args.legacy_tq01:
            result = convert_tq01(args.input, args.out, args.rows, args.cols, bits, seed, codebook,
                                  source_endianness=args.source_endianness, block_rows=args.block_rows,
                                  memory_budget=args.memory_budget or TQ_DEFAULT_BUDGET)
        else:
            result = convert_matrix(args.input, args.out, args.rows, args.cols,
                                    group_size, args.block_rows, codec=args.codec,
                                    bits=bits if args.codec == 'tq' else None,
                                    seed=seed if args.codec == 'tq' else None,
                                    memory_budget=args.memory_budget, codebook_f32le=codebook)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, ArithmeticError, MemoryError) as exc:
        print(f"NexaPack conversion failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
