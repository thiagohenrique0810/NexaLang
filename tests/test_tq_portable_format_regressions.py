"""Portable TQ records/containers checked without NumPy, Torch or libm codebooks."""
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
import struct
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.nexapack import format as fmt


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def codebook(bits):
    levels = 1 << bits
    values = [(2 * index + 1 - levels) / levels for index in range(levels)]
    return values, struct.pack("<" + "f" * levels, *values).hex()


def signs_for(dim, seed):
    """Independent integer-only SplitMix64 and xoshiro256** implementation."""
    mask = (1 << 64) - 1
    rotate = lambda x, n: ((x << n) | (x >> (64 - n))) & mask
    state = []
    for _ in range(4):
        seed = (seed + 0x9E3779B97F4A7C15) & mask
        z = ((seed ^ (seed >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        state.append(z ^ (z >> 31))
    signs = []
    for _ in range(dim):
        output = rotate((state[1] * 5) & mask, 7) * 9 & mask
        signs.append(1 if output & 1 else -1)
        t = state[1] << 17 & mask
        state[2] ^= state[0]
        state[3] ^= state[1]
        state[1] ^= state[2]
        state[0] ^= state[3]
        state[2] ^= t
        state[3] = rotate(state[3], 45)
    return signs


def hadamard(values):
    values = list(values)
    half = 1
    while half < len(values):
        for start in range(0, len(values), 2 * half):
            for offset in range(half):
                a, b = values[start + offset], values[start + offset + half]
                values[start + offset] = f32(a + b)
                values[start + offset + half] = f32(a - b)
        half *= 2
    return values


def reference_pack(values, bits, seed, centroids):
    """Format oracle: F32 SRHT, explicit centroids and little-endian bit indices."""
    values = list(map(f32, values))
    dim = len(values)
    norm = math.sqrt(sum(float(value) * value for value in values))
    payload = bytearray((dim * bits + 7) // 8)
    if norm:
        rotated = hadamard([f32(value / norm) * sign
                               for value, sign in zip(values, signs_for(dim, seed))])
        denominator = f32(math.sqrt(dim))
        inverse = f32(1 / denominator)
        rotated = [f32(value / denominator if dim < 4 else value * inverse)
                   for value in rotated]
        boundaries = [f32(f32(a + b) * 0.5) for a, b in zip(centroids, centroids[1:])]
        for index, value in enumerate(rotated):
            level = sum(value > boundary for boundary in boundaries)
            for bit in range(bits):
                if level & (1 << bit):
                    target = index * bits + bit
                    payload[target // 8] |= 1 << (target % 8)
    return b"TQ02" + struct.pack("<f", norm) + payload


def reference_decode(record, dim, bits, seed, centroids):
    norm = struct.unpack_from("<f", record, 4)[0]
    if not norm:
        return [0.0] * dim
    payload = int.from_bytes(record[8:], "little")
    values = [centroids[(payload >> (index * bits)) & ((1 << bits) - 1)]
              for index in range(dim)]
    denominator = f32(math.sqrt(dim))
    inverse = f32(1 / denominator)
    if dim < 4:
        values = [f32(value * f32(sign / denominator))
                  for value, sign in zip(hadamard(values), signs_for(dim, seed))]
    else:
        values = hadamard([f32(value * inverse) for value in values])
        values = [value * sign for value, sign in zip(values, signs_for(dim, seed))]
    return [f32(value * norm) for value in values]


class PortableTQFormatRegressions(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="nexa-tq-format-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "weights.nxp"
        self.centroids, self.codebook = codebook(3)
        self.values = [[((row * 3 + col * 5) % 17 - 8) / 4 for col in range(8)]
                       for row in range(5)]
        self.records = [reference_pack(row, 3, -7, self.centroids) for row in self.values]
        fmt.write_tq_records(self.path, 5, 8, 3, -7, self.codebook,
                             iter(self.records), block_rows=2)
        self.original = self.path.read_bytes()

    def rewrite(self, mutate=None, *, raw=None, payload=None):
        header = list(fmt.HEADER.unpack(self.original[:fmt.HEADER.size]))
        metadata = json.loads(self.original[fmt.HEADER.size:fmt.HEADER.size + header[3]])
        if mutate:
            mutate(metadata)
        if payload is not None:
            for block in metadata["blocks"]:
                start = block["offset"] - header[4]
                block["sha256"] = hashlib.sha256(payload[start:start + block["size"]]).hexdigest()
        else:
            payload = self.original[header[4]:]
        encoded = raw if raw is not None else json.dumps(metadata, separators=(",", ":")).encode()
        self.assertLessEqual(fmt.HEADER.size + len(encoded), header[4])
        header[3] = len(encoded)
        header[6] = hashlib.sha256(encoded).digest()
        self.path.write_bytes(fmt.HEADER.pack(*header) + encoded +
                              bytes(header[4] - fmt.HEADER.size - len(encoded)) + payload)

    def test_explicit_metadata_partial_rows_tail_and_detached_index(self):
        with fmt.NexaPackReader(self.path) as reader:
            self.assertEqual((reader.rows, reader.cols, reader.row_bytes), (5, 8, 11))
            self.assertEqual((reader.codec_id, reader.bits, reader.seed), ("TQ_MSE_SRHT", 3, -7))
            self.assertEqual(reader.codebook_f32le, self.codebook)
            self.assertNotIn("group_size", reader.metadata)
            self.assertEqual(reader.metadata["transform_id"], "SRHT_XOSHIRO256SS_V1")
            self.assertEqual(reader.metadata["storage_dtype"], "tq_mse")
            self.assertEqual(reader.payload_bytes_read, 0)
            self.assertEqual(reader.read_rows(1, 3), b"".join(self.records[1:4]))
            self.assertEqual(reader.payload_bytes_read, 4 * reader.row_bytes)
            target = bytearray(reader.row_bytes)
            reader.read_rows_into(4, 1, target)
            self.assertEqual(target, self.records[4])
            self.assertIsNone(reader.validate_row(target))
            self.assertEqual(reader.read_rows(5, 0), b"")
            metadata = reader.metadata
            metadata["shape"][0] = 1
            metadata["blocks"][0]["offset"] = 0
            self.assertEqual(reader.read_rows(0, 1), self.records[0])

    def test_metadata_read_and_record_write_need_no_native_runtime(self):
        from runtime.nexapack import tq
        with mock.patch.object(tq, "TQCodec", side_effect=AssertionError("must stay pure Python")):
            fmt.write_tq_records(self.path, 5, 8, 3, -7, self.codebook, self.records)
            with fmt.NexaPackReader(self.path) as reader:
                reader.validate_row(reader.read_rows(0, 1))

    def test_strict_codec_metadata_types_parameters_and_codebook(self):
        mutations = [
            {"bits": True}, {"bits": 0}, {"bits": 9}, {"bits": 3.0},
            {"seed": True}, {"seed": -(1 << 31) - 1}, {"seed": 1 << 31},
            {"shape": [5, 7]}, {"shape": [True, 8]}, {"shape": [5, 1 << 21]},
            {"codec_id": "TQ01"}, {"codec_version": True}, {"codec_version": 2},
            {"transform_id": "SRHT_UNVERSIONED"}, {"storage_dtype": "q4"},
            {"endianness": "big"}, {"row_bytes": 12}, {"group_size": 3},
            {"codebook_f32le": self.codebook[:-2]}, {"codebook_f32le": "Z" * 64},
            {"codebook_f32le": self.codebook.upper()},
            {"codebook_f32le": [0.0] * 8},
            {"codebook_f32le": struct.pack("<8f", *([0.0] * 8)).hex()},
            {"codebook_f32le": struct.pack("<8f", *reversed(self.centroids)).hex()},
            {"codebook_f32le": struct.pack("<8f", *self.centroids[:-1], float("inf")).hex()},
            {"codebook_f32le": struct.pack("<8f", *self.centroids[:6], 2e38, 3e38).hex()},
        ]
        for values in mutations:
            with self.subTest(values=values):
                self.rewrite(lambda metadata: metadata.update(values))
                with self.assertRaises(ValueError):
                    fmt.NexaPackReader(self.path)

    def test_duplicate_metadata_checksum_padding_and_index_rejected(self):
        metadata = json.loads(self.original[64:64 + fmt.HEADER.unpack(self.original[:64])[3]])
        encoded = json.dumps(metadata, separators=(",", ":")).encode()
        self.rewrite(raw=b'{"bits":3,' + encoded[1:])
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            fmt.NexaPackReader(self.path)
        for offset in (64, fmt.HEADER.unpack(self.original[:64])[4] - 1):
            damaged = bytearray(self.original)
            damaged[offset] ^= 1
            self.path.write_bytes(damaged)
            with self.assertRaises(ValueError):
                fmt.NexaPackReader(self.path)
        for mutation in (lambda m: m["blocks"][1].update(offset=m["blocks"][0]["offset"]),
                         lambda m: m["blocks"][0].update(size=True),
                         lambda m: m["blocks"][0].update(sha256="g" * 64)):
            self.rewrite(mutation)
            with self.assertRaises(ValueError):
                fmt.NexaPackReader(self.path)

    def test_payload_checksums_cover_unrequested_rows_and_remain_lazy(self):
        header = fmt.HEADER.unpack(self.original[:64])
        damaged = bytearray(self.original)
        damaged[header[4] + 3 * 11 + 8] ^= 1
        self.path.write_bytes(damaged)
        with fmt.NexaPackReader(self.path) as reader:
            self.assertEqual(reader.payload_bytes_read, 0)
            self.assertEqual(reader.read_rows(0, 1), self.records[0])
            with self.assertRaisesRegex(ValueError, "checksum"):
                reader.read_rows(2, 1)

    def test_semantically_invalid_rows_rejected_even_with_valid_checksums(self):
        records = [b"TQ01" + self.records[0][4:],
                   b"TQ02" + struct.pack("<f", -1) + self.records[0][8:],
                   b"TQ02" + struct.pack("<f", float("nan")) + self.records[0][8:],
                   b"TQ02" + struct.pack("<f", float("inf")) + self.records[0][8:],
                   b"TQ02" + struct.pack("<f", -0.0) + bytes(3),
                   b"TQ02" + bytes(4) + b"\x01\x00\x00"]
        for record in records:
            with self.subTest(record=record.hex()):
                self.rewrite(payload=record + b"".join(self.records[1:]))
                with fmt.NexaPackReader(self.path) as reader:
                    packed = reader.read_rows(0, 1)
                    with self.assertRaises(ValueError):
                        reader.validate_row(packed)

    def test_bit_padding_and_zero_record_are_canonical(self):
        for dim, bits in ((1, 1), (1, 3), (2, 3), (2, 1), (4, 1)):
            _, centroids = codebook(bits)
            row = b"TQ02" + bytes(4 + (dim * bits + 7) // 8)
            fmt.write_tq_records(self.path, 1, dim, bits, 0, centroids, [row])
            with fmt.NexaPackReader(self.path) as reader:
                reader.validate_row(reader.read_rows(0, 1))
                damaged = bytearray(row)
                damaged[-1] |= 0x80
                with self.assertRaises(ValueError):
                    reader.validate_row(damaged)
            with self.assertRaises(ValueError):
                fmt.write_tq_records(self.path, 1, dim, bits, 0, centroids, [damaged])

    def test_atomic_records_writer_rejects_shape_source_failure_and_invalid_record(self):
        def failing_source():
            yield self.records[0]
            raise RuntimeError("source failed")
        for source, exception in (([], ValueError), (self.records[:1], ValueError),
                                  (self.records + [self.records[0]], ValueError),
                                  ([b"bad"] * 5, ValueError),
                                  ([b"TQ01" + row[4:] for row in self.records], ValueError),
                                  (failing_source(), RuntimeError)):
            with self.subTest(source=type(source).__name__), self.assertRaises(exception):
                fmt.write_tq_records(self.path, 5, 8, 3, -7, self.codebook, source, block_rows=2)
            self.assertEqual(self.path.read_bytes(), self.original)
            self.assertEqual(list(self.directory.glob(".nexapack-*")), [])
        with mock.patch.object(fmt.os, "replace", side_effect=OSError("replace failed")):
            with self.assertRaises(OSError):
                fmt.write_tq_records(self.path, 5, 8, 3, -7, self.codebook, self.records)
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(list(self.directory.glob(".nexapack-*")), [])

    def test_record_limits_reject_before_source_iteration(self):
        class Unreadable:
            def __iter__(self):
                raise AssertionError("invalid preflight consumed source")
        for rows, dim, bits, seed, blocks in ((0, 8, 3, 0, 1), (1, 3, 3, 0, 1),
                                              (1, 8, 9, 0, 1), (1, 8, 3, True, 1),
                                              (1, 1 << 21, 3, 0, 1),
                                              (fmt.MAX_BLOCKS + 1, 8, 3, 0, 1)):
            with self.subTest(values=(rows, dim, bits, seed, blocks)), self.assertRaises(ValueError):
                fmt.write_tq_records(self.path, rows, dim, bits, seed, self.codebook,
                                     Unreadable(), block_rows=blocks)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_q4_file_is_byte_identical_to_pre_extension_golden(self):
        fmt.write_q4_matrix(self.path, 5, 5, 3,
                            ([7 * (row + 1), -7, 3.5, -3.5, 0] for row in range(5)), block_rows=2)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(),
                         "9223f22b59f870896a945ad1304b42716772294ac12a5d10c1739aaa05bbb839")
        with fmt.NexaPackReader(self.path) as reader:
            self.assertEqual(reader.codec_id, "Q4_GROUPED")
            reader.validate_row(reader.read_rows(0, 1))

    def test_q4_executor_rejects_tq_before_payload_or_kernel(self):
        from runtime.nexapack import executor
        with mock.patch.object(fmt.NexaPackReader, "read_rows_into",
                               side_effect=AssertionError("must not read payload")), \
             mock.patch.object(executor, "_load_kernel", side_effect=AssertionError("must not load kernel")):
            with self.assertRaisesRegex(ValueError, "Q4|codec|supported"):
                executor.run_packed_matmul(self.path)

    def test_q4_bundle_rejects_tq_even_when_manifest_size_and_hash_match(self):
        from compiler.model_config import ModelConfig
        from runtime.nexapack.bundle import ModelBundleReader, write_model_bundle
        config = ModelConfig(name="portable_guard", vocab_size=8, hidden_size=8,
                             intermediate_size=8, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=1,
                             max_position_embeddings=8, tie_word_embeddings=True)
        sources = {}
        for name, shape in config.required_tensor_shapes().items():
            sources[name] = (lambda shape=shape: iter([[1.0] * shape[-1]] *
                              (shape[0] if len(shape) == 2 else 1)))
        path = self.directory / "model"
        write_model_bundle(path, config, sources)
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        entry = manifest["tensors"]["model.embed_tokens.weight"]
        disguised = path / entry["path"]
        fmt.write_tq_records(disguised, 8, 8, 3, -7, self.codebook, [self.records[0]] * 8)
        with fmt.NexaPackReader(disguised) as reader:
            entry["file_bytes"] = disguised.stat().st_size
            entry["metadata_sha256"] = hashlib.sha256(json.dumps(
                reader.metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with mock.patch.object(fmt.NexaPackReader, "read_rows_into",
                               side_effect=AssertionError("must not read payload")):
            with self.assertRaisesRegex(ValueError, "Q4|codec|supported"):
                with ModelBundleReader(path) as bundle:
                    bundle.open_q4("model.embed_tokens.weight")


class PortableTQNativeInteropRegressions(unittest.TestCase):
    def test_public_parameters_cannot_invalidate_memory_preflight(self):
        from runtime.nexapack.tq import TQCodec
        with TQCodec(8, 3, 42) as codec:
            for name, value in (("dim", 1 << 20), ("bits", 8), ("seed", -7)):
                with self.subTest(name=name), self.assertRaises(AttributeError):
                    setattr(codec, name, value)
            self.assertEqual((codec.dim, codec.bits, codec.seed), (8, 3, 42))
            self.assertEqual(len(codec.encode_row([0] * 8)), 11)

    def test_reentrant_input_cannot_close_or_reuse_active_context(self):
        from runtime.nexapack.tq import TQCodec
        with TQCodec(8, 3, 42) as codec:
            zero = codec.encode_row([0] * 8)
            for action in (codec.close, lambda: codec.encode_row([0] * 8),
                           lambda: codec.decode_row(zero)):
                def reentrant():
                    action()
                    yield from [0] * 8
                with self.assertRaises(RuntimeError):
                    codec.encode_row(reentrant())
                self.assertEqual(codec.encode_row([0] * 8), zero)

    def test_shared_codec_serializes_threads_and_close_waits_for_active_row(self):
        from runtime.nexapack.tq import TQCodec
        codec = TQCodec(8, 3, 42)
        self.addCleanup(codec.close)
        values = [[row + col / 4 for col in range(8)] for row in range(12)]
        expected = [codec.encode_row(row) for row in values]
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(list(pool.map(codec.encode_row, values)), expected)
            entered, release, closing = threading.Event(), threading.Event(), threading.Event()
            def blocked_row():
                entered.set()
                if not release.wait(10):
                    raise AssertionError("test failed to release row")
                yield from values[0]
            def close():
                closing.set()
                codec.close()
            encoding = pool.submit(codec.encode_row, blocked_row())
            try:
                self.assertTrue(entered.wait(10))
                closed = pool.submit(close)
                self.assertTrue(closing.wait(10))
                self.assertFalse(closed.done())
            finally:
                release.set()
            self.assertEqual(encoding.result(timeout=10), expected[0])
            closed.result(timeout=10)
        with self.assertRaisesRegex(ValueError, "closed"):
            codec.encode_row(values[0])

    def test_frozen_explicit_codebook_record_and_decoded_vector(self):
        from runtime.nexapack.tq import TQCodec
        encoded_centroids = "000060bf000020bf0000c0be000000be0000003e0000c03e0000203f0000603f"
        expected = bytes.fromhex("5451303295866441a34566")
        decoded = [1.2624378204345703, -1.2624379396438599, 3.7873141765594482,
                   -3.7873141765594482, 3.7873141765594482, -6.312190532684326,
                   6.312190532684326, -8.837066650390625]
        with TQCodec(8, 3, -7, codebook_f32le=encoded_centroids) as codec:
            self.assertEqual(codec.encode_row([1, -2, 3, -4, 5, -6, 7, -8]), expected)
            for actual, wanted in zip(codec.decode_row(expected), decoded):
                self.assertAlmostEqual(actual, wanted, delta=2e-6)

    def test_native_encode_decode_matches_independent_explicit_codebook_oracle(self):
        from runtime.nexapack.tq import TQCodec
        rng = random.Random(197)
        for dim, bits, seed in ((1, 1, 0), (2, 3, -7), (4, 1, 42), (8, 3, -7),
                                (16, 5, -(1 << 31)), (32, 8, (1 << 31) - 1), (64, 2, 0)):
            centroids, encoded_centroids = codebook(bits)
            with self.subTest(dim=dim, bits=bits, seed=seed), \
                 TQCodec(dim, bits, seed, codebook_f32le=encoded_centroids) as codec:
                self.assertEqual(codec.codebook_f32le, encoded_centroids)
                for values in ([0.0] * dim, [rng.uniform(-4, 4) for _ in range(dim)]):
                    expected = reference_pack(values, bits, seed, centroids)
                    self.assertEqual(codec.encode_row(iter(values)), expected)
                    decoded = codec.decode_row(expected)
                    reference = reference_decode(expected, dim, bits, seed, centroids)
                    for actual, wanted in zip(decoded, reference):
                        self.assertAlmostEqual(actual, wanted, delta=2e-6 * max(1.0, abs(wanted)))

    def test_matrix_writer_matches_record_writer_and_is_atomic_on_bad_rows(self):
        from runtime.nexapack.tq import TQCodec
        with tempfile.TemporaryDirectory(prefix="nexa-tq-matrix-") as temporary:
            path, other = Path(temporary) / "a.nxp", Path(temporary) / "b.nxp"
            centroids, encoded_centroids = codebook(3)
            values = [[1, -2, 3, -4, 5, -6, 7, -8], [0] * 8, [1] * 8]
            records = [reference_pack(row, 3, 42, centroids) for row in values]
            fmt.write_tq_records(path, 3, 8, 3, 42, encoded_centroids, records, block_rows=2)
            fmt.write_tq_matrix(other, 3, 8, 3, 42, (iter(row) for row in values),
                                block_rows=2, codebook_f32le=encoded_centroids)
            self.assertEqual(path.read_bytes(), other.read_bytes())
            original = other.read_bytes()
            for rows in ([], values + [[1] * 8], [[1] * 7] * 3,
                         [[1] * 9] * 3, [[float("nan")] * 8] * 3):
                with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                    fmt.write_tq_matrix(other, 3, 8, 3, 42, rows,
                                        codebook_f32le=encoded_centroids)
                self.assertEqual(other.read_bytes(), original)
                self.assertEqual(list(Path(temporary).glob(".nexapack-*")), [])
            # Imported centroids, rather than a new Lloyd-Max run, reproduce bytes.
            with TQCodec(8, 3, 42) as generated:
                row = generated.encode_row(values[0])
                with TQCodec(8, 3, 42, codebook_f32le=generated.codebook_f32le) as restored:
                    self.assertEqual(restored.encode_row(values[0]), row)
                    self.assertEqual(restored.decode_row(row), generated.decode_row(row))

    def test_codec_memory_budget_boundary_and_closed_context(self):
        from runtime.nexapack.tq import TQCodec
        _, encoded_centroids = codebook(3)
        with TQCodec(64, 3, -7, codebook_f32le=encoded_centroids) as codec:
            report = codec.memory_report()
            budget = report["managed_buffers_peak_bound_bytes"]
        with self.assertRaisesRegex(ValueError, "closed"):
            codec.encode_row([0] * 64)
        codec.close()
        with TQCodec(64, 3, -7, codebook_f32le=encoded_centroids, memory_budget=budget) as exact:
            self.assertEqual(exact.memory_report()["budget_bytes"], budget)
            self.assertEqual(len(exact.encode_row([0] * 64)), 32)
        with self.assertRaises((ValueError, MemoryError)):
            TQCodec(64, 3, -7, codebook_f32le=encoded_centroids, memory_budget=budget - 1)

    def test_matrix_preflight_rejects_invalid_shape_and_budget_before_consuming_source(self):
        from runtime.nexapack import tq
        class Unreadable:
            def __iter__(self):
                raise AssertionError("preflight consumed source")
        with tempfile.TemporaryDirectory(prefix="nexa-tq-preflight-") as temporary:
            path = Path(temporary) / "a.nxp"
            path.write_bytes(b"original")
            with mock.patch.object(tq, "TQCodec", side_effect=AssertionError("preflight created context")):
                with self.assertRaises(ValueError):
                    fmt.write_tq_matrix(path, 0, 8, 3, 0, Unreadable())
                with self.assertRaises(ValueError):
                    fmt.write_tq_matrix(path, 1, 7, 3, 0, Unreadable())
            with self.assertRaises((ValueError, MemoryError)):
                fmt.write_tq_matrix(path, 1, 8, 3, 0, Unreadable(), memory_budget=1)
            self.assertEqual(path.read_bytes(), b"original")
            self.assertEqual(list(Path(temporary).glob(".nexapack-*")), [])


if __name__ == "__main__":
    unittest.main()
