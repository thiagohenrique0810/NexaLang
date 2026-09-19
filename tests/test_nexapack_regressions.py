"""NexaPack streaming, codec, corruption, and atomic-write regressions."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import tracemalloc
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.nexapack import (
    READ_CHUNK_BYTES, NexaPackError, NexaPackReader,
    decode_q4_row, quantize_q4_row, write_q4_matrix,
)
from runtime.nexapack import format as fmt


class CodecRegressions(unittest.TestCase):
    def test_exact_bytes_rounding_signed_codes_and_odd_padding(self):
        packed = quantize_q4_row([7, 0.5, -0.5, 1.5, -1.5], 5)
        self.assertEqual(packed.hex(), '0000803f172f0e')
        self.assertEqual(decode_q4_row(packed, 5, 5), [7, 1, -1, 2, -2])
        self.assertEqual(quantize_q4_row([0, -0.0, 0], 3), bytes(6))

    def test_partial_group_is_padded_and_float32_rounding_is_used(self):
        packed = quantize_q4_row(iter([7, -7, 0, 14]), 3)
        self.assertEqual(packed.hex(), '0000803f9700000000400700')
        self.assertEqual(decode_q4_row(packed, 4, 3), [7, -7, 0, 14])
        self.assertEqual(quantize_q4_row([7 + 1e-8, -7 - 1e-8], 2),
                         quantize_q4_row([7, -7], 2))

    def test_nonfinite_overflow_and_unrepresentable_scale_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf'), 1e40, 2 ** -149):
            with self.subTest(value=value), self.assertRaises(NexaPackError):
                quantize_q4_row([value], 1)

    def test_invalid_dimensions_and_empty_rows_rejected(self):
        for size in (0, -1, True, 1.5, fmt.MAX_GROUP_SIZE + 1):
            with self.subTest(size=size), self.assertRaises(NexaPackError):
                quantize_q4_row([1], size)
        with self.assertRaises(NexaPackError):
            quantize_q4_row([], 4)
        with self.assertRaises(NexaPackError):
            decode_q4_row(b'', 1, 0)

    def test_decode_rejects_invalid_scale_codes_and_padding(self):
        invalid = [struct.pack('<fB', scale, 0) for scale in (-1, float('nan'), float('inf'))]
        invalid += [struct.pack('<fB', 1, 8), struct.pack('<fB', 0, 1),
                    struct.pack('<fB', 1, 0x10), b'bad']
        for packed in invalid:
            for check in (decode_q4_row, fmt.validate_q4_row):
                with self.subTest(packed=packed, check=check.__name__), self.assertRaises(NexaPackError):
                    check(packed, 1, 1)
        with self.assertRaises(NexaPackError):
            decode_q4_row(struct.pack('<fB', 1, 0x10), 1, 2)
        self.assertEqual(decode_q4_row(struct.pack('<fB', -0.0, 0), 1, 1), [0])

    def test_codec_validation_does_not_materialize_a_decoded_row(self):
        cols, group = 65536, 32
        payload = bytes((cols // group) * (4 + group // 2))
        already_tracing = tracemalloc.is_tracing()
        if not already_tracing:
            tracemalloc.start()
        try:
            before = tracemalloc.get_traced_memory()[0]
            tracemalloc.reset_peak()
            self.assertIsNone(fmt.validate_q4_row(payload, cols, group))
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            if not already_tracing:
                tracemalloc.stop()
        # A decoded list of these values needs over 2 MiB. Streaming validation
        # uses only transient scalar values and memoryviews over the caller data.
        self.assertLess(peak - before, 256 * 1024)


class NexaPackRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nexa-pack-')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / 'weights.nxp'
        self.values = [[7 * (row + 1), -7, 3.5, -3.5, 0] for row in range(5)]
        write_q4_matrix(self.path, 5, 5, 3, (iter(row) for row in self.values), block_rows=2)
        self.original = self.path.read_bytes()

    def rewrite_metadata(self, mutate=None, raw=None):
        header = list(fmt.HEADER.unpack(self.original[:fmt.HEADER.size]))
        metadata = json.loads(self.original[fmt.HEADER.size:fmt.HEADER.size + header[3]])
        if mutate:
            mutate(metadata)
        encoded = raw if raw is not None else json.dumps(metadata, separators=(',', ':')).encode()
        self.assertLess(len(encoded) + fmt.HEADER.size, header[4])
        header[3] = len(encoded)
        header[6] = hashlib.sha256(encoded).digest()
        self.path.write_bytes(fmt.HEADER.pack(*header) + encoded +
                              bytes(header[4] - fmt.HEADER.size - len(encoded)) + self.original[header[4]:])

    def test_roundtrip_partial_rows_metadata_and_actual_io(self):
        with NexaPackReader(self.path) as reader:
            self.assertEqual((reader.rows, reader.cols, reader.group_size, reader.row_bytes), (5, 5, 3, 12))
            self.assertEqual(reader.payload_bytes_read, 0)
            self.assertEqual(reader.metadata['storage_dtype'], 'q4')
            expected = b''.join(quantize_q4_row(row, 3) for row in self.values[1:4])
            self.assertEqual(reader.read_rows(1, 3), expected)
            self.assertEqual(reader.payload_bytes_read, 4 * reader.row_bytes)
            destination = bytearray(reader.row_bytes)
            self.assertEqual(reader.read_rows_into(4, 1, destination), reader.row_bytes)
            self.assertEqual(destination, quantize_q4_row(self.values[4], 3))
            self.assertEqual(reader.payload_bytes_read, 5 * reader.row_bytes)
            reader.read_rows(4, 1)
            self.assertEqual(reader.payload_bytes_read, 6 * reader.row_bytes)
            self.assertEqual(reader.read_rows(5, 0), b'')

    def test_only_intersecting_blocks_are_read_but_checksums_cover_whole_blocks(self):
        with NexaPackReader(self.path) as reader:
            offset = reader.metadata['blocks'][1]['offset'] + reader.row_bytes
        damaged = bytearray(self.original)
        damaged[offset] ^= 1
        self.path.write_bytes(damaged)
        with NexaPackReader(self.path) as reader:
            self.assertEqual(reader.read_rows(0, 1), quantize_q4_row(self.values[0], 3))
            with self.assertRaisesRegex(NexaPackError, 'checksum'):
                reader.read_rows(2, 1)  # The corrupt row 3 is outside the requested rows.

    def test_io_uses_fixed_scratch_even_for_large_blocks(self):
        rows = 16000
        write_q4_matrix(self.path, rows, 1, 1, ([7] for _ in range(rows)), block_rows=rows)
        with NexaPackReader(self.path) as reader:
            stream = reader._stream
            proxy = mock.Mock(wraps=stream)
            lengths = []
            def readinto(destination):
                lengths.append(len(destination))
                return stream.readinto(destination)
            proxy.readinto.side_effect = readinto
            proxy.closed = False
            reader._stream = proxy
            self.assertEqual(reader.read_rows(12000, 1), quantize_q4_row([7], 1))
            self.assertEqual(reader.payload_bytes_read, rows * reader.row_bytes)
            self.assertEqual(lengths, [READ_CHUNK_BYTES, rows * reader.row_bytes - READ_CHUNK_BYTES])
            proxy.read.assert_not_called()

    def test_reader_metadata_copy_cannot_change_validated_offsets(self):
        with NexaPackReader(self.path) as reader:
            metadata = reader.metadata
            metadata['blocks'][0]['offset'] = 0
            metadata['shape'][0] = 10000
            self.assertEqual(reader.rows, 5)
            self.assertEqual(reader.read_rows(0, 1), quantize_q4_row(self.values[0], 3))

    def test_destination_and_request_validation(self):
        with NexaPackReader(self.path) as reader:
            for destination in (bytes(12), bytearray(11), memoryview(bytearray(24))[::2], None):
                with self.subTest(destination=type(destination)), self.assertRaises(NexaPackError):
                    reader.read_rows_into(0, 1, destination)
            for start, count in ((-1, 1), (0, -1), (5, 1), (6, 0), (True, 1), (0, 1.5)):
                with self.subTest(start=start, count=count), self.assertRaises(NexaPackError):
                    reader.read_rows(start, count)
            with mock.patch.object(fmt, 'MAX_READ_BYTES', 11), self.assertRaises(NexaPackError):
                reader.read_rows(0, 1)
            self.assertEqual(reader.payload_bytes_read, 0)
        with self.assertRaisesRegex(NexaPackError, 'closed'):
            reader.read_rows(0, 0)

    def test_reader_rejects_truncation_trailing_data_and_header_corruption(self):
        variants = [self.original[:n] for n in (0, 63, 100, len(self.original) - 1)]
        variants += [self.original + b'\0']
        for offset in (0, 8, 10, 32, fmt.HEADER.size):
            damaged = bytearray(self.original)
            damaged[offset] ^= 1
            variants.append(damaged)
        for data in variants:
            with self.subTest(size=len(data)):
                self.path.write_bytes(data)
                with self.assertRaises(NexaPackError):
                    NexaPackReader(self.path)

    def test_reader_rejects_bad_index_with_valid_metadata_checksum(self):
        mutations = [
            lambda m: m['blocks'][1].update(offset=m['blocks'][0]['offset']),
            lambda m: m['blocks'][0].update(offset=64),
            lambda m: m['blocks'][0].update(size=1),
            lambda m: m['blocks'][1].update(start_row=1),
            lambda m: m['blocks'][0].update(row_count=1),
            lambda m: m['blocks'][0].update(sha256='../elsewhere'),
            lambda m: m.update(shape=[True, 5]),
            lambda m: m.update(group_size=0),
            lambda m: m.update(row_bytes=13),
            lambda m: m.update(codec_id='UNKNOWN'),
            lambda m: m.update(codec_version=True),
            lambda m: m.update(endianness='big'),
            lambda m: m.update(path='../weights'),
            lambda m: m['blocks'].pop(),
        ]
        for mutation in mutations:
            self.rewrite_metadata(mutation)
            with self.assertRaises(NexaPackError):
                NexaPackReader(self.path)

    def test_reader_rejects_duplicate_keys_invalid_json_and_oversized_header(self):
        for raw in (b'{"shape":[1,1],"shape":[2,2]}', b'{"x":NaN}', b'[]', b'\xff'):
            self.rewrite_metadata(raw=raw)
            with self.assertRaises(NexaPackError):
                NexaPackReader(self.path)
        header = list(fmt.HEADER.unpack(self.original[:fmt.HEADER.size]))
        header[3] = fmt.MAX_METADATA_BYTES + 1
        self.path.write_bytes(fmt.HEADER.pack(*header) + self.original[fmt.HEADER.size:])
        with self.assertRaisesRegex(NexaPackError, 'oversized'):
            NexaPackReader(self.path)

    def test_reader_detects_truncation_after_open(self):
        with NexaPackReader(self.path) as reader:
            with self.path.open('r+b') as stream:
                stream.truncate(len(self.original) - 1)
            with self.assertRaisesRegex(NexaPackError, 'truncated'):
                reader.read_rows(4, 1)

    def test_writer_atomic_on_bad_rows_and_source_failure(self):
        def failing_source():
            yield [1]
            raise RuntimeError('source failed')
        sources = [([], NexaPackError), ([[1], [2]], NexaPackError),
                   ([[float('nan')]], NexaPackError), ([[1, 2]], NexaPackError),
                   ([[]], NexaPackError), (failing_source(), RuntimeError)]
        for source, exception in sources:
            with self.subTest(source=source), self.assertRaises(exception):
                write_q4_matrix(self.path, 1, 1, 1, source)
            self.assertEqual(self.path.read_bytes(), self.original)
            self.assertEqual(list(self.directory.glob('.nexapack-*')), [])

    def test_writer_atomic_if_replace_fails(self):
        with mock.patch.object(fmt.os, 'replace', side_effect=OSError('replace failed')):
            with self.assertRaises(OSError):
                write_q4_matrix(self.path, 1, 1, 1, [[1]])
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(list(self.directory.glob('.nexapack-*')), [])

    def test_writer_validates_limits_before_consuming_source(self):
        source = mock.Mock()
        for dimensions in ((0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1),
                           (fmt.MAX_BLOCKS + 1, 1, 1, 1),
                           (1, fmt.MAX_ROW_BYTES, 1, 1)):
            rows, cols, group, block = dimensions
            with self.subTest(dimensions=dimensions), self.assertRaises(NexaPackError):
                write_q4_matrix(self.path, rows, cols, group, source, block_rows=block)
        with mock.patch.object(fmt, 'MAX_METADATA_BYTES', 20), self.assertRaises(NexaPackError):
            write_q4_matrix(self.path, 1, 1, 1, source)
        self.assertEqual(source.mock_calls, [])
        self.assertEqual(self.path.read_bytes(), self.original)


if __name__ == '__main__':
    unittest.main()
