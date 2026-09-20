"""MIXED_GROUPED: per-block codec, per-block width, and what it costs.

The measured numbers live here as assertions rather than only in the guide, so
a change in the index layout has to move the documented cost too.
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import q3_reference
from runtime.nexapack import NexaPackError, NexaPackReader
from runtime.nexapack import format as fmt
from tools.nexa_inspect import inspect_artifact

ROWS, COLS, GROUP, BLOCK_ROWS = 12, 10, 4, 3
BLOCK_CODECS = ['q2', 'q3', 'q4', 'q8']
# One deterministic source with the same statistics in every row: no row block
# is easier or harder than another, so nothing here can be read as a matrix
# shaped to make mixing look good.
SOURCE = [[((row * 37 + col * 11) % 23) - 11 + 0.5 * ((row + col) % 3)
           for col in range(COLS)] for row in range(ROWS)]
# SHA-256 of the four uniform files, taken from the writer as it stood before
# MIXED_GROUPED existed (git show HEAD:runtime/nexapack/format.py). A shared
# encoder or a shared layout resolver that drifted would move these.
PINNED_UNIFORM = {
    fmt.CODEC_ID: '9b0f35b09a588bc157190ba4c60c2d243611c0cc7605f521e17548a5c7341bc7',
    fmt.Q8_CODEC_ID: 'dfb51f384502805f0fae74f420883ad945a21924ff14fe44ff97ccb2ac130e6e',
    fmt.Q3_CODEC_ID: '06738d4f63ccefecf921f0a5345975a490b3bf3c552ed5b7ac08787f7a127422',
    fmt.Q2_CODEC_ID: 'db196086ca198c8be827a553ed5123bd135501c056b8aa939f04b06a0ea6ed94',
}


def rows():
    return (list(row) for row in SOURCE)


class MixedWriterRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nexa-mixed-')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def write_uniform(self, codec, name='uniform.nxp', block_rows=BLOCK_ROWS):
        path = self.directory / name
        fmt.write_grouped_matrix(path, ROWS, COLS, GROUP, rows(),
                                 block_rows=block_rows, codec=codec)
        return path

    def write_mixed(self, codecs=BLOCK_CODECS, name='mixed.nxp', block_rows=BLOCK_ROWS):
        path = self.directory / name
        fmt.write_mixed_matrix(path, ROWS, COLS, GROUP, rows(),
                               block_codecs=codecs, block_rows=block_rows)
        return path

    @staticmethod
    def payload_of(path):
        data = path.read_bytes()
        return data[fmt.HEADER.unpack(data[:fmt.HEADER.size])[4]:]

    def test_mixed_payload_is_identical_to_the_uniform_file_it_imitates(self):
        uniform = self.write_uniform(fmt.Q3_CODEC_ID)
        mixed = self.write_mixed(['q3'] * 4)
        # Same rows, same codec, same group: only the index may differ. A
        # second encoder for mixed files would show up right here.
        self.assertEqual(self.payload_of(mixed), self.payload_of(uniform))
        self.assertNotEqual(uniform.read_bytes(), mixed.read_bytes())
        uniform_index = fmt.HEADER.unpack(uniform.read_bytes()[:fmt.HEADER.size])[3]
        mixed_index = fmt.HEADER.unpack(mixed.read_bytes()[:fmt.HEADER.size])[3]
        # Four blocks pay for a codec_id and a row_bytes each; the file-level
        # row_bytes key is given back once.
        self.assertEqual(mixed_index - uniform_index, 147)
        with NexaPackReader(mixed) as reader:
            self.assertEqual(reader.codec_id, fmt.MIXED_CODEC_ID)
            self.assertIsNone(reader.row_bytes)
            self.assertEqual({block['codec_id'] for block in reader.blocks}, {fmt.Q3_CODEC_ID})

    def test_every_block_holds_exactly_the_bytes_its_uniform_file_holds(self):
        uniform = {codec: self.write_uniform(fmt._block_codec(codec), f'{codec}.nxp')
                   for codec in BLOCK_CODECS}
        mixed = self.write_mixed()
        decoders = {'q2': fmt.decode_q2_row, 'q3': fmt.decode_q3_row,
                    'q4': fmt.decode_q4_row, 'q8': fmt.decode_q8_row}
        with NexaPackReader(mixed) as reader:
            blocks = reader.blocks
            self.assertEqual([block['codec_id'] for block in blocks],
                             [fmt._block_codec(codec) for codec in BLOCK_CODECS])
            for codec, block in zip(BLOCK_CODECS, blocks):
                width = block['row_bytes']
                self.assertEqual(width, fmt._row_bytes(COLS, GROUP, fmt._block_codec(codec)))
                with NexaPackReader(uniform[codec]) as reference:
                    for offset in range(block['row_count']):
                        row = block['start_row'] + offset
                        packed = reader.read_rows(row, 1)
                        self.assertEqual(len(packed), width)
                        self.assertEqual(packed, reference.read_rows(row, 1))
                        self.assertEqual(decoders[codec](packed, COLS, GROUP),
                                         decoders[codec](reference.read_rows(row, 1), COLS, GROUP))
                        reader.validate_row(packed, codec_id=codec)

    def test_the_q3_block_matches_the_independent_q3_reference(self):
        mixed = self.write_mixed()
        with NexaPackReader(mixed) as reader:
            block = next(item for item in reader.blocks
                         if item['codec_id'] == fmt.Q3_CODEC_ID)
            for offset in range(block['row_count']):
                row = block['start_row'] + offset
                packed = reader.read_rows(row, 1)
                self.assertEqual(packed, q3_reference.quantize_q3_row(SOURCE[row], GROUP))
                self.assertEqual(q3_reference.decode_q3_row(packed, COLS, GROUP),
                                 fmt.decode_q3_row(packed, COLS, GROUP))

    def test_reads_that_span_blocks_of_different_widths(self):
        mixed = self.write_mixed()
        with NexaPackReader(mixed) as reader:
            blocks = reader.blocks
            expected = b''.join(reader.read_rows(row, 1) for row in range(2, 8))
            reader._payload_bytes_read = 0
            self.assertEqual(reader.read_rows(2, 6), expected)
            # Rows 2..7 touch blocks 0..2, and a checksum covers whole blocks.
            self.assertEqual(reader.payload_bytes_read,
                             sum(block['size'] for block in blocks[:3]))
            self.assertEqual(reader.read_rows(ROWS, 0), b'')
            with self.assertRaises(NexaPackError):
                reader.read_rows(ROWS - 1, 2)
            destination = bytearray(blocks[3]['row_bytes'])
            self.assertEqual(reader.read_rows_into(ROWS - 1, 1, destination), len(destination))
            # Rows 2 and 3 straddle two widths, so a buffer sized by either
            # one of them alone is the wrong size.
            self.assertNotEqual(blocks[0]['row_bytes'], blocks[1]['row_bytes'])
            for width in (blocks[0]['row_bytes'], blocks[1]['row_bytes']):
                with self.subTest(width=width), self.assertRaisesRegex(NexaPackError, 'exactly'):
                    reader.read_rows_into(2, 2, bytearray(2 * width))
            self.assertEqual(len(reader.read_rows(2, 2)),
                             blocks[0]['row_bytes'] + blocks[1]['row_bytes'])

    def test_a_mixed_matrix_refuses_to_guess_a_row_codec(self):
        with NexaPackReader(self.write_mixed()) as reader:
            packed = reader.read_rows(0, 1)
            with self.assertRaisesRegex(NexaPackError, 'per block'):
                reader.validate_row(packed)
            for name in ('MIXED_GROUPED', 'TQ_MSE_SRHT', 'q9', 4, None):
                with self.subTest(name=name), self.assertRaises(NexaPackError):
                    reader.validate_row(packed, codec_id=name)

    def test_uniform_files_are_byte_for_byte_what_they_were(self):
        for codec, digest in PINNED_UNIFORM.items():
            path = self.write_uniform(codec, f'pin-{codec}.nxp', block_rows=3)
            with self.subTest(codec=codec):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
                with NexaPackReader(path) as reader:
                    self.assertEqual(set(reader.metadata), fmt._METADATA_KEYS)
                    self.assertEqual(set(reader.metadata['blocks'][0]), fmt._BLOCK_KEYS)
                    self.assertEqual(reader.row_bytes,
                                     fmt._row_bytes(COLS, GROUP, codec))

    def test_writer_validates_the_plan_before_touching_the_source(self):
        existing = self.write_uniform(fmt.CODEC_ID, 'survivor.nxp')
        original = existing.read_bytes()
        bad = [
            {'block_codecs': ['q3'] * 3},
            {'block_codecs': ['q3'] * 5},
            {'block_codecs': []},
            {'block_codecs': 'q3q3q3q3'},
            {'block_codecs': ['q3', 'q3', 'q3', 'q9']},
            {'block_codecs': ['q3', 'q3', 'q3', fmt.MIXED_CODEC_ID]},
            {'block_codecs': ['q3', 'q3', 'q3', None]},
            {'block_codecs': ['q3'] * 4, 'block_rows': 0},
        ]
        for keywords in bad:
            # A real generator, not a mock: a mock raises on iter() and would
            # make every one of these pass for the wrong reason.
            taken = []

            def source():
                for row in SOURCE:
                    taken.append(row)
                    yield list(row)

            with self.subTest(**keywords):
                with self.assertRaises(NexaPackError):
                    fmt.write_mixed_matrix(existing, ROWS, COLS, GROUP, source(),
                                           **{'block_rows': BLOCK_ROWS, **keywords})
                self.assertEqual(taken, [])
                self.assertEqual(existing.read_bytes(), original)
        with self.assertRaisesRegex(NexaPackError, 'iterable'):
            fmt.write_mixed_matrix(existing, ROWS, COLS, GROUP, mock.Mock(),
                                   block_codecs=BLOCK_CODECS, block_rows=BLOCK_ROWS)
        self.assertEqual(existing.read_bytes(), original)
        self.assertEqual(list(self.directory.glob('.nexapack-*')), [])

    def test_writer_is_atomic_when_the_source_disagrees_with_the_plan(self):
        existing = self.write_uniform(fmt.CODEC_ID, 'survivor.nxp')
        original = existing.read_bytes()
        sources = [[], SOURCE[:ROWS - 1], SOURCE + [SOURCE[0]],
                   [row[:-1] for row in SOURCE], [[float('nan')] * COLS] * ROWS]
        for source in sources:
            with self.subTest(length=len(source)), self.assertRaises(NexaPackError):
                fmt.write_mixed_matrix(existing, ROWS, COLS, GROUP,
                                       (list(row) for row in source),
                                       block_codecs=BLOCK_CODECS, block_rows=BLOCK_ROWS)
            self.assertEqual(existing.read_bytes(), original)
        self.assertEqual(list(self.directory.glob('.nexapack-*')), [])


class MixedIndexRegressions(unittest.TestCase):
    """Refuse each way a per-block index can lie, one mutation at a time."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nexa-mixed-index-')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / 'mixed.nxp'
        fmt.write_mixed_matrix(self.path, ROWS, COLS, GROUP, rows(),
                               block_codecs=BLOCK_CODECS, block_rows=BLOCK_ROWS)
        self.original = self.path.read_bytes()

    def rewrite_metadata(self, mutate):
        header = list(fmt.HEADER.unpack(self.original[:fmt.HEADER.size]))
        metadata = json.loads(self.original[fmt.HEADER.size:fmt.HEADER.size + header[3]])
        mutate(metadata)
        encoded = json.dumps(metadata, separators=(',', ':')).encode()
        self.assertLess(len(encoded) + fmt.HEADER.size, header[4])
        header[3] = len(encoded)
        header[6] = hashlib.sha256(encoded).digest()
        self.path.write_bytes(fmt.HEADER.pack(*header) + encoded +
                              bytes(header[4] - fmt.HEADER.size - len(encoded)) +
                              self.original[header[4]:])

    def test_unopened_file_is_readable_before_any_mutation(self):
        with NexaPackReader(self.path) as reader:
            self.assertEqual(len(reader.blocks), 4)

    def test_index_mutations_are_refused(self):
        q8_width = fmt._row_bytes(COLS, GROUP, fmt.Q8_CODEC_ID)
        mutations = {
            # size disagrees with row_count * the width this codec implies,
            # while the blocks still tile the payload without a gap.
            'size_for_another_codec':
                lambda m: m['blocks'][0].update(codec_id=fmt.Q8_CODEC_ID, row_bytes=q8_width),
            'declared_width_off_by_one':
                lambda m: m['blocks'][1].update(row_bytes=m['blocks'][1]['row_bytes'] + 1),
            'declared_width_of_a_neighbour':
                lambda m: m['blocks'][1].update(row_bytes=q8_width),
            'overlapping_offset':
                lambda m: m['blocks'][1].update(offset=m['blocks'][0]['offset']),
            'offset_gap':
                lambda m: m['blocks'][1].update(offset=m['blocks'][1]['offset'] + 1),
            'block_without_a_codec':
                lambda m: m['blocks'][3].pop('codec_id'),
            'block_without_a_width':
                lambda m: m['blocks'][3].pop('row_bytes'),
            'file_level_row_bytes_is_back':
                lambda m: m.update(row_bytes=fmt._row_bytes(COLS, GROUP, fmt.CODEC_ID)),
            'uniform_storage_dtype':
                lambda m: m.update(storage_dtype='q4'),
            'uniform_codec_id_over_a_mixed_index':
                lambda m: m.update(codec_id=fmt.CODEC_ID),
            'unknown_codec_version':
                lambda m: m.update(codec_version=2),
            'group_size_that_changes_every_width':
                lambda m: m.update(group_size=GROUP + 1),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                self.rewrite_metadata(mutation)
                with self.assertRaises(NexaPackError):
                    NexaPackReader(self.path)

    def test_an_unreadable_block_codec_is_refused_by_name(self):
        # Deriving the width from an unknown codec already fails, so the
        # membership check is redundant for the refusal. It is not redundant
        # for the message: without it the reader blames group_size, and an
        # operator goes looking at the wrong field of the wrong record.
        for name in ('Q9_GROUPED', fmt.MIXED_CODEC_ID, fmt.TQ_CODEC_ID, 4, None, ['q4']):
            with self.subTest(name=name):
                self.rewrite_metadata(lambda m: m['blocks'][2].update(codec_id=name))
                with self.assertRaisesRegex(NexaPackError, 'block codec_id'):
                    NexaPackReader(self.path)

    def test_a_swapped_pair_of_block_codecs_is_refused(self):
        # Both codecs exist and both widths are consistent with their codec;
        # only the sizes the payload was written with disagree.
        def swap(metadata):
            first, second = metadata['blocks'][0], metadata['blocks'][1]
            first['codec_id'], second['codec_id'] = second['codec_id'], first['codec_id']
            first['row_bytes'], second['row_bytes'] = second['row_bytes'], first['row_bytes']

        self.rewrite_metadata(swap)
        with self.assertRaises(NexaPackError):
            NexaPackReader(self.path)

    def test_a_corrupt_payload_byte_is_still_caught_per_block(self):
        with NexaPackReader(self.path) as reader:
            target = reader.blocks[2]['offset']
        damaged = bytearray(self.original)
        damaged[target] ^= 1
        self.path.write_bytes(damaged)
        with NexaPackReader(self.path) as reader:
            self.assertEqual(len(reader.read_rows(0, 3)), 3 * reader.blocks[0]['row_bytes'])
            with self.assertRaisesRegex(NexaPackError, 'checksum'):
                reader.read_rows(6, 1)


class MixedCostRegressions(unittest.TestCase):
    """What a per-block codec costs, and the block count where it stops paying."""

    def index_bytes(self, metadata):
        return len(fmt._json_bytes(metadata))

    def test_per_block_index_cost_is_a_few_dozen_bytes(self):
        cols, group = 64, 32
        for count in (256, 1024, 2048):
            uniform = fmt._new_metadata(count, cols, group, 1, fmt.Q3_CODEC_ID)[0]
            mixed = fmt._new_mixed_metadata(count, cols, group, 1,
                                            [fmt.Q3_CODEC_ID] * count)[0]
            per_block = (self.index_bytes(mixed) - self.index_bytes(uniform)) / count
            with self.subTest(blocks=count):
                self.assertAlmostEqual(per_block, 39.0, delta=0.1)

    def test_the_file_cost_is_zero_only_until_the_index_crosses_a_page(self):
        cols, group = 64, 32
        first = next(count for count in range(1, 200)
                     if fmt._new_mixed_metadata(count, cols, group, 1,
                                                [fmt.Q3_CODEC_ID] * count)[2]
                     != fmt._new_metadata(count, cols, group, 1, fmt.Q3_CODEC_ID)[2])
        # Below this the extra index bytes disappear into padding the uniform
        # file was already paying for. Reporting that zero as the cost of
        # mixing would be reporting the alignment, not the codec.
        self.assertEqual(first, 23)
        uniform = fmt._new_metadata(first, cols, group, 1, fmt.Q3_CODEC_ID)[2]
        mixed = fmt._new_mixed_metadata(first, cols, group, 1, [fmt.Q3_CODEC_ID] * first)[2]
        self.assertEqual(mixed - uniform, fmt.ALIGNMENT)

    def test_a_mixed_index_describes_fewer_blocks_than_a_uniform_one(self):
        def ceiling(build):
            low, high, best = 1, fmt.MAX_BLOCKS, 0
            while low <= high:
                middle = (low + high) // 2
                try:
                    build(middle)
                except NexaPackError:
                    high = middle - 1
                else:
                    best, low = middle, middle + 1
            return best

        uniform = ceiling(lambda n: fmt._new_metadata(n, 64, 32, 1, fmt.Q3_CODEC_ID))
        mixed = ceiling(lambda n: fmt._new_mixed_metadata(n, 64, 32, 1,
                                                          [fmt.Q3_CODEC_ID] * n))
        # The 1 MiB metadata cap binds before MAX_BLOCKS for both, and the
        # wider index reaches it sooner: matrices a uniform file can describe
        # are refused as mixed.
        self.assertEqual((uniform, mixed), (7716, 5996))
        self.assertLess(mixed, uniform)

    def test_where_mixing_stops_paying(self):
        rows, cols, group = 1024, 64, 32
        wide = fmt._row_bytes(cols, group, fmt.CODEC_ID)
        narrow = fmt._row_bytes(cols, group, fmt.Q2_CODEC_ID)
        self.assertEqual((wide, narrow), (40, 24))
        measured = {}
        for block_rows in (512, 256, 128, 64, 32, 16, 8, 4, 2, 1):
            count = rows // block_rows
            codecs = [fmt.CODEC_ID] * (count // 2) + [fmt.Q2_CODEC_ID] * (count - count // 2)
            uniform = fmt._new_metadata(rows, cols, group, block_rows, fmt.CODEC_ID)
            mixed = fmt._new_mixed_metadata(rows, cols, group, block_rows, codecs)
            saved = rows * wide - sum(block['size'] for block in mixed[0]['blocks'])
            self.assertEqual(saved, (rows // 2) * (wide - narrow))
            measured[count] = mixed[2] - uniform[2]
        # Half the rows are downgraded in every line, so the payload saving is
        # the same 8192 bytes throughout; only the block count changes. The
        # index eats that saving away and then overruns it.
        self.assertEqual(measured[2], -8192)
        self.assertEqual(measured[64], -8192)
        self.assertEqual(measured[128], -4096)
        self.assertEqual(measured[256], 0)
        self.assertEqual(measured[512], 12288)
        self.assertEqual(measured[1024], 32768)

    def test_the_same_curve_on_files_the_writer_actually_produced(self):
        rows, cols, group, block_rows = 1024, 64, 32, 1
        source = [[((row * 7 + col * 3) % 19) - 9 for col in range(cols)]
                  for row in range(rows)]
        codecs = ([fmt.CODEC_ID] * (rows // 2) + [fmt.Q2_CODEC_ID] * (rows // 2))
        with tempfile.TemporaryDirectory(prefix='nexa-mixed-cost-') as name:
            directory = Path(name)
            uniform, mixed = directory / 'u.nxp', directory / 'm.nxp'
            fmt.write_grouped_matrix(uniform, rows, cols, group,
                                     (list(row) for row in source),
                                     block_rows=block_rows, codec=fmt.CODEC_ID)
            fmt.write_mixed_matrix(mixed, rows, cols, group,
                                   (list(row) for row in source),
                                   block_codecs=codecs, block_rows=block_rows)
            with NexaPackReader(mixed) as reader:
                payload = sum(block['size'] for block in reader.blocks)
            self.assertEqual(rows * fmt._row_bytes(cols, group, fmt.CODEC_ID) - payload, 8192)
            # 8192 payload bytes saved, 32768 file bytes lost.
            self.assertEqual(mixed.stat().st_size - uniform.stat().st_size, 32768)


class MixedInspectRegressions(unittest.TestCase):
    def test_inspect_reports_the_composition_and_verifies_per_block(self):
        with tempfile.TemporaryDirectory(prefix='nexa-mixed-inspect-') as name:
            path = Path(name) / 'mixed.nxp'
            fmt.write_mixed_matrix(path, ROWS, COLS, GROUP, rows(),
                                   block_codecs=BLOCK_CODECS, block_rows=BLOCK_ROWS)
            report = inspect_artifact(path, verify=True)
            self.assertEqual(report['codec'], fmt.MIXED_CODEC_ID)
            self.assertEqual(report['shape'], [ROWS, COLS])
            self.assertEqual(sorted(report['composition']),
                             [fmt._block_codec(codec) for codec in sorted(BLOCK_CODECS)])
            for codec in BLOCK_CODECS:
                entry = report['composition'][fmt._block_codec(codec)]
                self.assertEqual(entry['blocks'], 1)
                self.assertEqual(entry['rows'], BLOCK_ROWS)
                self.assertEqual(entry['payload_bytes'],
                                 BLOCK_ROWS * fmt._row_bytes(COLS, GROUP,
                                                             fmt._block_codec(codec)))
            self.assertEqual(report['packed_payload_bytes'],
                             sum(item['size'] for item in report['blocks']))
            self.assertEqual(report['validation']['payload_bytes_read'],
                             report['packed_payload_bytes'])
            self.assertFalse(report['validation']['model_quality_measured'])
            self.assertTrue(report['validation']['codec_validated'])
            json.dumps(report, allow_nan=False)

    def test_inspect_still_reports_a_uniform_matrix_the_old_way(self):
        with tempfile.TemporaryDirectory(prefix='nexa-uniform-inspect-') as name:
            path = Path(name) / 'uniform.nxp'
            fmt.write_grouped_matrix(path, ROWS, COLS, GROUP, rows(),
                                     block_rows=BLOCK_ROWS, codec=fmt.CODEC_ID)
            report = inspect_artifact(path, verify=True)
            self.assertNotIn('composition', report)
            self.assertEqual(report['group_size'], GROUP)
            self.assertEqual(report['packed_payload_bytes'],
                             ROWS * fmt._row_bytes(COLS, GROUP, fmt.CODEC_ID))


if __name__ == '__main__':
    unittest.main()
