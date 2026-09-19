"""Local Safetensors parsing, sharding, provenance, and Llama import checks."""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compiler.importers.safetensors import (
    MAX_HEADER_BYTES, READ_CHUNK_BYTES, SafeTensorCheckpoint,
    SafeTensorError, SafeTensorReader,
)


def write_fixture(path, tensors, *, metadata=None):
    """Independent minimal Safetensors writer for locally generated fixtures."""
    header, data = {}, bytearray()
    if metadata is not None:
        header['__metadata__'] = metadata
    for name, (dtype, shape, values) in tensors.items():
        start = len(data)
        for value in values:
            if dtype == 'BF16':
                bits, = struct.unpack('<I', struct.pack('<f', value))
                data.extend(struct.pack('<H', bits >> 16))
            else:
                data.extend(struct.pack('<f' if dtype == 'F32' else '<e', value))
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [start, len(data)]}
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + data)


class SafetensorsRegressions(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='nexa-safetensors-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / 'model.safetensors'

    def raw(self, header, payload=b''):
        encoded = header if isinstance(header, bytes) else json.dumps(header).encode()
        self.path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)

    def test_float_dtypes_vectors_scalars_and_empty_tensors(self):
        tensors = {
            'f32': ('F32', [2, 3], [1, -2, 3.5, 0, 7, -7]),
            'f16': ('F16', [3], [0.5, -4, 8]),
            'bf16': ('BF16', [1, 2], [1.5, -2.5]),
            'scalar': ('F32', [], [7]),
            'empty': ('F32', [0, 3], []),
        }
        write_fixture(self.path, tensors, metadata={'format': 'pt'})
        with SafeTensorReader(self.path) as reader:
            self.assertEqual(reader.payload_bytes_read, 0)
            self.assertEqual(reader.metadata, {'format': 'pt'})
            for name, (_, shape, values) in tensors.items():
                rows = [list(row) for row in reader.iter_rows(name)]
                self.assertEqual([value for row in rows for value in row], values)
                self.assertEqual(reader.tensors[name]['shape'], tuple(shape))
            self.assertEqual([list(row) for row in reader.iter_rows('scalar')], [[7]])
            self.assertEqual(list(reader.iter_rows('empty')), [])

    def test_nan_and_inf_are_legal_tensor_values_but_not_header_values(self):
        write_fixture(self.path, {'x': ('F32', [2], [float('nan'), float('inf')])})
        with SafeTensorReader(self.path) as reader:
            row = list(next(reader.iter_rows('x')))
            self.assertTrue(math.isnan(row[0]))
            self.assertEqual(row[1], float('inf'))
        self.raw(b'{"__metadata__":{"bad":NaN}}')
        with self.assertRaises(SafeTensorError):
            SafeTensorReader(self.path)

    def test_rows_are_lazy_and_reads_are_chunk_bounded(self):
        count = READ_CHUNK_BYTES // 4 + 100
        write_fixture(self.path, {'large': ('F32', [1, count], (1 for _ in range(count)))})
        with SafeTensorReader(self.path) as reader:
            stream = reader._stream
            proxy = mock.Mock(wraps=stream)
            proxy.closed = False
            sizes = []
            def readinto(buffer):
                sizes.append(len(buffer))
                return stream.readinto(buffer)
            proxy.readinto.side_effect = readinto
            reader._stream = proxy
            row = next(reader.iter_rows('large'))
            self.assertEqual(reader.payload_bytes_read, 0)
            self.assertEqual(sum(row), count)
            self.assertEqual(sizes, [READ_CHUNK_BYTES, 400])
            self.assertEqual(reader.payload_bytes_read, count * 4)
            proxy.read.assert_not_called()

    def test_invalid_headers_offsets_shapes_and_metadata_rejected(self):
        entry = {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}
        headers = [
            {'x': dict(entry, dtype='I64')}, {'x': dict(entry, shape=[True])},
            {'x': dict(entry, shape=[-1])}, {'x': dict(entry, shape=[2])},
            {'x': dict(entry, data_offsets=[1, 5])},
            {'x': dict(entry, data_offsets=[4, 0])},
            {'x': entry, 'y': entry}, {'x': dict(entry, extra=1)},
            {'x': entry, '__metadata__': {'bad': 1}},
            {'x': entry, '__metadata__': []},
            {'x': dict(entry, shape=[1 << 63])},
        ]
        for header in headers:
            with self.subTest(header=header):
                self.raw(header, bytes(4))
                with self.assertRaises(SafeTensorError):
                    SafeTensorReader(self.path)
        for header in (b' {"x":1}', b'[]', b'{"x":1,"x":2}', b'{"__metadata__":{"x":1,"x":2}}', b'{\xff}'):
            self.raw(header)
            with self.assertRaises(SafeTensorError):
                SafeTensorReader(self.path)

    def test_truncation_trailing_bytes_and_oversized_headers_rejected(self):
        write_fixture(self.path, {'x': ('F32', [1], [1])})
        data = self.path.read_bytes()
        variants = [data[:size] for size in (0, 7, 10, len(data) - 1)]
        variants += [data + b'\0', struct.pack('<Q', MAX_HEADER_BYTES + 1) + b'{}']
        for variant in variants:
            self.path.write_bytes(variant)
            with self.assertRaises(SafeTensorError):
                SafeTensorReader(self.path)

    def test_changed_file_and_closed_reader_rejected(self):
        write_fixture(self.path, {'x': ('F32', [1], [1])})
        with SafeTensorReader(self.path) as reader:
            with self.path.open('ab') as stream:
                stream.write(b'\0')
            with self.assertRaisesRegex(SafeTensorError, 'changed'):
                list(reader.iter_rows('x'))
        with self.assertRaisesRegex(SafeTensorError, 'closed'):
            list(reader.iter_rows('x'))

    def index(self, assignments, *, metadata=None):
        path = self.directory / 'model.safetensors.index.json'
        path.write_text(json.dumps({'metadata': metadata or {}, 'weight_map': assignments}))

    def test_checkpoint_shards_provenance_and_reopening(self):
        first, second = self.directory / 'one.safetensors', self.directory / 'two.safetensors'
        write_fixture(first, {'a': ('F32', [1, 2], [1, 2])})
        write_fixture(second, {'b': ('BF16', [2], [3, 4])})
        self.index({'a': first.name, 'b': second.name}, metadata={'total_size': 12})
        checkpoint = SafeTensorCheckpoint(self.directory)
        self.assertEqual(checkpoint.tensor_shapes, {'a': (1, 2), 'b': (2,)})
        self.assertEqual([list(row) for row in checkpoint.iter_rows('b')], [[3, 4]])
        self.assertEqual([list(row) for row in checkpoint.iter_rows('a')], [[1, 2]])
        provenance = checkpoint.provenance()
        self.assertEqual(provenance['hash_scope'], 'complete_files')
        for entry in provenance['files']:
            self.assertEqual(entry['sha256'], hashlib.sha256((self.directory / entry['path']).read_bytes()).hexdigest())
        write_fixture(first, {'a': ('F32', [1, 2], [9, 2])})
        with self.assertRaisesRegex(SafeTensorError, 'changed'):
            list(checkpoint.iter_rows('a'))
        with self.assertRaisesRegex(SafeTensorError, 'changed'):
            checkpoint.provenance()

    def test_checkpoint_index_must_match_real_shards(self):
        shard = self.directory / 'one.safetensors'
        write_fixture(shard, {'a': ('F32', [1], [1])})
        invalid = [({'a': shard.name, 'b': shard.name}, {}),
                   ({'b': shard.name}, {}), ({'a': shard.name}, {'total_size': 5}),
                   ({'a': shard.name}, {'total_size': True})]
        for assignments, metadata in invalid:
            self.index(assignments, metadata=metadata)
            with self.assertRaises(SafeTensorError):
                SafeTensorCheckpoint(self.directory)
        other = self.directory / 'two.safetensors'
        write_fixture(other, {'a': ('F32', [1], [1])})
        self.index({'a': shard.name, 'b': other.name})
        with self.assertRaises(SafeTensorError):
            SafeTensorCheckpoint(self.directory)

    def test_checkpoint_rejects_traversal_absolute_paths_and_ambiguous_sources(self):
        for filename in ('../outside.safetensors', '/outside.safetensors',
                         'dir/../outside.safetensors', 'C:\\outside.safetensors'):
            self.index({'x': filename})
            with self.assertRaises(SafeTensorError):
                SafeTensorCheckpoint(self.directory)
        write_fixture(self.path, {'x': ('F32', [1], [1])})
        with self.assertRaises(SafeTensorError):
            SafeTensorCheckpoint(self.directory)

    @unittest.skipIf(os.name == 'nt', 'Symlinks need Windows developer mode')
    def test_checkpoint_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory(prefix='nexa-outside-') as outside:
            target = Path(outside) / 'outside.safetensors'
            write_fixture(target, {'x': ('F32', [1], [1])})
            (self.directory / 'shard.safetensors').symlink_to(target)
            self.index({'x': 'shard.safetensors'})
            with self.assertRaisesRegex(SafeTensorError, 'escapes'):
                SafeTensorCheckpoint(self.directory)


class LlamaImportRegressions(unittest.TestCase):
    def setUp(self):
        from compiler.model_config import ModelConfig
        self.temporary = tempfile.TemporaryDirectory(prefix='nexa-llama-import-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / 'source'
        self.source.mkdir()
        self.destination = self.directory / 'bundle'
        self.hf = {'model_type': 'llama', 'hidden_size': 4, 'intermediate_size': 8,
                   'num_hidden_layers': 1, 'num_attention_heads': 2, 'num_key_value_heads': 1,
                   'vocab_size': 8, 'max_position_embeddings': 32, 'tie_word_embeddings': True}
        self.config = ModelConfig.from_hf_config(self.hf)
        (self.source / 'config.json').write_text(json.dumps(self.hf))
        self.tensors = {name: ('F32', list(shape), [1.0] * math.prod(shape))
                        for name, shape in self.config.required_tensor_shapes().items()}

    def write(self):
        write_fixture(self.source / 'model.safetensors', self.tensors)

    def test_import_preserves_assets_aliases_and_verifiable_provenance(self):
        from compiler.importers.llama import import_llama_checkpoint
        from runtime.nexapack.bundle import ModelBundleReader
        self.write()
        tokenizer = b'{"fixture":"local tokenizer asset"}'
        (self.source / 'tokenizer.json').write_bytes(tokenizer)
        result = import_llama_checkpoint(self.source, self.destination, group_size=3, block_rows=2)
        self.assertEqual((result['format'], result['format_version']), ('NexaModelBundle', 1))
        self.assertEqual(result['tensor_count'], len(self.tensors))
        self.assertEqual(result['tensor_aliases'], {'lm_head.weight': 'model.embed_tokens.weight'})
        self.assertIn('tokenizer.json', result['assets'])
        reader = ModelBundleReader(self.destination)
        self.assertEqual(reader.config.to_dict(), self.config.to_dict())
        for entry in result['provenance']['files']:
            self.assertEqual(entry['sha256'], hashlib.sha256((self.source / entry['path']).read_bytes()).hexdigest())

    def test_present_tied_head_requires_matching_values(self):
        from compiler.importers.llama import import_llama_checkpoint
        self.tensors['lm_head.weight'] = self.tensors['model.embed_tokens.weight']
        self.write()
        result = import_llama_checkpoint(self.source, self.destination, group_size=3)
        self.assertEqual(result['tied_weights_verified'], ['lm_head.weight'])
        dtype, shape, values = self.tensors['lm_head.weight']
        self.tensors['lm_head.weight'] = dtype, shape, [2.0] + values[1:]
        self.write()
        with self.assertRaisesRegex(SafeTensorError, 'differs'):
            import_llama_checkpoint(self.source, self.directory / 'bad-bundle')
        self.assertFalse((self.directory / 'bad-bundle').exists())

    def test_sharded_half_and_bfloat_weights_import(self):
        from compiler.importers.llama import import_llama_checkpoint
        first, second = {}, {}
        assignments = {}
        for index, (name, (_, shape, values)) in enumerate(self.tensors.items()):
            target = first if index % 2 else second
            target[name] = 'F16' if index % 2 else 'BF16', shape, values
            assignments[name] = 'one.safetensors' if index % 2 else 'two.safetensors'
        write_fixture(self.source / 'one.safetensors', first)
        write_fixture(self.source / 'two.safetensors', second)
        (self.source / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': assignments}))
        result = import_llama_checkpoint(self.source, self.destination, group_size=3)
        self.assertEqual(result['tensor_count'], len(self.tensors))
        self.assertEqual(len(result['provenance']['files']), 3)

    def test_untied_head_required_and_remote_code_configuration_rejected(self):
        from compiler.importers.llama import import_llama_checkpoint
        self.hf['tie_word_embeddings'] = False
        (self.source / 'config.json').write_text(json.dumps(self.hf))
        self.write()
        with self.assertRaisesRegex(SafeTensorError, 'missing tensors: lm_head.weight'):
            import_llama_checkpoint(self.source, self.destination)
        self.hf['auto_map'] = {'AutoModel': 'custom_remote.Model'}
        (self.source / 'config.json').write_text(json.dumps(self.hf))
        with self.assertRaisesRegex(ValueError, 'auto_map'):
            import_llama_checkpoint(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_changed_config_rejected_before_writing(self):
        from compiler.importers.llama import import_llama_checkpoint
        self.write()
        def changed_checkpoint(directory):
            checkpoint = SafeTensorCheckpoint(directory)
            self.hf['rope_theta'] = 50000
            (self.source / 'config.json').write_text(json.dumps(self.hf))
            return checkpoint
        with mock.patch('compiler.importers.llama.SafeTensorCheckpoint', side_effect=changed_checkpoint):
            with self.assertRaisesRegex(SafeTensorError, 'config.json changed'):
                import_llama_checkpoint(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_assets_changed_after_hash_rejected_before_publication(self):
        from compiler.importers.llama import import_llama_checkpoint
        from runtime.nexapack.bundle import write_model_bundle
        self.write()
        for name in ('config.json', 'tokenizer.json'):
            (self.source / 'config.json').write_text(json.dumps(self.hf))
            (self.source / 'tokenizer.json').write_text('{"fixture":true}')
            def change_then_write(*args, **kwargs):
                path = self.source / name
                path.write_bytes(path.read_bytes() + b' ')
                return write_model_bundle(*args, **kwargs)
            with mock.patch('compiler.importers.llama.write_model_bundle', side_effect=change_then_write):
                with self.subTest(asset=name), self.assertRaisesRegex(ValueError, 'checksum|changed'):
                    import_llama_checkpoint(self.source, self.destination)
            self.assertFalse(self.destination.exists())

    def test_config_mutation_during_hash_is_rejected(self):
        from compiler.importers.llama import import_llama_checkpoint
        from compiler.importers.safetensors import file_sha256
        self.write()
        def change_during_hash(path):
            digest = file_sha256(path)
            if path.name == 'config.json':
                path.write_bytes(path.read_bytes() + b' ')
            return digest
        with mock.patch('compiler.importers.llama.file_sha256', side_effect=change_during_hash):
            with self.assertRaisesRegex(SafeTensorError, 'changed'):
                import_llama_checkpoint(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_extra_missing_wrong_shape_and_nonfinite_weights_fail_atomically(self):
        from compiler.importers.llama import import_llama_checkpoint
        name = 'model.norm.weight'
        original = dict(self.tensors)
        cases = [dict(original, unexpected=('F32', [1], [1])),
                 {key: value for key, value in original.items() if key != name},
                 dict(original, **{name: ('F32', [2], [1, 1])}),
                 dict(original, **{name: ('F32', [4], [float('nan'), 1, 1, 1])})]
        for tensors in cases:
            self.tensors = tensors
            self.write()
            with self.assertRaises(ValueError):
                import_llama_checkpoint(self.source, self.destination)
            self.assertFalse(self.destination.exists())


if __name__ == '__main__':
    unittest.main()
