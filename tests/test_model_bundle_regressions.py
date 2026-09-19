"""Multi-tensor bundle boundaries, integrity, lazy payloads and atomic publish."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compiler.model_config import ModelConfig
from runtime.nexapack import bundle
from runtime.nexapack.bundle import ModelBundleError, ModelBundleReader, write_model_bundle
from runtime.nexapack.format import NexaPackError, NexaPackReader, decode_q4_row


def tiny_config(tied=True):
    return ModelConfig(name="tiny", vocab_size=16, hidden_size=8, intermediate_size=16,
                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                       max_position_embeddings=32, tie_word_embeddings=tied)


def tensor_rows(shape):
    if len(shape) == 1:
        yield (1.0 + c / 8.0 for c in range(shape[0]))
    else:
        for row in range(shape[0]):
            yield (((row * 3 + col * 7) % 17 - 8) / 8.0 for col in range(shape[1]))


class ModelBundleRegressions(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="nexa-bundle-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "model"
        self.config = tiny_config()
        self.sources = {name: (lambda shape=shape: tensor_rows(shape))
                        for name, shape in self.config.required_tensor_shapes().items()}

    def write(self, **kwargs):
        write_model_bundle(self.path, self.config, self.sources, group_size=3, block_rows=3, **kwargs)

    def manifest(self):
        return json.loads((self.path / "manifest.json").read_text())

    def set_manifest(self, manifest):
        (self.path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def no_staging(self):
        self.assertEqual(list(self.directory.glob(".nexa-model-*")), [])

    def test_tiny_llama_shapes_alias_vectors_assets_and_v1_compatibility(self):
        asset = self.directory / "tokenizer.json"
        asset.write_text('{"tokens":["a","b"]}', encoding="utf-8")
        self.write(tokenizer_files={"tokenizer.json": asset}, provenance={"source": "tiny", "sha256": "a" * 64})
        with ModelBundleReader(self.path) as model:
            self.assertEqual(model.config, self.config)
            self.assertEqual(len(model.tensor_names), len(self.sources) + 1)
            self.assertIn("lm_head.weight", model.tensor_names)
            self.assertEqual(model.read_f32("model.norm.weight"), [1 + c / 8 for c in range(8)])
            self.assertEqual(model.inspect()["physical_tensors"], len(self.sources))
            self.assertEqual(model.inspect()["tokenizer_assets"], ["tokenizer.json"])
            self.assertEqual(model.inspect()["validation"]["q4_payload"], "lazy block checksums")
            summary = model.inspect()
            self.assertEqual(len(summary["tensors"]), len(self.sources))
            self.assertEqual(summary["logical_f32_bytes"], self.config.parameter_count() * 4)
            self.assertNotIn("lm_head.weight", [entry["name"] for entry in summary["tensors"]])
            embed = next(entry for entry in summary["tensors"] if entry["name"] == "model.embed_tokens.weight")
            self.assertEqual((embed["storage_bits"], embed["group_size"], embed["packed_payload_bytes"]), (4, 3, 288))
            self.assertEqual(embed["compression_vs_f32"], 512 / 288)
            for name, shape in self.config.required_tensor_shapes().items():
                if len(shape) == 2:
                    with model.open_q4(name) as tensor:
                        self.assertEqual((tensor.rows, tensor.cols), shape)
                        self.assertEqual(tensor.payload_bytes_read, 0)
                        self.assertEqual(len(decode_q4_row(tensor.read_rows(0, 1), shape[1], 3)), shape[1])
            with model.open_q4("lm_head.weight") as alias, model.open_q4("model.embed_tokens.weight") as embedding:
                self.assertEqual(alias.read_rows(0, 16), embedding.read_rows(0, 16))
            manifest = model.manifest
            manifest["tensors"].clear()
            self.assertEqual(model.inspect()["physical_tensors"], len(self.sources))
        manifest = self.manifest()
        entry = manifest["tensors"]["model.embed_tokens.weight"]
        with NexaPackReader(self.path / entry["path"]) as standalone:
            self.assertEqual((standalone.rows, standalone.cols), (16, 8))
            self.assertTrue(standalone.read_rows(0, 1))
        self.assertNotIn("lm_head.weight", manifest["tensors"])
        self.assertEqual(manifest["provenance"], {"schema_version": 1, "data": {"source": "tiny", "sha256": "a" * 64}})
        self.no_staging()

    def test_untied_head_has_a_physical_file(self):
        self.config = tiny_config(False)
        self.sources = {name: (lambda shape=shape: tensor_rows(shape))
                        for name, shape in self.config.required_tensor_shapes().items()}
        self.write()
        with ModelBundleReader(self.path) as model:
            self.assertEqual(model.manifest["aliases"], {})
            self.assertIn("lm_head.weight", model.manifest["tensors"])

    def test_preflight_does_not_read_weight_payloads(self):
        self.write()
        manifest = self.manifest()
        raw = self.path / manifest["tensors"]["model.norm.weight"]["path"]
        data = bytearray(raw.read_bytes())
        struct.pack_into("<f", data, 0, 99.0)
        raw.write_bytes(data)
        q4entry = manifest["tensors"]["model.embed_tokens.weight"]
        packed = self.path / q4entry["path"]
        data = bytearray(packed.read_bytes())
        data[-1] ^= 1
        packed.write_bytes(data)
        with mock.patch.object(NexaPackReader, "read_rows", side_effect=AssertionError("eager Q4")), \
             mock.patch.object(bundle, "_hash_file", side_effect=AssertionError("eager norm")):
            with ModelBundleReader(self.path) as model:
                self.assertEqual(model.inspect()["physical_tensors"], len(self.sources))
        with ModelBundleReader(self.path) as model:
            with self.assertRaisesRegex(ModelBundleError, "checksum"):
                model.read_f32("model.norm.weight")
            with model.open_q4("model.embed_tokens.weight") as reader:
                with self.assertRaisesRegex(NexaPackError, "checksum"):
                    reader.read_rows(15, 1)

    def test_raw_nonfinite_even_with_valid_checksum_is_rejected_lazily(self):
        self.write()
        manifest = self.manifest()
        entry = manifest["tensors"]["model.norm.weight"]
        raw = self.path / entry["path"]
        data = struct.pack("<f", float("nan")) + raw.read_bytes()[4:]
        raw.write_bytes(data)
        entry["sha256"] = hashlib.sha256(data).hexdigest()
        self.set_manifest(manifest)
        with ModelBundleReader(self.path) as model:
            with self.assertRaisesRegex(ModelBundleError, "Nonfinite"):
                model.read_f32("model.norm.weight")

    def test_source_names_must_exactly_match_physical_contract(self):
        for variant in (dict(list(self.sources.items())[1:]), {**self.sources, "lm_head.weight": lambda: []}):
            with self.subTest(names=list(variant)), self.assertRaises(ModelBundleError):
                write_model_bundle(self.path, self.config, variant)
            self.assertFalse(self.path.exists())
        self.no_staging()

    def test_bad_sources_leave_no_partial_bundle(self):
        bad_values = [lambda: [[float("nan")] * 8], lambda: [[1] * 7],
                      lambda: [[1] * 9], lambda: [[1] * 8, [1] * 8], lambda: []]
        for source in bad_values:
            with self.subTest(source=source):
                sources = {**self.sources, "model.norm.weight": source}
                with self.assertRaises((ModelBundleError, NexaPackError)):
                    write_model_bundle(self.path, self.config, sources)
                self.assertFalse(self.path.exists())
                self.no_staging()
        def failed():
            raise RuntimeError("source failed")
        with self.assertRaisesRegex(RuntimeError, "source failed"):
            write_model_bundle(self.path, self.config, {**self.sources, "model.norm.weight": failed})
        self.assertFalse(self.path.exists())
        self.no_staging()

    def test_existing_destination_and_publish_race_are_never_overwritten(self):
        self.path.mkdir()
        with self.assertRaises(FileExistsError):
            self.write()
        self.assertEqual(list(self.path.iterdir()), [])
        self.path.rmdir()
        publish = bundle._publish_directory
        def concurrent_destination(source, destination):
            destination.mkdir()
            publish(source, destination)
        with mock.patch.object(bundle, "_publish_directory", side_effect=concurrent_destination):
            with self.assertRaises(OSError):
                self.write()
        self.assertTrue(self.path.is_dir())
        self.assertEqual(list(self.path.iterdir()), [])
        self.no_staging()

    def test_writer_publication_failure_cleans_staging(self):
        with mock.patch.object(bundle, "_publish_directory", side_effect=OSError("publish error")):
            with self.assertRaisesRegex(OSError, "publish error"):
                self.write()
        self.assertFalse(self.path.exists())
        self.no_staging()

    def test_manifest_version_codec_shape_alias_and_duplicate_paths_rejected(self):
        self.write()
        original = self.manifest()
        mutations = [lambda m: m.update(format_version=True),
                     lambda m: m.update(architecture="other"),
                     lambda m: m["tensors"]["model.norm.weight"].update(codec="unknown"),
                     lambda m: m["tensors"]["model.norm.weight"].update(codec_version=True),
                     lambda m: m["tensors"]["model.norm.weight"].update(shape=[True]),
                     lambda m: m["tensors"]["model.norm.weight"].update(shape=[9]),
                     lambda m: m["aliases"].update({"lm_head.weight": "lm_head.weight"}),
                     lambda m: m["aliases"].update({"a": "b", "b": "a"}),
                     lambda m: m["tensors"]["model.norm.weight"].update(path=m["tensors"]["model.layers.0.input_layernorm.weight"]["path"]),
                     lambda m: m["provenance"].update(schema_version=True)]
        for change in mutations:
            metadata = json.loads(json.dumps(original))
            change(metadata)
            self.set_manifest(metadata)
            with self.subTest(change=change), self.assertRaises(ModelBundleError):
                ModelBundleReader(self.path)

    def test_traversal_absolute_windows_paths_and_symlinks_rejected(self):
        self.write()
        original = self.manifest()
        for invalid in ("../outside", "/etc/passwd", "C:/secret", "tensors\\data", "tensors//data", "tensors/./data"):
            metadata = json.loads(json.dumps(original))
            metadata["tensors"]["model.norm.weight"]["path"] = invalid
            self.set_manifest(metadata)
            with self.subTest(path=invalid), self.assertRaises(ModelBundleError):
                ModelBundleReader(self.path)
        self.set_manifest(original)
        entry = original["tensors"]["model.norm.weight"]
        vector = self.path / entry["path"]
        outside = self.directory / "outside.f32"
        vector.replace(outside)
        try:
            vector.symlink_to(outside)
        except OSError as error:
            self.skipTest(f"Symlinks unavailable: {error}")
        with self.assertRaisesRegex(ModelBundleError, "Symlinks"):
            ModelBundleReader(self.path)

    def test_q4_metadata_fingerprint_header_corruption_and_truncation(self):
        self.write()
        manifest = self.manifest()
        entry = manifest["tensors"]["model.embed_tokens.weight"]
        packed = self.path / entry["path"]
        original = packed.read_bytes()
        entry["metadata_sha256"] = "0" * 64
        self.set_manifest(manifest)
        with self.assertRaisesRegex(ModelBundleError, "metadata checksum"):
            ModelBundleReader(self.path)
        # Restore the record from the validated original metadata.
        with NexaPackReader(packed) as reader:
            entry["metadata_sha256"] = bundle._metadata_sha(reader)
        self.set_manifest(manifest)
        packed.write_bytes(b"badmagic" + original[8:])
        with self.assertRaises(NexaPackError):
            ModelBundleReader(self.path)
        packed.write_bytes(original[:-1])
        with self.assertRaisesRegex(ModelBundleError, "size"):
            ModelBundleReader(self.path)

    def test_partial_open_closes_all_q4_readers(self):
        self.write()
        metadata = self.manifest()
        metadata["tensors"]["model.layers.0.self_attn.v_proj.weight"]["metadata_sha256"] = "0" * 64
        self.set_manifest(metadata)
        readers = []
        class TrackedReader(NexaPackReader):
            def __init__(self, path):
                super().__init__(path)
                readers.append(self)
        with mock.patch.object(bundle, "NexaPackReader", TrackedReader):
            with self.assertRaises(ModelBundleError):
                ModelBundleReader(self.path)
        self.assertGreater(len(readers), 1)
        self.assertTrue(all(reader._stream.closed for reader in readers))

    def test_metadata_limits_duplicate_keys_and_layer_expansion_bound(self):
        self.write()
        manifest = self.path / "manifest.json"
        original = manifest.read_bytes()
        manifest.write_bytes(b'{"format":"a","format":"b"}')
        with self.assertRaisesRegex(ModelBundleError, "Duplicate"):
            ModelBundleReader(self.path)
        manifest.write_bytes(b" " * (bundle.MAX_MANIFEST_BYTES + 1))
        with self.assertRaisesRegex(ModelBundleError, "limit"):
            ModelBundleReader(self.path)
        manifest.write_bytes(original)
        metadata = self.manifest()
        metadata["config"]["num_hidden_layers"] = 10**12
        self.set_manifest(metadata)
        with self.assertRaisesRegex(ModelBundleError, "layer count"):
            ModelBundleReader(self.path)

    def test_tokenizer_integrity_limits_and_portable_names(self):
        asset = self.directory / "tokenizer.json"
        asset.write_bytes(b"{}")
        for name in ("../escape", "CON", "aux.json", "tokenizer."):
            with self.subTest(name=name), self.assertRaises(ModelBundleError):
                self.write(tokenizer_files={name: asset})
            self.no_staging()
        with mock.patch.object(bundle, "MAX_ASSET_BYTES", 1):
            with self.assertRaises(ModelBundleError):
                self.write(tokenizer_files={"tokenizer.json": asset})
        self.assertFalse(self.path.exists())
        self.write(tokenizer_files={"tokenizer.json": asset})
        (self.path / "assets/tokenizer.json").write_bytes(b"[]")
        with self.assertRaisesRegex(ModelBundleError, "Tokenizer checksum"):
            ModelBundleReader(self.path)

    def test_expected_asset_digest_is_checked_before_publish(self):
        asset = self.directory / "tokenizer.json"
        asset.write_bytes(b"{}")
        for checksums in ({}, {"tokenizer.json": "bad"}, {"tokenizer.json": "0" * 64}):
            with self.subTest(checksums=checksums), self.assertRaises(ModelBundleError):
                self.write(tokenizer_files={"tokenizer.json": asset}, asset_checksums=checksums)
            self.assertFalse(self.path.exists())
            self.no_staging()
        self.write(tokenizer_files={"tokenizer.json": asset},
                   asset_checksums={"tokenizer.json": hashlib.sha256(b"{}").hexdigest()})
        with ModelBundleReader(self.path) as model:
            self.assertEqual(model.inspect()["tokenizer_assets"], ["tokenizer.json"])

    def test_explicit_reader_lifetime_and_type_errors(self):
        self.write()
        model = ModelBundleReader(self.path)
        independent = model.open_q4("lm_head.weight")
        with self.assertRaises(ModelBundleError):
            model.open_q4("model.norm.weight")
        with self.assertRaises(ModelBundleError):
            model.read_f32("lm_head.weight")
        with self.assertRaises(ModelBundleError):
            model.open_q4("missing")
        model.close()
        with self.assertRaises(ModelBundleError):
            model.open_q4("lm_head.weight")
        with independent:
            self.assertTrue(independent.read_rows(0, 1))


if __name__ == "__main__":
    unittest.main()
