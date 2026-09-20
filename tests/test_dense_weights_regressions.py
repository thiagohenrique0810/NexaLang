"""Dense F32 weight matrices: per-codec dispatch, verified blocks and limits."""
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.model_config import ModelConfig
from runtime.nexapack.bundle import ModelBundleError, ModelBundleReader, write_model_bundle
from runtime.nexapack.transformer import TransformerSession
from test_transformer_forward_regressions import f32, random_bundle


def exact_q4_value(index):
    """A multiple of 0.5 inside [-3.5, 3.5]: Q4 stores it without any error.

    A group whose largest magnitude is 3.5 gets scale 3.5/7 = 0.5 exactly, and
    every multiple of 0.5 in range maps to an integer code, so the dense and
    packed kernels multiply the very same float32 values.
    """
    return f32(((index % 15) - 7) * 0.5)


class _DenseFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nexa-dense-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    @staticmethod
    def config(**overrides):
        options = {"name": "dense_fixture", "vocab_size": 12, "hidden_size": 8,
                   "intermediate_size": 12, "num_hidden_layers": 2, "num_attention_heads": 4,
                   "num_key_value_heads": 2, "max_position_embeddings": 12,
                   "tie_word_embeddings": False, "rms_norm_eps": 1e-5, "rope_theta": 10000.0}
        options.update(overrides)
        return ModelConfig(**options)

    def exact_weights(self, config, group_size=4):
        """Weights Q4 stores without error, so only the codec path differs."""
        values = {}
        for name, shape in config.required_tensor_shapes().items():
            if len(shape) == 1:
                values[name] = [f32(1.0 + (index % 3) * 0.25) for index in range(shape[0])]
                continue
            rows = []
            for row in range(shape[0]):
                line = [exact_q4_value(row + column) for column in range(shape[1])]
                # Scales are per group, so every group needs its own 3.5.
                for start in range(0, shape[1], group_size):
                    line[start] = f32(3.5)
                rows.append(line)
            values[name] = rows
        return values

    def write(self, destination, config, values, *, codecs=None, block_rows=3, group_size=4):
        shapes = config.required_tensor_shapes()

        def source(name):
            rows = values[name]
            return (lambda: iter([rows])) if len(shapes[name]) == 1 else (lambda: iter(rows))

        write_model_bundle(destination, config, {name: source(name) for name in shapes},
                           group_size=group_size, block_rows=block_rows, tensor_codecs=codecs)
        return destination

    def dense_codecs(self, config, names=None):
        matrices = [name for name, shape in config.required_tensor_shapes().items() if len(shape) == 2]
        return {name: "f32" for name in (names if names is not None else matrices)}

    def run_model(self, path, tokens=(1, 3, 5), **options):
        with TransformerSession(path, memory_budget=options.pop("memory_budget", "8MiB"),
                                max_sequence_length=8, tile_rows=options.pop("tile_rows", 3)) as session:
            return session.prefill(list(tokens)), session.report()


class DenseWeightRegressions(_DenseFixture):
    def test_dense_and_packed_agree_when_quantization_is_exact(self):
        config = self.config()
        values = self.exact_weights(config)
        packed = self.write(self.directory / "packed", config, values)
        dense = self.write(self.directory / "dense", config, values, codecs=self.dense_codecs(config))
        packed_logits, packed_report = self.run_model(packed)
        dense_logits, dense_report = self.run_model(dense)
        # Identical values reach both kernels, which share the reduction order.
        self.assertEqual(dense_logits, packed_logits)
        self.assertEqual(dense_report["logits_sha256"], packed_report["logits_sha256"])
        self.assertGreater(dense_report["io"]["raw_payload_bytes_read"],
                           packed_report["io"]["raw_payload_bytes_read"])
        self.assertEqual(dense_report["io"]["q4_payload_bytes_read"], 0)

    def test_dense_execution_is_deterministic_and_differs_only_by_quantization(self):
        path = self.directory / "random"
        config, values = random_bundle(path, layers=2, heads=4, kv_heads=2, tied=False, seed=77)
        dense = self.write(self.directory / "random-dense", config, values,
                           codecs=self.dense_codecs(config))
        first, report = self.run_model(dense)
        second, _ = self.run_model(dense)
        packed, _ = self.run_model(path)
        self.assertEqual(first, second)
        self.assertNotEqual(first, packed)
        scale = max(abs(value) for row in first for value in row)
        error = max(abs(a - b) for left, right in zip(first, packed) for a, b in zip(left, right))
        self.assertLess(error, scale)  # Quantization error, not a broken path.
        self.assertEqual(report["memory"]["budget_bytes"], 8 * 1024 * 1024)

    def test_a_mixed_bundle_dispatches_per_tensor(self):
        config = self.config()
        values = self.exact_weights(config)
        name = "model.layers.0.mlp.down_proj.weight"
        mixed = self.write(self.directory / "mixed", config, values, codecs={name: "f32"})
        with ModelBundleReader(mixed) as bundle:
            codecs = {item["name"]: item["codec"] for item in bundle.inspect()["tensors"]}
        self.assertEqual(codecs[name], "RAW_F32_MATRIX")
        self.assertEqual(codecs["model.layers.0.mlp.up_proj.weight"], "Q4_GROUPED")
        self.assertEqual(codecs["model.norm.weight"], "RAW_F32")
        logits, report = self.run_model(mixed)
        packed_logits, _ = self.run_model(self.write(self.directory / "all-packed", config, values))
        self.assertEqual(logits, packed_logits)  # Exact weights: the codec cannot change the result.
        self.assertGreater(report["io"]["q4_payload_bytes_read"], 0)
        self.assertGreater(report["io"]["raw_payload_bytes_read"], 0)

    def test_embedding_reads_only_the_blocks_holding_its_tokens(self):
        config = self.config(vocab_size=12)
        values = self.exact_weights(config)
        dense = self.write(self.directory / "embedding", config, values,
                           codecs={"model.embed_tokens.weight": "f32"}, block_rows=2)
        _, narrow = self.run_model(dense, tokens=(0, 1))
        _, wide = self.run_model(dense, tokens=(0, 5, 11))
        self.assertLess(narrow["io"]["raw_payload_bytes_read"], wide["io"]["raw_payload_bytes_read"])
        self.assertEqual(narrow["io"]["embedding_rows_read"], 2)
        self.assertEqual(wide["io"]["embedding_rows_read"], 3)

    def test_block_checksums_reject_corruption_and_truncation(self):
        config = self.config()
        values = self.exact_weights(config)
        source = self.write(self.directory / "verified", config, values, codecs=self.dense_codecs(config))
        with ModelBundleReader(source) as bundle:
            entry = bundle.manifest["tensors"]["model.layers.0.mlp.down_proj.weight"]
            relative = entry["path"]
        for index, mutate in enumerate((lambda data: data[:-4],
                                        lambda data: bytes([data[0] ^ 0xFF]) + data[1:],
                                        lambda data: data + b"\x00\x00\x00\x00")):
            with self.subTest(case=index):
                damaged = self.directory / f"damaged-{index}"
                shutil.copytree(source, damaged)
                path = damaged / relative
                path.write_bytes(mutate(path.read_bytes()))
                with self.assertRaises((ModelBundleError, ValueError)):
                    self.run_model(damaged)

    def test_manifest_blocks_must_tile_every_row_once(self):
        config = self.config()
        values = self.exact_weights(config)
        source = self.write(self.directory / "blocks", config, values, codecs=self.dense_codecs(config))
        name = "model.layers.0.mlp.down_proj.weight"

        def mutation(change):
            def mutate(manifest):
                blocks = manifest["tensors"][name]["blocks"]
                change(blocks)
            return mutate

        cases = [mutation(lambda blocks: blocks.pop()),
                 mutation(lambda blocks: blocks.insert(0, dict(blocks[0]))),
                 mutation(lambda blocks: blocks[0].__setitem__("start_row", 1)),
                 mutation(lambda blocks: blocks[0].__setitem__("row_count", 0)),
                 mutation(lambda blocks: blocks[0].__setitem__("sha256", "zz" * 32))]
        for index, mutate in enumerate(cases):
            with self.subTest(case=index):
                damaged = self.directory / f"blocks-{index}"
                shutil.copytree(source, damaged)
                manifest = json.loads((damaged / "manifest.json").read_text())
                mutate(manifest)
                (damaged / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(ModelBundleError):
                    ModelBundleReader(damaged).close()

    def test_a_block_beyond_the_reader_limit_is_rejected_with_a_clear_message(self):
        config = self.config(hidden_size=8, intermediate_size=12)
        values = self.exact_weights(config)
        # One block holding every row of a wide matrix exceeds the read bound.
        dense = self.write(self.directory / "huge-block", config, values,
                           codecs=self.dense_codecs(config), block_rows=1 << 20)
        with ModelBundleReader(dense) as bundle:
            blocks = bundle.matrix_blocks("model.layers.0.mlp.down_proj.weight")
        self.assertEqual(len(blocks), 1)
        from unittest.mock import patch
        with patch("runtime.nexapack.transformer.MAX_READ_BYTES", 16):
            with self.assertRaises(ValueError) as failure:
                TransformerSession(dense, memory_budget="8MiB", max_sequence_length=8, tile_rows=3)
        self.assertIn("block", str(failure.exception))

    def test_writer_rejects_invalid_codec_maps(self):
        config = self.config()
        values = self.exact_weights(config)
        for codecs in ({"model.norm.weight": "q4"}, {"missing.tensor": "f32"},
                       {"model.embed_tokens.weight": "q16"}, {"model.embed_tokens.weight": None}):
            with self.subTest(codecs=sorted(codecs)), self.assertRaises(ModelBundleError):
                self.write(self.directory / f"invalid-{abs(hash(str(codecs)))}", config, values, codecs=codecs)

    def test_reader_rejects_block_access_on_a_packed_tensor(self):
        config = self.config()
        packed = self.write(self.directory / "packed-only", config, self.exact_weights(config))
        with ModelBundleReader(packed) as bundle:
            with self.assertRaises(ModelBundleError):
                bundle.matrix_blocks("model.layers.0.mlp.down_proj.weight")
            dense = self.write(self.directory / "dense-only", config, self.exact_weights(config),
                               codecs=self.dense_codecs(config))
            with ModelBundleReader(dense) as other:
                name = "model.layers.0.mlp.down_proj.weight"
                block = other.matrix_blocks(name)[0]
                buffer = bytearray(block["bytes"])
                self.assertEqual(other.read_matrix_block_into(name, 0, buffer), block["bytes"])
                for index in (-1, len(other.matrix_blocks(name)), True, 1.0):
                    with self.assertRaises(ModelBundleError):
                        other.read_matrix_block_into(name, index, buffer)
                with self.assertRaises(ModelBundleError):
                    other.read_matrix_block_into(name, 0, bytearray(block["bytes"] - 1))


class DenseWeightCLIRegressions(_DenseFixture):
    def setUp(self):
        super().setUp()
        from model_checkpoint_fixture import create_checkpoint
        self.checkpoint = self.directory / "checkpoint"
        create_checkpoint(self.checkpoint)

    def run_tool(self, tool, *arguments, expect=0):
        import subprocess
        result = subprocess.run([sys.executable, str(ROOT / "tools" / tool), *arguments],
                                capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, expect, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        return json.loads(result.stdout) if expect == 0 else result.stderr

    def test_convert_inspect_and_run_a_dense_bundle(self):
        bundle = self.directory / "dense-bundle"
        converted = self.run_tool("nexa_convert.py", "--checkpoint", str(self.checkpoint),
                                  "--out", str(bundle), "--dense-all", "--block-rows", "3")
        self.assertTrue(converted["dense_tensors"])
        inspected = self.run_tool("nexa_inspect.py", str(bundle), "--verify")
        codecs = {item["codec"] for item in inspected["tensors"]}
        self.assertEqual(codecs, {"RAW_F32_MATRIX", "RAW_F32"})
        self.assertGreater(inspected["validation"]["dense_payload_bytes_read"], 0)
        self.assertEqual(inspected["validation"]["q4_payload_bytes_read"], 0)
        report = self.run_tool("nexa_run.py", str(bundle), "--tokens", "1,3", "--decode-tokens", "5",
                               "--tile-rows", "3", "--memory-budget", "4MiB")
        self.assertEqual(report["token_ids"], [1, 3, 5])
        self.assertEqual(report["io"]["q4_payload_bytes_read"], 0)
        self.assertGreater(report["io"]["raw_payload_bytes_read"], 0)

    def test_one_dense_tensor_keeps_the_rest_packed(self):
        bundle = self.directory / "mixed-bundle"
        name = "model.layers.0.mlp.down_proj.weight"
        self.run_tool("nexa_convert.py", "--checkpoint", str(self.checkpoint), "--out", str(bundle),
                      "--dense-tensor", name, "--block-rows", "3")
        inspected = self.run_tool("nexa_inspect.py", str(bundle), "--verify")
        codecs = {item["name"]: item["codec"] for item in inspected["tensors"]}
        self.assertEqual(codecs[name], "RAW_F32_MATRIX")
        self.assertEqual(codecs["model.layers.0.mlp.up_proj.weight"], "Q4_GROUPED")
        self.assertGreater(inspected["validation"]["q4_payload_bytes_read"], 0)
        self.assertGreater(inspected["validation"]["dense_payload_bytes_read"], 0)
        report = self.run_tool("nexa_run.py", str(bundle), "--tokens", "1,3",
                               "--tile-rows", "3", "--memory-budget", "4MiB")
        self.assertGreater(report["io"]["q4_payload_bytes_read"], 0)

    def test_dense_options_are_rejected_without_a_checkpoint_or_a_known_tensor(self):
        for arguments in (("--dense-all", "--out", str(self.directory / "a"), "--input",
                           str(self.checkpoint / "config.json"), "--rows", "2", "--cols", "2"),
                          ("--checkpoint", str(self.checkpoint), "--out", str(self.directory / "b"),
                           "--dense-tensor", "model.missing.weight"),
                          ("--checkpoint", str(self.checkpoint), "--out", str(self.directory / "c"),
                           "--dense-all", "--dense-tensor", "model.norm.weight")):
            with self.subTest(arguments=arguments[0]):
                message = self.run_tool("nexa_convert.py", *arguments, expect=2)
                self.assertTrue(message.strip())


if __name__ == "__main__":
    unittest.main()
