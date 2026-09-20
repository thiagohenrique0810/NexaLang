"""Synthetic checkpoints: determinism, declared shapes, init shape, importability.

Every model here is deliberately tiny. What a synthetic checkpoint is for --
measuring a 125M-parameter pipeline -- belongs in docs/NEXALM_MODELOS_SINTETICOS.md,
not in a suite that has to stay fast.
"""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import tracemalloc
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compiler.importers.llama import import_llama_checkpoint
from compiler.importers.safetensors import SafeTensorCheckpoint, read_json
from compiler.model_config import ModelConfig
from runtime.nexapack.bundle import ModelBundleReader
from tools.nexa_synth import (SyntheticCheckpointError, plan_shards, synthesize_checkpoint,
                              tensor_std)

SMALL = dict(name="SynthSmall", vocab_size=96, hidden_size=32, intermediate_size=64,
             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
             max_position_embeddings=32)
# Only for the streaming bound: a large embedding table and nothing else, so
# the payload dwarfs any per-chunk scratch without slowing the suite down.
STREAM = dict(name="SynthStream", vocab_size=16384, hidden_size=64, intermediate_size=128,
              num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
              max_position_embeddings=32)
# Wide enough that a standard deviation and a tail fraction are meaningful, and
# still under five megabytes so the suite stays fast.
WIDE = dict(name="SynthWide", vocab_size=4096, hidden_size=64, intermediate_size=128,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=32)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree_digests(directory):
    return {path.name: _digest(path) for path in sorted(Path(directory).iterdir())
            if path.name != "origin.json"}


class SyntheticCheckpointRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nexa-synth-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make(self, name, *, shape=None, seed=11, **kwargs):
        config = ModelConfig(**(shape or SMALL))
        return config, synthesize_checkpoint(self.root / name, config, seed=seed, **kwargs)

    def read_values(self, directory, tensor):
        checkpoint = SafeTensorCheckpoint(directory)
        return [value for row in checkpoint.iter_rows(tensor) for value in row]

    def cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "tools" / "nexa_synth.py"), *map(str, args)],
                              capture_output=True, text=True, timeout=120)

    def test_same_seed_produces_identical_bytes(self):
        _, first = self.make("first", seed=4242)
        _, second = self.make("second", seed=4242)
        self.assertEqual(_tree_digests(self.root / "first"), _tree_digests(self.root / "second"))
        self.assertEqual([entry["sha256"] for entry in first["files"]],
                         [entry["sha256"] for entry in second["files"]])
        # The declared hashes must be the hashes of the files actually written.
        on_disk = {path.name: _digest(path) for path in (self.root / "first").iterdir()}
        for entry in first["files"]:
            self.assertEqual(entry["sha256"], on_disk[entry["path"]], entry["path"])

    def test_a_different_seed_changes_every_sampled_tensor(self):
        self.make("a", seed=1)
        self.make("b", seed=2)
        left, right = _tree_digests(self.root / "a"), _tree_digests(self.root / "b")
        self.assertEqual(left["config.json"], right["config.json"])
        self.assertNotEqual(left["model.safetensors"], right["model.safetensors"])

    def test_shard_layout_never_splits_a_tensor_and_matches_its_index(self):
        config, result = self.make("sharded", max_shard_bytes=4096)
        self.assertGreater(result["shard_count"], 1)
        index = read_json(self.root / "sharded" / "model.safetensors.index.json")
        shapes = config.required_tensor_shapes()
        self.assertEqual(set(index["weight_map"]), set(shapes))
        self.assertEqual(index["metadata"]["total_size"],
                         sum(math.prod(shape) for shape in shapes.values()) * 4)
        placed = {}
        for name, filename in index["weight_map"].items():
            placed.setdefault(filename, set()).add(name)
        self.assertEqual(len(placed), result["shard_count"])
        for filename, names in placed.items():
            payload = sum(math.prod(shapes[name]) for name in names) * 4
            self.assertLessEqual(payload, max(4096, max(math.prod(shapes[name]) * 4 for name in names)))

    def test_shapes_and_scalar_count_match_the_declared_architecture(self):
        config, result = self.make("shapes")
        checkpoint = SafeTensorCheckpoint(self.root / "shapes")
        self.assertEqual(checkpoint.tensor_shapes, config.required_tensor_shapes())
        self.assertEqual(set(checkpoint.tensor_dtypes.values()), {"F32"})
        self.assertEqual(result["stored_scalar_count"], config.parameter_count())
        self.assertEqual(result["parameter_count"], config.parameter_count())

    def test_a_stored_tied_head_adds_storage_without_adding_parameters(self):
        config, result = self.make("tied", include_tied_head=True)
        self.assertTrue(result["tied_head_stored"])
        self.assertEqual(result["parameter_count"], config.parameter_count())
        self.assertEqual(result["stored_scalar_count"],
                         config.parameter_count() + config.vocab_size * config.hidden_size)
        # The importer decodes both tensors and rejects any difference, so this
        # passing is the proof the alias shares the embedding's exact stream.
        report = import_llama_checkpoint(self.root / "tied", self.root / "tied.bundle",
                                         group_size=32, block_rows=16)
        self.assertEqual(report["tied_weights_verified"], ["lm_head.weight"])

    def test_the_checkpoint_imports_into_a_bundle(self):
        config, _ = self.make("importable", max_shard_bytes=4096)
        report = import_llama_checkpoint(self.root / "importable", self.root / "bundle",
                                         group_size=32, block_rows=16)
        self.assertEqual(report["parameter_count"], config.parameter_count())
        with ModelBundleReader(self.root / "bundle") as bundle:
            self.assertEqual(ModelConfig.from_dict(bundle.manifest["config"]), config)
            self.assertEqual(set(bundle.manifest["tensors"]), set(config.required_tensor_shapes()))

    def test_origin_declares_the_checkpoint_synthetic_and_reproducible(self):
        _, result = self.make("origin", seed=99)
        origin = read_json(self.root / "origin" / "origin.json")
        self.assertTrue(origin["synthetic"])
        self.assertFalse(origin["trained"])
        self.assertFalse(origin["quality_measured"])
        self.assertEqual(origin["seed"], 99)
        self.assertEqual(origin["tool"], "tools/nexa_synth.py")
        self.assertEqual(origin["model_name"], SMALL["name"])
        self.assertEqual(origin["config"], ModelConfig(**SMALL).to_dict())
        self.assertEqual(origin["files"], result["files"])
        self.assertIn("do not transfer to a trained model", origin["warning"])

    def test_norm_gains_are_exactly_one_and_projections_scale_by_fan_in(self):
        config, _ = self.make("init", shape=WIDE)
        for name in ("model.norm.weight", "model.layers.0.input_layernorm.weight"):
            self.assertEqual(set(self.read_values(self.root / "init", name)), {1.0})
        expectations = {
            "model.embed_tokens.weight": 0.02,
            "model.layers.0.self_attn.q_proj.weight": 1.0 / math.sqrt(WIDE["hidden_size"]),
            "model.layers.0.mlp.down_proj.weight":
                1.0 / math.sqrt(WIDE["intermediate_size"] * 2.0 * WIDE["num_hidden_layers"]),
            "model.layers.0.self_attn.o_proj.weight":
                1.0 / math.sqrt(WIDE["hidden_size"] * 2.0 * WIDE["num_hidden_layers"]),
        }
        for name, expected in expectations.items():
            self.assertAlmostEqual(tensor_std(config, name), expected, places=12, msg=name)
            values = self.read_values(self.root / "init", name)
            measured = math.sqrt(sum(value * value for value in values) / len(values))
            self.assertAlmostEqual(measured / expected, 1.0, delta=0.12, msg=name)

    def test_the_sampled_distribution_has_normal_tails_not_uniform_ones(self):
        """A uniform fill of the same width has no |v| > 2 sigma at all.

        The codecs quantize by max|v| inside a group, so the tail is the part
        of the distribution that decides the reported quantization error.
        """
        config, _ = self.make("tails", shape=WIDE)
        name = "model.layers.0.mlp.gate_proj.weight"
        std = tensor_std(config, name)
        values = self.read_values(self.root / "tails", name)
        tail = sum(abs(value) > 2 * std for value in values) / len(values)
        self.assertGreater(tail, 0.02)
        self.assertLess(tail, 0.08)
        self.assertLessEqual(max(abs(value) for value in values), 6 * std)

    def test_generation_streams_instead_of_materializing_the_model(self):
        payload = sum(math.prod(shape) for shape in
                      ModelConfig(**STREAM).required_tensor_shapes().values()) * 4
        self.assertGreater(payload, 4 * 1024 * 1024)
        tracemalloc.start()
        try:
            self.make("streamed", shape=STREAM)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, payload // 4)

    def test_a_synthetic_checkpoint_never_overwrites_an_existing_directory(self):
        self.make("once")
        with self.assertRaises(SyntheticCheckpointError):
            self.make("once")

    def test_shard_and_seed_arguments_are_validated(self):
        config = ModelConfig(**SMALL)
        with self.assertRaises(SyntheticCheckpointError):
            synthesize_checkpoint(self.root / "bad-seed", config, seed=-1)
        with self.assertRaises(SyntheticCheckpointError):
            synthesize_checkpoint(self.root / "bad-seed", config, seed=True)
        with self.assertRaises(SyntheticCheckpointError):
            plan_shards(config, max_shard_bytes=64)
        with self.assertRaises(SyntheticCheckpointError):
            plan_shards(config, dtype="bf16")
        with self.assertRaises(SyntheticCheckpointError):
            plan_shards(ModelConfig(**{**SMALL, "tie_word_embeddings": False}), include_tied_head=True)

    def test_f16_storage_halves_the_payload_and_still_imports(self):
        config, result = self.make("half", dtype="f16")
        self.assertEqual(result["stored_dtype"], "F16")
        checkpoint = SafeTensorCheckpoint(self.root / "half")
        self.assertEqual(set(checkpoint.tensor_dtypes.values()), {"F16"})
        entry, = [item for item in result["files"] if item["path"] == "model.safetensors"]
        self.assertEqual(entry["payload_bytes"], config.parameter_count() * 2)
        import_llama_checkpoint(self.root / "half", self.root / "half.bundle",
                                group_size=32, block_rows=16)

    def test_cli_reports_the_declared_architecture_without_writing_weights(self):
        target = self.root / "never-written"
        run = self.cli("--definition", ROOT / "models/nexalm512/architecture.nxl",
                       "--model", "NexaLM512_R0", "--seed", 1, "--dry-run", "--out", target)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads(run.stdout)
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["parameter_count"], 125854464)
        self.assertEqual(report["stored_scalar_count"], 125854464)
        self.assertEqual(report["payload_bytes"], 125854464 * 4)
        self.assertFalse(target.exists())

    def test_cli_writes_a_checkpoint_the_importer_accepts(self):
        target = self.root / "from-cli"
        run = self.cli("--name", "SynthCli", "--vocab-size", 96, "--hidden-size", 32,
                       "--layers", 2, "--query-heads", 4, "--kv-heads", 2,
                       "--ffn-hidden", 64, "--context", 32, "--seed", 11, "--out", target)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["seed"], 11)
        self.make("from-api")
        self.assertEqual({name: digest for name, digest in _tree_digests(target).items()
                          if name != "config.json"},
                         {name: digest for name, digest in _tree_digests(self.root / "from-api").items()
                          if name != "config.json"})
        import_llama_checkpoint(target, self.root / "cli.bundle", group_size=32, block_rows=16)

    def test_cli_rejects_mixing_a_definition_with_command_line_shapes(self):
        run = self.cli("--definition", ROOT / "models/nexalm512/architecture.nxl",
                       "--model", "NexaLM512_R0", "--hidden-size", 8,
                       "--seed", 1, "--out", self.root / "mixed")
        self.assertEqual(run.returncode, 2)
        self.assertFalse((self.root / "mixed").exists())

    def test_cli_requires_a_named_model_when_the_definition_declares_several(self):
        run = self.cli("--definition", ROOT / "models/nexalm512/architecture.nxl",
                       "--seed", 1, "--dry-run", "--out", self.root / "ambiguous")
        self.assertEqual(run.returncode, 1)
        self.assertIn("NexaLM512_R0", run.stderr)


if __name__ == "__main__":
    unittest.main()
