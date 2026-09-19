"""Local architecture -> Safetensors -> bundle -> native Q4 integration."""
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.importers.llama import import_llama_checkpoint
from compiler.planner.memory import MemoryBudgetError
from model_checkpoint_fixture import create_checkpoint
from runtime.nexapack.bundle import ModelBundleReader
from runtime.nexapack.executor import run_packed_matmul
from runtime.nexapack.format import HEADER, NexaPackReader, write_q4_matrix
from tools.nexa_inspect import inspect_artifact
from tools.nexa_model import compile_definitions


class ModelPipelineRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nexa-model-pipeline-")
        self.root = Path(self.temp.name)
        self.source = self.root / "checkpoint"
        self.bundle = self.root / "bundle"

    def tearDown(self):
        self.temp.cleanup()

    def import_fixture(self, **kwargs):
        create_checkpoint(self.source, **kwargs)
        return import_llama_checkpoint(self.source, self.bundle, group_size=4, block_rows=3)

    def cli(self, tool, *args):
        return subprocess.run([sys.executable, str(ROOT / "tools" / tool), *map(str, args)],
                              capture_output=True, text=True, timeout=60)

    def test_canonical_definitions_compile_to_exact_physical_counts(self):
        result = compile_definitions(ROOT / "models/nexalm512/architecture.nxl")
        counts = sorted(model["parameter_count"] for model in result["models"].values())
        self.assertEqual(counts, [125854464, 394331136])
        for model in result["models"].values():
            self.assertEqual(model["aliases"]["lm_head.weight"], "model.embed_tokens.weight")
            self.assertNotIn("lm_head.weight", model["tensor_shapes"])
        output = self.root / "definitions.json"
        run = self.cli("nexa_model.py", ROOT / "models/nexalm512/architecture.nxl", "--out", output)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(output.read_text()), result)

    def test_import_shards_preserves_norms_aliases_and_tokenizer(self):
        self.import_fixture(include_tied_head=True)
        with ModelBundleReader(self.bundle) as bundle:
            manifest = bundle.manifest
            self.assertEqual(len(manifest["tensors"]), len(bundle.config.required_tensor_shapes()))
            self.assertNotIn("lm_head.weight", manifest["tensors"])
            self.assertEqual(manifest["aliases"]["lm_head.weight"], "model.embed_tokens.weight")
            self.assertIn("tokenizer.json", manifest["assets"])
            self.assertEqual(bundle.read_f32("model.norm.weight"),
                             [1.0 + ((j % 3) - 1) / 32.0 for j in range(8)])
            with bundle.open_q4("model.embed_tokens.weight") as embed, bundle.open_q4("lm_head.weight") as head:
                self.assertEqual(embed.read_rows(0, 3), head.read_rows(0, 3))
            record = manifest["tensors"]["model.layers.0.self_attn.q_proj.weight"]
            with NexaPackReader(self.bundle / record["path"]) as legacy:
                self.assertEqual((legacy.rows, legacy.cols), (8, 8))

    def test_inspection_is_lazy_and_explicit_verification_detects_corruption(self):
        self.import_fixture()
        with patch.object(NexaPackReader, "read_rows_into", side_effect=AssertionError("unexpected payload read")):
            with patch.object(ModelBundleReader, "read_f32", side_effect=AssertionError("unexpected norm read")):
                result = inspect_artifact(self.bundle)
        self.assertFalse(result["validation"]["payloads_verified"])
        verified = inspect_artifact(self.bundle, verify=True)
        self.assertTrue(verified["validation"]["payloads_verified"])
        self.assertGreater(verified["validation"]["q4_payload_bytes_read"], 0)
        with ModelBundleReader(self.bundle) as bundle:
            record = bundle.manifest["tensors"]["model.embed_tokens.weight"]
            filename = self.bundle / record["path"]
        with NexaPackReader(filename) as reader:
            offset = reader.metadata["blocks"][0]["offset"]
        with filename.open("r+b") as stream:
            stream.seek(offset)
            value = stream.read(1)[0]
            stream.seek(offset)
            stream.write(bytes([value ^ 1]))
        self.assertFalse(inspect_artifact(self.bundle)["validation"]["payloads_verified"])
        with self.assertRaises(ValueError):
            inspect_artifact(self.bundle, verify=True)

    def test_bundle_benchmark_reuses_v1_kernel_and_tied_alias(self):
        if not (shutil.which("clang") or shutil.which("cc")):
            self.skipTest("C compiler unavailable")
        self.import_fixture()
        results = []
        for name in ("model.embed_tokens.weight", "lm_head.weight"):
            results.append(run_packed_matmul(self.bundle, tensor_name=name, batch=2, tile_rows=3,
                                            memory_budget="96KiB", verify=True))
        self.assertEqual(results[0]["output_sha256_by_batch"], results[1]["output_sha256_by_batch"])
        self.assertEqual(results[0]["source"]["kind"], "model_bundle")
        self.assertTrue(results[0]["validation"]["verified"])
        with patch.object(NexaPackReader, "read_rows_into", side_effect=AssertionError("weights loaded")):
            with self.assertRaises(MemoryBudgetError):
                run_packed_matmul(self.bundle, tensor_name="lm_head.weight", memory_budget="1KiB")

    def test_inspection_rejects_invalid_q4_even_when_all_checksums_match(self):
        def rewrite_payload(path, change):
            raw = bytearray(path.read_bytes())
            header = list(HEADER.unpack(raw[:HEADER.size]))
            metadata = json.loads(raw[HEADER.size:HEADER.size + header[3]])
            change(raw, header[4])
            for block in metadata["blocks"]:
                begin = block["offset"]
                block["sha256"] = hashlib.sha256(raw[begin:begin + block["size"]]).hexdigest()
            encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
            header[3] = len(encoded)
            header[6] = hashlib.sha256(encoded).digest()
            path.write_bytes(HEADER.pack(*header) + encoded +
                             bytes(header[4] - HEADER.size - len(encoded)) + raw[header[4]:])
            return hashlib.sha256(encoded).hexdigest()

        matrix = self.root / "invalid.nxp"
        write_q4_matrix(matrix, 1, 2, 3, [[1.0, 2.0]])
        original = matrix.read_bytes()
        changes = [lambda raw, offset: struct.pack_into("<f", raw, offset, float("nan")),
                   lambda raw, offset: struct.pack_into("<f", raw, offset, -1),
                   lambda raw, offset: raw.__setitem__(offset + 4, 8),
                   lambda raw, offset: raw.__setitem__(offset + 5, 1),
                   lambda raw, offset: raw.__setitem__(offset + 5, 16),
                   lambda raw, offset: struct.pack_into("<f", raw, offset, 0)]
        for change in changes:
            matrix.write_bytes(original)
            rewrite_payload(matrix, change)
            self.assertFalse(inspect_artifact(matrix)["validation"]["q4_codec_validated"])
            with self.assertRaises(ValueError):
                inspect_artifact(matrix, verify=True)
        matrix.write_bytes(original)
        verified = inspect_artifact(matrix, verify=True)["validation"]
        self.assertTrue(verified["checksums_verified"])
        self.assertTrue(verified["q4_codec_validated"])

        self.import_fixture()
        manifest_path = self.bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        entry = manifest["tensors"]["model.embed_tokens.weight"]
        entry["metadata_sha256"] = rewrite_payload(self.bundle / entry["path"], changes[0])
        manifest_path.write_text(json.dumps(manifest))
        self.assertFalse(inspect_artifact(self.bundle)["validation"]["payloads_verified"])
        with self.assertRaisesRegex(ValueError, "scale must be finite"):
            inspect_artifact(self.bundle, verify=True)
        cli = self.cli("nexa_inspect.py", self.bundle, "--verify")
        self.assertEqual(cli.returncode, 1)
        self.assertIn("scale must be finite", cli.stderr)

    def test_cli_import_inspect_and_benchmark(self):
        create_checkpoint(self.source, sharded=False)
        imported = self.cli("nexa_convert.py", "--checkpoint", self.source, "--out", self.bundle,
                            "--group-size", 4, "--block-rows", 3)
        self.assertEqual(imported.returncode, 0, imported.stderr)
        self.assertIsInstance(json.loads(imported.stdout), dict)
        inspected = self.cli("nexa_inspect.py", self.bundle, "--verify")
        self.assertEqual(inspected.returncode, 0, inspected.stderr)
        self.assertTrue(json.loads(inspected.stdout)["validation"]["payloads_verified"])
        if shutil.which("clang") or shutil.which("cc"):
            report = self.root / "bench.json"
            bench = self.cli("nexa_bench.py", "--bundle", self.bundle, "--tensor", "lm_head.weight",
                             "--memory-budget", "96KiB", "--verify", "--report", report)
            self.assertEqual(bench.returncode, 0, bench.stderr)
            self.assertTrue(json.loads(report.read_text())["validation"]["verified"])

    def test_reports_cannot_replace_bundle_contents(self):
        self.import_fixture()
        target = self.bundle / "manifest.json"
        original = target.read_bytes()
        run = self.cli("nexa_bench.py", "--bundle", self.bundle, "--tensor", "lm_head.weight", "--report", target)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("outside", run.stderr)
        self.assertEqual(target.read_bytes(), original)

    def test_failed_reimport_leaves_published_bundle_unchanged(self):
        self.import_fixture()
        original = (self.bundle / "manifest.json").read_bytes()
        with self.assertRaises((OSError, ValueError)):
            import_llama_checkpoint(self.source, self.bundle)
        self.assertEqual((self.bundle / "manifest.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
