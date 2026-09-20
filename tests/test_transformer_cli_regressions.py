"""Offline token-ID execution and in-place norm loading contracts."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.importers.llama import import_llama_checkpoint
from model_checkpoint_fixture import create_checkpoint
from runtime.nexapack.bundle import ModelBundleReader


class TransformerCLIRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nexa-forward-cli-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.bundle = self.root / "source", self.root / "bundle"
        create_checkpoint(self.source)
        import_llama_checkpoint(self.source, self.bundle, group_size=4, block_rows=3)

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "tools/nexa_run.py"),
                               str(self.bundle), *map(str, args)], capture_output=True,
                              text=True, timeout=60)

    def test_raw_norm_into_buffer_matches_reader_and_rejects_bad_destination(self):
        import struct
        with ModelBundleReader(self.bundle) as bundle:
            target = bytearray(bundle.config.hidden_size * 4)
            self.assertEqual(bundle.read_f32_into("model.norm.weight", target), len(target))
            self.assertEqual([x[0] for x in struct.iter_unpack("<f", target)],
                             bundle.read_f32("model.norm.weight"))
            for bad in (bytes(target), bytearray(len(target) - 1), memoryview(target)[::2]):
                with self.assertRaises(ValueError):
                    bundle.read_f32_into("model.norm.weight", bad)
            with self.assertRaises(ValueError):
                bundle.read_f32_into("lm_head.weight", target)
            path = self.bundle / bundle.manifest["tensors"]["model.norm.weight"]["path"]
            content = bytearray(path.read_bytes())
            content[0] ^= 1
            path.write_bytes(content)
            with self.assertRaisesRegex(ValueError, "checksum"):
                bundle.read_f32_into("model.norm.weight", target)

    @unittest.skipUnless(shutil.which("clang") or shutil.which("cc"), "C compiler unavailable")
    def test_cli_prefill_decode_and_greedy_ids_reproduce(self):
        destination = self.root / "report.json"
        run = self.run_cli("--tokens", "1,3", "--decode-tokens", "5,7", "--include-logits",
                           "--memory-budget", "96KiB", "--tile-rows", 3, "--report", destination)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads(run.stdout)
        self.assertEqual(report, json.loads(destination.read_text()))
        self.assertEqual(report["token_ids"], [1, 3, 5, 7])
        self.assertEqual(len(report["logits"]), 4)
        self.assertEqual(len(report["steps"]), 3)
        # Paged KV is the default execution path; the baseline is opt-in now.
        self.assertTrue(report["persistent_kv_cache"])
        self.assertEqual(report["decode_strategy"], "paged_incremental_kv")
        self.assertFalse(report["tokenizer_executed"])
        direct = self.run_cli("--tokens", "1,3,5,7", "--include-logits", "--tile-rows", 2)
        self.assertEqual(direct.returncode, 0, direct.stderr)
        self.assertEqual(report["logits"], json.loads(direct.stdout)["logits"])
        recomputed = self.run_cli("--tokens", "1,3", "--decode-tokens", "5,7",
                                  "--include-logits", "--recompute", "--tile-rows", 3)
        self.assertEqual(recomputed.returncode, 0, recomputed.stderr)
        baseline = json.loads(recomputed.stdout)
        self.assertFalse(baseline["persistent_kv_cache"])
        # Both paths must agree: the cache is an optimization, not a variant.
        self.assertEqual(baseline["token_ids"], report["token_ids"])
        self.assertEqual(baseline["next_token_id"], report["next_token_id"])
        greedy = self.run_cli("--tokens", "1,3", "--generate", 2)
        self.assertEqual(greedy.returncode, 0, greedy.stderr)
        first = json.loads(greedy.stdout)
        replay = self.run_cli("--tokens", ",".join(map(str, first["token_ids"])))
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(first["logits_sha256"], json.loads(replay.stdout)["logits_sha256"])

    def test_cli_rejects_budget_context_bad_ids_and_source_overwrite(self):
        cases = [("--tokens", "1", "--memory-budget", "1KiB"),
                 ("--tokens", "1,3", "--generate", 8),
                 ("--tokens", "1,3", "--decode-tokens", "999"),
                 ("--tokens", "1,3", "--generate", -1),
                 ("--tokens", "1,3", "--report", self.bundle / "manifest.json"),
                 ("--tokens", "1,",)]
        original = (self.bundle / "manifest.json").read_bytes()
        for args in cases:
            with self.subTest(args=args):
                run = self.run_cli(*args)
                self.assertNotEqual(run.returncode, 0)
                self.assertNotIn("Traceback", run.stderr)
        self.assertEqual((self.bundle / "manifest.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
