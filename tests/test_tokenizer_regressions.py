"""NexaTokenizer: determinism, byte-exact round-trip, asset integrity and limits."""
import json
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.tokenizer_trainer import build_tokenizer, train_byte_bpe
from runtime.nexapack.tokenizer import (
    DEFAULT_SPECIAL_TOKENS, MAX_VOCAB_SIZE, NexaTokenizer, segment, write_tokenizer,
)

CORPUS = [
    "O NexaLang compila modelos grandes para rodar em 512 MB de memória.",
    "A quantização Q4 reduz a memória sem recomputar o prefixo inteiro.",
    "The runtime streams packed weights and keeps the KV cache paged.",
    "Attention reads the cache directly, without expanding a full prefix.",
    "def forward(x):\n    y = norm(x)\n    return attention(y) + x\n",
    "for index in range(32):\n    total += weights[index] * scale\n",
    "Números: 1234, 56.78 e 90% em uma linha com pontuação!",
]
SAMPLES = {
    "pt": "O NexaLang executa modelos com memória limitada e atenção paginada.",
    "en": "The paged attention kernel reads packed pages without expanding them.",
    "code": "def soma(a, b):\n    return a + b\n",
    "unicode": "acentuação, emoji 🙂, símbolos ±≈∞ e CJK 漢字",
}


class _TokenizerFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nexa-tokenizer-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def build(self, name="tokenizer", documents=None, vocab_size=400, **options):
        destination = self.directory / name
        manifest = build_tokenizer(destination, documents or CORPUS * 3, vocab_size, **options)
        return NexaTokenizer.load(destination), manifest, destination

    def corpus_directory(self, name="corpus"):
        path = self.directory / name
        path.mkdir()
        for index, document in enumerate(CORPUS * 3):
            (path / f"doc-{index:03d}.txt").write_text(document, encoding="utf-8")
        return path


class TokenizerTrainingRegressions(_TokenizerFixture):
    def test_training_is_deterministic_and_independent_of_document_order(self):
        documents = CORPUS * 3
        shuffled = list(documents)
        random.Random(17).shuffle(shuffled)
        first = train_byte_bpe(documents, 420)
        second = train_byte_bpe(shuffled, 420)
        self.assertEqual(first[0], second[0])  # Same vocabulary, in the same ids.
        self.assertEqual(first[1], second[1])  # Same merge order.
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[3], second[3])  # Corpus identity ignores order.
        self.assertEqual(first[3]["documents"], len(documents))
        self.assertTrue(first[3]["reached_target"])

    def test_every_byte_and_special_token_has_a_stable_id(self):
        tokenizer, manifest, _ = self.build()
        for value in range(256):
            self.assertEqual(tokenizer.token_bytes(value), bytes((value,)))
        for index, name in enumerate(DEFAULT_SPECIAL_TOKENS):
            self.assertEqual(tokenizer.special_id(name), 256 + index)
            self.assertEqual(manifest["special_tokens"][name], 256 + index)
        self.assertEqual(manifest["vocab_size"], tokenizer.vocab_size)
        self.assertGreater(tokenizer.vocab_size, 256 + len(DEFAULT_SPECIAL_TOKENS))

    def test_a_smaller_corpus_stops_early_and_reports_it(self):
        tokenizer, manifest, _ = self.build("small", documents=["abc abc"], vocab_size=MAX_VOCAB_SIZE // 1024)
        self.assertFalse(manifest["training"]["reached_target"])
        self.assertLess(tokenizer.vocab_size, manifest["training"]["requested_vocab_size"])
        # Everything still encodes: byte tokens make unknown text impossible.
        self.assertEqual(tokenizer.decode_bytes(tokenizer.encode("texto novo 🙂")), "texto novo 🙂".encode("utf-8"))

    def test_rejects_invalid_corpora_sizes_and_special_tokens(self):
        for documents, vocab_size, options in (
                ([], 400, {}), ("not a list", 400, {}), ([b"bytes"], 400, {}),
                (CORPUS, 267, {}), (CORPUS, MAX_VOCAB_SIZE + 1, {}), (CORPUS, 400.0, {}),
                (CORPUS, 400, {"min_frequency": 0}), (CORPUS, 400, {"special_tokens": ("",)}),
                (CORPUS, 400, {"special_tokens": ("a",)})):  # Collides with a byte token.
            with self.subTest(vocab_size=vocab_size, options=sorted(options)):
                with self.assertRaises(ValueError):
                    train_byte_bpe(documents, vocab_size, **options)


class TokenizerEncodingRegressions(_TokenizerFixture):
    def test_round_trip_is_byte_exact_for_text_the_corpus_never_saw(self):
        tokenizer, _, _ = self.build()
        rng = random.Random(4242)
        texts = [*SAMPLES.values(), "", " ", "\n\n\t ", "a" * 500, "🙂" * 40,
                 "mixed 42 tokens\r\nwith CRLF", "<|bos|><|eos|>", "\x00\x01 control bytes"]
        texts.extend("".join(chr(rng.randrange(0x20, 0x2FFF)) for _ in range(rng.randrange(1, 60)))
                     for _ in range(40))
        for text in texts:
            with self.subTest(text=text[:24]):
                ids = tokenizer.encode(text)
                self.assertEqual(tokenizer.decode_bytes(ids), text.encode("utf-8"))
                self.assertEqual(tokenizer.decode(ids), text)
                self.assertTrue(all(0 <= identifier < tokenizer.vocab_size for identifier in ids))

    def test_text_can_never_produce_a_special_token(self):
        tokenizer, _, _ = self.build()
        reserved = set(tokenizer.special_tokens.values())
        for name in DEFAULT_SPECIAL_TOKENS:
            for text in (name, f"prefix {name} suffix", name * 3, name.upper()):
                with self.subTest(text=text):
                    self.assertFalse(reserved.intersection(tokenizer.encode(text)))
        # Markers enter only through the explicit frame arguments.
        framed = tokenizer.encode("oi", prefix=["<|user|>"], suffix=["<|end|>"])
        self.assertEqual(framed[0], tokenizer.special_id("<|user|>"))
        self.assertEqual(framed[-1], tokenizer.special_id("<|end|>"))
        self.assertEqual(tokenizer.decode(framed, skip_special=True), "oi")
        self.assertIn("<|user|>", tokenizer.decode(framed))
        with self.assertRaises(ValueError):
            tokenizer.encode("oi", prefix=["<|missing|>"])

    def test_segmentation_keeps_digits_whitespace_and_classes_apart(self):
        self.assertEqual(segment("ab 12  c\n\nd"), ["ab", " 1", "2", "  ", "c", "\n\n", "d"])
        self.assertEqual(segment(""), [])
        self.assertEqual(segment("   "), ["   "])
        self.assertEqual(segment("a,b"), ["a", ",", "b"])
        self.assertEqual(segment(" 2024-01-02"), [" 2", "0", "2", "4", "-", "0", "1", "-", "0", "2"])
        self.assertEqual("".join(segment(SAMPLES["unicode"])), SAMPLES["unicode"])
        with self.assertRaises(ValueError):
            segment(b"bytes")

    def test_merges_never_cross_a_segment_boundary(self):
        tokenizer, _, _ = self.build()
        for text in ("memória 512", "x=1", "a\nb"):
            with self.subTest(text=text):
                pieces = segment(text)
                joined = tokenizer.encode(text)
                separate = [identifier for piece in pieces for identifier in tokenizer.encode(piece)]
                self.assertEqual(joined, separate)

    def test_efficiency_metrics_cover_each_domain(self):
        tokenizer, _, _ = self.build()
        report = tokenizer.measure(SAMPLES)
        self.assertEqual(set(report), set(SAMPLES))
        for domain, row in report.items():
            with self.subTest(domain=domain):
                self.assertGreater(row["tokens"], 0)
                self.assertEqual(row["bytes"], len(SAMPLES[domain].encode("utf-8")))
                self.assertAlmostEqual(row["bytes_per_token"], row["bytes"] / row["tokens"])
                self.assertGreaterEqual(row["bytes_per_token"], 1.0)

    def test_invalid_token_ids_are_rejected(self):
        tokenizer, _, _ = self.build()
        for identifier in (-1, tokenizer.vocab_size, True, 1.0, None):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                tokenizer.decode_bytes([identifier])


class TokenizerAssetRegressions(_TokenizerFixture):
    def damaged(self, mutate, name="damaged"):
        tokenizer, _, source = self.build(name)
        target = self.directory / f"{name}-copy"
        shutil.copytree(source, target)
        mutate(target)
        return target

    def test_checksum_size_and_version_mismatches_are_rejected(self):
        def truncate(path):
            data = (path / "vocab.bin").read_bytes()
            (path / "vocab.bin").write_bytes(data[:-4])

        def corrupt(path):
            data = bytearray((path / "merges.bin").read_bytes())
            data[-1] ^= 0xFF
            (path / "merges.bin").write_bytes(bytes(data))

        def rewrite_manifest(field, value):
            def mutate(path):
                manifest = json.loads((path / "manifest.json").read_text())
                manifest[field] = value
                (path / "manifest.json").write_text(json.dumps(manifest))
            return mutate

        def drop_file(path):
            (path / "merges.bin").unlink()

        def bad_magic(path):
            data = bytearray((path / "vocab.bin").read_bytes())
            data[:8] = b"OTHERTOK"
            (path / "vocab.bin").write_bytes(bytes(data))
            manifest = json.loads((path / "manifest.json").read_text())
            manifest["files"]["vocab.bin"]["sha256"] = __import__("hashlib").sha256(bytes(data)).hexdigest()
            (path / "manifest.json").write_text(json.dumps(manifest))

        mutations = [truncate, corrupt, drop_file, bad_magic,
                     rewrite_manifest("version", 2), rewrite_manifest("model", "wordpiece"),
                     rewrite_manifest("segmentation", "other"), rewrite_manifest("vocab_size", 3),
                     rewrite_manifest("files", {"vocab.bin": {}}),
                     rewrite_manifest("special_tokens", ["not", "a", "map"])]
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                path = self.damaged(mutate, f"case-{index}")
                with self.assertRaises((ValueError, OSError, KeyError, TypeError)):
                    NexaTokenizer.load(path)

    def test_merges_outside_the_vocabulary_and_duplicate_tokens_are_rejected(self):
        tokens = [bytes((value,)) for value in range(256)] + [b"<|bos|>"]
        destination = self.directory / "hand-made"
        write_tokenizer(destination, tokens, [(b"a", b"b")], {"<|bos|>": 256})
        data = bytearray((destination / "merges.bin").read_bytes())
        struct.pack_into("<I", data, 16, 9999)  # Left id past the vocabulary.
        (destination / "merges.bin").write_bytes(bytes(data))
        manifest = json.loads((destination / "manifest.json").read_text())
        manifest["files"]["merges.bin"]["sha256"] = __import__("hashlib").sha256(bytes(data)).hexdigest()
        (destination / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            NexaTokenizer.load(destination)
        with self.assertRaises(ValueError):
            NexaTokenizer(tokens + [b"<|bos|>"], [], {"<|bos|>": 256}, {})
        with self.assertRaises(ValueError):
            NexaTokenizer(tokens, [], {"<|bos|>": 3}, {})

    def test_publishing_twice_into_the_same_directory_fails(self):
        _, _, destination = self.build("published")
        with self.assertRaises(OSError):
            write_tokenizer(destination, [b"a"], [], {})

    def test_manifest_records_the_corpus_and_reloads_identically(self):
        tokenizer, manifest, destination = self.build("identity")
        training = manifest["training"]
        self.assertEqual(len(training["corpus_sha256"]), 64)
        self.assertEqual(training["documents"], len(CORPUS) * 3)
        self.assertGreater(training["bytes"], 0)
        reloaded = NexaTokenizer.load(destination)
        self.assertEqual(reloaded.manifest, tokenizer.manifest)
        self.assertEqual(reloaded.encode(SAMPLES["pt"]), tokenizer.encode(SAMPLES["pt"]))


class TokenizerCLIRegressions(_TokenizerFixture):
    def run_cli(self, *arguments, expect=0):
        result = subprocess.run([sys.executable, str(ROOT / "tools/nexa_tokenizer.py"), *arguments],
                                capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, expect, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        return json.loads(result.stdout) if expect == 0 else result.stderr

    def test_train_inspect_encode_and_decode_round_trip(self):
        corpus = self.corpus_directory()
        destination = self.directory / "cli-tokenizer"
        trained = self.run_cli("train", "--corpus", str(corpus), "--out", str(destination),
                               "--vocab-size", "420")
        self.assertEqual(trained["vocab_size"], 420)
        self.assertTrue(trained["training"]["reached_target"])
        samples = self.directory / "samples.json"
        samples.write_text(json.dumps(SAMPLES), encoding="utf-8")
        inspected = self.run_cli("inspect", str(destination), "--samples", str(samples))
        self.assertTrue(inspected["verified"])
        self.assertEqual(set(inspected["efficiency"]), set(SAMPLES))
        text = "O NexaLang <|system|> com 512 MB 🙂"
        encoded = self.run_cli("encode", str(destination), "--text", text, "--prefix", "<|bos|>")
        self.assertTrue(encoded["round_trip_exact"])
        self.assertEqual(encoded["special_tokens_from_text"], 0)
        decoded = self.run_cli("decode", str(destination), "--ids",
                               ",".join(str(identifier) for identifier in encoded["token_ids"]),
                               "--skip-special")
        self.assertEqual(decoded["text"], text)

    def test_cli_rejects_bad_arguments_and_existing_destinations(self):
        corpus = self.corpus_directory("cli-corpus")
        destination = self.directory / "existing"
        self.run_cli("train", "--corpus", str(corpus), "--out", str(destination), "--vocab-size", "300")
        for arguments in (("train", "--corpus", str(corpus), "--out", str(destination), "--vocab-size", "300"),
                          ("train", "--corpus", str(self.directory / "missing"), "--out",
                           str(self.directory / "new"), "--vocab-size", "300"),
                          ("train", "--corpus", str(corpus), "--out", str(self.directory / "tiny"),
                           "--vocab-size", "10"),
                          ("inspect", str(self.directory / "missing")),
                          ("encode", str(destination), "--text", "a", "--input", str(corpus / "doc-000.txt")),
                          ("decode", str(destination), "--ids", "999999")):
            with self.subTest(command=arguments[0], arguments=len(arguments)):
                message = self.run_cli(*arguments, expect=(2 if "--input" in arguments else 1))
                self.assertTrue(message.strip())


class TokenizerModelIntegrationRegressions(_TokenizerFixture):
    """Text in, native CPU execution, text out — no PyTorch anywhere."""
    @classmethod
    def setUpClass(cls):
        if not (shutil.which("clang") or shutil.which("cc")):
            raise unittest.SkipTest("C compiler unavailable")

    def paired_model(self, tokenizer):
        from test_transformer_forward_regressions import random_bundle
        bundle = self.directory / "bundle"
        random_bundle(bundle, layers=2, heads=4, kv_heads=2, tied=False, seed=909,
                      vocab_size=tokenizer.vocab_size, max_position_embeddings=64)
        return bundle

    def run_model(self, *arguments, expect=0):
        result = subprocess.run([sys.executable, str(ROOT / "tools/nexa_run.py"), *arguments],
                                capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, expect, result.stderr)
        return json.loads(result.stdout) if expect == 0 else result.stderr

    def test_a_text_prompt_drives_generation_and_decodes_back(self):
        tokenizer, _, asset = self.build("paired", vocab_size=400)
        bundle = self.paired_model(tokenizer)
        prompt = "O NexaLang compila <|system|> 42"
        report = self.run_model(str(bundle), "--prompt", prompt, "--tokenizer", str(asset),
                                "--bos", "--generate", "3", "--kv-cache", "--kv-page-tokens", "2",
                                "--max-sequence-length", "48", "--tile-rows", "4", "--memory-budget", "8MiB")
        expected = tokenizer.encode(prompt, prefix=["<|bos|>"])
        self.assertEqual(report["input_token_ids"], expected)
        self.assertEqual(report["prompt"], prompt)
        self.assertEqual(report["tokenizer"]["vocab_size"], tokenizer.vocab_size)
        self.assertTrue(report["tokenizer"]["framed_with_bos"])
        self.assertEqual(report["tokenizer"]["files"], {name: entry["sha256"] for name, entry
                                                        in tokenizer.manifest["files"].items()})
        generated = report["appended_token_ids"]
        self.assertEqual(len(generated), 3)
        self.assertEqual(report["generated_text"], tokenizer.decode(generated, skip_special=True))
        self.assertEqual(report["decoded_text"], tokenizer.decode(expected + generated, skip_special=True))
        # The literal role marker in the prompt stayed text, ids and all.
        reserved = set(tokenizer.special_tokens.values())
        self.assertEqual([identifier for identifier in expected[1:] if identifier in reserved], [])
        self.assertIn("<|system|>", report["decoded_text"])
        self.assertFalse(report["validation"]["verified"])  # No PyTorch oracle in this path.

    def test_a_tokenizer_from_another_vocabulary_is_rejected(self):
        tokenizer, _, asset = self.build("mismatch", vocab_size=400)
        from test_transformer_forward_regressions import random_bundle
        bundle = self.directory / "other-vocab"
        random_bundle(bundle, layers=1, heads=2, kv_heads=1, seed=31, vocab_size=tokenizer.vocab_size + 5)
        message = self.run_model(str(bundle), "--prompt", "texto", "--tokenizer", str(asset),
                                 "--generate", "1", "--max-sequence-length", "8",
                                 "--tile-rows", "4", "--memory-budget", "4MiB", expect=1)
        self.assertIn("Tokenizer vocabulary differs", message)
        self.assertNotIn("Traceback", message)

    def test_prompt_and_token_arguments_are_mutually_exclusive(self):
        _, _, asset = self.build("arguments", vocab_size=300)
        bundle = self.directory / "unused"
        for arguments in (("--prompt", "oi"),
                          ("--tokens", "1,2", "--prompt", "oi", "--tokenizer", str(asset)),
                          ("--tokenizer", str(asset), "--tokens", "1,2"),
                          ("--tokens", "1,2", "--bos")):
            with self.subTest(arguments=arguments[0]):
                message = self.run_model(str(bundle), *arguments, "--memory-budget", "4MiB", expect=2)
                self.assertNotIn("Traceback", message)


if __name__ == "__main__":
    unittest.main()
