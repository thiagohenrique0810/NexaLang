"""Offline CLI conversion, inspection and explicitly declared legacy migration."""
import hashlib
import gc
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import weakref
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.nexapack import format as fmt
from test_tq_portable_format_regressions import codebook, reference_pack


class PortableTQCLIRegressions(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="nexa-tq-cli-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = self.directory / "source.f32"
        self.output = self.directory / "weights.nxp"
        self.centroids, self.codebook = codebook(3)
        self.values = [[1, -2, 3, -4, 5, -6, 7, -8], [0] * 8, [1] * 8]
        self.records = [reference_pack(row, 3, 42, self.centroids) for row in self.values]
        self.source.write_bytes(struct.pack("<24f", *(x for row in self.values for x in row)))
        self.codebook_path = self.directory / "codebook.json"
        self.codebook_metadata = {"dim": 8, "bits": 3, "seed": 42,
                                  "transform_id": "SRHT_XOSHIRO256SS_V1",
                                  "codebook_f32le": self.codebook}
        self.codebook_path.write_text(json.dumps(self.codebook_metadata))

    def command(self, script, *arguments, ok=True):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "tools" / script),
                                 *map(str, arguments)], cwd=ROOT, capture_output=True,
                                text=True, timeout=60)
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("Traceback", result.stderr)
        return result

    def arguments(self, source=None):
        return [source or self.source, "--out", self.output, "--rows", 3, "--cols", 8,
                "--codec", "tq", "--bits", 3, "--seed", 42, "--block-rows", 2]

    def test_f32_conversion_and_inspection_without_site_packages(self):
        result = self.command("nexa_convert.py", *self.arguments(),
                              "--codebook", self.codebook_path, "--memory-budget", "96KiB")
        self.assertEqual(result["codec"], "TQ_MSE_SRHT")
        self.assertEqual(result["shape"], [3, 8])
        with fmt.NexaPackReader(self.output) as reader:
            self.assertEqual(reader.read_rows(0, 3), b"".join(self.records))
            self.assertEqual(reader.codebook_f32le, self.codebook)
        inspected = self.command("nexa_inspect.py", self.output)
        self.assertEqual(inspected["metadata"]["codec_id"], "TQ_MSE_SRHT")
        self.assertFalse(inspected["validation"]["payloads_verified"])
        verified = self.command("nexa_inspect.py", self.output, "--verify")
        self.assertTrue(verified["validation"]["payloads_verified"])
        self.assertTrue(verified["validation"]["checksums_verified"])

    def test_default_codec_remains_q4_and_explicit_q4_is_identical(self):
        self.command("nexa_convert.py", self.source, "--out", self.output, "--rows", 3, "--cols", 8)
        previous = self.output.read_bytes()
        self.command("nexa_convert.py", self.source, "--out", self.output, "--rows", 3, "--cols", 8,
                     "--codec", "q4")
        self.assertEqual(self.output.read_bytes(), previous)
        verified = self.command("nexa_inspect.py", self.output, "--verify")
        self.assertEqual(verified["metadata"]["codec_id"], "Q4_GROUPED")
        self.assertTrue(verified["validation"]["q4_codec_validated"])

    def test_legacy_little_and_big_endian_migration_preserves_indices_and_codebook(self):
        expected = None
        for endianness, spec in (("little", "<f"), ("big", ">f")):
            source = self.directory / (endianness + ".tq01")
            source.write_bytes(b"".join(b"TQ01" + struct.pack(spec, struct.unpack_from("<f", row, 4)[0])
                                         + row[8:] for row in self.records))
            self.command("nexa_convert.py", *self.arguments(source), "--legacy-tq01",
                         "--source-endianness", endianness, "--codebook", self.codebook_path)
            with fmt.NexaPackReader(self.output) as reader:
                self.assertEqual(reader.read_rows(0, 3), b"".join(self.records))
                self.assertEqual(reader.codebook_f32le, self.codebook)
            if expected is None:
                expected = self.output.read_bytes()
            else:
                self.assertEqual(self.output.read_bytes(), expected)
        original = source.read_bytes()
        malformed = [b"TQ02" + original[4:], original[:-1], original + b"\0",
                     original[:4] + struct.pack(">f", -1) + original[8:],
                     original[:4] + bytes(4) + b"\x01" + original[9:]]
        for content in malformed:
            with self.subTest(content=content[:12].hex()):
                source.write_bytes(content)
                self.command("nexa_convert.py", *self.arguments(source), "--legacy-tq01",
                             "--source-endianness", "big", "--codebook", self.codebook_path, ok=False)
                self.assertEqual(self.output.read_bytes(), expected)
                self.assertEqual(list(self.directory.glob(".nexapack-*")), [])

    def test_migration_requires_explicit_origin_and_codebook_and_rejects_codec_misuse(self):
        invalid = [self.arguments() + ["--legacy-tq01"],
                   self.arguments() + ["--legacy-tq01", "--source-endianness", "little"],
                   self.arguments() + ["--legacy-tq01", "--codebook", self.codebook_path],
                   self.arguments() + ["--group-size", 32],
                   self.arguments() + ["--source-endianness", "little"],
                   [self.source, "--out", self.output, "--rows", 3, "--cols", 8, "--bits", 3],
                   ["--checkpoint", self.directory, "--out", self.output, "--codec", "tq"]]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                self.command("nexa_convert.py", *arguments, ok=False)
                self.assertFalse(self.output.exists())

    def test_codebook_document_strict_shape_types_and_size(self):
        self.output.write_bytes(b"preserved")
        documents = []
        for fields in ({"dim": 16}, {"bits": 2}, {"seed": -7}, {"seed": True},
                       {"transform_id": "unknown"}, {"extra": 1}):
            documents.append(json.dumps(dict(self.codebook_metadata, **fields)))
        documents += ["[1,2,3]", "{" + '"dim":8,' + json.dumps(self.codebook_metadata)[1:],
                      " " * 8193 + json.dumps(self.codebook_metadata),
                      "[" * 1100 + "0" + "]" * 1100]
        for document in documents:
            with self.subTest(document=document[:100]):
                self.codebook_path.write_text(document)
                self.command("nexa_convert.py", *self.arguments(), "--codebook", self.codebook_path, ok=False)
                self.assertEqual(self.output.read_bytes(), b"preserved")

    def test_bad_source_budget_and_path_alias_preserve_destination(self):
        self.output.write_bytes(b"preserved")
        arguments = self.arguments() + ["--codebook", self.codebook_path]
        self.command("nexa_convert.py", *arguments, "--memory-budget", "1B", ok=False)
        self.assertEqual(self.output.read_bytes(), b"preserved")
        original = self.source.read_bytes()
        for payload in (original[:-1], original + b"\0"):
            self.source.write_bytes(payload)
            self.command("nexa_convert.py", *arguments, ok=False)
            self.assertEqual(self.output.read_bytes(), b"preserved")
        self.source.write_bytes(original)
        self.command("nexa_convert.py", self.source, "--out", self.source, "--rows", 3,
                     "--cols", 8, "--codec", "tq", ok=False)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(list(self.directory.glob(".nexapack-*")), [])
        # A caller may retain the error. Its traceback must not retain managed
        # conversion buffers and silently multiply the next conversion's peak.
        import tools.nexa_convert as converter
        buffers, errors = [], []
        class TrackedBuffer(bytearray):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                buffers.append(weakref.ref(self))
        self.source.write_bytes(struct.pack("<f", float("nan")) + original[4:])
        with mock.patch.object(converter, "bytearray", TrackedBuffer, create=True):
            try:
                converter.convert_matrix(self.source, self.output, 3, 8, codec="tq",
                                         bits=3, seed=42, codebook_f32le=self.codebook)
            except ValueError as error:
                errors.append(error)
        self.assertEqual(len(errors), 1)
        self.assertTrue(buffers)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in buffers))
        self.assertEqual(self.output.read_bytes(), b"preserved")
        legacy = self.directory / "bad.tq01"
        legacy.write_bytes(b"TQ01" + struct.pack("<f", -1) + self.records[0][8:])
        buffers.clear()
        with mock.patch.object(converter, "bytearray", TrackedBuffer, create=True):
            try:
                converter.convert_tq01(legacy, self.output, 1, 8, 3, 42, self.codebook,
                                        source_endianness="little")
            except ValueError as error:
                errors.append(error)
        self.assertEqual(len(errors), 2)
        self.assertTrue(buffers)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in buffers))
        self.assertEqual(self.output.read_bytes(), b"preserved")

    def test_inspection_checks_semantics_after_valid_payload_checksums(self):
        fmt.write_tq_records(self.output, 3, 8, 3, 42, self.codebook, self.records, block_rows=2)
        content = self.output.read_bytes()
        header = list(fmt.HEADER.unpack(content[:fmt.HEADER.size]))
        metadata = json.loads(content[fmt.HEADER.size:fmt.HEADER.size + header[3]])
        payload = bytearray(content[header[4]:])
        payload[4:8] = struct.pack("<f", -1)
        metadata["blocks"][0]["sha256"] = hashlib.sha256(payload[:22]).hexdigest()
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        header[3] = len(encoded)
        header[6] = hashlib.sha256(encoded).digest()
        self.output.write_bytes(fmt.HEADER.pack(*header) + encoded +
                                bytes(header[4] - fmt.HEADER.size - len(encoded)) + payload)
        self.command("nexa_inspect.py", self.output)
        self.command("nexa_inspect.py", self.output, "--verify", ok=False)
        import tools.nexa_inspect as inspector
        buffers, errors = [], []
        class TrackedBuffer(bytearray):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                buffers.append(weakref.ref(self))
        for corrupt_checksum in (False, True):
            if corrupt_checksum:
                content = bytearray(self.output.read_bytes())
                content[-1] ^= 1
                self.output.write_bytes(content)
            buffers.clear()
            with mock.patch.object(inspector, "bytearray", TrackedBuffer, create=True):
                with fmt.NexaPackReader(self.output) as reader:
                    try:
                        inspector.verify_matrix(reader)
                    except ValueError as error:
                        errors.append(error)
            self.assertEqual(len(errors), 2 if corrupt_checksum else 1)
            self.assertTrue(buffers)
            gc.collect()
            self.assertTrue(all(ref() is None for ref in buffers))

    def test_migration_api_uses_no_native_context(self):
        from runtime.nexapack import tq
        from tools.nexa_convert import convert_tq01
        source = self.directory / "legacy.tq01"
        source.write_bytes(b"".join(b"TQ01" + row[4:] for row in self.records))
        with mock.patch.object(tq, "TQCodec", side_effect=AssertionError("migration must not quantize")):
            convert_tq01(source, self.output, 3, 8, 3, 42, self.codebook,
                         source_endianness="little", block_rows=2)
        with fmt.NexaPackReader(self.output) as reader:
            self.assertEqual(reader.read_rows(0, 3), b"".join(self.records))


if __name__ == "__main__":
    unittest.main()
