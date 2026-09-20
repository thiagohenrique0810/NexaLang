"""`.nxb` container: layout, window confinement, byte identity and execution.

The silent failure this format can have is a window applied in one place and
forgotten in another: the reader then lands on the neighbouring section and
still satisfies that section's own checksums, because the index offsets and
the payload slide together. Every test here is built to fail when that
happens -- byte identity against the source directory, bit-identical logits
and I/O counters across the two storage forms, and four separate window
refusals in the reader.
"""
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.model_config import ModelConfig
from runtime.nexapack import container as container_module
from runtime.nexapack.bundle import ModelBundleError, ModelBundleReader, write_model_bundle
from runtime.nexapack.container import (FORMAT, MAGIC, NexaContainerError, NexaContainerReader,
                                        SUFFIX, is_container, pack_bundle, unpack_bundle)
from runtime.nexapack.format import ALIGNMENT, HEADER, NexaPackError, NexaPackReader

ALL_PATHS_REJECTED = ("../outside", "/etc/passwd", "C:/secret", "tensors\\data",
                      "tensors//data", "tensors/./data", "", "tensors/../escape")


def align(value):
    return ((value + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT


def tiny_config(tied=True):
    return ModelConfig(name="container", vocab_size=12, hidden_size=8, intermediate_size=16,
                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                       max_position_embeddings=16, tie_word_embeddings=tied,
                       rms_norm_eps=1e-5, rope_theta=10000.0)


def tensor_rows(shape):
    if len(shape) == 1:
        yield (1.0 + index / 8.0 for index in range(shape[0]))
    else:
        for row in range(shape[0]):
            yield (((row * 3 + col * 7) % 17 - 8) / 8.0 for col in range(shape[1]))


def build_bundle(path, config=None, *, asset=None):
    config = config or tiny_config()
    sources = {name: (lambda shape=shape: tensor_rows(shape))
               for name, shape in config.required_tensor_shapes().items()}
    files = {"tokenizer.json": asset} if asset is not None else None
    write_model_bundle(path, config, sources, group_size=4, block_rows=3, tokenizer_files=files)
    return config


def forge_container(path, specs, payloads, *, total_size=None, mutate=None):
    """Write a container byte by byte, with offsets given relative to the payload.

    The real writer can only produce valid files, so every malformed-index case
    below is assembled here instead of by corrupting a good one at random.
    """
    sections = [{"kind": spec["kind"], "path": spec["path"], "offset": 0,
                 "bytes": spec.get("bytes", len(payloads.get(spec["path"], b""))),
                 "sha256": spec.get("sha256") or hashlib.sha256(
                     payloads.get(spec["path"], b"")).hexdigest()}
                for spec in specs]
    index = {"format": FORMAT, "format_version": 1, "alignment": ALIGNMENT, "sections": sections}
    if mutate is not None:
        mutate(index)

    def encode():
        return json.dumps(index, sort_keys=True, separators=(",", ":")).encode("utf-8")

    payload_offset = align(HEADER.size + len(encode()))
    for _ in range(8):
        for section, spec in zip(sections, specs):
            section["offset"] = payload_offset + spec["relative_offset"]
        wanted = align(HEADER.size + len(encode()))
        if wanted == payload_offset:
            break
        payload_offset = wanted
    encoded = encode()
    end = total_size if total_size is not None else align(
        max((section["offset"] + section["bytes"] for section in sections), default=payload_offset))
    blob = bytearray(max(end, payload_offset + 1))
    blob[HEADER.size:HEADER.size + len(encoded)] = encoded
    for section in sections:
        data = payloads.get(section["path"], b"")
        blob[section["offset"]:section["offset"] + len(data)] = data
    blob[:HEADER.size] = HEADER.pack(MAGIC, 1, 0, len(encoded), payload_offset, end,
                                     hashlib.sha256(encoded).digest())
    path.write_bytes(bytes(blob))
    return payload_offset


class _ContainerFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="nexa-container-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.bundle = self.directory / "model"
        self.asset = self.directory / "tokenizer.json"
        self.asset.write_text('{"tokens":["a","b"]}', encoding="utf-8")
        self.config = build_bundle(self.bundle, asset=self.asset)
        self.nxb = self.directory / ("model" + SUFFIX)
        self.measurements = pack_bundle(self.bundle, self.nxb)

    def relative_files(self, root):
        return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())

    def section_specs(self):
        with NexaContainerReader(self.nxb) as container:
            offset = container.payload_offset
            return [{"kind": section["kind"], "path": section["path"],
                     "relative_offset": section["offset"] - offset,
                     "bytes": section["bytes"], "sha256": section["sha256"]}
                    for section in container.sections]

    def payloads(self):
        return {relative: (self.bundle / relative).read_bytes()
                for relative in self.relative_files(self.bundle)}

    def flip(self, path, offset):
        data = bytearray(path.read_bytes())
        data[offset] ^= 0x01
        path.write_bytes(bytes(data))


class ContainerIdentityRegressions(_ContainerFixture):
    def test_every_section_is_the_source_file_byte_for_byte(self):
        originals = self.payloads()
        with NexaContainerReader(self.nxb) as container:
            self.assertEqual(sorted(section["path"] for section in container.sections),
                             sorted(originals))
            for section in container.sections:
                with container.open_section(section["path"]) as stream:
                    extracted = stream.read(section["bytes"])
                self.assertEqual(extracted, originals[section["path"]], section["path"])
                self.assertEqual(hashlib.sha256(extracted).hexdigest(), section["sha256"])
            self.assertEqual(container.verify(), sum(len(value) for value in originals.values()))

    def test_unpack_reproduces_the_same_paths_with_identical_bytes(self):
        restored = self.directory / "restored"
        report = unpack_bundle(self.nxb, restored)
        self.assertEqual(self.relative_files(restored), self.relative_files(self.bundle))
        for relative in self.relative_files(self.bundle):
            self.assertEqual((restored / relative).read_bytes(),
                             (self.bundle / relative).read_bytes(), relative)
        self.assertEqual(report["unpacked_bytes"], sum(len(v) for v in self.payloads().values()))
        with ModelBundleReader(restored) as first, ModelBundleReader(self.bundle) as second:
            self.assertEqual(first.manifest, second.manifest)

    def test_container_and_directory_expose_the_same_tensor_bytes(self):
        with ModelBundleReader(self.bundle) as source, ModelBundleReader(self.nxb) as packed:
            self.assertEqual(source.manifest, packed.manifest)
            self.assertEqual(source.tensor_names, packed.tensor_names)
            self.assertEqual(source.source_kind, "directory")
            self.assertEqual(packed.source_kind, "container")
            for name, shape in self.config.required_tensor_shapes().items():
                if len(shape) == 2:
                    with source.open_packed(name) as left, packed.open_packed(name) as right:
                        self.assertEqual(right.window_offset > 0, True)
                        self.assertEqual(left.read_rows(0, shape[0]), right.read_rows(0, shape[0]))
                        self.assertEqual(left.payload_bytes_read, right.payload_bytes_read)
                else:
                    self.assertEqual(source.read_f32(name), packed.read_f32(name))

    def test_publication_is_atomic_and_never_overwrites(self):
        with self.assertRaises(FileExistsError):
            pack_bundle(self.bundle, self.nxb)
        self.assertEqual(list(self.directory.glob(".nexa-container-*")), [])
        restored = self.directory / "restored"
        unpack_bundle(self.nxb, restored)
        with self.assertRaises(FileExistsError):
            unpack_bundle(self.nxb, restored)
        self.assertEqual(list(self.directory.glob(".nexa-unpack-*")), [])
        broken = self.directory / "broken"
        self.flip(self.nxb, self.section_specs()[1]["relative_offset"] + align(HEADER.size + 1))
        with self.assertRaises(NexaPackError):
            unpack_bundle(self.nxb, broken)
        self.assertFalse(broken.exists())
        self.assertEqual(list(self.directory.glob(".nexa-unpack-*")), [])


class ContainerWindowRegressions(_ContainerFixture):
    """The four refusals that stop a matrix reader from leaving its section."""

    def packed_section(self):
        with NexaContainerReader(self.nxb) as container:
            section = next(entry for entry in container.sections if entry["path"].endswith(".nxp"))
        return section

    def test_a_window_one_byte_short_is_refused(self):
        section = self.packed_section()
        with NexaPackReader(self.nxb, window_offset=section["offset"],
                            window_bytes=section["bytes"]) as reader:
            self.assertEqual(reader.window_bytes, section["bytes"])
        with self.assertRaisesRegex(NexaPackError, "file size"):
            NexaPackReader(self.nxb, window_offset=section["offset"],
                           window_bytes=section["bytes"] - 1)

    def test_a_window_one_byte_long_reaching_into_the_padding_is_refused(self):
        section = self.packed_section()
        self.assertGreater(section["padding_bytes"], 0)
        with self.assertRaisesRegex(NexaPackError, "file size"):
            NexaPackReader(self.nxb, window_offset=section["offset"],
                           window_bytes=section["bytes"] + 1)

    def test_a_header_total_size_that_differs_from_the_window_is_refused(self):
        """The neighbour's size is the realistic wrong number to be handed."""
        with NexaContainerReader(self.nxb) as container:
            sections = [entry for entry in container.sections if entry["path"].endswith(".nxp")]
        first, second = sections[0], sections[1]
        self.assertNotEqual(first["bytes"], second["bytes"] or first["bytes"] + 1)
        with self.assertRaisesRegex(NexaPackError, "file size"):
            NexaPackReader(self.nxb, window_offset=first["offset"], window_bytes=second["bytes"])
        # Same matrix, correct size, wrong start: the header no longer parses.
        with self.assertRaises(NexaPackError):
            NexaPackReader(self.nxb, window_offset=first["offset"] + ALIGNMENT,
                           window_bytes=first["bytes"])

    def test_a_block_reaching_past_the_window_is_refused_before_any_payload_read(self):
        section = self.packed_section()
        forbidden = mock.patch.object(NexaPackReader, "read_rows_into",
                                      side_effect=AssertionError("payload read during open"))
        # A window whose end lies past the file: every block still inside the
        # header's total size, yet the last one has no bytes behind it.
        with forbidden, self.assertRaisesRegex(NexaPackError, "window"):
            NexaPackReader(self.nxb, window_offset=self.nxb.stat().st_size - section["bytes"] + 1,
                           window_bytes=section["bytes"])
        # A forged header that shortens the matrix: the index then covers more
        # bytes than the window owns. Measured honestly, this second case does
        # not isolate the `offset + size > total_size` clause -- deleting only
        # that clause leaves the file refused by the coverage check two lines
        # later, because blocks are contiguous and must end exactly at
        # total_size. It is the refusal that is proved here, not the clause.
        data = bytearray(self.nxb.read_bytes())
        start = section["offset"]
        magic, version, flags, length, payload_offset, total, digest = HEADER.unpack_from(data, start)
        with NexaPackReader(self.nxb, window_offset=start, window_bytes=section["bytes"]) as reader:
            row_bytes = reader.row_bytes
        HEADER.pack_into(data, start, magic, version, flags, length, payload_offset,
                         total - row_bytes, digest)
        forged = self.directory / "forged.nxp"
        forged.write_bytes(bytes(data))
        with forbidden, self.assertRaises(NexaPackError):
            NexaPackReader(forged, window_offset=start, window_bytes=section["bytes"] - row_bytes)

    def test_a_standalone_matrix_keeps_the_zero_window_default(self):
        entry = json.loads((self.bundle / "manifest.json").read_text())["tensors"]
        relative = next(item["path"] for item in entry.values() if item["path"].endswith(".nxp"))
        with NexaPackReader(self.bundle / relative) as reader:
            self.assertEqual((reader.window_offset, reader.window_bytes),
                             (0, (self.bundle / relative).stat().st_size))


class ContainerCorruptionRegressions(_ContainerFixture):
    def test_one_flipped_payload_byte_fails_only_its_own_section(self):
        for section in self.section_specs():
            with self.subTest(path=section["path"]):
                shutil.copyfile(self.nxb, self.directory / "copy.nxb")
                copy = self.directory / "copy.nxb"
                with NexaContainerReader(copy) as container:
                    offset = container.section(section["path"])["offset"]
                self.flip(copy, offset)
                with NexaContainerReader(copy) as container:
                    with self.assertRaisesRegex(NexaContainerError, section["path"]):
                        container.verify()
                    for other in container.sections:
                        if other["path"] == section["path"]:
                            continue
                        digest = hashlib.sha256()
                        with container.open_section(other["path"]) as stream:
                            digest.update(stream.read(other["bytes"]))
                        self.assertEqual(digest.hexdigest(), other["sha256"], other["path"])
                copy.unlink()

    def test_a_flipped_index_byte_is_refused_by_the_header_checksum(self):
        with NexaContainerReader(self.nxb) as container:
            measurements = container.measurements()
            target = container.sections[1]
        index = self.nxb.read_bytes()[HEADER.size:HEADER.size + measurements["index_bytes"]]
        # A checksum digit is the one index byte that survives JSON parsing,
        # the field checks and the layout checks: without the header digest
        # the file would open, and the lie would surface only on a read.
        position = HEADER.size + index.index(target["sha256"].encode("ascii"))
        copy = self.directory / "copy.nxb"
        shutil.copyfile(self.nxb, copy)
        data = bytearray(copy.read_bytes())
        data[position] = ord("b") if data[position] != ord("b") else ord("c")
        copy.write_bytes(bytes(data))
        with self.assertRaisesRegex(NexaContainerError, "index checksum"):
            NexaContainerReader(copy)
        copy.unlink()
        for offset in (HEADER.size, HEADER.size + measurements["index_bytes"] - 1):
            shutil.copyfile(self.nxb, copy)
            self.flip(copy, offset)
            with self.subTest(offset=offset), self.assertRaisesRegex(NexaContainerError,
                                                                     "index checksum"):
                NexaContainerReader(copy)
            copy.unlink()
        # The same digit changed in the header's own digest field is refused too.
        copy = self.directory / "copy.nxb"
        shutil.copyfile(self.nxb, copy)
        self.flip(copy, HEADER.size - 1)
        with self.assertRaisesRegex(NexaContainerError, "index checksum"):
            NexaContainerReader(copy)

    def test_a_flipped_padding_byte_is_refused_at_open(self):
        with NexaContainerReader(self.nxb) as container:
            section = container.sections[0]
            self.assertGreater(section["padding_bytes"], 0)
            padding_start = section["offset"] + section["bytes"]
            index_padding = HEADER.size + container.measurements()["index_bytes"]
        for offset in (padding_start, padding_start + 1, index_padding,
                       self.nxb.stat().st_size - 1):
            copy = self.directory / "copy.nxb"
            shutil.copyfile(self.nxb, copy)
            self.flip(copy, offset)
            with self.subTest(offset=offset), self.assertRaisesRegex(NexaContainerError, "adding"):
                NexaContainerReader(copy)
            copy.unlink()

    def test_corruption_inside_a_packed_section_names_the_tensor(self):
        manifest = json.loads((self.bundle / "manifest.json").read_text())
        name, entry = next((key, value) for key, value in manifest["tensors"].items()
                           if value["path"].endswith(".nxp"))
        with NexaContainerReader(self.nxb) as container:
            section = container.section(entry["path"])
        copy = self.directory / "copy.nxb"
        shutil.copyfile(self.nxb, copy)
        self.flip(copy, section["offset"] + section["bytes"] - 1)
        with ModelBundleReader(copy) as model, model.open_packed(name) as reader:
            with self.assertRaisesRegex(NexaPackError, "checksum"):
                reader.read_rows(0, reader.rows)


class ContainerIndexRegressions(_ContainerFixture):
    def forged(self, **kwargs):
        target = self.directory / "forged.nxb"
        if target.exists():
            target.unlink()
        forge_container(target, **kwargs)
        return target

    def test_a_forged_but_faithful_container_still_opens(self):
        """The forge itself has to be able to produce an accepted file."""
        payloads = self.payloads()
        specs, cursor = [], 0
        for spec in self.section_specs():
            specs.append({**spec, "relative_offset": cursor})
            cursor = align(cursor + spec["bytes"])
        target = self.forged(specs=specs, payloads=payloads)
        with ModelBundleReader(target) as model:
            self.assertEqual(model.config, self.config)

    def layout(self, transform):
        payloads = self.payloads()
        specs, cursor = [], 0
        for spec in self.section_specs():
            specs.append({**spec, "relative_offset": cursor})
            cursor = align(cursor + spec["bytes"])
        transform(specs)
        return {"specs": specs, "payloads": payloads}

    def test_overlapping_holed_and_out_of_order_sections_are_refused(self):
        def overlap(specs):
            specs[2]["relative_offset"] = specs[1]["relative_offset"]

        def hole(specs):
            for spec in specs[2:]:
                spec["relative_offset"] += ALIGNMENT

        def out_of_order(specs):
            specs[1]["relative_offset"], specs[2]["relative_offset"] = (
                specs[2]["relative_offset"], specs[1]["relative_offset"])
            specs[1], specs[2] = specs[2], specs[1]
            specs[1]["relative_offset"], specs[2]["relative_offset"] = (
                specs[2]["relative_offset"], specs[1]["relative_offset"])

        def unaligned(specs):
            for spec in specs[1:]:
                spec["relative_offset"] += 1

        for name, transform in (("overlap", overlap), ("hole", hole),
                                ("out_of_order", out_of_order), ("unaligned", unaligned)):
            with self.subTest(layout=name), self.assertRaises(NexaContainerError):
                NexaContainerReader(self.forged(**self.layout(transform)))

    def test_a_trailing_byte_outside_every_section_is_refused(self):
        arguments = self.layout(lambda specs: None)
        target = self.forged(total_size=None, **arguments)
        with NexaContainerReader(target):
            pass
        bigger = self.forged(total_size=target.stat().st_size + ALIGNMENT, **arguments)
        with self.assertRaisesRegex(NexaContainerError, "complete payload"):
            NexaContainerReader(bigger)

    def test_duplicate_and_unconfined_section_paths_are_refused(self):
        for invalid in ALL_PATHS_REJECTED:
            def rename(specs, invalid=invalid):
                specs[1]["path"] = invalid
            arguments = self.layout(rename)
            arguments["payloads"][invalid] = arguments["payloads"].pop(
                self.section_specs()[1]["path"])
            with self.subTest(path=invalid), self.assertRaises(NexaContainerError):
                NexaContainerReader(self.forged(**arguments))

        def duplicate(specs):
            specs[2]["path"] = specs[1]["path"]
        with self.assertRaisesRegex(NexaContainerError, "Duplicate"):
            NexaContainerReader(self.forged(**self.layout(duplicate)))

    def test_an_unknown_section_kind_is_refused_rather_than_skipped(self):
        for kind in ("plan", "kernel", "variant", "fallback", "", 1, None):
            def relabel(specs, kind=kind):
                specs[1]["kind"] = kind
            with self.subTest(kind=kind), self.assertRaisesRegex(NexaContainerError, "kind"):
                NexaContainerReader(self.forged(**self.layout(relabel)))

    def test_a_kind_cannot_claim_a_path_of_another_kind(self):
        def mislabel(specs):
            specs[1]["kind"] = "asset"
        with self.assertRaisesRegex(NexaContainerError, "cannot carry"):
            NexaContainerReader(self.forged(**self.layout(mislabel)))

        def demote_manifest(specs):
            specs[0]["kind"] = "tensor"
        with self.assertRaises(NexaContainerError):
            NexaContainerReader(self.forged(**self.layout(demote_manifest)))

    def test_the_manifest_section_must_come_first(self):
        def move(specs):
            specs.append(specs.pop(0))
            cursor = 0
            for spec in specs:
                spec["relative_offset"] = cursor
                cursor = align(cursor + spec["bytes"])
        with self.assertRaisesRegex(NexaContainerError, "manifest"):
            NexaContainerReader(self.forged(**self.layout(move)))

    def test_unsupported_header_and_index_fields_are_refused(self):
        arguments = self.layout(lambda specs: None)
        mutations = {"format": lambda index: index.update(format="NexaOther"),
                     "version": lambda index: index.update(format_version=2),
                     "alignment": lambda index: index.update(alignment=512),
                     "extra": lambda index: index.update(plan={}),
                     "missing": lambda index: index.pop("alignment"),
                     "field": lambda index: index["sections"][1].pop("sha256"),
                     "checksum": lambda index: index["sections"][1].update(sha256="z" * 64),
                     "negative": lambda index: index["sections"][1].update(bytes=-1)}
        for name, mutate in mutations.items():
            with self.subTest(mutation=name), self.assertRaises(NexaContainerError):
                NexaContainerReader(self.forged(mutate=mutate, **arguments))

    def test_a_truncated_or_alien_file_is_not_mistaken_for_a_container(self):
        alien = self.directory / "alien.nxb"
        alien.write_bytes(b"NEXAPACK" + bytes(4096))
        self.assertFalse(is_container(alien))
        with self.assertRaises(NexaContainerError):
            NexaContainerReader(alien)
        with self.assertRaises(ModelBundleError):
            ModelBundleReader(alien)
        short = self.directory / "short.nxb"
        short.write_bytes(MAGIC + b"\x00" * 8)
        self.assertTrue(is_container(short))
        with self.assertRaisesRegex(NexaContainerError, "Truncated"):
            NexaContainerReader(short)
        self.assertFalse(is_container(self.directory))


class ContainerMeasurementRegressions(_ContainerFixture):
    def test_published_numbers_are_the_file_and_can_be_negative(self):
        measurements = self.measurements
        payloads = self.payloads()
        self.assertEqual(measurements["file_bytes"], self.nxb.stat().st_size)
        self.assertEqual(measurements["section_bytes"], sum(len(v) for v in payloads.values()))
        self.assertEqual(measurements["container_overhead_bytes"],
                         measurements["file_bytes"] - measurements["section_bytes"])
        self.assertEqual(measurements["container_overhead_bytes"],
                         HEADER.size + measurements["index_bytes"]
                         + measurements["index_padding_bytes"]
                         + measurements["alignment_padding_bytes"])
        self.assertEqual(measurements["sections"], len(payloads))
        for entry in measurements["section_padding"]:
            self.assertEqual(entry["padding_bytes"],
                             align(entry["bytes"]) - entry["bytes"])
        with ModelBundleReader(self.nxb) as model:
            storage = model.inspect()["storage"]
        self.assertEqual(storage["stored_files"], 1)
        self.assertEqual(storage["directory_files"], len(payloads))
        self.assertEqual(storage["bundle_file_bytes"], measurements["section_bytes"])
        self.assertEqual(storage["nxb_vs_directory_bytes"],
                         measurements["file_bytes"] - storage["bundle_file_bytes"])
        # This fixture's tensors are far smaller than one page, so the signed
        # comparison is positive: packing costs more than it saves here.
        self.assertGreater(storage["nxb_vs_directory_bytes"], 0)
        with ModelBundleReader(self.bundle) as model:
            plain = model.inspect()["storage"]
        self.assertEqual(plain["stored_files"], plain["directory_files"])
        self.assertIsNone(plain["nxb_vs_directory_bytes"])

    def test_alignment_padding_shrinks_as_a_share_when_tensors_grow(self):
        """The overhead is per section, so it is a rate, not a constant share."""
        wider = tiny_config()
        wider = ModelConfig(name="wide", vocab_size=256, hidden_size=64, intermediate_size=128,
                            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=16, tie_word_embeddings=True,
                            rms_norm_eps=1e-5, rope_theta=10000.0)
        path = self.directory / "wide"
        build_bundle(path, wider)
        measurements = pack_bundle(path, self.directory / ("wide" + SUFFIX))
        small = self.measurements["container_overhead_bytes"] / self.measurements["section_bytes"]
        large = measurements["container_overhead_bytes"] / measurements["section_bytes"]
        self.assertLess(large, small)


class ContainerExecutionRegressions(_ContainerFixture):
    @classmethod
    def setUpClass(cls):
        if not (shutil.which("clang") or shutil.which("cc")):
            raise unittest.SkipTest("C compiler unavailable")

    def session(self, path):
        from runtime.nexapack.transformer import TransformerSession
        return TransformerSession(path, memory_budget="1MiB")

    def run_native(self, path, tokens):
        with self.session(path) as session:
            logits = session.prefill(tokens)
            return logits, session.report()

    def bits(self, rows, code="<f"):
        return [struct.pack(code, float(value)) for row in rows for value in row]

    def test_directory_container_and_unpacked_runs_are_bit_identical(self):
        tokens = [1, 5, 2, 7]
        restored = self.directory / "restored"
        unpack_bundle(self.nxb, restored)
        results = {name: self.run_native(path, tokens)
                   for name, path in (("directory", self.bundle), ("container", self.nxb),
                                      ("unpacked", restored))}
        reference_logits, reference_report = results["directory"]
        for name, (logits, report) in results.items():
            with self.subTest(source=name):
                self.assertEqual(self.bits(logits), self.bits(reference_logits))
                self.assertEqual(report["logits_sha256"], reference_report["logits_sha256"])
                self.assertEqual(report["io"], reference_report["io"])
                self.assertEqual(report["memory"]["managed_buffers_peak_bound_bytes"],
                                 reference_report["memory"]["managed_buffers_peak_bound_bytes"])
                self.assertGreater(report["io"]["q4_payload_bytes_read"], 0)

    def test_python_oracle_reconstructs_the_same_weights_from_both_forms(self):
        from transformer_reference import (compare_logits, llama_forward, load_bundle_weights,
                                           torch_available)
        if not torch_available():
            self.skipTest("PyTorch unavailable")
        tokens = [1, 5, 2, 7]
        directory_config, directory_weights = load_bundle_weights(self.bundle)
        container_config, container_weights = load_bundle_weights(self.nxb)
        self.assertEqual(directory_config, container_config)
        for name, tensor in directory_weights.items():
            self.assertEqual(tensor.tolist(), container_weights[name].tolist(), name)
        reference = llama_forward(directory_config, directory_weights, tokens)
        packed_reference = llama_forward(container_config, container_weights, tokens)
        self.assertEqual(self.bits(reference, "<d"), self.bits(packed_reference, "<d"))
        native, _ = self.run_native(self.nxb, tokens)
        comparison = compare_logits(native, packed_reference)
        self.assertTrue(comparison["passed"], comparison)

    def test_a_forgotten_window_offset_is_caught_by_the_block_checksum(self):
        """The bug this format is exposed to, injected on purpose.

        A window honoured when the index is parsed and dropped when the
        payload is read lands on whatever sits at the same offset from the
        start of the container. Nothing above the reader would notice.
        """
        original = NexaPackReader.read_rows_into

        def without_window(reader, start, count, destination):
            saved, reader._window_offset = reader._window_offset, 0
            try:
                return original(reader, start, count, destination)
            finally:
                reader._window_offset = saved

        manifest = json.loads((self.bundle / "manifest.json").read_text())
        name = next(key for key, value in manifest["tensors"].items()
                    if value["path"].endswith(".nxp"))
        with ModelBundleReader(self.nxb) as model:
            with model.open_packed(name) as reader:
                self.assertGreater(reader.window_offset, 0)
                expected = reader.read_rows(0, reader.rows)
            with mock.patch.object(NexaPackReader, "read_rows_into", without_window):
                with model.open_packed(name) as reader:
                    with self.assertRaisesRegex(NexaPackError, "checksum"):
                        reader.read_rows(0, reader.rows)
        with ModelBundleReader(self.bundle) as model, model.open_packed(name) as reader:
            # The directory reader is already at window zero, so the same
            # injection has to leave it untouched.
            self.assertEqual(reader.window_offset, 0)
            with mock.patch.object(NexaPackReader, "read_rows_into", without_window):
                self.assertEqual(reader.read_rows(0, reader.rows), expected)

    def test_two_swapped_sections_are_refused_by_index_and_by_manifest(self):
        with NexaContainerReader(self.nxb) as container:
            sizes = {}
            for section in container.sections:
                if section["path"].endswith(".nxp"):
                    sizes.setdefault(section["bytes"], []).append(section)
            first, second = next(pair for pair in sizes.values() if len(pair) > 1)[:2]
        swapped = self.directory / "swapped.nxb"
        data = bytearray(self.nxb.read_bytes())
        left = bytes(data[first["offset"]:first["offset"] + first["bytes"]])
        right = bytes(data[second["offset"]:second["offset"] + second["bytes"]])
        data[first["offset"]:first["offset"] + first["bytes"]] = right
        data[second["offset"]:second["offset"] + second["bytes"]] = left
        swapped.write_bytes(bytes(data))
        with NexaContainerReader(swapped) as container:
            with self.assertRaisesRegex(NexaContainerError, first["path"]):
                container.verify()
        with self.assertRaises(ModelBundleError):
            ModelBundleReader(swapped)


class ContainerToolRegressions(_ContainerFixture):
    def tool(self, module, argv):
        from importlib import import_module
        from io import StringIO
        from contextlib import redirect_stderr, redirect_stdout
        stream = StringIO()
        # The failing cases below print their diagnostic to stderr on purpose;
        # capturing it keeps a passing suite quiet.
        with redirect_stdout(stream), redirect_stderr(StringIO()):
            code = import_module(module).main(argv)
        return code, stream.getvalue()

    def test_pack_unpack_and_verify_report_measured_numbers(self):
        target = self.directory / "cli.nxb"
        code, output = self.tool("tools.nexa_pack", ["pack", str(self.bundle), str(target)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["file_bytes"], target.stat().st_size)
        code, output = self.tool("tools.nexa_pack", ["verify", str(target)])
        self.assertEqual(code, 0)
        report = json.loads(output)
        self.assertEqual(report["verified_bytes"], report["section_bytes"])
        restored = self.directory / "cli-restored"
        code, output = self.tool("tools.nexa_pack", ["unpack", str(target), str(restored)])
        self.assertEqual(code, 0)
        self.assertEqual(self.relative_files(restored), self.relative_files(self.bundle))
        self.assertEqual(self.tool("tools.nexa_pack", ["pack", str(self.bundle), str(target)])[0], 1)

    def test_inspect_accepts_a_container_and_publishes_its_cost(self):
        code, output = self.tool("tools.nexa_inspect", [str(self.nxb), "--verify"])
        self.assertEqual(code, 0)
        report = json.loads(output)
        self.assertEqual(report["storage"]["stored_files"], 1)
        self.assertEqual(report["storage"]["container"]["file_bytes"], self.nxb.stat().st_size)
        self.assertEqual(report["validation"]["container_bytes_read"],
                         report["storage"]["container"]["section_bytes"])
        code, output = self.tool("tools.nexa_inspect", [str(self.bundle)])
        self.assertIsNone(json.loads(output)["storage"]["container"])

    def test_run_refuses_to_write_its_report_over_the_model(self):
        before = self.nxb.read_bytes()
        code, _ = self.tool("tools.nexa_run",
                            [str(self.nxb), "--tokens", "1,2", "--report", str(self.nxb)])
        self.assertEqual(code, 1)
        self.assertEqual(self.nxb.read_bytes(), before)
        code, _ = self.tool("tools.nexa_run", [str(self.bundle), "--tokens", "1,2",
                                               "--report", str(self.bundle / "report.json")])
        self.assertEqual(code, 1)
        self.assertFalse((self.bundle / "report.json").exists())
        # The allowed case -- a report beside the container -- is what
        # test_run_executes_from_a_container exercises.

    def test_run_executes_from_a_container(self):
        if not (shutil.which("clang") or shutil.which("cc")):
            self.skipTest("C compiler unavailable")
        import tools.nexa_run as nexa_run
        report = self.directory / "run.json"
        code, _ = self.tool("tools.nexa_run", [str(self.nxb), "--tokens", "1,5,2",
                                               "--memory-budget", "8MiB",
                                               "--report", str(report)])
        self.assertEqual(code, 0)
        payload = json.loads(report.read_text())
        self.assertEqual(payload["steps"][0]["sequence_length"], 3)
        self.assertTrue(math.isfinite(payload["steps"][0]["timing"]["execution_wall_seconds"]))
        self.assertEqual(nexa_run.DEFAULT_PAGE_TOKENS, 16)


if __name__ == "__main__":
    unittest.main()
