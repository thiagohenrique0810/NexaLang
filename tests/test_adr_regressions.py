"""ADRs checked against the code, with the code as the oracle.

The central proof is a bijection. `walk_contract_identifiers` asks `compiler`
and `runtime.nexapack` what storage contracts they define, and the ADRs have to
claim exactly that set: every identifier claimed by exactly one ADR, every claim
still present in the code with the value it declares. Adding `Q5_GROUPED`,
raising a `FORMAT_VERSION` or renaming a `POLICY_ID` without writing the ADR
fails here.

A bijection is only worth as much as its oracle, so the extractor is proved
against a synthetic package with known identifiers: an extractor that found
nothing would make the bijection pass trivially against an empty set of claims.

The third proof is that the numbers are recomputed rather than quoted. Every
`adr-measurement` block names a source the test executes against the real
implementation, and a source that never reaches `compiler/` or `runtime/` is
refused -- it could not disagree with the ADR whatever the code did. The
measurements that genuinely cannot be recomputed are listed here by name, so
converting a recomputed number into a quoted one means editing this file.
"""
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import adr_registry
from adr_registry import (
    ADR_DIR, AdrFormatError, CITED, ExtractorError, REQUIRED_SECTIONS, SOURCES,
    assert_source_touches_implementation, declared_text, flatten_scalars,
    is_contract_name, load_adrs, parse_adr, source_touches_implementation,
    walk_contract_identifiers,
)

#: Wall-clock timings and whole-bundle fixtures that a unit test cannot
#: reproduce honestly. A quoted number is not a verified number; keeping the
#: list here means a silent conversion is impossible.
CITED_MEASUREMENTS = {
    "q4_decode_ns_per_value",
    "q2_decode_ns_per_value",
    "f32_decode_ns_per_value",
    "f16_decode_ns_per_value",
    "nxb_overhead_tiny_fixture_percent",
    "nxb_overhead_small_model_percent",
    "nxb_real_cost_tiny_fixture_bytes",
    "q2_slower_than_q4_percent",
    "f16_slower_than_f32_factor",
    "payload_plan_bundle_size_factor_tiny",
    "reload_cache_peak_bytes_256_slots",
    "reload_cache_peak_bytes_one_slot",
    "threads_admitted_against_ten_slots",
}

#: A floor on the census. The walk found 62 identifiers in 19 modules when this
#: registry was written; a run that finds a handful means the walker broke, and
#: a broken walker makes the bijection agree with almost anything.
MINIMUM_CENSUS = 40


def _write_synthetic_package(directory, body, name="synthpkg"):
    """Build a real importable package on disk for the extractor to walk."""
    package = Path(directory) / name
    (package / "sub").mkdir(parents=True)
    (package / "__init__.py").write_text("PACKAGE_LEVEL_CODEC_ID = 'PKG_CODEC'\n")
    (package / "leaf.py").write_text(textwrap.dedent(body))
    (package / "sub" / "__init__.py").write_text("")
    (package / "sub" / "deep.py").write_text("SCHEMA_VERSION = 7\n")
    return name


class ExtractorSelfTest(unittest.TestCase):
    """The oracle itself, proved against identifiers chosen in advance."""

    def setUp(self):
        self._saved_path = list(sys.path)
        self._saved_modules = set(sys.modules)

    def tearDown(self):
        sys.path[:] = self._saved_path
        for name in set(sys.modules) - self._saved_modules:
            del sys.modules[name]

    def test_finds_every_planted_identifier_in_a_real_package(self):
        body = """
            MAGIC = b'SYNTH'
            FORMAT = 'Synthetic'
            FORMAT_VERSION = 3
            SCHEMA_VERSION = 4
            WIDE_CODEC_ID = 'WIDE'
            WIDE_CODEC_VERSION = 2
            SOME_CODEC_IDS = {'a': 1, 'b': 2}
            SOME_CODECS = ('x', 'y')
            MY_TRANSFORM_ID = 'T_V1'
            MY_POLICY_ID = 'P_V1'
            MY_POLICY_IDS = {'one': 'P1', 'two': 'P2'}
            _PRIVATE_CODEC_ID = 'HIDDEN'
        """
        with tempfile.TemporaryDirectory() as directory:
            name = _write_synthetic_package(directory, body)
            sys.path.insert(0, directory)
            found = walk_contract_identifiers((name,))
        planted = {
            f"{name}:PACKAGE_LEVEL_CODEC_ID",
            f"{name}.leaf:MAGIC", f"{name}.leaf:FORMAT",
            f"{name}.leaf:FORMAT_VERSION", f"{name}.leaf:SCHEMA_VERSION",
            f"{name}.leaf:WIDE_CODEC_ID", f"{name}.leaf:WIDE_CODEC_VERSION",
            f"{name}.leaf:SOME_CODEC_IDS", f"{name}.leaf:SOME_CODECS",
            f"{name}.leaf:MY_TRANSFORM_ID", f"{name}.leaf:MY_POLICY_ID",
            f"{name}.leaf:MY_POLICY_IDS", f"{name}.leaf:_PRIVATE_CODEC_ID",
            f"{name}.sub.deep:SCHEMA_VERSION",
        }
        self.assertEqual(planted, set(found),
                         "the extractor must find every planted identifier and nothing else")
        self.assertEqual("b'SYNTH'", found[f"{name}.leaf:MAGIC"].declared)
        self.assertEqual("3", found[f"{name}.leaf:FORMAT_VERSION"].declared)
        self.assertEqual("7", found[f"{name}.sub.deep:SCHEMA_VERSION"].declared)

    def test_walks_into_subpackages(self):
        with tempfile.TemporaryDirectory() as directory:
            name = _write_synthetic_package(directory, "MAGIC = b'X'\n")
            sys.path.insert(0, directory)
            found = walk_contract_identifiers((name,))
        self.assertIn(f"{name}.sub.deep:SCHEMA_VERSION", found,
                      "a nested module must be reached, or its contracts are invisible")

    def test_ignores_names_that_are_not_contract_identifiers(self):
        body = """
            MAX_BLOCKS = 8192
            ALIGNMENT = 4096
            CODEC_BITS = {'a': 4}
            DENSE_WIDTH = {'a': 2}
            codec_id = 'lowercase is not a module constant'
            HEADER = 'not a contract'
        """
        with tempfile.TemporaryDirectory() as directory:
            name = _write_synthetic_package(directory, body, name="synthquiet")
            sys.path.insert(0, directory)
            found = walk_contract_identifiers((name,))
        self.assertEqual({f"{name}:PACKAGE_LEVEL_CODEC_ID",
                          f"{name}.sub.deep:SCHEMA_VERSION"}, set(found))

    def test_raises_instead_of_skipping_a_module_that_cannot_be_imported(self):
        """The failure that would make the bijection pass by finding nothing."""
        with tempfile.TemporaryDirectory() as directory:
            name = _write_synthetic_package(directory, "import a_module_that_does_not_exist\n",
                                            name="synthbroken")
            sys.path.insert(0, directory)
            with self.assertRaises(ExtractorError) as caught:
                walk_contract_identifiers((name,))
        self.assertIn("synthbroken.leaf", str(caught.exception))

    def test_flattens_containers_so_a_new_member_changes_the_value(self):
        before = declared_text({"q2": "Q2_GROUPED", "q4": "Q4_GROUPED"})
        after = declared_text({"q2": "Q2_GROUPED", "q4": "Q4_GROUPED", "q5": "Q5_GROUPED"})
        self.assertNotEqual(before, after)
        self.assertEqual(("a", 1, "b", 2), flatten_scalars({"a": 1, "b": 2}))
        self.assertEqual(("x", "y"), flatten_scalars(("x", "y")))
        self.assertEqual((1,), flatten_scalars(1))

    def test_contract_name_rule(self):
        for name in ("MAGIC", "FORMAT", "FORMAT_VERSION", "SCHEMA_VERSION",
                     "Q3_CODEC_ID", "_CODEC_IDS", "TQ_CODEC_VERSION", "PACKED_CODECS",
                     "TQ_TRANSFORM_ID", "RELOAD_POLICY_ID", "POLICY_IDS"):
            self.assertTrue(is_contract_name(name), name)
        for name in ("MAX_BLOCKS", "ALIGNMENT", "CODEC_BITS", "DENSE_WIDTH",
                     "HEADER", "codec_id", "Codec_Id", "", "MAX_CODEC"):
            self.assertFalse(is_contract_name(name), name)


class BijectionTest(unittest.TestCase):
    """Registry against code, with the code as the oracle."""

    @classmethod
    def setUpClass(cls):
        cls.census = walk_contract_identifiers()
        cls.adrs = load_adrs()

    def test_the_census_is_not_trivially_empty(self):
        self.assertGreaterEqual(
            len(self.census), MINIMUM_CENSUS,
            "the walk found almost nothing, so the bijection below proves nothing")

    def test_every_contract_identifier_is_claimed_by_exactly_one_adr(self):
        claimed = {}
        for adr in self.adrs:
            for identifier in adr.identifiers:
                claimed.setdefault(identifier.name, []).append(adr.id)
        unclaimed = sorted(set(self.census) - set(claimed))
        self.assertEqual([], unclaimed,
                         f"identifiers in the code that no ADR claims: {unclaimed}")
        phantom = sorted(set(claimed) - set(self.census))
        self.assertEqual([], phantom,
                         f"identifiers an ADR claims that the code no longer defines: {phantom}")
        twice = sorted(name for name, owners in claimed.items() if len(owners) > 1)
        self.assertEqual([], twice, f"identifiers claimed by more than one ADR: {twice}")

    def test_every_claimed_value_matches_what_the_code_defines(self):
        mismatches = []
        for adr in self.adrs:
            for identifier in adr.identifiers:
                actual = self.census.get(identifier.name)
                if actual is None:
                    continue
                if actual.declared != identifier.value:
                    mismatches.append(
                        f"{adr.id} {identifier.name}: ADR says {identifier.value}, "
                        f"code says {actual.declared}")
        self.assertEqual([], mismatches, "\n".join(mismatches))

    def test_adr_ids_are_unique_and_match_their_filenames(self):
        seen = set()
        for adr in self.adrs:
            self.assertNotIn(adr.id, seen, f"duplicate ADR id {adr.id}")
            seen.add(adr.id)
            self.assertTrue(adr.path.name.startswith(adr.id + "-"),
                            f"{adr.path.name} does not start with {adr.id}")

    def test_every_adr_carries_the_required_sections(self):
        for adr in self.adrs:
            missing = [section for section in REQUIRED_SECTIONS
                       if section not in adr.sections]
            self.assertEqual([], missing, f"{adr.id} is missing {missing}")

    def test_the_readme_lists_every_adr(self):
        readme = (ADR_DIR / "README.md").read_text(encoding="utf-8")
        for adr in self.adrs:
            self.assertIn(adr.path.name, readme,
                          f"{adr.id} is not linked from docs/adr/README.md")

    def test_prior_art_is_never_recorded_as_verification(self):
        for adr in self.adrs:
            for reference in adr.prior_art:
                self.assertIn(reference.role, adr_registry.PRIOR_ART_ROLES)
                self.assertTrue(reference.note.strip(),
                                f"{adr.id} cites {reference.id} without saying why")


class MeasurementTest(unittest.TestCase):
    """Numbers the test recomputes, and the ones it only quotes."""

    @classmethod
    def setUpClass(cls):
        cls.adrs = load_adrs()
        cls.measurements = [(adr, block) for adr in cls.adrs for block in adr.measurements]

    def test_there_are_measurements_to_check(self):
        self.assertGreater(len(self.measurements), 40)

    def test_measurement_names_are_unique_across_the_registry(self):
        names = [block.name for _adr, block in self.measurements]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        self.assertEqual([], duplicates, f"duplicate measurement names: {duplicates}")

    def test_every_recomputed_measurement_matches_the_code(self):
        """The assertion that makes an ADR a contract instead of a document."""
        mismatches = []
        for adr, block in self.measurements:
            if block.is_cited:
                continue
            self.assertIn(block.source, SOURCES,
                          f"{adr.id} names an unknown source {block.source!r}")
            actual = SOURCES[block.source]()
            if actual != block.value:
                mismatches.append(f"{adr.id} {block.name}: ADR says {block.value!r}, "
                                  f"code says {actual!r}")
        self.assertEqual([], mismatches, "\n".join(mismatches))

    def test_every_source_reaches_the_implementation(self):
        """A source that cannot disagree with its ADR is not a source."""
        inert = [name for name, function in SOURCES.items()
                 if not source_touches_implementation(function)]
        self.assertEqual([], inert, f"sources that never reach compiler/ or runtime/: {inert}")
        for name, function in SOURCES.items():
            assert_source_touches_implementation(name, function)

    def test_the_inert_source_check_rejects_a_literal(self):
        """Prove the guard above fires, rather than trusting that it would."""
        self.assertFalse(source_touches_implementation(lambda: 4096))
        with self.assertRaises(AssertionError):
            assert_source_touches_implementation("fake", lambda: 4096)

    def test_cited_measurements_are_exactly_the_declared_set(self):
        cited = {block.name for _adr, block in self.measurements if block.is_cited}
        self.assertEqual(CITED_MEASUREMENTS, cited,
                         "a measurement changed between recomputed and quoted; "
                         "a quoted number is not a verified number")
        for _adr, block in self.measurements:
            if block.is_cited:
                self.assertTrue(block.cited_from.strip())

    def test_no_source_is_dead(self):
        used = {block.source for _adr, block in self.measurements if not block.is_cited}
        unused = sorted(set(SOURCES) - used)
        self.assertEqual([], unused, f"sources no ADR uses: {unused}")

    def test_every_adr_carries_at_least_one_recomputed_measurement(self):
        for adr in self.adrs:
            recomputed = [block for block in adr.measurements if not block.is_cited]
            self.assertTrue(recomputed, f"{adr.id} has no recomputed measurement")


class ParserStrictnessTest(unittest.TestCase):
    """A parser that ignored a line would let an unchecked claim through."""

    VALID = (
        "---\n"
        "adr: ADR-9999\n"
        "title: Probe\n"
        "status: accepted\n"
        "identifiers:\n"
        "  - name: mod:NAME\n"
        "    value: 1\n"
        "prior_art: []\n"
        "---\n"
        "\n"
        "## Contexto\n"
    )

    def _parse(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ADR-9999-probe.md"
            path.write_text(text, encoding="utf-8")
            return parse_adr(path)

    def test_accepts_the_template_shape(self):
        adr = self._parse(self.VALID)
        self.assertEqual("ADR-9999", adr.id)
        self.assertEqual(1, len(adr.identifiers))
        self.assertEqual("mod:NAME", adr.identifiers[0].name)

    def test_refuses_a_document_without_front_matter(self):
        with self.assertRaises(AdrFormatError):
            self._parse("# ADR without front matter\n")

    def test_refuses_unclosed_front_matter(self):
        with self.assertRaises(AdrFormatError):
            self._parse("---\nadr: ADR-9999\ntitle: X\nstatus: accepted\n")

    def test_refuses_a_missing_identifiers_list(self):
        with self.assertRaises(AdrFormatError):
            self._parse("---\nadr: ADR-9999\ntitle: X\nstatus: accepted\nprior_art:\n---\n")

    def test_refuses_a_malformed_adr_id(self):
        with self.assertRaises(AdrFormatError):
            self._parse(self.VALID.replace("ADR-9999", "ADR-99"))

    def test_refuses_prior_art_presented_as_verification(self):
        text = self.VALID.replace(
            "prior_art: []",
            "prior_art:\n  - id: P07\n    role: verification\n    note: confirms our number")
        with self.assertRaises(AdrFormatError) as caught:
            self._parse(text)
        self.assertIn("independent verification", str(caught.exception))

    def test_refuses_an_unclosed_measurement_block(self):
        text = self.VALID + "\n```adr-measurement\nname: x\nvalue: 1\nunit: b\nsource: y\n"
        with self.assertRaises(AdrFormatError):
            self._parse(text)

    def test_refuses_a_measurement_missing_a_field(self):
        text = self.VALID + "\n```adr-measurement\nname: x\nvalue: 1\n```\n"
        with self.assertRaises(AdrFormatError):
            self._parse(text)

    def test_refuses_a_cited_measurement_without_a_source_document(self):
        text = self.VALID + f"\n```adr-measurement\nname: x\nvalue: 1\nunit: b\nsource: {CITED}\n```\n"
        with self.assertRaises(AdrFormatError) as caught:
            self._parse(text)
        self.assertIn("cited_from", str(caught.exception))

    def test_refuses_cited_from_on_a_recomputed_measurement(self):
        text = (self.VALID + "\n```adr-measurement\nname: x\nvalue: 1\nunit: b\n"
                             "source: real\ncited_from: somewhere\n```\n")
        with self.assertRaises(AdrFormatError):
            self._parse(text)

    def test_refuses_an_unparsable_front_matter_line(self):
        with self.assertRaises(AdrFormatError):
            self._parse(self.VALID.replace("status: accepted", "status accepted"))

    def test_refuses_a_duplicate_front_matter_key(self):
        with self.assertRaises(AdrFormatError):
            self._parse(self.VALID.replace("title: Probe", "title: Probe\ntitle: Again"))

    def test_refuses_an_empty_adr_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AdrFormatError):
                load_adrs(directory)


if __name__ == "__main__":
    unittest.main()
