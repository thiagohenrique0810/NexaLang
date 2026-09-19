"""Runtime publication when Windows holds an existing DLL open."""
import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
builder = importlib.import_module("runtime.build_runtime")


class RuntimeBuildRegressions(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="nexa-build-test-")
        self.addCleanup(temporary.cleanup)
        self.output_dir = Path(temporary.name).resolve()
        self.artifact = b"fake compiled runtime\x00\x01\x02"
        self.candidates = []

    def compile(self, command, *, check):
        self.assertTrue(check)
        candidate = Path(command[command.index("-o") + 1])
        self.candidates.append(candidate)
        candidate.write_bytes(self.artifact)
        return subprocess.CompletedProcess(command, 0)

    def assert_clean_temporaries(self):
        self.assertEqual(list(self.output_dir.glob("nexa-runtime-*")), [])
        self.assertTrue(all(not path.exists() for path in self.candidates))

    def test_windows_loaded_dll_uses_unique_persistent_compiled_artifact(self):
        original = self.output_dir / "nexa_transformer.dll"
        original.write_bytes(b"existing loaded DLL")
        replace = builder.os.replace

        def locked_destination(source, destination):
            if Path(destination) == original:
                raise PermissionError("DLL is loaded")
            replace(source, destination)

        with mock.patch.object(builder.platform, "system", return_value="Windows"), \
             mock.patch.object(builder.subprocess, "run", side_effect=self.compile), \
             mock.patch.object(builder.os, "replace", side_effect=locked_destination):
            first = builder.build_runtime("nexa_transformer", self.output_dir)
            second = builder.build_runtime("nexa_transformer", self.output_dir)
        self.assertNotEqual(first, original)
        self.assertNotEqual(first, second)
        for result in (first, second):
            self.assertEqual(result.parent, self.output_dir.resolve())
            self.assertEqual(result.suffix, ".dll")
            self.assertTrue(result.name.startswith("nexa_transformer-"))
            self.assertEqual(result.read_bytes(), self.artifact)
        self.assertEqual(original.read_bytes(), b"existing loaded DLL")
        self.assertEqual(set(self.output_dir.iterdir()), {original, first, second})
        self.assert_clean_temporaries()

    def test_windows_unlocked_destination_retains_normal_filename(self):
        original = self.output_dir / "nexa_transformer.dll"
        original.write_bytes(b"old artifact")
        with mock.patch.object(builder.platform, "system", return_value="Windows"), \
             mock.patch.object(builder.subprocess, "run", side_effect=self.compile):
            result = builder.build_runtime("nexa_transformer", self.output_dir)
        self.assertEqual(result, original)
        self.assertEqual(original.read_bytes(), self.artifact)
        self.assertEqual(list(self.output_dir.iterdir()), [original])
        self.assert_clean_temporaries()

    def test_failed_compilation_preserves_destination_and_never_publishes(self):
        original = self.output_dir / "nexa_transformer.dll"
        original.write_bytes(b"loaded DLL")

        def failed_compile(command, *, check):
            self.compile(command, check=check)
            raise subprocess.CalledProcessError(1, command)

        with mock.patch.object(builder.platform, "system", return_value="Windows"), \
             mock.patch.object(builder.subprocess, "run", side_effect=failed_compile), \
             mock.patch.object(builder.os, "replace") as publication:
            with self.assertRaises(subprocess.CalledProcessError):
                builder.build_runtime("nexa_transformer", self.output_dir)
            publication.assert_not_called()
        self.assertEqual(original.read_bytes(), b"loaded DLL")
        self.assertEqual(list(self.output_dir.iterdir()), [original])
        self.assert_clean_temporaries()

    def test_fallback_publication_failure_propagates_and_cleans_candidate(self):
        original = self.output_dir / "nexa_transformer.dll"
        original.write_bytes(b"loaded DLL")
        with mock.patch.object(builder.platform, "system", return_value="Windows"), \
             mock.patch.object(builder.subprocess, "run", side_effect=self.compile), \
             mock.patch.object(builder.os, "replace", side_effect=[PermissionError("DLL loaded"),
                                                                  PermissionError("fallback denied")]) as publication:
            with self.assertRaisesRegex(PermissionError, "fallback denied"):
                builder.build_runtime("nexa_transformer", self.output_dir)
            self.assertEqual(publication.call_count, 2)
        self.assertEqual(original.read_bytes(), b"loaded DLL")
        self.assertEqual(list(self.output_dir.iterdir()), [original])
        self.assert_clean_temporaries()

    def test_other_platforms_do_not_hide_permission_errors(self):
        for system, filename in (("Linux", "libnexa_transformer.so"), ("Darwin", "libnexa_transformer.dylib")):
            with self.subTest(system=system):
                original = self.output_dir / filename
                original.write_bytes(b"existing library")
                with mock.patch.object(builder.platform, "system", return_value=system), \
                     mock.patch.object(builder.subprocess, "run", side_effect=self.compile), \
                     mock.patch.object(builder.os, "replace", side_effect=PermissionError("denied")) as publication:
                    with self.assertRaisesRegex(PermissionError, "denied"):
                        builder.build_runtime("nexa_transformer", self.output_dir)
                    self.assertEqual(publication.call_count, 1)
                self.assertEqual(original.read_bytes(), b"existing library")
                original.unlink()
                self.assertEqual(list(self.output_dir.iterdir()), [])
                self.assert_clean_temporaries()


if __name__ == "__main__":
    unittest.main()
