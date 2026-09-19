"""Private KV scratch files: bounded direct I/O, verification and ownership."""
import ctypes
from dataclasses import FrozenInstanceError, replace
import errno
import gc
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compiler.model_config import ModelConfig
from compiler.tiered_kv_plan import TieredKVCachePlan
from runtime.nexapack import kv_store
from runtime.nexapack.kv_store import (
    KVPageRef, KVPageStore, KV_STORE_BUFFER_BYTES, KV_STORE_CHUNK_BYTES,
    KV_STORE_HEADER, KV_STORE_MAGIC, KV_STORE_MAX_METADATA_BYTES,
)


class _Interrupted(BaseException):
    pass


class _CheckedFile:
    def __init__(self, stream, events, *, short=None, fail_write=None, fail_read=None, error=OSError):
        self.stream, self.events, self.short = stream, events, short
        self.fail_write, self.fail_read, self.error = fail_write, fail_read, error
        self.writes = self.reads = 0

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def __enter__(self):
        self.stream.__enter__()
        return self

    def __exit__(self, *args):
        return self.stream.__exit__(*args)

    def write(self, data):
        self.writes += 1
        self.events.append(("write", len(data), type(data)))
        if self.writes == self.fail_write:
            raise self.error(errno.ENOSPC, "injected backing write failure")
        view = memoryview(data)[:self.short]
        try:
            result = self.stream.write(view)
            self.events.append(("written", result, None))
            return result
        finally:
            view.release()
            data = None

    def readinto(self, data):
        self.reads += 1
        self.events.append(("read", len(data), type(data)))
        if self.reads == self.fail_read:
            raise self.error("injected backing read failure")
        view = memoryview(data)[:self.short]
        try:
            result = self.stream.readinto(view)
            self.events.append(("readinto", result, None))
            return result
        finally:
            view.release()
            data = None


class KVStoreRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nexa-kv-store-test-")
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name) / "scratch"

    def page(self, page_tokens=3):
        config = ModelConfig("KVStoreTest", 16, 128, 160, 2, 2, 1, page_tokens * 4)
        plan = TieredKVCachePlan(config, page_tokens * 4, page_tokens)
        descriptor = plan.desired_pages(page_tokens * 4)[0]
        extent = plan.layout("q3").page_extent_bytes
        owner = (ctypes.c_uint8 * extent)()
        ctypes.memset(ctypes.addressof(owner), 0xA7, extent)
        # Include opaque padding in the byte identity; this store does not
        # decode grouped rows or regenerate alignment bytes.
        owner[0], owner[-1] = 13, 91
        return plan, descriptor, extent, owner

    def store(self, *, identity="model-and-layout-signature"):
        store = KVPageStore(self.parent, identity=identity)
        self.addCleanup(store.close)
        return store

    def assert_empty(self, store):
        self.assertEqual(list(store.directory.iterdir()), [])

    def capture(self, action, error_type):
        try:
            action()
        except error_type as error:
            self.assertIsNotNone(error.__traceback__)
            return error
        self.fail(f"expected {error_type.__name__}")

    def test_roundtrip_preserves_whole_extent_and_reference_is_immutable(self):
        _, descriptor, extent, source = self.page()
        store = self.store()
        reference = store.write_page(descriptor, ctypes.addressof(source), extent)
        destination = (ctypes.c_uint8 * (extent + 17))()
        ctypes.memset(ctypes.addressof(destination), 0xCC, extent + 17)
        self.assertEqual(store.read_page(reference, ctypes.addressof(destination), extent + 17), reference.file_bytes)
        self.assertEqual(bytes(destination[:extent]), bytes(source))
        self.assertEqual(bytes(destination[extent:]), b"\xCC" * 17)
        self.assertEqual(reference.extent_bytes, descriptor.page_allocation_bytes - 63)
        self.assertEqual(reference.checksum, hashlib.sha256(source).hexdigest())
        self.assertEqual(reference.file_bytes, reference.path.stat().st_size)
        self.assertEqual(reference.payload_offset + extent, reference.file_bytes)
        self.assertEqual((reference.page_index, reference.logical_start, reference.valid_tokens, reference.codec),
                         (descriptor.page_index, descriptor.logical_start, descriptor.valid_tokens, "q3"))
        with self.assertRaises(FrozenInstanceError):
            reference.extent_bytes = 1
        with self.assertRaises(AttributeError):
            store.identity = "changed"

    def test_header_metadata_are_canonical_versioned_and_contain_no_paths(self):
        _, descriptor, extent, source = self.page()
        store = self.store()
        reference = store.write_page(descriptor, ctypes.addressof(source), extent)
        data = reference.path.read_bytes()
        magic, version, flags, size, payload_extent, checksum = KV_STORE_HEADER.unpack_from(data)
        self.assertEqual((magic, version, flags, payload_extent), (KV_STORE_MAGIC, 1, 0, extent))
        metadata = data[KV_STORE_HEADER.size:KV_STORE_HEADER.size + size]
        self.assertLessEqual(size, KV_STORE_MAX_METADATA_BYTES)
        self.assertEqual(hashlib.sha256(metadata).digest(), checksum)
        parsed = json.loads(metadata)
        self.assertEqual(metadata, json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii"))
        self.assertEqual(parsed["identity"], store.identity)
        self.assertEqual(parsed["descriptor"], descriptor.to_dict())
        self.assertNotIn("path", parsed)
        self.assertNotIn(str(store.directory).encode(), metadata)

    def test_large_pages_use_bounded_direct_short_io_and_write_each_byte_once(self):
        _, descriptor, extent, source = self.page(4097)
        self.assertGreater(extent, KV_STORE_CHUNK_BYTES * 2)
        store, events = self.store(), []
        original = kv_store._open_file

        def open_file(*args):
            return _CheckedFile(original(*args), events, short=8191)

        with patch.object(kv_store, "_open_file", side_effect=open_file):
            reference = store.write_page(descriptor, ctypes.addressof(source), extent)
            destination = (ctypes.c_uint8 * extent)()
            self.assertEqual(store.read_page(reference, ctypes.addressof(destination), extent), reference.file_bytes)
        self.assertEqual(bytes(destination), bytes(source))
        self.assertEqual(sum(size for kind, size, _ in events if kind == "written"), reference.file_bytes)
        self.assertEqual(sum(size for kind, size, _ in events if kind == "readinto"), reference.file_bytes)
        self.assertTrue(all(size <= KV_STORE_CHUNK_BYTES for kind, size, _ in events if kind in ("read", "write")))
        self.assertTrue(all(kind_type is memoryview for kind, _, kind_type in events if kind == "write"))
        self.assertGreaterEqual(KV_STORE_BUFFER_BYTES, 2 * KV_STORE_MAX_METADATA_BYTES + 2 * KV_STORE_HEADER.size)
        self.assertLess(KV_STORE_BUFFER_BYTES, KV_STORE_CHUNK_BYTES)

    def test_invalid_descriptors_addresses_extents_and_slot_capacities_reject_before_io(self):
        plan, descriptor, extent, source = self.page()
        store = self.store()
        wrong_partial = replace(plan.desired_pages(12)[1], valid_tokens=2)
        cases = [(plan.desired_pages(12)[-1], ctypes.addressof(source), extent),
                 (plan.desired_pages(12)[-2], ctypes.addressof(source), extent),
                 (wrong_partial, ctypes.addressof(source), extent),
                 (descriptor, 0, extent), (descriptor, True, extent),
                 (descriptor, ctypes.addressof(source), extent + 63),
                 (descriptor, ctypes.addressof(source), True),
                 (descriptor, (1 << (8 * ctypes.sizeof(ctypes.c_void_p))) - 1, extent)]
        for args in cases:
            with self.subTest(args=args), patch.object(kv_store, "_open_file", side_effect=AssertionError("opened")):
                with self.assertRaises(ValueError):
                    store.write_page(*args)
        reference = store.write_page(descriptor, ctypes.addressof(source), extent)
        for capacity in (0, extent - 1, True, 1.5):
            with self.assertRaises(ValueError):
                store.read_page(reference, ctypes.addressof(source), capacity)

    def test_reference_ownership_rejects_forged_copied_foreign_or_removed_refs(self):
        _, descriptor, extent, source = self.page()
        store, other = self.store(), self.store()
        reference = store.write_page(descriptor, ctypes.addressof(source), extent)
        foreign = other.write_page(descriptor, ctypes.addressof(source), extent)
        forged = KVPageRef(self.parent / "other.txt", descriptor, extent, reference.checksum, reference.file_bytes)
        for ref in (replace(reference), foreign, forged, None):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                store.read_page(ref, ctypes.addressof(source), extent)
            with self.assertRaises(ValueError):
                store.remove(ref)
        store.remove(reference)
        self.assertFalse(reference.path.exists())
        with self.assertRaises(ValueError):
            store.read_page(reference, ctypes.addressof(source), extent)
        self.assertTrue(foreign.path.exists())

    def test_corrupt_payload_metadata_header_truncation_and_trailing_bytes_fail(self):
        _, descriptor, extent, source = self.page()
        store = self.store()
        reference = store.write_page(descriptor, ctypes.addressof(source), extent)
        original = reference.path.read_bytes()
        variants = []
        for offset in (0, 8, 10, 12, 16, 24, KV_STORE_HEADER.size, reference.payload_offset, len(original) - 1):
            changed = bytearray(original)
            changed[offset] ^= 0x40
            variants.append(bytes(changed))
        variants += [original[:KV_STORE_HEADER.size - 1], original[:-1], original + b"extra"]
        for data in variants:
            reference.path.write_bytes(data)
            with self.subTest(length=len(data)), self.assertRaises(ValueError):
                store.read_page(reference, ctypes.addressof(source), extent)
        reference.path.write_bytes(original)
        self.assertEqual(store.read_page(reference, ctypes.addressof(source), extent), reference.file_bytes)

    def test_metadata_with_recomputed_hash_cannot_change_identity_or_descriptor(self):
        _, descriptor, extent, source = self.page()
        store = self.store(identity="test-identity")
        reference = store.write_page(descriptor, ctypes.addressof(source), extent)
        original = reference.path.read_bytes()
        metadata = original[KV_STORE_HEADER.size:reference.payload_offset]
        variants = [metadata.replace(b"test-identity", b"evil-identity"),
                    metadata.replace(b'"age_rank":3', b'"age_rank":2'),
                    metadata.replace(reference.checksum.encode(), b"0" * 64),
                    metadata.replace(b'"group_size":32', b'"group_size":31')]
        for value in variants:
            self.assertNotEqual(value, metadata)
            header = KV_STORE_HEADER.pack(KV_STORE_MAGIC, 1, 0, len(value), extent, hashlib.sha256(value).digest())
            reference.path.write_bytes(header + value + original[reference.payload_offset:])
            with self.assertRaisesRegex(ValueError, "metadata identity"):
                store.read_page(reference, ctypes.addressof(source), extent)

    def test_oversized_metadata_is_rejected_before_allocation_or_payload(self):
        _, descriptor, extent, source = self.page()
        store = self.store()
        reference = store.write_page(descriptor, ctypes.addressof(source), extent)
        data = bytearray(reference.path.read_bytes())
        struct.pack_into("<I", data, 12, KV_STORE_MAX_METADATA_BYTES + 1)
        reference.path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "header"):
            store.read_page(reference, ctypes.addressof(source), extent)
        for identity in (None, "", True, "a" * 4097, "😀" * 1000):
            with self.assertRaises(ValueError):
                self.store(identity=identity)

    def test_no_space_and_cancellation_during_write_remove_private_temporaries(self):
        for error_type in (OSError, _Interrupted):
            _, descriptor, extent, source = self.page(4097)
            store, events = self.store(), []
            original = kv_store._open_file

            def open_file(*args):
                return _CheckedFile(original(*args), events, fail_write=2, error=error_type)

            with patch.object(kv_store, "_open_file", side_effect=open_file):
                saved = self.capture(lambda: store.write_page(descriptor, ctypes.addressof(source), extent), error_type)
            self.assert_empty(store)
            reference = store.write_page(descriptor, ctypes.addressof(source), extent)
            self.assertTrue(reference.path.exists())
            self.assertIsNotNone(saved.__traceback__)

    def test_fsync_and_replace_failure_or_post_replace_cancellation_roll_back(self):
        _, descriptor, extent, source = self.page()
        for operation in ("fsync", "replace", "cancel_after_replace"):
            with self.subTest(operation=operation):
                store = self.store()
                if operation == "fsync":
                    context = patch.object(kv_store.os, "fsync", side_effect=OSError("fsync failed"))
                elif operation == "replace":
                    context = patch.object(store, "_publish", side_effect=OSError("replace failed"))
                else:
                    real_publish = store._publish

                    def publish(*args):
                        real_publish(*args)
                        raise _Interrupted("cancel after successful replace")

                    context = patch.object(store, "_publish", side_effect=publish)
                with context:
                    saved = self.capture(lambda: store.write_page(descriptor, ctypes.addressof(source), extent),
                                         _Interrupted if operation == "cancel_after_replace" else OSError)
                self.assert_empty(store)
                self.assertIsNotNone(saved.__traceback__)

    def test_interrupted_read_is_retryable_and_retains_no_source_or_destination_owner(self):
        _, descriptor, extent, source = self.page(4097)
        store = self.store()
        source_ref = weakref.ref(source)
        reference = store.write_page(descriptor, ctypes.addressof(source), extent)
        source = None
        gc.collect()
        self.assertIsNone(source_ref(), "store retained the source allocation")
        destination = (ctypes.c_uint8 * extent)()
        destination_ref = weakref.ref(destination)
        address = ctypes.addressof(destination)
        original = kv_store._open_file
        events = []

        def open_file(*args):
            return _CheckedFile(original(*args), events, fail_read=4, error=_Interrupted)

        with patch.object(kv_store, "_open_file", side_effect=open_file):
            saved = self.capture(lambda: store.read_page(reference, address, extent), _Interrupted)
        destination = None
        gc.collect()
        self.assertIsNone(destination_ref(), "read failure retained its borrowed reload slot")
        destination = (ctypes.c_uint8 * extent)()
        store.read_page(reference, ctypes.addressof(destination), extent)
        self.assertEqual(hashlib.sha256(destination).hexdigest(), reference.checksum)
        self.assertIsNotNone(saved.__traceback__)

    def test_directory_is_exclusive_close_removes_only_owned_files(self):
        _, descriptor, extent, source = self.page()
        store, other = self.store(), self.store()
        self.assertNotEqual(store.directory, other.directory)
        parent_file = self.parent / "user-data.txt"
        parent_file.write_text("preserve")
        private_foreign = store.directory / "foreign.txt"
        private_foreign.write_text("also preserve")
        own = store.write_page(descriptor, ctypes.addressof(source), extent)
        other_ref = other.write_page(descriptor, ctypes.addressof(source), extent)
        store.close()
        store.close()
        self.assertTrue(store.closed)
        self.assertFalse(own.path.exists())
        self.assertEqual(parent_file.read_text(), "preserve")
        self.assertEqual(private_foreign.read_text(), "also preserve")
        self.assertTrue(other_ref.path.exists())
        with self.assertRaises(ValueError):
            store.write_page(descriptor, ctypes.addressof(source), extent)

    def test_close_deletion_failure_preserves_retryable_ownership(self):
        _, descriptor, extent, source = self.page()
        store = self.store()
        first = store.write_page(descriptor, ctypes.addressof(source), extent)
        second = store.write_page(descriptor, ctypes.addressof(source), extent)
        original = store._unlink

        def unlink(name):
            if name == first.path.name:
                raise PermissionError("injected delete failure")
            return original(name)

        with patch.object(store, "_unlink", side_effect=unlink):
            with self.assertRaises(PermissionError):
                store.close()
        self.assertFalse(store.closed)
        self.assertTrue(first.path.exists())
        self.assertFalse(second.path.exists())
        store.close()
        self.assertFalse(store.directory.exists())

    def test_failed_write_and_failed_cleanup_preserve_original_error_and_close_retry(self):
        _, descriptor, extent, source = self.page()
        store = self.store()
        foreign = store.directory / "foreign.txt"
        foreign.write_text("preserve")
        original_error = OSError(errno.ENOSPC, "original write failure")
        with patch.object(kv_store, "_write_all", side_effect=original_error):
            with patch.object(store, "_unlink", side_effect=PermissionError("cleanup denied")):
                saved = self.capture(lambda: store.write_page(descriptor, ctypes.addressof(source), extent), OSError)
                self.assertIs(saved, original_error)
                with self.assertRaises(PermissionError):
                    store.close()
        self.assertFalse(store.closed)
        self.assertEqual(len(store._pending_cleanup), 1)
        self.assertEqual(len(list(store.directory.glob("temporary-*"))), 1)
        self.assertFalse(store._refs)
        store.close()
        self.assertTrue(store.closed)
        self.assertEqual(list(store.directory.iterdir()), [foreign])
        self.assertEqual(foreign.read_text(), "preserve")
        self.assertIsNotNone(saved.__traceback__)

    def test_post_rename_cancellation_and_failed_cleanup_keep_published_inode_for_close(self):
        _, descriptor, extent, source = self.page()
        store = self.store()
        foreign = store.directory / "foreign.txt"
        foreign.write_text("preserve")
        publish = store._publish
        original_error = _Interrupted("original cancellation after rename")

        def interrupted_publish(*args):
            publish(*args)
            raise original_error

        with patch.object(store, "_publish", side_effect=interrupted_publish):
            with patch.object(store, "_unlink", side_effect=PermissionError("cleanup denied")):
                saved = self.capture(lambda: store.write_page(descriptor, ctypes.addressof(source), extent), _Interrupted)
                self.assertIs(saved, original_error)
                with self.assertRaises(PermissionError):
                    store.close()
        self.assertFalse(store.closed)
        self.assertEqual(len(store._pending_cleanup), 1)
        self.assertEqual(len(list(store.directory.glob("page-*.kvp"))), 1)
        self.assertFalse(store._refs)
        store.close()
        self.assertTrue(store.closed)
        self.assertEqual(list(store.directory.iterdir()), [foreign])
        self.assertEqual(foreign.read_text(), "preserve")
        self.assertIsNotNone(saved.__traceback__)

    def test_path_fallback_without_dir_fd_roundtrips_and_cleans_up(self):
        _, descriptor, extent, source = self.page()
        with patch.object(kv_store.os, "supports_dir_fd", set()):
            store = self.store()
        self.assertIsNone(store._directory_fd)
        reference = store.write_page(descriptor, ctypes.addressof(source), extent)
        destination = (ctypes.c_uint8 * extent)()
        store.read_page(reference, ctypes.addressof(destination), extent)
        self.assertEqual(bytes(source), bytes(destination))
        store.remove(reference)
        self.assert_empty(store)
        store.close()
        self.assertFalse(store.directory.exists())


if __name__ == "__main__":
    unittest.main()
