"""Slot admission, cyclic-scan replacement and ownership of reloaded pages."""
import ctypes
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime.nexapack.reload_cache import ReloadPageCache


class Cancelled(BaseException):
    pass


class _Page:
    allocation_bytes = 64

    def __init__(self):
        self.arena = (ctypes.c_uint8 * self.allocation_bytes)()
        self.address = ctypes.addressof(self.arena)
        self.released = False

    def release(self):
        self.address, self.arena, self.released = 0, None, True


class _Ref:
    """Stand-in for a published backing reference; identity is what matters."""
    def __init__(self, name, file_bytes=100):
        self.name, self.file_bytes = name, file_bytes


class _Harness:
    def __init__(self, capacity, failures=()):
        self.pages, self.loaded, self.failures = [], [], list(failures)
        self.cache = ReloadPageCache(capacity, self.allocate)

    def allocate(self):
        page = _Page()
        self.pages.append(page)
        return page

    def acquire(self, ref):
        def load(address):
            if self.failures:
                raise self.failures.pop(0)
            self.loaded.append((ref.name, address))
        return self.cache.acquire(ref, load)

    def scan(self, refs, passes):
        return [[self.acquire(ref) for ref in refs] for _ in range(passes)]


class ReloadCacheRegressions(unittest.TestCase):
    def test_first_touch_admits_and_a_full_cache_reuses_its_most_recent_slot(self):
        harness = _Harness(2)
        refs = [_Ref(f"page-{index}") for index in range(4)]
        harness.scan(refs, 2)
        # Slot zero pins the first page; the second slot streams the rest, so a
        # cyclic scan keeps exactly capacity-1 pages instead of missing always.
        self.assertEqual([name for name, _ in harness.loaded],
                         ["page-0", "page-1", "page-2", "page-3", "page-1", "page-2", "page-3"])
        self.assertEqual((harness.cache.hits, harness.cache.misses), (1, 7))
        self.assertEqual((harness.cache.admissions, harness.cache.evictions), (2, 5))
        self.assertEqual(len(harness.pages), 2)
        self.assertEqual(harness.cache.allocated_bytes, 2 * _Page.allocation_bytes)

    def test_capacity_above_the_cold_prefix_reads_every_page_once(self):
        harness = _Harness(4)
        refs = [_Ref(f"page-{index}", file_bytes=10 + index) for index in range(3)]
        addresses = harness.scan(refs, 5)
        self.assertEqual(len(harness.loaded), 3)
        self.assertEqual(addresses, [addresses[0]] * 5)  # Stable addresses per page.
        self.assertEqual(len(set(addresses[0])), 3)
        self.assertEqual(harness.cache.hits, 12)
        self.assertEqual(harness.cache.bytes_avoided, 4 * sum(ref.file_bytes for ref in refs))
        self.assertEqual(len(harness.pages), 3)  # A fourth slot is never allocated.

    def test_hits_serve_without_loading_and_report_counter_deltas(self):
        harness = _Harness(2)
        first, second = _Ref("a"), _Ref("b", file_bytes=7)
        harness.acquire(first)
        baseline = harness.cache.counters
        address = harness.acquire(second)
        self.assertEqual(harness.acquire(second), address)
        self.assertEqual(harness.cache.delta(baseline),
                         {"hits": 1, "misses": 1, "admissions": 1, "evictions": 0, "bytes_avoided": 7})
        self.assertEqual(len(harness.loaded), 2)

    def test_failed_load_discards_only_its_slot_and_keeps_other_entries(self):
        for error in (OSError("injected read error"), Cancelled("injected cancellation")):
            with self.subTest(error=type(error).__name__):
                harness = _Harness(2)
                pinned = _Ref("pinned")
                pinned_address = harness.acquire(pinned)
                broken = _Ref("broken")
                harness.failures.append(error)
                with self.assertRaises(type(error)):
                    harness.acquire(broken)
                self.assertEqual(harness.cache.entry_count, 1)
                # A partial read leaves undefined bytes: the page must reload.
                harness.acquire(broken)
                self.assertEqual([name for name, _ in harness.loaded], ["pinned", "broken"])
                self.assertEqual(harness.acquire(pinned), pinned_address)
                self.assertEqual(harness.cache.hits, 1)

    def test_discard_and_retain_free_entries_while_reusing_allocated_slots(self):
        harness = _Harness(3)
        refs = [_Ref(f"page-{index}") for index in range(3)]
        harness.scan(refs, 1)
        harness.cache.discard(refs[1])
        self.assertEqual((harness.cache.entry_count, harness.cache.slot_count), (2, 3))
        replacement = _Ref("page-3")
        harness.acquire(replacement)
        self.assertEqual(len(harness.pages), 3)  # The freed slot is reused in place.
        harness.cache.retain([refs[0]])
        self.assertEqual(harness.cache.entry_count, 1)
        self.assertEqual(harness.cache.allocated_bytes, 3 * _Page.allocation_bytes)
        harness.cache.discard(_Ref("never admitted"))
        self.assertEqual(harness.cache.entry_count, 1)
        harness.cache.retain([])
        self.assertEqual(harness.cache.entry_count, 0)
        for ref in refs + [replacement]:
            harness.acquire(ref)
        self.assertEqual(len(harness.pages), 3)

    def test_clear_releases_every_slot_and_a_released_slot_is_never_served(self):
        harness = _Harness(2)
        refs = [_Ref("a"), _Ref("b")]
        harness.scan(refs, 1)
        pages = list(harness.pages)
        harness.cache.clear()
        self.assertTrue(all(page.released and page.arena is None for page in pages))
        self.assertEqual((harness.cache.entry_count, harness.cache.slot_count, harness.cache.allocated_bytes), (0, 0, 0))
        harness.acquire(refs[0])
        self.assertEqual(len(harness.pages), 3)
        single = _Harness(1)
        single.acquire(_Ref("only"))
        single.pages[0].release()
        # A slot whose owner is gone must fail before any load writes into it.
        with self.assertRaises(ValueError):
            single.acquire(_Ref("next"))
        self.assertEqual(len(single.loaded), 1)

    def test_rejects_invalid_capacity_and_missing_reference(self):
        for capacity in (0, -1, True, 1.0, None):
            with self.assertRaises(ValueError):
                ReloadPageCache(capacity, _Page)
        harness = _Harness(1)
        with self.assertRaises(ValueError):
            harness.cache.acquire(None, lambda address: None)
        self.assertEqual(harness.cache.slot_count, 0)


if __name__ == "__main__":
    unittest.main()
