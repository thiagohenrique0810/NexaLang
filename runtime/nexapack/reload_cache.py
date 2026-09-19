"""Bounded reuse of immutable cold KV pages inside one offloaded session.

A slot holds the exact published bytes of one backing file. Cold pages are
immutable once published, so a loaded slot stays valid until its reference is
retired; nothing here re-encodes, promotes precision or alters a logical page.

Attention rescans every page of the prefix in ascending order, twice per layer.
Admission is therefore by first touch, and a full cache reuses the slot it
loaded last: least-recently-used replacement would miss on every page of such a
cyclic scan. With more slots than cold pages each page is read once per call.

The cache owns only its slots. References, files and committed pages belong to
the store and the session. A failed load discards its slot, because a partial
read can already have overwritten the previous contents.
"""
from __future__ import annotations

from compiler.model_ir import _integer


class ReloadPageCache:
    """Slots for reloaded cold pages, allocated on first use and never shared.

    `allocate` must return an owned page exposing address/allocation_bytes/
    release, sized for the largest cold page. Capacity is admitted memory: the
    cache allocates a slot only when a real miss needs one.
    """
    def __init__(self, capacity, allocate):
        self.capacity = _integer(capacity, "reload cache capacity", 1)
        self._allocate = allocate
        self._slots = []
        self._refs = []
        self._index = {}
        self._recent = None
        self.hits = self.misses = self.admissions = self.evictions = 0
        self.bytes_avoided = 0

    @property
    def entry_count(self):
        return len(self._index)

    @property
    def slot_count(self):
        return len(self._slots)

    @property
    def allocated_bytes(self):
        return sum(slot.allocation_bytes for slot in self._slots)

    @property
    def counters(self):
        return {"hits": self.hits, "misses": self.misses, "admissions": self.admissions,
                "evictions": self.evictions, "bytes_avoided": self.bytes_avoided}

    def delta(self, baseline):
        return {name: value - baseline[name] for name, value in self.counters.items()}

    def _slot_of(self, ref):
        # Entries keep a strong reference, so the identity key stays unique for
        # as long as it can be looked up. A retired reference is never reused.
        return self._index.get(id(ref))

    def _free_slot(self):
        return next((index for index, ref in enumerate(self._refs) if ref is None), None)

    def acquire(self, ref, load):
        """Return the address holding `ref`, calling `load(address)` on a miss."""
        if ref is None:
            raise ValueError("Reload cache requires a backing page reference")
        index = self._slot_of(ref)
        if index is not None:
            self.hits += 1
            self.bytes_avoided += ref.file_bytes
            return self._slots[index].address
        self.misses += 1
        index = self._free_slot()
        if index is None and len(self._slots) < self.capacity:
            index = self._admit()
        elif index is None:
            # Invalidating the most recent slot also frees it, so a full cache
            # still knows its victim; the last slot only guards that invariant.
            index = len(self._slots) - 1 if self._recent is None else self._recent
            self.evictions += 1
        self._invalidate(index)
        address = self._slots[index].address
        if not address:
            raise ValueError("Reload slot lost its own allocation")
        # Contents are undefined until the store verifies the whole payload.
        load(address)
        self._refs[index], self._index[id(ref)], self._recent = ref, index, index
        return address

    def _admit(self):
        slot = self._allocate()
        try:
            self._slots.append(slot)
            self._refs.append(None)
        except BaseException:
            slot.release()
            self._slots = self._slots[:len(self._refs)]
            raise
        self.admissions += 1
        return len(self._slots) - 1

    def _invalidate(self, index):
        ref = self._refs[index]
        if ref is not None:
            del self._index[id(ref)]
            self._refs[index] = None
        if self._recent == index:
            self._recent = None

    def discard(self, ref):
        index = self._slot_of(ref)
        if index is not None:
            self._invalidate(index)

    def retain(self, refs):
        """Drop every entry whose page is no longer committed to the session."""
        live = {id(ref) for ref in refs}
        for index in range(len(self._slots)):
            if self._refs[index] is not None and id(self._refs[index]) not in live:
                self._invalidate(index)

    def clear(self):
        """Release every slot; saved tracebacks must not retain reloaded pages."""
        slots, self._slots, self._refs = self._slots, [], []
        self._index, self._recent = {}, None
        for slot in slots:
            slot.release()
