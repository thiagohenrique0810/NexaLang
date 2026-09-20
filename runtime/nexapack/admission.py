"""Joint admission for sessions that share one process memory ceiling.

Each session already admits its own budget: the plan refuses to build when the
arena plus the reserves exceed what the caller declared. That check is local,
so two sessions of 300 MiB each pass their own admission and together exceed a
512 MiB process. A pool is the missing shared term: every session admits the
same upper bound it published, against one ceiling, before allocating anything.

The pool reserves an upper bound, never measured residence. Sharing a prefix
between derived sequences lowers what exists in RAM and does not lower the
reservation: the derived sequence may append, and then the pages stop being
shared. Admitting the lower number would be admitting a state the sequences are
free to leave.

Reservations are keyed by an opaque handle so the same session may be admitted
once and released exactly once, and the pool is safe to use from several
threads: admission is the one operation a caller may run concurrently.
"""
import threading

from compiler.model_ir import _integer


# Constante de módulo, como `tiered_kv_plan.POLICY_ID` e
# `offloaded_kv_plan.RELOAD_POLICY_ID`: um policy_id escrito inline não é
# visto pelo registro de ADRs e pode ser renomeado sem quebrar nada.
POLICY_ID = "PROCESS_JOINT_ADMISSION_UPPER_BOUND_V1"


class PoolAdmissionError(MemoryError):
    """A session did not fit the ceiling shared with the sessions already open."""

    def __init__(self, label, required_bytes, reserved_bytes, limit_bytes):
        self.label = label
        self.required_bytes = required_bytes
        self.reserved_bytes = reserved_bytes
        self.limit_bytes = limit_bytes
        super().__init__(
            f"{label} requires {required_bytes} bytes, {reserved_bytes} of {limit_bytes} "
            f"are already reserved by open sessions, {limit_bytes - reserved_bytes} available")


class _Reservation:
    __slots__ = ("label", "bytes", "released")

    def __init__(self, label, size):
        self.label, self.bytes, self.released = label, size, False


class SessionMemoryPool:
    """One ceiling shared by every session admitted against it."""

    def __init__(self, limit_bytes):
        from .transformer import parse_memory_size
        limit = parse_memory_size(limit_bytes)
        _integer(limit, "pool limit", 1)
        self._limit = limit
        self._lock = threading.Lock()
        self._reservations = {}
        self._reserved = 0
        self._peak_reserved = 0
        self._admissions = 0
        self._rejections = 0

    @property
    def limit_bytes(self):
        return self._limit

    @property
    def reserved_bytes(self):
        with self._lock:
            return self._reserved

    @property
    def available_bytes(self):
        with self._lock:
            return self._limit - self._reserved

    @property
    def members(self):
        with self._lock:
            return len(self._reservations)

    @property
    def counters(self):
        with self._lock:
            return {"limit_bytes": self._limit, "reserved_bytes": self._reserved,
                    "available_bytes": self._limit - self._reserved,
                    "peak_reserved_bytes": self._peak_reserved, "members": len(self._reservations),
                    "admissions": self._admissions, "rejections": self._rejections}

    def admit(self, label, required_bytes):
        """Reserve required_bytes or refuse; returns the handle that releases it."""
        _integer(required_bytes, "required bytes")
        with self._lock:
            if self._reserved + required_bytes > self._limit:
                self._rejections += 1
                raise PoolAdmissionError(label, required_bytes, self._reserved, self._limit)
            reservation = _Reservation(label, required_bytes)
            self._reservations[id(reservation)] = reservation
            self._reserved += required_bytes
            self._peak_reserved = max(self._peak_reserved, self._reserved)
            self._admissions += 1
            return reservation

    def release(self, reservation):
        """Return a reservation. Releasing twice is a no-op, not a double credit."""
        if reservation is None:
            return 0
        with self._lock:
            if reservation.released or id(reservation) not in self._reservations:
                return 0
            reservation.released = True
            del self._reservations[id(reservation)]
            self._reserved -= reservation.bytes
            return reservation.bytes

    def to_dict(self):
        counters = self.counters
        return {"schema_version": 1, "policy_id": POLICY_ID,
                "scope": "declared per-session upper bound, not measured residence", **counters}
