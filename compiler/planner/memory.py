"""Deterministic arena planning for explicit half-open tensor lifetimes.

Offsets are relative to an aligned arena base. ``peak_bytes`` is each arena's
high-water extent, including alignment holes but excluding its reserved bytes.
The budget check is peak_bytes + reserves <= budgets. Lifetimes describe one
serial execution schedule; asynchronous work must extend them until completion.
"""
from dataclasses import dataclass
import json
import re
from types import MappingProxyType
from typing import Mapping

from ..model_ir import _alignment, _integer, _keys, _load_json, _name


_UNITS = {"B": 1, "KB": 1000, "MB": 1000 ** 2, "GB": 1000 ** 3,
          "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3}


def parse_memory_size(value):
    """Parse bytes or an explicit SI/IEC size; reject ambiguous unit spellings."""
    if type(value) is int:
        return _integer(value, "memory size")
    if not isinstance(value, str):
        raise ValueError("memory size must be integer bytes or an explicit size such as 512MiB")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|KiB|MiB|GiB)", value.strip())
    if not match:
        raise ValueError("memory size requires an unambiguous B/KB/MB/GB/KiB/MiB/GiB unit")
    whole, _, fractional = match.group(1).partition(".")
    numerator = int(whole + fractional) * _UNITS[match.group(2)]
    count, remainder = divmod(numerator, 10 ** len(fractional))
    if remainder:
        raise ValueError("memory size must resolve to a whole number of bytes")
    return count


class MemoryBudgetError(ValueError):
    """A plan cannot fit before any arena or model allocation has occurred."""

    def __init__(self, tier, required_bytes, budget_bytes, reserve_bytes):
        self.tier = tier
        self.required_bytes = required_bytes
        self.budget_bytes = budget_bytes
        self.reserve_bytes = reserve_bytes
        super().__init__(f"{tier} memory budget exceeded: arena requires {required_bytes} bytes "
                         f"plus {reserve_bytes} reserved bytes, budget is {budget_bytes} bytes")


@dataclass(frozen=True)
class MemoryRequest:
    name: str
    size_bytes: int
    start: int
    end: int
    alignment: int = 64
    tier: str = "host"

    def __post_init__(self):
        _name(self.name, "request name")
        _name(self.tier, "memory tier")
        _integer(self.size_bytes, "size_bytes", 1)
        _integer(self.start, "lifetime start")
        _integer(self.end, "lifetime end", 1)
        if self.end <= self.start:
            raise ValueError("lifetime must satisfy 0 <= start < end")
        _alignment(self.alignment)

    def to_dict(self):
        return {"name": self.name, "size_bytes": self.size_bytes,
                "start": self.start, "end": self.end,
                "alignment": self.alignment, "tier": self.tier}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {"name", "size_bytes", "start", "end", "alignment", "tier"}, "MemoryRequest")
        return cls(**data)


@dataclass(frozen=True)
class MemoryAllocation(MemoryRequest):
    offset: int = 0

    def __post_init__(self):
        super().__post_init__()
        _integer(self.offset, "offset")
        if self.offset % self.alignment:
            raise ValueError(f"{self.name}: offset does not satisfy alignment")

    def to_dict(self):
        return {**super().to_dict(), "offset": self.offset}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {"name", "size_bytes", "start", "end", "alignment", "tier", "offset"},
              "MemoryAllocation")
        return cls(**data)


def _byte_map(values, label):
    if not isinstance(values, Mapping):
        raise ValueError(f"{label} must be a mapping of memory tier to integer bytes")
    result = {}
    for tier, size in values.items():
        _name(tier, "memory tier")
        result[tier] = _integer(size, f"{label}[{tier}]")
    return result


@dataclass(frozen=True)
class MemoryPlan:
    allocations: Mapping[str, MemoryAllocation]
    peak_bytes: Mapping[str, int]
    budgets: Mapping[str, int]
    reserves: Mapping[str, int]

    def __post_init__(self):
        if not isinstance(self.allocations, Mapping):
            raise ValueError("allocations must map names to MemoryAllocation objects")
        object.__setattr__(self, "allocations", MappingProxyType(dict(self.allocations)))
        for field in ("peak_bytes", "budgets", "reserves"):
            object.__setattr__(self, field, MappingProxyType(_byte_map(getattr(self, field), field)))
        self.validate()

    def validate(self, requests=None):
        """Check geometry, budgets and optionally an exact source-request contract."""
        tiers = set(self.budgets)
        if set(self.peak_bytes) != tiers or set(self.reserves) != tiers:
            raise ValueError("budgets, reserves and peak_bytes must describe the same tiers")
        computed_peak = dict.fromkeys(tiers, 0)
        by_tier = {tier: [] for tier in tiers}
        for name, allocation in self.allocations.items():
            if not isinstance(allocation, MemoryAllocation) or name != allocation.name:
                raise ValueError("allocation map keys must match MemoryAllocation names")
            if allocation.tier not in tiers:
                raise ValueError(f"missing budget for tier {allocation.tier}")
            computed_peak[allocation.tier] = max(computed_peak[allocation.tier],
                                                 allocation.offset + allocation.size_bytes)
            by_tier[allocation.tier].append(allocation)
        if computed_peak != dict(self.peak_bytes):
            raise ValueError("peak_bytes must equal the actual allocation extent per tier")
        for tier, allocations in by_tier.items():
            if self.peak_bytes[tier] + self.reserves[tier] > self.budgets[tier]:
                raise MemoryBudgetError(tier, self.peak_bytes[tier], self.budgets[tier], self.reserves[tier])
            active = []
            for allocation in sorted(allocations, key=lambda a: (a.start, a.end, a.name)):
                active = [other for other in active if other.end > allocation.start]
                for other in active:
                    if (allocation.offset < other.offset + other.size_bytes
                            and other.offset < allocation.offset + allocation.size_bytes):
                        raise ValueError(f"live allocations overlap: {other.name} and {allocation.name}")
                active.append(allocation)
        if requests is not None:
            expected = {}
            for request in requests:
                if not isinstance(request, MemoryRequest) or isinstance(request, MemoryAllocation):
                    raise ValueError("requests must contain MemoryRequest objects")
                if request.name in expected:
                    raise ValueError("request names must be unique")
                expected[request.name] = request.to_dict()
            actual = {name: MemoryRequest.to_dict(a) for name, a in self.allocations.items()}
            if actual != expected:
                raise ValueError("allocations do not exactly match source request sizes and lifetimes")
        return self

    def to_dict(self):
        return {"schema_version": 1, "budgets": dict(self.budgets),
                "reserves": dict(self.reserves), "peak_bytes": dict(self.peak_bytes),
                "allocations": {name: allocation.to_dict()
                                for name, allocation in sorted(self.allocations.items())}}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {"schema_version", "budgets", "reserves", "peak_bytes", "allocations"}, "MemoryPlan")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported MemoryPlan schema_version")
        if not isinstance(data["allocations"], dict):
            raise ValueError("allocations must be an object")
        return cls(allocations={name: MemoryAllocation.from_dict(value)
                                for name, value in data["allocations"].items()},
                   peak_bytes=data["peak_bytes"], budgets=data["budgets"], reserves=data["reserves"])

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(_load_json(text))


class MemoryPlanner:
    def plan(self, requests, budgets: dict[str, int], reserves: dict[str, int] | None = None):
        budgets = _byte_map(budgets, "budgets")
        reserves = _byte_map({} if reserves is None else reserves, "reserves")
        if not set(reserves).issubset(budgets):
            raise ValueError("every reserved tier must have a budget")
        reserves = {tier: reserves.get(tier, 0) for tier in budgets}
        for tier in budgets:
            if reserves[tier] > budgets[tier]:
                raise MemoryBudgetError(tier, 0, budgets[tier], reserves[tier])
        requests = tuple(requests)
        seen = set()
        for request in requests:
            if not isinstance(request, MemoryRequest) or isinstance(request, MemoryAllocation):
                raise ValueError("requests must contain MemoryRequest objects")
            if request.name in seen:
                raise ValueError(f"duplicate request name: {request.name}")
            seen.add(request.name)
            if request.tier not in budgets:
                raise ValueError(f"missing budget for tier {request.tier}")
        active = {tier: [] for tier in budgets}
        peaks = dict.fromkeys(budgets, 0)
        allocations = {}
        for request in sorted(requests, key=lambda r: (r.start, r.end, r.name)):
            tier = request.tier
            live = sorted((a for a in active[tier] if a.end > request.start), key=lambda a: a.offset)
            offset = 0
            for other in live:
                offset = (offset + request.alignment - 1) & -request.alignment
                if offset + request.size_bytes <= other.offset:
                    break
                offset = max(offset, other.offset + other.size_bytes)
            offset = (offset + request.alignment - 1) & -request.alignment
            extent = offset + request.size_bytes
            if extent + reserves[tier] > budgets[tier]:
                raise MemoryBudgetError(tier, extent, budgets[tier], reserves[tier])
            allocation = MemoryAllocation(**request.to_dict(), offset=offset)
            allocations[request.name] = allocation
            active[tier] = [*live, allocation]
            peaks[tier] = max(peaks[tier], extent)
        return MemoryPlan(allocations, peaks, budgets, reserves).validate(requests)
