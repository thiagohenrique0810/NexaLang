"""Conservative CPU discovery. This module never probes or claims a GPU."""
from dataclasses import dataclass, field
import math
import os
import platform
import struct
import sys
from typing import Mapping


@dataclass(frozen=True)
class HardwareMeasurement:
    value: float
    unit: str
    method: str

    def to_dict(self):
        if type(self.value) not in (int, float) or not math.isfinite(self.value) or self.value < 0:
            raise ValueError("measurement value must be finite and nonnegative")
        if not self.unit or not self.method:
            raise ValueError("measurements require a unit and measurement method")
        return {"value": self.value, "unit": self.unit, "method": self.method}


@dataclass(frozen=True)
class HardwareEstimate:
    value: float
    unit: str
    basis: str

    def to_dict(self):
        if type(self.value) not in (int, float) or not math.isfinite(self.value) or self.value < 0:
            raise ValueError("estimate value must be finite and nonnegative")
        if not self.unit or not self.basis:
            raise ValueError("estimates require a unit and stated basis")
        return {"value": self.value, "unit": self.unit, "basis": self.basis}


@dataclass(frozen=True)
class HardwareProfile:
    architecture: str
    os_name: str
    processor: str | None
    logical_cpu_count: int | None
    physical_cpu_count: int | None = None
    total_memory_bytes: int | None = None
    cache_bytes: int | None = None
    memory_bandwidth_bytes_per_second: float | None = None
    simd_width_bits: int | None = None
    capabilities: Mapping = field(default_factory=dict)
    measurements: Mapping[str, HardwareMeasurement] = field(default_factory=dict)
    estimates: Mapping[str, HardwareEstimate] = field(default_factory=dict)

    @classmethod
    def detect_cpu(cls):
        """Return OS-reported facts; unmeasured hardware properties stay unknown."""
        return cls(architecture=platform.machine(), os_name=platform.system(),
                   processor=platform.processor() or None, logical_cpu_count=os.cpu_count(),
                   capabilities={"pointer_bits": struct.calcsize("P") * 8,
                                 "byte_order": sys.byteorder})

    def to_dict(self):
        return {"schema_version": 1, "device_type": "cpu", "architecture": self.architecture,
                "os_name": self.os_name, "processor": self.processor,
                "logical_cpu_count": self.logical_cpu_count,
                "physical_cpu_count": self.physical_cpu_count,
                "total_memory_bytes": self.total_memory_bytes, "cache_bytes": self.cache_bytes,
                "memory_bandwidth_bytes_per_second": self.memory_bandwidth_bytes_per_second,
                "simd_width_bits": self.simd_width_bits, "capabilities": dict(self.capabilities),
                "measurements": {name: value.to_dict() for name, value in self.measurements.items()},
                "estimates": {name: value.to_dict() for name, value in self.estimates.items()}}
