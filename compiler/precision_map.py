"""Per-tensor codec selection under a byte budget, from measured calibration.

A precision map says which codec stores each tensor. It is a plan, not a
measurement: the numbers it optimizes come from a calibration report, and the
selection is a documented heuristic over them.

Two honesty constraints shape this module. Individual sensitivities do not add
up — quantizing two tensors is not the sum of quantizing each alone — so the
estimated cost of a map is a ranking aid, never a quality claim. And a map is
only valid for the model it was measured on: it carries the checkpoint identity
and the calibration tokens, and applying it elsewhere is rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from types import MappingProxyType

SCHEMA_VERSION = 1
POLICY_ID = "GREEDY_SENSITIVITY_PER_BYTE_V1"
CODECS = ("q4", "f32")
MAX_MAP_TENSORS = 4096


def _text(value, label):
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True)
class PrecisionMap:
    """Codec per tensor, with the provenance of the decision that produced it."""
    codecs: object
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        codecs = dict(self.codecs)
        if not 0 < len(codecs) <= MAX_MAP_TENSORS:
            raise ValueError("A precision map must name between one and 4096 tensors")
        for name, codec in codecs.items():
            _text(name, "tensor name")
            if codec not in CODECS:
                raise ValueError(f"Unsupported codec for {name}: {codec!r}")
        if not isinstance(self.provenance, dict):
            raise ValueError("Provenance must be a JSON object")
        object.__setattr__(self, "codecs", MappingProxyType(dict(sorted(codecs.items()))))
        object.__setattr__(self, "provenance", json.loads(json.dumps(self.provenance, sort_keys=True)))

    @property
    def dense_tensors(self):
        return tuple(name for name, codec in self.codecs.items() if codec == "f32")

    def bytes_for(self, sizes):
        """Total stored bytes, given {tensor: {"q4": n, "f32": n}} from calibration."""
        total = 0
        for name, codec in self.codecs.items():
            if name not in sizes or codec not in sizes[name]:
                raise ValueError(f"Missing measured size for {name}/{codec}")
            total += sizes[name][codec]
        return total

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, "policy_id": POLICY_ID,
                "codecs": dict(self.codecs), "provenance": self.provenance}

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {"schema_version", "policy_id", "codecs", "provenance"}:
            raise ValueError("Unexpected precision map fields")
        if data["schema_version"] != SCHEMA_VERSION or data["policy_id"] != POLICY_ID:
            raise ValueError("Unsupported precision map version or policy")
        if not isinstance(data["codecs"], dict):
            raise ValueError("Precision map codecs must be an object")
        return cls(data["codecs"], data["provenance"])

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))


def _measured(report):
    tensors = report.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ValueError("Calibration report contains no tensors")
    if not report.get("sensitivity_measured"):
        raise ValueError("Selection requires a calibration report with measured sensitivity")
    rows = {}
    for entry in tensors:
        name = _text(entry.get("name"), "tensor name")
        for key in ("dense_bytes", "packed_bytes"):
            size = entry.get(key)
            if type(size) is not int or size <= 0:
                raise ValueError(f"Calibration report lacks a valid {key} for {name}")
        sensitivity = entry.get("sensitivity", {}).get("rmse")
        if not isinstance(sensitivity, (int, float)) or not math.isfinite(sensitivity) or sensitivity < 0:
            raise ValueError(f"Calibration report lacks a finite sensitivity for {name}")
        rows[name] = {"dense": entry["dense_bytes"], "packed": entry["packed_bytes"],
                      "sensitivity": float(sensitivity)}
    return rows


def select_precision(report, budget_bytes):
    """Choose codecs under a byte budget, keeping the costliest tensors dense.

    Every tensor starts packed, which is the cheapest configuration. Whatever
    budget remains promotes tensors to dense in order of sensitivity per extra
    byte, so the bytes spent buy the most avoided logit error.

    The selection is greedy over a ratio: with a binary choice per tensor this
    is a heuristic, not an optimum, and the estimate assumes errors add, which
    they do not. It ranks; it does not certify quality.
    """
    if type(budget_bytes) is not int or budget_bytes < 0:
        raise ValueError("budget_bytes must be a non-negative integer")
    rows = _measured(report)
    packed_total = sum(row["packed"] for row in rows.values())
    if packed_total > budget_bytes:
        raise ValueError(f"Budget {budget_bytes} is below {packed_total} bytes with every tensor packed")
    codecs = {name: "q4" for name in rows}
    remaining = budget_bytes - packed_total
    promotions = []
    # Ties break on the tensor name so the same report always plans the same map.
    order = sorted(rows.items(),
                   key=lambda item: (-item[1]["sensitivity"] / max(item[1]["dense"] - item[1]["packed"], 1),
                                     item[0]))
    for name, row in order:
        extra = row["dense"] - row["packed"]
        if extra <= 0:
            # A codec that does not shrink this tensor: dense costs nothing.
            codecs[name] = "f32"
            promotions.append({"tensor": name, "extra_bytes": max(extra, 0),
                               "avoided_rmse": row["sensitivity"]})
            continue
        if extra > remaining or not row["sensitivity"]:
            continue
        codecs[name] = "f32"
        remaining -= extra
        promotions.append({"tensor": name, "extra_bytes": extra, "avoided_rmse": row["sensitivity"]})
    provenance = {
        "policy": POLICY_ID, "budget_bytes": budget_bytes,
        "packed_baseline_bytes": packed_total,
        "planned_bytes": budget_bytes - remaining,
        "unused_budget_bytes": remaining,
        "checkpoint": report.get("checkpoint"),
        "calibration_tokens": report.get("tokens"),
        "group_size": report.get("group_size"),
        "promotions": promotions,
        "estimated_avoided_rmse_sum": sum(item["avoided_rmse"] for item in promotions),
        "estimate_scope": ("sum of individually measured logit RMSE; errors do not add, "
                           "so this ranks plans and does not predict combined quality"),
        "quality_measured": False,
    }
    return PrecisionMap(codecs, provenance)
