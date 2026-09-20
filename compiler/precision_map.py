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
POLICY_ID = "GREEDY_SENSITIVITY_PER_BYTE_V2"
CODECS = ("q4", "q8", "f32")
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


def _rmse(value, label):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Calibration report lacks a finite sensitivity for {label}")
    return float(value)


def _options(name, entry):
    """Codec choices for one tensor, cheapest first, dominated ones removed.

    Dense is always available at its own size with zero measured error, since
    it is the reference every sensitivity was measured against.
    """
    dense_bytes = entry.get("dense_bytes")
    if type(dense_bytes) is not int or dense_bytes <= 0:
        raise ValueError(f"Calibration report lacks a valid dense_bytes for {name}")
    measured = entry.get("codecs")
    if measured is None and "packed_bytes" in entry:
        # A report from the single-codec calibration still plans correctly.
        measured = {"q4": {"packed_bytes": entry["packed_bytes"],
                           "sensitivity": entry.get("sensitivity", {})}}
    if not isinstance(measured, dict) or not measured:
        raise ValueError(f"Calibration report lacks measured codecs for {name}")
    choices = []
    for codec, item in measured.items():
        if codec not in CODECS or codec == "f32":
            raise ValueError(f"Unsupported measured codec for {name}: {codec!r}")
        size = item.get("packed_bytes")
        if type(size) is not int or size <= 0:
            raise ValueError(f"Calibration report lacks a valid packed_bytes for {name}/{codec}")
        choices.append({"codec": codec, "bytes": size,
                        "sensitivity": _rmse(item.get("sensitivity", {}).get("rmse"), f"{name}/{codec}")})
    choices.append({"codec": "f32", "bytes": dense_bytes, "sensitivity": 0.0})
    choices.sort(key=lambda item: (item["bytes"], item["sensitivity"], item["codec"]))
    # Drop a choice that costs more and is not more accurate than a cheaper one.
    frontier, best = [], math.inf
    for choice in choices:
        if choice["sensitivity"] < best:
            frontier.append(choice)
            best = choice["sensitivity"]
    return frontier


def _measured(report):
    tensors = report.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ValueError("Calibration report contains no tensors")
    if not report.get("sensitivity_measured"):
        raise ValueError("Selection requires a calibration report with measured sensitivity")
    return {_text(entry.get("name"), "tensor name"): _options(_text(entry.get("name"), "tensor name"), entry)
            for entry in tensors}


def select_precision(report, budget_bytes):
    """Choose a codec per tensor under a byte budget, from measured error.

    Every tensor starts at its cheapest measured codec. While budget remains,
    the upgrade with the best avoided error per extra byte is applied, anywhere
    in the model; a tensor can climb more than one step, and a step that buys
    no accuracy is never taken.

    The selection is greedy over a ratio, so it is a heuristic, not an optimum.
    The estimate also assumes errors add, which they do not: it ranks plans, it
    does not certify quality.
    """
    if type(budget_bytes) is not int or budget_bytes < 0:
        raise ValueError("budget_bytes must be a non-negative integer")
    rows = _measured(report)
    baseline = sum(options[0]["bytes"] for options in rows.values())
    if baseline > budget_bytes:
        raise ValueError(f"Budget {budget_bytes} is below {baseline} bytes at the cheapest codecs")
    chosen = {name: 0 for name in rows}
    remaining = budget_bytes - baseline
    upgrades = []
    while True:
        best = None
        for name in sorted(rows):
            options = rows[name]
            current = options[chosen[name]]
            for index in range(chosen[name] + 1, len(options)):
                candidate = options[index]
                extra = candidate["bytes"] - current["bytes"]
                avoided = current["sensitivity"] - candidate["sensitivity"]
                if avoided <= 0 or extra > remaining:
                    continue
                # A free upgrade is always worth taking; otherwise rank by the
                # error it avoids per extra byte, breaking ties by name.
                ratio = math.inf if extra <= 0 else avoided / extra
                # Tensors are visited in name order and steps cheapest first,
                # so a strict comparison keeps the first of any tie.
                key = (ratio, -extra)
                if best is None or key > best[0]:
                    best = (key, name, index, extra, avoided, candidate["codec"])
        if best is None:
            break
        _, name, index, extra, avoided, codec = best
        chosen[name] = index
        remaining -= max(extra, 0)
        upgrades.append({"tensor": name, "codec": codec, "extra_bytes": max(extra, 0),
                         "avoided_rmse": avoided})
    codecs = {name: rows[name][index]["codec"] for name, index in chosen.items()}
    provenance = {
        "policy": POLICY_ID, "budget_bytes": budget_bytes,
        "cheapest_baseline_bytes": baseline,
        "planned_bytes": budget_bytes - remaining,
        "unused_budget_bytes": remaining,
        "checkpoint": report.get("checkpoint"),
        "calibration_tokens": report.get("tokens"),
        "group_size": report.get("group_size"),
        "measured_codecs": report.get("measured_codecs", ["q4"]),
        "codec_counts": {codec: sum(value == codec for value in codecs.values()) for codec in CODECS},
        "upgrades": upgrades,
        "estimated_avoided_rmse_sum": sum(item["avoided_rmse"] for item in upgrades),
        "estimate_scope": ("sum of individually measured logit RMSE; errors do not add, "
                           "so this ranks plans and does not predict combined quality"),
        "quality_measured": False,
    }
    return PrecisionMap(codecs, provenance)
