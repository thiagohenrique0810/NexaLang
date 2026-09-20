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

from compiler.codec_speed import DECODE_RATE_POLICY_ID, decode_ns, rate_table

SCHEMA_VERSION = 1
# The cost a plan optimizes is part of the policy: the same sensitivities rank
# differently against payload bytes and against bytes the file actually holds.
POLICY_IDS = {"payload": "GREEDY_SENSITIVITY_PER_BYTE_V2",
              "physical": "GREEDY_SENSITIVITY_PER_PHYSICAL_BYTE_V3"}
POLICY_ID = POLICY_IDS["payload"]
COSTS = ("payload", "physical")
CODECS = ("q2", "q3", "q4", "q8", "f16", "f32")
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
    policy_id: str = POLICY_ID

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
        if self.policy_id not in POLICY_IDS.values():
            raise ValueError(f"Unsupported precision map policy: {self.policy_id!r}")
        object.__setattr__(self, "codecs", MappingProxyType(dict(sorted(codecs.items()))))
        object.__setattr__(self, "provenance", json.loads(json.dumps(self.provenance, sort_keys=True)))

    @property
    def dense_tensors(self):
        return tuple(name for name, codec in self.codecs.items() if codec == "f32")

    @property
    def cost_basis(self):
        """Which byte count this map was optimized against."""
        for cost, policy in POLICY_IDS.items():
            if policy == self.policy_id:
                return cost
        raise ValueError("Unsupported precision map policy")

    def bytes_for(self, sizes):
        """Total stored bytes, given {tensor: {"q4": n, "f32": n}} from calibration."""
        total = 0
        for name, codec in self.codecs.items():
            if name not in sizes or codec not in sizes[name]:
                raise ValueError(f"Missing measured size for {name}/{codec}")
            total += sizes[name][codec]
        return total

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, "policy_id": self.policy_id,
                "codecs": dict(self.codecs), "provenance": self.provenance}

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {"schema_version", "policy_id", "codecs", "provenance"}:
            raise ValueError("Unexpected precision map fields")
        if data["schema_version"] != SCHEMA_VERSION or data["policy_id"] not in POLICY_IDS.values():
            raise ValueError("Unsupported precision map version or policy")
        if not isinstance(data["codecs"], dict):
            raise ValueError("Precision map codecs must be an object")
        return cls(data["codecs"], data["provenance"], data["policy_id"])

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))


def _rmse(value, label):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Calibration report lacks a finite sensitivity for {label}")
    return float(value)


# What each cost basis reads from the report, per tensor and per codec.
_COST_FIELDS = {"payload": ("dense_bytes", "packed_bytes"),
                "physical": ("dense_physical_bytes", "physical_bytes")}


def _options(name, entry, cost="payload", rates=None):
    """Codec choices for one tensor, cheapest first, dominated ones removed.

    Dense is always available at its own size with zero measured error, since
    it is the reference every sensitivity was measured against.

    `cost` picks the byte count to optimize. "payload" counts the codec's own
    bytes; "physical" counts what the tensor file actually holds, including the
    container header, the per-block metadata and its checksums. The second is
    the number that has to fit on a device, and the two do not always rank the
    codecs the same way.

    `rates` are measured nanoseconds per stored byte. With them every choice
    also carries its predicted decode time, and dominance becomes a question in
    two dimensions: a choice survives unless some cheaper one is at least as
    accurate *and* at least as fast. Without them the time of every choice is
    zero, the second half of the test is vacuous, and the surviving frontier is
    exactly the one this function has always produced.
    """
    dense_field, packed_field = _COST_FIELDS[cost]
    # Decode reads the codec's own bytes whatever the cost basis counts: the
    # container header around them is checked, not dequantized.
    payload_field = _COST_FIELDS["payload"][1]
    dense_bytes = entry.get(dense_field)
    if type(dense_bytes) is not int or dense_bytes <= 0:
        raise ValueError(f"Calibration report lacks a valid {dense_field} for {name}")
    measured = entry.get("codecs")
    if measured is None and "packed_bytes" in entry:
        # A report from the single-codec calibration still plans correctly.
        measured = {"q4": {"packed_bytes": entry["packed_bytes"],
                           "physical_bytes": entry.get("physical_bytes"),
                           "sensitivity": entry.get("sensitivity", {})}}
    if not isinstance(measured, dict) or not measured:
        raise ValueError(f"Calibration report lacks measured codecs for {name}")
    choices = []
    for codec, item in measured.items():
        if codec not in CODECS or codec == "f32":
            raise ValueError(f"Unsupported measured codec for {name}: {codec!r}")
        # f32 is the reference itself and is appended below, never measured.
        size = item.get(packed_field)
        if type(size) is not int or size <= 0:
            raise ValueError(f"Calibration report lacks a valid {packed_field} for {name}/{codec}")
        choice = {"codec": codec, "bytes": size,
                  "sensitivity": _rmse(item.get("sensitivity", {}).get("rmse"), f"{name}/{codec}"),
                  "decode_ns": 0}
        if rates is not None:
            payload = item.get(payload_field)
            if type(payload) is not int or payload <= 0:
                raise ValueError(
                    f"A decode-time bound needs a valid {payload_field} for {name}/{codec}")
            choice["decode_ns"] = decode_ns(rates, codec, payload)
        choices.append(choice)
    dense = {"codec": "f32", "bytes": dense_bytes, "sensitivity": 0.0, "decode_ns": 0}
    if rates is not None:
        dense_payload = entry.get(_COST_FIELDS["payload"][0])
        if type(dense_payload) is not int or dense_payload <= 0:
            raise ValueError(f"A decode-time bound needs a valid dense_bytes for {name}")
        dense["decode_ns"] = decode_ns(rates, "f32", dense_payload)
    choices.append(dense)
    choices.sort(key=lambda item: (item["bytes"], item["sensitivity"], item["codec"]))
    # Drop a choice that costs more without being better on any axis measured.
    frontier = []
    for index, choice in enumerate(choices):
        if not any(earlier["sensitivity"] <= choice["sensitivity"]
                   and earlier["decode_ns"] <= choice["decode_ns"]
                   for earlier in choices[:index]):
            frontier.append(choice)
    return frontier


def _measured(report, cost="payload", rates=None):
    tensors = report.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ValueError("Calibration report contains no tensors")
    if not report.get("sensitivity_measured"):
        raise ValueError("Selection requires a calibration report with measured sensitivity")
    return {_text(entry.get("name"), "tensor name"):
            _options(_text(entry.get("name"), "tensor name"), entry, cost, rates)
            for entry in tensors}


def _decode_total(rows, chosen):
    """Predicted nanoseconds to decode every tensor once, under this plan."""
    return sum(rows[name][index]["decode_ns"] for name, index in chosen.items())


def _estimated_rmse(rows, chosen):
    """Root-sum-square of the chosen codecs' measured errors.

    Errors from different tensors are combined as if independent, which is an
    assumption, not a measurement: use it to compare plans, never as a quality
    figure for the model.
    """
    return math.sqrt(sum(rows[name][index]["sensitivity"] ** 2 for name, index in chosen.items()))


def select_precision(report, budget_bytes=None, *, max_rmse=None, max_decode_ns=None,
                     cost="payload"):
    """Choose a codec per tensor, bounded by bytes or by estimated error.

    Every tensor starts at its cheapest measured codec. The upgrade with the
    best avoided error per extra byte is applied repeatedly, anywhere in the
    model; a tensor can climb more than one step, and a step that buys no
    accuracy is never taken.

    With `budget_bytes`, upgrades stop when the budget runs out. With
    `max_rmse`, they stop as soon as the estimated error falls to the ceiling,
    which yields the cheapest plan this heuristic reaches for that quality.

    `max_decode_ns` adds a second, independent ceiling over the *measured*
    decode rate of each codec, in nanoseconds to decode the whole model once.
    It is a separate bound because bytes do not predict time: a codec that
    stores fewer bytes may take longer to unpack them, so the cheapest plan by
    bytes and the cheapest by time are different plans. The cheapest plan by
    bytes is made to fit the time ceiling first, spending the fewest extra
    bytes per nanosecond saved; only then does the error-per-byte climb run,
    and it refuses any step that would push the plan back over the ceiling.

    The selection is greedy over a ratio, so it is a heuristic, not an optimum,
    and the error estimate combines measurements that were taken one tensor at
    a time. It ranks plans; it does not certify quality.
    """
    if cost not in COSTS:
        raise ValueError(f"cost must be one of {COSTS}")
    if (budget_bytes is None) == (max_rmse is None):
        raise ValueError("Pass exactly one of budget_bytes or max_rmse")
    if budget_bytes is not None and (type(budget_bytes) is not int or budget_bytes < 0):
        raise ValueError("budget_bytes must be a non-negative integer")
    if max_rmse is not None and (not isinstance(max_rmse, (int, float)) or isinstance(max_rmse, bool)
                                 or not math.isfinite(max_rmse) or max_rmse < 0):
        raise ValueError("max_rmse must be a finite non-negative number")
    if max_decode_ns is not None and (type(max_decode_ns) is not int or max_decode_ns <= 0):
        raise ValueError("max_decode_ns must be a positive integer number of nanoseconds")
    rates = None if max_decode_ns is None else rate_table(report)
    rows = _measured(report, cost, rates)
    baseline = sum(options[0]["bytes"] for options in rows.values())
    if budget_bytes is not None and baseline > budget_bytes:
        raise ValueError(f"Budget {budget_bytes} is below {baseline} bytes at the cheapest codecs")
    chosen = {name: 0 for name in rows}
    remaining = math.inf if budget_bytes is None else budget_bytes - baseline
    upgrades = []
    decode_baseline = _decode_total(rows, chosen)
    decode_upgrades = []
    while max_decode_ns is not None and _decode_total(rows, chosen) > max_decode_ns:
        # Feasibility first: buy time with the fewest extra bytes per
        # nanosecond saved. Error is not consulted here, because a plan that
        # misses the ceiling is not a plan at all.
        best = None
        for name in sorted(rows):
            options = rows[name]
            current = options[chosen[name]]
            for index in range(chosen[name] + 1, len(options)):
                candidate = options[index]
                extra = candidate["bytes"] - current["bytes"]
                saved = current["decode_ns"] - candidate["decode_ns"]
                if saved <= 0 or extra > remaining:
                    continue
                key = (math.inf if extra <= 0 else saved / extra, -extra)
                if best is None or key > best[0]:
                    best = (key, name, index, extra, saved, candidate["codec"])
        if best is None:
            raise ValueError(
                f"No plan within {max_decode_ns} ns: the cheapest codecs decode in "
                f"{_decode_total(rows, chosen)} ns and no affordable upgrade is faster")
        _, name, index, extra, saved, codec = best
        chosen[name] = index
        remaining -= max(extra, 0) if remaining != math.inf else 0
        decode_upgrades.append({"tensor": name, "codec": codec, "extra_bytes": max(extra, 0),
                                "saved_decode_ns": saved})
    while True:
        if max_rmse is not None and _estimated_rmse(rows, chosen) <= max_rmse:
            break
        best = None
        decode_now = _decode_total(rows, chosen)
        for name in sorted(rows):
            options = rows[name]
            current = options[chosen[name]]
            for index in range(chosen[name] + 1, len(options)):
                candidate = options[index]
                extra = candidate["bytes"] - current["bytes"]
                avoided = current["sensitivity"] - candidate["sensitivity"]
                if avoided <= 0 or extra > remaining:
                    continue
                # An upgrade buys accuracy with bytes; it may also cost time,
                # and the time ceiling is not negotiable by the error ranking.
                if (max_decode_ns is not None and
                        decode_now - current["decode_ns"] + candidate["decode_ns"] > max_decode_ns):
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
        remaining -= max(extra, 0) if remaining != math.inf else 0
        upgrades.append({"tensor": name, "codec": codec, "extra_bytes": max(extra, 0),
                         "avoided_rmse": avoided})
    codecs = {name: rows[name][index]["codec"] for name, index in chosen.items()}
    planned = sum(rows[name][index]["bytes"] for name, index in chosen.items())
    estimate = _estimated_rmse(rows, chosen)
    provenance = {
        "policy": POLICY_IDS[cost], "cost_basis": cost,
        "cost_scope": ("codec payload only" if cost == "payload" else
                       "bytes the tensor file holds: container header, per-block metadata and checksums"),
        "budget_bytes": budget_bytes, "max_rmse": max_rmse,
        "bound": "bytes" if budget_bytes is not None else "estimated_rmse",
        "cheapest_baseline_bytes": baseline,
        "planned_bytes": planned,
        "unused_budget_bytes": (budget_bytes - planned) if budget_bytes is not None else None,
        "estimated_rmse": estimate,
        "meets_max_rmse": None if max_rmse is None else estimate <= max_rmse,
        "checkpoint": report.get("checkpoint"),
        "calibration_tokens": report.get("tokens"),
        "group_size": report.get("group_size"),
        "measured_codecs": report.get("measured_codecs", ["q4"]),
        "codec_counts": {codec: sum(value == codec for value in codecs.values()) for codec in CODECS},
        "upgrades": upgrades,
        "estimated_avoided_rmse_sum": sum(item["avoided_rmse"] for item in upgrades),
        "estimate_scope": ("individually measured logit RMSE, combined as independent errors; "
                           "this ranks plans and does not predict combined quality"),
        "quality_measured": False,
    }
    if max_decode_ns is not None:
        # Written only under a time bound, so a plan that never asked about
        # time keeps exactly the provenance it has always had.
        planned_decode = _decode_total(rows, chosen)
        provenance["decode_time"] = {
            "policy": DECODE_RATE_POLICY_ID,
            "max_decode_ns": max_decode_ns,
            "cheapest_baseline_decode_ns": decode_baseline,
            "planned_decode_ns": planned_decode,
            "unused_decode_ns": max_decode_ns - planned_decode,
            "meets_max_decode_ns": planned_decode <= max_decode_ns,
            "ns_per_byte": {codec: rates[codec] for codec in sorted(rates)},
            "feasibility_upgrades": decode_upgrades,
            "scope": ("predicted from the measured rate of the block matmul the executor runs, "
                      "times the codec payload of each tensor: one decode of every weight, "
                      "on the host that measured it"),
            "not_predicted": ("reads from disk, attention, activations and Python orchestration; "
                              "this is not end-to-end latency"),
        }
    return PrecisionMap(codecs, provenance, POLICY_IDS[cost])
