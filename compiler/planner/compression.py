"""Compression planning over the axes this repository can actually measure.

M6.03 asks for a planner over codec, sparsity and low rank "without assuming
speedup". Two of those three axes do not exist here: there is no sparse codec
in the pack format, no sparse kernel in the executor and no manifest field to
record one, and the same is true of a low-rank factorization. A planner that
accepted `sparsity=0.5` and quietly produced a codec-only plan would be worse
than no planner, because the caller would believe the axis was considered.

So this planner does two things and says so. It delegates the codec decision,
unchanged, to `select_precision`, and it carries a provenance block naming
every axis it did *not* consider and why. Passing an axis it cannot plan is an
error that names the axis; it is never accepted and ignored.

The third part is the axis that stopped being an assumption. "Without assuming
speedup" used to mean the planner said nothing about speed at all. The decode
rate of every codec is now measured (`compiler/codec_speed.py`), so a plan can
be bounded by time as well as by bytes — and the two bounds disagree, because
the smallest codec on this host is not the fastest one.
"""
from __future__ import annotations

import json

from compiler.codec_speed import DECODE_RATE_POLICY_ID, rate_table
from compiler.precision_map import COSTS, select_precision

PLANNER_POLICY_ID = "COMPRESSION_CODEC_ONLY_WITH_MEASURED_DECODE_TIME_V1"
# Each axis M6.03 names, and the reason this planner cannot decide it. The
# reasons are checked against the runtime by the regression suite: if a sparse
# codec ever lands in PACKED_CODECS, the claim below stops being true and the
# test that reads it fails.
UNSUPPORTED_AXES = {
    "sparsity": ("unsupported: no codec in runtime/nexapack/bundle.py PACKED_CODECS, "
                 "no kernel in runtime/nexapack/transformer.py _PACKED_KERNELS, "
                 "no manifest field to record one (M6.06)"),
    "low_rank": ("unsupported: no factorization in the importer, the manifest or the "
                 "graph, and no equivalence check for one (M6.07)"),
    "peak_vram": "not measurable: this repository has no GPU execution path",
    "transfer_bytes": "not measurable: this repository has no host-to-device transfer to time",
    "energy": "not measurable: no energy counter is read on this host",
}
CONSIDERED_AXES = ("codec", "decode_time")


class CompressionPlan:
    """A codec plan plus the record of what was left out of it.

    `to_json` is the precision map and nothing else, byte for byte, so the plan
    feeds `nexa_convert.py --precision-map` exactly as `select_precision` did.
    The axis provenance lives beside it in `to_report_json`, because a file that
    a converter reads must not grow fields the converter does not understand.
    """
    def __init__(self, precision_map, axes, constraints):
        self.precision_map = precision_map
        self.axes = axes
        self.constraints = constraints

    @property
    def codecs(self):
        return self.precision_map.codecs

    @property
    def provenance(self):
        return self.precision_map.provenance

    def to_json(self, *, indent=2):
        return self.precision_map.to_json(indent=indent)

    def to_report_dict(self):
        return {"planner_policy_id": PLANNER_POLICY_ID, "constraints": self.constraints,
                "axes": self.axes, "precision_map": self.precision_map.to_dict()}

    def to_report_json(self, *, indent=2):
        return json.dumps(self.to_report_dict(), indent=indent, sort_keys=True, allow_nan=False)


class CompressionPlanner:
    """Plan compression from a calibration report under explicitly declared constraints.

    Constraints are keyword-only and named after what they bound:
    `max_bytes` or `max_rmse` (exactly one, as the precision map requires) and
    an optional `max_decode_ns`. Any other keyword is an axis this planner
    cannot decide, and it is refused by name rather than dropped.
    """
    def __init__(self, report, *, max_bytes=None, max_rmse=None, max_decode_ns=None,
                 cost="payload", **axes):
        if not isinstance(report, dict):
            raise ValueError("A compression plan needs a calibration report object")
        refused = sorted(axes)
        if refused:
            reasons = "; ".join(
                f"{axis}: {UNSUPPORTED_AXES.get(axis, 'unknown axis: this planner has no such constraint')}"
                for axis in refused)
            raise ValueError(f"CompressionPlanner cannot plan these axes: {reasons}")
        if cost not in COSTS:
            raise ValueError(f"cost must be one of {COSTS}")
        if (max_bytes is None) == (max_rmse is None):
            raise ValueError("Pass exactly one of max_bytes or max_rmse")
        self.report = report
        self.max_bytes = max_bytes
        self.max_rmse = max_rmse
        self.max_decode_ns = max_decode_ns
        self.cost = cost

    def axis_provenance(self):
        """Every axis considered, every axis refused, and the reason for each refusal."""
        considered = {
            "codec": (f"planned by compiler/precision_map.py against measured {self.cost} bytes "
                      "and measured per-tensor logit sensitivity"),
            "decode_time": ("bounded by measured decode nanoseconds per stored byte"
                            if self.max_decode_ns is not None else
                            "measured but not bounded: no max_decode_ns was declared"),
        }
        return {"considered": considered, "not_considered": dict(UNSUPPORTED_AXES),
                "decode_rate_policy_id": DECODE_RATE_POLICY_ID,
                "decode_rate_available": self._rates_available()}

    def _rates_available(self):
        try:
            rate_table(self.report)
        except ValueError:
            return False
        return True

    def plan(self):
        precision = select_precision(self.report, self.max_bytes, max_rmse=self.max_rmse,
                                     max_decode_ns=self.max_decode_ns, cost=self.cost)
        constraints = {"max_bytes": self.max_bytes, "max_rmse": self.max_rmse,
                       "max_decode_ns": self.max_decode_ns, "cost": self.cost}
        return CompressionPlan(precision, self.axis_provenance(), constraints)
