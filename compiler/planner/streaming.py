"""Where a ModelGraph could be evaluated a tile of rows at a time.

Analysis only. Nothing here executes anything; it names the op ranges whose
interior activations never need all their rows at once, and sizes the memory
requests that would follow. The executor keeps evaluating whole tensors.

The cut rule is a property of compiler/model_ir.py, not a guess. Read along the
token axis: Embedding gathers one weight row per token, RMSNorm reduces over the
features of its own row, MatMul reduces over K inside its own row, RoPE rotates
lanes inside its own row, Add and SwiGLU are elementwise. Every one of them
computes row t from row t alone. CausalAttention does not: its row t reads rows
0..t of key and value, so no tile of its output exists before the tile of key
and value rows below it. Attention therefore ends a region, and a region never
contains one.

A second condition is easy to lose: an operand that is *not* row-aligned — the
weight of a MatMul, the vector of an RMSNorm — must be a whole tensor, never a
tile. Only graph constants and graph inputs qualify here, which keeps the rule
checkable instead of hopeful.

A third one bites in practice. Row-independent does not mean position-blind:
RoPE reads no other row and still needs to know *which* row it is rotating. A
driver must carry the absolute row offset of each tile, which is why a region
publishes row_slices() and why compiler/graph_eval.evaluate_op takes row_offset.
Forgetting it rotates every tile as if the sequence restarted, and the byte
comparison in the regression suite is what catches that.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..model_ir import ModelGraph, OpKind, _integer, _name, _sequence
from ..model_lowering import derive_activation_requests
from .memory import MemoryPlanner, MemoryRequest


STREAMING_AXIS = 0
# Which operands advance with the streaming axis. Everything not listed here is
# read whole: a weight matrix, a norm vector, a codebook.
_ROW_ALIGNED_OPERANDS = {
    OpKind.MATMUL: (0,),
    OpKind.EMBEDDING: (0,),
    OpKind.RMSNORM: (0,),
    OpKind.ROPE: (0,),
    OpKind.SWIGLU: (0, 1),
    OpKind.ADD: (0, 1),
}
# Spelled out rather than derived, so adding an operator to the IR forces a
# decision here instead of silently inheriting "streamable".
_BARRIER_OPS = (OpKind.CAUSAL_ATTENTION,)


@dataclass(frozen=True)
class StreamingRegion:
    """A half-open run of ops [start, end) that a row tile can walk through.

    boundary_tensors cross the frontier and stay whole: they enter the region
    from outside or leave it for a later consumer. interior_tensors are born
    and die inside, so a tile buffer of tile_rows rows is enough for them.
    live_tile_buffers is how many of those tiles exist at the same moment, which
    is the number that decides whether streaming actually pays.
    """
    name: str
    start: int
    end: int
    axis: int
    rows: int
    tile_rows: int
    boundary_tensors: tuple[str, ...]
    interior_tensors: tuple[str, ...]
    constant_tensors: tuple[str, ...]
    live_tile_buffers: int

    def __post_init__(self):
        _name(self.name, "region name")
        _integer(self.start, "region start")
        _integer(self.end, "region end", 1)
        if self.end <= self.start:
            raise ValueError("a region must span at least one operation")
        if self.axis != STREAMING_AXIS:
            raise ValueError("only the token axis is supported for streaming")
        _integer(self.rows, "region rows", 1)
        _integer(self.tile_rows, "region tile_rows", 1)
        if self.tile_rows > self.rows:
            raise ValueError("tile_rows cannot exceed the streamed extent")
        for field_name in ("boundary_tensors", "interior_tensors", "constant_tensors"):
            values = _sequence(getattr(self, field_name), field_name)
            for item in values:
                _name(item, field_name)
            object.__setattr__(self, field_name, values)
        _integer(self.live_tile_buffers, "live_tile_buffers")
        if self.live_tile_buffers > len(self.interior_tensors):
            raise ValueError("live_tile_buffers cannot exceed the interior tensor count")

    @property
    def tiles(self):
        return -(-self.rows // self.tile_rows)

    def retile(self, tile_rows):
        """Same region, different tile height; detection does not choose one."""
        return StreamingRegion(self.name, self.start, self.end, self.axis, self.rows,
                               tile_rows, self.boundary_tensors, self.interior_tensors,
                               self.constant_tensors, self.live_tile_buffers)

    def row_slices(self):
        """The half-open row intervals a driver would walk, in order."""
        return tuple((start, min(start + self.tile_rows, self.rows))
                     for start in range(0, self.rows, self.tile_rows))

    def to_dict(self):
        return {"name": self.name, "start": self.start, "end": self.end, "axis": self.axis,
                "rows": self.rows, "tile_rows": self.tile_rows, "tiles": self.tiles,
                "boundary_tensors": list(self.boundary_tensors),
                "interior_tensors": list(self.interior_tensors),
                "constant_tensors": list(self.constant_tensors),
                "live_tile_buffers": self.live_tile_buffers}


def row_aligned_operands(op):
    """Operand positions that advance with the streaming axis, for a driver."""
    if op.kind not in _ROW_ALIGNED_OPERANDS:
        raise ValueError(f"{op.kind.value} has no row-aligned operand contract")
    return _ROW_ALIGNED_OPERANDS[op.kind]


def _streamable(op, graph_rows, tensors, whole_tensors):
    """Row-independence plus the whole-operand condition, per operation."""
    if op.kind in _BARRIER_OPS or op.kind not in _ROW_ALIGNED_OPERANDS:
        return False
    aligned = _ROW_ALIGNED_OPERANDS[op.kind]
    for position, name in enumerate(op.inputs):
        tensor = tensors[name]
        if position in aligned:
            if tensor.shape[STREAMING_AXIS] != graph_rows:
                return False
        elif name not in whole_tensors:
            return False
    return tensors[op.outputs[0]].shape[STREAMING_AXIS] == graph_rows


def _graph_rows(graph, tensors):
    """The streamed extent: the token count every activation shares."""
    rows = {tensors[name].shape[STREAMING_AXIS] for name in graph.inputs}
    if len(rows) != 1:
        raise ValueError("streaming needs a single token extent across the graph inputs")
    return rows.pop()


def detect_streaming_regions(graph, *, tile_rows=None):
    """Find maximal runs of row-independent ops with at least one interior tensor.

    tile_rows is left at the full extent unless a caller names one: detection
    reports where streaming is possible, not how finely to cut. A run whose
    every produced tensor escapes it buys no buffer and is not reported.
    """
    if not isinstance(graph, ModelGraph):
        raise ValueError("graph must be a ModelGraph")
    graph.validate()
    tensors = {tensor.name: tensor for tensor in graph.tensors}
    rows = _graph_rows(graph, tensors)
    if tile_rows is not None:
        _integer(tile_rows, "tile_rows", 1)
        if tile_rows > rows:
            raise ValueError("tile_rows cannot exceed the streamed extent")
    whole = set(graph.constants) | set(graph.inputs)
    consumers = {}
    for index, op in enumerate(graph.ops):
        for name in op.inputs:
            consumers.setdefault(name, []).append(index)
    escapes = set(graph.outputs)
    regions, start = [], None
    for index in range(len(graph.ops) + 1):
        runnable = (index < len(graph.ops)
                    and _streamable(graph.ops[index], rows, tensors, whole))
        if runnable and start is None:
            start = index
        if runnable or start is None:
            continue
        region = _describe(graph, tensors, consumers, escapes, rows, start, index)
        if region is not None:
            regions.append(region if tile_rows is None else region.retile(tile_rows))
        start = None
    return tuple(regions)


def _describe(graph, tensors, consumers, escapes, rows, start, end):
    produced = {}
    for index in range(start, end):
        produced[graph.ops[index].outputs[0]] = index
    interior, boundary = [], set()
    for name, index in produced.items():
        uses = consumers.get(name, ())
        if name in escapes or any(use < start or use >= end for use in uses):
            boundary.add(name)
        else:
            interior.append(name)
    constants = set()
    for index in range(start, end):
        op = graph.ops[index]
        aligned = _ROW_ALIGNED_OPERANDS[op.kind]
        for position, name in enumerate(op.inputs):
            if name in produced:
                continue
            (boundary if position in aligned else constants).add(name)
    if not interior:
        return None
    return StreamingRegion(
        name=f"{graph.name}.region.{start}_{end}", start=start, end=end, axis=STREAMING_AXIS,
        rows=rows, tile_rows=rows, boundary_tensors=tuple(sorted(boundary)),
        interior_tensors=tuple(sorted(interior)), constant_tensors=tuple(sorted(constants)),
        live_tile_buffers=_live_tile_buffers(graph, produced, set(interior), start, end))


def _live_tile_buffers(graph, produced, interior, start, end):
    """Peak count of interior tiles alive together, on the serial op order.

    An op reads and writes in the same event, so a value stays live through its
    last consumer's event: a producer and that consumer cannot share a buffer.
    """
    last_use = {}
    for index in range(start, end):
        for name in graph.ops[index].inputs:
            if name in interior:
                last_use[name] = index
    peak = 0
    for index in range(start, end):
        live = sum(1 for name in interior
                   if produced[name] <= index <= last_use.get(name, produced[name]))
        peak = max(peak, live)
    return peak


def derive_streamed_activation_requests(graph, regions, tile_rows):
    """Activation requests with every region interior sized to one row tile.

    The lifetimes are the ones derive_activation_requests already computes; the
    only change is the size of a buffer that no longer has to hold every row.
    Boundary tensors and graph outputs keep their full extent, which is exactly
    why the saving is bounded by how much of the arena they occupy.
    """
    if not isinstance(graph, ModelGraph):
        raise ValueError("graph must be a ModelGraph")
    _integer(tile_rows, "tile_rows", 1)
    tensors = {tensor.name: tensor for tensor in graph.tensors}
    interior = {}
    for region in regions:
        if not isinstance(region, StreamingRegion):
            raise ValueError("regions must contain StreamingRegion objects")
        if tile_rows > region.rows:
            raise ValueError("tile_rows cannot exceed the streamed extent")
        for name in region.interior_tensors:
            if name in interior:
                raise ValueError(f"{name} is interior to more than one region")
            interior[name] = region
    requests = []
    for request in derive_activation_requests(graph):
        region = interior.get(request.name)
        if region is None:
            requests.append(request)
            continue
        tensor = tensors[request.name]
        rows = tensor.shape[STREAMING_AXIS]
        if request.size_bytes % rows:
            raise ValueError(f"{request.name}: rows do not divide its storage evenly")
        requests.append(MemoryRequest(request.name, request.size_bytes // rows * tile_rows,
                                      request.start, request.end,
                                      alignment=request.alignment, tier=request.tier))
    return requests


def streaming_summary(graph, regions, tile_rows) -> Mapping:
    """What the regions would save, planned by the same planner the runtime uses.

    The arena peak is the number that matters and the one to distrust: the
    planner already reuses dead activations, so shrinking a buffer only helps
    where it was the tall one. Requested bytes are reported beside it because
    the two disagree, and the gap is the point.
    """
    baseline = derive_activation_requests(graph)
    streamed = derive_streamed_activation_requests(graph, regions, tile_rows)
    before = sum(request.size_bytes for request in baseline)
    after = sum(request.size_bytes for request in streamed)
    peak_before, peak_after = (_arena_peak(requests) for requests in (baseline, streamed))
    return MappingProxyType({
        "ops": len(graph.ops), "regions": len(regions), "tile_rows": tile_rows,
        "streamed_ops": sum(region.end - region.start for region in regions),
        "interior_tensors": sum(len(region.interior_tensors) for region in regions),
        "live_tile_buffers": max((region.live_tile_buffers for region in regions), default=0),
        "activation_request_bytes": before, "streamed_request_bytes": after,
        "arena_peak_bytes": peak_before, "streamed_arena_peak_bytes": peak_after,
        "saved_bytes": peak_before - peak_after,
        "saved_fraction": 0.0 if not peak_before else (peak_before - peak_after) / peak_before,
        "largest_unstreamable_bytes": max((request.size_bytes for request in streamed), default=0),
    })


def _arena_peak(requests):
    tiers = {request.tier for request in requests}
    budget = sum(request.size_bytes for request in requests)
    plan = MemoryPlanner().plan(requests, {tier: budget for tier in tiers})
    return max(plan.peak_bytes.values(), default=0)
