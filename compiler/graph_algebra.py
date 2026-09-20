"""Graph rewrites with a pass manager that refuses to trust them.

Every rewrite declares what it promises: ``exact`` means the rewritten graph
must produce byte-identical float32 outputs, ``tolerance`` means it must stay
within a published bound. ``run_passes`` measures that promise with an
independent evaluator (compiler/graph_eval.py) before accepting a rewrite, so a
pass that matches a site it should not have matched fails here instead of in a
model. A pass that rewrites nothing is not verified, because there is nothing
to verify; the report still records the zero.

A fold can need a constant that no checkpoint contains — the product of two
projection matrices, for instance. The report publishes the recipe as a
ConstantDerivation instead of inventing the numbers, so whoever executes the
rewritten graph knows exactly which tensor to materialize and from what.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from types import MappingProxyType
from typing import Mapping

from .graph_eval import evaluate_graph, matmul, random_bindings, tensor_bytes
from .model_ir import (DType, ModelGraph, OpKind, TensorDesc, _keys,
                       _load_json, _name, _positive_real, _sequence)


REWRITE_SCHEMA_VERSION = 1
EXACT = "exact"
TOLERANCE = "tolerance"
_EXACTNESS = (EXACT, TOLERANCE)
# A rewrite loop that keeps matching its own output is a bug, not a deep graph.
MAX_REWRITE_ROUNDS = 64


class RewriteError(ValueError):
    """A rewrite broke its own contract; the graph before it stays authoritative."""


@dataclass(frozen=True)
class ConstantDerivation:
    """How to build a constant the rewritten graph needs and no bundle holds."""
    name: str
    recipe: str
    sources: tuple[str, ...]
    shape: tuple[int, ...]

    def __post_init__(self):
        _name(self.name, "derived constant name")
        if self.recipe != "matmul":
            raise ValueError("the only supported derivation recipe is matmul")
        object.__setattr__(self, "sources", _sequence(self.sources, "derivation sources"))
        object.__setattr__(self, "shape", _sequence(self.shape, "derivation shape"))
        if len(self.sources) != 2 or len(self.shape) != 2:
            raise ValueError("a matmul derivation takes two sources and a rank-2 shape")

    def materialize(self, bindings):
        """Compute the constant from already-bound (or earlier derived) tensors."""
        left, right = (bindings[name] for name in self.sources)
        result = matmul(left, right)
        if (len(result), len(result[0])) != tuple(self.shape):
            raise ValueError(f"{self.name}: derived shape does not match the declaration")
        return result

    def to_dict(self):
        return {"name": self.name, "recipe": self.recipe,
                "sources": list(self.sources), "shape": list(self.shape)}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {"name", "recipe", "sources", "shape"}, "ConstantDerivation")
        return cls(**data)


@dataclass(frozen=True)
class RewriteSite:
    """One matched location: which operations it touched and why it qualified."""
    ops: tuple[str, ...]
    detail: Mapping = None

    def __post_init__(self):
        object.__setattr__(self, "ops", _sequence(self.ops, "site ops"))
        if not self.ops:
            raise ValueError("a rewrite site must name at least one operation")
        for item in self.ops:
            _name(item, "site op")
        detail = {} if self.detail is None else dict(self.detail)
        for key in detail:
            _name(key, "site detail key")
        object.__setattr__(self, "detail", MappingProxyType(detail))

    def to_dict(self):
        return {"ops": list(self.ops), "detail": dict(self.detail)}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {"ops", "detail"}, "RewriteSite")
        return cls(ops=data["ops"], detail=data["detail"])


@dataclass(frozen=True)
class PassRecord:
    """What one pass did to one graph, including the zero it may have done."""
    name: str
    exactness: str
    tolerance: float | None
    sites: tuple[RewriteSite, ...]
    ops_before: int
    ops_after: int
    tensors_before: int
    tensors_after: int
    verification: Mapping | None = None

    def __post_init__(self):
        _name(self.name, "pass name")
        if self.exactness not in _EXACTNESS:
            raise ValueError(f"pass exactness must be one of {_EXACTNESS}")
        if self.exactness == TOLERANCE:
            _positive_real(self.tolerance, "pass tolerance")
            object.__setattr__(self, "tolerance", float(self.tolerance))
        elif self.tolerance is not None:
            raise ValueError("an exact pass must not publish a tolerance")
        object.__setattr__(self, "sites", _sequence(self.sites, "pass sites"))
        if any(not isinstance(site, RewriteSite) for site in self.sites):
            raise ValueError("sites must contain RewriteSite objects")
        if self.verification is not None:
            object.__setattr__(self, "verification", MappingProxyType(dict(self.verification)))

    @property
    def matched(self):
        return len(self.sites)

    def to_dict(self):
        return {"name": self.name, "exactness": self.exactness, "tolerance": self.tolerance,
                "sites": [site.to_dict() for site in self.sites],
                "matched": self.matched, "ops_before": self.ops_before,
                "ops_after": self.ops_after, "tensors_before": self.tensors_before,
                "tensors_after": self.tensors_after,
                "verification": None if self.verification is None else dict(self.verification)}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {"name", "exactness", "tolerance", "sites", "matched", "ops_before",
                     "ops_after", "tensors_before", "tensors_after", "verification"}, "PassRecord")
        record = cls(name=data["name"], exactness=data["exactness"], tolerance=data["tolerance"],
                     sites=tuple(RewriteSite.from_dict(site) for site in data["sites"]),
                     ops_before=data["ops_before"], ops_after=data["ops_after"],
                     tensors_before=data["tensors_before"], tensors_after=data["tensors_after"],
                     verification=data["verification"])
        if data["matched"] != record.matched:
            raise ValueError("PassRecord matched count disagrees with its sites")
        return record


@dataclass(frozen=True)
class RewriteReport:
    """Versioned, serializable account of a pass pipeline over one graph."""
    graph_name: str
    passes: tuple[PassRecord, ...]
    derivations: tuple[ConstantDerivation, ...] = ()

    def __post_init__(self):
        _name(self.graph_name, "graph name")
        for field_name, expected in (("passes", PassRecord), ("derivations", ConstantDerivation)):
            values = _sequence(getattr(self, field_name), field_name)
            if any(not isinstance(item, expected) for item in values):
                raise ValueError(f"{field_name} must contain {expected.__name__} objects")
            object.__setattr__(self, field_name, values)

    @property
    def matched(self):
        return sum(record.matched for record in self.passes)

    def to_dict(self):
        return {"schema_version": REWRITE_SCHEMA_VERSION, "graph_name": self.graph_name,
                "matched": self.matched,
                "passes": [record.to_dict() for record in self.passes],
                "derivations": [item.to_dict() for item in self.derivations]}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {"schema_version", "graph_name", "matched", "passes", "derivations"},
              "RewriteReport")
        if type(data["schema_version"]) is not int or data["schema_version"] != REWRITE_SCHEMA_VERSION:
            raise ValueError("unsupported RewriteReport schema_version")
        report = cls(graph_name=data["graph_name"],
                     passes=tuple(PassRecord.from_dict(item) for item in data["passes"]),
                     derivations=tuple(ConstantDerivation.from_dict(item)
                                       for item in data["derivations"]))
        if data["matched"] != report.matched:
            raise ValueError("RewriteReport matched count disagrees with its passes")
        return report

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(_load_json(text))


class GraphRewrite:
    """Contract: match sites, return a new validated graph, declare exactness."""
    name = "GraphRewrite"
    exactness = EXACT
    tolerance = None

    def rewrite(self, graph):
        """Return (graph, sites, derivations); the input graph is never mutated."""
        raise NotImplementedError


def _drop_tensors(tensors, names):
    return tuple(tensor for tensor in tensors if tensor.name not in names)


class DeadOpElimination(GraphRewrite):
    """Remove operations no declared output depends on, plus their tensors.

    Declared inputs and constants stay declared even when nothing reads them:
    dropping them would silently change the graph's external contract, which is
    a different decision from removing computation nobody asked for.
    """
    name = "DeadOpElimination"
    exactness = EXACT

    def rewrite(self, graph):
        live = set(graph.outputs)
        kept, sites = [], []
        for op in reversed(graph.ops):
            if set(op.outputs) & live:
                kept.append(op)
                live.update(op.inputs)
            else:
                sites.append(RewriteSite((op.name,), {"kind": op.kind.value,
                                                      "outputs": list(op.outputs)}))
        if not sites:
            return graph, (), ()
        kept.reverse()
        sites.reverse()
        removed = {name for site in sites for name in site.detail["outputs"]}
        return replace(graph, tensors=_drop_tensors(graph.tensors, removed),
                       ops=tuple(kept)), tuple(sites), ()


class CommonSubexpressionElimination(GraphRewrite):
    """Collapse operations that read the same inputs with the same attributes.

    A duplicate whose output the graph declares is kept: rewiring its consumers
    would leave a graph output with no producer, and renaming it would change
    the contract for the caller.
    """
    name = "CommonSubexpressionElimination"
    exactness = EXACT

    def rewrite(self, graph):
        tensors = {tensor.name: tensor for tensor in graph.tensors}
        outputs = set(graph.outputs)
        canonical, aliases, kept, sites = {}, {}, [], []
        for op in graph.ops:
            op = replace(op, inputs=tuple(aliases.get(name, name) for name in op.inputs))
            key = (op.kind.value, op.inputs, tuple(sorted(op.attributes.items())))
            produced = op.outputs[0]
            previous = canonical.get(key)
            if previous is not None and produced not in outputs and _interchangeable(
                    tensors[previous], tensors[produced]):
                aliases[produced] = previous
                sites.append(RewriteSite((op.name,), {"kind": op.kind.value,
                                                      "output": produced, "reuses": previous}))
                continue
            canonical.setdefault(key, produced)
            kept.append(op)
        if not sites:
            return graph, (), ()
        removed = {site.detail["output"] for site in sites}
        return replace(graph, tensors=_drop_tensors(graph.tensors, removed),
                       ops=tuple(kept)), tuple(sites), ()


def _interchangeable(first, second):
    return (first.shape == second.shape and first.logical_dtype == second.logical_dtype
            and first.storage_dtype == second.storage_dtype and first.tier == second.tier)


class MatMulProjectionFold(GraphRewrite):
    """Fold x@A^T@B^T into x@(B@A)^T when both projections are constants.

    Only folds when the folded matrix is no larger than the two it replaces,
    because the rewrite trades two reductions for one at the cost of a weight
    that must be materialized once. The folded constant is reported as a
    derivation, never fabricated here. The result is not exact: summing over
    the shared dimension in a different order moves the last float32 bits.
    """
    name = "MatMulProjectionFold"
    exactness = TOLERANCE
    # Measured on the fixtures of tests/test_graph_algebra_regressions.py; a
    # site that exceeds it fails the pass instead of being published.
    tolerance = 1e-4

    def rewrite(self, graph):
        sites, derivations = [], []
        for _ in range(MAX_REWRITE_ROUNDS):
            outcome = self._fold_once(graph)
            if outcome is None:
                break
            graph, site, derivation = outcome
            sites.append(site)
            derivations.append(derivation)
        return graph, tuple(sites), tuple(derivations)

    def _fold_once(self, graph):
        tensors = {tensor.name: tensor for tensor in graph.tensors}
        constants = set(graph.constants)
        producers = {op.outputs[0]: (index, op) for index, op in enumerate(graph.ops)}
        consumers = {}
        for op in graph.ops:
            for name in op.inputs:
                consumers.setdefault(name, []).append(op.name)
        for second in graph.ops:
            if not _is_projection(second, constants):
                continue
            intermediate = second.inputs[0]
            produced = producers.get(intermediate)
            if produced is None or intermediate in graph.outputs:
                continue
            first = produced[1]
            if not _is_projection(first, constants) or len(consumers[intermediate]) != 1:
                continue
            source, left_weight = first.inputs
            right_weight = second.inputs[1]
            inner = tensors[left_weight].shape[1]
            middle = tensors[left_weight].shape[0]
            columns = tensors[right_weight].shape[0]
            folded_elements = columns * inner
            if folded_elements > middle * inner + columns * middle:
                continue
            name = f"{second.name}.folded_weight"
            if name in tensors:
                raise RewriteError(f"folded constant name is already taken: {name}")
            derivation = ConstantDerivation(name, "matmul", (right_weight, left_weight),
                                            (columns, inner))
            weight = TensorDesc(name, (columns, inner), DType.F32, DType.F32)
            ops = [op for position, op in enumerate(graph.ops) if position != produced[0]]
            ops[[op.name for op in ops].index(second.name)] = replace(second, inputs=(source, name))
            folded = replace(graph, tensors=(*_drop_tensors(graph.tensors, {intermediate}), weight),
                             ops=tuple(ops), constants=(*graph.constants, name))
            unreferenced = sorted({left_weight, right_weight} - {
                item for op in folded.ops for item in op.inputs})
            site = RewriteSite((first.name, second.name),
                               {"folded_constant": name, "sources": [right_weight, left_weight],
                                "folded_elements": folded_elements,
                                "replaced_elements": middle * inner + columns * middle,
                                "unreferenced_constants": unreferenced})
            return folded, site, derivation
        return None


def _is_projection(op, constants):
    return (op.kind == OpKind.MATMUL and op.attributes.get("transpose_b", False)
            and op.inputs[1] in constants)


class GraphEquivalenceVerifier:
    """Measure output agreement between two graphs on shared random bindings.

    The bindings come from the original graph, so a rewrite cannot choose the
    numbers it is judged on. Derived constants are materialized in order, which
    also lets a derivation depend on an earlier one.
    """
    def __init__(self, *, seeds=(17, 23)):
        self.seeds = tuple(seeds)
        if not self.seeds:
            raise ValueError("a verifier needs at least one seed")

    def __call__(self, before, after, derivations=()):
        if set(before.outputs) != set(after.outputs):
            raise RewriteError("a rewrite must preserve the declared graph outputs")
        identical = True
        max_abs = max_rel = 0.0
        for seed in self.seeds:
            bindings = random_bindings(before, seed=seed)
            for derivation in derivations:
                bindings[derivation.name] = derivation.materialize(bindings)
            left = evaluate_graph(before, _restrict(before, bindings))
            right = evaluate_graph(after, _restrict(after, bindings))
            for name in sorted(before.outputs):
                identical = identical and tensor_bytes(left[name]) == tensor_bytes(right[name])
                for expected_row, actual_row in zip(left[name], right[name]):
                    for expected, actual in zip(expected_row, actual_row):
                        error = abs(actual - expected)
                        max_abs = max(max_abs, error)
                        max_rel = max(max_rel, error / max(abs(expected), 1e-12))
        return {"identical": identical, "max_abs_error": max_abs, "max_rel_error": max_rel,
                "seeds": list(self.seeds), "outputs": sorted(before.outputs)}


def _restrict(graph, bindings):
    return {name: bindings[name] for name in (*graph.inputs, *graph.constants)}


def run_passes(graph, passes, *, verifier=None):
    """Apply passes in order, verifying every rewrite that matched anything.

    A pass that matches nothing costs nothing and is recorded as a zero. A pass
    that matches is measured before its graph is accepted: an exact pass must
    reproduce the outputs bit for bit, a tolerance pass must stay under the
    tolerance it published. Either failure raises and leaves the caller with a
    graph that no unverified rewrite has touched.
    """
    if not isinstance(graph, ModelGraph):
        raise ValueError("graph must be a ModelGraph")
    graph.validate()
    records, derivations = [], []
    for item in passes:
        if not isinstance(item, GraphRewrite):
            raise ValueError("passes must contain GraphRewrite objects")
        rewritten, sites, produced = item.rewrite(graph)
        if not isinstance(rewritten, ModelGraph):
            raise RewriteError(f"{item.name} did not return a ModelGraph")
        rewritten.validate()
        verification = None
        if sites:
            if set(rewritten.outputs) != set(graph.outputs):
                raise RewriteError(f"{item.name} changed the declared graph outputs")
            if verifier is not None:
                verification = verifier(graph, rewritten, (*derivations, *produced))
                _enforce(item, verification)
        elif rewritten is not graph:
            raise RewriteError(f"{item.name} returned a new graph without reporting a site")
        records.append(PassRecord(item.name, item.exactness, item.tolerance, sites,
                                  len(graph.ops), len(rewritten.ops),
                                  len(graph.tensors), len(rewritten.tensors), verification))
        derivations.extend(produced)
        graph = rewritten
    return graph, RewriteReport(graph.name, tuple(records), tuple(derivations))


def _enforce(item, verification):
    if item.exactness == EXACT:
        if not verification["identical"]:
            raise RewriteError(f"{item.name} claims exactness but changed the output bytes "
                               f"by up to {verification['max_abs_error']}")
    elif verification["max_abs_error"] > item.tolerance:
        raise RewriteError(f"{item.name} exceeded its published tolerance {item.tolerance}: "
                           f"measured {verification['max_abs_error']}")


def default_passes():
    """The pipeline tools run: two exact cleanups, then the one fold."""
    return (DeadOpElimination(), CommonSubexpressionElimination(), MatMulProjectionFold())
