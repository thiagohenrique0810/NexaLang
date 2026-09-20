"""Parameter regions and plasticity validation over a frozen physical model.

PL.01a declares *where* a model could learn; it does not learn, train, mask
gradients or change any baseline path. Nothing in this module is read by the
lowering, the memory planner, the bundle writer or the executor, so a session
run with a plasticity config produces exactly the same plan and logits as one
run without it.

The region record deliberately omits ``importance``, ``drift_budget``,
``update_count``, ``last_update`` and ``residency``. The PDF proposes them, but
this repository has no producer and no consumer for any of the five today: a
field that nothing writes and nothing reads cannot be falsified by a test, and
a schema that only agrees with itself is what this item is easiest to fake as.
They belong to PL.02/PL.04 and to the residency work in CC/M3.

Resolution is the part a test can break. It maps declared tensor names onto the
physical inventory of a ``ModelConfig``, so a tied ``lm_head.weight`` and
``model.embed_tokens.weight`` are the *same* storage and overlap, while the same
declaration against an untied model names two distinct tensors and does not.
"""
from dataclasses import dataclass, fields
import json
import math
from math import prod
from types import MappingProxyType
from typing import Mapping

from .model_config import ModelConfig
from .model_ir import _integer, _keys, _load_json, _name


SCHEMA_VERSION = 1

# Lifecycle vocabulary of the plan's table. Residency (hot/warm/cold) and the
# candidate/committed state of a transaction are separate axes and are not here.
REGION_CLASSES = ("stable_core", "mature_expert", "plastic_expert",
                  "adapter_delta", "working_memory")

MAX_REGIONS = 4096
MAX_REGION_TENSORS = 4096
MAX_PROVENANCE_DEPTH = 8


def _unit_interval(value, label):
    # ``type(True) is bool``, so booleans never reach the float conversion:
    # the plan requires integer/float fields to reject booleans outright.
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    return value


def _boolean(value, label):
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _provenance(value, depth=0):
    """Accept only a JSON-round-trippable tree, so to_json can never fail late."""
    if depth > MAX_PROVENANCE_DEPTH:
        raise ValueError("provenance nests deeper than the supported limit")
    if value is None or type(value) is bool or isinstance(value, str):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("provenance rejects NaN and infinity")
        return value
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            _name(key, "provenance key")
            result[key] = _provenance(item, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_provenance(item, depth + 1) for item in value]
    raise ValueError("provenance must be a JSON object tree")


@dataclass(frozen=True)
class TensorSpan:
    """A whole physical tensor, or a half-open row range inside one.

    ``start_row``/``row_count`` are either both absent (the whole tensor) or
    both present. A one-dimensional norm vector has one scalar per row.
    """
    tensor: str
    start_row: int | None = None
    row_count: int | None = None

    def __post_init__(self):
        _name(self.tensor, "span tensor name")
        if (self.start_row is None) != (self.row_count is None):
            raise ValueError("a row range needs both start_row and row_count")
        if self.start_row is not None:
            _integer(self.start_row, "start_row", 0)
            _integer(self.row_count, "row_count", 1)

    @property
    def whole_tensor(self):
        return self.start_row is None

    def to_dict(self):
        return {"tensor": self.tensor, "start_row": self.start_row,
                "row_count": self.row_count}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {"tensor", "start_row", "row_count"}, "TensorSpan")
        return cls(**data)

    @classmethod
    def coerce(cls, value):
        """Accept a span, its object form, or a bare name meaning the whole tensor.

        to_dict always emits the object form, so a round trip is stable either way.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)
        if isinstance(value, Mapping):
            return cls.from_dict(dict(value))
        raise ValueError("a region tensor must be a TensorSpan, a name or its object form")


@dataclass(frozen=True)
class ParameterRegion:
    """One declared region of the physical parameter space.

    ``protected`` is the decision, not a hint: the plan states that for P0-P2 it
    prevails over plasticity, scores and any router request. A protected region
    therefore has to declare plasticity 0.0, so a config cannot claim both.
    """
    id: str
    domain: str
    region_class: str
    plasticity: float
    maturity: float
    protected: bool
    tensors: tuple
    provenance: Mapping

    def __post_init__(self):
        _name(self.id, "region id")
        _name(self.domain, "region domain")
        if self.region_class not in REGION_CLASSES:
            raise ValueError(f"region_class must be one of: {', '.join(REGION_CLASSES)}")
        object.__setattr__(self, "plasticity", _unit_interval(self.plasticity, "plasticity"))
        object.__setattr__(self, "maturity", _unit_interval(self.maturity, "maturity"))
        _boolean(self.protected, "protected")
        if self.protected and self.plasticity != 0.0:
            raise ValueError(f"protected region {self.id} must declare plasticity 0.0")
        if not isinstance(self.tensors, (list, tuple)) or not self.tensors:
            raise ValueError("a region must name at least one tensor")
        if len(self.tensors) > MAX_REGION_TENSORS:
            raise ValueError("region names more tensors than the supported limit")
        spans = tuple(TensorSpan.coerce(span) for span in self.tensors)
        if len(set(spans)) != len(spans):
            raise ValueError(f"region {self.id} repeats a tensor span")
        object.__setattr__(self, "tensors", spans)
        provenance = _provenance(self.provenance)
        if not isinstance(provenance, dict):
            raise ValueError("provenance must be a JSON object")
        object.__setattr__(self, "provenance", MappingProxyType(provenance))

    def to_dict(self):
        return {"id": self.id, "domain": self.domain, "region_class": self.region_class,
                "plasticity": self.plasticity, "maturity": self.maturity,
                "protected": self.protected,
                "tensors": [span.to_dict() for span in self.tensors],
                "provenance": dict(self.provenance)}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {field.name for field in fields(cls)}, "ParameterRegion")
        return cls(**data)


@dataclass(frozen=True)
class ModelPlasticityConfig:
    """Versioned container of regions for one named model.

    The container carries no defaults for the model shape: it is resolved
    against a ``ModelConfig`` and is meaningless on its own.
    """
    model_name: str
    regions: tuple

    def __post_init__(self):
        _name(self.model_name, "model_name")
        if not isinstance(self.regions, (list, tuple)):
            raise ValueError("regions must be a list of ParameterRegion")
        if len(self.regions) > MAX_REGIONS:
            raise ValueError(f"a plasticity config holds at most {MAX_REGIONS} regions")
        regions = tuple(region if isinstance(region, ParameterRegion)
                        else ParameterRegion.from_dict(dict(region)) for region in self.regions)
        identifiers = [region.id for region in regions]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("region ids must be unique")
        object.__setattr__(self, "regions", regions)

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, "model_name": self.model_name,
                "regions": [region.to_dict() for region in self.regions]}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {"schema_version", "model_name", "regions"}, "ModelPlasticityConfig")
        if type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported ModelPlasticityConfig schema_version")
        if not isinstance(data["regions"], list):
            raise ValueError("regions must be a JSON array")
        return cls(model_name=data["model_name"], regions=tuple(data["regions"]))

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text):
        # _load_json refuses duplicate object keys and NaN/Infinity literals,
        # which json.loads would otherwise accept silently.
        return cls.from_dict(_load_json(text))


@dataclass(frozen=True)
class ResolvedSpan:
    """A row range of one physical tensor, with its exact scalar count."""
    tensor: str
    start_row: int
    row_count: int
    scalars: int

    def to_dict(self):
        return {"tensor": self.tensor, "start_row": self.start_row,
                "row_count": self.row_count, "scalars": self.scalars}


@dataclass(frozen=True)
class ResolvedRegion:
    id: str
    domain: str
    region_class: str
    plasticity: float
    maturity: float
    protected: bool
    spans: tuple
    scalars: int

    def to_dict(self):
        return {"id": self.id, "domain": self.domain, "region_class": self.region_class,
                "plasticity": self.plasticity, "maturity": self.maturity,
                "protected": self.protected, "scalars": self.scalars,
                "spans": [span.to_dict() for span in self.spans]}


@dataclass(frozen=True)
class PlasticityMap:
    """A partition of the physical parameter space into regions and the rest.

    ``covered_scalars`` is summed from the resolved region spans, while
    ``unassigned_scalars`` is summed from the complement of those spans over the
    physical inventory, and ``total_scalars`` from the declared tensor shapes.
    The three are produced by different walks on purpose: their identity against
    ``ModelConfig.parameter_count()`` is what a caller can check, and an alias
    resolved as separate storage breaks it.
    """
    model_name: str
    regions: tuple
    unassigned: tuple
    covered_scalars: int
    unassigned_scalars: int
    total_scalars: int

    def region(self, identifier):
        for region in self.regions:
            if region.id == identifier:
                return region
        raise KeyError(identifier)

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, "model_name": self.model_name,
                "covered_scalars": self.covered_scalars,
                "unassigned_scalars": self.unassigned_scalars,
                "total_scalars": self.total_scalars,
                "regions": [region.to_dict() for region in self.regions],
                "unassigned": [span.to_dict() for span in self.unassigned]}


def physical_inventory(config: ModelConfig):
    """Physical tensors and the alias map that folds tied names onto them."""
    if not isinstance(config, ModelConfig):
        raise ValueError("config must be a ModelConfig")
    return config.required_tensor_shapes(), dict(config.tensor_aliases())


def resolve_plasticity_map(config: ModelConfig, plasticity: ModelPlasticityConfig) -> PlasticityMap:
    """Resolve regions onto physical storage; refuse overlap and unknown names.

    Two declarations that land on the same physical rows are rejected even when
    they were written with different names, which is the only reason the tied
    alias matters here at all.
    """
    shapes, aliases = physical_inventory(config)
    if not isinstance(plasticity, ModelPlasticityConfig):
        raise ValueError("plasticity must be a ModelPlasticityConfig")
    if plasticity.model_name != config.name:
        raise ValueError(f"plasticity config targets model {plasticity.model_name!r}, "
                         f"not {config.name!r}")
    occupancy = {}
    regions = []
    covered = 0
    for region in plasticity.regions:
        resolved = []
        for span in region.tensors:
            physical = aliases.get(span.tensor, span.tensor)
            if physical not in shapes:
                raise ValueError(f"region {region.id} names unknown tensor {span.tensor}")
            shape = shapes[physical]
            rows = shape[0]
            width = prod(shape[1:])
            start = 0 if span.whole_tensor else span.start_row
            count = rows if span.whole_tensor else span.row_count
            if start + count > rows:
                raise ValueError(f"region {region.id} span on {span.tensor} exceeds its "
                                 f"{rows} rows")
            for other_start, other_count, other_region in occupancy.get(physical, ()):
                if start < other_start + other_count and other_start < start + count:
                    raise ValueError(f"regions {other_region} and {region.id} overlap on "
                                     f"physical tensor {physical}")
            occupancy.setdefault(physical, []).append((start, count, region.id))
            scalars = count * width
            covered += scalars
            resolved.append(ResolvedSpan(physical, start, count, scalars))
        resolved.sort(key=lambda span: (span.tensor, span.start_row))
        regions.append(ResolvedRegion(region.id, region.domain, region.region_class,
                                      region.plasticity, region.maturity, region.protected,
                                      tuple(resolved), sum(span.scalars for span in resolved)))
    unassigned = []
    unassigned_scalars = 0
    total = 0
    for physical, shape in shapes.items():
        rows, width = shape[0], prod(shape[1:])
        total += rows * width
        cursor = 0
        for start, count, _ in sorted(occupancy.get(physical, [])):
            if start > cursor:
                span = ResolvedSpan(physical, cursor, start - cursor, (start - cursor) * width)
                unassigned.append(span)
                unassigned_scalars += span.scalars
            cursor = start + count
        if cursor < rows:
            span = ResolvedSpan(physical, cursor, rows - cursor, (rows - cursor) * width)
            unassigned.append(span)
            unassigned_scalars += span.scalars
    return PlasticityMap(config.name, tuple(regions), tuple(unassigned),
                         covered, unassigned_scalars, total)


def protected_tensors(plasticity_map: PlasticityMap):
    """Physical tensors any protected region claims, for a learning target check."""
    if not isinstance(plasticity_map, PlasticityMap):
        raise ValueError("plasticity_map must be a PlasticityMap")
    return frozenset(span.tensor for region in plasticity_map.regions if region.protected
                     for span in region.spans)


def validate_learning_target(plasticity_map: PlasticityMap, tensor, *, rows=None):
    """Refuse a learning target that a protected region covers.

    ``rows`` restricts the check to a half-open row range of the tensor; without
    it any protected row of that tensor refuses the target.
    """
    if not isinstance(plasticity_map, PlasticityMap):
        raise ValueError("plasticity_map must be a PlasticityMap")
    _name(tensor, "target tensor name")
    if rows is not None:
        start, count = rows
        _integer(start, "target start_row", 0)
        _integer(count, "target row_count", 1)
    for region in plasticity_map.regions:
        if not region.protected:
            continue
        for span in region.spans:
            if span.tensor != tensor:
                continue
            if rows is None or (start < span.start_row + span.row_count
                                and span.start_row < start + count):
                raise ValueError(f"tensor {tensor} is covered by protected region {region.id}")
    return True
