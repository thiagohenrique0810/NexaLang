"""Pure planning for transactional CPU KV page aging: F32 -> Q4 -> Q3.

Age counts logical pages from the newest page, including a partial newest page.
Only complete pages migrate, after the attention using the current codecs.
No page is evicted and no page is promoted. A large chunk may skip warm Q4.
All source pages remain resident until the whole operation is committed, so a
failed re-encode can discard its replacements without changing the prefix.

Byte counts cover owned page backing allocations, including alignment slack.
They exclude Python metadata, model weights, activations and RSS. Migration
scratch is reported separately for admission alongside those other resources.
"""
from dataclasses import dataclass, field, fields
import hashlib
import json
import math
from types import MappingProxyType

from .kv_plan import _strict_equal
from .model_config import ModelConfig
from .model_ir import _integer, _keys, _load_json
from .paged_kv_plan import (
    MAX_ALLOCATION_BYTES, MAX_PLAN_SEGMENTS, MAX_Q4_GROUP_SIZE,
    PagedKVCachePlan, make_paged_kv_cache_plan,
)


SCHEMA_VERSION = 1
POLICY_ID = "CPU_PAGE_AGE_F32_Q4_Q3_V1"
_CODECS = {"hot": "f32", "warm": "q4", "cold": "q3"}
# Higher rank means more precision: a page may be kept above what its age
# would give it, never below. Ageing does not run backwards.
_PRECISION_RANK = {"q3": 0, "q4": 1, "f32": 2}
_TIERS = {value: key for key, value in _CODECS.items()}
_CODEC_IDS = {"f32": "F32_NATIVE", "q4": "Q4_GROUPED", "q3": "Q3_GROUPED"}


def normalize_retained(retained):
    """Canonical retained-page form: pairs sorted by page index, JSON friendly."""
    if retained is None:
        return ()
    items = retained.items() if isinstance(retained, dict) else retained
    result = []
    seen = set()
    for pair in items:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("A retained page is a (page_index, codec) pair")
        page_index, codec = pair
        _integer(page_index, "retained page index")
        if page_index in seen:
            raise ValueError("A retained page index appears twice")
        if not isinstance(codec, str) or codec not in _PRECISION_RANK:
            raise ValueError("A retained page codec must be f32, q4 or q3")
        seen.add(page_index)
        result.append((page_index, codec))
    result.sort()
    return tuple(result)


def _byte_range(value, label):
    if value > MAX_ALLOCATION_BYTES:
        raise ValueError(f"{label} exceeds the supported allocation byte range")
    return value


class _JSON:
    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text):
        try:
            data = _load_json(text)
        except RecursionError as error:
            raise ValueError("Tiered KV JSON nesting is too deep") from error
        return cls.from_dict(data)


@dataclass(frozen=True)
class TieredKVPolicy(_JSON):
    hot_pages: int = 1
    warm_pages: int = 1
    group_size: int = 32
    # A page whose re-encode error exceeds this RMSE keeps its current codec,
    # up to retain_pages of them. Zero pages means ageing is unconditional.
    quality_max_rmse: float | None = None
    retain_pages: int = 0

    def __post_init__(self):
        _integer(self.hot_pages, "hot_pages", 1)
        _integer(self.warm_pages, "warm_pages")
        _integer(self.group_size, "group_size", 1)
        _integer(self.retain_pages, "retain_pages")
        if self.quality_max_rmse is not None:
            if (not isinstance(self.quality_max_rmse, (int, float))
                    or isinstance(self.quality_max_rmse, bool)
                    or not math.isfinite(self.quality_max_rmse) or self.quality_max_rmse < 0):
                raise ValueError("quality_max_rmse must be a finite non-negative number")
            object.__setattr__(self, "quality_max_rmse", float(self.quality_max_rmse))
        elif self.retain_pages:
            raise ValueError("retain_pages needs a quality_max_rmse to compare against")
        if self.retain_pages > MAX_PLAN_SEGMENTS:
            raise ValueError("retain_pages exceeds the page metadata limit")
        if self.hot_pages > MAX_PLAN_SEGMENTS or self.warm_pages > MAX_PLAN_SEGMENTS:
            raise ValueError("Tier counts exceed the page metadata limit")
        if self.group_size > MAX_Q4_GROUP_SIZE:
            raise ValueError(f"Tiered KV group_size exceeds {MAX_Q4_GROUP_SIZE}")

    def to_dict(self):
        return {"policy_id": POLICY_ID, "hot_pages": self.hot_pages,
                "warm_pages": self.warm_pages, "group_size": self.group_size,
                "quality_max_rmse": self.quality_max_rmse, "retain_pages": self.retain_pages}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {"policy_id", "hot_pages", "warm_pages", "group_size",
                     "quality_max_rmse", "retain_pages"}, "TieredKVPolicy")
        result = cls(data["hot_pages"], data["warm_pages"], data["group_size"],
                     data["quality_max_rmse"], data["retain_pages"])
        if not _strict_equal(data, result.to_dict()):
            raise ValueError("Tiered KV policy differs from its canonical contract")
        return result


@dataclass(frozen=True)
class TieredKVPage(_JSON):
    page_index: int
    logical_start: int
    valid_tokens: int
    tier: str
    age_rank: int
    codec: str
    codec_id: str
    codec_version: int
    group_size: int | None
    layout_identity: str
    page_allocation_bytes: int

    def __post_init__(self):
        for name in ("page_index", "logical_start", "age_rank"):
            _integer(getattr(self, name), name)
        for name in ("valid_tokens", "codec_version", "page_allocation_bytes"):
            _integer(getattr(self, name), name, 1)
        if (not isinstance(self.codec, str) or self.codec not in _TIERS
                or self.tier != _TIERS[self.codec]
                or self.codec_id != _CODEC_IDS[self.codec] or self.codec_version != 1):
            raise ValueError("Tiered KV page tier and codec must match")
        if self.codec == "f32":
            if self.group_size is not None:
                raise ValueError("F32 pages do not accept group_size")
        else:
            _integer(self.group_size, "group_size", 1)
            if self.group_size > MAX_Q4_GROUP_SIZE:
                raise ValueError("Page group_size exceeds the supported limit")
        if (not isinstance(self.layout_identity, str) or len(self.layout_identity) != 64
                or any(c not in "0123456789abcdef" for c in self.layout_identity)):
            raise ValueError("Page layout_identity must be a canonical SHA256 digest")
        _byte_range(self.page_allocation_bytes, "Page allocation")

    def to_dict(self):
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {item.name for item in fields(cls)}, "TieredKVPage")
        result = cls(**data)
        if not _strict_equal(data, result.to_dict()):
            raise ValueError("Tiered KV page differs from its canonical descriptor")
        return result


@dataclass(frozen=True)
class TieredKVMigration(_JSON):
    source: TieredKVPage
    target: TieredKVPage

    def __post_init__(self):
        if not isinstance(self.source, TieredKVPage) or not isinstance(self.target, TieredKVPage):
            raise ValueError("Migration requires source and target page descriptors")
        a, b = self.source, self.target
        if ((a.page_index, a.logical_start, a.valid_tokens, a.age_rank)
                != (b.page_index, b.logical_start, b.valid_tokens, b.age_rank)
                or (a.codec, b.codec) not in (("f32", "q4"), ("f32", "q3"), ("q4", "q3"))):
            raise ValueError("Migration must age a page while preserving its logical token range")

    def to_dict(self):
        return {"source": self.source.to_dict(), "target": self.target.to_dict()}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {"source", "target"}, "TieredKVMigration")
        return cls(TieredKVPage.from_dict(data["source"]), TieredKVPage.from_dict(data["target"]))


@dataclass(frozen=True)
class TieredKVCachePlan(_JSON):
    config: ModelConfig
    capacity: int
    page_tokens: int
    policy: TieredKVPolicy = field(default_factory=TieredKVPolicy)
    _layouts: object = field(init=False, repr=False, compare=False)
    _identities: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.config, ModelConfig) or not isinstance(self.policy, TieredKVPolicy):
            raise ValueError("Tiered KV requires ModelConfig and TieredKVPolicy")
        _integer(self.capacity, "KV capacity", 1)
        _integer(self.page_tokens, "page_tokens", 1)
        if self.capacity > self.config.max_position_embeddings:
            raise ValueError("KV capacity exceeds the configured context")
        if self.max_pages > MAX_PLAN_SEGMENTS:
            raise ValueError(f"Tiered KV capacity exceeds the {MAX_PLAN_SEGMENTS} page metadata limit")
        layouts = {codec: make_paged_kv_cache_plan(
            self.config, self.capacity, self.page_tokens, codec=codec,
            group_size=None if codec == "f32" else self.policy.group_size)
            for codec in _TIERS}
        identities = {codec: hashlib.sha256(layout.to_json(indent=None).encode("utf-8")).hexdigest()
                      for codec, layout in layouts.items()}
        object.__setattr__(self, "_layouts", MappingProxyType(layouts))
        object.__setattr__(self, "_identities", MappingProxyType(identities))
        _byte_range(self.capacity_resident_bytes_bound, "Tiered KV capacity")
        _byte_range(self.migration_scratch_bytes, "Migration scratch")

    def layout(self, codec) -> PagedKVCachePlan:
        if not isinstance(codec, str) or codec not in self._layouts:
            raise ValueError("Tiered KV layout codec must be f32, q4 or q3")
        return self._layouts[codec]

    @property
    def max_pages(self):
        return (self.capacity + self.page_tokens - 1) // self.page_tokens

    @property
    def migration_scratch_bytes(self):
        """One reconstructed head (F32) and bounded grouped-codec scalar staging."""
        return 4 * self.config.head_dim + 24

    @property
    def max_page_allocation_bytes(self):
        return max(layout.page_allocation_bytes for layout in self._layouts.values())

    @property
    def max_packed_page_allocation_bytes(self):
        return max(self.layout(codec).page_allocation_bytes for codec in ("q4", "q3"))

    @property
    def capacity_resident_bytes_bound(self):
        # The canonical residence is monotone in page count: each additional
        # page adds A_f32 while filling hot, A_q4 while filling warm, then A_q3.
        hot = min(self.max_pages, self.policy.hot_pages)
        warm = min(self.max_pages - hot, self.policy.warm_pages)
        cold = self.max_pages - hot - warm
        return (hot * self.layout("f32").page_allocation_bytes
                + warm * self.layout("q4").page_allocation_bytes
                + cold * self.layout("q3").page_allocation_bytes)

    def page_count(self, length):
        return self.layout("f32").page_count(length)

    @property
    def quality_retention_bytes(self):
        """Extra allocation the admitted retentions may cost over ageing."""
        if not self.policy.retain_pages:
            return 0
        spread = (self.layout("f32").page_allocation_bytes -
                  self.layout("q3").page_allocation_bytes)
        return self.policy.retain_pages * max(spread, 0)

    def allocation_limit_bytes(self, max_chunk_length):
        """Conservative page-only bound for arbitrary prefill replacement/append.

        Old canonical capacity, a fresh F32 chunk, and *all* possible packed
        replacements may coexist. Packed pages can exceed F32 for small heads
        or large groups; their bound does not assume compression.
        """
        self.layout("f32").reservation_pages(max_chunk_length)
        extra = self.page_count(max_chunk_length)
        bound = (self.capacity_resident_bytes_bound + self.quality_retention_bytes
                 + extra * self.layout("f32").page_allocation_bytes
                 + self.max_pages * self.max_packed_page_allocation_bytes)
        return _byte_range(bound, "Tiered KV reservation")

    def _page(self, page_index, length, codec):
        count = self.page_count(length)
        layout = self.layout(codec)
        return TieredKVPage(page_index, page_index * self.page_tokens,
                            min(self.page_tokens, length - page_index * self.page_tokens),
                            _TIERS[codec], count - 1 - page_index, codec, _CODEC_IDS[codec], 1,
                            layout.group_size, self._identities[codec], layout.page_allocation_bytes)

    def desired_pages(self, length, retained=None):
        """Canonical layout, with the pages quality kept above their age."""
        retained = {} if retained is None else dict(retained)
        if len(retained) > self.policy.retain_pages:
            raise ValueError("More retained pages than the policy admits")
        count = self.page_count(length)
        for page_index in retained:
            _integer(page_index, "retained page index")
            if page_index >= count:
                raise ValueError("A retained page index falls outside the prefix")
        result = []
        for page_index in range(count):
            age = count - 1 - page_index
            if age < self.policy.hot_pages:
                codec = "f32"
            elif age < self.policy.hot_pages + self.policy.warm_pages:
                codec = "q4"
            else:
                codec = "q3"
            kept = retained.get(page_index)
            if kept is not None:
                if kept not in _PRECISION_RANK or _PRECISION_RANK[kept] < _PRECISION_RANK[codec]:
                    raise ValueError("A retained page may only keep more precision than its age gives")
                codec = kept
            result.append(self._page(page_index, length, codec))
        return tuple(result)

    def plan_transition(self, past_length, chunk_length, mode, committed_pages, retained=None):
        return TieredKVTransition(self, past_length, chunk_length, mode, committed_pages,
                                  normalize_retained(retained))

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, "config": self.config.to_dict(),
                "capacity": self.capacity, "page_tokens": self.page_tokens,
                "policy": self.policy.to_dict(), "max_pages": self.max_pages,
                "layouts": {codec: layout.to_dict() for codec, layout in self._layouts.items()},
                "layout_identities": dict(self._identities),
                "capacity_resident_bytes_bound": self.capacity_resident_bytes_bound,
                "migration_scratch_bytes": self.migration_scratch_bytes}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {"schema_version", "config", "capacity", "page_tokens", "policy", "max_pages",
                     "layouts", "layout_identities", "capacity_resident_bytes_bound",
                     "migration_scratch_bytes"}, "TieredKVCachePlan")
        plan = cls(ModelConfig.from_dict(data["config"]), data["capacity"], data["page_tokens"],
                   TieredKVPolicy.from_dict(data["policy"]))
        if not _strict_equal(data, plan.to_dict()):
            raise ValueError("Tiered KV layout, identity, byte counts or version differ from canonical plan")
        return plan


@dataclass(frozen=True)
class TieredKVTransition(_JSON):
    cache_plan: TieredKVCachePlan
    past_length: int
    chunk_length: int
    mode: str
    committed_pages: tuple[TieredKVPage, ...]
    retained_pages: tuple = ()
    final_retained_pages: tuple = field(init=False)
    position_offset: int = field(init=False)
    new_length: int = field(init=False)
    attention_pages: tuple[TieredKVPage, ...] = field(init=False)
    final_pages: tuple[TieredKVPage, ...] = field(init=False)
    migrations: tuple[TieredKVMigration, ...] = field(init=False)
    fresh_pages: tuple[TieredKVPage, ...] = field(init=False)
    old_resident_bytes: int = field(init=False)
    attention_resident_bytes: int = field(init=False)
    migration_replacement_bytes: int = field(init=False)
    transaction_peak_bytes: int = field(init=False)
    final_resident_bytes: int = field(init=False)

    def __post_init__(self):
        if not isinstance(self.cache_plan, TieredKVCachePlan):
            raise ValueError("Transition requires a TieredKVCachePlan")
        cache = self.cache_plan
        object.__setattr__(self, "retained_pages", normalize_retained(self.retained_pages))
        retained = dict(self.retained_pages)
        old_count = cache.page_count(self.past_length)
        _integer(self.chunk_length, "chunk_length", 1)
        if self.mode not in ("prefill", "decode"):
            raise ValueError("Tiered KV mode must be prefill or decode")
        if self.mode == "decode" and not self.past_length:
            raise ValueError("Decode requires a committed KV prefix")
        if (not isinstance(self.committed_pages, tuple) or len(self.committed_pages) != old_count
                or any(not isinstance(page, TieredKVPage) for page in self.committed_pages)
                or self.committed_pages != cache.desired_pages(self.past_length, retained)):
            raise ValueError("Committed pages do not match the canonical policy and prefix layout")
        position = 0 if self.mode == "prefill" else self.past_length
        length = position + self.chunk_length
        count = cache.page_count(length)
        segments = (position % cache.page_tokens + self.chunk_length + cache.page_tokens - 1) // cache.page_tokens
        if segments > MAX_PLAN_SEGMENTS:
            raise ValueError(f"Transition exceeds the {MAX_PLAN_SEGMENTS} segment metadata limit")
        keep = 0 if self.mode == "prefill" else old_count
        attention = tuple(cache._page(index, length, self.committed_pages[index].codec
                                      if index < keep else "f32") for index in range(count))
        # A replaced prefix keeps nothing: retention belongs to the pages that
        # survive the transition, and prefill rebuilds every page from scratch.
        surviving = normalize_retained({index: codec for index, codec in retained.items()
                                        if index < keep})
        final = cache.desired_pages(length, dict(surviving))
        migrations = tuple(TieredKVMigration(source, target) for source, target in zip(attention, final)
                           if source.codec != target.codec)
        if any(m.source.valid_tokens != cache.page_tokens for m in migrations):
            raise ValueError("Only complete KV pages may migrate")
        fresh = attention[keep:]
        old_bytes = sum(page.page_allocation_bytes for page in self.committed_pages)
        attention_bytes = old_bytes + sum(page.page_allocation_bytes for page in fresh)
        replacement_bytes = sum(m.target.page_allocation_bytes for m in migrations)
        values = {"final_retained_pages": surviving,
                  "position_offset": position, "new_length": length,
                  "attention_pages": attention, "final_pages": final,
                  "migrations": migrations, "fresh_pages": fresh,
                  "old_resident_bytes": old_bytes, "attention_resident_bytes": attention_bytes,
                  "migration_replacement_bytes": replacement_bytes,
                  "transaction_peak_bytes": _byte_range(attention_bytes + replacement_bytes,
                                                       "Tiered KV transaction"),
                  "final_resident_bytes": sum(page.page_allocation_bytes for page in final)}
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def to_dict(self):
        result = {"schema_version": SCHEMA_VERSION, "cache_plan": self.cache_plan.to_dict(),
                  "past_length": self.past_length, "chunk_length": self.chunk_length,
                  "mode": self.mode,
                  "retained_pages": [[index, codec] for index, codec in self.retained_pages],
                  "final_retained_pages": [[index, codec] for index, codec in self.final_retained_pages]}
        for name in ("committed_pages", "attention_pages", "final_pages", "migrations", "fresh_pages"):
            result[name] = [item.to_dict() for item in getattr(self, name)]
        for name in ("position_offset", "new_length", "old_resident_bytes", "attention_resident_bytes",
                     "migration_replacement_bytes", "transaction_peak_bytes", "final_resident_bytes"):
            result[name] = getattr(self, name)
        return result

    @classmethod
    def from_dict(cls, data):
        required = {item.name for item in fields(cls)} | {"schema_version"}
        _keys(data, required, "TieredKVTransition")
        cache = TieredKVCachePlan.from_dict(data["cache_plan"])
        committed = data["committed_pages"]
        if not isinstance(committed, list) or len(committed) > MAX_PLAN_SEGMENTS:
            raise ValueError("Committed pages exceed the page metadata limit")
        kept = data["retained_pages"]
        if not isinstance(kept, list):
            raise ValueError("Retained pages must be a list of pairs")
        result = cls(cache, data["past_length"], data["chunk_length"], data["mode"],
                     tuple(TieredKVPage.from_dict(page) for page in committed),
                     normalize_retained(kept))
        if not _strict_equal(data, result.to_dict()):
            raise ValueError("Tiered KV transition differs from canonical pages, migrations or byte counts")
        return result
