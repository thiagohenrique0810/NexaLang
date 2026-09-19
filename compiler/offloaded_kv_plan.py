"""Physical CPU residence for age tiers whose immutable cold pages live on disk.

The tier transition describes logical codec history. This plan separately counts
resident sources, fresh pages, destinations, and a bounded reload cache. No
payload is needed to admit the worst case for either append or prompt replacement.

Reload slots are admitted capacity, not a promise of residency: a call occupies
at most one slot per cold page its own prefix touches. Slots hold copies of
published bytes, so they change residency and I/O, never the logical page,
its codec history or its numerical result.
"""
from dataclasses import dataclass

from .model_ir import _integer
from .paged_kv_plan import MAX_ALLOCATION_BYTES
from .tiered_kv_plan import TieredKVCachePlan


# First touch admits a slot; once every slot is occupied the most recently
# loaded one is reused. Attention rescans the same pages in ascending order,
# where least-recently-used replacement would miss on every single page.
RELOAD_POLICY_ID = "CPU_RELOAD_FIRST_TOUCH_MRU_V1"


@dataclass(frozen=True)
class OffloadedKVPlan:
    tiers: TieredKVCachePlan
    max_chunk_length: int
    reload_slots: int = 1

    def __post_init__(self):
        if not isinstance(self.tiers, TieredKVCachePlan):
            raise ValueError("Offloaded KV requires a tiered layout")
        _integer(self.reload_slots, "reload_slots", 1)
        if self.reload_slots > self.tiers.max_pages:
            raise ValueError("Reload slots exceed the pages this context can hold")
        self.tiers.layout("f32").reservation_pages(self.max_chunk_length)
        if self.allocation_limit_bytes + self.reload_cache_capacity_bytes > MAX_ALLOCATION_BYTES:
            raise ValueError("Offloaded KV reservation exceeds the allocation range")

    @property
    def resident_capacity_bytes(self):
        hot = min(self.tiers.max_pages, self.tiers.policy.hot_pages)
        warm = min(self.tiers.max_pages - hot, self.tiers.policy.warm_pages)
        return (hot * self.tiers.layout("f32").page_allocation_bytes +
                warm * self.tiers.layout("q4").page_allocation_bytes)

    @property
    def reload_slot_bytes(self):
        return self.tiers.layout("q3").page_allocation_bytes

    @property
    def reload_cache_capacity_bytes(self):
        return self.reload_slots * self.reload_slot_bytes

    @property
    def allocation_limit_bytes(self):
        fresh = self.tiers.page_count(self.max_chunk_length)
        # Cold pages never migrate again. At most the resident source pages and
        # this chunk's fresh pages can need replacement destinations.
        replacements = min(self.tiers.max_pages,
                           self.tiers.policy.hot_pages + self.tiers.policy.warm_pages + fresh)
        return (self.resident_capacity_bytes +
                fresh * self.tiers.layout("f32").page_allocation_bytes +
                replacements * self.tiers.max_packed_page_allocation_bytes)

    def transition_memory(self, transition):
        if transition.cache_plan != self.tiers or transition.chunk_length > self.max_chunk_length:
            raise ValueError("Transition exceeds the offloaded KV plan")
        old = sum(p.page_allocation_bytes for p in transition.committed_pages if p.tier != "cold")
        attention = old + sum(p.page_allocation_bytes for p in transition.fresh_pages)
        migration = attention + transition.migration_replacement_bytes
        if migration > self.allocation_limit_bytes:
            raise ValueError("Offloaded KV transaction exceeds admitted page allocation")
        # A slot can only ever hold a cold page of this very prefix, so the
        # touched pages bound residency even when the cache outlives the call.
        slots = min(self.reload_slots, sum(p.tier == "cold" for p in transition.attention_pages))
        return {"old_resident_bytes": old, "attention_pages_bytes": attention,
                "migration_pages_bytes": migration,
                "final_resident_bytes": sum(p.page_allocation_bytes for p in transition.final_pages
                                             if p.tier != "cold"),
                "reload_slots_used": slots, "reload_slot_bytes": slots * self.reload_slot_bytes}

    def to_dict(self):
        return {"schema_version": 1, "kind": "cpu_cold_q3_backing_store",
                "max_chunk_length": self.max_chunk_length,
                "resident_capacity_bytes": self.resident_capacity_bytes,
                "page_allocation_limit_bytes": self.allocation_limit_bytes,
                "reload_slot_capacity_bytes": self.reload_slot_bytes,
                "reload_cache_capacity_bytes": self.reload_cache_capacity_bytes,
                "reload_replacement_policy": RELOAD_POLICY_ID,
                "reload_slots": self.reload_slots, "logical_codec_plan": self.tiers.to_dict()}
