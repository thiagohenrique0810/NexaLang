"""Continuous Batching Scheduler — manages multiple inference requests."""

import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
from collections import deque

logger = logging.getLogger("turboruntime.scheduler")


class RequestState(Enum):
    QUEUED = auto()
    PREFILLING = auto()
    DECODING = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class InferenceRequest:
    request_id: str = ""
    prompt: str = ""
    input_ids: Any = None
    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    state: RequestState = RequestState.QUEUED
    generated_ids: list = field(default_factory=list)
    generated_text: str = ""
    created_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    ttft: float = 0.0  # Time to first token
    cache_pos: int = 0


class Scheduler:
    def __init__(self, max_batch: int = 8, max_ctx: int = 2048, timeout_ms: int = 30000):
        self.max_batch = max_batch
        self.max_ctx = max_ctx
        self.timeout_ms = timeout_ms
        self.queue: deque[InferenceRequest] = deque()
        self.active: dict[str, InferenceRequest] = {}
        self._lock = threading.Lock()
        self._counter = 0

    def submit(self, prompt: str, input_ids: Any = None,
               max_tokens: int = 128, temperature: float = 1.0,
               top_p: float = 1.0, top_k: int = 0) -> InferenceRequest:
        with self._lock:
            self._counter += 1
            req = InferenceRequest(
                request_id=f"req-{self._counter}",
                prompt=prompt,
                input_ids=input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                created_at=time.time(),
            )
            self.queue.append(req)
            return req

    def get_batch(self) -> list[InferenceRequest]:
        """Get next batch of requests to process."""
        with self._lock:
            batch = []
            # First, add active decoding requests
            for req in list(self.active.values()):
                if req.state == RequestState.DECODING:
                    batch.append(req)
                    if len(batch) >= self.max_batch:
                        break

            # Then fill remaining slots from queue
            while self.queue and len(batch) < self.max_batch:
                req = self.queue.popleft()
                req.state = RequestState.PREFILLING
                req.started_at = time.time()
                self.active[req.request_id] = req
                batch.append(req)

            return batch

    def complete(self, request_id: str, generated_text: str):
        with self._lock:
            if request_id in self.active:
                req = self.active.pop(request_id)
                req.state = RequestState.COMPLETED
                req.generated_text = generated_text
                req.finished_at = time.time()
                return req
        return None

    def fail(self, request_id: str, error: str):
        with self._lock:
            if request_id in self.active:
                req = self.active.pop(request_id)
                req.state = RequestState.FAILED
                req.generated_text = f"[ERROR] {error}"
                req.finished_at = time.time()

    @property
    def pending_count(self) -> int:
        return len(self.queue) + len(self.active)

    @property
    def metrics(self) -> dict:
        return {
            'queue_size': len(self.queue),
            'active_size': len(self.active),
            'max_batch': self.max_batch,
            'total_processed': self._counter,
        }
