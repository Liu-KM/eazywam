from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from eazywam.core.tracing import TraceWriter
from eazywam.core.types import InferenceRequest, InferenceResult


@dataclass
class BatchDispatchItem:
    request_id: str
    request: InferenceRequest
    event: str
    expected_horizon: int
    started_at: float
    payload: dict[str, Any]
    enqueued_at: float


@dataclass
class _PendingRequest:
    item: BatchDispatchItem
    done: threading.Event = field(default_factory=threading.Event)
    result: InferenceResult | None = None
    error: BaseException | None = None


BatchDispatchFn = Callable[[list[BatchDispatchItem], str], list[InferenceResult]]


class BatchDispatcher:
    """Dynamic request batcher for resident model serving.

    The dispatcher owns queueing only. The caller supplies the actual dispatch
    function so core batching can stay independent from a concrete backend or
    server implementation.
    """

    def __init__(
        self,
        *,
        dispatch_fn: BatchDispatchFn,
        trace: TraceWriter,
        max_batch_size: int,
        max_wait_time: float,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_wait_time < 0:
            raise ValueError("max_wait_time must be non-negative")
        self.dispatch_fn = dispatch_fn
        self.trace = trace
        self.max_batch_size = int(max_batch_size)
        self.max_wait_time = float(max_wait_time)
        self._condition = threading.Condition()
        self._queue: list[_PendingRequest] = []
        self._closed = False
        self._counter = itertools.count()
        self._worker = threading.Thread(
            target=self._run,
            name="eazywam-batch-dispatcher",
            daemon=True,
        )
        self._worker.start()

    def infer(
        self,
        *,
        request_id: str,
        request: InferenceRequest,
        event: str,
        expected_horizon: int,
        started_at: float,
        payload: dict[str, Any] | None = None,
    ) -> InferenceResult:
        pending = _PendingRequest(
            item=BatchDispatchItem(
                request_id=request_id,
                request=request,
                event=event,
                expected_horizon=expected_horizon,
                started_at=started_at,
                payload=payload or {},
                enqueued_at=time.perf_counter(),
            )
        )
        with self._condition:
            if self._closed:
                raise RuntimeError("batch dispatcher is closed")
            self._queue.append(pending)
            queue_size = len(self._queue)
            self._condition.notify()
        self.trace.write(
            "batch_request_enqueued",
            request_id=request_id,
            queue_size=queue_size,
            max_batch_size=self.max_batch_size,
            max_wait_time=self.max_wait_time,
        )
        pending.done.wait()
        if pending.error is not None:
            raise pending.error
        if pending.result is None:
            raise RuntimeError("batch dispatcher completed without a result")
        return pending.result

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._worker.join(timeout=5)

    def _run(self) -> None:
        while True:
            batch = self._next_batch()
            if not batch:
                return
            batch_id = f"batch-{next(self._counter):06d}"
            self._dispatch(batch, batch_id)

    def _next_batch(self) -> list[_PendingRequest]:
        with self._condition:
            while not self._queue and not self._closed:
                self._condition.wait()
            if not self._queue and self._closed:
                return []
            batch = [self._queue.pop(0)]
            deadline = batch[0].item.enqueued_at + self.max_wait_time
            while len(batch) < self.max_batch_size:
                if self._queue:
                    batch.append(self._queue.pop(0))
                    continue
                if self._closed or self.max_wait_time == 0:
                    break
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            return batch

    def _dispatch(self, batch: list[_PendingRequest], batch_id: str) -> None:
        items = [pending.item for pending in batch]
        queue_wait_ms = [
            max(0.0, (time.perf_counter() - item.enqueued_at) * 1000)
            for item in items
        ]
        self.trace.write(
            "batch_dispatch_start",
            batch_id=batch_id,
            request_ids=[item.request_id for item in items],
            batch_size=len(items),
            queue_wait_ms=queue_wait_ms,
        )
        start = time.perf_counter()
        try:
            results = self.dispatch_fn(items, batch_id)
            if len(results) != len(items):
                raise RuntimeError(
                    f"batch dispatch returned {len(results)} results for {len(items)} requests"
                )
        except BaseException as exc:
            dispatch_ms = (time.perf_counter() - start) * 1000
            self.trace.write(
                "batch_dispatch_end",
                batch_id=batch_id,
                batch_size=len(items),
                status="error",
                dispatch_ms=dispatch_ms,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            for pending in batch:
                pending.error = exc
                pending.done.set()
            return

        dispatch_ms = (time.perf_counter() - start) * 1000
        self.trace.write(
            "batch_dispatch_end",
            batch_id=batch_id,
            request_ids=[item.request_id for item in items],
            batch_size=len(items),
            status="ok",
            dispatch_ms=dispatch_ms,
        )
        for pending, result in zip(batch, results, strict=True):
            pending.result = result
            pending.done.set()
