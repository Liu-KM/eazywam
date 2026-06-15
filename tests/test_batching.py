from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eazywam.core.batching import BatchDispatchItem, BatchDispatcher
from eazywam.core.backend_session import BackendSession
from eazywam.core.manifest import load_builtin_manifest
from eazywam.core.tracing import TraceWriter
from eazywam.core.types import (
    ActionChunk,
    InferenceRequest,
    InferenceResult,
    Observation,
    RuntimeInfo,
)


def _read_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _request(prompt: str = "batch request", *, horizon: int = 2) -> InferenceRequest:
    return InferenceRequest(
        observation=Observation(images={"primary": []}, prompt=prompt),
        action_horizon=horizon,
        replan_steps=horizon,
    )


class _NoBatchBackend:
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, request: InferenceRequest) -> InferenceResult:
        self.calls += 1
        return InferenceResult(
            action_chunk=ActionChunk(
                actions=[[float(self.calls)] for _ in range(request.action_horizon)]
            )
        )

    def runtime_info(self) -> RuntimeInfo:
        return RuntimeInfo(
            manifest_id="fake-open-loop",
            model_name="Fake Open Loop",
            backend="fake",
            processor="passthrough",
            source_repo=None,
            mode="serve",
            device="cpu",
            dtype="fp32",
        )

    def close(self) -> None:
        return


def test_batch_dispatcher_groups_concurrent_requests(tmp_path) -> None:
    trace = TraceWriter(tmp_path / "trace.jsonl", "run")
    dispatched_batch_sizes: list[int] = []

    def dispatch(items: list[BatchDispatchItem], batch_id: str) -> list[InferenceResult]:
        dispatched_batch_sizes.append(len(items))
        return [
            InferenceResult(action_chunk=ActionChunk(actions=[[float(index)]]))
            for index, _item in enumerate(items)
        ]

    dispatcher = BatchDispatcher(
        dispatch_fn=dispatch,
        trace=trace,
        max_batch_size=2,
        max_wait_time=0.25,
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    dispatcher.infer,
                    request_id=f"req-{index}",
                    request=_request(f"request {index}"),
                    event="inference_end",
                    expected_horizon=1,
                    started_at=time.perf_counter(),
                    payload={"request_index": index},
                )
                for index in range(2)
            ]
            results = [future.result(timeout=2) for future in futures]
    finally:
        dispatcher.close()
        trace.close()

    assert dispatched_batch_sizes == [2]
    assert [result.action_chunk.actions for result in results] == [
        [[0.0]],
        [[1.0]],
    ]
    events = _read_events(trace.path)
    names = [str(event["event"]) for event in events]
    assert names.count("batch_request_enqueued") == 2
    dispatch_start = [event for event in events if event["event"] == "batch_dispatch_start"][0]
    dispatch_end = [event for event in events if event["event"] == "batch_dispatch_end"][0]
    assert dispatch_start["batch_size"] == 2
    assert dispatch_end["batch_size"] == 2
    assert dispatch_end["status"] == "ok"


def test_backend_session_batch_falls_back_to_loop_infer(tmp_path) -> None:
    manifest = load_builtin_manifest("fake-open-loop")
    backend = _NoBatchBackend()
    trace = TraceWriter(tmp_path / "trace.jsonl", "run", backend.runtime_info())
    session = BackendSession(
        manifest=manifest,
        profiles=[],
        backend=backend,
        processor=None,
        trace=trace,
    )
    started_at = time.perf_counter()
    items = [
        BatchDispatchItem(
            request_id="req-0",
            request=_request("first"),
            event="inference_end",
            expected_horizon=2,
            started_at=started_at,
            payload={"request_id": "req-0"},
            enqueued_at=started_at,
        ),
        BatchDispatchItem(
            request_id="req-1",
            request=_request("second"),
            event="inference_end",
            expected_horizon=2,
            started_at=started_at,
            payload={"request_id": "req-1"},
            enqueued_at=started_at,
        ),
    ]

    results = session.infer_batch_and_trace(items, batch_id="batch-test")
    trace.close()

    assert backend.calls == 2
    assert [result.action_chunk.actions[0][0] for result in results] == [1.0, 2.0]
    events = _read_events(trace.path)
    inference_events = [event for event in events if event["event"] == "inference_end"]
    assert len(inference_events) == 2
    assert {event["request_id"] for event in inference_events} == {"req-0", "req-1"}
    assert all(event["batch_id"] == "batch-test" for event in inference_events)
    assert all(event["batch_size"] == 2 for event in inference_events)
    assert all(
        event["batch_fallback_reason"] == "infer_batch_unavailable"
        for event in inference_events
    )
