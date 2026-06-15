from __future__ import annotations

import json
from typing import Any

from eazywam.core.batch_client import RemoteInferenceClient
from eazywam.core.types import InferenceRequest, Observation


class _Response:
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {
                "action_chunk": {"actions": [[1.0, 2.0]]},
                "backend_metadata": {"teacache_mode": "auto"},
            }
        ).encode("utf-8")


def test_remote_inference_client_serializes_runtime_options(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    request = InferenceRequest(
        observation=Observation(
            images={"primary": [[[0, 0, 0]]]},
            prompt="remote teacache",
        ),
        action_horizon=32,
        replan_steps=10,
        runtime_options={
            "dit_cache_mode": "video_kv",
            "teacache_mode": "auto",
            "teacache_threshold": 0.05,
            "teacache_warmup_steps": 1,
            "cuda_graph_mode": "off",
        },
    )

    result = RemoteInferenceClient("http://127.0.0.1:8000", timeout_s=3).infer(request)

    assert captured["url"] == "http://127.0.0.1:8000/infer"
    assert captured["timeout"] == 3
    assert captured["payload"]["runtime_options"] == {
        "dit_cache_mode": "video_kv",
        "teacache_mode": "auto",
        "teacache_threshold": 0.05,
        "teacache_warmup_steps": 1,
        "cuda_graph_mode": "off",
    }
    assert result.backend_metadata["teacache_mode"] == "auto"
