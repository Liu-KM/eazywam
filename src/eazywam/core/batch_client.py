from __future__ import annotations

import json
import urllib.request
from typing import Any

from eazywam.core.types import ActionChunk, InferenceRequest, InferenceResult


class RemoteInferenceClient:
    """Tiny JSON client for a resident EazyWAM `/infer` endpoint."""

    def __init__(self, endpoint: str, *, timeout_s: float = 300.0) -> None:
        self.endpoint = _infer_url(endpoint)
        self.timeout_s = float(timeout_s)

    def infer(self, request: InferenceRequest) -> InferenceResult:
        payload = {
            "observation": _jsonable(request.observation.to_dict()),
            "action_horizon": request.action_horizon,
            "replan_steps": request.replan_steps,
            "reset": request.reset,
            "runtime_options": _jsonable(request.runtime_options),
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(http_request, timeout=self.timeout_s) as response:
            raw = json.loads(response.read().decode("utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("remote inference response must be a JSON object")
        return inference_result_from_payload(raw)


def inference_result_from_payload(payload: dict[str, Any]) -> InferenceResult:
    action_chunk = payload.get("action_chunk")
    if not isinstance(action_chunk, dict):
        raise ValueError("remote inference response is missing action_chunk")
    actions = action_chunk.get("actions")
    if not isinstance(actions, list):
        raise ValueError("remote inference response action_chunk.actions must be a list")
    return InferenceResult(
        action_chunk=ActionChunk(
            actions=[
                [float(value) for value in row]
                for row in actions
                if isinstance(row, list)
            ]
        ),
        warnings=_str_list(payload.get("warnings")),
        backend_metadata=_dict_or_empty(payload.get("backend_metadata")),
        timing=_dict_or_empty(payload.get("timing")),
        memory=_dict_or_empty(payload.get("memory")),
        future_frames=payload.get("future_frames")
        if isinstance(payload.get("future_frames"), dict)
        else None,
        value=payload.get("value"),
    )


def _infer_url(endpoint: str) -> str:
    text = endpoint.rstrip("/")
    if text.endswith("/infer"):
        return text
    return f"{text}/infer"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist())
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _dict_or_empty(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
