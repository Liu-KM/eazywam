from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def episode_indices(total: int, *, shard_id: int = 0, num_shards: int = 1) -> list[int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must be in [0, num_shards)")
    return [index for index in range(total) if index % num_shards == shard_id]


def merge_shard_summaries(paths: list[str | Path]) -> dict[str, Any]:
    summaries = [_load_summary(Path(path)) for path in paths]
    successes = sum(int(summary.get("successes", 0)) for summary in summaries)
    completed = sum(int(summary.get("total_episodes", 0)) for summary in summaries)
    requested_values = [
        int(summary.get("requested_episodes", summary.get("total_episodes", 0)))
        for summary in summaries
    ]
    requested = max(requested_values) if requested_values else 0
    return {
        "status": "ok",
        "requested_episodes": requested,
        "completed_episodes": completed,
        "successes": successes,
        "failed_episodes": max(0, completed - successes),
        "skipped_episodes": max(0, requested - completed),
        "success_rate": float(successes) / float(completed) if completed else 0.0,
        "shards": summaries,
    }


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"shard summary must be a JSON object: {path}")
    metrics = payload.get("metrics")
    return dict(metrics) if isinstance(metrics, dict) else payload
