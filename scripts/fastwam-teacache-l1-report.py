#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TARGETS = {
    "LIBERO": {
        "compare": "fastwam-libero-teacache-compare.json",
        "baseline": "fastwam-libero-eager-cache-summary.json",
        "variant": "fastwam-libero-teacache-summary.json",
    },
    "RoboTwin": {
        "compare": "fastwam-robotwin-teacache-compare.json",
        "baseline": "fastwam-robotwin-eager-cache-summary.json",
        "variant": "fastwam-robotwin-teacache-summary.json",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract FastWAM TeaCache L1 report rows from SuperPod JSON outputs."
    )
    parser.add_argument(
        "--report-root",
        default="runs/fastwam-teacache-l1-reports",
        help="Directory containing helper output JSON files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable rows instead of markdown.",
    )
    args = parser.parse_args()

    rows = [_row(target, files, Path(args.report_root)) for target, files in TARGETS.items()]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(_markdown_table(rows))
    return 0


def _row(target: str, files: dict[str, str], root: Path) -> dict[str, object]:
    compare = _load_json(root / files["compare"])
    baseline = _load_json(root / files["baseline"])
    variant = _load_json(root / files["variant"])
    baseline_success_rate = _success_rate(baseline, target=target)
    variant_success_rate = _success_rate(variant, target=target)
    fallback_values = _path(
        compare,
        "variant",
        "backend_metadata_values",
        "teacache_fallback_reason",
        "values",
    )
    speedup_blockers = _speedup_blockers(
        compare,
        fallback_values=fallback_values,
        baseline_success_rate=baseline_success_rate,
        variant_success_rate=variant_success_rate,
    )
    latency_mean_speedup = _path(
        compare,
        "metric_comparisons",
        "latency_ms.mean",
        "speedup",
    )
    denoise_mean_speedup = _path(
        compare,
        "metric_comparisons",
        "backend_metadata.denoise_wall_ms.mean",
        "speedup",
    )
    action_drift = _action_drift(compare)
    hit_rate = _path(
        compare,
        "variant",
        "backend_metadata",
        "teacache_hit_rate",
        "mean",
    )
    skipped_steps = _path(
        compare,
        "variant",
        "backend_metadata",
        "teacache_skipped_steps",
        "mean",
    )
    drift_score = _path(
        compare,
        "variant",
        "backend_metadata",
        "teacache_drift_score",
        "mean",
    )
    for name, value in (
        ("teacache_hit_rate_missing", hit_rate),
        ("teacache_skipped_steps_missing", skipped_steps),
        ("teacache_drift_score_missing", drift_score),
    ):
        if value is None:
            speedup_blockers.append(name)
    for name, value in (
        ("latency_mean_speedup_missing", latency_mean_speedup),
        ("denoise_mean_speedup_missing", denoise_mean_speedup),
        ("action_drift_missing", action_drift),
    ):
        if value is None:
            speedup_blockers.append(name)
    speedup_allowed = not speedup_blockers
    return {
        "target": target,
        "latency_mean_speedup": latency_mean_speedup,
        "denoise_mean_speedup": denoise_mean_speedup,
        "latency_mean_speedup_reportable": latency_mean_speedup if speedup_allowed else None,
        "denoise_mean_speedup_reportable": denoise_mean_speedup if speedup_allowed else None,
        "hit_rate": hit_rate,
        "skipped_steps": skipped_steps,
        "drift_score": drift_score,
        "action_drift": action_drift,
        "success_rate_baseline": baseline_success_rate,
        "success_rate_teacache": variant_success_rate,
        "fallback_reason": ", ".join(str(value) for value in fallback_values)
        if isinstance(fallback_values, list) and fallback_values
        else None,
        "compare_decision": compare.get("decision"),
        "output_gate_passed": compare.get("output_gate_passed"),
        "speedup_reportable": speedup_allowed,
        "speedup_blockers": speedup_blockers,
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing required JSON file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object in {path}")
    return payload


def _path(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _speedup_blockers(
    compare: dict[str, Any],
    *,
    fallback_values: object,
    baseline_success_rate: float | None,
    variant_success_rate: float | None,
) -> list[str]:
    blockers: list[str] = []
    decision = compare.get("decision")
    if decision != "faster":
        blockers.append(f"compare_decision:{compare.get('decision')}")
    if compare.get("output_gate_passed") is not True:
        blockers.append("output_gate_not_passed")
    if isinstance(fallback_values, list) and len(fallback_values) > 0:
        blockers.append("teacache_fallback_reason")
    baseline_profiles = _path(compare, "baseline", "profiles")
    variant_profiles = _path(compare, "variant", "profiles")
    if not isinstance(baseline_profiles, list):
        blockers.append("baseline_profiles_missing")
    elif "teacache" in baseline_profiles:
        blockers.append("baseline_has_teacache_profile")
    if not isinstance(variant_profiles, list):
        blockers.append("variant_profiles_missing")
    elif "teacache" not in variant_profiles:
        blockers.append("variant_missing_teacache_profile")
    if baseline_success_rate is None:
        blockers.append("baseline_success_rate_missing")
    if variant_success_rate is None:
        blockers.append("teacache_success_rate_missing")
    if (
        baseline_success_rate is not None
        and variant_success_rate is not None
        and variant_success_rate < baseline_success_rate
    ):
        blockers.append("success_rate_drop")
    return blockers


def _action_drift(compare: dict[str, Any]) -> float | None:
    details = compare.get("output_gate_details")
    if not isinstance(details, dict):
        return None
    observed = details.get("observed")
    if isinstance(observed, dict):
        values = [float(value) for value in observed.values() if isinstance(value, int | float)]
        if values:
            return max(values)
    value = details.get("max_observed")
    if isinstance(value, int | float):
        return float(value)
    return None


def _success_rate(summary: dict[str, Any], *, target: str) -> float | None:
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return None
    paths = (
        ("success_rate",),
        ("overall", "clean_mean_success_rate"),
        ("overall", "random_mean_success_rate"),
    )
    if target == "RoboTwin":
        paths = (
            ("success_rate",),
            ("overall", "random_mean_success_rate"),
            ("overall", "clean_mean_success_rate"),
        )
    for path in paths:
        value = _path(metrics, *path)
        if isinstance(value, int | float):
            return float(value)
    return None


def _markdown_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "Target",
        "latency mean speedup",
        "denoise mean speedup",
        "hit rate",
        "skipped steps",
        "drift score",
        "action drift",
        "success rate baseline",
        "success rate TeaCache",
        "fallback reason",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["target"]),
                    _fmt(row["latency_mean_speedup"]),
                    _fmt(row["denoise_mean_speedup"]),
                    _fmt(row["hit_rate"]),
                    _fmt(row["skipped_steps"]),
                    _fmt(row["drift_score"]),
                    _fmt(row["action_drift"]),
                    _fmt(row["success_rate_baseline"]),
                    _fmt(row["success_rate_teacache"]),
                    _fmt(row["fallback_reason"]),
                ]
            )
            + " |"
        )
    blocker_lines = _speedup_blocker_notes(rows)
    if blocker_lines:
        lines.append("")
        lines.extend(blocker_lines)
    return "\n".join(lines)


def _speedup_blocker_notes(rows: list[dict[str, object]]) -> list[str]:
    notes = []
    for row in rows:
        blockers = row.get("speedup_blockers")
        if not isinstance(blockers, list) or not blockers:
            continue
        notes.append(
            f"- {row['target']} speedup blocked: "
            + ", ".join(str(blocker) for blocker in blockers)
        )
    return notes


def _fmt(value: object) -> str:
    if value is None:
        return "TODO"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
