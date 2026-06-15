from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SchedulerAuditError(RuntimeError):
    """Raised when scheduler sweep artifacts are incomplete."""


@dataclass(frozen=True)
class SchedulerArtifactAudit:
    manifest_dir: Path
    candidate_count: int
    summary_count: int
    trace_count: int
    scheduler_metadata_count: int
    job_id_count: int

    def message(self) -> str:
        return (
            "FastWAM scheduler artifact audit passed: "
            f"manifest_dir={self.manifest_dir} candidates={self.candidate_count} "
            f"summaries={self.summary_count} traces={self.trace_count} "
            f"scheduler_metadata={self.scheduler_metadata_count} "
            f"job_ids={self.job_id_count}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "ok",
            "manifest_dir": str(self.manifest_dir),
            "candidate_count": self.candidate_count,
            "summary_count": self.summary_count,
            "trace_count": self.trace_count,
            "scheduler_metadata_count": self.scheduler_metadata_count,
            "job_id_count": self.job_id_count,
        }


def audit_scheduler_artifacts(
    manifest_dir: str | Path,
    *,
    require_job_ids: bool = True,
    require_summaries: bool = True,
    require_trace_paths: bool = True,
    require_scheduler_metadata: bool = True,
    require_ok_status: bool = True,
) -> SchedulerArtifactAudit:
    root = Path(manifest_dir).expanduser()
    candidates_path = root / "scheduler_sweep_candidates.csv"
    job_map_path = root / "scheduler_sweep_job_map.csv"
    candidates = _read_csv(candidates_path)
    job_map = _read_job_map(job_map_path)
    errors: list[str] = []

    if not candidates:
        errors.append(f"candidate manifest is empty: {candidates_path}")

    summary_count = 0
    trace_count = 0
    scheduler_metadata_count = 0
    job_id_count = 0
    for row in candidates:
        candidate_id = str(row.get("candidate_id") or "")
        label = candidate_id or str(row.get("summary_path") or "<unknown>")
        summary_path = _required_path(row.get("summary_path"), label=label, field="summary_path")
        if summary_path is None:
            errors.append(f"{label} is missing summary_path")
            continue

        job_id = job_map.get(candidate_id) or job_map.get(str(summary_path)) or job_map.get(str(summary_path.resolve()))
        if job_id:
            job_id_count += 1
        elif require_job_ids:
            errors.append(f"{label} is missing job_id in {job_map_path}")

        if not summary_path.exists():
            if require_summaries:
                errors.append(f"{label} summary_path does not exist: {summary_path}")
            continue

        summary_count += 1
        summary = _read_summary(summary_path, label=label, errors=errors)
        if summary is None:
            continue
        if require_ok_status:
            status = str(summary.get("status") or "")
            if status != "ok":
                errors.append(f"{label} summary.status is {status!r}, expected 'ok'")
        trace_path = _trace_path(summary, summary_path)
        if trace_path is None:
            if require_trace_paths:
                errors.append(f"{label} summary is missing trace_path")
            continue
        if not trace_path.exists():
            if require_trace_paths:
                errors.append(f"{label} trace_path does not exist: {trace_path}")
            continue

        trace_count += 1
        events = _read_trace(trace_path, label=label, errors=errors)
        if events is None:
            continue
        if require_ok_status:
            run_end_status = _run_end_status(events)
            if run_end_status != "ok":
                errors.append(f"{label} trace run_end status is {run_end_status!r}, expected 'ok'")
        scheduler_metadata = _first_scheduler_metadata(events)
        if scheduler_metadata:
            scheduler_metadata_count += 1
        elif require_scheduler_metadata:
            errors.append(f"{label} trace has no scheduler backend_metadata: {trace_path}")
        missing_metadata = _missing_scheduler_metadata_fields(scheduler_metadata)
        if missing_metadata and require_scheduler_metadata:
            errors.append(
                f"{label} scheduler backend_metadata is missing {','.join(missing_metadata)}"
            )

    if errors:
        raise SchedulerAuditError("\n".join(errors))

    return SchedulerArtifactAudit(
        manifest_dir=root,
        candidate_count=len(candidates),
        summary_count=summary_count,
        trace_count=trace_count,
        scheduler_metadata_count=scheduler_metadata_count,
        job_id_count=job_id_count,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit FastWAM scheduler sweep artifacts")
    parser.add_argument("manifest_dir", help="Directory containing scheduler_sweep_*.csv files")
    parser.add_argument("--allow-missing-job-ids", action="store_true")
    parser.add_argument("--allow-missing-summaries", action="store_true")
    parser.add_argument("--allow-missing-trace-paths", action="store_true")
    parser.add_argument("--allow-missing-scheduler-metadata", action="store_true")
    parser.add_argument("--allow-non-ok-status", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    args = parser.parse_args(argv)

    try:
        report = audit_scheduler_artifacts(
            args.manifest_dir,
            require_job_ids=not args.allow_missing_job_ids,
            require_summaries=not args.allow_missing_summaries,
            require_trace_paths=not args.allow_missing_trace_paths,
            require_scheduler_metadata=not args.allow_missing_scheduler_metadata,
            require_ok_status=not args.allow_non_ok_status,
        )
    except (OSError, json.JSONDecodeError, SchedulerAuditError) as exc:
        print(f"scheduler artifact audit error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.message())
    return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_job_map(path: Path) -> dict[str, str]:
    rows = _read_csv(path)
    mapping: dict[str, str] = {}
    for row in rows:
        job_id = row.get("job_id") or row.get("superpod_job_id")
        if not job_id:
            continue
        candidate_id = row.get("candidate_id")
        summary_path = row.get("summary_path") or row.get("summary") or row.get("path")
        if candidate_id:
            mapping[str(candidate_id)] = str(job_id)
        if summary_path:
            path = Path(summary_path)
            mapping[str(path)] = str(job_id)
            mapping[str(path.resolve())] = str(job_id)
    return mapping


def _required_path(value: object, *, label: str, field: str) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    return Path(text)


def _read_summary(path: Path, *, label: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"{label} summary is not valid JSON: {path}: {exc}")
        return None
    if not isinstance(data, dict):
        errors.append(f"{label} summary is not a JSON object: {path}")
        return None
    return data


def _read_trace(path: Path, *, label: str, errors: list[str]) -> list[dict[str, Any]] | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"{label} trace could not be read: {path}: {exc}")
        return None
    events = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{label} trace line {index} is not valid JSON: {path}: {exc}")
            return None
        if isinstance(event, dict):
            events.append(event)
    return events


def _first_scheduler_metadata(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in events:
        metadata = event.get("backend_metadata")
        if isinstance(metadata, dict) and metadata.get("scheduler_name") is not None:
            return metadata
    return {}


def _run_end_status(events: list[dict[str, Any]]) -> str | None:
    for event in reversed(events):
        if event.get("event") == "run_end":
            status = event.get("status")
            return None if status is None else str(status)
    return None


def _missing_scheduler_metadata_fields(metadata: dict[str, Any]) -> list[str]:
    if not metadata:
        return []
    required = (
        "scheduler_name",
        "solver",
        "schedule_type",
        "num_inference_steps",
        "sigma_shift",
        "timestep_count",
        "timesteps",
        "sigmas",
        "deltas",
        "denoise_wall_ms",
    )
    return [key for key in required if key not in metadata]


def _trace_path(summary: dict[str, Any], summary_path: Path) -> Path | None:
    value = summary.get("trace_path")
    if value is None or str(value) == "":
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return (summary_path.parent / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
