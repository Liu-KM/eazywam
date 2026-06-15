from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eazywam.evals.scheduler_acceptance import SchedulerAcceptanceError, validate_scheduler_report
from eazywam.evals.scheduler_audit import SchedulerAuditError, audit_scheduler_artifacts


class SchedulerBundleError(RuntimeError):
    """Raised when final scheduler evidence cannot be bundled."""


@dataclass(frozen=True)
class BundleArtifact:
    name: str
    path: Path
    required: bool
    exists: bool
    size_bytes: int | None
    sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": str(self.path),
            "required": self.required,
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class SchedulerEvidenceBundle:
    bundle_path: Path
    report_path: Path
    manifest_dir: Path | None
    row_count: int
    config_count: int
    artifact_count: int
    candidate_ids: tuple[str, ...]
    job_ids: tuple[str, ...]
    trace_paths: tuple[str, ...]

    def message(self) -> str:
        manifest = str(self.manifest_dir) if self.manifest_dir is not None else "none"
        return (
            "FastWAM scheduler evidence bundle written: "
            f"bundle={self.bundle_path} report={self.report_path} "
            f"manifest_dir={manifest} rows={self.row_count} "
            f"configs={self.config_count} artifacts={self.artifact_count}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "ok",
            "bundle_path": str(self.bundle_path),
            "report_path": str(self.report_path),
            "manifest_dir": None if self.manifest_dir is None else str(self.manifest_dir),
            "row_count": self.row_count,
            "config_count": self.config_count,
            "artifact_count": self.artifact_count,
            "candidate_ids": list(self.candidate_ids),
            "job_ids": list(self.job_ids),
            "trace_paths": list(self.trace_paths),
        }


def build_scheduler_evidence_bundle(
    report_json: str | Path,
    *,
    output_dir: str | Path | None = None,
    manifest_dir: str | Path | None = None,
    baseline_steps: int = 10,
    min_config_repeats: int = 1,
    allow_missing_quality_reference: bool = False,
) -> SchedulerEvidenceBundle:
    report_path = Path(report_json).expanduser()
    validate_scheduler_report(
        report_path,
        baseline_steps=baseline_steps,
        require_quality_reference=not allow_missing_quality_reference,
        min_config_repeats=min_config_repeats,
    )
    manifest_path = Path(manifest_dir).expanduser() if manifest_dir is not None else None
    if manifest_path is not None:
        audit_scheduler_artifacts(manifest_path)

    rows = _read_rows(report_path)
    config_rows = _read_rows(report_path.parent / "scheduler_config_summary.json")
    artifacts = _report_artifacts(report_path)
    if manifest_path is not None:
        artifacts.extend(_manifest_artifacts(manifest_path))
    missing = [artifact for artifact in artifacts if artifact.required and not artifact.exists]
    if missing:
        names = ", ".join(artifact.name for artifact in missing)
        raise SchedulerBundleError(f"missing required bundle artifacts: {names}")

    out_dir = Path(output_dir).expanduser() if output_dir is not None else report_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "scheduler_evidence_bundle.json"
    payload = {
        "schema_version": 1,
        "status": "ok",
        "report_path": str(report_path),
        "manifest_dir": None if manifest_path is None else str(manifest_path),
        "baseline_steps": baseline_steps,
        "min_config_repeats": min_config_repeats,
        "allow_missing_quality_reference": allow_missing_quality_reference,
        "row_count": len(rows),
        "config_count": len(config_rows),
        "artifacts": [artifact.to_dict() for artifact in artifacts],
        "runs": [_run_evidence(row) for row in rows],
        "config_summaries": [_config_evidence(row) for row in config_rows],
    }
    bundle_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return SchedulerEvidenceBundle(
        bundle_path=bundle_path,
        report_path=report_path,
        manifest_dir=manifest_path,
        row_count=len(rows),
        config_count=len(config_rows),
        artifact_count=len(artifacts),
        candidate_ids=tuple(sorted(_nonempty(row.get("candidate_id") for row in rows))),
        job_ids=tuple(sorted(_nonempty(row.get("job_id") for row in rows))),
        trace_paths=tuple(sorted(_nonempty(row.get("trace_path") for row in rows))),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a FastWAM scheduler evidence bundle manifest")
    parser.add_argument("report_json", help="scheduler_results.json from scheduler_report")
    parser.add_argument("--output-dir", default=None, help="Directory for scheduler_evidence_bundle.json")
    parser.add_argument("--manifest-dir", default=None, help="Optional scheduler sweep manifest directory")
    parser.add_argument("--baseline-steps", type=int, default=10)
    parser.add_argument("--min-config-repeats", type=int, default=1)
    parser.add_argument("--allow-missing-quality-reference", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    args = parser.parse_args(argv)

    try:
        bundle = build_scheduler_evidence_bundle(
            args.report_json,
            output_dir=args.output_dir,
            manifest_dir=args.manifest_dir,
            baseline_steps=args.baseline_steps,
            min_config_repeats=args.min_config_repeats,
            allow_missing_quality_reference=args.allow_missing_quality_reference,
        )
    except (
        OSError,
        json.JSONDecodeError,
        SchedulerAcceptanceError,
        SchedulerAuditError,
        SchedulerBundleError,
    ) as exc:
        print(f"scheduler bundle error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(bundle.to_dict(), indent=2, sort_keys=True))
    else:
        print(bundle.message())
    return 0


def _report_artifacts(report_path: Path) -> list[BundleArtifact]:
    return [
        _artifact("scheduler_results.json", report_path),
        _artifact("scheduler_results.csv", report_path.parent / "scheduler_results.csv"),
        _artifact("scheduler_config_summary.json", report_path.parent / "scheduler_config_summary.json"),
        _artifact("scheduler_config_summary.csv", report_path.parent / "scheduler_config_summary.csv"),
        _artifact("scheduler_report.html", report_path.parent / "scheduler_report.html"),
        _artifact("scheduler_final_report.md", report_path.parent / "scheduler_final_report.md"),
    ]


def _manifest_artifacts(manifest_dir: Path) -> list[BundleArtifact]:
    return [
        _artifact("scheduler_sweep_candidates.json", manifest_dir / "scheduler_sweep_candidates.json"),
        _artifact("scheduler_sweep_candidates.csv", manifest_dir / "scheduler_sweep_candidates.csv"),
        _artifact("scheduler_sweep_commands.sh", manifest_dir / "scheduler_sweep_commands.sh"),
        _artifact("scheduler_sweep_job_map.csv", manifest_dir / "scheduler_sweep_job_map.csv"),
        _artifact("scheduler_report_command.sh", manifest_dir / "scheduler_report_command.sh"),
    ]


def _artifact(name: str, path: Path, *, required: bool = True) -> BundleArtifact:
    if not path.exists():
        return BundleArtifact(
            name=name,
            path=path,
            required=required,
            exists=False,
            size_bytes=None,
            sha256=None,
        )
    return BundleArtifact(
        name=name,
        path=path,
        required=required,
        exists=True,
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SchedulerBundleError(f"expected JSON list: {path}")
    rows = [row for row in data if isinstance(row, dict)]
    if len(rows) != len(data):
        raise SchedulerBundleError(f"expected object rows: {path}")
    return rows


def _run_evidence(row: dict[str, Any]) -> dict[str, object]:
    return {
        "candidate_id": row.get("candidate_id"),
        "model_id": row.get("model_id"),
        "workload": row.get("workload"),
        "config_key": row.get("config_key"),
        "num_inference_steps": row.get("num_inference_steps"),
        "sigma_shift": row.get("sigma_shift"),
        "speedup": row.get("speedup"),
        "success_rate": row.get("success_rate"),
        "action_drift": row.get("action_drift"),
        "recommendation": row.get("recommendation"),
        "job_id": row.get("job_id"),
        "trace_path": row.get("trace_path"),
        "summary_path": row.get("summary_path"),
        "command": row.get("command"),
    }


def _config_evidence(row: dict[str, Any]) -> dict[str, object]:
    return {
        "model_id": row.get("model_id"),
        "workload": row.get("workload"),
        "config_key": row.get("config_key"),
        "repeats": row.get("repeats"),
        "speedup_mean": row.get("speedup_mean"),
        "success_rate_mean": row.get("success_rate_mean"),
        "action_drift_mean": row.get("action_drift_mean"),
        "aggregate_pareto": row.get("aggregate_pareto"),
        "aggregate_recommendation": row.get("aggregate_recommendation"),
        "candidate_ids": row.get("candidate_ids"),
        "job_ids": row.get("job_ids"),
        "trace_paths": row.get("trace_paths"),
    }


def _nonempty(values: object) -> set[str]:
    return {str(value) for value in values if value not in {None, ""}}


if __name__ == "__main__":
    raise SystemExit(main())
