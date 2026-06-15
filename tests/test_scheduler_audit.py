from __future__ import annotations

import csv
import json

import pytest

from eazywam.evals.scheduler_audit import SchedulerAuditError, audit_scheduler_artifacts


def test_scheduler_audit_accepts_complete_manifest(tmp_path) -> None:
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    summary_path = tmp_path / "summaries" / "candidate-summary.json"
    trace_path = tmp_path / "traces" / "trace.jsonl"
    _write_summary(summary_path, trace_path)
    _write_manifest(
        manifest_dir,
        rows=[
            {
                "candidate_id": "fastwam-libero-steps10-shiftnull",
                "summary_path": str(summary_path),
                "job_id": "451000",
            }
        ],
    )

    report = audit_scheduler_artifacts(manifest_dir)

    assert report.candidate_count == 1
    assert report.summary_count == 1
    assert report.trace_count == 1
    assert report.scheduler_metadata_count == 1
    assert report.job_id_count == 1


def test_scheduler_audit_reports_missing_evidence(tmp_path) -> None:
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    summary_path = tmp_path / "summaries" / "candidate-summary.json"
    _write_manifest(
        manifest_dir,
        rows=[
            {
                "candidate_id": "fastwam-libero-steps6-shift3",
                "summary_path": str(summary_path),
                "job_id": "",
            }
        ],
    )

    with pytest.raises(SchedulerAuditError) as exc_info:
        audit_scheduler_artifacts(manifest_dir)

    message = str(exc_info.value)
    assert "missing job_id" in message
    assert "summary_path does not exist" in message


def test_scheduler_audit_reports_missing_trace_path(tmp_path) -> None:
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    summary_path = tmp_path / "summaries" / "candidate-summary.json"
    trace_path = tmp_path / "traces" / "missing.jsonl"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps({"trace_path": str(trace_path)}), encoding="utf-8")
    _write_manifest(
        manifest_dir,
        rows=[
            {
                "candidate_id": "fastwam-robotwin-steps8-shift5",
                "summary_path": str(summary_path),
                "job_id": "451100",
            }
        ],
    )

    with pytest.raises(SchedulerAuditError) as exc_info:
        audit_scheduler_artifacts(manifest_dir)

    assert "trace_path does not exist" in str(exc_info.value)


def test_scheduler_audit_reports_missing_scheduler_metadata(tmp_path) -> None:
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    summary_path = tmp_path / "summaries" / "candidate-summary.json"
    trace_path = tmp_path / "traces" / "trace.jsonl"
    summary_path.parent.mkdir(parents=True)
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text('{"event": "run_end", "status": "ok"}\n', encoding="utf-8")
    summary_path.write_text(json.dumps({"trace_path": str(trace_path)}), encoding="utf-8")
    _write_manifest(
        manifest_dir,
        rows=[
            {
                "candidate_id": "fastwam-libero-steps6-shift3",
                "summary_path": str(summary_path),
                "job_id": "451200",
            }
        ],
    )

    with pytest.raises(SchedulerAuditError) as exc_info:
        audit_scheduler_artifacts(manifest_dir)

    assert "trace has no scheduler backend_metadata" in str(exc_info.value)


def test_scheduler_audit_reports_incomplete_scheduler_metadata(tmp_path) -> None:
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    summary_path = tmp_path / "summaries" / "candidate-summary.json"
    trace_path = tmp_path / "traces" / "trace.jsonl"
    _write_summary(
        summary_path,
        trace_path,
        backend_metadata={
            "scheduler_name": "fastwam_flowmatch_euler",
            "num_inference_steps": 6,
        },
    )
    _write_manifest(
        manifest_dir,
        rows=[
            {
                "candidate_id": "fastwam-libero-steps6-shift3",
                "summary_path": str(summary_path),
                "job_id": "451201",
            }
        ],
    )

    with pytest.raises(SchedulerAuditError) as exc_info:
        audit_scheduler_artifacts(manifest_dir)

    message = str(exc_info.value)
    assert "scheduler backend_metadata is missing" in message
    assert "solver" in message
    assert "sigmas" in message


def test_scheduler_audit_reports_non_ok_summary_status(tmp_path) -> None:
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    summary_path = tmp_path / "summaries" / "candidate-summary.json"
    trace_path = tmp_path / "traces" / "trace.jsonl"
    _write_summary(summary_path, trace_path, summary_status="error")
    _write_manifest(
        manifest_dir,
        rows=[
            {
                "candidate_id": "fastwam-libero-steps6-shift3",
                "summary_path": str(summary_path),
                "job_id": "451202",
            }
        ],
    )

    with pytest.raises(SchedulerAuditError) as exc_info:
        audit_scheduler_artifacts(manifest_dir)

    assert "summary.status is 'error', expected 'ok'" in str(exc_info.value)


def test_scheduler_audit_reports_non_ok_trace_status(tmp_path) -> None:
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    summary_path = tmp_path / "summaries" / "candidate-summary.json"
    trace_path = tmp_path / "traces" / "trace.jsonl"
    _write_summary(summary_path, trace_path, run_end_status="error")
    _write_manifest(
        manifest_dir,
        rows=[
            {
                "candidate_id": "fastwam-libero-steps6-shift3",
                "summary_path": str(summary_path),
                "job_id": "451203",
            }
        ],
    )

    with pytest.raises(SchedulerAuditError) as exc_info:
        audit_scheduler_artifacts(manifest_dir)

    assert "trace run_end status is 'error', expected 'ok'" in str(exc_info.value)


def test_scheduler_audit_allows_partial_exploratory_runs(tmp_path) -> None:
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    summary_path = tmp_path / "summaries" / "candidate-summary.json"
    _write_manifest(
        manifest_dir,
        rows=[
            {
                "candidate_id": "fastwam-libero-steps6-shift3",
                "summary_path": str(summary_path),
                "job_id": "",
            }
        ],
    )

    report = audit_scheduler_artifacts(
        manifest_dir,
        require_job_ids=False,
        require_summaries=False,
        require_trace_paths=False,
        require_scheduler_metadata=False,
    )

    assert report.candidate_count == 1
    assert report.summary_count == 0
    assert report.trace_count == 0
    assert report.scheduler_metadata_count == 0
    assert report.job_id_count == 0


def _write_summary(
    summary_path,
    trace_path,
    backend_metadata=None,
    *,
    summary_status: str = "ok",
    run_end_status: str = "ok",
) -> None:
    summary_path.parent.mkdir(parents=True)
    trace_path.parent.mkdir(parents=True)
    metadata = backend_metadata or {
        "scheduler_name": "fastwam_flowmatch_euler",
        "solver": "euler",
        "schedule_type": "shifted_flowmatch",
        "num_inference_steps": 10,
        "sigma_shift": None,
        "timestep_count": 10,
        "timesteps": {"count": 10, "first": 1000.0, "last": 90.9, "min": 90.9, "max": 1000.0},
        "sigmas": {"count": 10, "first": 1.0, "last": 0.0909, "min": 0.0909, "max": 1.0},
        "deltas": {"count": 10, "first": -0.1818, "last": -0.0909, "min": -0.1818, "max": -0.0909},
        "denoise_wall_ms": 80.0,
    }
    trace_path.write_text(
        json.dumps({"event": "inference_end", "backend_metadata": metadata}) + "\n"
        + json.dumps({"event": "run_end", "status": run_end_status}) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps({"trace_path": str(trace_path), "status": summary_status}),
        encoding="utf-8",
    )


def _write_manifest(manifest_dir, *, rows: list[dict[str, str]]) -> None:
    fields = [
        "candidate_id",
        "model_id",
        "workload",
        "num_inference_steps",
        "sigma_shift",
        "phase",
        "command",
        "summary_path",
    ]
    candidate_rows = []
    for row in rows:
        candidate_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "model_id": "fastwam-libero",
                "workload": "libero-single-task",
                "num_inference_steps": "10",
                "sigma_shift": "null",
                "phase": "baseline",
                "command": "wam eval fastwam-libero",
                "summary_path": row["summary_path"],
            }
        )
    with (manifest_dir / "scheduler_sweep_candidates.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidate_rows)
    with (manifest_dir / "scheduler_sweep_job_map.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "summary_path", "job_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "summary_path": row["summary_path"],
                    "job_id": row["job_id"],
                }
            )
