from __future__ import annotations

import csv
import json

import pytest

from eazywam.evals.scheduler_acceptance import SchedulerAcceptanceError
from eazywam.evals.scheduler_bundle import build_scheduler_evidence_bundle
from eazywam.evals.scheduler_report import build_report, write_report_outputs


def test_scheduler_bundle_writes_hashed_evidence_manifest(tmp_path) -> None:
    summaries = _write_complete_report_inputs(tmp_path)
    runs = build_report(summaries)
    outputs = write_report_outputs(runs, tmp_path / "report")
    manifest_dir = tmp_path / "manifest"
    _write_manifest(manifest_dir, runs)

    bundle = build_scheduler_evidence_bundle(
        outputs["json"],
        output_dir=tmp_path / "bundle",
        manifest_dir=manifest_dir,
    )
    payload = json.loads(bundle.bundle_path.read_text(encoding="utf-8"))
    artifacts = {artifact["name"]: artifact for artifact in payload["artifacts"]}

    assert bundle.row_count == 6
    assert bundle.config_count == 6
    assert bundle.artifact_count == 11
    assert "fastwam-libero-steps6-shift3.0-libero-fast" in bundle.candidate_ids
    assert "451101" in bundle.job_ids
    assert artifacts["scheduler_results.json"]["sha256"]
    assert artifacts["scheduler_sweep_job_map.csv"]["sha256"]
    assert payload["runs"][1]["command"].startswith("wam eval fastwam-libero --opt scheduler")
    assert payload["config_summaries"][0]["aggregate_recommendation"]


def test_scheduler_bundle_rejects_invalid_report_artifacts(tmp_path) -> None:
    summaries = _write_complete_report_inputs(tmp_path)
    runs = build_report(summaries)
    outputs = write_report_outputs(runs, tmp_path / "report")
    (tmp_path / "report" / "scheduler_config_summary.csv").unlink()

    with pytest.raises(SchedulerAcceptanceError) as exc_info:
        build_scheduler_evidence_bundle(outputs["json"], output_dir=tmp_path / "bundle")

    assert "report artifact is missing" in str(exc_info.value)


def _write_complete_report_inputs(root):
    return [
        _write_run(
            root,
            name="libero-baseline",
            model_id="fastwam-libero",
            workload="libero-single-task",
            steps=10,
            total_ms=100.0,
            denoise_ms=80.0,
            success_rate=1.0,
            action_mean=0.10,
            job_id="451100",
        ),
        _write_run(
            root,
            name="libero-fast",
            model_id="fastwam-libero",
            workload="libero-single-task",
            steps=6,
            sigma_shift=3.0,
            total_ms=60.0,
            denoise_ms=45.0,
            success_rate=1.0,
            action_mean=0.12,
            job_id="451101",
        ),
        _write_run(
            root,
            name="libero-reference",
            model_id="fastwam-libero",
            workload="libero-single-task",
            steps=12,
            total_ms=120.0,
            denoise_ms=96.0,
            success_rate=1.0,
            action_mean=0.09,
            job_id="451102",
        ),
        _write_run(
            root,
            name="robotwin-baseline",
            model_id="fastwam-robotwin",
            workload="robotwin-single-task",
            steps=10,
            total_ms=140.0,
            denoise_ms=112.0,
            success_rate=1.0,
            action_mean=0.20,
            job_id="451200",
        ),
        _write_run(
            root,
            name="robotwin-fast",
            model_id="fastwam-robotwin",
            workload="robotwin-single-task",
            steps=8,
            sigma_shift=5.0,
            total_ms=100.0,
            denoise_ms=80.0,
            success_rate=1.0,
            action_mean=0.21,
            job_id="451201",
        ),
        _write_run(
            root,
            name="robotwin-reference",
            model_id="fastwam-robotwin",
            workload="robotwin-single-task",
            steps=12,
            total_ms=160.0,
            denoise_ms=128.0,
            success_rate=1.0,
            action_mean=0.19,
            job_id="451202",
        ),
    ]


def _write_manifest(manifest_dir, runs) -> None:
    manifest_dir.mkdir()
    rows = [
        {
            "candidate_id": run.candidate_id,
            "model_id": run.model_id,
            "workload": run.workload,
            "num_inference_steps": run.num_inference_steps,
            "sigma_shift": "null" if run.sigma_shift is None else f"{run.sigma_shift:.4g}",
            "phase": "test",
            "command": run.command,
            "summary_path": str(run.summary_path),
            "job_id": run.job_id,
        }
        for run in runs
    ]
    candidate_fields = [
        "candidate_id",
        "model_id",
        "workload",
        "num_inference_steps",
        "sigma_shift",
        "phase",
        "command",
        "summary_path",
        "job_id",
    ]
    with (manifest_dir / "scheduler_sweep_candidates.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=candidate_fields)
        writer.writeheader()
        writer.writerows(rows)
    (manifest_dir / "scheduler_sweep_candidates.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )
    with (manifest_dir / "scheduler_sweep_job_map.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "summary_path", "job_id"])
        writer.writeheader()
        writer.writerows(
            {
                "candidate_id": row["candidate_id"],
                "summary_path": row["summary_path"],
                "job_id": row["job_id"],
            }
            for row in rows
        )
    (manifest_dir / "scheduler_sweep_commands.sh").write_text(
        "\n".join(str(row["command"]) for row in rows) + "\n",
        encoding="utf-8",
    )
    (manifest_dir / "scheduler_report_command.sh").write_text(
        "python -m eazywam.evals.scheduler_report --output-dir report summaries/*.json\n",
        encoding="utf-8",
    )


def _write_run(
    root,
    *,
    name: str,
    model_id: str,
    workload: str,
    steps: int,
    total_ms: float,
    denoise_ms: float,
    success_rate: float,
    action_mean: float,
    sigma_shift: float | None = None,
    job_id: str,
):
    run_dir = root / name
    run_dir.mkdir()
    trace_path = run_dir / "trace.jsonl"
    candidate_id = f"{model_id}-steps{steps}-shift{sigma_shift or 'null'}-{name}"
    events = [
        _event(
            name,
            "native_eval_plan",
            manifest_id=model_id,
            runtime_options={
                "scheduler_name": "fastwam_flowmatch_euler",
                "solver": "euler",
                "schedule_type": "shifted_flowmatch",
                "num_inference_steps": steps,
                "sigma_shift": sigma_shift,
            },
            candidate_id=candidate_id,
            job_id=job_id,
        ),
        _event(
            name,
            "inference_end",
            manifest_id=model_id,
            timing={"total_ms": total_ms},
            action_summary={
                "mean": action_mean,
                "min": action_mean - 0.1,
                "max": action_mean + 0.1,
                "max_abs": abs(action_mean) + 0.1,
            },
            backend_metadata={
                "scheduler_name": "fastwam_flowmatch_euler",
                "solver": "euler",
                "schedule_type": "shifted_flowmatch",
                "num_inference_steps": steps,
                "sigma_shift": sigma_shift,
                "timestep_count": steps,
                "timesteps": {
                    "count": steps,
                    "first": 1000.0,
                    "last": 100.0,
                    "min": 100.0,
                    "max": 1000.0,
                },
                "sigmas": {
                    "count": steps,
                    "first": 1.0,
                    "last": 0.1,
                    "min": 0.1,
                    "max": 1.0,
                },
                "deltas": {
                    "count": steps,
                    "first": -0.1,
                    "last": -0.1,
                    "min": -0.1,
                    "max": -0.1,
                },
                "denoise_wall_ms": denoise_ms,
            },
        ),
        _event(name, "episode_end", manifest_id=model_id, success=success_rate == 1.0),
        _event(name, "run_end", manifest_id=model_id, status="ok"),
    ]
    trace_path.write_text(
        "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n",
        encoding="utf-8",
    )
    summary_path = run_dir / "summary.json"
    sigma = "null" if sigma_shift is None else f"{sigma_shift:.4g}"
    summary_path.write_text(
        json.dumps(
            {
                "run_id": name,
                "candidate_id": candidate_id,
                "model_id": model_id,
                "workload": workload,
                "trace_path": str(trace_path),
                "status": "ok",
                "command": {
                    "display": (
                        f"wam eval {model_id} --opt scheduler "
                        f"--set num_inference_steps={steps} --set sigma_shift={sigma}"
                    )
                },
                "metrics": {"success_rate": success_rate, "job_id": job_id},
            }
        ),
        encoding="utf-8",
    )
    return summary_path


def _event(run_id: str, event: str, **payload):
    manifest_id = payload.pop("manifest_id")
    base = {
        "schema_version": 1,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "run_id": run_id,
        "event": event,
        "manifest_id": manifest_id,
        "backend": "fastwam",
        "processor": "fastwam_libero",
        "mode": "simulator_eval",
        "model_name": "FastWAM",
        "source_repo": "yuantianyuan01/FastWAM",
    }
    base.update(payload)
    return base
