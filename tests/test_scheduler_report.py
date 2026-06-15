from __future__ import annotations

import json

import pytest

from eazywam.evals.scheduler_acceptance import validate_scheduler_report
from eazywam.evals.scheduler_report import (
    build_config_summaries,
    build_report,
    write_report_outputs,
)


def test_scheduler_report_computes_speedup_pareto_and_action_drift(tmp_path) -> None:
    baseline = _write_run(
        tmp_path,
        name="baseline",
        steps=10,
        total_ms=100.0,
        denoise_ms=80.0,
        success_rate=1.0,
        action_mean=0.10,
    )
    faster = _write_run(
        tmp_path,
        name="faster",
        steps=6,
        sigma_shift=3.0,
        total_ms=60.0,
        denoise_ms=45.0,
        success_rate=1.0,
        action_mean=0.12,
        job_id="451001",
    )
    weak = _write_run(
        tmp_path,
        name="weak",
        steps=4,
        sigma_shift=3.0,
        total_ms=50.0,
        denoise_ms=36.0,
        success_rate=0.5,
        action_mean=0.30,
    )
    ten_step_shifted = _write_run(
        tmp_path,
        name="ten-step-shifted",
        steps=10,
        sigma_shift=3.0,
        total_ms=95.0,
        denoise_ms=76.0,
        success_rate=1.0,
        action_mean=0.11,
    )

    runs = build_report([baseline, faster, weak, ten_step_shifted])
    by_config = {run.config_key: run for run in runs}

    assert by_config["steps=10,sigma_shift=null"].speedup == pytest.approx(1.0)
    assert by_config["steps=10,sigma_shift=null"].recommendation == "baseline"
    assert by_config["steps=10,sigma_shift=3"].recommendation != "baseline"
    assert by_config["steps=6,sigma_shift=3"].solver == "euler"
    assert by_config["steps=6,sigma_shift=3"].schedule_source == "generated"
    assert by_config["steps=6,sigma_shift=3"].speedup == pytest.approx(100.0 / 60.0)
    assert by_config["steps=6,sigma_shift=3"].success_delta == pytest.approx(0.0)
    assert by_config["steps=6,sigma_shift=3"].timestep_count == 6
    assert by_config["steps=6,sigma_shift=3"].timesteps is not None
    assert '"count":6' in by_config["steps=6,sigma_shift=3"].timesteps
    assert by_config["steps=6,sigma_shift=3"].sigmas is not None
    assert '"first":1.0' in by_config["steps=6,sigma_shift=3"].sigmas
    assert by_config["steps=6,sigma_shift=3"].deltas is not None
    assert by_config["steps=6,sigma_shift=3"].action_drift == pytest.approx(0.02)
    assert by_config["steps=6,sigma_shift=3"].action_drift_fields == "mean,min,max,max_abs"
    assert by_config["steps=6,sigma_shift=3"].job_id == "451001"
    assert by_config["steps=6,sigma_shift=3"].pareto is True
    assert by_config["steps=6,sigma_shift=3"].recommendation == "recommended_candidate"
    assert by_config["steps=4,sigma_shift=3"].pareto is True
    assert by_config["steps=4,sigma_shift=3"].recommendation == "not_recommended"
    assert (
        by_config["steps=4,sigma_shift=3"].recommendation_reason
        == "success_drop_exceeds_threshold"
    )
    assert by_config["steps=4,sigma_shift=3"].failure_count == 1
    assert by_config["steps=4,sigma_shift=3"].failure_cases is not None
    assert "episode_id=0" in by_config["steps=4,sigma_shift=3"].failure_cases
    assert "task_id=2" in by_config["steps=4,sigma_shift=3"].failure_cases
    assert by_config["steps=10,sigma_shift=null"].pareto is False

    outputs = write_report_outputs(runs, tmp_path / "report")

    assert set(outputs) == {"json", "csv", "config_json", "config_csv", "html", "markdown"}
    assert "FastWAM Scheduler Sweep Report" in (tmp_path / "report" / "scheduler_report.html").read_text(
        encoding="utf-8"
    )
    csv_text = (tmp_path / "report" / "scheduler_results.csv").read_text(encoding="utf-8")
    assert "speedup" in csv_text
    assert "solver" in csv_text
    assert "schedule_source" in csv_text
    assert "recommendation" in csv_text
    assert "failure_cases" in csv_text
    assert "timesteps" in csv_text
    assert "sigmas" in csv_text
    assert "episode_id=0" in csv_text
    assert "451001" in csv_text
    config_json = json.loads((tmp_path / "report" / "scheduler_config_summary.json").read_text(
        encoding="utf-8"
    ))
    assert any(row["config_key"] == "steps=6,sigma_shift=3" for row in config_json)
    assert any(row["aggregate_recommendation"] == "recommended_candidate" for row in config_json)
    config_csv_text = (tmp_path / "report" / "scheduler_config_summary.csv").read_text(
        encoding="utf-8"
    )
    assert "aggregate_recommendation" in config_csv_text
    assert "candidate_ids" in config_csv_text
    html_text = (tmp_path / "report" / "scheduler_report.html").read_text(encoding="utf-8")
    assert "Config-Level Decisions" in html_text
    assert "recommended_candidate" in html_text
    assert "wam eval fastwam-libero --opt scheduler" in html_text
    assert "num_inference_steps=6" in html_text
    assert "task_id=2" in html_text
    markdown_text = (tmp_path / "report" / "scheduler_final_report.md").read_text(
        encoding="utf-8"
    )
    assert "FastWAM Scheduler / Sampler Report" in markdown_text
    assert "solver" in markdown_text
    assert "schedule_source" in markdown_text
    assert "euler" in markdown_text
    assert "Per-Workload Conclusions" in markdown_text
    assert "Config-Level Decisions" in markdown_text
    assert "candidate_recommended_for_repeat" in markdown_text
    assert "needs_more_repeats" in markdown_text
    assert "Repeat Summary" in markdown_text
    assert "total_ms_std" in markdown_text
    assert "Recommended Candidates" in markdown_text
    assert "Not Recommended" in markdown_text
    assert "does not claim `parity_verified`" in markdown_text
    assert "total_ms" in markdown_text
    assert "denoise_wall_ms" in markdown_text
    assert "timestep_count" in markdown_text
    assert '"count":6' in markdown_text
    assert "command" in markdown_text
    assert "steps=6,sigma_shift=3" in markdown_text
    assert "wam eval fastwam-libero --opt scheduler" in markdown_text
    assert "num_inference_steps=6" in markdown_text
    assert "success_drop_exceeds_threshold" in markdown_text
    assert "fallback_reason" in markdown_text
    assert "failure_cases" in markdown_text
    assert "episode_id=0" in markdown_text


def test_scheduler_report_can_read_superpod_job_id_sidecar(tmp_path) -> None:
    baseline = _write_run(
        tmp_path,
        name="baseline",
        steps=10,
        total_ms=100.0,
        denoise_ms=80.0,
        success_rate=1.0,
        action_mean=0.10,
    )
    candidate = _write_run(
        tmp_path,
        name="candidate",
        steps=6,
        total_ms=60.0,
        denoise_ms=45.0,
        success_rate=1.0,
        action_mean=0.11,
    )
    job_map = tmp_path / "jobs.csv"
    job_map.write_text(
        "candidate_id,summary_path,job_id\n"
        f"fastwam-libero-steps6-shiftnull,{candidate},451777\n",
        encoding="utf-8",
    )

    runs = build_report([baseline, candidate], job_map_path=job_map)
    by_steps = {run.num_inference_steps: run for run in runs}

    assert by_steps[6].candidate_id == "fastwam-libero-steps6-shiftnull"
    assert by_steps[6].job_id == "451777"


def test_scheduler_report_prefers_manifest_command_for_reproducibility(tmp_path) -> None:
    candidate = _write_run(
        tmp_path,
        name="candidate",
        steps=6,
        total_ms=60.0,
        denoise_ms=45.0,
        success_rate=1.0,
        action_mean=0.11,
    )
    summary = json.loads(candidate.read_text(encoding="utf-8"))
    summary["command"] = {"display": "native LIBERO single-task eval fastwam-libero"}
    candidate.write_text(json.dumps(summary), encoding="utf-8")

    manifest_command = (
        "wam eval fastwam-libero --opt scheduler "
        "--set num_inference_steps=6 --set sigma_shift=null"
    )
    (tmp_path / "scheduler_sweep_candidates.csv").write_text(
        "candidate_id,summary_path,command\n"
        f"fastwam-libero-steps6-shiftnull,{candidate},{manifest_command}\n",
        encoding="utf-8",
    )
    job_map = tmp_path / "scheduler_sweep_job_map.csv"
    job_map.write_text(
        "candidate_id,summary_path,job_id\n"
        f"fastwam-libero-steps6-shiftnull,{candidate},451777\n",
        encoding="utf-8",
    )

    runs = build_report([candidate], job_map_path=job_map)

    assert runs[0].command == manifest_command
    assert runs[0].job_id == "451777"


def test_scheduler_report_keeps_requested_sigma_shift_for_command_validation(tmp_path) -> None:
    baseline = _write_run(
        tmp_path,
        name="baseline",
        steps=10,
        total_ms=100.0,
        denoise_ms=80.0,
        success_rate=1.0,
        action_mean=0.10,
        sigma_shift="null",
        actual_sigma_shift=5.0,
    )
    candidate = _write_run(
        tmp_path,
        name="candidate",
        steps=6,
        total_ms=60.0,
        denoise_ms=45.0,
        success_rate=1.0,
        action_mean=0.11,
        sigma_shift="null",
        actual_sigma_shift=5.0,
    )

    runs = build_report([baseline, candidate])
    outputs = write_report_outputs(runs, tmp_path / "report")
    rows = json.loads((tmp_path / "report" / "scheduler_results.json").read_text())

    assert rows[0]["sigma_shift"] == 5.0
    assert rows[0]["requested_sigma_shift"] == "null"
    validate_scheduler_report(
        outputs["json"],
        required_models=("fastwam-libero",),
        require_job_ids=False,
        require_action_drift=False,
        require_quality_reference=False,
    )


def test_scheduler_report_does_not_label_explicit_default_shift_as_baseline(tmp_path) -> None:
    baseline = _write_run(
        tmp_path,
        name="baseline",
        steps=10,
        total_ms=100.0,
        denoise_ms=80.0,
        success_rate=1.0,
        action_mean=0.10,
        sigma_shift="null",
        actual_sigma_shift=5.0,
    )
    explicit_default_shift = _write_run(
        tmp_path,
        name="explicit-default-shift",
        steps=10,
        total_ms=101.0,
        denoise_ms=81.0,
        success_rate=1.0,
        action_mean=0.10,
        sigma_shift=5.0,
        actual_sigma_shift=5.0,
    )
    faster = _write_run(
        tmp_path,
        name="faster",
        steps=6,
        total_ms=60.0,
        denoise_ms=48.0,
        success_rate=1.0,
        action_mean=0.11,
        sigma_shift="null",
        actual_sigma_shift=5.0,
    )
    quality_reference = _write_run(
        tmp_path,
        name="quality-reference",
        steps=12,
        total_ms=120.0,
        denoise_ms=95.0,
        success_rate=1.0,
        action_mean=0.10,
        sigma_shift="null",
        actual_sigma_shift=5.0,
    )

    runs = build_report([baseline, explicit_default_shift, faster, quality_reference])
    rows = {run.candidate_id: run for run in runs}

    baseline_row = rows["fastwam-libero-steps10-shiftnull-baseline"]
    assert baseline_row.is_baseline_reference is True
    assert baseline_row.recommendation == "baseline"

    explicit = rows["fastwam-libero-steps10-shift5.0-explicit-default-shift"]
    assert explicit.is_baseline_reference is False
    assert explicit.recommendation != "baseline"
    assert explicit.speedup == pytest.approx(100.0 / 101.0)

    outputs = write_report_outputs(runs, tmp_path / "report")
    validate_scheduler_report(
        outputs["json"],
        required_models=("fastwam-libero",),
        require_job_ids=False,
        require_action_drift=False,
        require_quality_reference=False,
    )


def test_scheduler_report_keys_distinguish_custom_schedule_values(tmp_path) -> None:
    baseline = _write_run(
        tmp_path,
        name="baseline",
        steps=10,
        total_ms=100.0,
        denoise_ms=80.0,
        success_rate=1.0,
        action_mean=0.10,
    )
    custom_a = _write_run(
        tmp_path,
        name="custom-a",
        steps=3,
        total_ms=70.0,
        denoise_ms=50.0,
        success_rate=1.0,
        action_mean=0.11,
        schedule_source="custom_sigmas",
        sigma_values=[1.0, 0.5, 0.125],
    )
    custom_b = _write_run(
        tmp_path,
        name="custom-b",
        steps=3,
        total_ms=72.0,
        denoise_ms=52.0,
        success_rate=1.0,
        action_mean=0.12,
        schedule_source="custom_sigmas",
        sigma_values=[1.0, 0.75, 0.25],
    )

    runs = build_report([baseline, custom_a, custom_b])
    custom_keys = [
        run.config_key
        for run in runs
        if run.schedule_source == "custom_sigmas"
    ]

    assert len(set(custom_keys)) == 2
    assert all("schedule_source=custom_sigmas" in key for key in custom_keys)
    assert all("schedule_hash=" in key for key in custom_keys)


def test_scheduler_report_outputs_pass_scheduler_acceptance(tmp_path) -> None:
    summaries = [
        _write_run(
            tmp_path,
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
            tmp_path,
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
            tmp_path,
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
            tmp_path,
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
            tmp_path,
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
            tmp_path,
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
    runs = build_report(summaries)
    outputs = write_report_outputs(runs, tmp_path / "report")

    report = validate_scheduler_report(outputs["json"])

    assert report.row_count == 6
    assert report.model_ids == ("fastwam-libero", "fastwam-robotwin")
    assert report.pareto_count >= 2


def test_scheduler_report_accepts_repeated_stage_c_baselines(tmp_path) -> None:
    summaries = []
    for index, total_ms in enumerate((100.0, 104.0, 96.0), start=1):
        summaries.append(
            _write_run(
                tmp_path,
                name=f"libero-baseline-{index}",
                model_id="fastwam-libero",
                workload="libero-single-task",
                steps=10,
                total_ms=total_ms,
                denoise_ms=total_ms * 0.8,
                success_rate=1.0,
                action_mean=0.10,
                job_id=f"45110{index}",
            )
        )
        summaries.append(
            _write_run(
                tmp_path,
                name=f"libero-fast-{index}",
                model_id="fastwam-libero",
                workload="libero-single-task",
                steps=6,
                sigma_shift=3.0,
                total_ms=60.0 + index,
                denoise_ms=48.0 + index,
                success_rate=1.0,
                action_mean=0.12,
                job_id=f"45111{index}",
            )
        )
        summaries.append(
            _write_run(
                tmp_path,
                name=f"libero-reference-{index}",
                model_id="fastwam-libero",
                workload="libero-single-task",
                steps=12,
                total_ms=120.0 + index,
                denoise_ms=96.0 + index,
                success_rate=1.0,
                action_mean=0.09,
                job_id=f"45112{index}",
            )
        )
        summaries.append(
            _write_run(
                tmp_path,
                name=f"robotwin-baseline-{index}",
                model_id="fastwam-robotwin",
                workload="robotwin-single-task",
                steps=10,
                total_ms=140.0 + index,
                denoise_ms=112.0 + index,
                success_rate=1.0,
                action_mean=0.20,
                job_id=f"45120{index}",
            )
        )
        summaries.append(
            _write_run(
                tmp_path,
                name=f"robotwin-fast-{index}",
                model_id="fastwam-robotwin",
                workload="robotwin-single-task",
                steps=8,
                sigma_shift=5.0,
                total_ms=100.0 + index,
                denoise_ms=80.0 + index,
                success_rate=1.0,
                action_mean=0.21,
                job_id=f"45121{index}",
            )
        )
        summaries.append(
            _write_run(
                tmp_path,
                name=f"robotwin-reference-{index}",
                model_id="fastwam-robotwin",
                workload="robotwin-single-task",
                steps=12,
                total_ms=160.0 + index,
                denoise_ms=128.0 + index,
                success_rate=1.0,
                action_mean=0.19,
                job_id=f"45122{index}",
            )
        )

    runs = build_report(summaries)
    summaries_by_config = {
        (summary.model_id, summary.config_key): summary
        for summary in build_config_summaries(runs)
    }
    outputs = write_report_outputs(runs, tmp_path / "report")

    report = validate_scheduler_report(outputs["json"], min_config_repeats=3)
    baseline_rows = [
        run
        for run in runs
        if run.num_inference_steps == 10 and run.sigma_shift is None
    ]
    markdown_text = (tmp_path / "report" / "scheduler_final_report.md").read_text(
        encoding="utf-8"
    )

    assert report.row_count == 18
    assert all(run.speedup == pytest.approx(1.0) for run in baseline_rows)
    assert all(run.success_delta == pytest.approx(0.0) for run in baseline_rows)
    assert summaries_by_config[
        ("fastwam-libero", "steps=6,sigma_shift=3")
    ].aggregate_recommendation == "recommended_candidate"
    assert summaries_by_config[
        ("fastwam-robotwin", "steps=8,sigma_shift=5")
    ].aggregate_pareto is True
    assert "## Repeat Summary" in markdown_text
    assert "## Config-Level Decisions" in markdown_text
    assert "steps=10,sigma_shift=null | 3 |" in markdown_text
    assert "aggregate_recommendation" in markdown_text
    assert "total_ms_std" in markdown_text
    assert "| no |" in markdown_text


def _write_run(
    root,
    *,
    name: str,
    model_id: str = "fastwam-libero",
    workload: str = "libero-single-task",
    steps: int,
    total_ms: float,
    denoise_ms: float,
    success_rate: float,
    action_mean: float,
    sigma_shift: object = None,
    actual_sigma_shift: float | None = None,
    schedule_source: str = "generated",
    sigma_values: list[float] | None = None,
    job_id: str | None = None,
):
    run_dir = root / name
    run_dir.mkdir()
    trace_path = run_dir / "trace.jsonl"
    events = [
        _event(
            name,
            "run_start",
            manifest_id=model_id,
            optimization_profiles=[{"name": "scheduler", "enabled": True, "params": {}}],
        ),
        _event(
            name,
            "native_eval_plan",
            runtime_options={
                "scheduler_name": "fastwam_flowmatch_euler",
                "solver": "euler",
                "schedule_type": "shifted_flowmatch",
                "schedule_source": schedule_source,
                "num_inference_steps": steps,
                "sigma_shift": sigma_shift,
            },
            job_id=job_id,
        ),
        _event(
            name,
            "inference_end",
            timing={"total_ms": total_ms},
            action_summary={
                "shape": [32, 7],
                "finite": True,
                "mean": action_mean,
                "min": action_mean - 0.1,
                "max": action_mean + 0.1,
                "max_abs": abs(action_mean) + 0.1,
            },
            backend_metadata={
                "scheduler_name": "fastwam_flowmatch_euler",
                "schedule_type": "shifted_flowmatch",
                "schedule_source": schedule_source,
                "num_inference_steps": steps,
                "sigma_shift": actual_sigma_shift if actual_sigma_shift is not None else sigma_shift,
                "timestep_count": steps,
                "timesteps": _summary(
                    [value * 1000.0 for value in sigma_values]
                    if sigma_values is not None
                    else None,
                    count=steps,
                    first=1000.0,
                    last=100.0,
                    min_value=100.0,
                    max_value=1000.0,
                ),
                "sigmas": _summary(
                    sigma_values,
                    count=steps,
                    first=1.0,
                    last=0.1,
                    min_value=0.1,
                    max_value=1.0,
                ),
                "deltas": _summary(
                    _sigma_deltas(sigma_values),
                    count=steps,
                    first=-0.1,
                    last=-0.1,
                    min_value=-0.1,
                    max_value=-0.1,
                ),
                "denoise_wall_ms": denoise_ms,
            },
        ),
        _event(
            name,
            "episode_end",
            success=success_rate == 1.0,
            episode_id=0,
            task_id=2,
            task_name="libero_goal_task",
            steps=700,
        ),
        _event(name, "run_end", status="ok"),
    ]
    trace_path.write_text(
        "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n",
        encoding="utf-8",
    )
    summary_path = run_dir / "summary.json"
    summary = {
        "run_id": name,
        "candidate_id": f"{model_id}-steps{steps}-shift{sigma_shift or 'null'}-{name}",
        "model_id": model_id,
        "workload": workload,
        "trace_path": str(trace_path),
        "status": "ok",
        "command": {
            "display": (
                f"wam eval {model_id} --opt scheduler "
                f"--set num_inference_steps={steps} --set sigma_shift={sigma_shift or 'null'}"
            )
        },
        "metrics": {"success_rate": success_rate},
    }
    if job_id is not None:
        summary["metrics"]["job_id"] = job_id
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path


def _summary(
    values,
    *,
    count: int,
    first: float,
    last: float,
    min_value: float,
    max_value: float,
) -> dict[str, object]:
    if values is None:
        return {
            "count": count,
            "first": first,
            "last": last,
            "min": min_value,
            "max": max_value,
        }
    return {
        "count": len(values),
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def _sigma_deltas(sigmas: list[float] | None) -> list[float] | None:
    if sigmas is None:
        return None
    path = [*sigmas, 0.0]
    return [path[index + 1] - path[index] for index in range(len(sigmas))]


def _event(run_id: str, event: str, **payload):
    manifest_id = payload.pop("manifest_id", "fastwam-libero")
    base = {
        "schema_version": 1,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "run_id": run_id,
        "event": event,
        "manifest_id": manifest_id,
        "backend": "fastwam",
        "processor": "fastwam_libero",
        "mode": "simulator_eval",
        "model_name": "FastWAM LIBERO",
        "source_repo": "yuantianyuan01/FastWAM",
    }
    base.update(payload)
    return base
