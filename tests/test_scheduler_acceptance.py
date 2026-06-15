from __future__ import annotations

import json

import pytest

from eazywam.evals.scheduler_acceptance import (
    SchedulerAcceptanceError,
    validate_scheduler_report,
)


def test_scheduler_acceptance_validates_completed_report(tmp_path) -> None:
    report_path = _write_report(
        tmp_path,
        rows=[
            _row(tmp_path, model_id="fastwam-libero", steps=10, sigma_shift=None, recommendation="baseline"),
            _row(
                tmp_path,
                model_id="fastwam-libero",
                steps=6,
                sigma_shift=3.0,
                recommendation="recommended_candidate",
                speedup=1.6,
                pareto=True,
            ),
            _row(tmp_path, model_id="fastwam-libero", steps=12, sigma_shift=None, speedup=0.9),
            _row(tmp_path, model_id="fastwam-robotwin", steps=10, sigma_shift=None, recommendation="baseline"),
            _row(
                tmp_path,
                model_id="fastwam-robotwin",
                steps=8,
                sigma_shift=5.0,
                recommendation="experimental",
                speedup=1.25,
                pareto=True,
            ),
            _row(tmp_path, model_id="fastwam-robotwin", steps=12, sigma_shift=None, speedup=0.92),
        ],
    )

    report = validate_scheduler_report(report_path, require_recommendation=True)

    assert report.row_count == 6
    assert report.model_ids == ("fastwam-libero", "fastwam-robotwin")
    assert report.baseline_steps == 10
    assert report.recommended_count == 1
    assert report.not_recommended_count == 2
    assert report.pareto_count == 2


def test_scheduler_acceptance_reports_missing_required_evidence(tmp_path) -> None:
    report_path = _write_report(
        tmp_path,
        rows=[
            _row(
                tmp_path,
                model_id="fastwam-libero",
                steps=10,
                sigma_shift=None,
                recommendation="baseline",
                candidate_id="",
                job_id="",
            ),
            _row(tmp_path, model_id="fastwam-libero", steps=6, sigma_shift=3.0, pareto=True),
        ],
    )

    with pytest.raises(SchedulerAcceptanceError) as exc_info:
        validate_scheduler_report(report_path)

    message = str(exc_info.value)
    assert "report is missing model_id=fastwam-robotwin" in message
    assert "fastwam-libero has no quality-reference candidate above baseline" in message
    assert "is missing candidate_id" in message
    assert "is missing job_id" in message


def test_scheduler_acceptance_requires_source_summary_paths_by_default(tmp_path) -> None:
    rows = [
        _row(tmp_path, model_id="fastwam-libero", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(tmp_path, model_id="fastwam-libero", steps=6, sigma_shift=3.0, pareto=True),
        _row(tmp_path, model_id="fastwam-libero", steps=12, sigma_shift=None),
        _row(tmp_path, model_id="fastwam-robotwin", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(tmp_path, model_id="fastwam-robotwin", steps=8, sigma_shift=5.0, pareto=True),
        _row(tmp_path, model_id="fastwam-robotwin", steps=12, sigma_shift=None),
    ]
    missing_summary = tmp_path / "missing-summary.json"
    rows[1]["summary_path"] = str(missing_summary)
    report_path = _write_report(tmp_path, rows=rows)

    with pytest.raises(SchedulerAcceptanceError) as exc_info:
        validate_scheduler_report(report_path)

    assert "summary_path does not exist" in str(exc_info.value)

    report = validate_scheduler_report(report_path, require_summary_paths=False)

    assert report.row_count == 6


def test_scheduler_acceptance_requires_scheduler_and_failure_fields(tmp_path) -> None:
    rows = [
        _row(tmp_path, model_id="fastwam-libero", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(tmp_path, model_id="fastwam-libero", steps=6, sigma_shift=3.0, pareto=True),
        _row(tmp_path, model_id="fastwam-libero", steps=12, sigma_shift=None),
        _row(tmp_path, model_id="fastwam-robotwin", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(tmp_path, model_id="fastwam-robotwin", steps=8, sigma_shift=5.0, pareto=True),
        _row(tmp_path, model_id="fastwam-robotwin", steps=12, sigma_shift=None),
    ]
    for key in (
        "scheduler_name",
        "solver",
        "schedule_type",
        "schedule_source",
        "timestep_count",
        "timesteps",
        "sigmas",
        "deltas",
        "sigma_shift",
        "failure_count",
        "failure_cases",
        "fallback_reason",
    ):
        rows[1].pop(key)
    report_path = _write_report(tmp_path, rows=rows)

    with pytest.raises(SchedulerAcceptanceError) as exc_info:
        validate_scheduler_report(report_path)

    message = str(exc_info.value)
    assert "is missing scheduler_name" in message
    assert "is missing solver" in message
    assert "is missing schedule_type" in message
    assert "is missing schedule_source" in message
    assert "is missing timestep_count" in message
    assert "is missing timesteps" in message
    assert "is missing sigmas" in message
    assert "is missing deltas" in message
    assert "is missing sigma_shift" in message
    assert "is missing failure_count" in message
    assert "is missing failure_cases" in message
    assert "is missing fallback_reason" in message


def test_scheduler_acceptance_validates_baseline_comparison_fields(tmp_path) -> None:
    rows = [
        _row(tmp_path, model_id="fastwam-libero", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(tmp_path, model_id="fastwam-libero", steps=6, sigma_shift=3.0, pareto=True),
        _row(tmp_path, model_id="fastwam-libero", steps=12, sigma_shift=None),
        _row(tmp_path, model_id="fastwam-robotwin", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(tmp_path, model_id="fastwam-robotwin", steps=8, sigma_shift=5.0, pareto=True),
        _row(tmp_path, model_id="fastwam-robotwin", steps=12, sigma_shift=None),
    ]
    rows[0]["speedup"] = 1.2
    rows[0]["success_delta"] = -0.1
    report_path = _write_report(tmp_path, rows=rows)

    with pytest.raises(SchedulerAcceptanceError) as exc_info:
        validate_scheduler_report(report_path)

    message = str(exc_info.value)
    assert "baseline speedup is 1.2, expected 1.0" in message
    assert "baseline success_delta is -0.1, expected 0.0" in message


def test_scheduler_acceptance_validates_recommendation_consistency(tmp_path) -> None:
    rows = [
        _row(tmp_path, model_id="fastwam-libero", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(
            tmp_path,
            model_id="fastwam-libero",
            steps=6,
            sigma_shift=3.0,
            recommendation="recommended_candidate",
            speedup=1.0,
            pareto=False,
        ),
        _row(
            tmp_path,
            model_id="fastwam-libero",
            steps=12,
            sigma_shift=None,
            recommendation="baseline",
        ),
        _row(tmp_path, model_id="fastwam-robotwin", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(
            tmp_path,
            model_id="fastwam-robotwin",
            steps=8,
            sigma_shift=5.0,
            recommendation="experimental",
            speedup=1.25,
            pareto=True,
            fallback_reason="scheduler_hook_unavailable",
        ),
        _row(tmp_path, model_id="fastwam-robotwin", steps=12, sigma_shift=None),
    ]
    report_path = _write_report(tmp_path, rows=rows)

    with pytest.raises(SchedulerAcceptanceError) as exc_info:
        validate_scheduler_report(report_path)

    message = str(exc_info.value)
    assert "is recommended_candidate but is not Pareto" in message
    assert "is recommended_candidate but speedup is 1.0" in message
    assert "is labeled baseline but is not exact 10-step baseline" in message
    assert "has fallback_reason but recommendation is 'experimental'" in message


def test_scheduler_acceptance_validates_metric_ranges(tmp_path) -> None:
    rows = [
        _row(tmp_path, model_id="fastwam-libero", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(tmp_path, model_id="fastwam-libero", steps=6, sigma_shift=3.0, pareto=True),
        _row(tmp_path, model_id="fastwam-libero", steps=12, sigma_shift=None),
        _row(tmp_path, model_id="fastwam-robotwin", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(tmp_path, model_id="fastwam-robotwin", steps=8, sigma_shift=5.0, pareto=True),
        _row(tmp_path, model_id="fastwam-robotwin", steps=12, sigma_shift=None),
    ]
    rows[1]["total_ms"] = -1.0
    rows[1]["denoise_wall_ms"] = 0.0
    rows[1]["timestep_count"] = 0
    rows[1]["speedup"] = 0.0
    rows[1]["success_rate"] = 1.2
    rows[1]["success_delta"] = -1.2
    rows[1]["failure_count"] = -1
    rows[1]["action_drift"] = -0.01
    report_path = _write_report(tmp_path, rows=rows)

    with pytest.raises(SchedulerAcceptanceError) as exc_info:
        validate_scheduler_report(report_path)

    message = str(exc_info.value)
    assert "total_ms is -1.0, expected > 0" in message
    assert "denoise_wall_ms is 0.0, expected > 0" in message
    assert "timestep_count is 0.0, expected > 0" in message
    assert "speedup is 0.0, expected > 0" in message
    assert "success_rate is 1.2, expected 0..1" in message
    assert "success_delta is -1.2, expected -1..1" in message
    assert "failure_count is -1.0, expected >= 0" in message
    assert "action_drift is -0.01, expected >= 0" in message


def test_scheduler_acceptance_validates_schedule_summary_consistency(tmp_path) -> None:
    rows = [
        _row(tmp_path, model_id="fastwam-libero", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(tmp_path, model_id="fastwam-libero", steps=6, sigma_shift=3.0, pareto=True),
        _row(tmp_path, model_id="fastwam-libero", steps=12, sigma_shift=None),
        _row(tmp_path, model_id="fastwam-robotwin", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(tmp_path, model_id="fastwam-robotwin", steps=8, sigma_shift=5.0, pareto=True),
        _row(tmp_path, model_id="fastwam-robotwin", steps=12, sigma_shift=None),
    ]
    rows[1]["timesteps"] = '{"count":5,"first":1000.0,"last":90.9,"min":90.9,"max":1000.0}'
    rows[1]["sigmas"] = '{"count":6,"first":1.0,"last":0.0909,"min":0.0909}'
    rows[1]["deltas"] = "not-json"
    rows[2]["timestep_count"] = 11
    report_path = _write_report(tmp_path, rows=rows)

    with pytest.raises(SchedulerAcceptanceError) as exc_info:
        validate_scheduler_report(report_path)

    message = str(exc_info.value)
    assert "timesteps.count is 5, expected timestep_count 6" in message
    assert "sigmas.max is missing" in message
    assert "deltas is not valid JSON summary" in message
    assert "timestep_count is 11, expected num_inference_steps 12" in message


def test_scheduler_acceptance_validates_scheduler_identity(tmp_path) -> None:
    rows = [
        _row(tmp_path, model_id="fastwam-libero", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(tmp_path, model_id="fastwam-libero", steps=6, sigma_shift=3.0, pareto=True),
        _row(tmp_path, model_id="fastwam-libero", steps=12, sigma_shift=None),
        _row(tmp_path, model_id="fastwam-robotwin", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(tmp_path, model_id="fastwam-robotwin", steps=8, sigma_shift=5.0, pareto=True),
        _row(tmp_path, model_id="fastwam-robotwin", steps=12, sigma_shift=None),
    ]
    for row in rows:
        row["scheduler_name"] = "unipc"
        row["solver"] = "multistep"
        row["schedule_type"] = "custom"
    report_path = _write_report(tmp_path, rows=rows)

    with pytest.raises(SchedulerAcceptanceError) as exc_info:
        validate_scheduler_report(report_path)

    message = str(exc_info.value)
    assert "scheduler_name is 'unipc', expected 'fastwam_flowmatch_euler'" in message
    assert "solver is 'multistep', expected 'euler'" in message
    assert "schedule_type is 'custom', expected 'shifted_flowmatch'" in message

    report = validate_scheduler_report(
        report_path,
        expected_scheduler_name="unipc",
        expected_solver="multistep",
        expected_schedule_type="custom",
    )

    assert report.row_count == 6


def test_scheduler_acceptance_validates_command_reproducibility(tmp_path) -> None:
    rows = [
        _row(tmp_path, model_id="fastwam-libero", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(tmp_path, model_id="fastwam-libero", steps=6, sigma_shift=3.0, pareto=True),
        _row(tmp_path, model_id="fastwam-libero", steps=12, sigma_shift=None),
        _row(tmp_path, model_id="fastwam-robotwin", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(tmp_path, model_id="fastwam-robotwin", steps=8, sigma_shift=5.0, pareto=True),
        _row(tmp_path, model_id="fastwam-robotwin", steps=12, sigma_shift=None),
    ]
    rows[1]["command"] = "wam eval other-model --set num_inference_steps=5"
    report_path = _write_report(tmp_path, rows=rows)

    with pytest.raises(SchedulerAcceptanceError) as exc_info:
        validate_scheduler_report(report_path)

    message = str(exc_info.value)
    assert "command does not include 'wam eval fastwam-libero'" in message
    assert "command does not include '--opt scheduler'" in message
    assert "command does not include 'num_inference_steps=6'" in message
    assert "command does not include 'sigma_shift=3'" in message


def test_scheduler_acceptance_requires_not_recommended_for_quality_references(tmp_path) -> None:
    rows = [
        _row(tmp_path, model_id="fastwam-libero", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(
            tmp_path,
            model_id="fastwam-libero",
            steps=6,
            sigma_shift=3.0,
            recommendation="recommended_candidate",
            speedup=1.6,
            pareto=True,
        ),
        _row(tmp_path, model_id="fastwam-libero", steps=12, sigma_shift=None, recommendation="experimental"),
        _row(tmp_path, model_id="fastwam-robotwin", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(
            tmp_path,
            model_id="fastwam-robotwin",
            steps=8,
            sigma_shift=5.0,
            recommendation="experimental",
            speedup=1.25,
            pareto=True,
        ),
        _row(tmp_path, model_id="fastwam-robotwin", steps=12, sigma_shift=None, recommendation="experimental"),
    ]
    report_path = _write_report(tmp_path, rows=rows)

    with pytest.raises(SchedulerAcceptanceError) as exc_info:
        validate_scheduler_report(report_path)

    assert "quality-reference rows but no not_recommended rows" in str(exc_info.value)


def test_scheduler_acceptance_allows_stage_c_without_not_recommended_rows(tmp_path) -> None:
    rows = [
        _row(tmp_path, model_id="fastwam-libero", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(
            tmp_path,
            model_id="fastwam-libero",
            steps=6,
            sigma_shift=3.0,
            recommendation="recommended_candidate",
            speedup=1.6,
            pareto=True,
        ),
        _row(tmp_path, model_id="fastwam-robotwin", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(
            tmp_path,
            model_id="fastwam-robotwin",
            steps=8,
            sigma_shift=5.0,
            recommendation="experimental",
            speedup=1.25,
            pareto=True,
        ),
    ]
    report_path = _write_report(tmp_path, rows=rows)

    report = validate_scheduler_report(report_path, require_quality_reference=False)

    assert report.row_count == 4
    assert report.not_recommended_count == 0


def test_scheduler_acceptance_can_explicitly_require_not_recommended_rows(tmp_path) -> None:
    rows = [
        _row(tmp_path, model_id="fastwam-libero", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(
            tmp_path,
            model_id="fastwam-libero",
            steps=6,
            sigma_shift=3.0,
            recommendation="recommended_candidate",
            speedup=1.6,
            pareto=True,
        ),
        _row(tmp_path, model_id="fastwam-robotwin", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(
            tmp_path,
            model_id="fastwam-robotwin",
            steps=8,
            sigma_shift=5.0,
            recommendation="experimental",
            speedup=1.25,
            pareto=True,
        ),
    ]
    report_path = _write_report(tmp_path, rows=rows)

    with pytest.raises(SchedulerAcceptanceError) as exc_info:
        validate_scheduler_report(
            report_path,
            require_not_recommended=True,
            require_quality_reference=False,
        )

    assert "report has no not_recommended rows" in str(exc_info.value)


def test_scheduler_acceptance_requires_action_drift_by_default(tmp_path) -> None:
    rows = [
        _row(tmp_path, model_id="fastwam-libero", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(
            tmp_path,
            model_id="fastwam-libero",
            steps=6,
            sigma_shift=3.0,
            recommendation="recommended_candidate",
            speedup=1.6,
            pareto=True,
        ),
        _row(tmp_path, model_id="fastwam-libero", steps=12, sigma_shift=None, speedup=0.9),
        _row(tmp_path, model_id="fastwam-robotwin", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(tmp_path, model_id="fastwam-robotwin", steps=8, sigma_shift=5.0, speedup=1.25, pareto=True),
        _row(tmp_path, model_id="fastwam-robotwin", steps=12, sigma_shift=None, speedup=0.92),
    ]
    rows[1]["action_drift"] = None
    rows[1]["action_drift_fields"] = None
    report_path = _write_report(tmp_path, rows=rows)

    with pytest.raises(SchedulerAcceptanceError) as exc_info:
        validate_scheduler_report(report_path)

    message = str(exc_info.value)
    assert "is missing action_drift" in message
    assert "is missing action_drift_fields" in message

    report = validate_scheduler_report(report_path, require_action_drift=False)

    assert report.row_count == 6


def test_scheduler_acceptance_requires_report_artifacts_by_default(tmp_path) -> None:
    report_path = _write_report(
        tmp_path,
        rows=[
            _row(tmp_path, model_id="fastwam-libero", steps=10, sigma_shift=None, recommendation="baseline"),
            _row(tmp_path, model_id="fastwam-libero", steps=6, sigma_shift=3.0, pareto=True),
            _row(tmp_path, model_id="fastwam-libero", steps=12, sigma_shift=None),
            _row(tmp_path, model_id="fastwam-robotwin", steps=10, sigma_shift=None, recommendation="baseline"),
            _row(tmp_path, model_id="fastwam-robotwin", steps=8, sigma_shift=5.0, pareto=True),
            _row(tmp_path, model_id="fastwam-robotwin", steps=12, sigma_shift=None),
        ],
    )
    (tmp_path / "scheduler_report.html").unlink()

    with pytest.raises(SchedulerAcceptanceError) as exc_info:
        validate_scheduler_report(report_path)

    assert "report artifact is missing" in str(exc_info.value)

    report = validate_scheduler_report(report_path, require_report_artifacts=False)

    assert report.row_count == 6


def test_scheduler_acceptance_validates_config_summary_consistency(tmp_path) -> None:
    report_path = _write_report(
        tmp_path,
        rows=[
            _row(tmp_path, model_id="fastwam-libero", steps=10, sigma_shift=None, recommendation="baseline"),
            _row(tmp_path, model_id="fastwam-libero", steps=6, sigma_shift=3.0, pareto=True),
            _row(tmp_path, model_id="fastwam-libero", steps=12, sigma_shift=None),
            _row(tmp_path, model_id="fastwam-robotwin", steps=10, sigma_shift=None, recommendation="baseline"),
            _row(tmp_path, model_id="fastwam-robotwin", steps=8, sigma_shift=5.0, pareto=True),
            _row(tmp_path, model_id="fastwam-robotwin", steps=12, sigma_shift=None),
        ],
    )
    config_summary_path = tmp_path / "scheduler_config_summary.json"
    summaries = json.loads(config_summary_path.read_text(encoding="utf-8"))
    summaries[1]["speedup_mean"] = 99.0
    summaries[1]["candidate_ids"] = ""
    config_summary_path.write_text(json.dumps(summaries), encoding="utf-8")

    with pytest.raises(SchedulerAcceptanceError) as exc_info:
        validate_scheduler_report(report_path)

    message = str(exc_info.value)
    assert "speedup_mean is 99.0" in message
    assert "candidate_ids is missing" in message


def test_scheduler_acceptance_requires_report_chart_markers(tmp_path) -> None:
    report_path = _write_report(
        tmp_path,
        rows=[
            _row(tmp_path, model_id="fastwam-libero", steps=10, sigma_shift=None, recommendation="baseline"),
            _row(tmp_path, model_id="fastwam-libero", steps=6, sigma_shift=3.0, pareto=True),
            _row(tmp_path, model_id="fastwam-libero", steps=12, sigma_shift=None),
            _row(tmp_path, model_id="fastwam-robotwin", steps=10, sigma_shift=None, recommendation="baseline"),
            _row(tmp_path, model_id="fastwam-robotwin", steps=8, sigma_shift=5.0, pareto=True),
            _row(tmp_path, model_id="fastwam-robotwin", steps=12, sigma_shift=None),
        ],
    )
    (tmp_path / "scheduler_report.html").write_text(
        "<h1>FastWAM Scheduler Sweep Report</h1><svg></svg>Total latency ms",
        encoding="utf-8",
    )

    with pytest.raises(SchedulerAcceptanceError) as exc_info:
        validate_scheduler_report(report_path)

    message = str(exc_info.value)
    assert "missing marker 'Speedup vs success rate'" in message
    assert "missing marker 'Drift vs speedup'" in message


def test_scheduler_acceptance_requires_final_report_sections(tmp_path) -> None:
    report_path = _write_report(
        tmp_path,
        rows=[
            _row(tmp_path, model_id="fastwam-libero", steps=10, sigma_shift=None, recommendation="baseline"),
            _row(tmp_path, model_id="fastwam-libero", steps=6, sigma_shift=3.0, pareto=True),
            _row(tmp_path, model_id="fastwam-libero", steps=12, sigma_shift=None),
            _row(tmp_path, model_id="fastwam-robotwin", steps=10, sigma_shift=None, recommendation="baseline"),
            _row(tmp_path, model_id="fastwam-robotwin", steps=8, sigma_shift=5.0, pareto=True),
            _row(tmp_path, model_id="fastwam-robotwin", steps=12, sigma_shift=None),
        ],
    )
    (tmp_path / "scheduler_final_report.md").write_text(
        "# FastWAM Scheduler / Sampler Report\n",
        encoding="utf-8",
    )

    with pytest.raises(SchedulerAcceptanceError) as exc_info:
        validate_scheduler_report(report_path)

    message = str(exc_info.value)
    assert "missing marker 'Per-Workload Conclusions'" in message
    assert "missing marker 'Repeat Summary'" in message
    assert "missing marker 'does not claim `parity_verified`'" in message


def test_scheduler_acceptance_accepts_cwd_relative_trace_paths(tmp_path, monkeypatch) -> None:
    report_dir = tmp_path / "runs" / "report"
    report_dir.mkdir(parents=True)
    trace_dir = tmp_path / "runs" / "traces"
    trace_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    rows = [
        _row(tmp_path, model_id="fastwam-libero", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(tmp_path, model_id="fastwam-libero", steps=6, sigma_shift=3.0, pareto=True),
        _row(tmp_path, model_id="fastwam-libero", steps=12, sigma_shift=None),
        _row(tmp_path, model_id="fastwam-robotwin", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(tmp_path, model_id="fastwam-robotwin", steps=8, sigma_shift=5.0, pareto=True),
        _row(tmp_path, model_id="fastwam-robotwin", steps=12, sigma_shift=None),
    ]
    for index, row in enumerate(rows):
        trace_path = trace_dir / f"trace-{index}.jsonl"
        trace_path.write_text('{"event": "run_end", "status": "ok"}\n', encoding="utf-8")
        row["trace_path"] = str(trace_path.relative_to(tmp_path))
    report_path = _write_report(report_dir, rows=rows)

    report = validate_scheduler_report(report_path)

    assert report.row_count == 6


def test_scheduler_acceptance_validates_min_config_repeats(tmp_path) -> None:
    rows = [
        _row(tmp_path, model_id="fastwam-libero", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(
            tmp_path,
            model_id="fastwam-libero",
            steps=6,
            sigma_shift=3.0,
            recommendation="recommended_candidate",
            speedup=1.6,
            pareto=True,
        ),
        _row(tmp_path, model_id="fastwam-libero", steps=12, sigma_shift=None, speedup=0.9),
        _row(tmp_path, model_id="fastwam-robotwin", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(tmp_path, model_id="fastwam-robotwin", steps=8, sigma_shift=5.0, speedup=1.25, pareto=True),
        _row(tmp_path, model_id="fastwam-robotwin", steps=12, sigma_shift=None, speedup=0.92),
    ]
    report_path = _write_report(tmp_path, rows=rows + rows)

    report = validate_scheduler_report(report_path, min_config_repeats=2)

    assert report.row_count == 12
    assert report.min_config_repeats == 2


def test_scheduler_acceptance_reports_missing_config_repeats(tmp_path) -> None:
    rows = [
        _row(tmp_path, model_id="fastwam-libero", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(
            tmp_path,
            model_id="fastwam-libero",
            steps=6,
            sigma_shift=3.0,
            recommendation="recommended_candidate",
            speedup=1.6,
            pareto=True,
        ),
        _row(tmp_path, model_id="fastwam-libero", steps=12, sigma_shift=None, speedup=0.9),
        _row(tmp_path, model_id="fastwam-robotwin", steps=10, sigma_shift=None, recommendation="baseline"),
        _row(tmp_path, model_id="fastwam-robotwin", steps=8, sigma_shift=5.0, speedup=1.25, pareto=True),
        _row(tmp_path, model_id="fastwam-robotwin", steps=12, sigma_shift=None, speedup=0.92),
    ]
    report_path = _write_report(tmp_path, rows=rows + rows[:-1])

    with pytest.raises(SchedulerAcceptanceError) as exc_info:
        validate_scheduler_report(report_path, min_config_repeats=2)

    message = str(exc_info.value)
    assert (
        "fastwam-robotwin:robotwin-single-task:steps=12:sigma_shift=None "
        "has 1 repeat(s), expected at least 2"
    ) in message


def _write_report(tmp_path, *, rows: list[dict[str, object]]):
    report_path = tmp_path / "scheduler_results.json"
    report_path.write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / "scheduler_results.csv").write_text("model_id,speedup\nfastwam-libero,1.0\n", encoding="utf-8")
    (tmp_path / "scheduler_config_summary.json").write_text(
        json.dumps(_config_summary_rows(rows)),
        encoding="utf-8",
    )
    (tmp_path / "scheduler_config_summary.csv").write_text(
        "aggregate_recommendation,candidate_ids,job_ids\nbaseline,candidate,451000\n",
        encoding="utf-8",
    )
    (tmp_path / "scheduler_report.html").write_text(
        "<h1>FastWAM Scheduler Sweep Report</h1>"
        "<svg></svg>"
        "Config-Level Decisions"
        "Total latency ms"
        "Success rate"
        "Speedup vs success rate"
        "Drift vs speedup"
        "Results",
        encoding="utf-8",
    )
    (tmp_path / "scheduler_final_report.md").write_text(
        "# FastWAM Scheduler / Sampler Report\n"
        "This report does not claim `parity_verified`.\n"
        "## Per-Workload Conclusions\n"
        "## Config-Level Decisions\n"
        "## Repeat Summary\n"
        "## Recommended Candidates\n"
        "## Not Recommended\n"
        "## Full Table\n",
        encoding="utf-8",
    )
    return report_path


def _config_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        sigma = "null" if row.get("sigma_shift") is None else f"{float(row['sigma_shift']):.4g}"
        key = (
            str(row.get("model_id") or ""),
            str(row.get("workload") or ""),
            str(row.get("solver") or ""),
            f"steps={row.get('num_inference_steps')},sigma_shift={sigma}",
        )
        groups.setdefault(key, []).append(row)

    summaries = []
    for (model_id, workload, solver, config_key), group in sorted(groups.items()):
        first = group[0]
        summaries.append(
            {
                "model_id": model_id,
                "workload": workload,
                "solver": solver,
                "config_key": config_key,
                "repeats": len(group),
                "num_inference_steps": first.get("num_inference_steps"),
                "sigma_shift": first.get("sigma_shift"),
                "total_ms_mean": _mean(group, "total_ms"),
                "denoise_wall_ms_mean": _mean(group, "denoise_wall_ms"),
                "success_rate_mean": _mean(group, "success_rate"),
                "success_delta_mean": _mean(group, "success_delta"),
                "speedup_mean": _mean(group, "speedup"),
                "action_drift_mean": _mean(group, "action_drift"),
                "failure_count": sum(int(row.get("failure_count") or 0) for row in group),
                "candidate_ids": ",".join(str(row.get("candidate_id") or "") for row in group),
                "job_ids": ",".join(str(row.get("job_id") or "") for row in group),
                "trace_paths": "<br>".join(str(row.get("trace_path") or "") for row in group),
                "aggregate_pareto": any(row.get("pareto") is True for row in group),
                "aggregate_recommendation": (
                    "baseline"
                    if any(row.get("recommendation") == "baseline" for row in group)
                    else "not_recommended"
                ),
                "aggregate_reason": "test",
            }
        )
    return summaries


def _mean(rows: list[dict[str, object]], field: str) -> float | None:
    values = [
        float(row[field])
        for row in rows
        if row.get(field) not in {None, "", "null", "None"}
    ]
    return sum(values) / len(values) if values else None


def _row(
    tmp_path,
    *,
    model_id: str,
    steps: int,
    sigma_shift: float | None,
    recommendation: str = "not_recommended",
    speedup: float = 1.0,
    pareto: bool = False,
    candidate_id: str | None = None,
    job_id: str = "451000",
    fallback_reason: str | None = None,
) -> dict[str, object]:
    trace_path = tmp_path / f"{model_id}-steps{steps}-shift{sigma_shift or 'null'}.jsonl"
    trace_path.write_text('{"event": "run_end", "status": "ok"}\n', encoding="utf-8")
    summary_path = tmp_path / f"{model_id}-steps{steps}-summary.json"
    summary_path.write_text('{"status": "ok"}\n', encoding="utf-8")
    sigma = "null" if sigma_shift is None else f"{sigma_shift:.4g}"
    config_key = f"steps={steps},sigma_shift={sigma}"
    return {
        "summary_path": str(summary_path),
        "trace_path": str(trace_path),
        "candidate_id": (
            f"{model_id}-steps{steps}-shift{sigma}" if candidate_id is None else candidate_id
        ),
        "run_id": f"{model_id}-steps{steps}",
        "model_id": model_id,
        "workload": "libero-single-task" if model_id == "fastwam-libero" else "robotwin-single-task",
        "config_key": config_key,
        "status": "ok",
        "command": (
            f"wam eval {model_id} --opt scheduler "
            f"--set num_inference_steps={steps} --set sigma_shift={sigma}"
        ),
        "job_id": job_id,
        "scheduler_name": "fastwam_flowmatch_euler",
        "solver": "euler",
        "schedule_type": "shifted_flowmatch",
        "schedule_source": "generated",
        "num_inference_steps": steps,
        "sigma_shift": sigma_shift,
        "timestep_count": steps,
        "timesteps": _schedule_summary(steps, first=1000.0, last=90.9, min_value=90.9, max_value=1000.0),
        "sigmas": _schedule_summary(steps, first=1.0, last=0.0909, min_value=0.0909, max_value=1.0),
        "deltas": _schedule_summary(
            steps,
            first=-0.1818,
            last=-0.0909,
            min_value=-0.1818,
            max_value=-0.0909,
        ),
        "total_ms": 100.0 / speedup,
        "denoise_wall_ms": 80.0 / speedup,
        "speedup": speedup,
        "success_rate": 1.0,
        "success_delta": 0.0,
        "action_drift": 0.01,
        "action_drift_fields": "mean,min,max,max_abs",
        "failure_count": 0,
        "failure_cases": None,
        "baseline_key": "steps=10,sigma_shift=null",
        "pareto": pareto,
        "fallback_reason": fallback_reason,
        "recommendation": recommendation,
        "recommendation_reason": "test",
    }


def _schedule_summary(
    count: int,
    *,
    first: float,
    last: float,
    min_value: float,
    max_value: float,
) -> str:
    return json.dumps(
        {
            "count": count,
            "first": first,
            "last": last,
            "min": min_value,
            "max": max_value,
        },
        sort_keys=True,
    )
