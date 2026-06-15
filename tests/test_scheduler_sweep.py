import json

import pytest

from eazywam.evals.scheduler_sweep import (
    build_confirm_sweep,
    build_coarse_sweep,
    build_refine_sweep,
    main,
    write_sweep_manifest,
)


def test_scheduler_sweep_manifest_includes_baseline_faster_and_quality_reference(tmp_path) -> None:
    candidates = build_coarse_sweep(trace_dir="/runs/scheduler", baseline_steps=10)

    steps = {candidate.num_inference_steps for candidate in candidates}
    phases = {candidate.phase for candidate in candidates}
    model_ids = {candidate.model_id for candidate in candidates}

    assert 10 in steps
    assert min(steps) < 10
    assert max(steps) > 10
    assert {"baseline", "faster_candidate", "quality_reference"} <= phases
    assert model_ids == {"fastwam-libero", "fastwam-robotwin"}
    assert any("--opt scheduler" in candidate.command for candidate in candidates)
    assert any("--set num_inference_steps=10" in candidate.command for candidate in candidates)
    assert any("--set sigma_shift=null" in candidate.command for candidate in candidates)

    outputs = write_sweep_manifest(candidates, tmp_path)

    assert set(outputs) == {"json", "csv", "shell", "job_map", "report_shell"}
    assert "fastwam-libero" in (tmp_path / "scheduler_sweep_candidates.csv").read_text(
        encoding="utf-8"
    )
    command_text = (tmp_path / "scheduler_sweep_commands.sh").read_text(encoding="utf-8")
    assert "mkdir -p /runs/scheduler/fastwam-libero" in command_text
    assert "wam eval fastwam-robotwin" in command_text
    job_map_text = (tmp_path / "scheduler_sweep_job_map.csv").read_text(encoding="utf-8")
    assert "candidate_id,summary_path,job_id" in job_map_text
    assert "/runs/scheduler/summaries/fastwam-libero-steps10-shiftnull-summary.json" in job_map_text
    report_text = (tmp_path / "scheduler_report_command.sh").read_text(encoding="utf-8")
    assert "python -m eazywam.evals.scheduler_audit" in report_text
    assert "python -m eazywam.evals.scheduler_report" in report_text
    assert "--job-map" in report_text
    assert "python -m eazywam.evals.scheduler_acceptance" in report_text
    assert "python -m eazywam.evals.scheduler_bundle" in report_text
    assert "--manifest-dir" in report_text
    assert "/runs/scheduler/report/scheduler_results.json" in report_text


def test_scheduler_sweep_accepts_custom_coarse_space(tmp_path) -> None:
    candidates = build_coarse_sweep(
        trace_dir="/runs/scheduler",
        baseline_steps=10,
        step_counts=[5, 8, 12],
        sigma_shifts=["null", "2.5"],
    )

    steps = {candidate.num_inference_steps for candidate in candidates}
    shifts = {candidate.sigma_shift for candidate in candidates}

    assert steps == {5, 8, 10, 12}
    assert shifts == {"null", "2.5"}
    assert any(candidate.phase == "baseline" for candidate in candidates)
    assert any(candidate.phase == "faster_candidate" for candidate in candidates)
    assert any(candidate.phase == "quality_reference" for candidate in candidates)
    assert any("--set sigma_shift=2.5" in candidate.command for candidate in candidates)

    exit_code = main(
        [
            "--trace-dir",
            "/runs/scheduler-custom",
            "--output-dir",
            str(tmp_path),
            "--step-counts",
            "5,8,12",
            "--sigma-shifts",
            "null,2.5",
        ]
    )

    assert exit_code == 0
    manifest = json.loads((tmp_path / "scheduler_sweep_candidates.json").read_text())
    assert {row["num_inference_steps"] for row in manifest} == {5, 8, 10, 12}
    assert {row["sigma_shift"] for row in manifest} == {"null", "2.5"}


def test_scheduler_sweep_can_filter_model_ids(tmp_path) -> None:
    candidates = build_coarse_sweep(
        trace_dir="/runs/scheduler",
        baseline_steps=10,
        model_ids=["fastwam-libero"],
        step_counts=[6, 12],
        sigma_shifts=["null"],
    )

    assert {candidate.model_id for candidate in candidates} == {"fastwam-libero"}
    assert all("wam eval fastwam-libero" in candidate.command for candidate in candidates)

    exit_code = main(
        [
            "--trace-dir",
            "/runs/scheduler-libero",
            "--output-dir",
            str(tmp_path),
            "--model-ids",
            "fastwam-libero",
            "--step-counts",
            "6,12",
            "--sigma-shifts",
            "null",
        ]
    )

    assert exit_code == 0
    manifest = json.loads((tmp_path / "scheduler_sweep_candidates.json").read_text())
    assert {row["model_id"] for row in manifest} == {"fastwam-libero"}
    command_text = (tmp_path / "scheduler_sweep_commands.sh").read_text(encoding="utf-8")
    assert "wam eval fastwam-libero" in command_text
    assert "wam eval fastwam-robotwin" not in command_text


def test_scheduler_sweep_commands_can_include_eval_runtime_context() -> None:
    candidates = build_coarse_sweep(
        trace_dir="/runs/scheduler",
        baseline_steps=10,
        step_counts=[6, 12],
        sigma_shifts=["null"],
        cache_dir="/scratch/cache",
        upstream_dir="/scratch/FastWAM",
        eval_sets=["seed=42"],
        libero_sets=["mujoco_gl=egl"],
        robotwin_sets=["robotwin_root=/scratch/RoboTwin", "gpu_id=0"],
    )
    by_model = {candidate.model_id: candidate.command for candidate in candidates}

    assert "--cache-dir /scratch/cache" in by_model["fastwam-libero"]
    assert "--upstream-dir /scratch/FastWAM" in by_model["fastwam-libero"]
    assert "--set seed=42" in by_model["fastwam-libero"]
    assert "--set mujoco_gl=egl" in by_model["fastwam-libero"]
    assert "robotwin_root" not in by_model["fastwam-libero"]

    assert "--cache-dir /scratch/cache" in by_model["fastwam-robotwin"]
    assert "--upstream-dir /scratch/FastWAM" in by_model["fastwam-robotwin"]
    assert "--set robotwin_root=/scratch/RoboTwin" in by_model["fastwam-robotwin"]
    assert "--set gpu_id=0" in by_model["fastwam-robotwin"]
    assert "mujoco_gl" not in by_model["fastwam-robotwin"]


def test_scheduler_sweep_rejects_custom_coarse_space_without_quality_reference() -> None:
    with pytest.raises(ValueError, match="quality-reference"):
        build_coarse_sweep(
            trace_dir="/runs/scheduler",
            baseline_steps=10,
            step_counts=[5, 8, 10],
        )


def test_scheduler_refine_sweep_expands_around_reported_pareto_candidates(tmp_path) -> None:
    report = tmp_path / "scheduler_results.json"
    report.write_text(
        json.dumps(
            [
                {
                    "model_id": "fastwam-libero",
                    "num_inference_steps": 10,
                    "sigma_shift": None,
                    "pareto": False,
                    "recommendation": "baseline",
                },
                {
                    "model_id": "fastwam-libero",
                    "num_inference_steps": 6,
                    "sigma_shift": 3.0,
                    "pareto": True,
                    "recommendation": "recommended_candidate",
                },
                {
                    "model_id": "fastwam-libero",
                    "num_inference_steps": 2,
                    "sigma_shift": 9.0,
                    "pareto": False,
                    "recommendation": "experimental",
                },
                {
                    "model_id": "fastwam-robotwin",
                    "num_inference_steps": 8,
                    "sigma_shift": 5.0,
                    "pareto": True,
                    "recommendation": "experimental",
                },
            ]
        ),
        encoding="utf-8",
    )

    candidates = build_refine_sweep(
        report_json=report,
        trace_dir="/runs/scheduler-stage-b",
        baseline_steps=10,
    )
    by_model = {
        model_id: [candidate for candidate in candidates if candidate.model_id == model_id]
        for model_id in {candidate.model_id for candidate in candidates}
    }

    assert any(
        candidate.num_inference_steps == 10 and candidate.sigma_shift == "null"
        for candidate in by_model["fastwam-libero"]
    )
    assert {5, 6, 7} <= {
        candidate.num_inference_steps for candidate in by_model["fastwam-libero"]
    }
    assert {"2", "3", "4"} <= {
        candidate.sigma_shift for candidate in by_model["fastwam-libero"]
    }
    assert not any(
        candidate.model_id == "fastwam-libero" and candidate.num_inference_steps == 2
        for candidate in candidates
    )
    assert {7, 8, 9} <= {
        candidate.num_inference_steps for candidate in by_model["fastwam-robotwin"]
    }
    assert any(candidate.phase == "refine_faster_candidate" for candidate in candidates)
    assert all("--opt scheduler" in candidate.command for candidate in candidates)

    write_sweep_manifest(candidates, tmp_path / "manifest")
    report_text = (tmp_path / "manifest" / "scheduler_report_command.sh").read_text(
        encoding="utf-8"
    )
    assert "--allow-missing-quality-reference" in report_text
    assert "--min-config-repeats" not in report_text


def test_scheduler_confirm_sweep_repeats_baseline_and_pareto_candidates(tmp_path) -> None:
    report = tmp_path / "scheduler_results.json"
    report.write_text(
        json.dumps(
            [
                {
                    "model_id": "fastwam-libero",
                    "num_inference_steps": 10,
                    "sigma_shift": None,
                    "pareto": False,
                    "recommendation": "baseline",
                },
                {
                    "model_id": "fastwam-libero",
                    "num_inference_steps": 6,
                    "sigma_shift": 3.0,
                    "pareto": True,
                    "recommendation": "recommended_candidate",
                },
                {
                    "model_id": "fastwam-libero",
                    "num_inference_steps": 4,
                    "sigma_shift": 7.0,
                    "pareto": False,
                    "recommendation": "not_recommended",
                },
                {
                    "model_id": "fastwam-robotwin",
                    "num_inference_steps": 10,
                    "sigma_shift": None,
                    "pareto": False,
                    "recommendation": "baseline",
                },
                {
                    "model_id": "fastwam-robotwin",
                    "num_inference_steps": 8,
                    "sigma_shift": 5.0,
                    "pareto": True,
                    "recommendation": "experimental",
                },
            ]
        ),
        encoding="utf-8",
    )

    candidates = build_confirm_sweep(
        report_json=report,
        trace_dir="/runs/scheduler-stage-c",
        baseline_steps=10,
        repeats=2,
    )

    assert len(candidates) == 8
    assert {candidate.phase for candidate in candidates} == {
        "confirm_baseline",
        "confirm_pareto",
    }
    assert any(candidate.candidate_id == "fastwam-libero-steps10-shiftnull-confirm1" for candidate in candidates)
    assert any(candidate.candidate_id == "fastwam-libero-steps6-shift3-confirm2" for candidate in candidates)
    assert any(candidate.candidate_id == "fastwam-robotwin-steps8-shift5-confirm1" for candidate in candidates)
    assert not any("steps4" in candidate.candidate_id for candidate in candidates)
    assert len({candidate.summary_path for candidate in candidates}) == len(candidates)
    assert all("--opt scheduler" in candidate.command for candidate in candidates)

    write_sweep_manifest(candidates, tmp_path)
    report_text = (tmp_path / "scheduler_report_command.sh").read_text(encoding="utf-8")
    assert "--allow-missing-quality-reference" in report_text
    assert "--min-config-repeats 2" in report_text
    assert "python -m eazywam.evals.scheduler_bundle" in report_text
