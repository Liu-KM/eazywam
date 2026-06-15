from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SchedulerAcceptanceError(RuntimeError):
    """Raised when scheduler sweep results do not prove acceptance."""


@dataclass(frozen=True)
class SchedulerAcceptanceReport:
    report_path: Path
    row_count: int
    model_ids: tuple[str, ...]
    baseline_steps: int
    recommended_count: int
    not_recommended_count: int
    pareto_count: int
    min_config_repeats: int

    def message(self) -> str:
        return (
            "FastWAM scheduler report acceptance passed: "
            f"report={self.report_path} rows={self.row_count} "
            f"models={','.join(self.model_ids)} "
            f"baseline_steps={self.baseline_steps} "
            f"recommended={self.recommended_count} "
            f"not_recommended={self.not_recommended_count} "
            f"pareto={self.pareto_count} "
            f"min_config_repeats={self.min_config_repeats}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "ok",
            "report_path": str(self.report_path),
            "row_count": self.row_count,
            "model_ids": list(self.model_ids),
            "baseline_steps": self.baseline_steps,
            "recommended_count": self.recommended_count,
            "not_recommended_count": self.not_recommended_count,
            "pareto_count": self.pareto_count,
            "min_config_repeats": self.min_config_repeats,
        }


def validate_scheduler_report(
    report_path: str | Path,
    *,
    baseline_steps: int = 10,
    required_models: tuple[str, ...] = ("fastwam-libero", "fastwam-robotwin"),
    require_candidate_ids: bool = True,
    require_job_ids: bool = True,
    require_summary_paths: bool = True,
    require_trace_paths: bool = True,
    require_action_drift: bool = True,
    require_report_artifacts: bool = True,
    require_recommendation: bool = False,
    require_not_recommended: bool = False,
    require_quality_reference: bool = True,
    min_config_repeats: int = 1,
    expected_scheduler_name: str = "fastwam_flowmatch_euler",
    expected_solver: str = "euler",
    expected_schedule_type: str = "shifted_flowmatch",
) -> SchedulerAcceptanceReport:
    path = Path(report_path).expanduser()
    rows = _load_rows(path)
    errors: list[str] = []

    if min_config_repeats < 1:
        errors.append(f"min_config_repeats is {min_config_repeats}, expected >= 1")
    if not rows:
        errors.append("scheduler report has no rows")
    if require_report_artifacts:
        errors.extend(_validate_report_artifacts(path, rows))
    errors.extend(_validate_config_repeats(rows, min_config_repeats=min_config_repeats))

    model_ids = tuple(sorted({str(row.get("model_id")) for row in rows if row.get("model_id")}))
    for model_id in required_models:
        model_rows = [row for row in rows if row.get("model_id") == model_id]
        if not model_rows:
            errors.append(f"report is missing model_id={model_id}")
            continue
        errors.extend(
            _validate_model_rows(
                model_id,
                model_rows,
                baseline_steps=baseline_steps,
                require_quality_reference=require_quality_reference,
            )
        )

    for row in rows:
        label = _row_label(row)
        if require_candidate_ids and not row.get("candidate_id"):
            errors.append(f"{label} is missing candidate_id")
        if require_job_ids and not row.get("job_id"):
            errors.append(f"{label} is missing job_id")
        if require_summary_paths:
            summary_path = _path_value(row.get("summary_path"), base_dir=path.parent)
            if summary_path is None:
                errors.append(f"{label} is missing summary_path")
            elif not summary_path.exists():
                errors.append(f"{label} summary_path does not exist: {summary_path}")
        if require_trace_paths:
            trace_path = _path_value(row.get("trace_path"), base_dir=path.parent)
            if trace_path is None:
                errors.append(f"{label} is missing trace_path")
            elif not trace_path.exists():
                errors.append(f"{label} trace_path does not exist: {trace_path}")
        for key in (
            "command",
            "config_key",
            "scheduler_name",
            "solver",
            "schedule_type",
            "schedule_source",
            "num_inference_steps",
            "timestep_count",
            "timesteps",
            "sigmas",
            "deltas",
            "total_ms",
            "denoise_wall_ms",
            "success_rate",
            "speedup",
            "failure_count",
            "recommendation",
            "recommendation_reason",
        ):
            if row.get(key) is None or row.get(key) == "":
                errors.append(f"{label} is missing {key}")
        for key in ("sigma_shift", "fallback_reason", "failure_cases"):
            if key not in row:
                errors.append(f"{label} is missing {key}")
        if require_action_drift:
            if row.get("action_drift") is None or row.get("action_drift") == "":
                errors.append(f"{label} is missing action_drift")
            if row.get("action_drift_fields") is None or row.get("action_drift_fields") == "":
                errors.append(f"{label} is missing action_drift_fields")
        errors.extend(_validate_metric_ranges(row, label, require_action_drift=require_action_drift))
        errors.extend(
            _validate_recommendation_consistency(
                row,
                label,
                baseline_steps=baseline_steps,
            )
        )
        errors.extend(
            _validate_scheduler_identity(
                row,
                label,
                expected_scheduler_name=expected_scheduler_name,
                expected_solver=expected_solver,
                expected_schedule_type=expected_schedule_type,
            )
        )
        errors.extend(_validate_command(row, label))
        errors.extend(_validate_schedule_summaries(row, label))

    recommended_count = sum(1 for row in rows if row.get("recommendation") == "recommended_candidate")
    not_recommended_count = sum(1 for row in rows if row.get("recommendation") == "not_recommended")
    pareto_count = sum(1 for row in rows if row.get("pareto") is True)
    if pareto_count == 0:
        errors.append("report has no Pareto rows")
    if require_recommendation and recommended_count == 0:
        errors.append("report has no recommended_candidate rows")
    if require_not_recommended and not_recommended_count == 0:
        errors.append("report has no not_recommended rows")
    if _has_quality_reference(rows, baseline_steps=baseline_steps) and not_recommended_count == 0:
        errors.append("report has quality-reference rows but no not_recommended rows")

    if errors:
        raise SchedulerAcceptanceError("\n".join(errors))

    return SchedulerAcceptanceReport(
        report_path=path,
        row_count=len(rows),
        model_ids=model_ids,
        baseline_steps=baseline_steps,
        recommended_count=recommended_count,
        not_recommended_count=not_recommended_count,
        pareto_count=pareto_count,
        min_config_repeats=min_config_repeats,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate completed FastWAM scheduler report artifacts")
    parser.add_argument("report_json", help="scheduler_results.json from scheduler_report")
    parser.add_argument("--baseline-steps", type=int, default=10)
    parser.add_argument(
        "--require-model",
        action="append",
        dest="required_models",
        default=None,
        help="Required model id; may be repeated",
    )
    parser.add_argument("--allow-missing-job-ids", action="store_true")
    parser.add_argument("--allow-missing-candidate-ids", action="store_true")
    parser.add_argument("--allow-missing-summary-paths", action="store_true")
    parser.add_argument("--allow-missing-trace-paths", action="store_true")
    parser.add_argument("--allow-missing-action-drift", action="store_true")
    parser.add_argument("--allow-missing-report-artifacts", action="store_true")
    parser.add_argument("--allow-missing-quality-reference", action="store_true")
    parser.add_argument("--require-recommendation", action="store_true")
    parser.add_argument("--require-not-recommended", action="store_true")
    parser.add_argument("--expected-scheduler-name", default="fastwam_flowmatch_euler")
    parser.add_argument("--expected-solver", default="euler")
    parser.add_argument("--expected-schedule-type", default="shifted_flowmatch")
    parser.add_argument(
        "--min-config-repeats",
        type=int,
        default=1,
        help="Minimum repeat count for each model/workload/step/sigma config",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    args = parser.parse_args(argv)

    try:
        report = validate_scheduler_report(
            args.report_json,
            baseline_steps=args.baseline_steps,
            required_models=tuple(args.required_models or ("fastwam-libero", "fastwam-robotwin")),
            require_candidate_ids=not args.allow_missing_candidate_ids,
            require_job_ids=not args.allow_missing_job_ids,
            require_summary_paths=not args.allow_missing_summary_paths,
            require_trace_paths=not args.allow_missing_trace_paths,
            require_action_drift=not args.allow_missing_action_drift,
            require_report_artifacts=not args.allow_missing_report_artifacts,
            require_recommendation=args.require_recommendation,
            require_not_recommended=args.require_not_recommended,
            require_quality_reference=not args.allow_missing_quality_reference,
            min_config_repeats=args.min_config_repeats,
            expected_scheduler_name=args.expected_scheduler_name,
            expected_solver=args.expected_solver,
            expected_schedule_type=args.expected_schedule_type,
        )
    except (OSError, json.JSONDecodeError, SchedulerAcceptanceError) as exc:
        print(f"scheduler acceptance error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.message())
    return 0


def _validate_model_rows(
    model_id: str,
    rows: list[dict[str, Any]],
    *,
    baseline_steps: int,
    require_quality_reference: bool,
) -> list[str]:
    errors = []
    baseline_rows = [
        row
        for row in rows
        if row.get("num_inference_steps") == baseline_steps
        and _requested_sigma_shift(row) is None
    ]
    if not baseline_rows:
        errors.append(f"{model_id} is missing exact {baseline_steps}-step baseline row")
    elif not any(row.get("recommendation") == "baseline" for row in baseline_rows):
        errors.append(f"{model_id} exact baseline row is not labeled baseline")
    for baseline_row in baseline_rows:
        baseline_label = _row_label(baseline_row)
        speedup = _float_value(baseline_row.get("speedup"))
        success_delta = _float_value(baseline_row.get("success_delta"))
        if speedup is not None and abs(speedup - 1.0) > 1e-6:
            errors.append(f"{baseline_label} baseline speedup is {speedup!r}, expected 1.0")
        if success_delta is not None and abs(success_delta) > 1e-6:
            errors.append(
                f"{baseline_label} baseline success_delta is {success_delta!r}, expected 0.0"
            )
    if not any(_step_count(row) is not None and _step_count(row) < baseline_steps for row in rows):
        errors.append(f"{model_id} has no faster-than-baseline candidate")
    if require_quality_reference and not any(
        _step_count(row) is not None and _step_count(row) > baseline_steps for row in rows
    ):
        errors.append(f"{model_id} has no quality-reference candidate above baseline")
    if not any(row.get("pareto") is True for row in rows):
        errors.append(f"{model_id} has no Pareto row")
    return errors


def _has_quality_reference(rows: list[dict[str, Any]], *, baseline_steps: int) -> bool:
    return any(_step_count(row) is not None and _step_count(row) > baseline_steps for row in rows)


def _validate_report_artifacts(report_path: Path, rows: list[dict[str, Any]]) -> list[str]:
    errors = []
    expected = {
        "scheduler_results.csv": ("speedup",),
        "scheduler_config_summary.json": (
            "aggregate_recommendation",
            "aggregate_pareto",
            "speedup_mean",
        ),
        "scheduler_config_summary.csv": (
            "aggregate_recommendation",
            "candidate_ids",
            "job_ids",
        ),
        "scheduler_report.html": (
            "<svg",
            "Config-Level Decisions",
            "Total latency ms",
            "Success rate",
            "Speedup vs success rate",
            "Drift vs speedup",
            "Results",
        ),
        "scheduler_final_report.md": (
            "FastWAM Scheduler / Sampler Report",
            "Per-Workload Conclusions",
            "Config-Level Decisions",
            "Repeat Summary",
            "Recommended Candidates",
            "Not Recommended",
            "Full Table",
            "does not claim `parity_verified`",
        ),
    }
    for filename, markers in expected.items():
        path = report_path.parent / filename
        if not path.exists():
            errors.append(f"report artifact is missing: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                errors.append(f"report artifact {path} is missing marker {marker!r}")
    config_summary_path = report_path.parent / "scheduler_config_summary.json"
    if config_summary_path.exists():
        errors.extend(_validate_config_summary_artifact(config_summary_path, rows))
    return errors


def _validate_config_summary_artifact(
    config_summary_path: Path,
    rows: list[dict[str, Any]],
) -> list[str]:
    data = json.loads(config_summary_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return [f"config summary JSON must be a list: {config_summary_path}"]
    summaries = [row for row in data if isinstance(row, dict)]
    if len(summaries) != len(data):
        return [f"config summary rows must be objects: {config_summary_path}"]

    errors = []
    groups = _config_groups(rows)
    summary_by_key = {_config_summary_key(summary): summary for summary in summaries}
    for key, group in groups.items():
        label = _config_group_label(key)
        summary = summary_by_key.get(key)
        if summary is None:
            errors.append(f"{label} is missing from scheduler_config_summary.json")
            continue
        errors.extend(_validate_config_summary_row(label, summary, group))
    for key in summary_by_key:
        if key not in groups:
            errors.append(f"{_config_group_label(key)} is unexpected in scheduler_config_summary.json")
    return errors


def _validate_config_summary_row(
    label: str,
    summary: dict[str, Any],
    group: list[dict[str, Any]],
) -> list[str]:
    errors = []
    first = group[0]
    expected_steps = _step_count(first)
    expected_sigma_shift = _sigma_shift(first.get("sigma_shift"))
    repeats = _step_count_value(summary.get("repeats"))
    if repeats != len(group):
        errors.append(f"{label} summary repeats is {repeats!r}, expected {len(group)}")
    if _step_count_value(summary.get("num_inference_steps")) != expected_steps:
        errors.append(f"{label} summary num_inference_steps does not match rows")
    actual_sigma_shift = _sigma_shift(summary.get("sigma_shift"))
    if actual_sigma_shift != expected_sigma_shift:
        errors.append(f"{label} summary sigma_shift does not match rows")
    for summary_field, row_field in (
        ("total_ms_mean", "total_ms"),
        ("denoise_wall_ms_mean", "denoise_wall_ms"),
        ("success_rate_mean", "success_rate"),
        ("success_delta_mean", "success_delta"),
        ("speedup_mean", "speedup"),
        ("action_drift_mean", "action_drift"),
    ):
        expected = _mean_row_field(group, row_field)
        actual = _float_value(summary.get(summary_field))
        if expected is not None and not _close(actual, expected):
            errors.append(f"{label} {summary_field} is {actual!r}, expected {expected!r}")
    expected_failure_count = sum(_int_value(row.get("failure_count")) or 0 for row in group)
    actual_failure_count = _int_value(summary.get("failure_count"))
    if actual_failure_count != expected_failure_count:
        errors.append(
            f"{label} failure_count is {actual_failure_count!r}, expected {expected_failure_count}"
        )
    for summary_field, row_field in (("candidate_ids", "candidate_id"), ("job_ids", "job_id")):
        expected_values = {str(row[row_field]) for row in group if row.get(row_field)}
        actual_values = _split_summary_values(summary.get(summary_field))
        missing = expected_values - actual_values
        if missing:
            errors.append(f"{label} {summary_field} is missing {','.join(sorted(missing))}")
    if summary.get("aggregate_recommendation") in {None, ""}:
        errors.append(f"{label} is missing aggregate_recommendation")
    if "aggregate_pareto" not in summary:
        errors.append(f"{label} is missing aggregate_pareto")
    return errors


def _validate_config_repeats(
    rows: list[dict[str, Any]],
    *,
    min_config_repeats: int,
) -> list[str]:
    if min_config_repeats <= 1:
        return []
    counts: dict[tuple[str, str, str], int] = {}
    labels: dict[tuple[str, str, str], str] = {}
    for row in rows:
        key = (
            str(row.get("model_id")),
            str(row.get("workload")),
            _config_key(row),
        )
        counts[key] = counts.get(key, 0) + 1
        labels[key] = _row_label(row)

    errors = []
    for key, count in counts.items():
        if count < min_config_repeats:
            errors.append(
                f"{labels[key]} has {count} repeat(s), expected at least {min_config_repeats}"
            )
    return errors


def _config_groups(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = _config_row_key(row)
        groups.setdefault(key, []).append(row)
    return groups


def _config_row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("model_id") or ""),
        str(row.get("workload") or ""),
        str(row.get("solver") or ""),
        _config_key(row),
    )


def _config_summary_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("model_id") or ""),
        str(row.get("workload") or ""),
        str(row.get("solver") or ""),
        str(row.get("config_key") or _config_key(row)),
    )


def _config_key(row: dict[str, Any]) -> str:
    configured = row.get("config_key")
    if configured not in {None, ""}:
        return str(configured)
    return (
        f"steps={_step_count(row)},"
        f"sigma_shift={_command_sigma_shift(row.get('sigma_shift'))}"
    )


def _config_group_label(key: tuple[str, str, str, str]) -> str:
    model_id, workload, solver, config_key = key
    return f"{model_id}:{workload}:{solver}:{config_key}"


def _mean_row_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [_float_value(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _int_value(value: object) -> int | None:
    number = _float_value(value)
    return int(number) if number is not None else None


def _close(actual: float | None, expected: float, *, tolerance: float = 1e-6) -> bool:
    return actual is not None and abs(actual - expected) <= tolerance


def _split_summary_values(value: object) -> set[str]:
    if value is None:
        return set()
    return {item for item in str(value).replace("<br>", ",").split(",") if item}


def _validate_metric_ranges(
    row: dict[str, Any],
    label: str,
    *,
    require_action_drift: bool,
) -> list[str]:
    errors = []
    positive_fields = (
        "num_inference_steps",
        "timestep_count",
        "total_ms",
        "denoise_wall_ms",
        "speedup",
    )
    for field in positive_fields:
        value = _float_value(row.get(field))
        if value is not None and value <= 0:
            errors.append(f"{label} {field} is {value!r}, expected > 0")
    success_rate = _float_value(row.get("success_rate"))
    if success_rate is not None and not 0.0 <= success_rate <= 1.0:
        errors.append(f"{label} success_rate is {success_rate!r}, expected 0..1")
    success_delta = _float_value(row.get("success_delta"))
    if success_delta is not None and not -1.0 <= success_delta <= 1.0:
        errors.append(f"{label} success_delta is {success_delta!r}, expected -1..1")
    failure_count = _float_value(row.get("failure_count"))
    if failure_count is not None and failure_count < 0:
        errors.append(f"{label} failure_count is {failure_count!r}, expected >= 0")
    action_drift = _float_value(row.get("action_drift"))
    if require_action_drift and action_drift is not None and action_drift < 0:
        errors.append(f"{label} action_drift is {action_drift!r}, expected >= 0")
    return errors


def _validate_recommendation_consistency(
    row: dict[str, Any],
    label: str,
    *,
    baseline_steps: int,
) -> list[str]:
    errors = []
    recommendation = row.get("recommendation")
    steps = _step_count(row)
    sigma_shift = _requested_sigma_shift(row)
    speedup = _float_value(row.get("speedup"))
    fallback_reason = row.get("fallback_reason")
    if recommendation == "baseline":
        if steps != baseline_steps or sigma_shift is not None:
            errors.append(
                f"{label} is labeled baseline but is not exact {baseline_steps}-step baseline"
            )
    if fallback_reason and recommendation != "not_recommended":
        errors.append(f"{label} has fallback_reason but recommendation is {recommendation!r}")
    if recommendation == "recommended_candidate":
        if row.get("pareto") is not True:
            errors.append(f"{label} is recommended_candidate but is not Pareto")
        if speedup is not None and speedup <= 1.0:
            errors.append(f"{label} is recommended_candidate but speedup is {speedup!r}")
        if fallback_reason:
            errors.append(f"{label} is recommended_candidate but has fallback_reason")
    return errors


def _validate_scheduler_identity(
    row: dict[str, Any],
    label: str,
    *,
    expected_scheduler_name: str,
    expected_solver: str,
    expected_schedule_type: str,
) -> list[str]:
    errors = []
    expected = {
        "scheduler_name": expected_scheduler_name,
        "solver": expected_solver,
        "schedule_type": expected_schedule_type,
    }
    for key, expected_value in expected.items():
        actual = row.get(key)
        if actual is not None and actual != "" and str(actual) != expected_value:
            errors.append(f"{label} {key} is {actual!r}, expected {expected_value!r}")
    return errors


def _validate_command(row: dict[str, Any], label: str) -> list[str]:
    command = row.get("command")
    if command is None or command == "":
        return []
    text = str(command)
    errors = []
    model_id = str(row.get("model_id") or "")
    if model_id and f"wam eval {model_id}" not in text:
        errors.append(f"{label} command does not include 'wam eval {model_id}'")
    if "--opt scheduler" not in text:
        errors.append(f"{label} command does not include '--opt scheduler'")
    steps = _step_count_value(row.get("num_inference_steps"))
    if steps is not None and f"num_inference_steps={steps}" not in text:
        errors.append(f"{label} command does not include 'num_inference_steps={steps}'")
    sigma_value = row.get("requested_sigma_shift") if "requested_sigma_shift" in row else row.get("sigma_shift")
    sigma_shift = _command_sigma_shift(sigma_value)
    if sigma_shift is not None and f"sigma_shift={sigma_shift}" not in text:
        errors.append(f"{label} command does not include 'sigma_shift={sigma_shift}'")
    return errors


def _command_sigma_shift(value: object) -> str | None:
    if value is None:
        return "null"
    text = str(value)
    if text.lower() in {"", "none", "null"}:
        return "null"
    number = _float_value(value)
    if number is None:
        return text
    return f"{number:.4g}"


def _validate_schedule_summaries(row: dict[str, Any], label: str) -> list[str]:
    errors = []
    timestep_count = _step_count_value(row.get("timestep_count"))
    if timestep_count is None:
        return errors
    num_inference_steps = _step_count_value(row.get("num_inference_steps"))
    if num_inference_steps is not None and timestep_count != num_inference_steps:
        errors.append(
            f"{label} timestep_count is {timestep_count}, expected num_inference_steps "
            f"{num_inference_steps}"
        )
    for field in ("timesteps", "sigmas", "deltas"):
        summary = _summary_value(row.get(field), label=label, field=field, errors=errors)
        if summary is None:
            continue
        count = _step_count_value(summary.get("count"))
        if count is None:
            errors.append(f"{label} {field}.count is missing or invalid")
        elif count != timestep_count:
            errors.append(
                f"{label} {field}.count is {count}, expected timestep_count {timestep_count}"
            )
        if count is not None and count > 0:
            for key in ("first", "last", "min", "max"):
                if summary.get(key) is None:
                    errors.append(f"{label} {field}.{key} is missing")
    return errors


def _summary_value(
    value: object,
    *,
    label: str,
    field: str,
    errors: list[str],
) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if value is None:
        return None
    text = str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        errors.append(f"{label} {field} is not valid JSON summary: {exc.msg}")
        return None
    if not isinstance(parsed, dict):
        errors.append(f"{label} {field} summary is not an object")
        return None
    return parsed


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SchedulerAcceptanceError(f"scheduler report JSON must be a list: {path}")
    rows = [row for row in data if isinstance(row, dict)]
    if len(rows) != len(data):
        raise SchedulerAcceptanceError(f"scheduler report rows must be objects: {path}")
    return rows


def _path_value(value: object, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if text.lower() in {"", "none", "null"}:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return (base_dir / path).resolve()


def _step_count(row: dict[str, Any]) -> int | None:
    return _step_count_value(row.get("num_inference_steps"))


def _step_count_value(value: object) -> int | None:
    if value is None:
        return None
    return int(float(str(value)))


def _float_value(value: object) -> float | None:
    if value is None:
        return None
    text = str(value)
    if text.lower() in {"", "none", "null"}:
        return None
    return float(text)


def _sigma_shift(value: object) -> float | None:
    if value is None:
        return None
    text = str(value)
    if text.lower() in {"", "none", "null"}:
        return None
    return float(text)


def _requested_sigma_shift(row: dict[str, Any]) -> float | None:
    if "requested_sigma_shift" in row:
        return _sigma_shift(row.get("requested_sigma_shift"))
    return _sigma_shift(row.get("sigma_shift"))


def _row_label(row: dict[str, Any]) -> str:
    candidate_id = row.get("candidate_id")
    prefix = f"{candidate_id}:" if candidate_id else ""
    return (
        f"{prefix}{row.get('model_id')}:{row.get('workload')}:"
        f"steps={row.get('num_inference_steps')}:sigma_shift={row.get('sigma_shift')}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
