from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


class SchedulerReportError(ValueError):
    """Raised when scheduler sweep artifacts cannot be summarized."""


@dataclass(frozen=True)
class JobMapEntry:
    candidate_id: str | None = None
    job_id: str | None = None
    command: str | None = None


@dataclass(frozen=True)
class SchedulerRun:
    summary_path: Path
    trace_path: Path
    candidate_id: str | None
    run_id: str
    model_id: str
    workload: str | None
    status: str
    command: str
    job_id: str | None
    scheduler_name: str | None
    solver: str | None
    schedule_type: str | None
    schedule_source: str | None
    num_inference_steps: int | None
    sigma_shift: float | None
    requested_sigma_shift: str | None
    timestep_count: int | None
    timesteps: str | None
    sigmas: str | None
    deltas: str | None
    total_ms: float | None
    denoise_wall_ms: float | None
    success_rate: float | None
    action_drift: float | None = None
    action_drift_fields: str | None = None
    failure_count: int = 0
    failure_cases: str | None = None
    speedup: float | None = None
    success_delta: float | None = None
    baseline_key: str | None = None
    is_baseline_reference: bool = False
    pareto: bool = False
    fallback_reason: str | None = None
    recommendation: str = "needs_review"
    recommendation_reason: str = "not_compared"

    @property
    def run_key(self) -> str:
        return _run_key(self.model_id, self.workload)

    @property
    def config_key(self) -> str:
        key = f"steps={self.num_inference_steps},sigma_shift={_format_optional(self.sigma_shift)}"
        if self.schedule_source and self.schedule_source != "generated":
            key += f",schedule_source={self.schedule_source},schedule_hash={self._schedule_hash()}"
        return key

    def _schedule_hash(self) -> str:
        payload = {
            "timesteps": self.timesteps,
            "sigmas": self.sigmas,
            "deltas": self.deltas,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:12]

    def with_comparison(
        self,
        *,
        speedup: float | None,
        action_drift: float | None,
        action_drift_fields: str | None,
        baseline_key: str | None,
        pareto: bool,
        is_baseline_reference: bool | None = None,
        success_delta: float | None = None,
        recommendation: str | None = None,
        recommendation_reason: str | None = None,
    ) -> "SchedulerRun":
        return SchedulerRun(
            summary_path=self.summary_path,
            trace_path=self.trace_path,
            candidate_id=self.candidate_id,
            run_id=self.run_id,
            model_id=self.model_id,
            workload=self.workload,
            status=self.status,
            command=self.command,
            job_id=self.job_id,
            scheduler_name=self.scheduler_name,
            solver=self.solver,
            schedule_type=self.schedule_type,
            schedule_source=self.schedule_source,
            num_inference_steps=self.num_inference_steps,
            sigma_shift=self.sigma_shift,
            requested_sigma_shift=self.requested_sigma_shift,
            timestep_count=self.timestep_count,
            timesteps=self.timesteps,
            sigmas=self.sigmas,
            deltas=self.deltas,
            total_ms=self.total_ms,
            denoise_wall_ms=self.denoise_wall_ms,
            success_rate=self.success_rate,
            action_drift=action_drift,
            action_drift_fields=action_drift_fields,
            failure_count=self.failure_count,
            failure_cases=self.failure_cases,
            speedup=speedup,
            success_delta=success_delta,
            baseline_key=baseline_key,
            is_baseline_reference=(
                self.is_baseline_reference
                if is_baseline_reference is None
                else is_baseline_reference
            ),
            pareto=pareto,
            fallback_reason=self.fallback_reason,
            recommendation=recommendation or self.recommendation,
            recommendation_reason=recommendation_reason or self.recommendation_reason,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "summary_path": str(self.summary_path),
            "trace_path": str(self.trace_path),
            "candidate_id": self.candidate_id,
            "run_id": self.run_id,
            "model_id": self.model_id,
            "workload": self.workload,
            "config_key": self.config_key,
            "status": self.status,
            "command": self.command,
            "job_id": self.job_id,
            "scheduler_name": self.scheduler_name,
            "solver": self.solver,
            "schedule_type": self.schedule_type,
            "schedule_source": self.schedule_source,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "requested_sigma_shift": self.requested_sigma_shift,
            "timestep_count": self.timestep_count,
            "timesteps": self.timesteps,
            "sigmas": self.sigmas,
            "deltas": self.deltas,
            "total_ms": self.total_ms,
            "denoise_wall_ms": self.denoise_wall_ms,
            "speedup": self.speedup,
            "success_rate": self.success_rate,
            "success_delta": self.success_delta,
            "action_drift": self.action_drift,
            "action_drift_fields": self.action_drift_fields,
            "failure_count": self.failure_count,
            "failure_cases": self.failure_cases,
            "baseline_key": self.baseline_key,
            "is_baseline_reference": self.is_baseline_reference,
            "pareto": self.pareto,
            "fallback_reason": self.fallback_reason,
            "recommendation": self.recommendation,
            "recommendation_reason": self.recommendation_reason,
        }


@dataclass(frozen=True)
class SchedulerConfigSummary:
    model_id: str
    workload: str | None
    solver: str | None
    config_key: str
    repeats: int
    num_inference_steps: int | None
    sigma_shift: float | None
    total_ms_mean: float | None
    total_ms_std: float | None
    denoise_wall_ms_mean: float | None
    success_rate_mean: float | None
    success_delta_mean: float | None
    speedup_mean: float | None
    speedup_std: float | None
    action_drift_mean: float | None
    failure_count: int
    job_ids: str
    candidate_ids: str
    trace_paths: str
    aggregate_pareto: bool = False
    aggregate_recommendation: str = "needs_review"
    aggregate_reason: str = "not_compared"

    @property
    def run_key(self) -> str:
        return _run_key(self.model_id, self.workload)

    def with_decision(
        self,
        *,
        aggregate_pareto: bool,
        aggregate_recommendation: str,
        aggregate_reason: str,
    ) -> "SchedulerConfigSummary":
        return SchedulerConfigSummary(
            model_id=self.model_id,
            workload=self.workload,
            solver=self.solver,
            config_key=self.config_key,
            repeats=self.repeats,
            num_inference_steps=self.num_inference_steps,
            sigma_shift=self.sigma_shift,
            total_ms_mean=self.total_ms_mean,
            total_ms_std=self.total_ms_std,
            denoise_wall_ms_mean=self.denoise_wall_ms_mean,
            success_rate_mean=self.success_rate_mean,
            success_delta_mean=self.success_delta_mean,
            speedup_mean=self.speedup_mean,
            speedup_std=self.speedup_std,
            action_drift_mean=self.action_drift_mean,
            failure_count=self.failure_count,
            job_ids=self.job_ids,
            candidate_ids=self.candidate_ids,
            trace_paths=self.trace_paths,
            aggregate_pareto=aggregate_pareto,
            aggregate_recommendation=aggregate_recommendation,
            aggregate_reason=aggregate_reason,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "workload": self.workload,
            "solver": self.solver,
            "config_key": self.config_key,
            "repeats": self.repeats,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "total_ms_mean": self.total_ms_mean,
            "total_ms_std": self.total_ms_std,
            "denoise_wall_ms_mean": self.denoise_wall_ms_mean,
            "success_rate_mean": self.success_rate_mean,
            "success_delta_mean": self.success_delta_mean,
            "speedup_mean": self.speedup_mean,
            "speedup_std": self.speedup_std,
            "action_drift_mean": self.action_drift_mean,
            "failure_count": self.failure_count,
            "job_ids": self.job_ids,
            "candidate_ids": self.candidate_ids,
            "trace_paths": self.trace_paths,
            "aggregate_pareto": self.aggregate_pareto,
            "aggregate_recommendation": self.aggregate_recommendation,
            "aggregate_reason": self.aggregate_reason,
        }


def build_report(
    summary_paths: list[str | Path],
    *,
    baseline_steps: int = 10,
    job_map_path: str | Path | None = None,
    min_recommend_speedup: float = 1.05,
    max_success_drop: float = 0.05,
    max_action_drift: float | None = None,
) -> list[SchedulerRun]:
    job_map = _read_job_map(job_map_path) if job_map_path is not None else {}
    runs = [_apply_job_map(load_scheduler_run(path), job_map) for path in summary_paths]
    baselines = _baseline_runs(runs, baseline_steps=baseline_steps)
    compared = [
        _compare_to_baseline(run, baselines.get(run.run_key))
        for run in runs
    ]
    pareto_keys = _pareto_keys(compared)
    return [
        _classify_recommendation(
            run.with_comparison(
                speedup=run.speedup,
                action_drift=run.action_drift,
                action_drift_fields=run.action_drift_fields,
                success_delta=run.success_delta,
                baseline_key=run.baseline_key,
                pareto=run.config_key in pareto_keys.get(run.run_key, set()),
            ),
            min_recommend_speedup=min_recommend_speedup,
            max_success_drop=max_success_drop,
            max_action_drift=max_action_drift,
        )
        for run in compared
    ]


def load_scheduler_run(summary_path: str | Path) -> SchedulerRun:
    path = Path(summary_path)
    summary = _read_json(path)
    trace_path = Path(str(summary.get("trace_path", "")))
    if not trace_path.is_absolute():
        trace_path = (path.parent / trace_path).resolve() if not trace_path.exists() else trace_path
    events = _read_trace(trace_path)
    plan = _first_event(events, "native_eval_plan") or {}
    inference_events = [event for event in events if event.get("event") == "inference_end"]
    backend_metadata = [
        metadata
        for metadata in (
            event.get("backend_metadata")
            for event in inference_events
        )
        if isinstance(metadata, dict)
    ]

    runtime_options = plan.get("runtime_options")
    if not isinstance(runtime_options, dict):
        runtime_options = {}
    scheduler_metadata = _first_scheduler_metadata(backend_metadata)
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    command = summary.get("command") if isinstance(summary.get("command"), dict) else {}
    command_text = str(command.get("display") or " ".join(command.get("argv", [])))
    failure_cases = _failure_cases(events)

    return SchedulerRun(
        summary_path=path,
        trace_path=trace_path,
        candidate_id=_candidate_id(summary, events),
        run_id=str(summary.get("run_id") or _first_value(events, "run_id") or ""),
        model_id=str(summary.get("model_id") or _first_value(events, "manifest_id") or ""),
        workload=_optional_str(summary.get("workload")),
        status=str(summary.get("status") or ""),
        command=command_text,
        job_id=_job_id(summary, events),
        scheduler_name=_optional_str(
            scheduler_metadata.get("scheduler_name")
            or runtime_options.get("scheduler_name")
        ),
        solver=_optional_str(scheduler_metadata.get("solver") or runtime_options.get("solver")),
        schedule_type=_optional_str(
            scheduler_metadata.get("schedule_type")
            or runtime_options.get("schedule_type")
        ),
        schedule_source=_optional_str(
            scheduler_metadata.get("schedule_source")
            or runtime_options.get("schedule_source")
            or "generated"
        ),
        num_inference_steps=_optional_int(
            scheduler_metadata.get("num_inference_steps")
            or runtime_options.get("num_inference_steps")
        ),
        sigma_shift=_optional_float(
            scheduler_metadata.get("sigma_shift")
            or runtime_options.get("sigma_shift")
        ),
        requested_sigma_shift=_requested_sigma_shift(runtime_options.get("sigma_shift")),
        timestep_count=_optional_int(scheduler_metadata.get("timestep_count")),
        timesteps=_summary_json(scheduler_metadata.get("timesteps")),
        sigmas=_summary_json(scheduler_metadata.get("sigmas")),
        deltas=_summary_json(scheduler_metadata.get("deltas")),
        total_ms=_mean_timing(inference_events, "total_ms", fallback_key="wall_ms"),
        denoise_wall_ms=_mean_backend_value(backend_metadata, "denoise_wall_ms"),
        success_rate=_success_rate(metrics, events),
        failure_count=len(failure_cases),
        failure_cases="; ".join(failure_cases) if failure_cases else None,
        fallback_reason=_fallback_reason(backend_metadata, events),
    )


def write_report_outputs(runs: list[SchedulerRun], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "scheduler_results.json"
    csv_path = out / "scheduler_results.csv"
    config_json_path = out / "scheduler_config_summary.json"
    config_csv_path = out / "scheduler_config_summary.csv"
    html_path = out / "scheduler_report.html"
    markdown_path = out / "scheduler_final_report.md"
    config_summaries = build_config_summaries(runs)
    json_path.write_text(
        json.dumps([run.to_dict() for run in runs], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_json_path.write_text(
        json.dumps([summary.to_dict() for summary in config_summaries], indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _write_csv(runs, csv_path)
    _write_config_csv(config_summaries, config_csv_path)
    html_path.write_text(_render_html(runs, config_summaries), encoding="utf-8")
    markdown_path.write_text(_render_markdown(runs, config_summaries), encoding="utf-8")
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "config_json": str(config_json_path),
        "config_csv": str(config_csv_path),
        "html": str(html_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a FastWAM scheduler sweep report")
    parser.add_argument("summaries", nargs="+", help="Eval summary JSON files")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV/JSON/HTML report")
    parser.add_argument("--baseline-steps", type=int, default=10)
    parser.add_argument("--min-recommend-speedup", type=float, default=1.05)
    parser.add_argument("--max-success-drop", type=float, default=0.05)
    parser.add_argument("--max-action-drift", type=float, default=None)
    parser.add_argument(
        "--job-map",
        default=None,
        help="Optional CSV with candidate_id,summary_path,job_id columns for SuperPod runs",
    )
    args = parser.parse_args(argv)

    runs = build_report(
        args.summaries,
        baseline_steps=args.baseline_steps,
        job_map_path=args.job_map,
        min_recommend_speedup=args.min_recommend_speedup,
        max_success_drop=args.max_success_drop,
        max_action_drift=args.max_action_drift,
    )
    outputs = write_report_outputs(runs, args.output_dir)
    print(json.dumps({"runs": len(runs), "outputs": outputs}, indent=2, sort_keys=True))
    return 0


def _compare_to_baseline(
    run: SchedulerRun,
    baseline: SchedulerRun | None,
) -> SchedulerRun:
    speedup = None
    action_drift = None
    action_drift_fields = None
    success_delta = None
    baseline_key = None
    is_baseline_reference = False
    if baseline is not None:
        baseline_key = baseline.config_key
        is_baseline_reference = _is_baseline_reference_run(run, baseline)
        if is_baseline_reference:
            speedup = 1.0
        elif baseline.total_ms is not None and run.total_ms is not None and run.total_ms > 0:
            speedup = baseline.total_ms / run.total_ms
        if is_baseline_reference:
            success_delta = 0.0
        elif baseline.success_rate is not None and run.success_rate is not None:
            success_delta = run.success_rate - baseline.success_rate
        action_drift, action_drift_fields = _action_drift(baseline.trace_path, run.trace_path)
    return run.with_comparison(
        speedup=speedup,
        action_drift=action_drift,
        action_drift_fields=action_drift_fields,
        success_delta=success_delta,
        baseline_key=baseline_key,
        pareto=False,
        is_baseline_reference=is_baseline_reference,
    )


def _same_run(left: SchedulerRun, right: SchedulerRun) -> bool:
    if left.summary_path == right.summary_path:
        return True
    return bool(left.candidate_id and left.candidate_id == right.candidate_id)


def _is_baseline_reference_run(run: SchedulerRun, baseline: SchedulerRun) -> bool:
    if _same_run(run, baseline):
        return True
    if not _requested_sigma_is_default(baseline):
        return False
    return (
        run.run_key == baseline.run_key
        and run.num_inference_steps == baseline.num_inference_steps
        and _requested_sigma_is_default(run)
    )


def _baseline_runs(runs: list[SchedulerRun], *, baseline_steps: int) -> dict[str, SchedulerRun]:
    baselines: dict[str, SchedulerRun] = {}
    for run in runs:
        if run.num_inference_steps != baseline_steps:
            continue
        current = baselines.get(run.run_key)
        if current is None or _score_baseline_candidate(run) > _score_baseline_candidate(current):
            baselines[run.run_key] = run
    return baselines


def _score_baseline_candidate(run: SchedulerRun) -> tuple[int, int]:
    sigma_is_default = _requested_sigma_is_default(run)
    status_ok = run.status == "ok"
    return (int(status_ok), int(sigma_is_default))


def _requested_sigma_is_default(run: SchedulerRun) -> bool:
    if run.requested_sigma_shift is None:
        return run.sigma_shift is None
    return run.requested_sigma_shift.strip().lower() in {"", "none", "null"}


def _pareto_keys(runs: list[SchedulerRun]) -> dict[str, set[str]]:
    by_group: dict[str, list[SchedulerRun]] = {}
    for run in runs:
        if run.speedup is None or run.success_rate is None:
            continue
        by_group.setdefault(run.run_key, []).append(run)

    result: dict[str, set[str]] = {}
    for key, group in by_group.items():
        frontier: set[str] = set()
        for candidate in group:
            dominated = False
            for other in group:
                if other is candidate:
                    continue
                if _dominates(other, candidate):
                    dominated = True
                    break
            if not dominated:
                frontier.add(candidate.config_key)
        result[key] = frontier
    return result


def _dominates(left: SchedulerRun, right: SchedulerRun) -> bool:
    if left.speedup is None or right.speedup is None:
        return False
    if left.success_rate is None or right.success_rate is None:
        return False
    at_least_equal = left.speedup >= right.speedup and left.success_rate >= right.success_rate
    strictly_better = left.speedup > right.speedup or left.success_rate > right.success_rate
    return at_least_equal and strictly_better


def _classify_recommendation(
    run: SchedulerRun,
    *,
    min_recommend_speedup: float,
    max_success_drop: float,
    max_action_drift: float | None,
) -> SchedulerRun:
    recommendation = "needs_review"
    reason = "insufficient_comparison_data"
    if run.baseline_key is None:
        reason = "missing_10_step_baseline"
    elif run.is_baseline_reference:
        recommendation = "baseline"
        reason = "baseline_reference"
    elif run.fallback_reason:
        recommendation = "not_recommended"
        reason = f"fallback:{run.fallback_reason}"
    elif run.speedup is None or run.success_delta is None:
        recommendation = "needs_review"
        reason = "missing_speed_or_success_metric"
    elif run.success_delta < -max_success_drop:
        recommendation = "not_recommended"
        reason = "success_drop_exceeds_threshold"
    elif max_action_drift is not None and run.action_drift is not None and run.action_drift > max_action_drift:
        recommendation = "not_recommended"
        reason = "action_drift_exceeds_threshold"
    elif run.speedup >= min_recommend_speedup and run.pareto:
        recommendation = "recommended_candidate"
        reason = "pareto_speedup_with_success_within_threshold"
    elif run.speedup > 1.0:
        recommendation = "experimental"
        reason = "faster_but_not_recommended_candidate"
    else:
        recommendation = "not_recommended"
        reason = "no_speedup"
    return run.with_comparison(
        speedup=run.speedup,
        action_drift=run.action_drift,
        action_drift_fields=run.action_drift_fields,
        success_delta=run.success_delta,
        baseline_key=run.baseline_key,
        pareto=run.pareto,
        is_baseline_reference=run.is_baseline_reference,
        recommendation=recommendation,
        recommendation_reason=reason,
    )


def build_config_summaries(runs: list[SchedulerRun]) -> list[SchedulerConfigSummary]:
    grouped = _config_groups(runs)
    summaries = [_summarize_config(group) for group in grouped.values()]
    pareto_keys = _aggregate_pareto_keys(summaries)
    return [
        _classify_config_summary(
            summary.with_decision(
                aggregate_pareto=summary.config_key in pareto_keys.get(summary.run_key, set()),
                aggregate_recommendation=summary.aggregate_recommendation,
                aggregate_reason=summary.aggregate_reason,
            ),
            grouped[(summary.model_id, summary.workload or "", summary.solver or "", summary.config_key)],
        )
        for summary in summaries
    ]


def _config_groups(
    runs: list[SchedulerRun],
) -> dict[tuple[str, str, str, str], list[SchedulerRun]]:
    groups: dict[tuple[str, str, str, str], list[SchedulerRun]] = {}
    for run in runs:
        key = (run.model_id, run.workload or "", run.solver or "", run.config_key)
        groups.setdefault(key, []).append(run)
    return dict(sorted(groups.items()))


def _summarize_config(group: list[SchedulerRun]) -> SchedulerConfigSummary:
    first = group[0]
    return SchedulerConfigSummary(
        model_id=first.model_id,
        workload=first.workload,
        solver=first.solver,
        config_key=first.config_key,
        repeats=len(group),
        num_inference_steps=first.num_inference_steps,
        sigma_shift=first.sigma_shift,
        total_ms_mean=_mean_field(group, "total_ms"),
        total_ms_std=_std_field(group, "total_ms"),
        denoise_wall_ms_mean=_mean_field(group, "denoise_wall_ms"),
        success_rate_mean=_mean_field(group, "success_rate"),
        success_delta_mean=_mean_field(group, "success_delta"),
        speedup_mean=_mean_field(group, "speedup"),
        speedup_std=_std_field(group, "speedup"),
        action_drift_mean=_mean_field(group, "action_drift"),
        failure_count=sum(run.failure_count for run in group),
        job_ids=",".join(run.job_id or "" for run in group),
        candidate_ids=",".join(run.candidate_id or "" for run in group),
        trace_paths="<br>".join(str(run.trace_path) for run in group),
    )


def _aggregate_pareto_keys(
    summaries: list[SchedulerConfigSummary],
) -> dict[str, set[str]]:
    by_group: dict[str, list[SchedulerConfigSummary]] = {}
    for summary in summaries:
        if summary.speedup_mean is None or summary.success_rate_mean is None:
            continue
        by_group.setdefault(summary.run_key, []).append(summary)

    result: dict[str, set[str]] = {}
    for key, group in by_group.items():
        frontier: set[str] = set()
        for candidate in group:
            dominated = False
            for other in group:
                if other is candidate:
                    continue
                if _dominates_summary(other, candidate):
                    dominated = True
                    break
            if not dominated:
                frontier.add(candidate.config_key)
        result[key] = frontier
    return result


def _dominates_summary(left: SchedulerConfigSummary, right: SchedulerConfigSummary) -> bool:
    if left.speedup_mean is None or right.speedup_mean is None:
        return False
    if left.success_rate_mean is None or right.success_rate_mean is None:
        return False
    at_least_equal = (
        left.speedup_mean >= right.speedup_mean
        and left.success_rate_mean >= right.success_rate_mean
    )
    strictly_better = (
        left.speedup_mean > right.speedup_mean
        or left.success_rate_mean > right.success_rate_mean
    )
    return at_least_equal and strictly_better


def _classify_config_summary(
    summary: SchedulerConfigSummary,
    group: list[SchedulerRun],
) -> SchedulerConfigSummary:
    recommendations = {run.recommendation for run in group}
    fallback = any(run.fallback_reason for run in group)
    if recommendations == {"baseline"}:
        recommendation = "baseline"
        reason = "baseline_reference"
    elif fallback:
        recommendation = "not_recommended"
        reason = "fallback_in_repeated_config"
    elif "not_recommended" in recommendations:
        recommendation = "not_recommended"
        reason = "one_or_more_repeats_not_recommended"
    elif summary.aggregate_pareto and "recommended_candidate" in recommendations:
        recommendation = "recommended_candidate"
        reason = "aggregate_pareto_with_recommended_repeats"
    elif summary.speedup_mean is not None and summary.speedup_mean > 1.0:
        recommendation = "experimental"
        reason = "aggregate_speedup_needs_review"
    else:
        recommendation = "needs_review"
        reason = "insufficient_aggregate_evidence"
    return summary.with_decision(
        aggregate_pareto=summary.aggregate_pareto,
        aggregate_recommendation=recommendation,
        aggregate_reason=reason,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SchedulerReportError(f"could not read summary: {path}") from exc
    if not isinstance(data, dict):
        raise SchedulerReportError(f"summary must be a JSON object: {path}")
    return data


def _read_job_map(path: str | Path) -> dict[str, JobMapEntry]:
    job_map_path = Path(path)
    with job_map_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    manifest_entries = _read_manifest_entries(job_map_path.parent)
    mapping: dict[str, JobMapEntry] = {}
    for row in rows:
        summary = row.get("summary_path") or row.get("summary") or row.get("path")
        candidate_id = row.get("candidate_id") or row.get("candidate")
        job_id = row.get("job_id") or row.get("superpod_job_id")
        if not summary:
            continue
        entry = JobMapEntry(
            candidate_id=str(candidate_id) if candidate_id else None,
            job_id=str(job_id) if job_id else None,
            command=str(row["command"]) if row.get("command") else None,
        )
        entry = _merge_job_map_entry(entry, manifest_entries.get(str(candidate_id or "")))
        summary_path = Path(summary)
        entry = _merge_job_map_entry(entry, manifest_entries.get(str(summary_path)))
        entry = _merge_job_map_entry(entry, manifest_entries.get(str(summary_path.resolve())))
        mapping[str(summary_path)] = entry
        mapping[str(summary_path.resolve())] = entry
        if entry.candidate_id:
            mapping[entry.candidate_id] = entry
    return mapping


def _read_manifest_entries(manifest_dir: Path) -> dict[str, JobMapEntry]:
    rows = _read_manifest_rows(manifest_dir)
    mapping: dict[str, JobMapEntry] = {}
    for row in rows:
        candidate_id = row.get("candidate_id") or row.get("candidate")
        summary = row.get("summary_path") or row.get("summary") or row.get("path")
        command = row.get("command")
        if not command:
            continue
        entry = JobMapEntry(
            candidate_id=str(candidate_id) if candidate_id else None,
            command=str(command),
        )
        if candidate_id:
            mapping[str(candidate_id)] = entry
        if summary:
            summary_path = Path(summary)
            mapping[str(summary_path)] = entry
            mapping[str(summary_path.resolve())] = entry
    return mapping


def _read_manifest_rows(manifest_dir: Path) -> list[dict[str, Any]]:
    csv_path = manifest_dir / "scheduler_sweep_candidates.csv"
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    json_path = manifest_dir / "scheduler_sweep_candidates.json"
    if not json_path.exists():
        return []
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def _merge_job_map_entry(
    entry: JobMapEntry,
    extra: JobMapEntry | None,
) -> JobMapEntry:
    if extra is None:
        return entry
    return JobMapEntry(
        candidate_id=entry.candidate_id or extra.candidate_id,
        job_id=entry.job_id or extra.job_id,
        command=entry.command or extra.command,
    )


def _apply_job_map(run: SchedulerRun, job_map: dict[str, JobMapEntry]) -> SchedulerRun:
    mapped = (
        job_map.get(str(run.summary_path))
        or job_map.get(str(run.summary_path.resolve()))
        or job_map.get(str(run.candidate_id or ""))
    )
    if mapped is None:
        return run
    return SchedulerRun(
        summary_path=run.summary_path,
        trace_path=run.trace_path,
        candidate_id=mapped.candidate_id or run.candidate_id,
        run_id=run.run_id,
        model_id=run.model_id,
        workload=run.workload,
        status=run.status,
        command=mapped.command or run.command,
        job_id=mapped.job_id or run.job_id,
        scheduler_name=run.scheduler_name,
        solver=run.solver,
        schedule_type=run.schedule_type,
        schedule_source=run.schedule_source,
        num_inference_steps=run.num_inference_steps,
        sigma_shift=run.sigma_shift,
        requested_sigma_shift=run.requested_sigma_shift,
        timestep_count=run.timestep_count,
        timesteps=run.timesteps,
        sigmas=run.sigmas,
        deltas=run.deltas,
        total_ms=run.total_ms,
        denoise_wall_ms=run.denoise_wall_ms,
        success_rate=run.success_rate,
        action_drift=run.action_drift,
        action_drift_fields=run.action_drift_fields,
        failure_count=run.failure_count,
        failure_cases=run.failure_cases,
        speedup=run.speedup,
        success_delta=run.success_delta,
        baseline_key=run.baseline_key,
        is_baseline_reference=run.is_baseline_reference,
        pareto=run.pareto,
        fallback_reason=run.fallback_reason,
        recommendation=run.recommendation,
        recommendation_reason=run.recommendation_reason,
    )


def _read_trace(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SchedulerReportError(f"could not read trace: {path}") from exc
    events = []
    for line in lines:
        if not line.strip():
            continue
        event = json.loads(line)
        if isinstance(event, dict):
            events.append(event)
    return events


def _first_event(events: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((event for event in events if event.get("event") == name), None)


def _first_value(events: list[dict[str, Any]], key: str) -> object | None:
    for event in events:
        if key in event:
            return event[key]
    return None


def _first_scheduler_metadata(metadatas: list[dict[str, Any]]) -> dict[str, Any]:
    for metadata in metadatas:
        if metadata.get("scheduler_name") is not None:
            return metadata
    return {}


def _summary_json(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _mean_timing(
    events: list[dict[str, Any]],
    key: str,
    *,
    fallback_key: str | None = None,
) -> float | None:
    values = []
    for event in events:
        timing = event.get("timing")
        if not isinstance(timing, dict):
            continue
        value = _optional_float(timing.get(key))
        if value is None and fallback_key is not None:
            value = _optional_float(timing.get(fallback_key))
        if value is not None:
            values.append(value)
    return mean(values) if values else None


def _mean_backend_value(metadatas: list[dict[str, Any]], key: str) -> float | None:
    values = [_optional_float(metadata.get(key)) for metadata in metadatas]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def _success_rate(metrics: dict[str, Any], events: list[dict[str, Any]]) -> float | None:
    for key in ("success_rate", "clean_success_rate", "randomized_success_rate"):
        value = _optional_float(metrics.get(key))
        if value is not None:
            return value
    successes = _optional_float(metrics.get("successes"))
    total = _optional_float(metrics.get("num_trials") or metrics.get("num_episodes"))
    if successes is not None and total is not None and total > 0:
        return successes / total
    episode_successes = [
        event.get("success")
        for event in events
        if event.get("event") == "episode_end" and event.get("success") is not None
    ]
    if episode_successes:
        return sum(1 for item in episode_successes if item) / len(episode_successes)
    return None


def _fallback_reason(
    metadatas: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> str | None:
    keys = (
        "scheduler_fallback_reason",
        "cuda_graph_fallback_reason",
        "torch_compile_fallback_reason",
        "batch_fallback_reason",
    )
    for metadata in metadatas:
        for key in keys:
            if metadata.get(key):
                return str(metadata[key])
    statuses = [
        item
        for event in events
        if event.get("event") == "optimization_profile_status"
        for item in event.get("profiles", [])
        if isinstance(item, dict)
    ]
    for status in statuses:
        if status.get("state") == "fallback" and status.get("reason"):
            return str(status["reason"])
    return None


def _failure_cases(events: list[dict[str, Any]]) -> list[str]:
    cases = []
    for event in events:
        event_name = event.get("event")
        if event_name == "episode_end" and event.get("success") is False:
            cases.append(_episode_failure_case(event))
        elif event_name == "error":
            cases.append(_error_failure_case(event))
        elif event_name == "run_end" and event.get("status") not in {None, "ok", "planned"}:
            cases.append(_run_end_failure_case(event))
    return cases


def _episode_failure_case(event: dict[str, Any]) -> str:
    parts = ["episode"]
    _append_case_part(parts, "episode_id", event)
    _append_case_part(parts, "episode_idx", event)
    _append_case_part(parts, "task_id", event)
    _append_case_part(parts, "task_name", event)
    _append_case_part(parts, "steps", event)
    _append_case_part(parts, "model_calls", event)
    return ",".join(parts)


def _error_failure_case(event: dict[str, Any]) -> str:
    parts = ["error"]
    _append_case_part(parts, "stage", event)
    _append_case_part(parts, "error_type", event)
    message = event.get("message")
    if message:
        parts.append(f"message={_short_text(message)}")
    return ",".join(parts)


def _run_end_failure_case(event: dict[str, Any]) -> str:
    parts = ["run_end"]
    _append_case_part(parts, "status", event)
    warnings = event.get("warnings")
    if warnings:
        parts.append(f"warnings={_short_text(warnings)}")
    return ",".join(parts)


def _append_case_part(parts: list[str], key: str, source: dict[str, Any]) -> None:
    value = source.get(key)
    if value is not None:
        parts.append(f"{key}={_short_text(value)}")


def _short_text(value: object, *, max_length: int = 120) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _job_id(summary: dict[str, Any], events: list[dict[str, Any]]) -> str | None:
    for source in (summary.get("metadata"), summary.get("metrics"), summary):
        if isinstance(source, dict):
            for key in ("job_id", "superpod_job_id"):
                if source.get(key) is not None:
                    return str(source[key])
    for event in events:
        for key in ("job_id", "superpod_job_id"):
            if event.get(key) is not None:
                return str(event[key])
    return None


def _candidate_id(summary: dict[str, Any], events: list[dict[str, Any]]) -> str | None:
    for source in (summary.get("metadata"), summary.get("metrics"), summary):
        if isinstance(source, dict) and source.get("candidate_id") is not None:
            return str(source["candidate_id"])
    for event in events:
        if event.get("candidate_id") is not None:
            return str(event["candidate_id"])
    return None


def _action_drift(baseline_trace: Path, variant_trace: Path) -> tuple[float | None, str | None]:
    baseline = _action_summary_values(_read_trace(baseline_trace))
    variant = _action_summary_values(_read_trace(variant_trace))
    if not baseline or not variant:
        return None, None
    count = min(len(baseline), len(variant))
    if count == 0:
        return None, None
    fields = [
        field
        for field in ("mean", "min", "max", "max_abs")
        if any(field in baseline[index] and field in variant[index] for index in range(count))
    ]
    if not fields:
        return None, None
    distances = []
    for index in range(count):
        for field in fields:
            if field in baseline[index] and field in variant[index]:
                distances.append(abs(variant[index][field] - baseline[index][field]))
    if not distances:
        return None, None
    return mean(distances), ",".join(fields)


def _action_summary_values(events: list[dict[str, Any]]) -> list[dict[str, float]]:
    summaries = []
    for event in events:
        summary = event.get("action_summary")
        if not isinstance(summary, dict):
            continue
        values = {
            field: value
            for field in ("mean", "min", "max", "max_abs")
            if (value := _optional_float(summary.get(field))) is not None
        }
        if values:
            summaries.append(values)
    return summaries


def _write_csv(runs: list[SchedulerRun], path: Path) -> None:
    fields = list(runs[0].to_dict()) if runs else list(SchedulerRun.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow(run.to_dict())


def _write_config_csv(summaries: list[SchedulerConfigSummary], path: Path) -> None:
    fields = (
        list(summaries[0].to_dict())
        if summaries
        else list(SchedulerConfigSummary.__dataclass_fields__)
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.to_dict())


def _render_html(runs: list[SchedulerRun], summaries: list[SchedulerConfigSummary]) -> str:
    rows = "\n".join(_render_table_row(run) for run in runs)
    summary_rows = "\n".join(_render_config_table_row(summary) for summary in summaries)
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>FastWAM Scheduler Sweep Report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2937; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #d1d5db; padding: 6px 8px; text-align: left; }}
th {{ background: #f3f4f6; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 20px; }}
svg {{ width: 100%; height: 260px; border: 1px solid #d1d5db; background: white; }}
</style>
<h1>FastWAM Scheduler Sweep Report</h1>
<p>Generated from {len(runs)} eval summaries. Baseline is selected per model/workload at
10 denoise steps.</p>
<div class="grid">
{_bar_chart(runs, "total_ms", "Total latency ms", lower_is_better=True)}
{_bar_chart(runs, "success_rate", "Success rate", lower_is_better=False)}
{_scatter_chart(runs, "speedup", "success_rate", "Speedup vs success rate")}
{_scatter_chart(runs, "speedup", "action_drift", "Drift vs speedup")}
</div>
<h2>Config-Level Decisions</h2>
<table>
<thead><tr>
<th>Model</th><th>Workload</th><th>Solver</th><th>Config</th><th>Repeats</th>
<th>Total ms mean</th><th>Total ms std</th><th>Denoise ms mean</th><th>Speedup mean</th>
<th>Speedup std</th><th>Success mean</th><th>Success delta mean</th><th>Drift mean</th>
<th>Failures</th><th>Aggregate Pareto</th><th>Aggregate Recommendation</th><th>Reason</th>
<th>Candidate IDs</th><th>Job IDs</th>
</tr></thead>
<tbody>
{summary_rows}
</tbody>
</table>
<h2>Results</h2>
<table>
<thead><tr>
<th>Candidate</th><th>Model</th><th>Workload</th><th>Solver</th><th>Schedule source</th><th>Steps</th><th>Sigma shift</th><th>Total ms</th>
<th>Timestep count</th><th>Denoise ms</th><th>Speedup</th><th>Success</th><th>Success delta</th><th>Drift</th>
<th>Drift fields</th><th>Failures</th><th>Failure cases</th><th>Pareto</th>
<th>Recommendation</th><th>Reason</th><th>Job</th><th>Trace</th><th>Timesteps</th><th>Sigmas</th><th>Deltas</th><th>Command</th><th>Fallback</th>
</tr></thead>
<tbody>
{rows}
</tbody>
</table>
</html>
"""


def _render_markdown(runs: list[SchedulerRun], summaries: list[SchedulerConfigSummary]) -> str:
    baselines = [run for run in runs if run.recommendation == "baseline"]
    pareto = [run for run in runs if run.pareto]
    recommended = [run for run in runs if run.recommendation == "recommended_candidate"]
    not_recommended = [run for run in runs if run.recommendation == "not_recommended"]
    missing_job_ids = [run for run in runs if not run.job_id]
    missing_candidate_ids = [run for run in runs if not run.candidate_id]
    missing_trace_paths = [run for run in runs if not run.trace_path]
    sections = [
        "# FastWAM Scheduler / Sampler Report",
        "",
        "This report summarizes measured FastWAM scheduler/sampler sweep artifacts. "
        "It does not claim `parity_verified`; parity requires repeated trials and "
        "statistical evidence beyond this table.",
        "",
        "## Per-Workload Conclusions",
        "",
        _conclusion_table(runs),
        "",
        "## Config-Level Decisions",
        "",
        _config_summary_table(summaries),
        "",
        "## Baseline",
        "",
        _markdown_table(baselines),
        "",
        "## Pareto Frontier",
        "",
        _markdown_table(pareto),
        "",
        "## Repeat Summary",
        "",
        _repeat_summary_table(runs),
        "",
        "## Recommended Candidates",
        "",
        _markdown_table(recommended),
        "",
        "## Not Recommended",
        "",
        _markdown_table(not_recommended),
        "",
        "## Artifact Completeness",
        "",
        f"- Rows: {len(runs)}",
        f"- Missing candidate id rows: {len(missing_candidate_ids)}",
        f"- Missing job id rows: {len(missing_job_ids)}",
        f"- Missing trace path rows: {len(missing_trace_paths)}",
        "",
        "## Full Table",
        "",
        _markdown_table(runs),
        "",
    ]
    return "\n".join(sections)


def _conclusion_table(runs: list[SchedulerRun]) -> str:
    headers = [
        "model",
        "workload",
        "baseline",
        "best_recommended",
        "fastest_pareto",
        "fastest_pareto_speedup",
        "conclusion",
        "needs_more_repeats",
    ]
    if not runs:
        return "_No rows._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for group in _runs_by_key(runs).values():
        baseline = next((run for run in group if run.recommendation == "baseline"), None)
        recommended = _best_speedup(
            run for run in group if run.recommendation == "recommended_candidate"
        )
        experimental = _best_speedup(run for run in group if run.recommendation == "experimental")
        fastest_pareto = _best_speedup(run for run in group if run.pareto)
        conclusion = _group_conclusion(recommended=recommended, experimental=experimental)
        first = group[0]
        values = [
            first.model_id,
            first.workload or "",
            baseline.config_key if baseline is not None else "",
            recommended.config_key if recommended is not None else "",
            fastest_pareto.config_key if fastest_pareto is not None else "",
            _format_optional(fastest_pareto.speedup if fastest_pareto is not None else None),
            conclusion,
            _needs_more_repeats(group),
        ]
        lines.append("| " + " | ".join(_escape_markdown(value) for value in values) + " |")
    return "\n".join(lines)


def _runs_by_key(runs: list[SchedulerRun]) -> dict[str, list[SchedulerRun]]:
    groups: dict[str, list[SchedulerRun]] = {}
    for run in runs:
        groups.setdefault(run.run_key, []).append(run)
    return dict(sorted(groups.items()))


def _best_speedup(runs: Iterable[SchedulerRun]) -> SchedulerRun | None:
    candidates = [run for run in runs if run.speedup is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda run: (run.speedup or 0.0, run.success_rate or -1.0))


def _group_conclusion(
    *,
    recommended: SchedulerRun | None,
    experimental: SchedulerRun | None,
) -> str:
    if recommended is not None:
        return "candidate_recommended_for_repeat"
    if experimental is not None:
        return "faster_candidate_needs_review"
    return "no_candidate_recommended"


def _needs_more_repeats(runs: list[SchedulerRun]) -> str:
    selected = [
        run
        for run in runs
        if run.recommendation == "baseline" or run.pareto or run.recommendation == "recommended_candidate"
    ]
    if not selected:
        return "yes"
    counts: dict[str, int] = {}
    for run in selected:
        counts[run.config_key] = counts.get(run.config_key, 0) + 1
    return "no" if counts and min(counts.values()) >= 3 else "yes"


def _repeat_summary_table(runs: list[SchedulerRun]) -> str:
    groups = _config_groups(runs)
    headers = [
        "model",
        "workload",
        "solver",
        "config",
        "repeats",
        "total_ms_mean",
        "total_ms_std",
        "denoise_ms_mean",
        "success_rate_mean",
        "speedup_mean",
        "speedup_std",
        "action_drift_mean",
        "job_ids",
        "candidate_ids",
        "trace_paths",
    ]
    if not groups:
        return "_No rows._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for (model_id, workload, solver, config_key), group in sorted(groups.items()):
        values = [
            model_id,
            workload,
            solver,
            config_key,
            len(group),
            _format_optional(_mean_field(group, "total_ms")),
            _format_optional(_std_field(group, "total_ms")),
            _format_optional(_mean_field(group, "denoise_wall_ms")),
            _format_optional(_mean_field(group, "success_rate")),
            _format_optional(_mean_field(group, "speedup")),
            _format_optional(_std_field(group, "speedup")),
            _format_optional(_mean_field(group, "action_drift")),
            ",".join(run.job_id or "" for run in group),
            ",".join(run.candidate_id or "" for run in group),
            "<br>".join(str(run.trace_path) for run in group),
        ]
        lines.append("| " + " | ".join(_escape_markdown(value) for value in values) + " |")
    return "\n".join(lines)


def _config_summary_table(summaries: list[SchedulerConfigSummary]) -> str:
    headers = [
        "model",
        "workload",
        "solver",
        "config",
        "repeats",
        "total_ms_mean",
        "total_ms_std",
        "denoise_ms_mean",
        "success_rate_mean",
        "success_delta_mean",
        "speedup_mean",
        "speedup_std",
        "action_drift_mean",
        "failure_count",
        "aggregate_pareto",
        "aggregate_recommendation",
        "aggregate_reason",
        "candidate_ids",
        "job_ids",
        "trace_paths",
    ]
    if not summaries:
        return "_No rows._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for summary in summaries:
        values = [
            summary.model_id,
            summary.workload or "",
            summary.solver or "",
            summary.config_key,
            summary.repeats,
            _format_optional(summary.total_ms_mean),
            _format_optional(summary.total_ms_std),
            _format_optional(summary.denoise_wall_ms_mean),
            _format_optional(summary.success_rate_mean),
            _format_optional(summary.success_delta_mean),
            _format_optional(summary.speedup_mean),
            _format_optional(summary.speedup_std),
            _format_optional(summary.action_drift_mean),
            summary.failure_count,
            "yes" if summary.aggregate_pareto else "",
            summary.aggregate_recommendation,
            summary.aggregate_reason,
            summary.candidate_ids,
            summary.job_ids,
            summary.trace_paths,
        ]
        lines.append("| " + " | ".join(_escape_markdown(value) for value in values) + " |")
    return "\n".join(lines)


def _mean_field(runs: list[SchedulerRun], field: str) -> float | None:
    values = _numeric_field_values(runs, field)
    return mean(values) if values else None


def _std_field(runs: list[SchedulerRun], field: str) -> float | None:
    values = _numeric_field_values(runs, field)
    if len(values) < 2:
        return None
    average = mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _numeric_field_values(runs: list[SchedulerRun], field: str) -> list[float]:
    values = []
    for run in runs:
        value = _field_float(run, field)
        if value is not None:
            values.append(value)
    return values


def _markdown_table(runs: list[SchedulerRun]) -> str:
    headers = [
        "model",
        "candidate_id",
        "workload",
        "solver",
        "schedule_source",
        "steps",
        "sigma_shift",
        "total_ms",
        "denoise_wall_ms",
        "speedup",
        "success_rate",
        "success_delta",
        "action_drift",
        "action_drift_fields",
        "timestep_count",
        "timesteps",
        "sigmas",
        "deltas",
        "pareto",
        "failure_count",
        "failure_cases",
        "recommendation",
        "reason",
        "fallback_reason",
        "job_id",
        "trace_path",
        "command",
    ]
    if not runs:
        return "_No rows._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for run in runs:
        values = [
            run.model_id,
            run.candidate_id or "",
            run.workload or "",
            run.solver or "",
            run.schedule_source or "",
            _format_optional(run.num_inference_steps),
            _format_optional(run.sigma_shift),
            _format_optional(run.total_ms),
            _format_optional(run.denoise_wall_ms),
            _format_optional(run.speedup),
            _format_optional(run.success_rate),
            _format_optional(run.success_delta),
            _format_optional(run.action_drift),
            run.action_drift_fields or "",
            _format_optional(run.timestep_count),
            run.timesteps or "",
            run.sigmas or "",
            run.deltas or "",
            "yes" if run.pareto else "",
            _format_optional(run.failure_count),
            run.failure_cases or "",
            run.recommendation,
            run.recommendation_reason,
            run.fallback_reason or "",
            run.job_id or "",
            str(run.trace_path),
            run.command,
        ]
        lines.append("| " + " | ".join(_escape_markdown(value) for value in values) + " |")
    return "\n".join(lines)


def _escape_markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _render_table_row(run: SchedulerRun) -> str:
    cells = [
        run.candidate_id or "",
        run.model_id,
        run.workload or "",
        run.solver or "",
        run.schedule_source or "",
        _format_optional(run.num_inference_steps),
        _format_optional(run.sigma_shift),
        _format_optional(run.total_ms),
        _format_optional(run.timestep_count),
        _format_optional(run.denoise_wall_ms),
        _format_optional(run.speedup),
        _format_optional(run.success_rate),
        _format_optional(run.success_delta),
        _format_optional(run.action_drift),
        run.action_drift_fields or "",
        _format_optional(run.failure_count),
        run.failure_cases or "",
        "yes" if run.pareto else "",
        run.recommendation,
        run.recommendation_reason,
        run.job_id or "",
        str(run.trace_path),
        run.timesteps or "",
        run.sigmas or "",
        run.deltas or "",
        run.command,
        run.fallback_reason or "",
    ]
    return "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in cells) + "</tr>"


def _render_config_table_row(summary: SchedulerConfigSummary) -> str:
    cells = [
        summary.model_id,
        summary.workload or "",
        summary.solver or "",
        summary.config_key,
        summary.repeats,
        _format_optional(summary.total_ms_mean),
        _format_optional(summary.total_ms_std),
        _format_optional(summary.denoise_wall_ms_mean),
        _format_optional(summary.speedup_mean),
        _format_optional(summary.speedup_std),
        _format_optional(summary.success_rate_mean),
        _format_optional(summary.success_delta_mean),
        _format_optional(summary.action_drift_mean),
        summary.failure_count,
        "yes" if summary.aggregate_pareto else "",
        summary.aggregate_recommendation,
        summary.aggregate_reason,
        summary.candidate_ids,
        summary.job_ids,
    ]
    return "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in cells) + "</tr>"


def _bar_chart(
    runs: list[SchedulerRun],
    field: str,
    title: str,
    *,
    lower_is_better: bool,
) -> str:
    values = [(run.config_key, _field_float(run, field)) for run in runs]
    values = [(label, value) for label, value in values if value is not None]
    if not values:
        return f"<section><h2>{html.escape(title)}</h2><svg><text x='20' y='40'>No data</text></svg></section>"
    max_value = max(value for _, value in values) or 1.0
    width = 640
    height = 240
    bar_width = max(8, int((width - 80) / max(1, len(values))))
    bars = []
    for index, (label, value) in enumerate(values):
        bar_height = int((height - 70) * value / max_value)
        x = 50 + index * bar_width
        y = height - 35 - bar_height
        color = "#2563eb" if lower_is_better else "#059669"
        bars.append(
            f"<rect x='{x}' y='{y}' width='{bar_width - 3}' height='{bar_height}' fill='{color}'/>"
            f"<title>{html.escape(label)}: {_format_optional(value)}</title>"
        )
    return (
        f"<section><h2>{html.escape(title)}</h2><svg viewBox='0 0 {width} {height}'>"
        f"<text x='20' y='24'>{html.escape(title)}</text>{''.join(bars)}</svg></section>"
    )


def _scatter_chart(runs: list[SchedulerRun], x_field: str, y_field: str, title: str) -> str:
    points = [
        (run, _field_float(run, x_field), _field_float(run, y_field))
        for run in runs
    ]
    points = [(run, x, y) for run, x, y in points if x is not None and y is not None]
    if not points:
        return f"<section><h2>{html.escape(title)}</h2><svg><text x='20' y='40'>No data</text></svg></section>"
    width = 640
    height = 240
    xs = [x for _, x, _ in points]
    ys = [y for _, _, y in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    circles = []
    for run, x_value, y_value in points:
        x = 45 + int((width - 90) * (x_value - min_x) / span_x)
        y = height - 35 - int((height - 75) * (y_value - min_y) / span_y)
        color = "#dc2626" if run.pareto else "#6b7280"
        circles.append(
            f"<circle cx='{x}' cy='{y}' r='5' fill='{color}'/>"
            f"<title>{html.escape(run.config_key)} x={x_value:.4g} y={y_value:.4g}</title>"
        )
    return (
        f"<section><h2>{html.escape(title)}</h2><svg viewBox='0 0 {width} {height}'>"
        f"<text x='20' y='24'>{html.escape(title)}</text>{''.join(circles)}</svg></section>"
    )


def _field_float(run: SchedulerRun, field: str) -> float | None:
    value = getattr(run, field)
    return value if isinstance(value, float) else None


def _run_key(model_id: str, workload: str | None) -> str:
    return f"{model_id}:{workload or ''}"


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def _requested_sigma_shift(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text.lower() in {"", "none"}:
        return None
    if text.lower() == "null":
        return "null"
    number = _optional_float(value)
    if number is None:
        return text
    return f"{number:.4g}"


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value)
    if text.lower() in {"", "none", "null"}:
        return None
    return int(float(text))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value)
    if text.lower() in {"", "none", "null"}:
        return None
    return float(text)


def _format_optional(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
