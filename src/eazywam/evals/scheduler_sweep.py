from __future__ import annotations

import argparse
import csv
import json
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SUPPORTED_MODEL_IDS = ("fastwam-libero", "fastwam-robotwin")


@dataclass(frozen=True)
class SweepCandidate:
    candidate_id: str
    model_id: str
    workload: str
    num_inference_steps: int
    sigma_shift: str
    phase: str
    command: str
    summary_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "model_id": self.model_id,
            "workload": self.workload,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "phase": self.phase,
            "command": self.command,
            "summary_path": self.summary_path,
        }


def build_coarse_sweep(
    *,
    trace_dir: str,
    baseline_steps: int = 10,
    model_ids: Sequence[str] | None = None,
    step_counts: Sequence[int] | None = None,
    sigma_shifts: Sequence[str] | None = None,
    cache_dir: str | None = None,
    upstream_dir: str | None = None,
    eval_sets: Sequence[str] | None = None,
    libero_sets: Sequence[str] | None = None,
    robotwin_sets: Sequence[str] | None = None,
    libero_task_id: int = 0,
    libero_num_trials: int = 1,
    robotwin_task_name: str = "click_alarmclock",
    robotwin_num_episodes: int = 1,
) -> list[SweepCandidate]:
    steps = _coarse_steps(baseline_steps, step_counts=step_counts)
    normalized_sigma_shifts = _coarse_sigma_shifts(sigma_shifts)
    candidates = []
    for model_id in _selected_model_ids(model_ids):
        workload = _workload_for_model(model_id)
        for step_count in steps:
            for sigma_shift in normalized_sigma_shifts:
                phase = _phase(step_count, baseline_steps)
                candidate_id = _candidate_id(model_id, step_count, sigma_shift)
                summary_path = (
                    f"{trace_dir}/summaries/{candidate_id}-summary.json"
                )
                command = _command(
                    model_id=model_id,
                    trace_dir=trace_dir,
                    summary_path=summary_path,
                    step_count=step_count,
                    sigma_shift=sigma_shift,
                    cache_dir=cache_dir,
                    upstream_dir=upstream_dir,
                    eval_sets=eval_sets,
                    libero_sets=libero_sets,
                    robotwin_sets=robotwin_sets,
                    libero_task_id=libero_task_id,
                    libero_num_trials=libero_num_trials,
                    robotwin_task_name=robotwin_task_name,
                    robotwin_num_episodes=robotwin_num_episodes,
                )
                candidates.append(
                    SweepCandidate(
                        candidate_id=candidate_id,
                        model_id=model_id,
                        workload=workload,
                        num_inference_steps=step_count,
                        sigma_shift=sigma_shift,
                        phase=phase,
                        command=command,
                        summary_path=summary_path,
                    )
                )
    return candidates


def build_refine_sweep(
    *,
    report_json: str | Path,
    trace_dir: str,
    baseline_steps: int = 10,
    model_ids: Sequence[str] | None = None,
    cache_dir: str | None = None,
    upstream_dir: str | None = None,
    eval_sets: Sequence[str] | None = None,
    libero_sets: Sequence[str] | None = None,
    robotwin_sets: Sequence[str] | None = None,
    libero_task_id: int = 0,
    libero_num_trials: int = 1,
    robotwin_task_name: str = "click_alarmclock",
    robotwin_num_episodes: int = 1,
) -> list[SweepCandidate]:
    rows = _read_report_rows(report_json)
    selected_model_ids = _selected_model_ids(model_ids)
    rows = [row for row in rows if str(row.get("model_id") or "") in selected_model_ids]
    selected = [
        row
        for row in rows
        if row.get("pareto") is True
        or row.get("recommendation") == "recommended_candidate"
    ]
    model_ids = sorted({str(row["model_id"]) for row in rows if row.get("model_id")})
    candidates_by_key: dict[tuple[str, int, str], SweepCandidate] = {}
    for model_id in model_ids:
        _add_candidate(
            candidates_by_key,
            model_id=model_id,
            trace_dir=trace_dir,
            baseline_steps=baseline_steps,
            step_count=baseline_steps,
            sigma_shift="null",
            phase="baseline",
            cache_dir=cache_dir,
            upstream_dir=upstream_dir,
            eval_sets=eval_sets,
            libero_sets=libero_sets,
            robotwin_sets=robotwin_sets,
            libero_task_id=libero_task_id,
            libero_num_trials=libero_num_trials,
            robotwin_task_name=robotwin_task_name,
            robotwin_num_episodes=robotwin_num_episodes,
        )

    for row in selected:
        model_id = str(row["model_id"])
        step_count = _positive_int(row.get("num_inference_steps"))
        if step_count is None:
            continue
        for refined_step in _neighbor_steps(step_count, baseline_steps):
            for refined_shift in _neighbor_sigma_shifts(row, rows):
                _add_candidate(
                    candidates_by_key,
                    model_id=model_id,
                    trace_dir=trace_dir,
                    baseline_steps=baseline_steps,
                    step_count=refined_step,
                    sigma_shift=refined_shift,
                    phase=_refine_phase(refined_step, baseline_steps),
                    cache_dir=cache_dir,
                    upstream_dir=upstream_dir,
                    eval_sets=eval_sets,
                    libero_sets=libero_sets,
                    robotwin_sets=robotwin_sets,
                    libero_task_id=libero_task_id,
                    libero_num_trials=libero_num_trials,
                    robotwin_task_name=robotwin_task_name,
                    robotwin_num_episodes=robotwin_num_episodes,
                )
    return sorted(
        candidates_by_key.values(),
        key=lambda item: (item.model_id, item.num_inference_steps, item.sigma_shift),
    )


def build_confirm_sweep(
    *,
    report_json: str | Path,
    trace_dir: str,
    baseline_steps: int = 10,
    model_ids: Sequence[str] | None = None,
    repeats: int = 3,
    cache_dir: str | None = None,
    upstream_dir: str | None = None,
    eval_sets: Sequence[str] | None = None,
    libero_sets: Sequence[str] | None = None,
    robotwin_sets: Sequence[str] | None = None,
    libero_task_id: int = 0,
    libero_num_trials: int = 1,
    robotwin_task_name: str = "click_alarmclock",
    robotwin_num_episodes: int = 1,
) -> list[SweepCandidate]:
    rows = _read_report_rows(report_json)
    selected_model_ids = _selected_model_ids(model_ids)
    rows = [row for row in rows if str(row.get("model_id") or "") in selected_model_ids]
    configs = _confirm_configs(rows, baseline_steps=baseline_steps)
    candidates = []
    for repeat_index in range(1, repeats + 1):
        for model_id, step_count, sigma_shift, phase in configs:
            base_id = _candidate_id(model_id, step_count, sigma_shift)
            candidate_id = f"{base_id}-confirm{repeat_index}"
            summary_path = f"{trace_dir}/summaries/{candidate_id}-summary.json"
            candidates.append(
                SweepCandidate(
                    candidate_id=candidate_id,
                    model_id=model_id,
                    workload=_workload_for_model(model_id),
                    num_inference_steps=step_count,
                    sigma_shift=sigma_shift,
                    phase=phase,
                    command=_command(
                        model_id=model_id,
                        trace_dir=trace_dir,
                        summary_path=summary_path,
                        step_count=step_count,
                        sigma_shift=sigma_shift,
                        cache_dir=cache_dir,
                        upstream_dir=upstream_dir,
                        eval_sets=eval_sets,
                        libero_sets=libero_sets,
                        robotwin_sets=robotwin_sets,
                        libero_task_id=libero_task_id,
                        libero_num_trials=libero_num_trials,
                        robotwin_task_name=robotwin_task_name,
                        robotwin_num_episodes=robotwin_num_episodes,
                    ),
                    summary_path=summary_path,
                )
            )
    return candidates


def write_sweep_manifest(candidates: list[SweepCandidate], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "scheduler_sweep_candidates.json"
    csv_path = out / "scheduler_sweep_candidates.csv"
    shell_path = out / "scheduler_sweep_commands.sh"
    job_map_path = out / "scheduler_sweep_job_map.csv"
    report_shell_path = out / "scheduler_report_command.sh"
    json_path.write_text(
        json.dumps([candidate.to_dict() for candidate in candidates], indent=2) + "\n",
        encoding="utf-8",
    )
    fields = list(candidates[0].to_dict()) if candidates else list(SweepCandidate.__dataclass_fields__)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(candidate.to_dict())
    with job_map_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "summary_path", "job_id"])
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "candidate_id": candidate.candidate_id,
                    "summary_path": candidate.summary_path,
                    "job_id": "",
                }
            )
    shell_path.write_text(
        _render_command_script(candidates)
        + "\n".join(candidate.command for candidate in candidates)
        + "\n",
        encoding="utf-8",
    )
    report_shell_path.write_text(_render_report_script(candidates, job_map_path), encoding="utf-8")
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "shell": str(shell_path),
        "job_map": str(job_map_path),
        "report_shell": str(report_shell_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate FastWAM scheduler sweep commands")
    parser.add_argument("--stage", choices=("coarse", "refine", "confirm"), default="coarse")
    parser.add_argument("--report-json", default=None, help="scheduler_results.json for refine/confirm")
    parser.add_argument("--trace-dir", required=True, help="SuperPod trace/output directory")
    parser.add_argument("--output-dir", required=True, help="Directory for manifest files")
    parser.add_argument("--baseline-steps", type=int, default=10)
    parser.add_argument(
        "--model-ids",
        default=None,
        help=(
            "Comma-separated model ids to include. Defaults to fastwam-libero "
            "and fastwam-robotwin."
        ),
    )
    parser.add_argument(
        "--step-counts",
        default=None,
        help=(
            "Comma-separated coarse step counts. The baseline step count is added "
            "automatically; the final set must include faster and quality-reference counts."
        ),
    )
    parser.add_argument(
        "--sigma-shifts",
        default=None,
        help="Comma-separated coarse sigma shifts. Use null for the default shift.",
    )
    parser.add_argument("--cache-dir", default=None, help="Optional cache dir for generated wam eval commands")
    parser.add_argument("--upstream-dir", default=None, help="Optional upstream dir for generated wam eval commands")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra eval override added to every generated wam eval command",
    )
    parser.add_argument(
        "--libero-set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra eval override added only to fastwam-libero commands",
    )
    parser.add_argument(
        "--robotwin-set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra eval override added only to fastwam-robotwin commands",
    )
    parser.add_argument("--confirm-repeats", type=int, default=3)
    parser.add_argument("--libero-task-id", type=int, default=0)
    parser.add_argument("--libero-num-trials", type=int, default=1)
    parser.add_argument("--robotwin-task-name", default="click_alarmclock")
    parser.add_argument("--robotwin-num-episodes", type=int, default=1)
    args = parser.parse_args(argv)

    if args.stage in {"refine", "confirm"}:
        if args.report_json is None:
            parser.error(f"--stage {args.stage} requires --report-json")
    if args.confirm_repeats <= 0:
        parser.error("--confirm-repeats must be positive")
    for flag, values in (
        ("--set", args.set),
        ("--libero-set", args.libero_set),
        ("--robotwin-set", args.robotwin_set),
    ):
        try:
            _validate_extra_sets(values, flag)
        except ValueError as exc:
            parser.error(str(exc))

    try:
        if args.stage == "refine":
            candidates = build_refine_sweep(
                report_json=args.report_json,
                trace_dir=args.trace_dir,
                baseline_steps=args.baseline_steps,
                model_ids=_parse_model_ids(args.model_ids),
                cache_dir=args.cache_dir,
                upstream_dir=args.upstream_dir,
                eval_sets=args.set,
                libero_sets=args.libero_set,
                robotwin_sets=args.robotwin_set,
                libero_task_id=args.libero_task_id,
                libero_num_trials=args.libero_num_trials,
                robotwin_task_name=args.robotwin_task_name,
                robotwin_num_episodes=args.robotwin_num_episodes,
            )
        elif args.stage == "confirm":
            candidates = build_confirm_sweep(
                report_json=args.report_json,
                trace_dir=args.trace_dir,
                baseline_steps=args.baseline_steps,
                model_ids=_parse_model_ids(args.model_ids),
                repeats=args.confirm_repeats,
                cache_dir=args.cache_dir,
                upstream_dir=args.upstream_dir,
                eval_sets=args.set,
                libero_sets=args.libero_set,
                robotwin_sets=args.robotwin_set,
                libero_task_id=args.libero_task_id,
                libero_num_trials=args.libero_num_trials,
                robotwin_task_name=args.robotwin_task_name,
                robotwin_num_episodes=args.robotwin_num_episodes,
            )
        else:
            candidates = build_coarse_sweep(
                trace_dir=args.trace_dir,
                baseline_steps=args.baseline_steps,
                model_ids=_parse_model_ids(args.model_ids),
                step_counts=_parse_step_counts(args.step_counts),
                sigma_shifts=_parse_sigma_shifts(args.sigma_shifts),
                cache_dir=args.cache_dir,
                upstream_dir=args.upstream_dir,
                eval_sets=args.set,
                libero_sets=args.libero_set,
                robotwin_sets=args.robotwin_set,
                libero_task_id=args.libero_task_id,
                libero_num_trials=args.libero_num_trials,
                robotwin_task_name=args.robotwin_task_name,
                robotwin_num_episodes=args.robotwin_num_episodes,
            )
    except ValueError as exc:
        parser.error(str(exc))
    outputs = write_sweep_manifest(candidates, args.output_dir)
    print(json.dumps({"candidates": len(candidates), "outputs": outputs}, indent=2))
    return 0


def _coarse_steps(
    baseline_steps: int,
    *,
    step_counts: Sequence[int] | None = None,
) -> list[int]:
    if baseline_steps <= 0:
        raise ValueError("baseline_steps must be positive")
    if step_counts is None:
        faster = {
            max(1, round(baseline_steps * factor))
            for factor in (0.4, 0.5, 0.6, 0.8)
        }
        references = {baseline_steps, round(baseline_steps * 1.2), round(baseline_steps * 1.6)}
        steps = faster | references
    else:
        steps = set(step_counts) | {baseline_steps}
    if not all(step > 0 for step in steps):
        raise ValueError("coarse step counts must be positive")
    if not any(step < baseline_steps for step in steps):
        raise ValueError("coarse step counts must include a faster-than-baseline candidate")
    if not any(step > baseline_steps for step in steps):
        raise ValueError("coarse step counts must include a quality-reference candidate")
    return sorted(steps)


def _coarse_sigma_shifts(sigma_shifts: Sequence[str] | None = None) -> list[str]:
    values = list(sigma_shifts) if sigma_shifts is not None else ["null", "3.0", "5.0", "7.0"]
    normalized = {_parse_sigma_shift(value) for value in values}
    normalized.add("null")
    return sorted(normalized, key=lambda item: (-1.0 if item == "null" else float(item)))


def _parse_step_counts(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("--step-counts must contain at least one integer")
    return [_positive_int_required(part, "--step-counts") for part in parts]


def _parse_sigma_shifts(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("--sigma-shifts must contain at least one value")
    return [_parse_sigma_shift(part) for part in parts]


def _parse_model_ids(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("--model-ids must contain at least one model id")
    return parts


def _selected_model_ids(model_ids: Sequence[str] | None) -> tuple[str, ...]:
    if model_ids is None:
        return _SUPPORTED_MODEL_IDS
    selected = tuple(dict.fromkeys(model_ids))
    unknown = [model_id for model_id in selected if model_id not in _SUPPORTED_MODEL_IDS]
    if unknown:
        raise ValueError(f"unsupported --model-ids value: {unknown[0]}")
    return selected


def _positive_int_required(value: object, flag: str) -> int:
    try:
        parsed = int(str(value))
    except ValueError as exc:
        raise ValueError(f"{flag} contains a non-integer value: {value}") from exc
    if parsed <= 0:
        raise ValueError(f"{flag} values must be positive")
    return parsed


def _parse_sigma_shift(value: object) -> str:
    parsed = _optional_float(value)
    if parsed is None:
        return "null"
    if parsed <= 0:
        raise ValueError("--sigma-shifts values must be positive or null")
    return _format_float(parsed)


def _validate_extra_sets(values: Sequence[str], flag: str) -> None:
    for value in values:
        if "=" not in value or not value.split("=", 1)[0]:
            raise ValueError(f"{flag} values must use KEY=VALUE syntax")


def _phase(step_count: int, baseline_steps: int) -> str:
    if step_count < baseline_steps:
        return "faster_candidate"
    if step_count == baseline_steps:
        return "baseline"
    return "quality_reference"


def _refine_phase(step_count: int, baseline_steps: int) -> str:
    if step_count == baseline_steps:
        return "refine_baseline_region"
    if step_count < baseline_steps:
        return "refine_faster_candidate"
    return "refine_quality_reference"


def _candidate_id(model_id: str, step_count: int, sigma_shift: str) -> str:
    normalized_shift = sigma_shift.replace(".", "p")
    return f"{model_id}-steps{step_count}-shift{normalized_shift}"


def _workload_for_model(model_id: str) -> str:
    return "libero-single-task" if model_id == "fastwam-libero" else "robotwin-single-task"


def _add_candidate(
    candidates: dict[tuple[str, int, str], SweepCandidate],
    *,
    model_id: str,
    trace_dir: str,
    baseline_steps: int,
    step_count: int,
    sigma_shift: str,
    phase: str,
    cache_dir: str | None,
    upstream_dir: str | None,
    eval_sets: Sequence[str] | None,
    libero_sets: Sequence[str] | None,
    robotwin_sets: Sequence[str] | None,
    libero_task_id: int,
    libero_num_trials: int,
    robotwin_task_name: str,
    robotwin_num_episodes: int,
) -> None:
    key = (model_id, step_count, sigma_shift)
    if key in candidates:
        return
    candidate_id = _candidate_id(model_id, step_count, sigma_shift)
    summary_path = f"{trace_dir}/summaries/{candidate_id}-summary.json"
    candidates[key] = SweepCandidate(
        candidate_id=candidate_id,
        model_id=model_id,
        workload=_workload_for_model(model_id),
        num_inference_steps=step_count,
        sigma_shift=sigma_shift,
        phase=phase if not (step_count == baseline_steps and sigma_shift == "null") else "baseline",
        command=_command(
            model_id=model_id,
            trace_dir=trace_dir,
            summary_path=summary_path,
            step_count=step_count,
            sigma_shift=sigma_shift,
            cache_dir=cache_dir,
            upstream_dir=upstream_dir,
            eval_sets=eval_sets,
            libero_sets=libero_sets,
            robotwin_sets=robotwin_sets,
            libero_task_id=libero_task_id,
            libero_num_trials=libero_num_trials,
            robotwin_task_name=robotwin_task_name,
            robotwin_num_episodes=robotwin_num_episodes,
        ),
        summary_path=summary_path,
    )


def _read_report_rows(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("scheduler report JSON must contain a list of rows")
    return [row for row in data if isinstance(row, dict)]


def _confirm_configs(
    rows: list[dict[str, Any]],
    *,
    baseline_steps: int,
) -> list[tuple[str, int, str, str]]:
    configs: set[tuple[str, int, str, str]] = set()
    model_ids = sorted({str(row["model_id"]) for row in rows if row.get("model_id")})
    for model_id in model_ids:
        configs.add((model_id, baseline_steps, "null", "confirm_baseline"))
    for row in rows:
        if row.get("pareto") is not True:
            continue
        model_id = str(row.get("model_id") or "")
        step_count = _positive_int(row.get("num_inference_steps"))
        if not model_id or step_count is None:
            continue
        sigma_shift = _report_sigma_shift(row.get("sigma_shift"))
        if step_count == baseline_steps and sigma_shift == "null":
            continue
        configs.add((model_id, step_count, sigma_shift, "confirm_pareto"))
    return sorted(configs, key=lambda item: (item[0], item[3], item[1], item[2]))


def _report_sigma_shift(value: object) -> str:
    parsed = _optional_float(value)
    if parsed is None:
        return "null"
    return _format_float(parsed)


def _neighbor_steps(step_count: int, baseline_steps: int) -> list[int]:
    neighbors = {
        max(1, step_count - 1),
        step_count,
        step_count + 1,
    }
    if step_count < baseline_steps:
        neighbors.add(max(1, step_count - 2))
    if step_count >= baseline_steps:
        neighbors.add(step_count + 2)
    return sorted(neighbors)


def _neighbor_sigma_shifts(row: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    current = _optional_float(row.get("sigma_shift"))
    observed = sorted(
        {
            value
            for value in (_optional_float(item.get("sigma_shift")) for item in rows)
            if value is not None
        }
    )
    if current is None:
        shifts = ["null"]
        if observed:
            shifts.append(_format_float(observed[0]))
        return shifts
    positive_gaps = [
        right - left
        for left, right in zip(observed, observed[1:])
        if right > left
    ]
    step = min(positive_gaps) / 2 if positive_gaps else 1.0
    values = {
        max(0.1, current - step),
        current,
        current + step,
    }
    return [_format_float(value) for value in sorted(values)]


def _positive_int(value: object) -> int | None:
    if value is None:
        return None
    parsed = int(float(str(value)))
    return parsed if parsed > 0 else None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value)
    if text.lower() in {"", "none", "null"}:
        return None
    return float(text)


def _format_float(value: float) -> str:
    return f"{value:.4g}"


def _command(
    *,
    model_id: str,
    trace_dir: str,
    summary_path: str,
    step_count: int,
    sigma_shift: str,
    cache_dir: str | None,
    upstream_dir: str | None,
    eval_sets: Sequence[str] | None,
    libero_sets: Sequence[str] | None,
    robotwin_sets: Sequence[str] | None,
    libero_task_id: int,
    libero_num_trials: int,
    robotwin_task_name: str,
    robotwin_num_episodes: int,
) -> str:
    argv = [
        "wam",
        "eval",
        model_id,
        "--opt",
        "scheduler",
        "--trace-dir",
        f"{trace_dir}/{model_id}",
        "--summary-path",
        summary_path,
        "--set",
        f"num_inference_steps={step_count}",
        "--set",
        f"sigma_shift={sigma_shift}",
    ]
    if cache_dir is not None:
        argv.extend(["--cache-dir", cache_dir])
    if upstream_dir is not None:
        argv.extend(["--upstream-dir", upstream_dir])
    for value in eval_sets or ():
        argv.extend(["--set", value])
    if model_id == "fastwam-libero":
        for value in libero_sets or ():
            argv.extend(["--set", value])
        argv.extend(
            [
                "--set",
                f"task_id={libero_task_id}",
                "--set",
                f"num_trials={libero_num_trials}",
            ]
        )
    else:
        for value in robotwin_sets or ():
            argv.extend(["--set", value])
        argv.extend(
            [
                "--set",
                f"task_name={robotwin_task_name}",
                "--set",
                f"num_episodes={robotwin_num_episodes}",
            ]
        )
    return " ".join(shlex.quote(item) for item in argv)


def _render_command_script(candidates: list[SweepCandidate]) -> str:
    trace_dirs = sorted(
        {
            str(Path(candidate.summary_path).parent)
            for candidate in candidates
        }
        | {
            f"{_trace_root(candidate)}/{candidate.model_id}"
            for candidate in candidates
        }
    )
    mkdirs = " ".join(shlex.quote(path) for path in trace_dirs)
    return f"#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p {mkdirs}\n"


def _render_report_script(candidates: list[SweepCandidate], job_map_path: Path) -> str:
    if not candidates:
        return "#!/usr/bin/env bash\nset -euo pipefail\n"
    trace_root = _trace_root(candidates[0])
    summary_paths = " ".join(shlex.quote(candidate.summary_path) for candidate in candidates)
    acceptance_args = _acceptance_args(candidates)
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"mkdir -p {shlex.quote(f'{trace_root}/report')}\n"
        "python -m eazywam.evals.scheduler_audit "
        f"{shlex.quote(str(job_map_path.parent))}\n"
        "python -m eazywam.evals.scheduler_report "
        f"--output-dir {shlex.quote(f'{trace_root}/report')} "
        f"--job-map {shlex.quote(str(job_map_path))} "
        f"{summary_paths}\n"
        "python -m eazywam.evals.scheduler_acceptance "
        f"{shlex.quote(f'{trace_root}/report/scheduler_results.json')}"
        f"{acceptance_args}\n"
        "python -m eazywam.evals.scheduler_bundle "
        f"{shlex.quote(f'{trace_root}/report/scheduler_results.json')} "
        f"--output-dir {shlex.quote(f'{trace_root}/report')} "
        f"--manifest-dir {shlex.quote(str(job_map_path.parent))}"
        f"{acceptance_args}\n"
    )


def _acceptance_args(candidates: list[SweepCandidate]) -> str:
    if not candidates:
        return ""
    if any(candidate.phase.startswith("refine_") for candidate in candidates):
        return " --allow-missing-quality-reference"
    if not all(candidate.phase.startswith("confirm_") for candidate in candidates):
        return ""
    counts: dict[tuple[str, str, int, str], int] = {}
    for candidate in candidates:
        key = (
            candidate.model_id,
            candidate.workload,
            candidate.num_inference_steps,
            candidate.sigma_shift,
        )
        counts[key] = counts.get(key, 0) + 1
    min_repeats = min(counts.values())
    return f" --allow-missing-quality-reference --min-config-repeats {min_repeats}"


def _trace_root(candidate: SweepCandidate) -> str:
    summary_parent = Path(candidate.summary_path).parent
    if summary_parent.name == "summaries":
        return str(summary_parent.parent)
    return str(summary_parent)


if __name__ == "__main__":
    raise SystemExit(main())
