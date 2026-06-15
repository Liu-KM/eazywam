#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/fastwam-teacache-l1-superpod.sh [options]

Purpose:
  Generate or run the scheduler-agnostic FastWAM TeaCache L1 SuperPod
  baseline/candidate experiment commands. The default mode prints commands
  only; pass --execute inside a prepared GPU environment to run them.

Options:
  --execute                Run commands instead of printing them.
  --cache-dir PATH         WAM cache path. Default: ${WAM_CACHE_DIR:-unset}
  --upstream-dir PATH      FastWAM checkout path. Default: unset.
  --robotwin-root PATH     RoboTwin root path with assets. Default: unset.
  --trace-root PATH        Trace root. Default: ${WAM_TRACE_DIR:-runs/fastwam-teacache-l1}
  --report-root PATH       Report root. Default: ${WAM_REPORT_DIR:-runs/fastwam-teacache-l1-reports}
  --run-id ID              Run group id. Default: UTC timestamp.
  --seed N                 Seed for all runs. Default: 42
  --num-inference-steps N  Inference steps for all runs. Default: 20
  --teacache-threshold V   TeaCache drift threshold. Default: 0.05
  --teacache-warmup-steps N TeaCache warmup steps. Default: 1
  --libero-task-id ID      LIBERO task id. Default: 0
  --libero-num-trials N    LIBERO trial count. Default: 5
  --robotwin-task NAME     RoboTwin task name. Default: click_alarmclock
  --robotwin-num-episodes N RoboTwin episode count. Default: 5
  --robotwin-task-config C RoboTwin task config. Default: demo_randomized
  --max-action-drift V     Compare action drift gate. Default: 0.001
  --help                   Show this help.

The script does not submit Slurm/PBS/LSF jobs and does not fabricate results.
Record only real SuperPod or maintainer-approved GPU/simulator numbers in
docs/fastwam_teacache_l1_report.md. In --execute mode it writes
fastwam-teacache-l1-report.md and fastwam-teacache-l1-report.json under the
run-specific report directory.
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

quote_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run_or_print() {
  quote_cmd "$@"
  if [[ "$execute" == "1" ]]; then
    "$@"
  fi
}

latest_trace() {
  local dir="$1"
  local trace
  trace="$(
    find "$dir" -name trace.jsonl -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | head -n 1 \
      | cut -d ' ' -f 2-
  )"
  [[ -n "$trace" ]] || die "no trace.jsonl found under $dir"
  printf '%s\n' "$trace"
}

run_compare() {
  local baseline_dir="$1"
  local variant_dir="$2"
  local output_path="$3"
  if [[ "$execute" == "1" ]]; then
    local baseline_trace
    local variant_trace
    baseline_trace="$(latest_trace "$baseline_dir")"
    variant_trace="$(latest_trace "$variant_dir")"
    quote_cmd wam compare "$baseline_trace" "$variant_trace" --max-action-drift "$max_action_drift"
    wam compare "$baseline_trace" "$variant_trace" --max-action-drift "$max_action_drift" \
      | tee "$output_path"
  else
    printf '+ wam compare %q %q --max-action-drift %q > %q\n' \
      "$baseline_dir/<run>/trace.jsonl" \
      "$variant_dir/<run>/trace.jsonl" \
      "$max_action_drift" \
      "$output_path"
  fi
}

run_report() {
  local markdown_path="$1"
  local json_path="$2"
  if [[ "$execute" == "1" ]]; then
    quote_cmd "$report_helper" --report-root "$report_run_root"
    "$report_helper" --report-root "$report_run_root" | tee "$markdown_path"
    quote_cmd "$report_helper" --report-root "$report_run_root" --json
    "$report_helper" --report-root "$report_run_root" --json | tee "$json_path"
  else
    printf '+ %q --report-root %q > %q\n' \
      "$report_helper" \
      "$report_run_root" \
      "$markdown_path"
    printf '+ %q --report-root %q --json > %q\n' \
      "$report_helper" \
      "$report_run_root" \
      "$json_path"
  fi
}

script_path="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1; then
  resolved_script_path="$(readlink -f "$script_path" 2>/dev/null || true)"
  if [[ -n "$resolved_script_path" ]]; then
    script_path="$resolved_script_path"
  fi
fi
script_dir="$(cd -- "$(dirname -- "$script_path")" && pwd)"
report_helper="$script_dir/fastwam-teacache-l1-report.py"

execute=0
cache_dir="${WAM_CACHE_DIR:-}"
upstream_dir=""
robotwin_root=""
trace_root="${WAM_TRACE_DIR:-runs/fastwam-teacache-l1}"
report_root="${WAM_REPORT_DIR:-runs/fastwam-teacache-l1-reports}"
run_id=""
seed="42"
num_inference_steps="20"
teacache_threshold="0.05"
teacache_warmup_steps="1"
libero_task_id="0"
libero_num_trials="5"
robotwin_task="click_alarmclock"
robotwin_num_episodes="5"
robotwin_task_config="demo_randomized"
max_action_drift="0.001"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      execute=1
      shift
      ;;
    --cache-dir)
      cache_dir="${2:-}"
      shift 2
      ;;
    --upstream-dir)
      upstream_dir="${2:-}"
      shift 2
      ;;
    --robotwin-root)
      robotwin_root="${2:-}"
      shift 2
      ;;
    --trace-root)
      trace_root="${2:-}"
      shift 2
      ;;
    --report-root)
      report_root="${2:-}"
      shift 2
      ;;
    --run-id)
      run_id="${2:-}"
      shift 2
      ;;
    --seed)
      seed="${2:-}"
      shift 2
      ;;
    --num-inference-steps)
      num_inference_steps="${2:-}"
      shift 2
      ;;
    --teacache-threshold)
      teacache_threshold="${2:-}"
      shift 2
      ;;
    --teacache-warmup-steps)
      teacache_warmup_steps="${2:-}"
      shift 2
      ;;
    --libero-task-id)
      libero_task_id="${2:-}"
      shift 2
      ;;
    --libero-num-trials)
      libero_num_trials="${2:-}"
      shift 2
      ;;
    --robotwin-task)
      robotwin_task="${2:-}"
      shift 2
      ;;
    --robotwin-num-episodes)
      robotwin_num_episodes="${2:-}"
      shift 2
      ;;
    --robotwin-task-config)
      robotwin_task_config="${2:-}"
      shift 2
      ;;
    --max-action-drift)
      max_action_drift="${2:-}"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -n "$trace_root" ]] || die "--trace-root must not be empty"
[[ -n "$report_root" ]] || die "--report-root must not be empty"
if [[ -z "$run_id" ]]; then
  run_id="$(date -u +%Y%m%dT%H%M%SZ)"
fi
[[ -n "$run_id" ]] || die "--run-id must not be empty"

cache_args=()
if [[ -n "$cache_dir" ]]; then
  cache_args=(--cache-dir "$cache_dir")
fi
upstream_args=()
if [[ -n "$upstream_dir" ]]; then
  upstream_args=(--upstream-dir "$upstream_dir")
fi
robotwin_root_args=()
if [[ -n "$robotwin_root" ]]; then
  robotwin_root_args=(--set "robotwin_root=$robotwin_root")
fi

trace_run_root="$trace_root/$run_id"
libero_baseline_trace="$trace_run_root/fastwam-libero-eager-cache"
libero_teacache_trace="$trace_run_root/fastwam-libero-teacache"
robotwin_baseline_trace="$trace_run_root/fastwam-robotwin-eager-cache"
robotwin_teacache_trace="$trace_run_root/fastwam-robotwin-teacache"
report_run_root="$report_root/$run_id"
report_markdown="$report_run_root/fastwam-teacache-l1-report.md"
report_json="$report_run_root/fastwam-teacache-l1-report.json"

if [[ "$execute" == "1" ]]; then
  mkdir -p "$report_run_root"
fi

run_or_print wam eval fastwam-libero \
  "${cache_args[@]}" \
  "${upstream_args[@]}" \
  --workload libero-single-task \
  --task-id "$libero_task_id" \
  --num-trials "$libero_num_trials" \
  --set "seed=$seed" \
  --set "num_inference_steps=$num_inference_steps" \
  --set dit_cache_mode=video_kv \
  --set cuda_graph_mode=off \
  --trace-dir "$libero_baseline_trace" \
  --summary-path "$report_run_root/fastwam-libero-eager-cache-summary.json"

run_or_print wam eval fastwam-libero \
  "${cache_args[@]}" \
  "${upstream_args[@]}" \
  --workload libero-single-task \
  --task-id "$libero_task_id" \
  --num-trials "$libero_num_trials" \
  --opt teacache \
  --set "seed=$seed" \
  --set "num_inference_steps=$num_inference_steps" \
  --set dit_cache_mode=video_kv \
  --set cuda_graph_mode=off \
  --set "teacache_threshold=$teacache_threshold" \
  --set "teacache_warmup_steps=$teacache_warmup_steps" \
  --trace-dir "$libero_teacache_trace" \
  --summary-path "$report_run_root/fastwam-libero-teacache-summary.json"

run_compare \
  "$libero_baseline_trace" \
  "$libero_teacache_trace" \
  "$report_run_root/fastwam-libero-teacache-compare.json"

run_or_print wam eval fastwam-robotwin \
  "${cache_args[@]}" \
  "${upstream_args[@]}" \
  --workload robotwin-single-task \
  --task-name "$robotwin_task" \
  --num-episodes "$robotwin_num_episodes" \
  --set "task_config=$robotwin_task_config" \
  "${robotwin_root_args[@]}" \
  --set "seed=$seed" \
  --set "num_inference_steps=$num_inference_steps" \
  --set dit_cache_mode=video_kv \
  --set cuda_graph_mode=off \
  --trace-dir "$robotwin_baseline_trace" \
  --summary-path "$report_run_root/fastwam-robotwin-eager-cache-summary.json"

run_or_print wam eval fastwam-robotwin \
  "${cache_args[@]}" \
  "${upstream_args[@]}" \
  --workload robotwin-single-task \
  --task-name "$robotwin_task" \
  --num-episodes "$robotwin_num_episodes" \
  --set "task_config=$robotwin_task_config" \
  "${robotwin_root_args[@]}" \
  --opt teacache \
  --set "seed=$seed" \
  --set "num_inference_steps=$num_inference_steps" \
  --set dit_cache_mode=video_kv \
  --set cuda_graph_mode=off \
  --set "teacache_threshold=$teacache_threshold" \
  --set "teacache_warmup_steps=$teacache_warmup_steps" \
  --trace-dir "$robotwin_teacache_trace" \
  --summary-path "$report_run_root/fastwam-robotwin-teacache-summary.json"

run_compare \
  "$robotwin_baseline_trace" \
  "$robotwin_teacache_trace" \
  "$report_run_root/fastwam-robotwin-teacache-compare.json"

run_report "$report_markdown" "$report_json"
