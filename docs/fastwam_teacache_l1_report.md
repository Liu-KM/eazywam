# FastWAM TeaCache L1 Report

This report is the required SuperPod evidence record for the FastWAM TeaCache
L1 profile. Do not fill result fields from local workstation smoke tests. Only
record numbers from real FastWAM checkpoint runs on SuperPod or an equivalent
maintainer-approved GPU/simulator environment.

## Status

- Current state: SuperPod measurement completed in Slurm job `454086`.
- Profile: `teacache`
- Family: `feature_cache`
- Default: disabled
- Scope: FastWAM action-only inference, request-local cache, `dit_cache_mode=video_kv`
- Exclusions: no CUDA Graph default combination, no PAB, no FasterCache, no
  token pruning, no cross-replan cache, no native/reference parity claim.
- Evidence root:
  `/scratch/peilab/qliucc/wam-teacache-l1/reports/teacache-l1-454086`
- Trace root:
  `/scratch/peilab/qliucc/wam-teacache-l1/traces/teacache-l1-454086`

## Experiment Matrix

| Target | Run | Command | Trace | Eval summary |
|---|---|---|---|---|
| LIBERO | baseline | `wam eval fastwam-libero ... --set dit_cache_mode=video_kv --set cuda_graph_mode=off` | `/scratch/peilab/qliucc/wam-teacache-l1/traces/teacache-l1-454086/fastwam-libero-eager-cache/3ba392561740/trace.jsonl` | `fastwam-libero-eager-cache-summary.json` |
| LIBERO | TeaCache | `wam eval fastwam-libero ... --opt teacache --set dit_cache_mode=video_kv --set cuda_graph_mode=off` | `/scratch/peilab/qliucc/wam-teacache-l1/traces/teacache-l1-454086/fastwam-libero-teacache/e53205b6a9b6/trace.jsonl` | `fastwam-libero-teacache-summary.json` |
| RoboTwin | baseline | `wam eval fastwam-robotwin ... --set robotwin_root=/scratch/peilab/qliucc/wam-upstreams/RoboTwin --set dit_cache_mode=video_kv --set cuda_graph_mode=off` | `/scratch/peilab/qliucc/wam-teacache-l1/traces/teacache-l1-454086/fastwam-robotwin-eager-cache/3e51affa6e05/trace.jsonl` | `fastwam-robotwin-eager-cache-summary.json` |
| RoboTwin | TeaCache | `wam eval fastwam-robotwin ... --opt teacache --set robotwin_root=/scratch/peilab/qliucc/wam-upstreams/RoboTwin --set dit_cache_mode=video_kv --set cuda_graph_mode=off` | `/scratch/peilab/qliucc/wam-teacache-l1/traces/teacache-l1-454086/fastwam-robotwin-teacache/e993f36d34ee/trace.jsonl` | `fastwam-robotwin-teacache-summary.json` |

Keep task, seed, episode/trial count, action horizon, replan steps, and
`num_inference_steps` fixed between baseline and TeaCache runs for each target.

## Required Commands

Prefer generating the exact command set with:

```bash
scripts/fastwam-teacache-l1-superpod.sh
scripts/fastwam-teacache-l1-superpod.sh --execute \
  --cache-dir /path/to/wam-cache \
  --upstream-dir /path/to/FastWAM \
  --robotwin-root /path/to/RoboTwin \
  --trace-root /path/to/traces/fastwam-teacache-l1 \
  --report-root /path/to/reports/fastwam-teacache-l1 \
  --run-id teacache-l1-YYYYMMDD
scripts/fastwam-teacache-l1-report.py \
  --report-root /path/to/reports/fastwam-teacache-l1/teacache-l1-YYYYMMDD
```

The script is scheduler-agnostic: run it inside an existing prepared SuperPod
allocation or container, not as a cluster submission wrapper. The commands
below are the explicit equivalent. In `--execute` mode the script writes
`fastwam-teacache-l1-report.md` and `fastwam-teacache-l1-report.json` under the
run-specific report directory after the compare steps complete. The standalone
report helper can re-read saved JSON outputs; it does not run experiments or
fabricate missing fields. The helper reports observed speedups for auditability
and separately reports `speedup_reportable` plus `speedup_blockers`. Treat a
speedup as a claim only when `speedup_reportable=true`.

Runtime options can enable TeaCache for ad hoc inference, but this report helper
is scoped to the SuperPod acceptance path generated above. Use `--opt teacache`
for the candidate run so `wam compare` can verify candidate profile metadata;
current compare JSON does not record per-request runtime options as a substitute
for profile metadata.

LIBERO baseline:

```bash
wam eval fastwam-libero \
  --workload libero-single-task \
  --task-id 0 \
  --num-trials 5 \
  --set seed=42 \
  --set num_inference_steps=20 \
  --set dit_cache_mode=video_kv \
  --set cuda_graph_mode=off \
  --trace-dir /path/to/traces/fastwam-libero-eager-cache \
  --summary-path /path/to/reports/fastwam-libero-eager-cache-summary.json
```

LIBERO TeaCache:

```bash
wam eval fastwam-libero \
  --workload libero-single-task \
  --task-id 0 \
  --num-trials 5 \
  --opt teacache \
  --set seed=42 \
  --set num_inference_steps=20 \
  --set dit_cache_mode=video_kv \
  --set cuda_graph_mode=off \
  --set teacache_threshold=0.05 \
  --set teacache_warmup_steps=1 \
  --trace-dir /path/to/traces/fastwam-libero-teacache \
  --summary-path /path/to/reports/fastwam-libero-teacache-summary.json
```

RoboTwin baseline:

```bash
wam eval fastwam-robotwin \
  --workload robotwin-single-task \
  --task-name click_alarmclock \
  --num-episodes 5 \
  --set task_config=demo_randomized \
  --set robotwin_root=/path/to/RoboTwin \
  --set seed=42 \
  --set num_inference_steps=20 \
  --set dit_cache_mode=video_kv \
  --set cuda_graph_mode=off \
  --trace-dir /path/to/traces/fastwam-robotwin-eager-cache \
  --summary-path /path/to/reports/fastwam-robotwin-eager-cache-summary.json
```

RoboTwin TeaCache:

```bash
wam eval fastwam-robotwin \
  --workload robotwin-single-task \
  --task-name click_alarmclock \
  --num-episodes 5 \
  --set task_config=demo_randomized \
  --set robotwin_root=/path/to/RoboTwin \
  --opt teacache \
  --set seed=42 \
  --set num_inference_steps=20 \
  --set dit_cache_mode=video_kv \
  --set cuda_graph_mode=off \
  --set teacache_threshold=0.05 \
  --set teacache_warmup_steps=1 \
  --trace-dir /path/to/traces/fastwam-robotwin-teacache \
  --summary-path /path/to/reports/fastwam-robotwin-teacache-summary.json
```

Compare traces:

```bash
wam compare \
  /path/to/traces/fastwam-libero-eager-cache/<run>/trace.jsonl \
  /path/to/traces/fastwam-libero-teacache/<run>/trace.jsonl \
  --max-action-drift 0.001

wam compare \
  /path/to/traces/fastwam-robotwin-eager-cache/<run>/trace.jsonl \
  /path/to/traces/fastwam-robotwin-teacache/<run>/trace.jsonl \
  --max-action-drift 0.001
```

## Required Result Fields

Fill these fields from `wam compare` JSON and eval summaries:

| Target | latency mean speedup | denoise mean speedup | hit rate | skipped steps | drift score | action drift | success rate baseline | success rate TeaCache | fallback reason |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LIBERO | 1.547 | 1.653 | 0.3974 | 7.949 | 0.3665 | 0.9731 | 1 | 1 | TODO |
| RoboTwin | 1.427 | 1.525 | 0.35 | 7 | 0.3137 | 0.03143 | 1 | 1 | TODO |

Speedup blockers:

- LIBERO: `compare_decision:invalid`, `output_gate_not_passed`.
- RoboTwin: `compare_decision:invalid`, `output_gate_not_passed`.

Both targets show real TeaCache call reduction and unchanged 5/5 success rate,
but the output gate did not pass at `--max-action-drift 0.001`; therefore
`speedup_reportable=false` for both rows.

Use:

- `metric_comparisons["latency_ms.mean"].speedup` for total latency speedup.
- `metric_comparisons["backend_metadata.denoise_wall_ms.mean"].speedup` for
  denoise-loop speedup.
- `backend_metadata.teacache_hit_rate`.
- `backend_metadata.teacache_skipped_steps`.
- `backend_metadata.teacache_drift_score`.
- `output_gate_details` for action drift.
- `eval_metrics.success_rate`,
  `metric_comparisons["eval_metrics.success_rate.mean"]`, or RoboTwin manager
  success-rate metrics. For the default RoboTwin `demo_randomized` run, use
  `overall.random_mean_success_rate` when both clean and randomized metrics are
  present.
- `backend_metadata_values.teacache_fallback_reason` for fallback strings.

## Acceptance Notes

- Observed speedups may be listed for auditability, but do not claim speedup as
  reportable if `wam compare` does not return `decision=faster`, or if
  `output_gate_passed` is not `true`.
- Do not claim speedup as reportable if the compared baseline trace includes
  `teacache` or the candidate trace does not include `teacache`.
- Do not claim speedup as reportable if either compared trace lacks profile
  metadata.
- Do not claim speedup as reportable if the TeaCache candidate trace lacks
  hit-rate, skipped-step, or drift-score telemetry.
- Do not claim speedup as reportable if the compare output lacks total-latency
  speedup, denoise-loop speedup, or action drift.
- Do not claim native/reference parity. TeaCache is an approximate acceleration
  profile and must be evaluated by action drift and simulator success rate.
- If success rate drops, report the drop and do not report speedup.
- If success rate is missing from either eval summary, do not report speedup.
- If fallback reasons are present, report them and treat speedup claims as
  unsupported unless the fallback is understood and scoped.
