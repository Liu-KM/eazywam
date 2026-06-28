# FastWAM Full-Task Evidence Plan

This plan defines the evidence required before FastWAM acceleration profiles
such as `scheduler`, `teacache`, and future `feature_cache` methods can move
from opt-in experimental use toward stronger recommendations. It is a plan for
future experiments. It does not run experiments, change profile defaults, or
upgrade existing single-task evidence into default-enabled evidence.

The public contract remains scheduler-agnostic. Run these commands inside an
already prepared GPU allocation, container, or backend runtime. Keep Slurm,
site queue wrappers, recovery scripts, account names, partitions, and scratch
layout outside the public contract. The tracked evidence must record only the
job id, command, trace path, summary path, profile metadata, fallback reason,
metrics, and evidence root.

## Current Non-Claims

Current FastWAM scheduler evidence is useful for opt-in candidate selection,
but it was collected on one LIBERO task and one RoboTwin task with repeated
confirmation. It does not prove full-task or simulator-wide parity.

Current FastWAM TeaCache L1 evidence is also opt-in evidence. It showed
unchanged success on small single-task runs, but the action output gate did not
pass at the recorded drift threshold, so reportable speedup and default
enablement must not be claimed from that evidence.

Until the matrices in this document pass, `scheduler`, `teacache`, and future
approximate acceleration profiles must stay explicit opt-in profiles. Do not
say that they are `parity_verified` or safe to enable by default.

## Evidence Levels

The labels below describe evidence maturity, not implementation status. A
profile can be implemented while still having only `experimental` evidence.

| Label | Required evidence | Allowed claim | Not allowed |
|---|---|---|---|
| `experimental` | Trace-complete smoke, single-task, or exploratory representative evidence. | "Opt-in experimental profile with measured telemetry." | Do not recommend as a stable candidate, claim parity, or default it. |
| `recommended_candidate` | Representative matrix passes for every target simulator where the profile will be recommended. | "Recommended opt-in candidate for the measured target and config." | Do not call it parity or enable it by default. |
| `parity_verified` | Full-suite paired baseline/candidate evidence passes success, latency, drift, and trace completeness gates. | "Measured parity for the named model, workload, profile, and config." | Do not default it without the separate default-enabled gate. |
| `default_enabled` | `parity_verified` plus operational review, fallback review, rollback plan, and a separate default-change issue/PR. | "Default for the named model and workload." | Do not default from representative or single-task evidence. |

Minimum gates:

| Gate | Success-rate gate | Latency gate | Drift and fallback gate |
|---|---|---|---|
| `experimental` | Success rate is present for simulator runs; failures are documented. | Latency telemetry is present if a speedup is discussed. | Fallback reasons and profile metadata are present. Missing drift blocks speedup claims for approximate profiles. |
| `recommended_candidate` | Representative aggregate success delta is no worse than `-0.03`, and no task or phase regresses by more than `0.15` absolute unless documented as baseline-noisy and rerun. | Mean total latency speedup and target-stage speedup are both at least `1.10x`, or the profile-specific report explains why only one latency metric applies. | No unexplained fallback. Action drift is below the profile's configured threshold. If no threshold exists, the profile stays `experimental`. |
| `parity_verified` | Full-suite aggregate success delta is no worse than `-0.02`; paired bootstrap or Wilson-style lower bound is no worse than `-0.05`; no task or phase regresses by more than `0.10` absolute without a documented rerun. | Mean and p50 total latency improve; p95 latency is not worse than baseline by more than `5%`; denoise wall time or the profile's target stage is reported. | Trace completeness is `100%`; fallback is absent or explicitly scoped as non-impacting; approximate profiles pass action-drift gates. |
| `default_enabled` | Same as `parity_verified`, repeated on the seed band or release target that will become the default. | Speedup remains positive after warmup and fallback filtering. | Default change is reviewed in a separate issue with rollback instructions and updated docs. |

## Baseline And Candidate Pairing

Every candidate run must have a paired baseline run with the same model id,
workload, task, episode/trial count, seed list, action horizon, replan steps,
checkpoint, dataset stats, device class, dtype, runtime path, and simulator
settings. Only the requested profile parameters may differ.

Use these pairing rules:

- Baseline is the current product path with the profile disabled or left at the
  current backend default. For FastWAM scheduler comparisons, keep the current
  10-step baseline unless the manifest default changes in a separate issue.
- Candidate uses `--opt <profile>` and explicit `--set` values for every
  profile parameter needed to reproduce the run.
- If a runner cannot preserve a per-episode seed list inside one command, run
  one command per seed and aggregate with the same table schema.
- Do not compare native runs against reference-script runs for profile speedup.
  Reference runs are useful for native/reference parity, not profile rollout.

## LIBERO Matrix

Target model id: `fastwam-libero`.

Primary suite: `libero_10`.

Full-suite task set:

- Tasks: every `libero_10` task, represented today by `task_id=0..9`.
- Trials per task: `50` for `parity_verified` and `default_enabled`.
- Seeds: fixed paired seed list per task:
  `420000 + task_id * 1000 + trial_index`, where `trial_index=0..49`.
- Execution shape: use `libero-manager` if a native manager can preserve the
  paired seed list and produce per-task summaries. Otherwise run sequential
  `libero-single-task` commands for every task id and aggregate the summaries.

Representative task set:

- Tasks: `task_id=0,2,4,6,9`.
- Rationale: includes the historical hard/mismatch task `task_id=6`, two
  endpoints, and spread across the current `libero_10` task order.
- Trials per task: `20` for `recommended_candidate`.
- Seeds: `430000 + task_id * 1000 + trial_index`, where `trial_index=0..19`.
- If later evidence identifies different hard tasks, replace this subset only
  before a run starts and record the selection reason in the evidence bundle.

Exploratory subset:

- Tasks: `task_id=0,6`.
- Trials per task: `5`.
- Purpose: early `experimental` telemetry only.

Command template for one paired single-task row:

```bash
wam eval fastwam-libero \
  --workload libero-single-task \
  --task-id TASK_ID \
  --num-trials NUM_TRIALS \
  --set seed=SEED_START \
  --set num_steps_wait=30 \
  --trace-dir "$EVIDENCE_ROOT/traces/libero/baseline/task-TASK_ID" \
  --summary-path "$EVIDENCE_ROOT/summaries/libero/baseline-task-TASK_ID.json"

wam eval fastwam-libero \
  --workload libero-single-task \
  --task-id TASK_ID \
  --num-trials NUM_TRIALS \
  --opt PROFILE_NAME \
  --set seed=SEED_START \
  --set num_steps_wait=30 \
  --set PROFILE_PARAM=VALUE \
  --trace-dir "$EVIDENCE_ROOT/traces/libero/candidate/task-TASK_ID" \
  --summary-path "$EVIDENCE_ROOT/summaries/libero/candidate-task-TASK_ID.json"
```

LIBERO success table format:

| model_id | workload | suite | task_id | task_name | profile | config_key | trials | seeds | baseline_successes | candidate_successes | baseline_success_rate | candidate_success_rate | success_delta | failed_episode_ids | trace_path | summary_path | job_id |
|---|---|---|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---|---|
| `fastwam-libero` | `libero-single-task` | `libero_10` | `0` | `TBD` | `scheduler` | `steps=2,sigma_shift=3.25` | `20` | `430000..430019` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `[]` | `.../trace.jsonl` | `...summary.json` | `TBD` |

LIBERO speed table format:

| model_id | task_id | profile | config_key | total_ms_mean_baseline | total_ms_std_baseline | total_ms_mean_candidate | total_ms_std_candidate | total_speedup | denoise_wall_ms_mean_baseline | denoise_wall_ms_mean_candidate | denoise_speedup | action_drift_mean | action_drift_p95 | action_drift_max | fallback_reason | command |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `fastwam-libero` | `0` | `scheduler` | `steps=2,sigma_shift=3.25` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `null` | `wam eval ...` |

## RoboTwin Matrix

Target model id: `fastwam-robotwin`.

Primary suite: `robotwin2.0`.

Full-suite task set:

- Tasks: every task in RoboTwin's current `_eval_step_limit.yml` task list,
  which is 50 tasks in the current maintained environment.
- Phases: both manager phases, `clean` (`demo_clean`) and `random`
  (`demo_randomized`).
- Episodes per task and phase: `10` valid episodes for `parity_verified` and
  `default_enabled`.
- Seed policy: top-level paired seed list
  `520000 + task_index * 1000 + phase_index * 100 + episode_index`, where
  `phase_index=0` for clean and `1` for random. If the manager only accepts one
  top-level seed for a multi-episode phase, record the manager's derived
  candidate episode seeds in the summary and use the same top-level seed for
  baseline and candidate.
- Invalid setup policy: invalid simulator setup is not a policy failure. The
  run is acceptable only when requested valid episodes are completed or the
  evidence bundle records invalid setup exhaustion as a separate environment
  blocker. Do not hide invalid setup as a failed policy episode.

Representative task set:

- Tasks: `10` tasks selected before execution from the 50-task list.
- Required composition: at least three previously verified easy tasks, at least
  three randomized-sensitive tasks, at least two tasks with known setup
  fragility or previous invalid setup records, and at least two long-horizon or
  manipulation-heavy tasks.
- Current task names already seen in maintained evidence include
  `click_alarmclock`, `click_bell`, `press_stapler`, and
  `put_bottles_dustbin`; use the live RoboTwin task list to fill the remaining
  slots and record the final list in the evidence bundle.
- Episodes per task and phase: `5` valid episodes for
  `recommended_candidate`.
- Seed policy: `530000 + task_index * 1000 + phase_index * 100 + episode_index`.

Exploratory subset:

- Tasks: `click_alarmclock` and one known setup-fragile task if available.
- Phases: both clean and random.
- Episodes per task and phase: `2`.
- Purpose: early `experimental` telemetry only.

Command template for one paired manager row:

```bash
wam eval fastwam-robotwin \
  --workload robotwin-manager \
  --set task_name=TASK_NAME_OR_null \
  --set num_episodes=NUM_EPISODES \
  --set seed=SEED_START \
  --set robotwin_root="$ROBOTWIN_ROOT" \
  --set max_worker_restarts_on_invalid_setup=MAX_RESTARTS \
  --trace-dir "$EVIDENCE_ROOT/traces/robotwin/baseline/TASK_KEY" \
  --summary-path "$EVIDENCE_ROOT/summaries/robotwin/baseline-TASK_KEY.json"

wam eval fastwam-robotwin \
  --workload robotwin-manager \
  --set task_name=TASK_NAME_OR_null \
  --set num_episodes=NUM_EPISODES \
  --opt PROFILE_NAME \
  --set seed=SEED_START \
  --set robotwin_root="$ROBOTWIN_ROOT" \
  --set max_worker_restarts_on_invalid_setup=MAX_RESTARTS \
  --set PROFILE_PARAM=VALUE \
  --trace-dir "$EVIDENCE_ROOT/traces/robotwin/candidate/TASK_KEY" \
  --summary-path "$EVIDENCE_ROOT/summaries/robotwin/candidate-TASK_KEY.json"
```

RoboTwin success table format:

| model_id | workload | suite | task_name | phase | profile | config_key | requested_valid_episodes | completed_valid_episodes | attempted_candidate_episodes | invalid_candidate_episodes | invalid_setup_reasons | baseline_success_rate | candidate_success_rate | success_delta | policy_failure_count | trace_path | summary_path | job_id |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|---|
| `fastwam-robotwin` | `robotwin-manager` | `robotwin2.0` | `click_alarmclock` | `random` | `teacache` | `threshold=0.05,warmup=1` | `5` | `5` | `5` | `0` | `[]` | `TBD` | `TBD` | `TBD` | `0` | `.../trace.jsonl` | `...summary.json` | `TBD` |

RoboTwin speed table format:

| model_id | task_name | phase | profile | config_key | total_ms_mean_baseline | total_ms_std_baseline | total_ms_mean_candidate | total_ms_std_candidate | total_speedup | denoise_wall_ms_mean_baseline | denoise_wall_ms_mean_candidate | denoise_speedup | action_drift_mean | action_drift_p95 | action_drift_max | fallback_reason | command |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `fastwam-robotwin` | `click_alarmclock` | `random` | `teacache` | `threshold=0.05,warmup=1` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `null` | `wam eval ...` |

## Required Metrics

Every report must compute these metrics per row and in aggregate:

- `success_rate`: successes divided by valid policy episodes or trials.
- `success_delta`: candidate success rate minus paired baseline success rate.
- `latency_ms.mean`, `latency_ms.std`, `latency_ms.p50`, and
  `latency_ms.p95` from native inference or eval summary timing.
- `denoise_wall_ms.mean` and `denoise_wall_ms.std` when the profile affects
  the denoise loop.
- `speedup`: baseline mean latency divided by candidate mean latency. Report
  both total latency speedup and target-stage speedup.
- `action_drift`: mean, p95, and max scalar drift from paired action summaries
  or `wam compare` output. Approximate profiles must have a configured gate.
- `trace_completeness`: percentage of runs with complete trace, summary,
  profile metadata, fallback reason field, command, and job id.
- `fallback_reason`: `null` when the requested profile state was honored;
  non-null values must be counted and explained.
- `profile_metadata`: requested profile name, params, enabled flag, runtime
  applied state, hook name when available, and profile-specific telemetry.

Do not report a speedup as valid when:

- success metrics are missing for either baseline or candidate;
- `wam compare` returns `invalid` or the output gate fails;
- the baseline trace already used the candidate profile;
- the candidate trace lacks profile metadata;
- fallback reason is non-null and the fallback invalidates the candidate path;
- trace and summary metrics disagree on success counts.

## Required Trace Fields

Each baseline and candidate trace must include the common fields defined in
`docs/trace_schema.md`, plus the rollout fields below:

- `run_start.output_dir`
- `run_start.device`
- `run_start.dtype`
- `run_start.optimization_profiles`
- `run_start.manifest_defaults`
- `run_start.config_overrides`
- `runtime_contract.backend`
- `runtime_contract.processor`
- `runtime_contract.workload`
- `runtime_contract.runtime_mode`
- `runtime_contract.supported_optimizations`
- `optimization_profile_status.stage`
- `optimization_profile_status.name`
- `optimization_profile_status.enabled`
- `optimization_profile_status.params`
- `optimization_profile_status.state`
- `optimization_profile_status.hook`
- `optimization_profile_status.reason`
- `inference_end.timing.total_ms`
- `inference_end.backend_metadata.denoise_wall_ms`
- `inference_end.backend_metadata.<profile fields>`
- `episode_end.success` or manager-equivalent success metrics
- `native_eval_end.successes`
- `native_eval_end.total_episodes`
- `native_eval_end.success_rate`
- `run_end.status`
- `run_end.trace_path`

Scheduler-specific trace fields:

- `scheduler_name`
- `solver`
- `num_inference_steps`
- `sigma_shift`
- `timestep_count`
- `timesteps`
- `sigmas`
- `schedule_type`
- `schedule_source`
- `denoise_wall_ms`
- `total_ms`
- `scheduler_fallback_reason` or profile status `reason`

TeaCache and future feature-cache trace fields:

- `teacache_enabled`
- `teacache_mode`
- `teacache_layers`
- `teacache_threshold`
- `teacache_warmup_steps`
- `teacache_hit_rate`
- `teacache_skipped_steps`
- `teacache_drift_score`
- `teacache_fallback_reason`

RoboTwin manager summaries must also preserve:

- requested valid episodes;
- completed valid episodes;
- attempted candidate episodes;
- invalid candidate episodes;
- invalid candidate reasons;
- `policy_failure=false` for simulator setup invalidity;
- policy failure counts separately from invalid setup counts.

## Evidence Bundle

Write one evidence bundle per profile/config/report. The bundle can be JSON,
Markdown plus JSON, or a directory with a manifest, but it must include these
fields:

```json
{
  "schema_version": 1,
  "profile": "scheduler",
  "config_key": "steps=2,sigma_shift=3.25",
  "evidence_level_requested": "recommended_candidate",
  "evidence_level_result": "TBD",
  "evidence_root": "/path/to/evidence/root",
  "created_at": "YYYY-MM-DDTHH:MM:SSZ",
  "model_ids": ["fastwam-libero", "fastwam-robotwin"],
  "commands": ["wam eval ..."],
  "runs": [
    {
      "role": "baseline",
      "model_id": "fastwam-libero",
      "workload": "libero-single-task",
      "task_id": 0,
      "task_name": null,
      "phase": null,
      "seed_start": 430000,
      "trial_or_episode_count": 20,
      "command": "wam eval ...",
      "job_id": "TBD",
      "trace_path": "/path/to/trace.jsonl",
      "summary_path": "/path/to/summary.json",
      "profile_metadata": {},
      "fallback_reason": null,
      "success_metrics": {},
      "latency_metrics": {},
      "action_drift": {}
    }
  ],
  "aggregate_success_metrics": {},
  "aggregate_latency_metrics": {},
  "aggregate_action_drift": {},
  "trace_completeness": {},
  "fallback_summary": {},
  "decision": "experimental|recommended_candidate|parity_verified|default_enabled|not_recommended",
  "non_claims": [
    "This evidence does not change profile defaults.",
    "This evidence does not claim default_enabled unless the default gate is explicitly passed."
  ]
}
```

Required evidence fields, repeated for clarity:

- trace path;
- summary path;
- job id;
- command;
- profile metadata;
- fallback reason;
- success metrics;
- latency metrics;
- action drift;
- evidence root.

## SuperPod Run Bookkeeping

Use SuperPod or an equivalent maintainer-approved GPU/simulator environment as
the measurement venue, but keep site mechanics outside the public contract.

Required tracked bookkeeping:

- `evidence_root`: stable root for the report and bundle.
- `trace_root`: directory containing trace subdirectories.
- `summary_root`: directory containing eval summaries.
- `report_root`: directory containing tables, compare JSON, and bundle JSON.
- `job_id`: site scheduler or allocation id, recorded as data only.
- `commands`: exact `wam eval` and `wam compare` commands.
- `environment`: GPU type, container or uv runtime label, EazyWAM commit, and
  FastWAM vendored commit.

Local-only bookkeeping that must not become public contract:

- Slurm account, partition, QoS, reservation, and retry wrapper.
- Site-specific scratch path conventions.
- Site-specific module load or venv activation snippets.
- Any assumption that users must run the same scheduler.

## Decision Procedure

1. Select the target evidence level and profile config before running.
2. Generate paired baseline and candidate commands for the required matrix.
3. Run inside an already prepared allocation or container.
4. Fill job ids into the run manifest.
5. Verify every summary points to an existing trace and every trace ends with
   `run_end.status=ok`.
6. Run `wam compare` for paired baseline/candidate traces with the configured
   action-drift gate.
7. Build the success and speed tables above.
8. Build the evidence bundle and mark `decision`.
9. Write explicit non-claim language into the final report.

The final report must include:

- candidate and baseline config;
- LIBERO and RoboTwin matrix coverage;
- success-rate tables;
- latency and denoise wall-time tables;
- action drift tables;
- fallback summary;
- trace completeness summary;
- evidence bundle path;
- exact reasons for `experimental`, `recommended_candidate`,
  `parity_verified`, `default_enabled`, or `not_recommended`.

## Follow-Up Experiment Issues

Create separate experiment issues instead of combining execution with this
planning document:

- Run FastWAM scheduler representative-matrix evidence for LIBERO and RoboTwin
  using the gates in this document.
- Run FastWAM scheduler full-suite parity evidence for any
  `recommended_candidate` config that survives the representative matrix.
- Run FastWAM TeaCache representative-matrix evidence with an explicit action
  drift threshold and compare gate.
- Run FastWAM TeaCache full-suite parity evidence only after representative
  evidence passes without output-gate failure.
- Add a report verifier that checks this bundle schema and refuses
  `parity_verified` or `default_enabled` when required fields are missing.
