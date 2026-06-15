# FastWAM Scheduler / Sampler Baseline And Sweep Plan

## Baseline Note

FastWAM LIBERO and RoboTwin eval currently use a 10-step action denoise
baseline. Do not report the product baseline as 20 steps.

- `src/eazywam/manifests/fastwam-libero.yaml` sets LIBERO eval
  `num_inference_steps: "10"`.
- `src/eazywam/manifests/fastwam-robotwin.yaml` sets RoboTwin eval
  `num_inference_steps: "10"`.
- `src/fastwam/configs/train.yaml` sets `eval_num_inference_steps: 10`.
- `FastWAMModelAdapter._num_inference_steps()` resolves request
  `runtime_options["num_inference_steps"]`, then scheduler profile params,
  then eval config, then the model-level fallback.
- `FastWAM.infer_action(..., num_inference_steps=20)` is only a function
  fallback and is not the current eval default.

The scheduler/sampler route is training-free inference-time acceleration. It
changes the denoise schedule, step count, or sampler update rule; it does not
train a new checkpoint.

## FastWAM / Diffusers FlowMatch Alignment

FastWAM's current action scheduler is a continuous FlowMatch-style Euler
sampler:

- schedule source: `WanContinuousFlowMatchScheduler`
- inference path: `FastWAM.infer_action()`
- sigma transform: `phi(u) = shift * u / (1 + (shift - 1) * u)`
- timesteps: `sigma * num_train_timesteps`
- update rule: `sample = sample + model_output * delta_sigma`

This matches the part of Diffusers `FlowMatchEulerDiscreteScheduler` that is
most relevant for the first implementation: shifted sigmas, timesteps derived
from sigmas, and Euler updates over sigma deltas. The first profile therefore
keeps the existing FastWAM-native scheduler and makes its schedule explicit and
traceable instead of inserting DPM-Solver++ or UniPC before proving the model
output parameterization and update equations match.

Implemented first:

- `scheduler_name=fastwam_flowmatch_euler`
- `schedule_type=shifted_flowmatch`
- configurable `num_inference_steps`
- configurable `sigma_shift`
- Diffusers-style custom `timesteps` or custom `sigmas` as mutually exclusive
  explicit profile/runtime options
- trace metadata for timesteps, sigmas, deltas, denoise wall time, and total
  harness timing, plus `schedule_source` to distinguish generated, custom
  timestep, and custom sigma schedules

Deferred until experiments justify them:

- alternate timestep spacing
- Karras-like or AYS-like schedules
- DPM-Solver++ / UniPC adapters

## Coarse-To-Fine SuperPod Search

The sweep must include the current 10-step baseline, faster candidates, and a
small quality-reference region above baseline.

Stage A, coarse:

- steps: `4, 5, 6, 8, 10, 12, 16`
- sigma shifts: `null, 2.5, 3.0, 5.0, 7.0`
- run at least one LIBERO task and one RoboTwin task per candidate
- record latency, denoise wall time, success rate, action drift, trace path,
  command, job id, and fallback reason

Stage B, refine:

- keep all non-dominated candidates from Stage A
- add neighboring step counts and sigma shifts around promising regions
- rerun the 10-step baseline in the same job shape
- include at least one higher-cost reference if Stage A shows instability

Stage C, confirm:

- repeat only Pareto candidates and the 10-step baseline
- do not claim `parity_verified` without enough repeated trials and statistical
  evidence

Example commands:

```bash
wam eval fastwam-libero --opt scheduler --set task_id=0 --set num_trials=1 \
  --set num_inference_steps=6 --set sigma_shift=3.0

wam eval fastwam-robotwin --opt scheduler --set task_name=click_alarmclock \
  --set num_episodes=1 --set num_inference_steps=6 --set sigma_shift=3.0
```

Portable coarse-sweep command generation inside a prepared SuperPod allocation:

```bash
export WAM_TRACE_DIR=/path/to/superpod/runs/fastwam-scheduler-stage-a
python -m eazywam.evals.scheduler_sweep \
  --trace-dir "$WAM_TRACE_DIR" \
  --output-dir "$WAM_TRACE_DIR/manifest" \
  --baseline-steps 10

bash "$WAM_TRACE_DIR/manifest/scheduler_sweep_commands.sh"
```

The default Stage A generator derives its step counts from `--baseline-steps`
so the search is anchored on the real 10-step baseline while still covering
faster candidates and quality references. If the first SuperPod allocation has
a different budget, override the coarse space explicitly instead of editing the
source code:

```bash
python -m eazywam.evals.scheduler_sweep \
  --trace-dir "$WAM_TRACE_DIR" \
  --output-dir "$WAM_TRACE_DIR/manifest" \
  --baseline-steps 10 \
  --step-counts 4,5,6,8,12,16 \
  --sigma-shifts null,2.5,3.0,5.0,7.0
```

The generator always adds the exact baseline step count and the default
`sigma_shift=null` path. It rejects a coarse step space that lacks either a
faster-than-baseline candidate or a quality-reference candidate.

Inside a site-managed SuperPod allocation, keep scheduler/queue details in a
local wrapper and pass only normal `wam eval` context into the generated
manifest. For the current maintained SuperPod uv runtime this usually means:

```bash
python -m eazywam.evals.scheduler_sweep \
  --trace-dir "$WAM_TRACE_DIR" \
  --output-dir "$WAM_TRACE_DIR/manifest" \
  --baseline-steps 10 \
  --step-counts 4,5,6,8,12,16 \
  --sigma-shifts null,2.5,3.0,5.0,7.0 \
  --cache-dir "$CACHE_DIR" \
  --upstream-dir "$FASTWAM_ROOT" \
  --set seed=42 \
  --libero-set "mujoco_gl=${MUJOCO_GL:-egl}" \
  --libero-set "pyopengl_platform=${PYOPENGL_PLATFORM:-egl}" \
  --robotwin-set "robotwin_root=$ROBOTWIN_ROOT" \
  --robotwin-set gpu_id=0
```

The public repo should not commit Slurm account, partition, venv, scratch, or
site recovery logic. Keep those in ignored `.local/superpod/*.sbatch` wrappers
and only write the resulting commands, job ids, trace paths, reports, and
general conclusions back into tracked docs.

The generated manifest contains:

- `scheduler_sweep_candidates.csv`
- `scheduler_sweep_candidates.json`
- `scheduler_sweep_commands.sh`
- `scheduler_sweep_job_map.csv`
- `scheduler_report_command.sh`

After each SuperPod submission or allocation run, record the queue job id next
to the eval summary path in `scheduler_sweep_job_map.csv`. If the scheduler
does not inject job ids into the summary JSON automatically, fill in the
generated sidecar CSV:

```csv
candidate_id,summary_path,job_id
fastwam-libero-steps10-shiftnull,/path/to/fastwam-libero-steps10-shiftnull-summary.json,451000
```

The report tool accepts extra columns in that sidecar and reads
`candidate_id`, `summary_path`, and `job_id`. When the sidecar lives next to
the generated `scheduler_sweep_candidates.csv` or
`scheduler_sweep_candidates.json`, the report uses the manifest's outer
`wam eval ... --opt scheduler ...` command for reproducibility instead of the
native backend's internal eval display string:

```csv
candidate_id,summary_path,job_id
fastwam-libero-steps10-shiftnull,/path/to/fastwam-libero-steps10-shiftnull-summary.json,451000
```

Before building the report, audit the run directory evidence:

```bash
python -m eazywam.evals.scheduler_audit "$WAM_TRACE_DIR/manifest"
```

This checks that every manifest candidate has a completed summary, an existing
trace path, `summary.status=ok`, a terminal `run_end` with `status=ok`,
scheduler backend metadata with schedule summaries, and a filled job id. For
exploratory local dry runs, the audit can be relaxed with
`--allow-missing-job-ids`, `--allow-missing-summaries`,
`--allow-missing-trace-paths`, `--allow-missing-scheduler-metadata`, or
`--allow-non-ok-status`; do not use those relaxed flags for final SuperPod
evidence.

Build the machine-readable table and HTML figures from completed summaries:

```bash
python -m eazywam.evals.scheduler_report \
  --output-dir "$WAM_TRACE_DIR/report" \
  --job-map "$WAM_TRACE_DIR/manifest/scheduler_sweep_job_map.csv" \
  "$WAM_TRACE_DIR"/summaries/*.json
```

Or run the generated helper script:

```bash
bash "$WAM_TRACE_DIR/manifest/scheduler_report_command.sh"
```

The helper script runs `scheduler_audit`, `scheduler_report`,
`scheduler_acceptance`, and then `scheduler_bundle`. Missing summary files,
trace files, scheduler backend metadata, non-ok terminal statuses, job ids, or
inconsistent final report artifacts fail before the evidence bundle is written.
Fill in `scheduler_sweep_job_map.csv` before using the helper for final
acceptance.

Before using the report as final evidence, validate that it contains both
FastWAM LIBERO and RoboTwin rows, an exact 10-step baseline, faster candidates,
quality-reference candidates, Pareto rows, source summary paths, trace paths,
candidate ids, job ids, commands, scheduler config fields,
timestep/sigma/delta summaries, fallback reason fields, failure case fields,
and metrics, including action-drift fields. The same check also
verifies that baseline rows have speedup 1.0 and success delta 0.0, and that
the expected scheduler identity is recorded (`fastwam_flowmatch_euler`,
`euler`, `shifted_flowmatch` by default), `timestep_count` matches
`num_inference_steps`, schedule summary counts match `timestep_count`, and
latency, speedup, success-rate, failure-count, and action-drift values are in
valid ranges. It also verifies each row's command contains `wam eval
<model-id>`, `--opt scheduler`, the row's `num_inference_steps`, and the row's
`sigma_shift`. It also verifies recommendation labels are consistent with the
row evidence: only exact 10-step rows may be `baseline`, fallback rows must be
`not_recommended`, and `recommended_candidate` rows must be Pareto candidates
with measured speedup. It also verifies that aggregate config-summary artifacts
exist, include aggregate Pareto/recommendation fields, and match the raw result
rows for repeats, means, failure counts, candidate ids, and job ids. It also
verifies that the Markdown report includes conclusions, config-level decisions,
repeat summaries, recommended/not-recommended sections, the full table, and
the explicit `parity_verified` non-claim. It also verifies that
`scheduler_results.csv`, `scheduler_config_summary.json`,
`scheduler_config_summary.csv`, `scheduler_report.html`, and
`scheduler_final_report.md` exist next to `scheduler_results.json`, and that
the HTML report includes the config-level decision table plus the latency,
success-rate, speedup-vs-success, and drift-vs-speedup figures:

```bash
python -m eazywam.evals.scheduler_acceptance \
  "$WAM_TRACE_DIR/report/scheduler_results.json"
```

After acceptance, write the final evidence bundle manifest. The generated
helper does this automatically:

```bash
python -m eazywam.evals.scheduler_bundle \
  "$WAM_TRACE_DIR/report/scheduler_results.json" \
  --output-dir "$WAM_TRACE_DIR/report" \
  --manifest-dir "$WAM_TRACE_DIR/manifest"
```

For a deliberate later experiment with another scheduler family, override the
identity gate explicitly with `--expected-scheduler-name`, `--expected-solver`,
and `--expected-schedule-type`; do not mix scheduler families in one final
acceptance report.

Exploratory Stage A/B reports must include quality-reference rows above the
10-step baseline, and if such rows exist acceptance requires at least one
`not_recommended` row. Stage C confirmation repeats only the exact baseline and
Pareto candidates, so the generated Stage C report helper adds
`--allow-missing-quality-reference`.

For exploratory local dry runs where SuperPod job ids or trace files are not
available yet, use `--allow-missing-job-ids`,
`--allow-missing-summary-paths`, or `--allow-missing-trace-paths`.
For dry runs without comparable action summaries, use
`--allow-missing-action-drift`. For partial checks before rendering all report
artifacts, use `--allow-missing-report-artifacts`. Do not use those relaxed
flags for final acceptance.

Generate the Stage B refinement manifest from Stage A results:

```bash
export WAM_TRACE_DIR_B=/path/to/superpod/runs/fastwam-scheduler-stage-b
python -m eazywam.evals.scheduler_sweep \
  --stage refine \
  --report-json "$WAM_TRACE_DIR/report/scheduler_results.json" \
  --trace-dir "$WAM_TRACE_DIR_B" \
  --output-dir "$WAM_TRACE_DIR_B/manifest" \
  --baseline-steps 10

bash "$WAM_TRACE_DIR_B/manifest/scheduler_sweep_commands.sh"
```

Refinement keeps the exact 10-step baseline and expands neighboring step counts
and sigma shifts around Pareto or recommended Stage A rows. Non-Pareto
`experimental` rows remain report evidence but are not refinement centers, which
keeps Stage B focused. The generator writes the same CSV/JSON/shell manifest
shape as Stage A.

Generate the Stage C confirmation manifest after Stage B reporting:

```bash
export WAM_TRACE_DIR_C=/path/to/superpod/runs/fastwam-scheduler-stage-c
python -m eazywam.evals.scheduler_sweep \
  --stage confirm \
  --report-json "$WAM_TRACE_DIR_B/report/scheduler_results.json" \
  --trace-dir "$WAM_TRACE_DIR_C" \
  --output-dir "$WAM_TRACE_DIR_C/manifest" \
  --baseline-steps 10 \
  --confirm-repeats 3

bash "$WAM_TRACE_DIR_C/manifest/scheduler_sweep_commands.sh"
```

Confirmation repeats only the exact 10-step baseline and Pareto candidates from
the input report. Each repeat gets a unique summary path and candidate id so job
ids, traces, and failure cases remain auditable.

After generating the Stage C report, run strict acceptance with the repeat
count gate:

```bash
python -m eazywam.evals.scheduler_acceptance \
  "$WAM_TRACE_DIR_C/report/scheduler_results.json" \
  --allow-missing-quality-reference \
  --min-config-repeats 3
```

The generated Stage C `scheduler_report_command.sh` includes the same
`--allow-missing-quality-reference` flag and the same `--min-config-repeats`
value inferred from `--confirm-repeats`.

This writes:

- `scheduler_results.json`
- `scheduler_results.csv`
- `scheduler_config_summary.json`
- `scheduler_config_summary.csv`
- `scheduler_report.html`
- `scheduler_final_report.md`
- `scheduler_evidence_bundle.json`

The report selects a 10-step baseline per model/workload, computes speedup from
mean `inference_end.timing.total_ms`, records mean `denoise_wall_ms`, copies
success rate from eval metrics or episode events, estimates action-summary drift
from comparable `action_summary` scalar fields (`mean`, `min`, `max`, and
`max_abs` when present), records which drift fields were used, and marks the
speedup/success-rate Pareto frontier. The HTML report uses inline SVG so the
core package does not need matplotlib.
For custom timestep or sigma schedules, the report includes `schedule_source`
and a schedule hash in `config_key` so repeated-config aggregation does not
merge distinct custom schedules that happen to use the same step count.

The report also extracts compact failure cases from traces: failed
`episode_end` events record episode/task identifiers and step counts, `error`
events record stage/type/message summaries, and non-ok `run_end` events record
the terminal status. These fields are exported as `failure_count` and
`failure_cases` in JSON, CSV, HTML, and Markdown.

The Markdown final report keeps the same decision evidence as the machine
tables: candidate id, exact eval command, trace path, SuperPod job id, total
latency, denoise latency, speedup, success-rate delta, action drift, scheduler
timestep/sigma/delta summaries, solver, failure cases, fallback reason, Pareto
flag, recommendation label, and recommendation reason.
It also includes a per-model/workload conclusion table that names the selected
10-step baseline, best recommended candidate, fastest Pareto candidate, and
whether more repeats are needed. It also includes a config-level decision table
that aggregates repeated configs by model/workload/solver/config and records
mean/std latency, mean success rate, mean speedup, mean action drift, aggregate
Pareto status, aggregate recommendation, candidate ids, job ids, and trace
paths. For Stage C repeated runs, it also includes a repeat summary table with
per-config repeat counts, latency means/stddev, success-rate means, speedup
means/stddev, drift means, job ids, and trace paths. It also lists the
candidate ids that produced each repeated config.
Exact 10-step baseline rows stay normalized to speedup 1.0 and success delta
0.0; latency variation is reported in the repeat summary instead.

The report also emits a conservative recommendation label:

- `baseline`: the selected 10-step baseline row.
- `recommended_candidate`: Pareto candidate with measured speedup, no fallback,
  and success-rate drop within the configured threshold.
- `experimental`: faster than baseline but not strong enough for the recommended
  candidate bucket.
- `not_recommended`: fallback, no speedup, excessive success drop, or excessive
  drift when a drift threshold is configured.
- `needs_review`: missing comparison data.

These labels are experiment recommendations only. They must not be reported as
`parity_verified`; parity still requires repeated trials and statistical
evidence.

The evidence bundle records the accepted report path, optional sweep manifest
directory, all required report/manifest artifact paths, file sizes, SHA-256
hashes, row-level candidate/job/trace evidence, and config-level aggregate
decisions. Use it as the final handoff manifest for the SuperPod run directory.

Final reporting must include the full candidate table, figures for latency and
success-rate tradeoffs, candidate ids, trace paths, job ids, recommended and
not recommended configs, and a clear statement of whether more repeats are
needed.

## SuperPod Evidence, 2026-06-16 HKT

Final accepted evidence was produced on SuperPod with the normal `wam eval
<model-id> --opt scheduler` path, cached FastWAM assets, `seed=42`, LIBERO
`task_id=0`, and RoboTwin `task_name=click_alarmclock`. All report artifacts
below include `scheduler_results.json`, `scheduler_results.csv`,
`scheduler_config_summary.json`, `scheduler_config_summary.csv`,
`scheduler_report.html`, `scheduler_final_report.md`, and
`scheduler_evidence_bundle.json`. The HTML report contains the latency,
success-rate, speedup-vs-success, and drift-vs-speedup figures.

| Stage | Job id | Rows | Acceptance | Evidence root |
| --- | ---: | ---: | --- | --- |
| A coarse | 454117 | 70 | accepted after report replay; 2 recommended, 19 not recommended, 2 Pareto | `/scratch/peilab/qliucc/wam-superpod-eval/evidence/fastwam-scheduler-stage-a-20260615-223813` |
| B refine | 454238 | 26 | accepted after report replay; 2 recommended, 0 not recommended, 2 Pareto | `/scratch/peilab/qliucc/wam-superpod-eval/evidence/fastwam-scheduler-stage-b-20260616-013913` |
| C confirm | 454260 | 12 | Slurm completed `0:0`; accepted with `--min-config-repeats 3`; 6 recommended rows, 6 Pareto rows | `/scratch/peilab/qliucc/wam-superpod-eval/evidence/fastwam-scheduler-stage-c-20260616-024507` |

The final Stage C evidence bundle is:

```text
/scratch/peilab/qliucc/wam-superpod-eval/evidence/fastwam-scheduler-stage-c-20260616-024507/report/scheduler_evidence_bundle.json
```

Stage C config-level results:

The config keys below use the actual scheduler metadata value. The baseline
commands request `sigma_shift=null`, which resolves to FastWAM's default shift
of `5`.

| Model | Config | Repeats | Total ms mean | Denoise ms mean | Success mean | Speedup mean | Speedup std | Drift mean | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fastwam-libero | `steps=10,sigma_shift=5` | 3 | 157.50 | 84.80 | 1.00 | 1.000 | 0.000 | 0.0000 | baseline |
| fastwam-libero | `steps=2,sigma_shift=3.25` | 3 | 89.57 | 16.51 | 1.00 | 1.761 | 0.015 | 0.0261 | recommended |
| fastwam-robotwin | `steps=10,sigma_shift=5` | 3 | 313.17 | 169.97 | 1.00 | 1.000 | 0.000 | 0.0046 | baseline |
| fastwam-robotwin | `steps=2,sigma_shift=7` | 3 | 253.84 | 101.92 | 1.00 | 1.246 | 0.019 | 0.0123 | recommended |

Recommended opt-in commands from the final confirmation sweep:

```bash
wam eval fastwam-libero --opt scheduler \
  --set num_inference_steps=2 --set sigma_shift=3.25 \
  --cache-dir /scratch/peilab/qliucc/wam-superpod-eval/cache \
  --upstream-dir /scratch/peilab/qliucc/wam-upstreams/FastWAM \
  --set seed=42 --set mujoco_gl=egl --set pyopengl_platform=egl \
  --set task_id=0 --set num_trials=1

wam eval fastwam-robotwin --opt scheduler \
  --set num_inference_steps=2 --set sigma_shift=7 \
  --cache-dir /scratch/peilab/qliucc/wam-superpod-eval/cache \
  --upstream-dir /scratch/peilab/qliucc/wam-upstreams/FastWAM \
  --set seed=42 --set robotwin_root=/scratch/peilab/qliucc/wam-upstreams/RoboTwin \
  --set task_config=demo_randomized --set gpu_id=0 \
  --set task_name=click_alarmclock --set num_episodes=1
```

Representative Stage C trace paths:

```text
fastwam-libero baseline:
/scratch/peilab/qliucc/wam-superpod-eval/evidence/fastwam-scheduler-stage-c-20260616-024507/fastwam-libero/9b49cd4cac6e/trace.jsonl

fastwam-libero recommended:
/scratch/peilab/qliucc/wam-superpod-eval/evidence/fastwam-scheduler-stage-c-20260616-024507/fastwam-libero/1b0bc5ab25a3/trace.jsonl

fastwam-robotwin baseline:
/scratch/peilab/qliucc/wam-superpod-eval/evidence/fastwam-scheduler-stage-c-20260616-024507/fastwam-robotwin/a3b012a302d1/trace.jsonl

fastwam-robotwin recommended:
/scratch/peilab/qliucc/wam-superpod-eval/evidence/fastwam-scheduler-stage-c-20260616-024507/fastwam-robotwin/e56db5412fa7/trace.jsonl
```

Recommendation:

- Keep the product default at the 10-step baseline.
- Treat `steps=2,sigma_shift=3.25` for LIBERO and `steps=2,sigma_shift=7`
  for RoboTwin as explicit scheduler-profile candidates, not defaults.
- Do not claim `parity_verified`; this evidence is a scheduler/sampler
  acceleration result on one LIBERO task and one RoboTwin task, with 3
  confirmation repeats per selected config.
